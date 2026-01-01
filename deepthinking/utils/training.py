import typing
from dataclasses import dataclass
from random import randrange

import torch
from tqdm import tqdm

from deepthinking.utils.maze_solver import MazeSolver
from deepthinking.utils.testing import get_predicted

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114, W0611


@dataclass
class TrainingSetup:
    """Attributes to describe the training precedure"""

    optimizer: "typing.Any"
    scheduler: "typing.Any"
    warmup: "typing.Any"
    clip: "typing.Any"
    alpha: "typing.Any"
    max_iters: "typing.Any"
    problem: "typing.Any"
    mazesolver_mode: "typing.Any"
    step: "typing.Any"


def build_oracle_batch(inputs_batch, solver: MazeSolver, step: int):
    """
    Args
        inputs : (B, 3, H, W) float tensor on GPU
        solver : MazeSolver() instance on CPU
    Returns
        paths      : list[list[np.ndarray]]  length B
        path_lens  : torch.LongTensor  (B,)
        T_max      : int   max(path_lens)
    """
    paths = []
    lengths = []
    max_len = 0
    for b in range(inputs_batch.size(0)):
        path_b = solver.get_intermediate_supervision_masks(
            inputs_batch[b].cpu().numpy(), step
        )
        paths.append(path_b)
        lengths.append(len(path_b))
        max_len = max(max_len, len(path_b))
    return (
        paths,
        torch.tensor(lengths, device=inputs_batch.device),
        max_len,
    )


def loss_mask(all_out, path_lens):
    """
    all_out : (B, T_max, 2, H, W)
    path_lens : (B,)  number of valid steps for every sample
    Returns
        mask : (B, T_max) bool  True where step is valid

    e.g.: if (B=3, T_max=6) and path_lens = [4, 2, 5]
    mask = [[T T T T F F]
            [T T F F F F]
            [T T T T T F]]
    """
    B, T_max = all_out.shape[:2]
    idx = torch.arange(T_max, device=all_out.device)[None, :].expand(B, T_max)
    return idx < path_lens[:, None]


def get_output_for_prog_loss(inputs, max_iters, net):
    # get features from n iterations to use as input
    n = randrange(0, max_iters)

    # do k iterations using intermediate features as input
    k = randrange(1, max_iters - n + 1)

    if n > 0:
        _, interim_thought, _ = net(inputs, iters_to_do=n)
        interim_thought = interim_thought.detach()
    else:
        interim_thought = None

    outputs, _, _ = net(
        inputs, iters_elapsed=n, iters_to_do=k, interim_thought=interim_thought
    )
    return outputs, k


def train(net, loaders, mode, train_setup, device):
    """Updated train function with combined mode"""
    # if mode == "combined":
    #     train_loss, acc = train_combined(net, loaders, train_setup, device)
    #     return train_loss, acc
    if mode == "progressive":
        train_loss, acc = train_progressive(net, loaders, train_setup, device)
    elif mode == "intermediate":
        train_loss, acc = train_with_intermediate_supervision(
            net, loaders, train_setup, device
        )
    else:
        train_loss, acc = train_default(net, loaders, train_setup, device)

    return train_loss, acc


def train_with_intermediate_supervision(net, loaders, train_setup, device):
    trainloader = loaders["train"]
    net.train()

    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters  # upper budget
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    solver = MazeSolver(mode=train_setup.mazesolver_mode)

    train_loss = 0
    correct = 0
    total = 0
    T_max = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)

        optimizer.zero_grad()

        mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

        if alpha != 0:
            paths, path_lens, T_max = build_oracle_batch(
                inputs, solver, train_setup.step
            )

        outputs_max_iters, _, all_outputs = net(inputs, iters_to_do=T_max)

        if alpha != 1:
            outputs_max_iters = outputs_max_iters.view(
                outputs_max_iters.size(0), outputs_max_iters.size(1), -1
            )
            loss_max_iters = criterion(outputs_max_iters, targets)
            loss_max_iters = loss_max_iters * mask
            loss_max_iters = loss_max_iters[mask > 0]
        else:
            loss_max_iters = torch.zeros_like(targets).float()

        # === Intermediate supervision loss ===
        if alpha != 0:
            B, _, C, H, W = all_outputs.shape
            oracle = torch.zeros((B, T_max, H, W), device=device, dtype=torch.long)

            for b, steps in enumerate(paths):
                for t, mask_np in enumerate(steps[:T_max]):
                    oracle[b, t] = torch.from_numpy(mask_np)

            pred_flat = all_outputs.view(B * T_max, C, H, W)  # (B*T, 2, H, W)
            targ_flat = oracle.view(B * T_max, H, W)  # (B*T, H, W)

            loss_map = criterion(pred_flat, targ_flat)  # (B*T, H, W)
            loss_map = loss_map.view(B, T_max, H, W)  # (B, T, H, W)

            mask_valid = loss_mask(
                all_outputs, path_lens
            )  # (B, T)  True where t<path_len
            mask_valid = mask_valid.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
            valid_losses = loss_map * mask_valid

            # per-sample normalization
            sample_losses = valid_losses.sum(dim=[1, 2, 3])  # Sum over T,H,W
            sample_counts = mask_valid.sum(dim=[1, 2, 3])  # Count valid positions
            sample_avg_losses = sample_losses / (sample_counts + 1e-8)  # Avoid div by 0
            interm_loss = sample_avg_losses.mean()

        else:
            interm_loss = torch.tensor(0.0, device=inputs.device)

        # === Combine losses ===
        loss = (1 - alpha) * loss_max_iters.mean() + alpha * interm_loss.mean()
        loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        train_loss += loss.item()

        with torch.no_grad():
            predicted = get_predicted(inputs, outputs_max_iters, problem)
            correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
            total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc


def train_progressive(net, loaders, train_setup, device):
    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        if problem == "mazes":
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).
        outputs_max_iters, _, _= net(inputs, iters_to_do=max_iters)
        if alpha != 1:
            outputs_max_iters = outputs_max_iters.view(
                outputs_max_iters.size(0), outputs_max_iters.size(1), -1
            )
            loss_max_iters = criterion(outputs_max_iters, targets)
        else:
            loss_max_iters = torch.zeros_like(targets).float()

        # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # so we save time by setting it equal to 0).
        if alpha != 0:
            outputs, k = get_output_for_prog_loss(inputs, max_iters, net)
            outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
            loss_progressive = criterion(outputs, targets)
        else:
            loss_progressive = torch.zeros_like(targets).float()

        if problem == "mazes":
            loss_max_iters = loss_max_iters * mask
            loss_max_iters = loss_max_iters[mask > 0]
            loss_progressive = loss_progressive * mask
            loss_progressive = loss_progressive[mask > 0]

        loss_max_iters_mean = loss_max_iters.mean()
        loss_progressive_mean = loss_progressive.mean()

        loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc


def train_default(net, loaders, train_setup, device):
    """
    one forward pass for `max_iters`, single CE loss against target, optional pixel mask for mazes.
    """
    trainloader = loaders["train"]
    net.train()

    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_sched = train_setup.warmup
    max_iters = train_setup.max_iters
    clip = train_setup.clip
    problem = train_setup.problem

    ce = torch.nn.CrossEntropyLoss(reduction="none")

    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in tqdm(trainloader, leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device).long().view(inputs.size(0), -1)

        # pixel-wise maze mask (same as original code)
        if problem == "mazes":
            maze_mask = (
                inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0
            )

        optimizer.zero_grad()

        # ---------- forward for full budget ----------
        logits, _, _ = net(inputs, iters_to_do=max_iters)  # (B, 2, H, W)
        logits = logits.view(logits.size(0), logits.size(1), -1)

        loss = ce(logits, targets)  # (B, P)

        if problem == "mazes":
            loss = (loss * maze_mask).sum(-1) / maze_mask.sum(
                -1
            )  # mean over true pixels
        else:
            loss = loss.mean(-1)  # mean over pixels

        loss = loss.mean()  # mean over batch
        loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        running_loss += loss.item()

        with torch.no_grad():
            preds = get_predicted(inputs, logits, problem)
            correct += torch.amin(preds == targets, dim=-1).sum().item()
            total += targets.size(0)

    lr_scheduler.step()
    warmup_sched.dampen()

    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc
