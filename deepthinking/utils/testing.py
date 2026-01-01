"""testing.py
Utilities for testing models

Collaboratively developed
by Avi Schwarzschild, Eitan Borgnia,
Arpit Bansal, and Zeyad Emam.

Developed for DeepThinking project
October 2021
"""

from collections import deque

import einops
import torch
import torch.nn.functional as F
from icecream import ic
from tqdm import tqdm

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114


def test(
    net,
    loaders,
    mode,
    iters,
    problem,
    device,
    return_outputs=False,
    n_outputs=10,
    p_thresh=0.9,
):
    accs = []
    outputs = None
    for i, loader in enumerate(loaders):
        if mode == "default":
            accuracy, out = test_default(
                net, loader, iters, problem, device, return_outputs, n_outputs
            )
            if i == 0 and return_outputs:
                outputs = out
        elif mode == "max_conf":
            accuracy = test_max_conf(net, loader, iters, problem, device)
        elif mode == "stable":
            accuracy, out = test_stable(
                net, loader, iters, problem, device, return_outputs, n_outputs, p_thresh
            )
            if i == 0 and return_outputs:
                outputs = out

        else:
            raise ValueError(f"{ic.format()}: test_{mode}() not implemented.")
        accs.append(accuracy)

    if return_outputs:
        return accs, outputs  # outputs only for test set
    else:
        return accs


def get_predicted(inputs, outputs, problem):
    outputs = outputs.clone()
    predicted = outputs.argmax(1)
    predicted = predicted.view(predicted.size(0), -1)
    if problem == "mazes":
        predicted = predicted * (inputs.max(1)[0].view(inputs.size(0), -1))
    elif problem == "chess":
        outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
        top_2 = torch.topk(outputs[:, 1], 2, dim=1)[0].min(dim=1)[0]
        top_2 = einops.repeat(top_2, "n -> n k", k=8)
        top_2 = einops.repeat(top_2, "n m -> n m k", k=8).view(-1, 64)
        outputs[:, 1][outputs[:, 1] < top_2] = -float("Inf")
        outputs[:, 0] = -float("Inf")
        predicted = outputs.argmax(1)

    return predicted


def test_default(
    net, testloader, iters, problem, device, return_outputs=False, n_outputs=10
):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters)
    total = 0
    first_n_outputs = []
    i = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            i += 1
            inputs, targets = inputs.to(device), targets.to(device)

            _, _, all_outputs = net(inputs, iters_to_do=max_iters)
            if return_outputs and i <= n_outputs:
                first_n_outputs.append(all_outputs.cpu())

            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                predicted = get_predicted(inputs, outputs, problem)
                targets = targets.view(targets.size(0), -1)
                corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()

            total += targets.size(0)

    accuracy = 100.0 * corrects / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite - 1].item()

    if return_outputs:
        return ret_acc, torch.cat(
            first_n_outputs, dim=0
        )  # (batch_size_total, max_iters, 2, H, W)
    else:
        return ret_acc, None


def test_max_conf(net, testloader, iters, problem, device):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters).to(device)
    total = 0
    softmax = torch.nn.functional.softmax

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(targets.size(0), -1)
            total += targets.size(0)

            _, _, all_outputs = net(inputs, iters_to_do=max_iters)

            confidence_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            corrects_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                conf = softmax(outputs.detach(), dim=1).max(1)[0]
                conf = conf.view(conf.size(0), -1)
                if problem == "mazes":
                    conf = conf * inputs.max(1)[0].view(conf.size(0), -1)
                confidence_array[i] = conf.sum([1])
                predicted = get_predicted(inputs, outputs, problem)
                corrects_array[i] = torch.amin(predicted == targets, dim=[1])

            correct_this_iter = corrects_array[
                torch.cummax(confidence_array, dim=0)[1],
                torch.arange(corrects_array.size(1)),
            ]
            corrects += correct_this_iter.sum(dim=1)

    accuracy = 100 * corrects.long().cpu() / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite - 1].item()
    return ret_acc


def _one_iter(net, x, interim, prev):
    """
    Run exactly one recurrent iteration by exploiting the
    (iters_to_do=1) path and returning the new state.
    """
    logits, interim, _ = net(
        x, iters_to_do=1, interim_thought=interim, prev_output=prev
    )
    return logits.squeeze(1), interim, logits.detach()


def test_stable(
    net,
    loader,
    iters,
    problem,
    device,
    return_outputs=False,
    n_outputs=10,
    p_thresh=0.9,
):
    """
    Per-sample stopping: stop after 5 stable, high-confidence masks.
    Returns:
        ret_acc : {iter: acc} with iter == # steps actually used
        collected : (n, T_used, 2, H, W) if return_outputs else None
    """

    net.eval()
    total = 0
    correct = 0
    max_iters = max(iters)
    samples_collected = 0
    collected_outputs = []

    # statistics
    all_stop_iters = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            B, _, H, W = inputs.shape
            targets = targets.view(B, -1)

            finished = torch.zeros(B, dtype=torch.bool, device=device)
            stop_iter = torch.zeros(B, dtype=torch.long, device=device)
            final_predictions = torch.zeros_like(targets, dtype=torch.float)
            last_masks = [deque(maxlen=5) for _ in range(B)]

            sample_outputs = (
                [[] for _ in range(B)]
                if return_outputs and samples_collected < n_outputs
                else None
            )

            interim, prev_out = None, None
            t = 0

            while (not finished.all()) and t < max_iters:
                t += 1
                logits, interim, prev_out = _one_iter(net, inputs, interim, prev_out)

                if sample_outputs is not None:
                    for b in range(B):
                        if samples_collected + b < n_outputs:
                            sample_outputs[b].append(logits[b : b + 1].cpu())

                predicted = get_predicted(inputs, logits, problem)
                probs = F.softmax(logits, dim=1)[:, 1]  # (B,H,W)
                masks = (probs > 0.5).float()  # binary

                # check for stability
                for b in range(B):
                    if finished[b]:  # already converged
                        continue

                    # add current mask to history
                    last_masks[b].append(masks[b].cpu())

                    # stability criterion:  5 identical masks, all with high confidence
                    if len(last_masks[b]) == 5:
                        same = all(
                            (last_masks[b][k] == last_masks[b][0]).all()
                            for k in range(1, 5)
                        )
                        non_wall_mask = inputs[b, 0] > 0
                        valid_mask = (masks[b] == 1) & non_wall_mask
                        high = (
                            (probs[b][valid_mask] >= p_thresh).all()
                            if valid_mask.sum() > 0
                            else False
                        )

                        if same and high:
                            finished[b] = True
                            stop_iter[b] = t
                            final_predictions[b] = predicted[b]

            # for unfinished samples use last iteration
            unfinished = ~finished
            if unfinished.any():
                stop_iter[unfinished] = t
                final_predictions[unfinished] = predicted[unfinished]

            all_stop_iters.extend(stop_iter.cpu().tolist())
            all_predictions.append(final_predictions.cpu())
            all_targets.append(targets.cpu())

            correct += torch.amin(final_predictions == targets, dim=[1]).sum().item()
            total += B

            if sample_outputs is not None:
                for b in range(B):
                    if samples_collected < n_outputs:
                        if len(sample_outputs[b]) > 0:
                            collected_outputs.append(sample_outputs[b])
                            samples_collected += 1  # keep track of how many outputs we collected regardless of batch
                            if samples_collected >= n_outputs:
                                break

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    stop_iter_tensor = torch.tensor(all_stop_iters)
    mean_iters = stop_iter_tensor.float().mean().item()
    iter_hist = torch.bincount(stop_iter_tensor, minlength=max_iters + 1).tolist()

    # accuracy by iteration
    acc_by_iter = {}
    for t in range(1, max_iters + 1):
        indices = (stop_iter_tensor == t).nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            preds_t = all_predictions[indices]
            targets_t = all_targets[indices]
            acc_t = torch.amin(preds_t == targets_t, dim=1).sum().item() / len(indices)
            acc_by_iter[t] = round(100.0 * acc_t, 2)

    ret_acc = {
        "mean_iters": round(mean_iters, 2),
        "iter_hist": iter_hist,
        "acc_by_iter": acc_by_iter,
        "stable_acc": round(100.0 * correct / total, 2),
    }

    # pad collected outputs to the same length using all zeros
    if return_outputs and collected_outputs:
        max_T = max(len(seq) for seq in collected_outputs)
        padded_outputs = []
        for seq in collected_outputs:
            if len(seq) < max_T:
                padding_shape = seq[-1].shape
                padding = [
                    torch.zeros(padding_shape, device="cpu")
                    for _ in range(max_T - len(seq))
                ]
                padded_seq = seq + padding
            else:
                padded_seq = seq
            padded_outputs.append(torch.cat(padded_seq, dim=0))
        collected = torch.stack(padded_outputs)

        return ret_acc, collected
    else:
        return ret_acc, None
