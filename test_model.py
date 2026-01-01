import json
import logging
import os
import sys
from collections import OrderedDict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import deepthinking as dt
from deepthinking.utils.plot import animate_prediction_sequence, plot_heatmap_sequence

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     No cfg in main call (E1120), No specified encoding (W1514), Use lazy % formatting in logging functions (W1203)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, E1120, W1514, W1203


@hydra.main(config_path="config", config_name="test_model_config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    if cfg.problem.hyp.save_period is None:
        cfg.problem.hyp.save_period = cfg.problem.hyp.epochs
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("test_model.py main() running.")
    log.info(OmegaConf.to_yaml(cfg))

    training_args = OmegaConf.load(
        os.path.join(cfg.problem.model.model_path, ".hydra/config.yaml")
    )
    cfg_keys_to_load = [
        ("hyp", "alpha"),
        ("hyp", "epochs"),
        ("hyp", "lr"),
        ("hyp", "lr_factor"),
        ("model", "max_iters"),
        ("model", "model"),
        ("hyp", "optimizer"),
        ("hyp", "train_mode"),
        ("model", "width"),
        ("hyp", "mazesolver_mode"),
        ("hyp", "step"),
        ("hyp", "train_batch_size"),
    ]
    for k1, k2 in cfg_keys_to_load:
        cfg["problem"][k1][k2] = training_args["problem"][k1][k2]
    cfg.problem.train_data = cfg.problem.train_data

    log.info(OmegaConf.to_yaml(cfg))

    ####################################################
    #               Dataset and Network and Optimizer
    loaders = dt.utils.get_dataloaders(cfg.problem)

    cfg.problem.model.model_path = os.path.join(
        cfg.problem.model.model_path, "model_best.pth"
    )
    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(
        cfg.problem.name, cfg.problem.model, device
    )
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    log.info(
        f"This {cfg.problem.model.model} has {pytorch_total_params / 1e6:0.3f} million parameters."
    )
    ####################################################

    ####################################################
    #        Test
    log.info("==> Starting testing...")
    if "feedforward" in cfg.problem.model.model:
        test_iterations = [cfg.problem.model.max_iters]
    else:
        test_iterations = list(
            range(
                cfg.problem.model.test_iterations["low"],
                cfg.problem.model.test_iterations["high"] + 1,
            )
        )

    if cfg.quick_test:
        test_acc = dt.test(
            net,
            [loaders["test"]],
            cfg.problem.hyp.test_mode,
            test_iterations,
            cfg.problem.name,
            device,
        )
        test_acc = test_acc[0]
        val_acc, train_acc = None, None

    elif cfg.plot_outputs:
        print(
            "Testing with plot_outputs=True and test_mode=", cfg.problem.hyp.test_mode
        )
        plot_n = 5

        accs, test_outputs = dt.test(
            net,
            [loaders["test"], loaders["val"], loaders["train"]],
            cfg.problem.hyp.test_mode,
            test_iterations,
            cfg.problem.name,
            device,
            return_outputs=True,
            n_outputs=plot_n,
        )

        test_acc, val_acc, train_acc = accs

        inputs, targets = next(iter(loaders["test"]))
        inputs = inputs.to(device)
        for idx in range(min(plot_n, inputs.size(0))):
            logits_seq = test_outputs[idx]  # (T, 2, H, W)
            probs_seq = torch.softmax(logits_seq, dim=1)[:, 1].cpu().numpy()
            plot_heatmap_sequence(
                input_maze=inputs[idx].cpu().numpy(),
                target=targets[idx].cpu().numpy(),
                probs_seq=probs_seq,
                steps=range(1, probs_seq.shape[0], 1),
                title_prefix=f"sample_{idx}",
            )
            try:
                animate_prediction_sequence(
                    input_maze=inputs[idx].cpu().numpy(),
                    target=targets[idx].cpu().numpy(),
                    probs_seq=probs_seq,
                    title_prefix=f"sample_{idx}",
                    frame_duration=0.5,
                )
            except (ValueError, RuntimeError, ImportError, IOError) as e:
                log.error(f"Error animating prediction sequence: {e}")

    else:
        test_acc, val_acc, train_acc = dt.test(
            net,
            [loaders["test"], loaders["val"], loaders["train"]],
            cfg.problem.hyp.test_mode,
            test_iterations,
            cfg.problem.name,
            device,
        )

    log.info(f"{dt.utils.now()} Training accuracy: {train_acc}")
    log.info(f"{dt.utils.now()} Val accuracy: {val_acc}")
    log.info(f"{dt.utils.now()} Testing accuracy (hard data): {test_acc}")

    model_name_str = f"{cfg.problem.model.model}_width={cfg.problem.model.width}"
    stats = OrderedDict(
        [
            ("epochs", cfg.problem.hyp.epochs),
            ("lr", cfg.problem.hyp.lr),
            ("lr_factor", cfg.problem.hyp.lr_factor),
            ("max_iters", cfg.problem.model.max_iters),
            ("model", model_name_str),
            ("model_path", cfg.problem.model.model_path),
            ("num_params", pytorch_total_params),
            ("optimizer", cfg.problem.hyp.optimizer),
            ("val_acc", val_acc),
            ("run_id", cfg.run_id),
            ("test_acc", test_acc),
            ("test_data", cfg.problem.test_data),
            ("test_iters", test_iterations),
            ("test_mode", cfg.problem.hyp.test_mode),
            ("train_data", cfg.problem.train_data),
            ("train_acc", train_acc),
            ("train_batch_size", cfg.problem.hyp.train_batch_size),
            ("train_mode", cfg.problem.hyp.train_mode),
            ("alpha", cfg.problem.hyp.alpha),
        ]
    )
    with open(os.path.join("stats.json"), "w") as fp:
        json.dump(stats, fp)
    log.info(stats)


if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    if not any(arg.startswith("+run_id=") for arg in sys.argv):
        sys.argv.append(f"+run_id={run_id}")
    main()
