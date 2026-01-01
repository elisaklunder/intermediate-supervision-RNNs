import argparse
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

mpl.rcParams.update(
    {
        "font.family": "Raleway",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)
PALETTE = {
    "default": "#33b0f2",  # blue
    "baseline": "#33b0f2",  # blue
    "reverse": "#cc0000",  # red
    "incremental": "#ff9900",  # yellow
    "bidirectional": "#6aa84f",  # green
    "dfs": "#9966cc",  # purple
}

sns.set_theme(style="whitegrid", font_scale=1.2)


def collect_stats(main_folder, model_names, sizes):
    all_stats = []

    for strategy_folder in os.listdir(main_folder):
        if not strategy_folder.startswith("testing-"):
            continue

        strategy_name = strategy_folder.replace("testing-", "")
        if not any(model in strategy_name for model in model_names):
            continue

        strategy_path = os.path.join(main_folder, strategy_folder)
        if not os.path.isdir(strategy_path):
            continue

        for run_folder in os.listdir(strategy_path):
            if not run_folder.startswith(f"testing-{strategy_name}-"):
                continue

            folder_parts = run_folder.replace(f"testing-{strategy_name}-", "").split(
                "-"
            )
            if not folder_parts:
                continue

            size_part = folder_parts[0]
            if "x" not in size_part:
                continue

            maze_size = size_part.split("x")[0]
            if maze_size not in sizes:
                continue

            run_number = 0
            if len(folder_parts) > 1:
                try:
                    run_number = int(folder_parts[1])
                except ValueError:
                    run_number = 0

            stats_path = os.path.join(strategy_path, run_folder, "stats.json")
            if not os.path.exists(stats_path):
                continue

            try:
                with open(stats_path, "r") as f:
                    stats = json.load(f)
            except json.JSONDecodeError:
                continue

            all_stats.append(
                {
                    "model": strategy_name,
                    "size": maze_size,
                    "run_number": run_number,
                    "stable_acc": stats.get("test_acc", {}).get("stable_acc", None),
                    "test_iters": stats.get("test_iters", []),
                    "iter_hist": stats.get("test_acc", {}).get("iter_hist", []),
                    "acc_by_iter": stats.get("test_acc", {}).get("acc_by_iter", {}),
                    "num_params": stats.get("num_params", 0),
                }
            )

    return all_stats


def collect_percolation_stats(main_folder, model_names):
    all_stats = []

    for strategy_folder in os.listdir(main_folder):
        if not strategy_folder.endswith("-percolation"):
            continue

        strategy_name = strategy_folder.replace("-percolation", "").replace(
            "testing-", ""
        )
        if not any(model in strategy_name for model in model_names):
            continue

        strategy_path = os.path.join(main_folder, strategy_folder)
        if not os.path.isdir(strategy_path):
            continue

        for run_folder in os.listdir(strategy_path):
            if not run_folder.startswith(f"testing-{strategy_name}-"):
                continue

            folder_parts = run_folder.replace(f"testing-{strategy_name}-", "")
            if not folder_parts.startswith("p-"):
                continue

            try:
                p_value = float(folder_parts.replace("p-", ""))
            except ValueError:
                continue

            stats_path = os.path.join(strategy_path, run_folder, "stats.json")
            if not os.path.exists(stats_path):
                continue

            try:
                with open(stats_path, "r") as f:
                    stats = json.load(f)
            except json.JSONDecodeError:
                continue

            all_stats.append(
                {
                    "model": strategy_name,
                    "p_value": p_value,
                    "stable_acc": stats.get("test_acc", {}).get("stable_acc", None),
                    "test_iters": stats.get("test_iters", []),
                    "iter_hist": stats.get("test_acc", {}).get("iter_hist", []),
                    "acc_by_iter": stats.get("test_acc", {}).get("acc_by_iter", {}),
                    "num_params": stats.get("num_params", 0),
                }
            )

    return all_stats


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_plot(fig, name, save_dir):
    fig_path = os.path.join(save_dir, name)
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_stable_accuracy(data, save_dir):
    df = pd.DataFrame(data)
    df = df.dropna(subset=["stable_acc"])
    df["size"] = pd.to_numeric(df["size"])
    df["stable_acc"] = pd.to_numeric(df["stable_acc"])

    summary = (
        df.groupby(["model", "size"])["stable_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary = summary.sort_values("size")
    summary["se"] = summary["std"] / (summary["count"] ** 0.5)
    summary.loc[summary["count"] <= 1, "se"] = 0

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, model in enumerate(summary["model"].unique()):
        model_data = summary[summary["model"] == model]
        color = PALETTE.get(model, f"C{idx}")
        marker = "o"
        linestyle = "-"

        ax.plot(
            model_data["size"],
            model_data["mean"],
            label=model,
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=3,
            markersize=7,
        )

        valid_se = model_data["se"] > 0
        if valid_se.any():
            valid_data = model_data[valid_se]
            ax.fill_between(
                valid_data["size"],
                valid_data["mean"] - valid_data["se"],
                valid_data["mean"] + valid_data["se"],
                alpha=0.2,
                color=color,
                label="", 
            )

    ax.set_title("Stable Accuracy by Maze Size", fontsize=16, fontweight="bold")
    ax.set_xlabel("Maze Size", fontsize=14, fontweight="bold")
    ax.set_ylabel("Stable Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_xticks(sorted(df["size"].unique()))
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(
        title="Model",
        fontsize=11,
        title_fontsize=12,
        frameon=True,
        fancybox=True,
        edgecolor="black",
    )

    fig.tight_layout()
    save_plot(fig, "stable_accuracy.png", save_dir)


def plot_iteration_distributions(data, save_dir):
    data = [
        entry
        for entry in data
        if all(iteration <= 1000 for iteration in entry.get("test_iters", []))
    ]
    iters_expanded = []
    for entry in data:
        test_iters = entry["test_iters"]
        iter_hist = entry.get("iter_hist", [])
        if not test_iters or not iter_hist:
            continue
        for i, count in enumerate(iter_hist):
            if i < len(test_iters):
                iters_expanded.extend(
                    [
                        {
                            "model": entry["model"],
                            "size": entry["size"],
                            "iterations": test_iters[i],
                        }
                    ]
                    * count
                )

    df = pd.DataFrame(iters_expanded)
    df["size"] = df["size"].astype(str)

    df["size_numeric"] = pd.to_numeric(df["size"])
    df = df.sort_values("size_numeric")

    size_order = df.sort_values("size_numeric")["size"].unique()

    fig, ax = plt.subplots(figsize=(10, 6))

    flierprops = dict(
        marker="o", markersize=1, linestyle="none", markerfacecolor="gray", alpha=0.4
    )
    boxprops = dict(linewidth=0)
    whiskerprops = dict(linewidth=1.0)
    capprops = dict(linewidth=1.0)
    medianprops = dict(linewidth=1.0, color="black")

    sns.boxplot(
        data=df,
        x="size",
        y="iterations",
        hue="model",
        palette=PALETTE,
        ax=ax,
        flierprops=flierprops,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        order=size_order,
    )

    ax.set_title(
        "Test Iteration Distribution by Maze Size", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Maze Size", fontsize=14, fontweight="bold")
    ax.set_ylabel("Iterations", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(
        title="Model",
        fontsize=11,
        title_fontsize=12,
        frameon=True,
        fancybox=True,
        edgecolor="black",
    )

    fig.tight_layout()
    save_plot(fig, "iteration_distribution.png", save_dir)

def plot_stable_accuracy_bars(data, save_dir):
    df = pd.DataFrame(data)
    df = df.dropna(subset=["stable_acc"])
    df["size"] = pd.to_numeric(df["size"])
    df["stable_acc"] = pd.to_numeric(df["stable_acc"])

    summary = (
        df.groupby(["model", "size"])["stable_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("size")
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=df,
        x="size",
        y="stable_acc",
        hue="model",
        palette=PALETTE,
        errorbar="sd",
        ax=ax,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_title("Stable Accuracy by Maze Size (Bar Chart)", fontsize=16)
    ax.set_xlabel("Maze Size", fontsize=14)
    ax.set_ylabel("Stable Accuracy (%)", fontsize=14)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(title="Model", fontsize=11, title_fontsize=12, frameon=True)

    fig.tight_layout()
    save_plot(fig, "stable_accuracy_barplot.png", save_dir)


def plot_percolation_accuracy(data, save_dir):
    df = pd.DataFrame(data)
    df = df.dropna(subset=["stable_acc"])
    df["p_value"] = pd.to_numeric(df["p_value"])
    df["stable_acc"] = pd.to_numeric(df["stable_acc"])

    models = df["model"].unique()
    n_models = len(models)

    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else list(axes)
    else:
        axes = axes.flatten()

    for idx, model in enumerate(models):
        model_data = df[df["model"] == model]
        summary = (
            model_data.groupby("p_value")["stable_acc"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("p_value")
        )

        summary["se"] = summary["std"] / (summary["count"] ** 0.5)
        summary.loc[summary["count"] <= 1, "se"] = 0

        ax = axes[idx]
        color = PALETTE.get(model, f"C{idx}")

        ax.plot(
            summary["p_value"],
            summary["mean"],
            marker="o",
            linestyle="-",
            color=color,
            linewidth=3,
            markersize=7,
            label=model,
        )

        valid_se = summary["se"] > 0
        if valid_se.any():
            valid_data = summary[valid_se]
            ax.fill_between(
                valid_data["p_value"],
                valid_data["mean"] - valid_data["se"],
                valid_data["mean"] + valid_data["se"],
                alpha=0.2,
                color=color,
            )

        ax.set_title(f"{model}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Percolation Value", fontsize=12, fontweight="bold")
        ax.set_ylabel("Stable Accuracy (%)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(axis="both", labelsize=10)

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Stable Accuracy vs Percolation Value", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "percolation_accuracy.png", save_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Compare model results and generate plots."
    )
    parser.add_argument(
        "main_folder", type=str, help="Path to main folder containing stats.json files"
    )
    parser.add_argument(
        "--models", nargs="+", required=True, help="List of model name substrings"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        help="List of test_data maze sizes (not used with --percolation)",
    )
    parser.add_argument(
        "--percolation",
        action="store_true",
        help="Plot percolation results instead of regular model comparison",
    )
    args = parser.parse_args()

    if args.percolation:
        stats = collect_percolation_stats(args.main_folder, args.models)
        if not stats:
            print("No matching percolation stats.json files found.")
            return

        models_str = "_".join(args.models)
        save_dir = os.path.join(args.main_folder, f"figures_percolation_{models_str}")
        ensure_dir(save_dir)

        print(f"Saving percolation figures to: {save_dir}")
        plot_percolation_accuracy(stats, save_dir)
        print("Percolation plots saved successfully.")
    else:
        if not args.sizes:
            print("--sizes is required when not using --percolation mode")
            return

        stats = collect_stats(args.main_folder, args.models, args.sizes)
        if not stats:
            print("No matching stats.json files found.")
            return

        models_str = "_".join(args.models)
        sizes_str = "_".join(args.sizes)
        save_dir = os.path.join(args.main_folder, f"figures_{models_str}_{sizes_str}")
        ensure_dir(save_dir)

        print(f"Saving figures to: {save_dir}")

        plot_stable_accuracy(stats, save_dir)
        plot_iteration_distributions(stats, save_dir)
        plot_stable_accuracy_bars(stats, save_dir)

        print("Plots saved successfully.")


if __name__ == "__main__":
    main()
