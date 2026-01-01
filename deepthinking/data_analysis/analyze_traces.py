import os
import pickle
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from tqdm import tqdm

project_root = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deepthinking.utils.maze_solver import MazeSolver


def compute_and_save_path_data(
    dataset_path: str = "data/maze_data_train_9/inputs.npy",
    cache_dir: str = "cache",
    force_recompute: bool = False,
):
    """
    Compute path lengths for all algorithms and save to cache.

    Args:
        dataset_path: Path to the maze dataset
        cache_dir: Directory to save cached results
        force_recompute: If True, recompute even if cache exists

    Returns:
        df: DataFrame with path length data
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    dataset_path_obj = Path(dataset_path)
    parent_dir = dataset_path_obj.parent.name
    filename = dataset_path_obj.stem
    cache_file = cache_path / f"{parent_dir}_{filename}_path_data.pkl"

    if cache_file.exists() and not force_recompute:
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"Computing path data for {dataset_path}")
    solver = MazeSolver()
    mazes = np.load(dataset_path)

    data = []
    for i, maze in enumerate(tqdm(mazes, desc="Computing path lengths")):
        inter_len = len(solver.get_incremental_path_masks(maze, step=3))
        reverse_len = len(solver.get_reverse_exploration_masks(maze, step=3))
        bidirectional_len = len(
            solver.get_incremental_path_masks_bidirectional(maze, step=3)
        )
        dfs_len = len(solver.get_dfs_masks(maze, step=3))

        data.append(
            {
                "maze_id": i,
                "incremental": inter_len,
                "bidirectional": bidirectional_len,
                "reverse": reverse_len,
                "dfs": dfs_len,
            }
        )

    df = pd.DataFrame(data)

    print(f"Saving computed data to {cache_file}")
    with open(cache_file, "wb") as f:
        pickle.dump(df, f)

    return df


def plot_training_maze_analysis(
    dataset_path: str = "data/maze_data_train_9/inputs.npy",
    cache_dir: str = "cache",
    force_recompute: bool = False,
):
    """
    Create detailed analysis plots with larger fonts for training maze size (9x9)
    """
    print(f"Processing training dataset: {dataset_path}")

    df = compute_and_save_path_data(dataset_path, cache_dir, force_recompute)

    algorithms = ["incremental", "bidirectional", "reverse", "dfs"]
    stats = {}
    colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]

    for alg in algorithms:
        stats[alg] = {
            "mean": df[alg].mean(),
            "median": df[alg].median(),
            "std": df[alg].std(),
            "min": df[alg].min(),
            "max": df[alg].max(),
            "q25": df[alg].quantile(0.25),
            "q75": df[alg].quantile(0.75),
        }

    stats_df = pd.DataFrame(stats).round(2)

    print("\nTraining Data (9x9 mazes) - Algorithm Statistics:")
    print("=" * 60)
    print(stats_df.to_string())

    fig_dist, ax_dist = plt.subplots(figsize=(18, 12), dpi=120)
    box_data = [df[alg] for alg in algorithms]
    violin_parts = ax_dist.violinplot(
        box_data,
        positions=range(1, len(algorithms) + 1),
        showmeans=True,
        showextrema=True,
        bw_method=0.3,
    )

    violin_parts["cmeans"].set_color("black")
    violin_parts["cmaxes"].set_color("black")
    violin_parts["cmins"].set_color("black")
    violin_parts["cbars"].set_color("black")

    for i, (patch, color) in enumerate(zip(violin_parts["bodies"], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_dist.set_xticks(range(1, len(algorithms) + 1))
    ax_dist.set_xticklabels([alg.capitalize() for alg in algorithms])

    legend_elements = [
        Patch(facecolor=color, alpha=0.7, label=alg.capitalize())
        for alg, color in zip(algorithms, colors)
    ]
    ax_dist.legend(handles=legend_elements, loc="upper left", fontsize=28)

    ax_dist.set_title(
        "Trace Length Distribution Comparison (9x9 Mazes)",
        fontsize=36,
        weight="bold",
    )
    ax_dist.set_ylabel("Path Length", fontsize=32)
    ax_dist.tick_params(axis="both", which="major", labelsize=28)
    ax_dist.grid(alpha=0.3)

    plt.tight_layout()
    fig_dist.savefig(
        "figures/training_maze_9x9_violinplot",
        dpi=300,
        bbox_inches="tight",
    )

    fig_detailed, axes = plt.subplots(2, 2, figsize=(24, 20), dpi=120)
    axes = axes.flatten()

    for i, (alg, color) in enumerate(zip(algorithms, colors)):
        ax = axes[i]

        data_alg = df[alg]

        bins = range(
            0,
            50,
            2,
        )
        n, bins_edges, patches = ax.hist(
            data_alg, bins=bins, color=color, alpha=0.7, edgecolor="black"
        )

        ax.axvline(
            stats[alg]["mean"], color="red", linestyle="--", linewidth=3, label="Mean"
        )
        ax.axvline(
            stats[alg]["median"],
            color="blue",
            linestyle=":",
            linewidth=3,
            label="Median",
        )

        ax.set_title(
            f"{alg.capitalize()}",
            fontsize=48,
            weight="bold",
        )
        ax.set_xlabel("Path Length", fontsize=40)
        ax.set_ylabel("Frequency", fontsize=40)
        ax.legend(fontsize=36)
        ax.tick_params(axis="both", which="major", labelsize=36)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_detailed.savefig(
        "figures/training_maze_9x9_detailed_histograms", dpi=300, bbox_inches="tight"
    )

    return stats_df, df


def analyze_maze_sizes_comparison(
    dataset_paths: dict = None,
    cache_dir: str = "cache",
    force_recompute: bool = False,
    save_plot: bool = True,
):
    """
    Analyze path lengths across different maze sizes for all algorithms.

    Args:
        dataset_paths: Dict mapping size names to dataset paths
                      e.g., {"9x9": "data/maze_data_train_9/inputs.npy"}
        cache_dir: Directory to save cached results
        force_recompute: If True, recompute even if cache exists
        save_plot: If True, save the generated plot

    Returns:
        summary_stats: DataFrame with mean/std for each algorithm and size
        all_data: Dict with raw data for each size
    """
    if dataset_paths is None:
        dataset_paths = {
            "9x9": "data/maze_data_test_9/inputs.npy",
            "11x11": "data/maze_data_test_11/inputs.npy",
            "13x13": "data/maze_data_test_13/inputs.npy",
            "15x15": "data/maze_data_test_15/inputs.npy",
            "19x19": "data/maze_data_test_19/inputs.npy",
            "25x25": "data/maze_data_test_25/inputs.npy",
            "33x33": "data/maze_data_test_33/inputs.npy",
            "59x59": "data/maze_data_test_59/inputs.npy",
        }

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    cache_file = cache_path / "maze_size_comparison_analysis.pkl"

    all_data = {}
    summary_stats = None

    if cache_file.exists() and not force_recompute:
        print(f"Loading cached comparison data from {cache_file}")
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
            all_data = cached_data["all_data"]
            summary_stats = cached_data["summary_stats"]
    else:
        print("Computing maze size comparison analysis...")

        algorithms = ["incremental", "bidirectional", "reverse", "dfs"]
        summary_data = []

        for size_name, dataset_path in dataset_paths.items():
            if not os.path.exists(dataset_path):
                print(
                    f"Warning: Dataset {dataset_path} not found, skipping {size_name}"
                )
                continue

            print(f"Processing {size_name} mazes from {dataset_path}")

            df = compute_and_save_path_data(dataset_path, cache_dir, force_recompute)
            all_data[size_name] = df

            for alg in algorithms:
                if alg in df.columns:
                    summary_data.append(
                        {
                            "size": size_name,
                            "algorithm": alg,
                            "mean": df[alg].mean(),
                            "std": df[alg].std(),
                            "median": df[alg].median(),
                            "count": len(df[alg]),
                            "min": df[alg].min(),
                            "max": df[alg].max(),
                        }
                    )

        summary_stats = pd.DataFrame(summary_data)

        cache_data = {
            "summary_stats": summary_stats,
            "all_data": all_data,
        }
        print(f"Saving comparison analysis to {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)

    if save_plot:
        _create_maze_size_comparison_plot(summary_stats, all_data)

    return summary_stats, all_data


def _create_maze_size_comparison_plot(summary_stats, all_data):
    """Create and save the maze size comparison plot with error bars."""

    fig, ax = plt.subplots(figsize=(16, 12), dpi=120)

    algorithms = ["incremental", "bidirectional", "reverse", "dfs"]
    colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]
    markers = ["o", "s", "^", "D"]

    sizes = sorted(summary_stats["size"].unique(), key=lambda x: int(x.split("x")[0]))
    x_positions = np.arange(len(sizes))

    for i, (alg, color, marker) in enumerate(zip(algorithms, colors, markers)):
        alg_data = summary_stats[summary_stats["algorithm"] == alg]

        means = []
        stds = []

        for size in sizes:
            size_data = alg_data[alg_data["size"] == size]
            if len(size_data) > 0:
                means.append(size_data["mean"].iloc[0])
                stds.append(size_data["std"].iloc[0])
            else:
                means.append(np.nan)
                stds.append(np.nan)

        ax.errorbar(
            x_positions,
            means,
            yerr=stds,
            label=alg.capitalize(),
            color=color,
            marker=marker,
            linewidth=3,
            markersize=12,
            capsize=8,
            capthick=3,
        )

    ax.set_xlabel("Maze Size", fontsize=32, weight="bold")
    ax.set_ylabel("Mean Path Length", fontsize=32, weight="bold")
    ax.set_title(
        "Supervision Traces Length Across Maze Sizes", fontsize=36, weight="bold"
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(sizes, fontsize=28)
    ax.tick_params(axis="y", which="major", labelsize=28)

    ax.legend(fontsize=28, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    fig.savefig(
        "figures/maze_size_algorithm_comparison_2.png", dpi=300, bbox_inches="tight"
    )
    print("Saved comparison plot to figures/maze_size_algorithm_comparison.png")

    pivot_mean = summary_stats.pivot(index="size", columns="algorithm", values="mean")
    pivot_std = summary_stats.pivot(index="size", columns="algorithm", values="std")

    print("\nMean Path Lengths by Size and Algorithm:")
    print("=" * 60)
    print(pivot_mean.round(2).to_string())

    print("\nStandard Deviations by Size and Algorithm:")
    print("=" * 60)
    print(pivot_std.round(2).to_string())

    return fig, ax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/maze_data_train_9/inputs.npy",
        help="Path to the maze dataset (.npy file). Default: data/maze_data_train_9/inputs.npy",
    )
    args = parser.parse_args()

    stats, raw_data = plot_training_maze_analysis(
        args.input_path, force_recompute=False
    )

    print("Running maze size comparison analysis...")
    summary_stats, all_data = analyze_maze_sizes_comparison(
        force_recompute=False, save_plot=True
    )
    
if __name__ == "__main__":
    main()

