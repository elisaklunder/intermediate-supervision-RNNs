import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


def read_hydra_config(stats_path):
    """Read Hydra config file and extract solver parameters."""
    config_path = Path(stats_path).parent / ".hydra/config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        mode = (
            config.get("problem", {}).get("hyp", {}).get("mazesolver_mode", "unknown")
        )
        steps = config.get("problem", {}).get("hyp", {}).get("step", "unknown")
        return f"Mode: {mode}, Step size: {steps}"
    except (FileNotFoundError, yaml.YAMLError):
        return "Config not found or invalid"


def visualize_stats(json_path, loader="test"):
    """
    Visualize data from stats.json file

    Args:
        json_path (str): Path to the stats.json file
        output_dir (str, optional): Directory to save visualizations
    """

    output_dir = Path(json_path).parent / "figures"

    with open(json_path, "r") as f:
        stats = json.load(f)

    sns.set_theme(style="whitegrid")

    stats = stats.get(f"{loader}_acc", {})

    solver_params = read_hydra_config(json_path)

    if "acc_by_iter" in stats:
        iterations = [int(k) for k in stats["acc_by_iter"].keys()]
        accuracies = list(stats["acc_by_iter"].values())

        plt.figure(figsize=(12, 6))
        plt.plot(
            iterations, accuracies, marker="o", linestyle="-", color="steelblue", markersize=3
        )
        plt.axhline(
            y=stats.get("stable_acc", 0),
            color="r",
            linestyle="--",
            label=f"Mean Accuracy: {stats.get('stable_acc', 0):.2f}%\nSet: {loader}, {solver_params}",
        )
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy per Iteration")
        plt.legend()
        plt.grid(True)

        if output_dir:
            plt.savefig(
                f"{output_dir}/accuracy_per_iteration.png", dpi=300, bbox_inches="tight"
            )
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(accuracies, bins=20, color='steelblue')
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Accuracies\nSet: {loader}, {solver_params}")

        if output_dir:
            plt.savefig(
                f"{output_dir}/accuracy_distribution.png", dpi=300, bbox_inches="tight"
            )
        plt.show()

    if "iter_hist" in stats:
        mean_iters = stats.get("mean_iters", 0)
        
        plt.figure(figsize=(15, 8))
        
        iter_hist = stats["iter_hist"]
        non_zero_indices = [i for i, count in enumerate(iter_hist) if count > 0]
        if non_zero_indices:
            min_idx = max(0, min(non_zero_indices) - 5)
            max_idx = min(len(iter_hist), max(non_zero_indices) + 5)
            
            x_values = list(range(min_idx, max_idx))
            y_values = iter_hist[min_idx:max_idx]
            plt.bar(x_values, y_values, color='steelblue', edgecolor='steelblue', alpha=0.7)
            
            for i, count in enumerate(y_values):
                if count > max(y_values) * 0.05:
                    plt.text(x_values[i], count + (max(y_values) * 0.01), str(count), 
                            ha='center', fontweight='bold')
            
            plt.xlabel("Iteration", fontsize=12)
            plt.ylabel("Count (samples)", fontsize=12)
            plt.title(f"Iteration Histogram\nSet: {loader}, {solver_params}", fontsize=14)
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.axvline(
                x=mean_iters,
                color='crimson',
                linestyle='--',
                linewidth=2,
                label=f"Mean Iteration: {mean_iters}"
            )
            plt.legend(fontsize=12)
            
            if output_dir:
                output_dir.mkdir(exist_ok=True)
                plt.savefig(
                    f"{output_dir}/iteration_histogram.png", dpi=300, bbox_inches="tight"
                )
            plt.show()
            
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, help="Path to the stats.json file.")
    args = parser.parse_args()
    visualize_stats(args.json_path)

if __name__ == "__main__":
    main()

