import os
import re
from datetime import datetime
from collections import defaultdict
import statistics
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

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
    "baseline": "#33b0f2",  # blue
    "reverse": "#cc0000",  # red
    "incremental": "#ff9900",  # yellow
    "bidirectional": "#6aa84f",  # green
    "dfs": "#9966cc",  # purple
}

sns.set_theme(style="whitegrid", font_scale=1.2)

STRATEGIES = ["bidirectional", "incremental", "reverse", "dfs"]

TIMESTAMP_FMT = "%m/%d/%Y %H:%M:%S"
TIMESTAMP_RE = re.compile(r"\[(\d{2}\/\d{2}\/\d{4} \d{2}:\d{2}:\d{2}) INFO\]")

TRAIN_START_RE = re.compile(r"Starting training")

TEST_START_RE = re.compile(r"Starting testing")
TEST_END_RE = re.compile(r"OrderedDict")


def parse_timestamp(line):
    m = TIMESTAMP_RE.search(line)
    if m:
        return datetime.strptime(m.group(1), TIMESTAMP_FMT)
    return None


def extract_duration(log_path, start_re = None, end_re = None):
    start, end = None, None
    first_timestamp = None
    last_timestamp = None

    with open(log_path, "r") as f:
        for line in f:
            ts = parse_timestamp(line)
            if ts is None:
                continue

            if first_timestamp is None:
                first_timestamp = ts
            last_timestamp = ts

            if start is None:
                if start_re is None:
                    start = ts
                elif start_re.search(line):
                    start = ts
            
            elif start is not None:
                if end_re is None:
                    end = ts 
                elif end_re.search(line):
                    end = ts
                    break

    if start_re is None and first_timestamp:
        start = first_timestamp
    if end_re is None and last_timestamp:
        end = last_timestamp

    if start and end:
        return (end - start).total_seconds() / 60.0
    return None


def base_strategy(name):
    name = name.replace("training-", "").replace("testing-", "")
    return name.split("-")[0]


def extract_maze_size(name):
    m = re.search(r"(\d+)x\1", name)
    return int(m.group(1)) if m else None


def analyze_training(training_root):
    times = defaultdict(list)

    for run in os.listdir(training_root):
        run_dir = os.path.join(training_root, run)
        log_path = os.path.join(run_dir, "train.log")

        if not os.path.isfile(log_path):
            continue

        strategy = base_strategy(run)

        if strategy not in STRATEGIES:
            continue

        duration = extract_duration(log_path, TRAIN_START_RE)
        if duration is not None:
            times[strategy].append(duration)

    return times


def analyze_testing(outputs_root):
    times = defaultdict(lambda: defaultdict(list))
    run_info = defaultdict(lambda: defaultdict(list))

    for d in os.listdir(outputs_root):
        if not d.startswith("testing-"):
            continue

        strategy = d.replace("testing-", "")
        if strategy not in STRATEGIES:
            continue

        strategy_dir = os.path.join(outputs_root, d)

        for run in os.listdir(strategy_dir):
            run_dir = os.path.join(strategy_dir, run)
            log_path = os.path.join(run_dir, "testing.log")

            if not os.path.isfile(log_path):
                continue

            maze_size = extract_maze_size(run)
            duration = extract_duration(log_path, TEST_START_RE, TEST_END_RE)

            if maze_size is not None and duration is not None:
                times[strategy][maze_size].append(duration)
                run_info[strategy][maze_size].append((duration, run))

    return times, run_info


def print_testing_details(testing_times, run_info):
    all_sizes = set()
    for sizes in testing_times.values():
        all_sizes.update(sizes.keys())
    
    for size in sorted(all_sizes):
        print(f"\n=== MAZE SIZE {size}x{size} ===")
        for strategy in sorted(testing_times.keys()):
            if size in testing_times[strategy]:
                print(f"\n{strategy:15s}:")
                for duration, run_name in run_info[strategy][size]:
                    print(f"  {run_name:30s}: {duration:.2f} min")


def summarize(values):
    mean = statistics.mean(values)
    se = statistics.stdev(values) / (len(values) ** 0.5) if len(values) > 1 else 0.0
    return mean, se


def print_report(training_times, testing_times):
    print("\n=== TRAINING TIME (9x9 mazes) ===\n")
    for strat, vals in training_times.items():
        mean, se = summarize(vals)
        print(f"{strat:15s}  mean = {mean:.2f} min   SE = {se:.2f} min")

    print("\n=== TESTING TIME (by maze size) ===\n")
    for strat, sizes in testing_times.items():
        print(f"\nStrategy: {strat}")
        for size in sorted(sizes):
            mean, se = summarize(sizes[size])
            print(f"  {size:>3}x{size:<3}  mean = {mean:.2f} min   SE = {se:.2f} min")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_plot(fig, name, save_dir="figures"):
    ensure_dir(save_dir)
    fig_path = os.path.join(save_dir, name)
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot(training_times, testing_times, save_dir="figures"):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = []
    ses = [] 
    colors = []
    
    for strat in STRATEGIES:
        values = training_times[strat]
        mean, se = summarize(values)
        means.append(mean / 60.0)
        ses.append(se / 60.0)
        colors.append(PALETTE.get(strat, "#33b0f2"))
    
    bars = ax.bar(STRATEGIES, means, color=colors, edgecolor="black", linewidth=0.6, alpha=0.8)
    ax.errorbar(STRATEGIES, means, yerr=ses, fmt='none', capsize=5, color='black', linewidth=1.5)
    
    ax.set_title("Training Time by Strategy", fontsize=16, fontweight="bold")
    ax.set_ylabel("Training Time (hours)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Strategy", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    
    fig.tight_layout()
    save_plot(fig, "time_analysis_training.png", save_dir)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    all_sizes = set()
    for sizes in testing_times.values():
        all_sizes.update(sizes.keys())
    all_sizes = sorted(all_sizes)
    
    bar_width = 0.15
    n_strategies = len(STRATEGIES)
    x = range(len(all_sizes))
    
    for i, strat in enumerate(STRATEGIES):
        means = []
        ses = []
        
        for size in all_sizes:
            if size in testing_times[strat]:
                values = testing_times[strat][size]
                mean, se = summarize(values)
                means.append(mean / 60.0) 
                ses.append(se / 60.0)
            else:
                means.append(0)
                ses.append(0)
        
        x_pos = [pos + i * bar_width - (n_strategies - 1) * bar_width / 2 for pos in x]
        
        color = PALETTE.get(strat, "#33b0f2")
        ax.bar(x_pos, means, bar_width, label=strat, color=color, 
               edgecolor="black", linewidth=0.6, alpha=0.8)
        ax.errorbar(x_pos, means, yerr=ses, fmt='none', capsize=3, 
                   color='black', linewidth=1)
    
    ax.set_title("Testing Time by Strategy and Maze Size", fontsize=16, fontweight="bold")
    ax.set_xlabel("Maze Size", fontsize=14, fontweight="bold")
    ax.set_ylabel("Testing Time (hours)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{size}x{size}" for size in all_sizes])
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(title="Strategy", fontsize=11, title_fontsize=12, 
              frameon=True, fancybox=True, edgecolor="black")
    
    fig.tight_layout()
    save_plot(fig, "time_analysis_testing.png", save_dir)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_root", type=str, help="Path to the training outputsdirectory.")
    parser.add_argument("--outputs_root", type=str, help="Path to the testing outputs directory.")
    args = parser.parse_args()

    training_times = analyze_training(args.training_root)
    testing_times, run_info = analyze_testing(args.outputs_root)

    print_testing_details(testing_times, run_info)
    print_report(training_times, testing_times)
    plot(training_times, testing_times)
    
if __name__ == "__main__":
    main()