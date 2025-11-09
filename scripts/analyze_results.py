#!/usr/bin/env python3
"""
Enhanced analysis and visualization for GPU profiling results.
Supports single run analysis and comparative analysis across multiple runs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)


class ProfileAnalyzer:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.req_df = None
        self.gpu_df = None
        self.summary = None
        self.config = None
        self.load_data()

    def load_data(self):
        """Load all result files."""
        req_file = self.result_dir / "request_log.csv"
        gpu_file = self.result_dir / "gpu_metrics.csv"
        summary_file = self.result_dir / "summary.json"
        config_file = self.result_dir / "config.json"

        if req_file.exists():
            self.req_df = pd.read_csv(req_file)
        if gpu_file.exists():
            self.gpu_df = pd.read_csv(
                gpu_file,
                names=[
                    "timestamp",
                    "gpu",
                    "name",
                    "gpu_util",
                    "mem_util",
                    "mem_used",
                    "mem_total",
                    "temp",
                    "power",
                ],
            )
        if summary_file.exists():
            with open(summary_file) as f:
                self.summary = json.load(f)
        if config_file.exists():
            with open(config_file) as f:
                self.config = json.load(f)

    def plot_gpu_utilization(self, output_path: str = None):
        """Plot GPU utilization over time."""
        if self.gpu_df is None:
            print("No GPU data available")
            return

        fig, ax = plt.subplots()
        ax.plot(
            self.gpu_df.index,
            self.gpu_df["gpu_util"],
            label="GPU Utilization",
            linewidth=2,
        )
        ax.fill_between(self.gpu_df.index, self.gpu_df["gpu_util"], alpha=0.3)

        ax.set_title(
            f"GPU Utilization - {self.config['model']['name']}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Samples (0.5s interval)")
        ax.set_ylabel("Utilization (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_path is None:
            output_path = self.result_dir / "plots" / "gpu_utilization.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved: {output_path}")

    def plot_latency_distribution(self, output_path: str = None):
        """Plot latency distribution."""
        if self.req_df is None:
            print("No request data available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(
            self.req_df["latency"],
            bins=50,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )
        ax1.set_xlabel("Latency (seconds)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Latency Distribution")
        ax1.grid(True, alpha=0.3)

        # CDF
        sorted_latency = np.sort(self.req_df["latency"])
        cdf = np.arange(1, len(sorted_latency) + 1) / len(sorted_latency)
        ax2.plot(sorted_latency, cdf * 100, linewidth=2, color="darkblue")
        ax2.set_xlabel("Latency (seconds)")
        ax2.set_ylabel("Cumulative %")
        ax2.set_title("Latency CDF")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=99, color="r", linestyle="--", label="p99")
        ax2.legend()

        if output_path is None:
            output_path = self.result_dir / "plots" / "latency_distribution.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved: {output_path}")

    def plot_gpu_memory(self, output_path: str = None):
        """Plot GPU memory utilization over time."""
        if self.gpu_df is None:
            print("No GPU data available")
            return

        fig, ax = plt.subplots()
        ax.plot(
            self.gpu_df.index,
            self.gpu_df["mem_util"],
            label="Memory Utilization",
            linewidth=2,
            color="orange",
        )
        ax.fill_between(
            self.gpu_df.index, self.gpu_df["mem_util"], alpha=0.3, color="orange"
        )

        ax.set_title(
            f"GPU Memory Utilization - {self.config['model']['name']}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Samples (0.5s interval)")
        ax.set_ylabel("Memory Utilization (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_path is None:
            output_path = self.result_dir / "plots" / "memory_utilization.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved: {output_path}")

    def plot_latency_over_time(self, output_path: str = None):
        """Plot latency over request sequence."""
        if self.req_df is None:
            print("No request data available")
            return

        fig, ax = plt.subplots()
        ax.scatter(
            self.req_df["idx"],
            self.req_df["latency"],
            alpha=0.6,
            s=20,
            color="steelblue",
        )

        # Add rolling average
        window = min(50, len(self.req_df) // 10)
        if window > 1:
            rolling_mean = self.req_df["latency"].rolling(window=window).mean()
            ax.plot(
                self.req_df["idx"],
                rolling_mean,
                color="red",
                linewidth=2,
                label=f"Rolling avg ({window} requests)",
            )

        ax.set_title(
            f"Latency Over Time - {self.config['model']['name']}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Request Index")
        ax.set_ylabel("Latency (seconds)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_path is None:
            output_path = self.result_dir / "plots" / "latency_over_time.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved: {output_path}")

    def print_summary(self):
        """Print summary statistics."""
        if not self.summary:
            return

        print("\n" + "=" * 60)
        print(f"Profile Summary: {self.config['model']['name']}")
        print("=" * 60)
        print(f"\nRequests:")
        print(f"  Total: {self.summary['requests']['total']}")
        print(f"  Successful: {self.summary['requests']['successful']}")
        print(f"  Failed: {self.summary['requests']['failed']}")
        print(f"\nLatency:")
        print(f"  Mean: {self.summary['requests']['latency_mean']:.3f}s")
        print(f"  Median: {self.summary['requests']['latency_median']:.3f}s")
        print(f"  P99: {self.summary['requests']['latency_p99']:.3f}s")
        print(f"  Max: {self.summary['requests']['latency_max']:.3f}s")
        print(f"  Throughput: {self.summary['requests']['throughput_rps']:.2f} req/s")
        print(f"\nGPU:")
        print(f"  Util Mean: {self.summary['gpu']['utilization_mean']:.1f}%")
        print(f"  Util Max: {self.summary['gpu']['utilization_max']:.1f}%")
        print(f"  Memory Mean: {self.summary['gpu']['memory_util_mean']:.1f}%")
        print(f"  Memory Max: {self.summary['gpu']['memory_util_max']:.1f}%")
        print(f"  Temp Max: {self.summary['gpu']['temperature_max']:.1f}°C")
        print(f"  Power Mean: {self.summary['gpu']['power_draw_mean']:.1f}W")
        print("=" * 60 + "\n")


def compare_results(result_dirs: List[str], output_dir: str = None):
    """Compare metrics across multiple profiling runs."""
    analyzers = [ProfileAnalyzer(d) for d in result_dirs]

    # Collect summaries
    summaries = []
    for analyzer in analyzers:
        if analyzer.summary:
            summaries.append(
                {
                    "name": f"{analyzer.config['model']['name']}_{analyzer.config['metadata']['name'].replace(' ', '_')}",
                    "data": analyzer.summary,
                }
            )

    if not summaries:
        print("No summaries found to compare")
        return

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = [s["name"] for s in summaries]
    latency_means = [s["data"]["requests"]["latency_mean"] for s in summaries]
    throughputs = [s["data"]["requests"]["throughput_rps"] for s in summaries]
    gpu_utils = [s["data"]["gpu"]["utilization_mean"] for s in summaries]
    memory_utils = [s["data"]["gpu"]["memory_util_mean"] for s in summaries]

    # Latency comparison
    axes[0, 0].bar(range(len(names)), latency_means, color="steelblue", alpha=0.7)
    axes[0, 0].set_ylabel("Mean Latency (s)")
    axes[0, 0].set_title("Latency Comparison")
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha="right")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Throughput comparison
    axes[0, 1].bar(range(len(names)), throughputs, color="green", alpha=0.7)
    axes[0, 1].set_ylabel("Throughput (req/s)")
    axes[0, 1].set_title("Throughput Comparison")
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, rotation=45, ha="right")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # GPU Util comparison
    axes[1, 0].bar(range(len(names)), gpu_utils, color="orange", alpha=0.7)
    axes[1, 0].set_ylabel("GPU Utilization (%)")
    axes[1, 0].set_title("GPU Utilization Comparison")
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha="right")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Memory Util comparison
    axes[1, 1].bar(range(len(names)), memory_utils, color="purple", alpha=0.7)
    axes[1, 1].set_ylabel("Memory Utilization (%)")
    axes[1, 1].set_title("Memory Utilization Comparison")
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=45, ha="right")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_dir is None:
        output_dir = ".results"

    output_path = Path(output_dir) / "comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparison: {output_path}")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python analyze_results.py <result_directory> [result_directory2 ...]"
        )
        print("Examples:")
        print("  python analyze_results.py .results/tinyllama_light_20251107_120000/")
        print("  python analyze_results.py .results/*/  # Compare all runs")
        sys.exit(1)

    result_dirs = sys.argv[1:]

    if len(result_dirs) == 1:
        # Single run analysis
        analyzer = ProfileAnalyzer(result_dirs[0])
        analyzer.print_summary()

        print("[*] Generating plots...")
        analyzer.plot_gpu_utilization()
        analyzer.plot_gpu_memory()
        analyzer.plot_latency_distribution()
        analyzer.plot_latency_over_time()

    else:
        # Comparative analysis
        print("[*] Running comparative analysis...")
        for result_dir in result_dirs:
            analyzer = ProfileAnalyzer(result_dir)
            analyzer.print_summary()

        compare_results(result_dirs)


if __name__ == "__main__":
    main()
