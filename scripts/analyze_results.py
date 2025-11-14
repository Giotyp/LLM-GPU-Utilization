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
import re

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

    def plot_per_token_latency(self, output_path: str = None):
        """Plot per-token latency distribution."""
        if self.req_df is None or "completion_tokens" not in self.req_df.columns:
            print("No request data or completion_tokens available")
            return

        # Filter out requests with 0 tokens to avoid division by zero
        valid_df = self.req_df[self.req_df["completion_tokens"] > 0].copy()
        if len(valid_df) == 0:
            print("No valid requests with completion tokens")
            return

        valid_df["latency_per_token"] = (
            valid_df["latency"] / valid_df["completion_tokens"]
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(
            valid_df["latency_per_token"],
            bins=50,
            color="darkgreen",
            alpha=0.7,
            edgecolor="black",
        )
        ax1.set_xlabel("Latency per Token (seconds)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Per-Token Latency Distribution")
        ax1.grid(True, alpha=0.3)

        # CDF
        sorted_latency = np.sort(valid_df["latency_per_token"])
        cdf = np.arange(1, len(sorted_latency) + 1) / len(sorted_latency)
        ax2.plot(sorted_latency, cdf * 100, linewidth=2, color="darkgreen")
        ax2.set_xlabel("Latency per Token (seconds)")
        ax2.set_ylabel("Cumulative %")
        ax2.set_title("Per-Token Latency CDF")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=99, color="r", linestyle="--", label="p99")
        ax2.legend()

        if output_path is None:
            output_path = self.result_dir / "plots" / "per_token_latency.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved: {output_path}")

    def print_summary(self):
        """Print summary statistics."""
        if not self.summary:
            return

        print("\n" + "=" * 60)
        print(f"Profile Summary: {self.config['metadata']['name']}")
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
        print(
            f"  Throughput: {self.summary['requests'].get('throughput_rps', 0.0):.2f} req/s"
        )

        # Tokens: new summary stores token stats under 'tokens'
        tokens = self.summary["requests"].get("tokens", {})
        tokens_sum = tokens.get("sum", None)
        if tokens_sum is not None:
            # compute tokens/sec from request log if available
            tokens_per_second = None
            if self.req_df is not None and len(self.req_df) > 0:
                duration = float(self.req_df["t1"].max() - self.req_df["t0"].min())
                tokens_per_second = tokens_sum / duration if duration > 0 else 0.0

            print(f"  Tokens Generated: {tokens_sum}")
            if tokens_per_second is not None:
                print(f"  Token Throughput: {tokens_per_second:.2f} tok/s")

        # Per-token latency (if present in summary)
        if "latency_per_token_mean" in self.summary["requests"]:
            print(
                f"  Per-Token Latency Mean: {self.summary['requests']['latency_per_token_mean']:.4f}s"
            )
        if "latency_per_token_p99" in self.summary["requests"]:
            print(
                f"  Per-Token Latency P99: {self.summary['requests']['latency_per_token_p99']:.4f}s"
            )
        print(f"\nGPU:")
        # GPU metrics: new summary nests fields under gpu_util / mem_used / power
        gpu_util_mean = self.summary.get("gpu", {}).get("gpu_util", {}).get("mean", 0.0)
        gpu_util_max = self.summary.get("gpu", {}).get("gpu_util", {}).get("max", 0.0)
        mem_used_mean = self.summary.get("gpu", {}).get("mem_used", {}).get("mean", 0.0)
        mem_used_max = self.summary.get("gpu", {}).get("mem_used", {}).get("max", 0.0)
        power_mean = self.summary.get("gpu", {}).get("power", {}).get("mean", 0.0)

        print(f"  Util Mean: {gpu_util_mean:.1f}%")
        print(f"  Util Max: {gpu_util_max:.1f}%")
        print(f"  Memory Mean: {mem_used_mean:.1f}")
        print(f"  Memory Max: {mem_used_max:.1f}")
        print(f"  Power Mean: {power_mean:.1f}W")
        print("=" * 60 + "\n")


def compare_results(result_dirs: List[str], output_dir: str = None):
    """Compare metrics across multiple profiling runs."""
    analyzers = [ProfileAnalyzer(d) for d in result_dirs]

    # Collect summaries
    summaries = []
    for analyzer in analyzers:
        if analyzer.summary:
            # compute token throughput if request log is available
            tokens_sum = analyzer.summary["requests"].get("tokens", {}).get("sum", None)
            token_throughput = None
            if (
                tokens_sum is not None
                and analyzer.req_df is not None
                and len(analyzer.req_df) > 0
            ):
                duration = float(
                    analyzer.req_df["t1"].max() - analyzer.req_df["t0"].min()
                )
                token_throughput = tokens_sum / duration if duration > 0 else 0.0

            summaries.append(
                {
                    "name": f"{analyzer.config['metadata']['name']}",
                    "data": analyzer.summary,
                    "token_throughput": token_throughput,
                }
            )

    if not summaries:
        print("No summaries found to compare")
        return

    # Create comparison plots
    # Sort summaries so that for each base model we show: base (no LoRA) first,
    # then increasing LoRA percentages in order. This makes plots
    # comparable across runs with different LoRA usage.

    def _extract_base_and_pct(name: str):
        # Try to find a percentage like '1%', '30%', '100%'
        pct = 0
        m = re.search(r"(\d+)\s*%", name)
        if not m:
            # fallback: look for patterns like '1p' or '100p'
            m = re.search(r"(\d+)p\b", name, flags=re.I)
        if m:
            try:
                pct = int(m.group(1))
            except Exception:
                pct = 0

        # Remove LoRA mentions and numeric suffixes to get the base model name
        base = re.sub(r"(?i)\b(lora)\b", "", name)
        base = re.sub(r"(\d+)\s*%", "", base)
        base = re.sub(r"(\d+)p\b", "", base, flags=re.I)
        base = re.sub(r"[_\-]+", " ", base)
        base = re.sub(r"\s+", " ", base).strip()
        return base.lower(), pct

    desired_order = [0, 1, 30, 100]
    prepared = []
    for s in summaries:
        base, pct = _extract_base_and_pct(s["name"])
        if pct in desired_order:
            order_idx = desired_order.index(pct)
        else:
            # place unknown percentages after the known order, ordered by pct value
            order_idx = len(desired_order) + (pct if pct is not None else 0)
        prepared.append((base, order_idx, s))

    prepared.sort(key=lambda x: (x[0], x[1]))
    slist = [p[2] for p in prepared]

    names = [s["name"] for s in slist]
    latency_means = [s["data"]["requests"]["latency_mean"] for s in slist]
    throughputs = [s["data"]["requests"].get("throughput_rps", 0.0) for s in slist]
    gpu_utils = [s["data"]["gpu"].get("gpu_util", {}).get("mean", 0.0) for s in slist]
    memory_utils = [
        s["data"]["gpu"].get("mem_used", {}).get("mean", 0.0) for s in slist
    ]

    # Color list so the same LoRA% always maps to the same color across models.
    pct_list = []
    for s in slist:
        _, pct = (
            _extract_base_and_pct(s["name"])
            if "_extract_base_and_pct" in locals()
            else (None, 0)
        )
        pct_list.append(pct)

    pal = sns.color_palette("tab10")
    color_map = {0: pal[0], 1: pal[1], 30: pal[2], 100: pal[3]}
    colors = [color_map.get(p if p is not None else 0, "gray") for p in pct_list]

    # Check if per-token metrics are available (we computed token_throughput earlier)
    has_token_metrics = any(s.get("token_throughput") is not None for s in slist)
    if has_token_metrics:
        token_throughputs = [s.get("token_throughput", 0.0) for s in slist]
        latency_per_token = [
            s["data"]["requests"].get("latency_per_token_mean", 0.0) for s in slist
        ]
        fig, axes = plt.subplots(2, 2, figsize=(14, 15))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    # Select axes handles depending on layout and draw bars with consistent colors
    if has_token_metrics:
        ax_lat = axes[0, 0]
        ax_thr = axes[0, 1]
    else:
        ax_lat = axes[0]
        ax_thr = axes[1]

    # Latency comparison
    ax_lat.bar(range(len(names)), latency_means, color=colors, alpha=0.9)
    ax_lat.set_ylabel("Mean Latency (s)")
    ax_lat.set_title("Latency Comparison")
    ax_lat.set_xticks(range(len(names)))
    ax_lat.set_xticklabels(names, rotation=45, ha="right")
    ax_lat.grid(True, alpha=0.3, axis="y")

    # Throughput comparison
    ax_thr.bar(range(len(names)), throughputs, color=colors, alpha=0.9)
    ax_thr.set_ylabel("Throughput (req/s)")
    ax_thr.set_title("Throughput Comparison")
    ax_thr.set_xticks(range(len(names)))
    ax_thr.set_xticklabels(names, rotation=45, ha="right")
    ax_thr.grid(True, alpha=0.3, axis="y")

    # # GPU Util comparison
    # axes[1, 0].bar(range(len(names)), gpu_utils, color="orange", alpha=0.7)
    # axes[1, 0].set_ylabel("GPU Utilization (%)")
    # axes[1, 0].set_title("GPU Utilization Comparison")
    # axes[1, 0].set_xticks(range(len(names)))
    # axes[1, 0].set_xticklabels(names, rotation=45, ha="right")
    # axes[1, 0].grid(True, alpha=0.3, axis="y")

    # # Memory Util comparison
    # axes[1, 1].bar(range(len(names)), memory_utils, color="purple", alpha=0.7)
    # axes[1, 1].set_ylabel("Memory Utilization (%)")
    # axes[1, 1].set_title("Memory Utilization Comparison")
    # axes[1, 1].set_xticks(range(len(names)))
    # axes[1, 1].set_xticklabels(names, rotation=45, ha="right")
    # axes[1, 1].grid(True, alpha=0.3, axis="y")

    if has_token_metrics:
        # Token Throughput comparison
        ax_tok = axes[1, 1]
        ax_tok.bar(range(len(names)), token_throughputs, color=colors, alpha=0.9)
        ax_tok.set_ylabel("Token Throughput (tok/s)")
        ax_tok.set_title("Token Throughput Comparison")
        ax_tok.set_xticks(range(len(names)))
        ax_tok.set_xticklabels(names, rotation=45, ha="right")
        ax_tok.grid(True, alpha=0.3, axis="y")

        # Per-token Latency comparison
        ax_per_tok = axes[1, 0]
        ax_per_tok.bar(range(len(names)), latency_per_token, color=colors, alpha=0.9)
        ax_per_tok.set_ylabel("Latency per Token (s)")
        ax_per_tok.set_title("Per-Token Latency Comparison")
        ax_per_tok.set_xticks(range(len(names)))
        ax_per_tok.set_xticklabels(names, rotation=45, ha="right")
        ax_per_tok.grid(True, alpha=0.3, axis="y")

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
        analyzer.plot_per_token_latency()

    else:
        # Comparative analysis
        print("[*] Running comparative analysis...")
        for result_dir in result_dirs:
            analyzer = ProfileAnalyzer(result_dir)
            analyzer.print_summary()

        compare_results(result_dirs)


if __name__ == "__main__":
    main()
