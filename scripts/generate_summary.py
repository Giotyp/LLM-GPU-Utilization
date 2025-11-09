#!/usr/bin/env python3
"""
Generate summary statistics from profiling results.
"""

import json
import pandas as pd
import sys
from pathlib import Path


def generate_summary(
    request_log: str,
    gpu_log: str,
    config_file: str,
    model_name: str,
    workload_name: str,
    timestamp: str,
    result_dir: str,
):
    """Generate summary statistics from profiling results."""

    # Load data
    req_df = pd.read_csv(request_log)
    gpu_df = pd.read_csv(
        gpu_log,
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

    # Generate summary
    summary = {
        "config_file": config_file,
        "model": model_name,
        "workload": workload_name,
        "timestamp": timestamp,
        "requests": {
            "total": int(len(req_df)),
            "successful": int((req_df["status"] == 200).sum()),
            "failed": int((req_df["status"] != 200).sum()),
            "latency_mean": float(req_df["latency"].mean()),
            "latency_median": float(req_df["latency"].median()),
            "latency_p99": float(req_df["latency"].quantile(0.99)),
            "latency_max": float(req_df["latency"].max()),
            "throughput_rps": float(
                len(req_df) / (req_df["t1"].max() - req_df["t0"].min())
            ),
        },
        "gpu": {
            "utilization_mean": float(gpu_df["gpu_util"].mean()),
            "utilization_max": float(gpu_df["gpu_util"].max()),
            "memory_util_mean": float(gpu_df["mem_util"].mean()),
            "memory_util_max": float(gpu_df["mem_util"].max()),
            "temperature_max": float(gpu_df["temp"].max()),
            "power_draw_mean": float(gpu_df["power"].mean()),
        },
    }

    # Write summary JSON
    summary_path = Path(result_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary to stdout
    print("Summary Statistics:")
    print(
        f"  Requests: {summary['requests']['total']} total, {summary['requests']['successful']} successful"
    )
    print(
        f"  Latency: {summary['requests']['latency_mean']:.3f}s mean, {summary['requests']['latency_p99']:.3f}s p99"
    )
    print(f"  Throughput: {summary['requests']['throughput_rps']:.2f} req/s")
    print(
        f"  GPU Util: {summary['gpu']['utilization_mean']:.1f}% mean, {summary['gpu']['utilization_max']:.1f}% max"
    )


def main():
    if len(sys.argv) != 8:
        print(
            "Usage: python generate_summary.py <request_log> <gpu_log> <config_file> <model_name> <workload_name> <timestamp> <result_dir>"
        )
        sys.exit(1)

    request_log = sys.argv[1]
    gpu_log = sys.argv[2]
    config_file = sys.argv[3]
    model_name = sys.argv[4]
    workload_name = sys.argv[5]
    timestamp = sys.argv[6]
    result_dir = sys.argv[7]

    generate_summary(
        request_log,
        gpu_log,
        config_file,
        model_name,
        workload_name,
        timestamp,
        result_dir,
    )


if __name__ == "__main__":
    main()
