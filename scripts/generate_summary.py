#!/usr/bin/env python3
"""
Generate summary statistics from profiling results.
"""

import json
import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path
from typing import Dict, Any


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
    # Basic request-level aggregates
    total_requests = int(len(req_df))
    successful = int((req_df["status"] == 200).sum())
    failed = int((req_df["status"] != 200).sum())

    # Latency stats
    latency_mean = float(req_df["latency"].mean())
    latency_median = float(req_df["latency"].median())
    latency_p99 = float(req_df["latency"].quantile(0.99))
    latency_p95 = float(req_df["latency"].quantile(0.95))
    latency_p90 = float(req_df["latency"].quantile(0.90))
    latency_max = float(req_df["latency"].max())

    # Response length and token stats
    resp_len_stats = {
        "count": int(req_df["response_len"].count()),
        "mean": float(req_df["response_len"].mean()),
        "median": float(req_df["response_len"].median()),
        "p90": float(req_df["response_len"].quantile(0.9)),
        "p95": float(req_df["response_len"].quantile(0.95)),
        "p99": float(req_df["response_len"].quantile(0.99)),
        "min": int(req_df["response_len"].min()),
        "max": int(req_df["response_len"].max()),
    }

    token_stats = {
        "count": int(req_df["completion_tokens"].count()),
        "sum": int(req_df["completion_tokens"].sum()),
        "mean": float(req_df["completion_tokens"].mean()),
        "median": float(req_df["completion_tokens"].median()),
        "p90": float(req_df["completion_tokens"].quantile(0.9)),
        "p95": float(req_df["completion_tokens"].quantile(0.95)),
        "p99": float(req_df["completion_tokens"].quantile(0.99)),
    }

    # Status code distribution
    status_counts = req_df["status"].value_counts().to_dict()

    # Inter-arrival times (based on t0)
    sorted_t0 = np.sort(req_df["t0"].values)
    inter_arrival = np.diff(sorted_t0) if len(sorted_t0) > 1 else np.array([])
    inter_arrival_stats = {
        "mean": float(inter_arrival.mean()) if inter_arrival.size else 0.0,
        "median": float(np.median(inter_arrival)) if inter_arrival.size else 0.0,
        "p95": float(np.percentile(inter_arrival, 95)) if inter_arrival.size else 0.0,
        "p99": float(np.percentile(inter_arrival, 99)) if inter_arrival.size else 0.0,
    }

    # Peak concurrency (sweep-line over start/end times)
    events = []
    for _, r in req_df.iterrows():
        events.append((float(r["t0"]), 1))
        events.append((float(r["t1"]), -1))
    events.sort()
    cur = 0
    peak_concurrency = 0
    for _, delta in events:
        cur += delta
        if cur > peak_concurrency:
            peak_concurrency = cur

    # (throughput series removed per request; see throughput_rps for average)

    # Per-task aggregations
    per_task = {}
    for task_id, group in req_df.groupby("task_id"):
        per_task[str(task_id)] = {
            "count": int(len(group)),
            "success_rate": float((group["status"] == 200).sum() / len(group)),
            "latency_mean": float(group["latency"].mean()),
            "latency_p95": float(group["latency"].quantile(0.95)),
            "latency_p99": float(group["latency"].quantile(0.99)),
            "tokens_total": int(group["completion_tokens"].sum()),
        }

    # GPU percentiles
    gpu_stats = {
        "gpu_util": {
            "mean": float(gpu_df["gpu_util"].mean()),
            "p50": float(gpu_df["gpu_util"].quantile(0.5)),
            "p90": float(gpu_df["gpu_util"].quantile(0.9)),
            "p95": float(gpu_df["gpu_util"].quantile(0.95)),
            "p99": float(gpu_df["gpu_util"].quantile(0.99)),
            "max": float(gpu_df["gpu_util"].max()),
        },
        "mem_used": {
            "mean": float(gpu_df["mem_used"].mean()),
            "p90": float(gpu_df["mem_used"].quantile(0.9)),
            "p99": float(gpu_df["mem_used"].quantile(0.99)),
            "max": float(gpu_df["mem_used"].max()),
        },
        "power": {
            "mean": float(gpu_df["power"].mean()),
            "p90": float(gpu_df["power"].quantile(0.9)),
            "p99": float(gpu_df["power"].quantile(0.99)),
        },
    }

    summary: Dict[str, Any] = {
        "config_file": config_file,
        "model": model_name,
        "workload": workload_name,
        "timestamp": timestamp,
        "requests": {
            "total": total_requests,
            "successful": successful,
            "failed": failed,
            "status_counts": status_counts,
            "latency_mean": latency_mean,
            "latency_median": latency_median,
            "latency_p90": latency_p90,
            "latency_p95": latency_p95,
            "latency_p99": latency_p99,
            "latency_max": latency_max,
            "throughput_rps": float(
                len(req_df) / (req_df["t1"].max() - req_df["t0"].min())
            ),
            "tokens": token_stats,
            "response_length": resp_len_stats,
            "latency_per_token_mean": float(
                (req_df["latency"] / req_df["completion_tokens"].replace(0, 1)).mean()
            ),
            "latency_per_token_p99": float(
                (
                    req_df["latency"] / req_df["completion_tokens"].replace(0, 1)
                ).quantile(0.99)
            ),
            "inter_arrival": inter_arrival_stats,
            "peak_concurrency": int(peak_concurrency),
            "per_task": per_task,
        },
        "gpu": gpu_stats,
    }

    # Write summary JSON
    summary_path = Path(result_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    # Also write some lightweight CSVs for downstream plotting/debugging
    # Per-task CSV
    per_task_df = (
        req_df.groupby("task_id")
        .agg(
            count=("idx", "count"),
            mean_latency=("latency", "mean"),
            p95_latency=("latency", lambda x: x.quantile(0.95)),
        )
        .reset_index()
    )
    per_task_csv = Path(result_dir) / "per_task_summary.csv"
    per_task_df.to_csv(per_task_csv, index=False)
    print(f"[+] Wrote per-task summary CSV: {per_task_csv}")

    # Print summary to stdout
    print("Summary Statistics:")
    print(
        f"  Requests: {summary['requests']['total']} total, {summary['requests']['successful']} successful"
    )
    print(
        f"  Latency: {summary['requests']['latency_mean']:.3f}s mean, {summary['requests']['latency_p99']:.3f}s p99"
    )
    print(f"  Throughput: {summary['requests']['throughput_rps']:.2f} req/s")
    # Tokens: total and tokens/sec (compute safely)
    token_sum = summary["requests"].get("tokens", {}).get("sum", 0)
    duration = (
        float(req_df["t1"].max() - req_df["t0"].min()) if len(req_df) > 0 else 0.0
    )
    tokens_per_second = token_sum / duration if duration > 0 else 0.0
    print(f"  Tokens: {token_sum} total, {tokens_per_second:.2f} tok/s")
    print(
        f"  Per-token latency: {summary['requests']['latency_per_token_mean']:.4f}s mean, {summary['requests']['latency_per_token_p99']:.4f}s p99"
    )
    # GPU utilization (mean, max)
    gpu_util_mean = summary.get("gpu", {}).get("gpu_util", {}).get("mean", 0.0)
    gpu_util_max = summary.get("gpu", {}).get("gpu_util", {}).get("max", 0.0)
    print(f"  GPU Util: {gpu_util_mean:.1f}% mean, {gpu_util_max:.1f}% max")


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary statistics from profiling results."
    )
    parser.add_argument(
        "-r", "--request-log", dest="request_log", help="path to request_log.csv"
    )
    parser.add_argument(
        "-g", "--gpu-log", dest="gpu_log", help="path to gpu metrics CSV"
    )
    parser.add_argument(
        "-c", "--config", dest="config_file", help="path to config JSON"
    )
    parser.add_argument("-m", "--model", dest="model_name", help="model name")
    parser.add_argument("-w", "--workload", dest="workload_name", help="workload name")
    parser.add_argument("-t", "--timestamp", dest="timestamp", help="timestamp string")
    parser.add_argument(
        "-o", "--out", dest="result_dir", help="result directory to write summary"
    )

    args, unknown = parser.parse_known_args()

    # Backwards-compatible positional fallback
    if not args.request_log and len(sys.argv) >= 2:
        args.request_log = sys.argv[1]
    if not args.gpu_log and len(sys.argv) >= 3:
        args.gpu_log = sys.argv[2]
    if not args.config_file and len(sys.argv) >= 4:
        args.config_file = sys.argv[3]
    if not args.model_name and len(sys.argv) >= 5:
        args.model_name = sys.argv[4]
    if not args.workload_name and len(sys.argv) >= 6:
        args.workload_name = sys.argv[5]
    if not args.timestamp and len(sys.argv) >= 7:
        args.timestamp = sys.argv[6]
    if not args.result_dir and len(sys.argv) >= 8:
        args.result_dir = sys.argv[7]

    if not (
        args.request_log
        and args.gpu_log
        and args.config_file
        and args.model_name
        and args.workload_name
        and args.timestamp
        and args.result_dir
    ):
        parser.print_usage()
        sys.exit(1)

    request_log = args.request_log
    gpu_log = args.gpu_log
    config_file = args.config_file
    model_name = args.model_name
    workload_name = args.workload_name
    timestamp = args.timestamp
    result_dir = args.result_dir

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
