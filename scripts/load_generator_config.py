#!/usr/bin/env python3
"""
Parametrized load generator that accepts a workload-model config file.
"""

import asyncio
import aiohttp
import time
import json
import random
import sys
import csv
from pathlib import Path
from typing import Dict, List, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load workload-model configuration."""
    with open(config_path, "r") as f:
        return json.load(f)


def generate_workload(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate request schedule from config."""
    workload_tasks = config["workload"]
    load_profile = config["load_profile"]
    num_requests = load_profile["num_requests"]

    random.seed(load_profile.get("seed", 42))

    schedule = []
    for i in range(num_requests):
        # Select task based on weights
        weights = [task["weight"] for task in workload_tasks]
        task = random.choices(workload_tasks, weights=weights, k=1)[0]

        req = {
            "task_id": task["id"],
            "messages": task["messages"],
            "max_tokens": task["max_tokens"],
        }

        # Add optional LoRA adapter if specified in task
        if "lora_adapter" in task and task["lora_adapter"]:
            req["lora_adapter"] = task["lora_adapter"]

        schedule.append(req)

    return schedule


async def send_request(
    session: aiohttp.ClientSession,
    req: Dict,
    idx: int,
    server_url: str,
    model_name: str,
    timeout: int = 120,
):
    """Send a single request to the server."""
    t0 = time.time()
    try:
        payload = {
            "model": model_name,
            "messages": req["messages"],
            "max_tokens": req["max_tokens"],
            "temperature": 0.7,
        }
        # Add optional LoRA adapter if specified in request
        if "lora_adapter" in req and req["lora_adapter"]:
            payload["lora_adapter"] = req["lora_adapter"]

        async with session.post(server_url, json=payload, timeout=timeout) as r:
            text = await r.text()
            status = r.status
    except Exception as e:
        text = str(e)
        status = -1
    t1 = time.time()

    return {
        "idx": idx,
        "task_id": req["task_id"],
        "t0": t0,
        "t1": t1,
        "latency": t1 - t0,
        "status": status,
        "response_len": len(text),
        "response_snippet": text[:300].replace("\n", " "),
    }


async def run_load_test(config: Dict[str, Any], output_csv: str):
    """Execute load test with given configuration."""
    model_name = config["model"]["path"]
    server_port = config["server"]["port"]
    inter_arrival_lambda = config["load_profile"]["inter_arrival_lambda"]

    server_url = f"http://localhost:{server_port}/v1/chat/completions"

    print(
        f"[*] Generating workload ({config['load_profile']['num_requests']} requests)..."
    )
    schedule = generate_workload(config)

    print(f"[*] Connecting to server at {server_url}...")

    max_concurrent = 128
    semaphore = asyncio.Semaphore(max_concurrent)

    async def send_request_with_semaphore(req, idx):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                return await send_request(
                    session,
                    req,
                    idx,
                    server_url,
                    model_name,
                    config["server"]["timeout"],
                )

    tasks = []
    for i, req in enumerate(schedule):
        task = asyncio.create_task(send_request_with_semaphore(req, i))
        tasks.append(task)

    print(
        f"[*] Sending {len(tasks)} requests concurrently (max {max_concurrent} parallel)..."
    )
    results = await asyncio.gather(*tasks)

    print(f"[*] Writing results to {output_csv}...")
    keys = [
        "idx",
        "task_id",
        "t0",
        "t1",
        "latency",
        "status",
        "response_len",
        "response_snippet",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"[+] Results written to {output_csv}")
    return results


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python load_generator_config.py <workload-model.json> [output_csv]"
        )
        print(
            "Example: python load_generator_config.py config/workload-model/tinyllama-heavy.json results/requests.csv"
        )
        sys.exit(1)

    config_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "request_log.csv"

    config = load_config(config_path)
    print(f"[+] Loaded config: {config['metadata']['name']}")
    print(f"    Model: {config['model']['name']}")
    print(f"    Requests: {config['load_profile']['num_requests']}")
    print()

    asyncio.run(run_load_test(config, output_csv))


if __name__ == "__main__":
    main()
