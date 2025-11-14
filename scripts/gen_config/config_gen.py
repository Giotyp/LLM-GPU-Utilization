#!/usr/bin/env python3
"""
Generate profiling configuration JSON files from generic templates.

Usage examples:

# Basic: generate config for model at ./models/Llama2-7b with 256 max tokens and summary+code-review
python scripts/gen_config/config_gen.py \
    --model_path ./models/Llama2-7b \
    --max_tokens 256 \
    --messages summary code-review \
    --config_dir ./config

# With LoRA adapters and 1% LoRA usage
python scripts/gen_config/config_gen.py \
    --model_path ./models/Llama2-7b \
    --max_tokens 256 \
    --messages summary code-review qa-technical \
    --lora_perc 0.01 \
    --loras alpaca-lora-7b,llama2-7b-chat-lora-adaptor \
    --config_dir ./config

This script expects the following template files in the same directory:
 - model_params.json  (contains generic metadata, model, server, load_profile)
 - workload.json      (contains a dictionary of workload message templates)

The generated configuration will be written to: <config_dir>/workload-<model_name>/<model_name>-<num_requests>[-lora-<p>p].json

"""

import argparse
import json
import os
from pathlib import Path
from typing import List
import math

HERE = Path(__file__).parent


def _read_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)


def _safe_model_name(name: str):
    # Lowercase, replace spaces with dashes and remove unsafe chars
    s = name.lower().strip()
    s = s.replace(" ", "-")
    s = s.replace("/", "-")
    s = s.replace("\\", "-")
    return s


def generate_config(
    model_path: str,
    max_tokens: int,
    messages: List[str],
    lora_perc: float,
    loras: List[str],
    config_dir: str,
    model_params_path: Path = None,
    workload_path: Path = None,
):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")

    # Load the model's config.json to get the canonical model name
    model_cfg_file = model_path / "config.json"
    if not model_cfg_file.exists():
        raise FileNotFoundError(f"model config.json not found in {model_path}")

    model_cfg = _read_json(model_cfg_file)
    model_name = (
        model_cfg.get("model_name")
        or model_cfg.get("name")
        or model_cfg.get("_name")
        or model_path.name
    )
    safe_name = _safe_model_name(model_name)

    # load templates
    model_params_path = model_params_path or (HERE / "model_params.json")
    workload_path = workload_path or (HERE / "workload.json")
    model_params = _read_json(model_params_path)
    workload_templates = _read_json(workload_path).get("workloads", {})

    # Start building config from model_params
    cfg = dict(model_params)
    # override model.path and model.name
    cfg["model"]["path"] = str(model_path)
    cfg["model"]["name"] = model_name

    # Compose workload entries
    selected_msgs = []
    for m in messages:
        if m not in workload_templates:
            raise ValueError(f"message id '{m}' not found in workload templates")
        tpl = workload_templates[m]
        # copy template and add max_tokens (weight/lora handled later)
        entry = {
            "id": tpl.get("id", m),
            "description": tpl.get("description", ""),
            "messages": tpl.get("messages", []),
        }
        selected_msgs.append(entry)

    n = len(selected_msgs)
    lora_total = float(lora_perc) if lora_perc is not None else 0.0
    loras = loras or []

    # Determine weights: base_total and lora_total
    base_total = max(0.0, 1.0 - lora_total)
    # For base messages each gets base_total / n
    base_weight = base_total / n if n > 0 else 0.0
    # For lora-enabled messages we create a LoRA variant per message with weight lora_total / n
    lora_weight = lora_total / n if n > 0 else 0.0

    final_workload = []
    # assign base entries
    for entry in selected_msgs:
        e = dict(entry)
        e["max_tokens"] = int(max_tokens)
        e["weight"] = round(base_weight, 6)
        final_workload.append(e)

    # assign LoRA entries (one per selected message) if requested
    if lora_total > 0 and len(loras) > 0:
        # cycle through provided loras
        for idx, entry in enumerate(selected_msgs):
            lora_adapter = loras[idx % len(loras)]
            e = dict(entry)
            e["id"] = f"{entry['id']}_lora_{idx}"
            e["max_tokens"] = int(max_tokens)
            e["weight"] = round(lora_weight, 6)
            e["lora_adapter"] = lora_adapter
            final_workload.append(e)
    elif lora_total > 0 and len(loras) == 0:
        # If user asked for lora_perc but provided no adapters, treat as base (no lora)
        pass

    # Normalize weights to sum to 1.0 (avoid float rounding issues), then
    # convert to two-decimal weights that sum exactly to 1.00.
    weights = [e.get("weight", 0.0) for e in final_workload]
    total_w = sum(weights)
    if total_w > 0:
        # normalize to sum 1.0 first
        norm = [w / total_w for w in weights]

        # Convert to integer 'cents' (0..100) while preserving distribution.
        scaled = [w * 100.0 for w in norm]
        floors = [int(math.floor(s)) for s in scaled]
        remainder = int(round(100 - sum(floors)))

        # Distribute remaining 1-cent increments to the entries with largest fractional parts
        fracs = [(scaled[i] - floors[i], i) for i in range(len(scaled))]
        fracs.sort(reverse=True)
        for j in range(remainder):
            idx = fracs[j][1]
            floors[idx] += 1

        # Assign back as two-decimal weights
        for i, e in enumerate(final_workload):
            e["weight"] = round(floors[i] / 100.0, 2)

    cfg["workload"] = final_workload

    # Update metadata.name and description
    num_requests = cfg.get("load_profile", {}).get("num_requests", 1000)
    lora_pct_display = f"{int(lora_total * 100)}% LoRA" if lora_total > 0 else "No LoRA"
    cfg["metadata"]["name"] = f"{model_name} {num_requests} - {lora_pct_display}"
    cfg["metadata"][
        "description"
    ] = f"Auto-generated profile for {model_name}. {lora_pct_display}."

    # Prepare output path
    out_dir = Path(config_dir) / f"workload-{safe_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_suffix = f"-lora-{int(lora_total*100)}p" if lora_total > 0 else ""
    out_file = out_dir / f"{safe_name}-{num_requests}{file_suffix}.json"

    with open(out_file, "w") as f:
        json.dump(cfg, f, indent=4)

    print(f"[+] Generated config: {out_file}")
    return out_file


def _parse_list_arg(val: str):
    # Accept comma-separated or space-separated values
    if not val:
        return []
    if isinstance(val, list):
        return val
    if "," in val:
        return [v.strip() for v in val.split(",") if v.strip()]
    return [v.strip() for v in val.split() if v.strip()]


def _parse_messages_arg(val):
    """Normalize the --messages argument.

    Accepts:
    - a list of strings (from argparse nargs='*') where items may include trailing commas
    - a single comma-separated string
    - space-separated items
    Returns a clean list of message ids (no commas, trimmed).
    """
    if not val:
        return []
    # If argparse already gave a list (nargs='*')
    if isinstance(val, list):
        out = []
        for item in val:
            if item is None:
                continue
            # split any comma-containing items
            parts = [p.strip() for p in item.split(",") if p.strip()]
            out.extend(parts)
        # final pass: also split on whitespace-only tokens just in case
        final = []
        for it in out:
            final.extend([p.strip() for p in it.split() if p.strip()])
        return final

    # If a single string was passed
    s = str(val)
    # Split on commas first, then whitespace
    parts = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend([p.strip() for p in chunk.split() if p.strip()])
    return parts


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path",
        required=True,
        help="Path to model directory (contains config.json)",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        required=True,
        help="Maximum tokens for each workload message",
    )
    p.add_argument(
        "--messages",
        nargs="*",
        help="List of message ids to include from workload.json (e.g. summary code-review)",
    )
    p.add_argument(
        "--lora_perc",
        type=float,
        default=0.0,
        help="Fraction of requests to use LoRA (0.0-1.0)",
    )
    p.add_argument(
        "--loras",
        type=str,
        default="",
        help="Comma-separated list of LoRA adapter names/paths to cycle through",
    )
    p.add_argument(
        "--config_dir",
        type=str,
        default=str(Path(__file__).parents[1] / "config"),
        help="Output config directory",
    )

    args = p.parse_args()

    messages = _parse_messages_arg(args.messages)
    loras = _parse_list_arg(args.loras)

    generate_config(
        model_path=args.model_path,
        max_tokens=args.max_tokens,
        messages=messages,
        lora_perc=args.lora_perc,
        loras=loras,
        config_dir=args.config_dir,
    )


if __name__ == "__main__":
    main()
