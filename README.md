# vLLM GPU Profiling Workflow

A pluggable, extensible framework for profiling GPU utilization of vLLM serving under different models and heterogeneous workloads.

## Quick Start

### 1. Run a Single Profiling Test

```bash
cd /home/george/gpu_util
source /path/to/venv/bin/activate

# Run profile with a specific config
bash scripts/run_profile.sh config/workload-model/model-conf.json

# Results will be saved to: .results/{model}_{workload}_{timestamp}/
```

### 2. Run Multiple Tests (Comparative Analysis)

```bash
# Run all configs and generate comparison plots
python scripts/workflow.py config/workload-model/

# Run specific configs only
python scripts/workflow.py config/workload-model/ --filter 1p 100p

# Custom results directory
python scripts/workflow.py config/workload-model/ --results benchmarks/
```

### 3. Analyze Individual Results

```bash
# Analyze a single profiling run
python scripts/analyze_results.py .results/res-folder/

# Compare multiple runs
python scripts/analyze_results.py .results/model1_*/  .results/model2_*/
```

## Configuration Files

Workload-model configurations are stored in `config/workload-model/` as JSON files.

### Configuration Schema

```json
{
    "metadata": {
        "name": "GENERIC MODEL PROFILE",
        "description": "Replace with specific model/run name after generation.",
        "created": "2025-11-14",
        "version": "1.0"
    },
    "model": {
        "path": "./models/ModelName",
        "name": "ModelName",
        "dtype": "bfloat16",
        "max_seq_length": 2048,
        "tensor_parallel_size": 1
    },
    "server": {
        "port": 8500,
        "gpu_memory_utilization": 0.9,
        "max_num_batched_tokens": 16384,
        "disable_log_requests": true,
        "timeout": 300
    },
    "load_profile": {
        "num_requests": 1000,
        "inter_arrival_lambda": 10.0,
        "seed": 42,
        "max_concurrent": 128,
        "disable_prompt_cache": true,
        "disable_optimizations": true
    },
  "workload": [
    {
      "id": "task_id",
      "description": "Task description",
      "messages": [
                {
                    "role": "system",
                    "content": "You are a text summarization assistant."
                },
                {
                    "role": "user",
                    "content": "Write a detailed summary of the current state of quantum computing technology..."
                }
            ],
      "max_tokens": 256,
      "weight": 0.3
    }
  ]
}
```

## Creating Custom Configurations

1. Examine the existing configurations in `scripts/gen_config/` to understand the structure.

2. Make any necessary modifications to `model_params.json` or `workload.json` to match your requirements.

3. Use script `scripts/gen_config/config_gen.py` to create a new configuration file.

```bash
python scripts/gen_config/config_gen.py --model_path <path_to_model> 
--max_tokens <max_tokens> --messages <messages to includ from workload.json> 
--lora_perc <optional lora percentage> --loras <optional path to lora adapters>
```

## Output Structure

Each profiling run creates a results directory with:

```bash
.results/{model}_{workload}_{timestamp}/
├── config.json                 # Copy of the config used
├── server.log                  # vLLM server logs
├── load_output.log            # Load generator output
├── gpu_metrics/                # Per-GPU CSV files (0.5s sampling)
│   ├── gpu_5_metrics.csv       # GPU-specific metrics for GPU index 5
│   └── gpu_4_metrics.csv       # (if multiple GPUs monitored)
├── request_log.csv             # Per-request metrics (latency, status, etc.)
├── summary.json               # Aggregate statistics
└── plots/
    ├── gpu_utilization.png    # GPU util over time
    ├── memory_utilization.png # GPU memory over time
    ├── latency_distribution.png
    └── latency_over_time.png
```

### CSV Columns

**gpu_metrics.csv:**

- timestamp, gpu, name, gpu_util, mem_util, mem_used, mem_total, temp, power

GPU metrics are written as separate per-GPU CSV files under the `gpu_metrics/` directory (e.g. `gpu_5_metrics.csv`). Each per-GPU CSV uses the same column layout:

- `timestamp, gpu, name, gpu_util, mem_util, mem_used, mem_total, temp, power`

- `gpu_util` and `mem_util` are percentages reported by `nvidia-smi`.
- `mem_used` and `mem_total` are absolute memory values (MiB) reported by `nvidia-smi` and are useful when computing absolute memory consumption.

The `summary.json` aggregates GPU stats using `gpu.mem_used` (absolute MB aggregates) and `gpu.gpu_util` (percent aggregates).

**request_log.csv:**

- idx, task_id, t0, t1, latency, status, response_len, response_snippet

## Performance Metrics Collected

### Request Metrics

- **Latency**: End-to-end request processing time
- **Throughput**: Requests per second
- **Success rate**: Percentage of successful requests
- **Response size**: Length of model outputs

### GPU Metrics (sampled every 0.5s)

- **GPU Utilization**: Percentage of GPU compute in use
- **Memory Utilization**: Percentage of GPU memory in use
- **Power Draw**: GPU power consumption in watts
- **Temperature**: GPU temperature in Celsius

Additional details recorded and summarized in `summary.json`:

- `gpu.gpu_util`: aggregated percent utilization (mean/p50/p90/p95/p99/max)
- `gpu.mem_used`: aggregated absolute memory used (MB) statistics (mean/p90/p99/max)
- `gpu.power`: aggregated power draw statistics (mean/p90/p99)

The `summary.json` also includes per-request aggregates and per-token metrics under `requests`, for example:

```json
"requests": {
    "total": 1000,
    "successful": 1000,
    "failed": 0,
    "latency_mean": 67.48,
    "latency_median": 68.19,
    "latency_p99": 123.15,
    "throughput_rps": 8.11,
    "tokens": {"count": 1000, "sum": 256000, "mean": 256.0},
    "latency_per_token_mean": 0.2636,
    "latency_per_token_p99": 0.4810
}
```

Use the per-GPU CSVs in `gpu_metrics/` for time-series plotting and `summary.json` for aggregated statistics used by `scripts/analyze_results.py`.
