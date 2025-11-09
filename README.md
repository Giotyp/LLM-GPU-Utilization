# vLLM GPU Profiling Workflow

A pluggable, extensible framework for profiling GPU utilization of vLLM serving under different models and heterogeneous workloads.

## Quick Start

### 1. Run a Single Profiling Test

```bash
cd /home/george/gpu_util
source /path/to/venv/bin/activate

# Run profile with a specific config
bash scripts/run_profile.sh config/workload-model/tinyllama-light.json

# Results will be saved to: .results/{model}_{workload}_{timestamp}/
```

### 2. Run Multiple Tests (Comparative Analysis)

```bash
# Run all configs and generate comparison plots
python scripts/workflow.py config/workload-model/

# Run specific configs only
python scripts/workflow.py config/workload-model/ --filter light heavy

# Custom results directory
python scripts/workflow.py config/workload-model/ --results benchmarks/
```

### 3. Analyze Individual Results

```bash
# Analyze a single profiling run
python scripts/analyze_results.py .results/tinyllama_light_20251107_120000/

# Compare multiple runs
python scripts/analyze_results.py .results/tinyllama_light_*/  .results/tinyllama_heavy_*/
```

## Configuration Files

Workload-model configurations are stored in `config/workload-model/` as JSON files.

### Configuration Schema

```json
{
  "metadata": {
    "name": "string - descriptive name",
    "description": "string - what this workload tests",
    "created": "ISO timestamp",
    "version": "1.0"
  },
  "model": {
    "path": "string - path to model or HF model ID",
    "name": "string - short model name for logging",
    "dtype": "bfloat16|float16|float32",
    "max_seq_length": 2048,
    "tensor_parallel_size": 1
  },
  "server": {
    "port": 8500,
    "gpu_memory_utilization": 0.9,
    "max_num_batched_tokens": 8192,
    "disable_log_requests": true,
    "timeout": 120
  },
  "load_profile": {
    "num_requests": 200,
    "inter_arrival_lambda": 0.5,
    "seed": 42
  },
  "workload": [
    {
      "id": "task_id",
      "description": "Task description",
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
      ],
      "max_tokens": 64,
      "weight": 0.3
    }
  ]
}
```

### Pre-configured Workloads

#### tinyllama-light.json

- 100 requests at 1 req/s inter-arrival
- Short summarization tasks
- Low GPU memory utilization
- **Use case**: Testing baseline performance

#### tinyllama-heavy.json

- 500 requests at 5 req/s inter-arrival (lambda=0.2)
- Long essay generation tasks
- Max token outputs
- **Use case**: Stress testing peak performance

#### tinyllama-heterogeneous.json

- 300 requests with mixed task types
- Varying input/output sizes
- Realistic production patterns
- **Use case**: Real-world workload simulation

## Creating Custom Configurations

1. Copy an existing config as a template:

```bash
cp config/workload-model/tinyllama-light.json config/workload-model/custom-workload.json
```

2. Edit the JSON to customize:
   - Model path and parameters
   - Server settings (GPU memory, batch size, etc.)
   - Load profile (request count, inter-arrival rate)
   - Workload tasks and weights

3. Run profiling:

```bash
bash scripts/run_profile.sh config/workload-model/custom-workload.json
```

## Output Structure

Each profiling run creates a results directory with:

```bash
.results/{model}_{workload}_{timestamp}/
├── config.json                 # Copy of the config used
├── server.log                  # vLLM server logs
├── load_output.log            # Load generator output
├── gpu_metrics.csv            # GPU metrics (0.5s sampling)
├── request_log.csv            # Per-request metrics (latency, status, etc.)
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

**request_log.csv:**

- idx, task_id, t0, t1, latency, status, response_len, response_snippet

### Summary Statistics (summary.json)

```json
{
  "requests": {
    "total": 100,
    "successful": 100,
    "failed": 0,
    "latency_mean": 0.268,
    "latency_median": 0.250,
    "latency_p99": 0.350,
    "latency_max": 0.500,
    "throughput_rps": 5.23
  },
  "gpu": {
    "utilization_mean": 45.2,
    "utilization_max": 78.5,
    "memory_util_mean": 32.1,
    "memory_util_max": 55.3,
    "temperature_max": 65.0,
    "power_draw_mean": 85.5
  }
}
```

## Scripts

### `load_generator_config.py`

Generates and executes load tests from a configuration file.

```bash
python scripts/load_generator_config.py config/workload-model/tinyllama-light.json output.csv
```

### `run_profile.sh`

Main profiling orchestrator. Starts server, monitors GPU, runs load test, and generates summary.

```bash
bash scripts/run_profile.sh config/workload-model/tinyllama-light.json [results_dir]
```

### `analyze_results.py`

Analyzes and visualizes single or multiple profiling runs.

```bash
# Single run
python scripts/analyze_results.py .results/tinyllama_light_20251107_120000/

# Comparative analysis
python scripts/analyze_results.py .results/tinyllama_*/
```

### `workflow.py`

Orchestrates multiple profiling runs for batch testing.

```bash
# Run all configs
python scripts/workflow.py config/workload-model/

# Run with filtering
python scripts/workflow.py config/workload-model/ --filter light heavy

# Custom output
python scripts/workflow.py config/workload-model/ --results benchmarks/
```

## Example Workflows

### Profile Three Model Configurations

```bash
# Assuming you have models at: ./models/{model}/

python scripts/workflow.py config/workload-model/ --filter light heavy heterogeneous
```

### Compare Performance Across Workloads

```bash
# Run all tests
python scripts/workflow.py config/workload-model/

# Analyze with comparison plots
python scripts/analyze_results.py .results/tinyllama_*/
```

### Profile New Model

1. Download/add model to `./models/`
2. Create config: `config/workload-model/my-model-light.json`
3. Run: `bash scripts/run_profile.sh config/workload-model/my-model-light.json`

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

## Extensibility

### Adding New Workload Types

1. Create a new config file in `config/workload-model/`
2. Define your tasks with different weights
3. Run with the workflow orchestrator

### Adding New Models

1. Download/place model in `./models/{model-name}/`
2. Create configs for different workloads
3. Run profiling workflow

### Custom Analysis

Extend `analyze_results.py` with additional plotting or statistical functions. The `ProfileAnalyzer` class exposes:

- `req_df`: Request metrics DataFrame
- `gpu_df`: GPU metrics DataFrame
- `summary`: Aggregate statistics dictionary
- `config`: Configuration used for the run

## Troubleshooting

### Server won't start

- Check that port is not in use: `lsof -i :8500`
- Ensure model path is correct
- Check vLLM logs in results directory

### GPU metrics not collected

- Ensure `nvidia-smi` is installed and in PATH
- Check GPU is accessible: `nvidia-smi`

### Analysis fails

- Verify result directory has required CSV files
- Check file permissions

## Requirements

- Python 3.8+
- vLLM >= 0.11.0
- pandas, matplotlib, seaborn
- nvidia-utils (for nvidia-smi)
- CUDA-capable GPU

## Notes

- Inter-arrival times are modeled as exponential distributions
- All timestamps use Unix epoch seconds
- GPU metrics sample at 0.5s intervals (tunable in run_profile.sh)
- Results are organized by model/workload/timestamp for easy organization
