#!/bin/bash
# Parametrized Profile Runner
# Usage: ./run_profile.sh <config.json> [results_dir]

set -euo pipefail

# Cleanup function to ensure processes are killed
cleanup() {
    echo ""
    echo "[!] Interrupt received. Cleaning up..."
    if [ ! -z "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "    Killing vLLM server (PID: $SERVER_PID)..."
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    # Kill all GPU monitor processes
    if [ ! -z "${GPU_MON_PIDS:-}" ]; then
        for gpu_id in "${!GPU_MON_PIDS[@]}"; do
            if kill -0 "${GPU_MON_PIDS[$gpu_id]}" 2>/dev/null; then
                echo "    Killing GPU monitor for GPU $gpu_id (PID: ${GPU_MON_PIDS[$gpu_id]})..."
                kill -9 "${GPU_MON_PIDS[$gpu_id]}" 2>/dev/null || true
            fi
        done
    fi
    # Kill any remaining vllm processes
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    echo "[!] Cleanup complete"
    exit 1
}

# Set trap to call cleanup on SIGINT, SIGTERM
trap cleanup SIGINT SIGTERM

if [ $# -lt 1 ]; then
    echo "Usage: $0 <workload-model-config.json> [results_directory]"
    echo "Example: $0 config/workload-model/tinyllama-heavy.json results/"
    exit 1
fi

# SET GPU DEVICE(S) TO USE
export CUDA_VISIBLE_DEVICES=5
echo "Using GPU device(s): $CUDA_VISIBLE_DEVICES"

CONFIG_FILE="$1"
RESULTS_BASE_DIR="${2:-.results}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Parse config
DIR_PATH="/home/george/gpu_util"
MODEL_PATH=$(python3 -c "import json; cfg=json.load(open('$CONFIG_FILE')); print(cfg['model']['path'])")
MODEL_NAME=$(python3 -c "import json; cfg=json.load(open('$CONFIG_FILE')); print(cfg['model']['name'])")
SERVER_PORT=$(python3 -c "import json; cfg=json.load(open('$CONFIG_FILE')); print(cfg['server']['port'])")
GPU_MEM_UTIL=$(python3 -c "import json; cfg=json.load(open('$CONFIG_FILE')); print(cfg['server']['gpu_memory_utilization'])")
MAX_BATCH=$(python3 -c "import json; cfg=json.load(open('$CONFIG_FILE')); print(cfg['server']['max_num_batched_tokens'])")
WORKLOAD_NAME=$(python3 -c "import json; cfg=json.load(open('$CONFIG_FILE')); print(cfg['metadata']['name'].lower().replace(' ', '_'))")
DISABLE_PROMPT_CACHE=$(python3 -c "import json; cfg=json.load(open('$CONFIG_FILE')); print(cfg['load_profile']['disable_prompt_cache'])")
DISABLE_OPTIMIZATIONS=$(python3 -c "import json; cfg=json.load(open('$CONFIG_FILE')); print(cfg['load_profile']['disable_optimizations'])")
MAX_CONCURRENCY=$(python3 -c "import json; cfg=json.load(open('$CONFIG_FILE')); print(cfg['load_profile']['max_concurrent'])")

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="$RESULTS_BASE_DIR/${MODEL_NAME}_${WORKLOAD_NAME}_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

REQUEST_LOG="$RESULT_DIR/request_log.csv"
SERVER_LOG="$RESULT_DIR/server.log"
LOAD_LOG="$RESULT_DIR/load_output.log"
CONFIG_COPY="$RESULT_DIR/config.json"
GPU_LOGS_DIR="$RESULT_DIR/gpu_metrics"

mkdir -p "$GPU_LOGS_DIR"
cp "$CONFIG_FILE" "$CONFIG_COPY"

echo "=========================================="
echo "GPU Profiling: $MODEL_NAME"
echo "Workload: $WORKLOAD_NAME"
echo "Results: $RESULT_DIR"
echo "=========================================="
echo ""

echo "[1/5] Starting vLLM server..."
# Check if config specifies LoRA adapters
LORA_ENABLED=$(python3 -c "import json; cfg=json.load(open('$CONFIG_FILE')); print('true' if any('lora_adapter' in task for task in cfg.get('workload', [])) else 'false')")

if [ "$DISABLE_PROMPT_CACHE" = "True" ]; then 
    echo "    Prompt caching disabled as per configuration."
    PROMPT_CACHE_FLAG="--no-enable-prefix-caching"
else
    PROMPT_CACHE_FLAG=""
fi

if [ "$DISABLE_OPTIMIZATIONS" = "True" ]; then
    echo "    Optimizations disabled as per configuration."
    DISABLE_OPTS="--disable-custom-all-reduce"
else 
    DISABLE_OPTS=""
fi

if [ "$LORA_ENABLED" = "true" ]; then
    echo "    LoRA support detected. Enabling adapters..."
    # Get unique LoRA adapters from config
    LORA_MODULES=$(python3 -c "
import json
cfg = json.load(open('$CONFIG_FILE'))
adapters = set()
for task in cfg.get('workload', []):
    if 'lora_adapter' in task and task['lora_adapter']:
        adapters.add(task['lora_adapter'])
modules = []
for adapter in sorted(adapters):
    path = f'$(pwd)/models/lora-adapters/{adapter}'
    modules.append(f'{adapter}={path}')
print(' '.join(modules))
")

    echo "    Using LoRA adapters: $LORA_MODULES"

    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --port $SERVER_PORT \
      --gpu-memory-utilization $GPU_MEM_UTIL \
      --max-num-batched-tokens $MAX_BATCH \
      --disable-log-requests \
      $PROMPT_CACHE_FLAG \
      --enable-lora \
      --lora-modules $LORA_MODULES \
      --max-loras 3 \
      --max-lora-rank 64 \
      --max-num-seqs $MAX_CONCURRENCY \
      $DISABLE_OPTS \
      > "$SERVER_LOG" 2>&1 &
else
    echo "    Standard mode (no LoRA)."
    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --port $SERVER_PORT \
      --gpu-memory-utilization $GPU_MEM_UTIL \
      --max-num-batched-tokens $MAX_BATCH \
      --max-num-seqs $MAX_CONCURRENCY \
      --disable-log-requests \
        $PROMPT_CACHE_FLAG \
        $DISABLE_OPTS \
      > "$SERVER_LOG" 2>&1 &
fi

SERVER_PID=$!
echo "    Server PID: $SERVER_PID"
sleep 90  # Allow model to fully initialize and warm up

echo "[2/5] Starting GPU monitor..."
# Monitor each GPU in CUDA_VISIBLE_DEVICES with separate log files
for gpu_id in $(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' '); do
    GPU_LOG="$GPU_LOGS_DIR/gpu_${gpu_id}_metrics.csv"
    nvidia-smi --id=$gpu_id --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,\
memory.used,memory.total,temperature.gpu,power.draw \
--format=csv,noheader,nounits -lms 500 > "$GPU_LOG" 2>&1 &
    GPU_MON_PIDS[$gpu_id]=$!
done
echo "    GPU Monitor PIDs: ${GPU_MON_PIDS[@]}"

echo "[3/5] Running load generator..."
echo "    Max concurrency: $MAX_CONCURRENCY"
python "$DIR_PATH/scripts/load_generator_config.py" -o "$REQUEST_LOG" -m "$MAX_CONCURRENCY" "$CONFIG_FILE" \
    > "$LOAD_LOG" 2>&1 || true

echo "[4/5] Stopping background processes..."
sleep 2  # Give GPU monitor time to finish
# Kill all GPU monitor processes
for gpu_id in $(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' '); do
    if [ ! -z "${GPU_MON_PIDS[$gpu_id]:-}" ] && kill -0 "${GPU_MON_PIDS[$gpu_id]}" 2>/dev/null; then
        kill "${GPU_MON_PIDS[$gpu_id]}" 2>/dev/null || true
        wait "${GPU_MON_PIDS[$gpu_id]}" 2>/dev/null || true
    fi
done

sleep 2
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# Ensure all vllm processes are dead
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 1

echo "[5/5] Collecting summary statistics..."

# Generate summary JSON
python3 "$DIR_PATH/scripts/generate_summary.py" \
    -r "$REQUEST_LOG" \
    -g "$GPU_LOG" \
    -c "$CONFIG_COPY" \
    -m "$MODEL_NAME" \
    -w "$WORKLOAD_NAME" \
    -t "$TIMESTAMP" \
    -o "$RESULT_DIR"

echo ""
echo "=========================================="
echo "[+] Profiling complete!"
echo "Results saved to: $RESULT_DIR"
echo "=========================================="
