#!/bin/bash
# Parametrized Profile Runner
# Usage: ./run_profile.sh <config.json> [results_dir]

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <workload-model-config.json> [results_directory]"
    echo "Example: $0 config/workload-model/tinyllama-heavy.json results/"
    exit 1
fi

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
MAX_CONCURRENCY=1000

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="$RESULTS_BASE_DIR/${MODEL_NAME}_${WORKLOAD_NAME}_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

GPU_LOG="$RESULT_DIR/gpu_metrics.csv"
REQUEST_LOG="$RESULT_DIR/request_log.csv"
SERVER_LOG="$RESULT_DIR/server.log"
LOAD_LOG="$RESULT_DIR/load_output.log"
CONFIG_COPY="$RESULT_DIR/config.json"

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

if [ "$LORA_ENABLED" = "true" ]; then
    echo "    LoRA support detected. Enabling adapters..."
    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --port $SERVER_PORT \
      --gpu-memory-utilization $GPU_MEM_UTIL \
      --max-num-batched-tokens $MAX_BATCH \
      --disable-log-requests \
      --enable-lora \
      --lora-modules alpaca=$(pwd)/models/lora-adapters/alpaca-lora-7b \
                     llama2chat=$(pwd)/models/lora-adapters/llama2-7b-chat-lora-adaptor \
      --max-lora-rank 64 \
      --max-num-seqs 512 \
      > "$SERVER_LOG" 2>&1 &
else
    echo "    Standard mode (no LoRA)."
    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --port $SERVER_PORT \
      --gpu-memory-utilization $GPU_MEM_UTIL \
      --max-num-batched-tokens $MAX_BATCH \
      --max-num-seqs 512 \
      --disable-log-requests \
      > "$SERVER_LOG" 2>&1 &
fi

SERVER_PID=$!
echo "    Server PID: $SERVER_PID"
sleep 45  # Allow model to fully initialize and warm up

echo "[2/5] Starting GPU monitor..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,\
memory.used,memory.total,temperature.gpu,power.draw \
--format=csv,noheader,nounits -lms 500 > "$GPU_LOG" 2>&1 &
GPU_MON_PID=$!
echo "    GPU Monitor PID: $GPU_MON_PID"

echo "[3/5] Running load generator..."
python "$DIR_PATH/scripts/load_generator_config.py" "$CONFIG_FILE" "$REQUEST_LOG" "$MAX_CONCURRENCY" \
  > "$LOAD_LOG" 2>&1 || true

echo "[4/5] Stopping background processes..."
sleep 2  # Give GPU monitor time to finish
kill $GPU_MON_PID 2>/dev/null || true
wait $GPU_MON_PID 2>/dev/null || true

sleep 2
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "[5/5] Collecting summary statistics..."

# Generate summary JSON
python3 "$DIR_PATH/scripts/generate_summary.py" "$REQUEST_LOG" "$GPU_LOG" "$CONFIG_FILE" "$MODEL_NAME" "$WORKLOAD_NAME" "$TIMESTAMP" "$RESULT_DIR"

echo ""
echo "=========================================="
echo "[+] Profiling complete!"
echo "Results saved to: $RESULT_DIR"
echo "=========================================="
