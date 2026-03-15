#!/bin/bash
# Benchmark Trinity-Mini quantizations on mlx_lm.server
#
# Measures prompt processing (prefill) and token generation speed.
# Starts one server and switches models via the API.
#
# Usage: ./benchmark.sh [quant...]
#   ./benchmark.sh              # benchmarks 4bit, 5bit, 6bit, 8bit
#   ./benchmark.sh 8bit         # benchmark just 8-bit

set -euo pipefail

PORT=8081  # Use a different port to avoid conflicts
if [ $# -eq 0 ]; then
    QUANTS=(4bit 5bit 6bit 8bit)
else
    QUANTS=("$@")
fi

MAX_TOKENS=200
ITERATIONS=10
PROMPT="Write a Python function that implements a binary search tree with insert, search, and delete operations. Include docstrings and type hints."

echo "============================================"
echo "  Trinity-Mini MLX Benchmark"
echo "============================================"
echo "  Max tokens:  $MAX_TOKENS"
echo "  Iterations:  $ITERATIONS"
echo "  Port:        $PORT"
echo "  Quants:      ${QUANTS[*]}"
echo "============================================"
echo ""

# Start server once with the first model
FIRST_MODEL="mlx-community/Trinity-Mini-${QUANTS[0]}"
uvx --from "mlx-lm>=0.28.4" mlx_lm.server \
    --model "$FIRST_MODEL" \
    --port "$PORT" 2>/dev/null &
SERVER_PID=$!

trap 'kill $SERVER_PID 2>/dev/null; pkill -P $SERVER_PID 2>/dev/null; lsof -ti ":${PORT}" | xargs kill -9 2>/dev/null; wait 2>/dev/null' EXIT

echo -n "Starting server..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo " ready."
        break
    fi
    if [ $i -eq 120 ]; then
        echo " TIMEOUT. Exiting."
        exit 1
    fi
    sleep 1
done
echo ""

for quant in "${QUANTS[@]}"; do
    MODEL="mlx-community/Trinity-Mini-${quant}"
    echo "--- Benchmarking $MODEL ---"

    # Warm-up run (loads model + compiles Metal kernels)
    echo "  Warm-up run (loading model)..."
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 10}" \
        >/dev/null 2>&1

    # Benchmark runs
    echo "  Running $ITERATIONS iterations ($MAX_TOKENS tokens each)..."
    SPEEDS=()
    for iter in $(seq 1 "$ITERATIONS"); do
        TMPFILE=$(mktemp)
        START=$(python3 -c "import time; print(time.time())")

        curl -s "http://localhost:${PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$MODEL\",
                \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
                \"max_tokens\": $MAX_TOKENS
            }" > "$TMPFILE"

        END=$(python3 -c "import time; print(time.time())")

        SPEED=$(python3 - "$TMPFILE" "$START" "$END" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    response = json.load(f)
tokens = response.get('usage', {}).get('completion_tokens', 0)
elapsed = float(sys.argv[3]) - float(sys.argv[2])
if tokens > 0:
    print(f'{tokens / elapsed:.1f}')
else:
    print('0')
PYEOF
)
        rm -f "$TMPFILE"

        if [ "$SPEED" = "0" ]; then
            echo "    Iteration $iter: ERROR (no tokens generated)"
        else
            echo "    Iteration $iter: ${SPEED} tok/s"
            SPEEDS+=("$SPEED")
        fi
    done

    # Compute average, min, max
    SPEEDS_CSV=$(IFS=,; echo "${SPEEDS[*]}")
    python3 -c "
speeds = [float(s) for s in '${SPEEDS_CSV}'.split(',') if s]
if speeds:
    avg = sum(speeds) / len(speeds)
    print(f'  Results ({len(speeds)}/$ITERATIONS successful):')
    print(f'    Average: {avg:.1f} tok/s')
    print(f'    Min:     {min(speeds):.1f} tok/s')
    print(f'    Max:     {max(speeds):.1f} tok/s')
    print(f'    Stddev:  {(sum((s - avg)**2 for s in speeds) / len(speeds))**0.5:.1f} tok/s')
else:
    print('  ERROR: All iterations failed')
" || echo "  ERROR: Failed to compute results"
    echo ""
done

echo "============================================"
echo "  Benchmark complete"
echo "============================================"
