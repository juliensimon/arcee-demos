#!/bin/bash
# Start Trinity-Mini MLX server with OpenAI-compatible API
#
# Usage:
#   ./start-server.sh              # defaults to 8-bit
#   ./start-server.sh 4bit         # use 4-bit quantization
#   ./start-server.sh 6bit         # use 6-bit quantization
#   ./start-server.sh 8bit         # use 8-bit quantization

set -euo pipefail

QUANT="${1:-8bit}"
MODEL="mlx-community/Trinity-Mini-${QUANT}"
PORT=8080

echo "============================================"
echo "  Trinity-Mini MLX Server"
echo "============================================"
echo "  Model:  $MODEL"
echo "  Port:   $PORT"
echo "  URL:    http://localhost:${PORT}/v1"
echo "============================================"
echo ""
echo "Starting server... (first run will download the model)"
echo ""

uvx --from "mlx-lm>=0.28.4" mlx_lm.server \
    --model "$MODEL" \
    --port "$PORT"
