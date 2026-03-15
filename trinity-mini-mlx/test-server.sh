#!/bin/bash
# Quick test to verify the MLX server is running and responding
set -euo pipefail

PORT="${1:-8080}"
URL="http://localhost:${PORT}/v1"

echo "Testing MLX server at $URL..."
echo ""

# Test 1: List models
echo "--- Available models ---"
curl -s "$URL/models" | python3 -m json.tool 2>/dev/null || echo "ERROR: Server not responding at $URL"
echo ""

# Test 2: Chat completion
echo "--- Chat completion test ---"
RESPONSE=$(curl -s "$URL/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "user", "content": "Write a Python function that returns the factorial of n. Just the code, no explanation."}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }')

echo "$RESPONSE" | python3 -c "
import sys, json
r = json.load(sys.stdin)
if 'choices' in r:
    print(r['choices'][0]['message']['content'])
    print()
    usage = r.get('usage', {})
    print(f\"Tokens — prompt: {usage.get('prompt_tokens', '?')}, completion: {usage.get('completion_tokens', '?')}\")
else:
    print('Unexpected response:', json.dumps(r, indent=2))
"

echo ""
echo "Server is working!"
