#!/bin/bash
# Patch mlx-lm to enable native tool calling for Trinity models.
#
# Fixes three issues:
# 1. tokenizer_utils.py: Removes vocab guard that disables tool parsing
#    when <tool_call> isn't a single token (issue #984)
# 2. server.py: Switches streaming parser from exact token matching to
#    buffer-based substring matching for multi-token delimiters (issue #984)
# 3. server.py: Disables batching to prevent KV cache crash on concurrent
#    requests with different prompt lengths (issue #983)
# 4. tokenizer_config.json: Adds tool_parser_type to model configs
#
# Safe to run multiple times. Patches are idempotent.

set -euo pipefail

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
UV_CACHE="$HOME/.cache/uv"
QUANT="${1:-8bit}"
MODEL="mlx-community/Trinity-Mini-${QUANT}"

# Ensure mlx-lm is cached (downloads the package + model if needed)
echo "=== Ensuring mlx-lm and model are cached ==="
echo "  Model: $MODEL"
echo "  (This may download ~16 GB on first run)"
echo ""
uvx --from "mlx-lm>=0.28.4" python -c "
from mlx_lm import load
print('mlx-lm package cached.')
try:
    load('$MODEL')
    print('Model cached: $MODEL')
except Exception as e:
    print(f'Model will be downloaded when the server starts: {e}')
"
echo ""

echo "=== Patching mlx-lm server files ==="

patched_server=0
for server_py in $(find "$UV_CACHE" -path "*/mlx_lm/server.py" 2>/dev/null); do
    if grep -q "exact match or substring\|text matching" "$server_py" 2>/dev/null; then
        echo "Already patched: $(dirname "$server_py")"
        continue
    fi
    if ! grep -q "gen.text == ctx.tool_call_start" "$server_py" 2>/dev/null; then
        echo "Skipped (unknown version): $(dirname "$server_py")"
        continue
    fi

    python3 -c "
import re

with open('$server_py') as f:
    code = f.read()

# Patch the streaming tool call detection: token matching -> buffer matching
old = '''            # Gather the text in tool calling or text variables
            if in_reasoning:
                if gen.text == ctx.think_end:
                    in_reasoning = False
                else:
                    reasoning_text += gen.text
            elif ctx.has_tool_calling and gen.text == ctx.tool_call_start:
                made_tool_call = True
                in_tool_call = True
            elif in_tool_call:
                if gen.text == ctx.tool_call_end:
                    tool_calls.append(tool_text)
                    tool_text = \"\"
                    in_tool_call = False
                else:
                    tool_text += gen.text
            else:
                text += gen.text
                segment += gen.text'''

new = '''            # Gather the text in tool calling or text variables
            if in_reasoning:
                if gen.text == ctx.think_end:
                    in_reasoning = False
                else:
                    reasoning_text += gen.text
            elif in_tool_call:
                tool_text += gen.text
                # Check for end tag via exact match or substring (text matching)
                if gen.text == ctx.tool_call_end or ctx.tool_call_end in tool_text:
                    tool_text = tool_text[:tool_text.index(ctx.tool_call_end)]
                    tool_calls.append(tool_text)
                    tool_text = \"\"
                    in_tool_call = False
            elif ctx.has_tool_calling:
                text += gen.text
                segment += gen.text
                # Check for start tag via exact match or substring (text matching)
                if ctx.tool_call_start in text:
                    text = text[:text.index(ctx.tool_call_start)]
                    segment = \"\"
                    made_tool_call = True
                    in_tool_call = True
            else:
                text += gen.text
                segment += gen.text'''

if old not in code:
    print('WARNING: Could not find expected code block in server.py')
    print('  File may have a different version. Skipping.')
    exit(1)

code = code.replace(old, new)
with open('$server_py', 'w') as f:
    f.write(code)
print('Patched: $server_py')
" && patched_server=$((patched_server + 1)) || echo "Skipped: $server_py"
done

echo ""
echo "=== Disabling batching to prevent KV cache crash (issue #983) ==="

patched_batch=0
for server_py in $(find "$UV_CACHE" -path "*/mlx_lm/server.py" 2>/dev/null); do
    if grep -q "Disabled: BatchRotatingKVCache" "$server_py" 2>/dev/null; then
        echo "Already patched: $(dirname "$server_py")"
        continue
    fi
    if ! grep -q "self.is_batchable = all(" "$server_py" 2>/dev/null; then
        echo "Skipped (unknown version or already modified): $(dirname "$server_py")"
        continue
    fi

    python3 -c "
with open('$server_py') as f:
    code = f.read()

old = '''            self.is_batchable = all(
                hasattr(c, \"merge\") for c in make_prompt_cache(self.model)
            )'''

new = '''            self.is_batchable = False  # Disabled: BatchRotatingKVCache.merge() crashes on different prompt lengths (issue #983)'''

if old not in code:
    exit(0)  # Already patched or different version

code = code.replace(old, new)
with open('$server_py', 'w') as f:
    f.write(code)
print('Patched: $server_py')
" && patched_batch=$((patched_batch + 1)) || true
done

echo ""
echo "=== Patching mlx-lm tokenizer_utils files ==="

patched_tokenizer=0
for tu_py in $(find "$UV_CACHE" -path "*/mlx_lm/tokenizer_utils.py" 2>/dev/null); do
    if grep -q "text matching still works" "$tu_py" 2>/dev/null; then
        echo "Already patched: $tu_py"
        continue
    fi

    python3 -c "
with open('$tu_py') as f:
    code = f.read()

old = '''        # Disable tool calling if tool call tokens aren't in vocab
        if (tool_call_start and tool_call_start not in vocab) or (
            tool_call_end and tool_call_end not in vocab
        ):
            self._tool_call_start = None
            self._tool_call_end = None
            self._tool_parser = None'''

new = '''        # Warn if tool call tokens aren't in vocab (text matching still works)
        if (tool_call_start and tool_call_start not in vocab) or (
            tool_call_end and tool_call_end not in vocab
        ):
            import logging
            logging.warning(
                f\"Tool call tokens ({tool_call_start}, {tool_call_end}) not \"
                \"found as single tokens in vocab. Tool parsing will use text matching.\"
            )'''

if old not in code:
    print('WARNING: Could not find expected code block in tokenizer_utils.py')
    print('  File may have a different version. Skipping.')
    exit(1)

code = code.replace(old, new)
with open('$tu_py', 'w') as f:
    f.write(code)
print('Patched: $tu_py')
" && patched_tokenizer=$((patched_tokenizer + 1)) || echo "Skipped: $tu_py"
done

echo ""
echo "=== Patching model tokenizer configs ==="

patched_model=0
for model_dir in "$HF_CACHE"/models--mlx-community--Trinity-*/snapshots/*/; do
    config="$model_dir/tokenizer_config.json"
    [ -f "$config" ] || continue

    if python3 -c "
import json, sys
with open('$config') as f:
    c = json.load(f)
if c.get('tool_parser_type') == 'json_tools':
    sys.exit(1)
c['tool_parser_type'] = 'json_tools'
with open('$config', 'w') as f:
    json.dump(c, f, indent=2, ensure_ascii=False)
" 2>/dev/null; then
        echo "Patched: $(basename "$(dirname "$(dirname "$model_dir")")")"
        patched_model=$((patched_model + 1))
    else
        echo "Already patched: $(basename "$(dirname "$(dirname "$model_dir")")")"
    fi
done

echo ""
total=$((patched_server + patched_batch + patched_tokenizer + patched_model))
if [ $total -eq 0 ]; then
    echo "Everything already patched."
else
    echo "Done. Patched $patched_server server parser(s), $patched_batch batching fix(es), $patched_tokenizer tokenizer_utils, $patched_model model config(s)."
    echo "Restart the server for changes to take effect."
fi
