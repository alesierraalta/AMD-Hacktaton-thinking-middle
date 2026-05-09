#!/usr/bin/env bash
set -euo pipefail

MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-results/phase6_merged_model}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-llama.cpp}"
F16_GGUF="${F16_GGUF:-results/codepause-qwen15-merged-f16.gguf}"
Q4_GGUF="${Q4_GGUF:-results/codepause-qwen15-merged-Q4_K_M.gguf}"
QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"

if [[ ! -d "$MERGED_MODEL_DIR" ]]; then
  echo "Missing merged model directory: $MERGED_MODEL_DIR" >&2
  echo "Run: python scripts/merge_phase6_adapter.py" >&2
  exit 1
fi

if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi

python -m pip install -r "$LLAMA_CPP_DIR/requirements.txt"

QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
if [[ ! -x "$QUANTIZE_BIN" ]]; then
  if ! command -v cmake >/dev/null 2>&1; then
    echo "Missing cmake. Install cmake before building llama.cpp." >&2
    exit 1
  fi

  cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build"
  cmake --build "$LLAMA_CPP_DIR/build" --config Release
fi

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$MERGED_MODEL_DIR" \
  --outfile "$F16_GGUF" \
  --outtype f16

if [[ ! -x "$QUANTIZE_BIN" ]]; then
  echo "Missing llama-quantize binary: $QUANTIZE_BIN" >&2
  echo "Try: find $LLAMA_CPP_DIR -name 'llama-quantize*'" >&2
  exit 1
fi

"$QUANTIZE_BIN" "$F16_GGUF" "$Q4_GGUF" "$QUANT_TYPE"

echo "Quantized GGUF saved to $Q4_GGUF"
echo "Import with: lms import $Q4_GGUF"
