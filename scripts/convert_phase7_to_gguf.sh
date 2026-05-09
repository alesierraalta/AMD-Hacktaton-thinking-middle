#!/usr/bin/env bash
# scripts/convert_phase7_to_gguf.sh
# Merges and converts Phase 7 Qwen2.5-Coder-7B-Instruct adapter to GGUF format.
set -euo pipefail

MERGED_MODEL_DIR="${1:-results/phase7_merged_model}"
GGUF_OUT_DIR="${2:-results/phase7_gguf}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-llama.cpp}"

mkdir -p "$GGUF_OUT_DIR"
F16_GGUF="$GGUF_OUT_DIR/phase7-final-qwen25-coder-7b-f16.gguf"
Q4_GGUF="$GGUF_OUT_DIR/phase7-final-qwen25-coder-7b.gguf"
QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"

echo "=== Phase 7 GGUF Export Readiness ==="

if [[ ! -d "$MERGED_MODEL_DIR" ]]; then
  echo "Error: Missing merged model directory: $MERGED_MODEL_DIR" >&2
  echo "Please run 'python scripts/merge_phase7_adapter.py' first." >&2
  exit 1
fi

if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
  echo "Cloning llama.cpp..."
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi

echo "Installing requirements for llama.cpp..."
python -m pip install -r "$LLAMA_CPP_DIR/requirements.txt"

QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"

# Build if needed
if [[ ! -x "$QUANTIZE_BIN" ]]; then
  echo "Building llama.cpp..."
  if ! command -v cmake >/dev/null 2>&1; then
    echo "Error: cmake is required but not installed." >&2
    exit 1
  fi
  cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build"
  cmake --build "$LLAMA_CPP_DIR/build" --config Release
fi

echo "Converting HF to F16 GGUF..."
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$MERGED_MODEL_DIR" \
  --outfile "$F16_GGUF" \
  --outtype f16

if [[ ! -x "$QUANTIZE_BIN" ]]; then
  echo "Error: llama-quantize binary not found at $QUANTIZE_BIN" >&2
  exit 1
fi

echo "Quantizing to $QUANT_TYPE..."
"$QUANTIZE_BIN" "$F16_GGUF" "$Q4_GGUF" "$QUANT_TYPE"

echo "=== Success ==="
echo "Phase 7 GGUF saved to: $Q4_GGUF"
echo "Import into LM Studio using: lms import $Q4_GGUF"
