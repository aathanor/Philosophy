#!/bin/bash
# Correct GGUF conversion script for llama.cpp (two-step process)
# Solves the "invalid choice: 'q4_K_M'" error

set -e

echo "=========================================="
echo "GGUF Conversion Script (Two-Step Process)"
echo "=========================================="
echo ""

# Configuration
MERGED_MODEL_DIR="../qwen7b_philosophy_merged"
OUTPUT_F16="philosophy_qwen_f16.gguf"
OUTPUT_Q4="philosophy_qwen_q4_K_M.gguf"

# Check if merged model exists
if [ ! -d "$MERGED_MODEL_DIR" ]; then
    echo "ERROR: Merged model directory not found: $MERGED_MODEL_DIR"
    echo "Please run merge_lora.py first to create the merged model"
    exit 1
fi

# Step 1: Convert to f16 GGUF
echo "Step 1: Converting HF model to f16 GGUF format..."
echo "This may take 5-10 minutes depending on your system..."
python convert_hf_to_gguf.py "$MERGED_MODEL_DIR" \
    --outfile "$OUTPUT_F16" \
    --outtype f16

echo ""
echo "✓ Step 1 complete: f16 GGUF created"
echo "  File: $OUTPUT_F16"
echo "  Size: $(du -h $OUTPUT_F16 | cut -f1)"
echo ""

# Step 2: Quantize to q4_K_M
echo "Step 2: Quantizing f16 to q4_K_M format..."
echo "This will reduce the model size significantly..."
../build/bin/llama-quantize "$OUTPUT_F16" "$OUTPUT_Q4" q4_K_M

echo ""
echo "=========================================="
echo "✓ Conversion Complete!"
echo "=========================================="
echo ""
echo "Output file: $OUTPUT_Q4"
echo "  F16 size: $(du -h $OUTPUT_F16 | cut -f1)"
echo "  Q4 size:  $(du -h $OUTPUT_Q4 | cut -f1)"
echo ""
echo "You can now import this into Ollama:"
echo "  ollama create philosophy-qwen -f Modelfile"
echo ""
echo "Where Modelfile contains:"
echo "  FROM ./$OUTPUT_Q4"
echo "=========================================="
