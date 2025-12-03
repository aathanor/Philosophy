# GGUF Conversion Guide

## Problem Solved

The error you encountered:
```bash
error: argument --outtype: invalid choice: 'q4_K_M'
```

This happens because **llama.cpp changed their API**. The `convert_hf_to_gguf.py` script no longer supports quantization types like `q4_K_M`, `q4_0`, `q5_K_M`, etc.

## Solution: Two-Step Process

### Step 1: Convert to f16 GGUF
First, convert your HuggingFace model to f16 GGUF format (unquantized):

```bash
cd llama.cpp
python convert_hf_to_gguf.py ../qwen7b_philosophy_merged \
    --outfile philosophy_qwen_f16.gguf \
    --outtype f16
```

Supported `--outtype` values:
- `f32` - Full precision (largest size)
- `f16` - Half precision (recommended)
- `bf16` - BFloat16
- `q8_0` - 8-bit quantization
- `auto` - Automatic selection

### Step 2: Quantize to q4_K_M
Then use llama.cpp's `llama-quantize` tool to quantize:

```bash
./build/bin/llama-quantize philosophy_qwen_f16.gguf philosophy_qwen_q4_K_M.gguf q4_K_M
```

Available quantization types for `llama-quantize`:
- `q4_0` - 4-bit (smallest, fastest, lower quality)
- `q4_1` - 4-bit (better than q4_0)
- `q4_K_M` - 4-bit with K-quantization (good balance)
- `q4_K_S` - 4-bit K-quant small
- `q5_0` - 5-bit
- `q5_1` - 5-bit (better)
- `q5_K_M` - 5-bit with K-quantization (recommended for quality)
- `q5_K_S` - 5-bit K-quant small
- `q6_K` - 6-bit (best quality, larger size)
- `q8_0` - 8-bit (very high quality)

## Quick Start

### Option 1: Use the provided script
```bash
cd "LLM experiments/llama.cpp"
./convert_to_gguf.sh
```

### Option 2: Manual commands
```bash
cd "LLM experiments/llama.cpp"

# Step 1: Convert to f16
python convert_hf_to_gguf.py ../qwen7b_philosophy_merged \
    --outfile philosophy_qwen_f16.gguf \
    --outtype f16

# Step 2: Quantize
./build/bin/llama-quantize \
    philosophy_qwen_f16.gguf \
    philosophy_qwen_q4_K_M.gguf \
    q4_K_M
```

## Complete Workflow

If you haven't done these steps yet:

### 1. Merge LoRA with Base Model
```bash
cd "LLM experiments"
python merge_lora.py
```
This creates `qwen7b_philosophy_merged/` directory.

### 2. Convert to GGUF
```bash
cd llama.cpp
./convert_to_gguf.sh
```

### 3. Import to Ollama
Create a `Modelfile`:
```bash
cat > Modelfile << EOF
FROM ./philosophy_qwen_q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM You are a philosophical reasoning assistant trained on the Stanford Encyclopedia of Philosophy.
EOF
```

Then import:
```bash
ollama create philosophy-qwen -f Modelfile
```

### 4. Test the Model
```bash
ollama run philosophy-qwen "What is epistemology?"
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
If you get this when running `merge_lora.py`:
```bash
pip install torch transformers peft accelerate
```

### "CUDA out of memory" during merge
The merge script already uses CPU offloading. If it still fails:
```bash
# Edit merge_lora.py and change max_shard_size
max_shard_size="1GB"  # Instead of "2GB"
```

### Conversion is very slow
This is normal. Converting a 7B model to GGUF takes 5-15 minutes depending on your CPU.

### Want a different quantization level?
Replace `q4_K_M` in step 2 with:
- `q5_K_M` - Better quality, larger file
- `q4_K_S` - Smaller file, faster
- `q6_K` - Highest quality before f16

## File Sizes (Approximate)

For Qwen 7B:
- **Base model (HF)**: ~14 GB
- **f16 GGUF**: ~14 GB
- **q8_0**: ~7.5 GB
- **q6_K**: ~5.8 GB
- **q5_K_M**: ~5.0 GB (good balance)
- **q4_K_M**: ~4.1 GB (recommended)
- **q4_0**: ~3.8 GB (smaller, lower quality)

## Summary

The key insight is that **llama.cpp now separates conversion and quantization**:
1. **Conversion** (`convert_hf_to_gguf.py`) - Only creates f16/f32/bf16/q8_0
2. **Quantization** (`llama-quantize`) - Creates all other quantization levels

This two-step process gives you more control and is more flexible.
