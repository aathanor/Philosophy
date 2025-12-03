# Next Steps: Complete Your Model Conversion

## ✅ What's Been Fixed

The GGUF conversion error you encountered has been solved. The issue was that llama.cpp's `convert_hf_to_gguf.py` no longer supports direct quantization to formats like `q4_K_M`.

**Solution**: Use a two-step process (convert → quantize)

## 📋 What You Need to Do Now

Run these commands **on your local machine** where your trained model exists:

### Step 1: Merge LoRA Adapters (if not done)
```bash
cd ~/PHILOSOPHY/LLM\ experiments
python merge_lora.py
```

**Expected output**: Creates `qwen7b_philosophy_merged/` directory with the merged model

**Time**: 5-10 minutes

---

### Step 2: Convert to GGUF
```bash
cd llama.cpp
./convert_to_gguf.sh
```

This script will:
1. Convert your merged model to f16 GGUF format
2. Quantize it to q4_K_M format
3. Give you a file called `philosophy_qwen_q4_K_M.gguf`

**Expected output**:
- `philosophy_qwen_f16.gguf` (~14 GB)
- `philosophy_qwen_q4_K_M.gguf` (~4.1 GB)

**Time**: 10-15 minutes total

**Alternative manual commands** (if you prefer):
```bash
cd llama.cpp

# Convert to f16
python convert_hf_to_gguf.py ../qwen7b_philosophy_merged \
    --outfile philosophy_qwen_f16.gguf \
    --outtype f16

# Quantize to q4_K_M
./build/bin/llama-quantize \
    philosophy_qwen_f16.gguf \
    philosophy_qwen_q4_K_M.gguf \
    q4_K_M
```

---

### Step 3: Import to Ollama
```bash
cd llama.cpp

# Create Modelfile
cp ../Modelfile.template Modelfile

# Import to Ollama
ollama create philosophy-qwen -f Modelfile
```

**Expected output**:
```
transferring model data
creating model layer
writing manifest
success
```

---

### Step 4: Test Your Model
```bash
ollama run philosophy-qwen "What is epistemology?"
```

You should see a philosophical response based on your Stanford Encyclopedia of Philosophy training data!

---

## 📁 Files Created for You

1. **`convert_to_gguf.sh`** - Automated conversion script (two-step process)
2. **`GGUF_CONVERSION_GUIDE.md`** - Detailed guide explaining the solution
3. **`Modelfile.template`** - Template for importing to Ollama
4. **`NEXT_STEPS.md`** - This file

## 🔍 Verification Checklist

Before running the commands, verify:
- [ ] `qwen7b_philosophy/` directory exists (trained LoRA adapters)
- [ ] `llama.cpp/` directory exists and is built
- [ ] `llama.cpp/build/bin/llama-quantize` exists
- [ ] `llama.cpp/convert_hf_to_gguf.py` exists
- [ ] Ollama is installed and running

## 📊 Expected File Sizes

After completion, you should have:
```
qwen7b_philosophy/          ~50 MB (LoRA adapters)
qwen7b_philosophy_merged/   ~14 GB (merged model)
philosophy_qwen_f16.gguf    ~14 GB (f16 GGUF)
philosophy_qwen_q4_K_M.gguf ~4.1 GB (quantized - this is what you'll use!)
```

## ⚠️ Troubleshooting

### If merge_lora.py fails with "No module named 'torch'":
```bash
pip install torch transformers peft accelerate
```

### If you want a different quantization level:
Edit `convert_to_gguf.sh` and change `q4_K_M` to:
- `q5_K_M` - Better quality, ~5 GB
- `q4_K_S` - Smaller size, ~3.8 GB
- `q6_K` - Best quality, ~5.8 GB

### If conversion is slow:
This is normal. Converting a 7B model takes time. Get a coffee! ☕

## 🎯 Summary

The conversion process is now:
1. **Old way** (doesn't work): `convert_hf_to_gguf.py ... --outtype q4_K_M` ❌
2. **New way** (works):
   - Convert to f16: `convert_hf_to_gguf.py ... --outtype f16` ✅
   - Quantize: `llama-quantize ... q4_K_M` ✅

Your corrected script (`convert_to_gguf.sh`) implements this two-step process automatically.

---

## 🚀 Quick Start

If you're confident everything is ready:
```bash
cd ~/PHILOSOPHY/LLM\ experiments/llama.cpp
./convert_to_gguf.sh && ollama create philosophy-qwen -f Modelfile && ollama run philosophy-qwen
```

This will convert, import, and start your model in one command!

---

**Questions?** Check `GGUF_CONVERSION_GUIDE.md` for detailed explanations.
