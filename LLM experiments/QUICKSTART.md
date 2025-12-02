# Quick Start Guide - SEP Alpaca Prep

Get started in 3 minutes! ⚡

## TL;DR

```bash
# Install dependencies
pip install requests beautifulsoup4 html2text tqdm

# Download 10 test articles
python3 sep_alpaca_prep.py --max-articles 10

# View the results
cat sep_data/alpaca_datasets/alpaca_philosophy.json | head -100
```

## Using Make (Recommended)

If you have `make` installed:

```bash
# Install dependencies
make install

# Test run (10 articles)
make test

# Download sample (50 articles)
make download-sample

# View results
make check-dataset
make view-sample
```

## Step-by-Step

### 1. Install Dependencies

```bash
pip install -r requirements_minimal.txt
```

### 2. Test Run

Download just 10 articles to test:

```bash
python3 sep_alpaca_prep.py --max-articles 10
```

This creates:
- `sep_data/markdown/` - Clean markdown files
- `sep_data/alpaca_datasets/` - Training datasets
- `sep_data/by_topic/` - Organized by philosophical topic

### 3. Check Results

```bash
# List downloaded articles
ls sep_data/markdown/

# View an article
cat sep_data/markdown/epistemology.md | head -50

# Check the training dataset
cat sep_data/alpaca_datasets/alpaca_philosophy.json | head -100
```

### 4. Download More

```bash
# 50 articles (~5 minutes)
python3 sep_alpaca_prep.py --max-articles 50

# 200 articles (~20 minutes)
python3 sep_alpaca_prep.py --max-articles 200

# All articles (~3-4 hours)
python3 sep_alpaca_prep.py
```

## Output Structure

```
sep_data/
├── alpaca_datasets/
│   ├── alpaca_philosophy.json     ← Use this for training
│   ├── alpaca_philosophy.jsonl
│   ├── alpaca_philosophy_hf.json
│   └── alpaca_philosophy.csv
├── markdown/           ← Human-readable articles
├── plain_text/         ← Plain text versions
├── by_topic/          ← Organized by topic
│   ├── ethics/
│   ├── epistemology/
│   ├── metaphysics/
│   └── ...
└── metadata/          ← Statistics and metadata
```

## Training with Alpaca

Once you have the dataset:

```bash
# Clone Stanford Alpaca
git clone https://github.com/tatsu-lab/stanford_alpaca.git
cd stanford_alpaca

# Copy your dataset
cp ../sep_data/alpaca_datasets/alpaca_philosophy.json ./

# Train (requires GPU)
python train.py \
  --data_path alpaca_philosophy.json \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --output_dir ./philosophy_model
```

## Using with Different LLM Frameworks

### LLaMA.cpp (Local CPU inference)

```bash
# Convert your fine-tuned model
python convert.py --model ./philosophy_model

# Run inference
./main -m model.gguf -p "Explain Kant's categorical imperative"
```

### Hugging Face Transformers

```python
from datasets import load_dataset

dataset = load_dataset(
    'json',
    data_files='sep_data/alpaca_datasets/alpaca_philosophy.json'
)

# Use with your training script
```

### OpenAI Fine-tuning Format

```python
import json

# Load dataset
with open('sep_data/alpaca_datasets/alpaca_philosophy.json', 'r') as f:
    data = json.load(f)

# Convert to OpenAI format
openai_data = []
for item in data:
    openai_data.append({
        "messages": [
            {"role": "system", "content": "You are a philosophy expert."},
            {"role": "user", "content": item['instruction']},
            {"role": "assistant", "content": item['output']}
        ]
    })

# Save for OpenAI fine-tuning
with open('openai_format.jsonl', 'w') as f:
    for item in openai_data:
        f.write(json.dumps(item) + '\n')
```

## Common Commands

```bash
# Test installation
python3 sep_alpaca_prep.py --help

# Quick test (10 articles)
make test

# View statistics
make stats

# See available topics
make list-topics

# Clean everything
make clean
```

## Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'bs4'`
```bash
pip install beautifulsoup4
```

**Problem**: Connection timeout
```bash
# Increase delay between requests
python3 sep_alpaca_prep.py --delay 2.0 --max-articles 50
```

**Problem**: Out of disk space
```bash
# Download fewer articles first
python3 sep_alpaca_prep.py --max-articles 50
```

## Next Steps

1. **Explore the dataset**: Check `sep_data/by_topic/` folders
2. **Run examples**: `python3 example_usage.py`
3. **Filter by topic**: See `example_usage.py` for code
4. **Train your model**: Use the Alpaca training pipeline
5. **Fine-tune further**: Combine with other philosophy datasets

## Resources

- Full Documentation: `README_SEP_ALPACA.md`
- Example Scripts: `example_usage.py`
- Stanford Encyclopedia: https://plato.stanford.edu/
- Alpaca Project: https://github.com/tatsu-lab/stanford_alpaca

## Get Help

```bash
# View all options
python3 sep_alpaca_prep.py --help

# View Make commands
make help

# Check dataset
make check-dataset
```

---

**Ready to create your philosophy LLM!** 🧠🦙
