# Stanford Encyclopedia of Philosophy → Alpaca LLM Preparation

A comprehensive script to download, clean, and prepare Stanford Encyclopedia of Philosophy (SEP) content for fine-tuning Alpaca-style LLMs on Linux.

## Features

- **Automated Downloading**: Fetches all SEP articles with respectful rate limiting
- **Content Cleaning**: Converts HTML to clean Markdown and plain text
- **Smart Organization**: Automatically categorizes articles by philosophical topics
- **Alpaca Dataset Generation**: Creates instruction-response pairs for LLM fine-tuning
- **Multiple Export Formats**: JSON, JSONL, CSV, and Hugging Face formats
- **Resumable**: Caches progress to resume interrupted downloads
- **Topic-Based Structure**: Organizes content by metaphysics, epistemology, ethics, etc.

## Directory Structure

After running the script, you'll have:

```
sep_data/
├── raw_html/              # Original HTML files
├── markdown/              # Cleaned markdown versions
├── plain_text/            # Plain text versions
├── alpaca_datasets/       # Training datasets
│   ├── alpaca_philosophy.json      # Standard Alpaca format
│   ├── alpaca_philosophy.jsonl     # One JSON per line
│   ├── alpaca_philosophy_hf.json   # Hugging Face format
│   └── alpaca_philosophy.csv       # CSV format
├── metadata/              # Article metadata and statistics
├── by_topic/              # Articles organized by topic
│   ├── metaphysics/
│   ├── epistemology/
│   ├── ethics/
│   ├── logic/
│   └── ...
└── logs/                  # Processing logs
```

## Installation

### 1. Install Python dependencies

```bash
# Minimal installation (just for SEP downloading)
pip install requests beautifulsoup4 html2text tqdm

# Or install from requirements file
pip install -r requirements.txt
```

### 2. Make the script executable

```bash
chmod +x sep_alpaca_prep.py
```

## Usage

### Basic Usage

Download first 50 articles (good for testing):

```bash
python sep_alpaca_prep.py --max-articles 50
```

### Download All Articles

This will download all ~1,700+ SEP articles (takes several hours):

```bash
python sep_alpaca_prep.py --output-dir ./sep_complete
```

### Faster Downloads

Reduce delay between requests (use carefully to avoid overwhelming the server):

```bash
python sep_alpaca_prep.py --delay 0.5 --max-articles 100
```

### Custom Output Directory

```bash
python sep_alpaca_prep.py --output-dir ~/datasets/philosophy --max-articles 200
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `./sep_data` | Output directory for all files |
| `--max-articles` | `None` (all) | Maximum number of articles to download |
| `--delay` | `1.0` | Delay between requests in seconds |

## Output Formats

### 1. Alpaca Format (Standard)

```json
[
  {
    "instruction": "What is the philosophical topic of Epistemology?",
    "input": "",
    "output": "Epistemology is the study of knowledge...",
    "source": "https://plato.stanford.edu/entries/epistemology/",
    "topics": ["epistemology", "philosophy_of_mind"]
  }
]
```

### 2. JSONL Format (Line-delimited)

One JSON object per line, ideal for streaming:

```jsonl
{"instruction": "...", "input": "", "output": "..."}
{"instruction": "...", "input": "", "output": "..."}
```

### 3. Hugging Face Format

```json
[
  {
    "text": "### Instruction:\nWhat is epistemology?\n\n### Response:\nEpistemology is..."
  }
]
```

### 4. CSV Format

Tab or comma-separated values, importable into spreadsheets.

## Topic Categories

Articles are automatically classified into these categories:

- **Metaphysics**: Reality, existence, causation, time, space
- **Epistemology**: Knowledge, belief, truth, skepticism
- **Ethics**: Morality, virtue, consequentialism, deontology
- **Political Philosophy**: Justice, rights, democracy, law
- **Logic**: Formal reasoning, arguments, proofs
- **Philosophy of Mind**: Consciousness, intentionality, perception
- **Philosophy of Science**: Scientific method, explanation, causation
- **Philosophy of Language**: Meaning, reference, semantics
- **Aesthetics**: Beauty, art, taste
- **Ancient Philosophy**: Plato, Aristotle, Stoics
- **Medieval Philosophy**: Aquinas, Augustine, scholasticism
- **Modern Philosophy**: Descartes, Kant, Hume
- **Contemporary Philosophy**: Analytic, phenomenology, pragmatism

## Using the Dataset with Alpaca

### 1. With Stanford Alpaca

```bash
# Clone Alpaca repository
git clone https://github.com/tatsu-lab/stanford_alpaca.git
cd stanford_alpaca

# Copy your dataset
cp /path/to/sep_data/alpaca_datasets/alpaca_philosophy.json ./

# Fine-tune
python train.py \
  --data_path alpaca_philosophy.json \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --output_dir ./philosophy_model
```

### 2. With Hugging Face Transformers

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load dataset
dataset = load_dataset('json', data_files='sep_data/alpaca_datasets/alpaca_philosophy_hf.json')

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Fine-tune
training_args = TrainingArguments(
    output_dir="./philosophy_llm",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()
```

### 3. With LLaMA.cpp (for local inference)

```bash
# Convert to GGUF format for efficient inference
python convert.py --model ./philosophy_model --output philosophy.gguf

# Run locally
./main -m philosophy.gguf -p "Explain Kant's categorical imperative"
```

## Performance & Resource Usage

### Download Statistics
- **Total Articles**: ~1,700
- **Total Size**: ~2-3 GB (raw HTML)
- **Average Article**: 50-100 KB
- **Download Time**: 2-4 hours (with 1s delay)

### Generated Dataset
- **Instruction Pairs**: ~6,000-8,000 (from 1,700 articles)
- **Average Pairs per Article**: 3-5
- **Training Examples**: Varies by topic

### System Requirements
- **RAM**: 2 GB minimum (8 GB recommended for full corpus)
- **Disk Space**: 5 GB free space
- **Network**: Stable internet connection
- **Python**: 3.7+

## Advanced Usage

### Filter by Topic

You can manually filter the dataset by topic:

```python
import json

with open('sep_data/alpaca_datasets/alpaca_philosophy.json', 'r') as f:
    data = json.load(f)

# Filter only ethics articles
ethics_data = [item for item in data if 'ethics' in item.get('topics', [])]

with open('alpaca_ethics_only.json', 'w') as f:
    json.dump(ethics_data, f, indent=2)
```

### Combine with Other Datasets

```python
import json

# Load SEP dataset
with open('sep_data/alpaca_datasets/alpaca_philosophy.json', 'r') as f:
    sep_data = json.load(f)

# Load another dataset
with open('other_dataset.json', 'r') as f:
    other_data = json.load(f)

# Combine
combined = sep_data + other_data

# Save
with open('combined_dataset.json', 'w') as f:
    json.dump(combined, f, indent=2)
```

### Custom Instruction Templates

Modify the `generate_alpaca_dataset()` method in `sep_alpaca_prep.py` to create custom instruction formats:

```python
# Example: Question-answering format
alpaca_data.append({
    'instruction': f"Answer this question about {topic}:",
    'input': f"What is {concept}?",
    'output': answer_text
})
```

## Troubleshooting

### Connection Errors

If you get connection timeouts:
```bash
python sep_alpaca_prep.py --delay 2.0  # Increase delay
```

### Memory Issues

For systems with limited RAM:
```bash
python sep_alpaca_prep.py --max-articles 100  # Process in batches
```

### Resume Interrupted Download

The script automatically caches the article list. If interrupted, just run again:
```bash
python sep_alpaca_prep.py  # Will resume from cache
```

To start fresh:
```bash
rm -rf sep_data/metadata/sep_entries_list.json
python sep_alpaca_prep.py
```

## Examples

### Example 1: Quick Test Run

```bash
# Download 10 articles to test
python sep_alpaca_prep.py --max-articles 10 --output-dir test_run

# Check results
ls test_run/markdown/
cat test_run/alpaca_datasets/alpaca_philosophy.json | head -50
```

### Example 2: Topic-Specific Corpus

```bash
# Download full corpus
python sep_alpaca_prep.py

# Then extract specific topics
cd sep_data/by_topic/ethics
ls -lh  # See all ethics articles
```

### Example 3: Create Training Subset

```bash
# Download moderate sized corpus
python sep_alpaca_prep.py --max-articles 500

# Check statistics
cat sep_data/metadata/corpus_statistics.json
```

## Ethical Considerations

- **Respectful Crawling**: Script includes delays to avoid overwhelming SEP servers
- **Attribution**: All content is properly attributed to SEP and original authors
- **Academic Use**: Intended for educational and research purposes
- **License**: SEP content is copyrighted; check their terms of use

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{sep_alpaca_2024,
  title={Stanford Encyclopedia of Philosophy Alpaca Dataset},
  author={Your Name},
  year={2024},
  note={Prepared from Stanford Encyclopedia of Philosophy}
}

@misc{zalta_sep,
  title={The Stanford Encyclopedia of Philosophy},
  author={Zalta, Edward N.},
  editor={Zalta, Edward N.},
  howpublished={Metaphysics Research Lab, Stanford University},
  year={1997--},
  url={https://plato.stanford.edu/}
}
```

## Contributing

Improvements welcome! Areas for contribution:
- Better categorization algorithms
- Additional export formats
- Integration with more LLM frameworks
- Improved instruction generation

## License

This script is provided as-is for educational purposes. The SEP content itself is copyrighted by Stanford University.

## Resources

- **Stanford Encyclopedia of Philosophy**: https://plato.stanford.edu/
- **Alpaca Project**: https://github.com/tatsu-lab/stanford_alpaca
- **LLaMA**: https://github.com/facebookresearch/llama
- **Hugging Face**: https://huggingface.co/

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the logs in `sep_data/logs/`
3. Open an issue in the repository

---

**Happy Fine-tuning!** 🦙📚🧠
