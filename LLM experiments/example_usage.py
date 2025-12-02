#!/usr/bin/env python3
"""
Example usage of SEP Alpaca datasets with various LLM frameworks

This script demonstrates how to load and use the generated datasets
for fine-tuning different LLM models.
"""

import json
from pathlib import Path
from typing import List, Dict

def load_alpaca_dataset(file_path: str) -> List[Dict]:
    """Load the Alpaca format dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_by_topic(data: List[Dict], topic: str) -> List[Dict]:
    """Filter dataset by specific topic"""
    return [item for item in data if topic in item.get('topics', [])]


def create_training_split(data: List[Dict], train_ratio: float = 0.8) -> tuple:
    """Split data into training and validation sets"""
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def format_for_llama(item: Dict) -> str:
    """Format instruction-response pair for LLaMA training"""
    return f"""<s>[INST] {item['instruction']} [/INST]
{item['output']} </s>"""


def format_for_alpaca(item: Dict) -> str:
    """Format instruction-response pair for Alpaca training"""
    if item.get('input'):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Response:
{item['output']}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{item['instruction']}

### Response:
{item['output']}"""


def format_for_chatgpt(item: Dict) -> Dict:
    """Format instruction-response pair for ChatGPT/OpenAI format"""
    return {
        "messages": [
            {"role": "system", "content": "You are a knowledgeable philosophy assistant."},
            {"role": "user", "content": item['instruction']},
            {"role": "assistant", "content": item['output']}
        ]
    }


def example_basic_usage():
    """Example 1: Basic dataset loading and exploration"""
    print("="*60)
    print("Example 1: Basic Dataset Loading")
    print("="*60)

    # Load dataset
    data = load_alpaca_dataset('sep_data/alpaca_datasets/alpaca_philosophy.json')

    print(f"\nTotal instruction pairs: {len(data)}")

    # Show first example
    if data:
        print("\nFirst example:")
        print(f"Instruction: {data[0]['instruction']}")
        print(f"Output: {data[0]['output'][:200]}...")
        print(f"Topics: {data[0].get('topics', [])}")


def example_filter_by_topic():
    """Example 2: Filter dataset by philosophical topic"""
    print("\n" + "="*60)
    print("Example 2: Filter by Topic")
    print("="*60)

    data = load_alpaca_dataset('sep_data/alpaca_datasets/alpaca_philosophy.json')

    # Filter ethics-related content
    ethics_data = filter_by_topic(data, 'ethics')
    print(f"\nEthics-related pairs: {len(ethics_data)}")

    # Filter epistemology content
    epistemology_data = filter_by_topic(data, 'epistemology')
    print(f"Epistemology-related pairs: {len(epistemology_data)}")

    # Save filtered dataset
    output_path = 'sep_data/alpaca_datasets/alpaca_ethics.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ethics_data, f, indent=2, ensure_ascii=False)
    print(f"\nEthics dataset saved to: {output_path}")


def example_create_splits():
    """Example 3: Create training/validation splits"""
    print("\n" + "="*60)
    print("Example 3: Create Training/Validation Splits")
    print("="*60)

    data = load_alpaca_dataset('sep_data/alpaca_datasets/alpaca_philosophy.json')

    train_data, val_data = create_training_split(data, train_ratio=0.8)

    print(f"\nTraining examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    # Save splits
    with open('sep_data/alpaca_datasets/train.json', 'w') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open('sep_data/alpaca_datasets/val.json', 'w') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    print("\nSplits saved to train.json and val.json")


def example_format_conversion():
    """Example 4: Convert to different formats"""
    print("\n" + "="*60)
    print("Example 4: Format Conversion")
    print("="*60)

    data = load_alpaca_dataset('sep_data/alpaca_datasets/alpaca_philosophy.json')

    if data:
        item = data[0]

        print("\n--- LLaMA Format ---")
        print(format_for_llama(item))

        print("\n--- Alpaca Format ---")
        print(format_for_alpaca(item))

        print("\n--- ChatGPT Format ---")
        print(json.dumps(format_for_chatgpt(item), indent=2))


def example_statistics():
    """Example 5: Dataset statistics"""
    print("\n" + "="*60)
    print("Example 5: Dataset Statistics")
    print("="*60)

    data = load_alpaca_dataset('sep_data/alpaca_datasets/alpaca_philosophy.json')

    # Topic distribution
    topic_counts = {}
    for item in data:
        for topic in item.get('topics', []):
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    print("\nTopic Distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic}: {count}")

    # Average output length
    avg_length = sum(len(item['output'].split()) for item in data) / len(data)
    print(f"\nAverage output length: {avg_length:.1f} words")

    # Instruction diversity
    unique_instructions = len(set(item['instruction'] for item in data))
    print(f"Unique instructions: {unique_instructions}")


def example_huggingface_usage():
    """Example 6: Using with Hugging Face datasets"""
    print("\n" + "="*60)
    print("Example 6: Hugging Face Integration")
    print("="*60)

    print("\nTo use with Hugging Face Transformers:")
    print("""
from datasets import load_dataset

# Load dataset
dataset = load_dataset(
    'json',
    data_files='sep_data/alpaca_datasets/alpaca_philosophy.json',
    split='train'
)

# Tokenize for training
def tokenize_function(examples):
    prompts = [f"Instruction: {inst}\\nResponse:"
               for inst in examples['instruction']]
    return tokenizer(prompts, examples['output'],
                    truncation=True, padding='max_length')

tokenized_dataset = dataset.map(tokenize_function, batched=True)
    """)


def example_custom_prompts():
    """Example 7: Create custom prompt templates"""
    print("\n" + "="*60)
    print("Example 7: Custom Prompt Templates")
    print("="*60)

    data = load_alpaca_dataset('sep_data/alpaca_datasets/alpaca_philosophy.json')

    # Create Socratic dialogue format
    socratic_prompts = []
    for item in data[:5]:  # First 5 examples
        prompt = {
            'conversation': [
                {'speaker': 'Student', 'text': item['instruction']},
                {'speaker': 'Socrates', 'text': item['output'][:300] + '...'}
            ],
            'topics': item.get('topics', [])
        }
        socratic_prompts.append(prompt)

    print("\nSocratic dialogue format example:")
    print(json.dumps(socratic_prompts[0], indent=2))

    # Save custom format
    with open('sep_data/alpaca_datasets/socratic_dialogue.json', 'w') as f:
        json.dump(socratic_prompts, f, indent=2, ensure_ascii=False)

    print("\nSaved Socratic dialogue format")


def main():
    """Run all examples"""
    print("SEP Alpaca Dataset Usage Examples")
    print("Make sure you've run sep_alpaca_prep.py first!")
    print()

    # Check if dataset exists
    dataset_path = Path('sep_data/alpaca_datasets/alpaca_philosophy.json')
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please run sep_alpaca_prep.py first to generate the dataset.")
        return

    # Run examples
    example_basic_usage()
    example_filter_by_topic()
    example_create_splits()
    example_format_conversion()
    example_statistics()
    example_huggingface_usage()
    example_custom_prompts()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
