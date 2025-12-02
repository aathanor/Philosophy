#!/usr/bin/env python3
"""
Create a fine-tuned adapter for Ollama models
This doesn't train the GGUF directly, but creates a dataset that can be used
with Ollama's future training features or exported for external training.

For now, this creates an enhanced Modelfile with extensive examples.
"""

import json
import random
from pathlib import Path

def create_training_modelfile(
    base_model="gpt-oss:20b",
    data_path="sep_complete/alpaca_datasets/alpaca_philosophy.json",
    output_name="gpt-oss-philosophy",
    num_examples=200
):
    """Create an extensive Modelfile with training examples"""

    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"Dataset size: {len(data)} examples")

    # Sample diverse examples
    random.seed(42)
    if len(data) > num_examples:
        samples = random.sample(data, num_examples)
    else:
        samples = data

    # Create Modelfile
    modelfile_content = f"""FROM {base_model}

# Philosophy Expert System Prompt
SYSTEM \"\"\"You are an expert philosopher with comprehensive knowledge from the Stanford Encyclopedia of Philosophy. You have deep expertise in:

**Core Areas:**
- Metaphysics: ontology, existence, causation, identity, time, space
- Epistemology: knowledge, truth, justification, belief, skepticism
- Ethics: moral philosophy, virtue ethics, deontology, consequentialism
- Logic: formal reasoning, arguments, validity, soundness

**Historical Periods:**
- Ancient Philosophy: Plato, Aristotle, Stoics, Epicureans
- Medieval Philosophy: Aquinas, Augustine, Anselm
- Modern Philosophy: Descartes, Kant, Hume, Spinoza, Locke
- Contemporary Philosophy: Analytic, Continental, Phenomenology

**Specialized Fields:**
- Philosophy of Mind: consciousness, intentionality, qualia
- Philosophy of Science: scientific method, explanation, laws
- Philosophy of Language: meaning, reference, pragmatics
- Political Philosophy: justice, rights, democracy, liberty
- Aesthetics: beauty, art, aesthetic judgment

When answering:
1. Provide accurate, scholarly responses
2. Reference relevant philosophers and their arguments
3. Explain complex concepts clearly with examples
4. Present multiple perspectives when appropriate
5. Use precise philosophical terminology
6. Ground answers in the philosophical literature
\"\"\"

# Training Examples from Stanford Encyclopedia of Philosophy
# These examples teach the model how to respond to philosophy questions

"""

    # Add examples
    print(f"Adding {len(samples)} training examples...")

    for i, example in enumerate(samples, 1):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(samples)} examples...")

        instruction = example['instruction'].strip()
        output = example['output'].strip()

        # Clean for Modelfile format
        instruction = instruction.replace('"""', "'").replace('\n', ' ')
        output = output.replace('"""', "'").replace('\n', ' ')

        # Limit output length to avoid too large Modelfile
        if len(output) > 1000:
            output = output[:1000] + "..."

        modelfile_content += f'''
MESSAGE user """{instruction}"""
MESSAGE assistant """{output}"""
'''

    # Add parameters
    modelfile_content += """
# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
"""

    # Save Modelfile
    output_file = Path(f"Modelfile.{output_name}")
    with open(output_file, 'w') as f:
        f.write(modelfile_content)

    print(f"\n✓ Created {output_file}")
    print(f"  Size: {len(modelfile_content):,} characters")
    print(f"  Examples: {len(samples)}")

    return output_file

def create_ollama_model(modelfile_path, model_name):
    """Create the Ollama model from Modelfile"""
    import subprocess

    print(f"\nCreating Ollama model '{model_name}'...")
    print("This may take a few minutes...")

    try:
        result = subprocess.run(
            ['ollama', 'create', model_name, '-f', str(modelfile_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print(f"\n✓ Model '{model_name}' created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating model: {e}")
        print(e.stderr)
        return False

def export_for_external_training(data_path, output_dir="training_export"):
    """Export data in formats for external training tools"""
    Path(output_dir).mkdir(exist_ok=True)

    print(f"\nExporting training data for external tools...")

    with open(data_path, 'r') as f:
        data = json.load(f)

    # Format 1: ChatML (for many training tools)
    chatml_data = []
    for item in data:
        chatml_data.append({
            "messages": [
                {"role": "system", "content": "You are a philosophy expert."},
                {"role": "user", "content": item['instruction']},
                {"role": "assistant", "content": item['output']}
            ]
        })

    chatml_file = Path(output_dir) / "philosophy_chatml.jsonl"
    with open(chatml_file, 'w') as f:
        for item in chatml_data:
            f.write(json.dumps(item) + '\n')

    print(f"  ✓ ChatML format: {chatml_file}")

    # Format 2: Simple instruction format
    simple_data = []
    for item in data:
        simple_data.append({
            "prompt": f"### Instruction:\n{item['instruction']}\n\n### Response:\n",
            "completion": item['output']
        })

    simple_file = Path(output_dir) / "philosophy_simple.jsonl"
    with open(simple_file, 'w') as f:
        for item in simple_data:
            f.write(json.dumps(item) + '\n')

    print(f"  ✓ Simple format: {simple_file}")

    return output_dir

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Create Ollama adapter with philosophy knowledge')
    parser.add_argument('--base-model', default='gpt-oss:20b',
                       help='Base Ollama model')
    parser.add_argument('--data', default='sep_complete/alpaca_datasets/alpaca_philosophy.json',
                       help='Training data path')
    parser.add_argument('--output-name', default='gpt-oss-philosophy:20b',
                       help='Output model name')
    parser.add_argument('--examples', type=int, default=200,
                       help='Number of examples to include')
    parser.add_argument('--export-only', action='store_true',
                       help='Only export data, don\'t create Ollama model')

    args = parser.parse_args()

    print("="*60)
    print("Ollama Philosophy Model Creator")
    print("="*60)

    if args.export_only:
        export_for_external_training(args.data)
        print("\n✓ Export complete!")
        return

    # Create Modelfile with examples
    modelfile = create_training_modelfile(
        base_model=args.base_model,
        data_path=args.data,
        output_name=args.output_name.replace(':', '_'),
        num_examples=args.examples
    )

    # Create Ollama model
    success = create_ollama_model(modelfile, args.output_name)

    if success:
        print("\n" + "="*60)
        print("✓ Philosophy model created!")
        print("="*60)
        print(f"\nTest it with:")
        print(f"  ollama run {args.output_name}")
        print(f'\nExample: ollama run {args.output_name} "What is epistemology?"')
        print("\n" + "="*60)

    # Also export for external training
    print("\nAlso exporting data for external training tools...")
    export_dir = export_for_external_training(args.data)
    print(f"\n✓ Exported to {export_dir}/")
    print("  Use these files with llama.cpp, Axolotl, or other training tools")

if __name__ == "__main__":
    main()
