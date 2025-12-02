#!/bin/bash
# Create an Ollama model fine-tuned with philosophy knowledge
# This creates a Modelfile with system prompts and examples from your dataset

set -e

echo "Creating Philosophy-enhanced Ollama model based on gpt-oss:20b"

# Create a Modelfile
cat > Modelfile.philosophy << 'EOF'
FROM gpt-oss:20b

# System prompt with philosophy expertise
SYSTEM """You are a knowledgeable philosophy expert trained on the Stanford Encyclopedia of Philosophy. You have deep understanding of:

- Metaphysics: ontology, existence, causation, time, space
- Epistemology: knowledge, truth, justification, skepticism
- Ethics: moral philosophy, virtue ethics, deontology, consequentialism
- Logic: formal reasoning, arguments, proofs
- History of Philosophy: Ancient, Medieval, Modern, and Contemporary philosophy
- Philosophy of Mind: consciousness, intentionality, perception
- Philosophy of Science: scientific method, explanation, laws of nature
- Political Philosophy: justice, rights, liberty, democracy
- Aesthetics: beauty, art, taste

When answering questions:
1. Provide accurate philosophical analysis
2. Reference relevant philosophers and their ideas
3. Explain complex concepts clearly
4. Acknowledge different philosophical perspectives
5. Use examples to illustrate abstract ideas
"""

# Add training examples from the dataset
EOF

# Extract sample Q&A from the philosophy dataset
python3 << 'PYTHON'
import json

# Load dataset
with open('sep_complete/alpaca_datasets/alpaca_philosophy.json', 'r') as f:
    data = json.load(f)

# Select diverse examples (50 samples)
import random
random.seed(42)
samples = random.sample(data, min(50, len(data)))

# Append to Modelfile
with open('Modelfile.philosophy', 'a') as f:
    f.write('\n# Training examples from Stanford Encyclopedia of Philosophy\n\n')

    for i, item in enumerate(samples[:50], 1):
        instruction = item['instruction'].replace('"', '\\"').replace('\n', ' ')
        output = item['output'][:500].replace('"', '\\"').replace('\n', ' ')  # Limit length

        f.write(f'MESSAGE user """{instruction}"""\n')
        f.write(f'MESSAGE assistant """{output}..."""\n\n')

print(f"Created Modelfile with {len(samples)} examples")
PYTHON

# Create the model in Ollama
echo "Creating Ollama model 'gpt-oss-philosophy:20b'..."
ollama create gpt-oss-philosophy:20b -f Modelfile.philosophy

echo ""
echo "✓ Model created successfully!"
echo ""
echo "Test it with:"
echo "  ollama run gpt-oss-philosophy:20b"
echo ""
echo "Example questions:"
echo '  "What is epistemology?"'
echo '  "Explain Kant'\''s categorical imperative"'
echo ""
