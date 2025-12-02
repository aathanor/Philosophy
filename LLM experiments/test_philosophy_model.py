#!/usr/bin/env python3
"""
Test/inference script for the trained Philosophy LLM

Usage:
    python test_philosophy_model.py --prompt "Explain Kant's categorical imperative"
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

def load_model(base_model_name, lora_path):
    """Load the base model and LoRA weights"""
    print(f"Loading base model: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(lora_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"Loading LoRA weights from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)

    return model, tokenizer


def generate_response(model, tokenizer, instruction, input_text="", max_length=512):
    """Generate a response to an instruction"""

    # Format prompt
    if input_text:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response


def interactive_mode(model, tokenizer):
    """Interactive chat mode"""
    print("\n" + "="*60)
    print("Philosophy LLM - Interactive Mode")
    print("="*60)
    print("Type your philosophical questions. Type 'quit' to exit.\n")

    while True:
        try:
            instruction = input("You: ").strip()

            if instruction.lower() in ['quit', 'exit', 'q']:
                break

            if not instruction:
                continue

            print("\nPhilosophy LLM: ", end="", flush=True)
            response = generate_response(model, tokenizer, instruction)
            print(response)
            print()

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description='Test Philosophy LLM')
    parser.add_argument('--model-path', type=str,
                       default='./philosophy_llm_lora',
                       help='Path to trained LoRA model')
    parser.add_argument('--base-model', type=str,
                       default='meta-llama/Llama-2-7b-hf',
                       help='Base model name')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Single prompt to test')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum response length')

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.base_model, args.model_path)

    if args.prompt:
        # Single prompt mode
        print(f"\nPrompt: {args.prompt}\n")
        response = generate_response(model, tokenizer, args.prompt, max_length=args.max_length)
        print(f"Response: {response}\n")

    elif args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer)

    else:
        # Demo mode - run some example prompts
        print("\n" + "="*60)
        print("Philosophy LLM - Demo Mode")
        print("="*60 + "\n")

        examples = [
            "What is epistemology?",
            "Explain Kant's categorical imperative",
            "What is the difference between deontology and consequentialism?",
            "Who was Plato and what were his main ideas?",
            "Explain the concept of free will",
        ]

        for i, example in enumerate(examples, 1):
            print(f"\n[Example {i}]")
            print(f"Q: {example}")
            print(f"A: ", end="", flush=True)
            response = generate_response(model, tokenizer, example, max_length=args.max_length)
            print(response)
            print("-" * 60)


if __name__ == "__main__":
    main()
