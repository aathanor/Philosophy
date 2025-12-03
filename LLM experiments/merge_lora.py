#!/usr/bin/env python3
"""
Merge LoRA adapters with base model for 8GB VRAM
Uses CPU offloading to handle memory constraints
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Create offload directory
os.makedirs("offload_merge", exist_ok=True)

print("="*60)
print("Merging LoRA adapters with base model")
print("="*60)

print("\nLoading base model with CPU offload...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload_merge",
    trust_remote_code=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(
    base_model,
    "./qwen7b_philosophy",
    offload_folder="offload_merge"
)

print("Merging... (this may take 5-10 minutes)")
merged_model = model.merge_and_unload()

print("Saving merged model...")
merged_model.save_pretrained(
    "./qwen7b_philosophy_merged",
    max_shard_size="2GB"  # Split into smaller files
)
tokenizer.save_pretrained("./qwen7b_philosophy_merged")

print("\n" + "="*60)
print("✓ Success! Merged model saved to ./qwen7b_philosophy_merged")
print("="*60)
print("\nNext: Convert to GGUF format")
