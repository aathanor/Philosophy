#!/usr/bin/env python3
"""
Minimal Qwen 7B training script for 8GB VRAM
Ultra-conservative settings to avoid OOM

Usage:
    python train_qwen_minimal.py
"""

import os
import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import logging

# Critical memory settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()
gc.collect()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    DATA_PATH = "sep_complete/alpaca_datasets/alpaca_philosophy.json"
    OUTPUT_DIR = "./qwen7b_philosophy"

    logger.info("="*60)
    logger.info("Qwen 7B Minimal Training (8GB VRAM)")
    logger.info("="*60)

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset('json', data_files=DATA_PATH, split='train')
    split = dataset.train_test_split(test_size=0.05, seed=42)  # Smaller eval set
    train_dataset = split['train']
    eval_dataset = split['test']
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Model with 4-bit - ultra conservative
    logger.info("Loading model (4-bit NF4)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Use float16 not bfloat16
        bnb_4bit_use_double_quant=False,  # Disable for memory
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},  # Force everything on GPU 0
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Disable caching
    model.config.use_cache = False

    # LoRA - very small config
    logger.info("Setting up LoRA (minimal config)...")
    peft_config = LoraConfig(
        r=8,  # Smaller rank
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Only Q and V, not all projections
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Format data - simpler format
    def format_prompt(example):
        return {"text": f"Question: {example['instruction']}\nAnswer: {example['output']}"}

    train_dataset = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_prompt, remove_columns=eval_dataset.column_names)

    # Tokenize
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=384, padding="max_length")

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Training arguments - minimal
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,  # Only 2 epochs
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_steps=1000,
        eval_steps=500,
        save_total_limit=1,
        eval_strategy="steps",
        warmup_steps=50,
        lr_scheduler_type="linear",
        optim="adamw_torch",  # Regular AdamW
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Train
    logger.info("="*60)
    logger.info("Starting training...")
    logger.info("Settings: Batch=1, GradAccum=16, Epochs=2")
    logger.info("="*60)

    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.info("Trying to save what we have...")
        trainer.save_model(f"{OUTPUT_DIR}_partial")
        raise

    # Save
    logger.info("Saving...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logger.info("="*60)
    logger.info(f"Complete! Saved to {OUTPUT_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
