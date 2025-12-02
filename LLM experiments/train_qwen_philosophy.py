#!/usr/bin/env python3
"""
Train Qwen2.5-14B with Philosophy dataset using QLoRA
Optimized for RTX 2060 8GB VRAM

Usage:
    python train_qwen_philosophy.py
"""

import os
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train Qwen2.5-14B with philosophy dataset"""

    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
    DATA_PATH = "sep_complete/alpaca_datasets/alpaca_philosophy.json"
    OUTPUT_DIR = "./qwen_philosophy"
    MAX_LENGTH = 512

    logger.info("="*60)
    logger.info("Qwen2.5-14B Philosophy Training")
    logger.info("="*60)

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return

    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load dataset
    logger.info(f"Loading dataset from {DATA_PATH}...")
    dataset = load_dataset('json', data_files=DATA_PATH, split='train')
    logger.info(f"Loaded {len(dataset)} examples")

    # Split dataset
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    # Qwen uses specific chat format
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with 4-bit quantization
    logger.info("Loading model with 4-bit quantization...")
    logger.info("This may take a few minutes...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offload
    )

    # Calculate available GPU memory (use 90% to be safe)
    max_memory = {0: "7GB", "cpu": "24GB"}

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_folder="offload",  # Offload to disk if needed
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration - conservative for 14B
    logger.info("Setting up LoRA...")
    lora_config = LoraConfig(
        r=8,  # Low rank for memory
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")

    # Format data for Qwen
    def format_qwen_chat(example):
        """Format in Qwen's chat format"""
        instruction = example['instruction']
        output = example['output']

        # Qwen chat format
        messages = [
            {"role": "system", "content": "You are a philosophy expert with comprehensive knowledge from the Stanford Encyclopedia of Philosophy."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]

        # Use Qwen's chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        return {"text": text}

    logger.info("Formatting datasets...")
    train_dataset = train_dataset.map(format_qwen_chat)
    eval_dataset = eval_dataset.map(format_qwen_chat)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )

    logger.info("Tokenizing...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    # Training arguments - very conservative for 14B on 8GB
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Batch size 1 for 14B
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,  # Effective batch = 32
        learning_rate=1e-4,
        fp16=True,
        save_steps=500,
        eval_steps=250,
        logging_steps=25,
        save_total_limit=2,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to="none",
        dataloader_num_workers=0,  # Avoid memory issues
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("="*60)
    logger.info("Starting training...")
    logger.info("Note: 14B model will train slower than smaller models")
    logger.info("Estimated time: 12-16 hours on RTX 2060")
    logger.info("="*60)

    trainer.train()

    # Save
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logger.info("="*60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {OUTPUT_DIR}")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Merge LoRA adapters with base model")
    logger.info("2. Convert to GGUF format")
    logger.info("3. Import to Ollama")
    logger.info("\nSee README for conversion instructions.")


if __name__ == "__main__":
    main()
