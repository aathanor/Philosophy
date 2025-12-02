#!/usr/bin/env python3
"""
Train Qwen2.5-7B with 4-bit quantization for 8GB VRAM
This is the ONLY configuration that reliably works on RTX 2060

Usage:
    python train_qwen_7b_4bit.py
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

# Set environment variable for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train Qwen2.5-7B with 4-bit quantization"""

    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    DATA_PATH = "sep_complete/alpaca_datasets/alpaca_philosophy.json"
    OUTPUT_DIR = "./qwen7b_philosophy"
    MAX_LENGTH = 512

    logger.info("="*60)
    logger.info("Qwen2.5-7B Philosophy Training (4-bit)")
    logger.info("="*60)

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return

    logger.info(f"CUDA: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset('json', data_files=DATA_PATH, split='train')
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset, eval_dataset = split['train'], split['test']
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    logger.info("Loading model with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Setup LoRA
    logger.info("Setting up LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable:,}")

    # Format data
    def format_prompt(example):
        return {
            "text": f"""<|im_start|>system
You are a philosophy expert with knowledge from the Stanford Encyclopedia of Philosophy.<|im_end|>
<|im_start|>user
{example['instruction']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""
        }

    logger.info("Formatting datasets...")
    train_dataset = train_dataset.map(format_prompt)
    eval_dataset = eval_dataset.map(format_prompt)

    # Tokenize
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    logger.info("Tokenizing...")
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Batch size 2 for 4-bit
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # Effective batch = 16
        learning_rate=2e-4,
        fp16=False,
        bf16=True,  # Use bfloat16 for stability
        save_steps=500,
        eval_steps=250,
        logging_steps=25,
        save_total_limit=2,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
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
    logger.info("Estimated time: 4-6 hours on RTX 2060")
    logger.info("="*60)

    trainer.train()

    # Save
    logger.info("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logger.info("="*60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {OUTPUT_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
