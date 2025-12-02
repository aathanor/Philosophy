#!/usr/bin/env python3
"""
Train a Philosophy LLM using LoRA (Low-Rank Adaptation)
Optimized for RTX 2060 8GB VRAM

Usage:
    python train_philosophy_lora.py
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhilosophyLLMTrainer:
    def __init__(
        self,
        data_path="sep_complete/alpaca_datasets/alpaca_philosophy.json",
        model_name="meta-llama/Llama-2-7b-hf",  # Can also use "mistralai/Mistral-7B-v0.1"
        output_dir="./philosophy_llm_lora",
        max_length=512,
        use_8bit=True
    ):
        self.data_path = data_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.use_8bit = use_8bit

        logger.info(f"Initializing trainer with model: {model_name}")
        logger.info(f"Output directory: {output_dir}")

    def load_and_prepare_data(self):
        """Load and format the philosophy dataset"""
        logger.info(f"Loading dataset from {self.data_path}")

        # Load dataset
        dataset = load_dataset('json', data_files=self.data_path, split='train')
        logger.info(f"Loaded {len(dataset)} training examples")

        # Split into train/validation
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']

        logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

        return train_dataset, eval_dataset

    def format_instruction(self, example):
        """Format instruction-response pairs for training"""
        instruction = example['instruction']
        output = example['output']
        input_text = example.get('input', '')

        if input_text:
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""

        return {"text": prompt}

    def load_model_and_tokenizer(self):
        """Load model with 8-bit quantization and prepare for LoRA"""
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Fix tokenizer padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        logger.info("Loading model with 8-bit quantization...")

        # Configure 8-bit quantization for memory efficiency
        if self.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offload for 8GB VRAM
            )

            # Set memory limits for 8GB GPU
            max_memory = {0: "7GB", "cpu": "20GB"}

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        logger.info("Model loaded successfully")
        return model, tokenizer

    def setup_lora(self, model):
        """Configure LoRA for efficient fine-tuning"""
        logger.info("Setting up LoRA configuration...")

        # LoRA configuration optimized for 8GB VRAM
        lora_config = LoraConfig(
            r=16,  # LoRA rank (lower = less memory, but less capacity)
            lora_alpha=32,  # LoRA scaling factor
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

        # Apply LoRA to model
        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())

        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
        logger.info(f"All parameters: {all_params:,}")

        return model

    def train(self):
        """Main training loop"""
        logger.info("="*60)
        logger.info("Starting Philosophy LLM Training")
        logger.info("="*60)

        # Load data
        train_dataset, eval_dataset = self.load_and_prepare_data()

        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()

        # Setup LoRA
        model = self.setup_lora(model)

        # Format datasets
        logger.info("Formatting datasets...")
        train_dataset = train_dataset.map(self.format_instruction)
        eval_dataset = eval_dataset.map(self.format_instruction)

        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )

        logger.info("Tokenizing datasets...")
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

        # Training arguments optimized for RTX 2060
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,  # Adjust based on VRAM
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,  # Effective batch size = 16
            learning_rate=2e-4,
            fp16=True,  # Mixed precision training
            save_steps=200,
            eval_steps=100,
            logging_steps=10,
            save_total_limit=3,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            warmup_steps=100,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",  # Memory-efficient optimizer
            gradient_checkpointing=True,  # Save memory
            report_to="none",  # Disable wandb/tensorboard
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train!
        logger.info("Starting training...")
        trainer.train()

        # Save the final model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        logger.info("="*60)
        logger.info("Training complete!")
        logger.info(f"Model saved to: {self.output_dir}")
        logger.info("="*60)

        return model, tokenizer


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Train Philosophy LLM with LoRA')
    parser.add_argument('--data', type=str,
                       default='sep_complete/alpaca_datasets/alpaca_philosophy.json',
                       help='Path to training data')
    parser.add_argument('--model', type=str,
                       default='meta-llama/Llama-2-7b-hf',
                       help='Base model to fine-tune')
    parser.add_argument('--output', type=str,
                       default='./philosophy_llm_lora',
                       help='Output directory')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Training will be very slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create trainer and train
    trainer = PhilosophyLLMTrainer(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        max_length=args.max_length
    )

    trainer.train()


if __name__ == "__main__":
    main()
