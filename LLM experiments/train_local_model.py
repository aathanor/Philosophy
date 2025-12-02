#!/usr/bin/env python3
"""
Fine-tune a local model (e.g., GPT OSS 20B) with Philosophy dataset using QLoRA
Optimized for 8GB VRAM using 4-bit quantization

Usage:
    python train_local_model.py --model-path /path/to/your/gpt-oss-20b
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


class LocalModelTrainer:
    def __init__(
        self,
        model_path,
        data_path="sep_complete/alpaca_datasets/alpaca_philosophy.json",
        output_dir="./philosophy_finetuned",
        max_length=512,
        use_4bit=True
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_length = max_length
        self.use_4bit = use_4bit

        logger.info(f"Local model path: {model_path}")
        logger.info(f"Output directory: {output_dir}")

    def load_and_prepare_data(self):
        """Load and format the philosophy dataset"""
        logger.info(f"Loading dataset from {self.data_path}")

        dataset = load_dataset('json', data_files=self.data_path, split='train')
        logger.info(f"Loaded {len(dataset)} training examples")

        # Split into train/validation
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']

        logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
        return train_dataset, eval_dataset

    def format_instruction(self, example):
        """Format instruction-response pairs (Alpaca format)"""
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
        """Load local model with 4-bit quantization"""
        logger.info(f"Loading tokenizer from {self.model_path}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except:
            # If tokenizer not found locally, try common alternatives
            logger.warning("Tokenizer not found in model path, trying GPT-NeoX tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

        # Fix tokenizer padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        logger.info(f"Loading model from {self.model_path}...")
        logger.info("Using 4-bit quantization for memory efficiency...")

        if self.use_4bit:
            # 4-bit quantization config for 20B model on 8GB VRAM
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

        logger.info("Model loaded successfully")
        return model, tokenizer

    def setup_lora(self, model):
        """Configure LoRA - more aggressive for 20B model"""
        logger.info("Setting up LoRA configuration...")

        # More conservative LoRA for larger model
        lora_config = LoraConfig(
            r=8,  # Lower rank for memory efficiency
            lora_alpha=16,
            target_modules=[
                "query_key_value",  # GPT-NeoX style
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())

        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
        logger.info(f"All parameters: {all_params:,}")

        return model

    def train(self):
        """Main training loop"""
        logger.info("="*60)
        logger.info("Starting Philosophy Fine-tuning")
        logger.info("="*60)

        # Load data
        train_dataset, eval_dataset = self.load_and_prepare_data()

        # Load model
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

        # Training arguments - very conservative for 20B on 8GB
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Very small batch for 20B model
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,  # Effective batch size = 16
            learning_rate=2e-4,
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

        # Train
        logger.info("Starting training...")
        logger.info("Note: With 20B parameters, this will take a while!")
        trainer.train()

        # Save
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        logger.info("="*60)
        logger.info("Training complete!")
        logger.info(f"Model saved to: {self.output_dir}")
        logger.info("="*60)

        return model, tokenizer


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fine-tune local model with philosophy dataset')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to local model directory')
    parser.add_argument('--data', type=str,
                       default='sep_complete/alpaca_datasets/alpaca_philosophy.json',
                       help='Path to training data')
    parser.add_argument('--output', type=str,
                       default='./philosophy_finetuned',
                       help='Output directory')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--use-8bit', action='store_true',
                       help='Use 8-bit instead of 4-bit (may not fit in 8GB for 20B)')

    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available! This script requires a GPU.")
        return

    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Check model path exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        logger.info("Please provide the correct path to your GPT OSS 20B model")
        return

    # Create trainer
    trainer = LocalModelTrainer(
        model_path=args.model_path,
        data_path=args.data,
        output_dir=args.output,
        max_length=args.max_length,
        use_4bit=not args.use_8bit
    )

    trainer.train()


if __name__ == "__main__":
    main()
