"""
training/finetune.py

Fine-tunes CodeLlama-7B on PR diff data using LoRA / QLoRA.

Usage:
    python training/finetune.py \
        --model  codellama/CodeLlama-7b-hf \
        --data   data/processed/train.jsonl \
        --output models/codellama-review-lora \
        --epochs 3 \
        --lora-rank 16
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from loguru import logger
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


# ─── Args ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune CodeLlama for code review")
    p.add_argument("--model",      default="codellama/CodeLlama-7b-hf")
    p.add_argument("--data",       default="data/processed/train.jsonl")
    p.add_argument("--val-data",   default="data/processed/val.jsonl")
    p.add_argument("--output",     default="models/codellama-review-lora")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--grad-accum", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--lora-rank",  type=int,   default=16)
    p.add_argument("--lora-alpha", type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-length", type=int,   default=2048)
    p.add_argument("--use-4bit",   action="store_true", default=True)
    p.add_argument("--wandb",      action="store_true", default=False)
    return p.parse_args()


# ─── Prompt Template ─────────────────────────────────────────────────────────

SYSTEM = (
    "You are an expert code reviewer. Analyze the diff and return a JSON "
    "object with a 'findings' list identifying bugs, anti-patterns, or security issues."
)

def format_example(row: dict) -> str:
    """Convert a dataset row into a training prompt."""
    diff   = row.get("diff", "")
    label  = row.get("label_json", "{\"findings\": []}")
    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM}\n<</SYS>>\n\n"
        f"File: {row.get('file_path', 'unknown')}\n"
        f"Language: {row.get('language', 'python')}\n\n"
        f"```diff\n{diff}\n```\n[/INST] {label} </s>"
    )


def tokenize(example, tokenizer, max_length):
    text = format_example(example)
    return tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.wandb:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "llm-code-review"),
            config=vars(args),
        )

    # ── Tokenizer ────────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"Loading base model: {args.model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ) if args.use_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # ── LoRA Config ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset from {args.data}")
    raw = load_dataset(
        "json",
        data_files={"train": args.data, "validation": args.val_data},
    )

    tokenized = raw.map(
        lambda ex: tokenize(ex, tokenizer, args.max_length),
        remove_columns=raw["train"].column_names,
        desc="Tokenizing",
    )

    logger.info(f"Train: {len(tokenized['train'])} examples")
    logger.info(f"Val:   {len(tokenized['validation'])} examples")

    # ── Training Args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=50,
        eval_steps=200,
        save_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb" if args.wandb else "none",
        dataloader_num_workers=4,
        group_by_length=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit" if args.use_4bit else "adamw_torch",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    logger.info("Starting fine-tuning …")
    trainer.train()

    logger.info(f"Saving adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("Done ✓")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
