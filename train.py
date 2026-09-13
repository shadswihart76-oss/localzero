#!/usr/bin/env python3
"""
DeepSpeed ZeRO-3 + QLoRA fine-tuning for cybersecurity models
Trains 7B-70B models on 8GB VRAM via CPU/NVMe offload
"""

import os
import json
import argparse
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig


def load_cyber_datasets():
    """Load all available cybersecurity + CTF datasets"""
    print("Loading datasets...")
    all_datasets = []
    
    # 1. Fenrir v2.1 - 99K defensive cybersecurity
    print("  Loading AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1...")
    ds = load_dataset("AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1", split="train")
    print(f"    Loaded {len(ds)} examples")
    all_datasets.append(("fenrir", ds))
    
    # 2. CTFtime - 18K CTF writeups
    print("  Loading justinwangx/CTFtime...")
    ds = load_dataset("justinwangx/CTFtime", split="train")
    print(f"    Loaded {len(ds)} examples")
    all_datasets.append(("ctftime", ds))
    
    # 3. SATML CTF datasets
    print("  Loading ethz-spylab/ctf-satml24...")
    for config in ["attack", "defense_teams"]:
        try:
            ds = load_dataset("ethz-spylab/ctf-satml24", config, split="train")
            print(f"    {config}: {len(ds)} examples")
            all_datasets.append((f"ctf_satml24_{config}", ds))
        except Exception as e:
            print(f"    {config}: ERROR - {e}")
    
    # 4. Agent CTF
    print("  Loading invariantlabs/agent-ctf24-public...")
    ds = load_dataset("invariantlabs/agent-ctf24-public", split="train")
    print(f"    Loaded {len(ds)} examples")
    all_datasets.append(("agent_ctf24", ds))
    
    # 5. Prompt injection CTF
    for name in ["cgoosen/prompt_injection_ctf_dataset_2", "cgoosen/prompt_injection_ctf_dataset_3"]:
        try:
            ds = load_dataset(name, split="train")
            print(f"    {name}: {len(ds)} examples")
            all_datasets.append((name.replace("/", "_"), ds))
        except Exception as e:
            print(f"    {name}: ERROR - {e}")
    
    return all_datasets


def format_fenrir(example):
    messages = []
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})
    if example.get("user"):
        messages.append({"role": "user", "content": example["user"]})
    if example.get("assistant"):
        messages.append({"role": "assistant", "content": example["assistant"]})
    return {"messages": messages} if len(messages) >= 2 else {"messages": []}


def format_ctftime(example):
    text = example.get("text_chunk", "")
    if not text:
        return {"messages": []}
    messages = [
        {"role": "user", "content": "Analyze this CTF challenge writeup and explain the solution:"},
        {"role": "assistant", "content": text}
    ]
    return {"messages": messages}


def format_ctf_satml24_attack(example):
    messages = []
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})
    if example.get("user") or example.get("prompt"):
        messages.append({"role": "user", "content": example.get("user") or example.get("prompt")})
    if example.get("assistant") or example.get("response"):
        messages.append({"role": "assistant", "content": example.get("assistant") or example.get("response")})
    return {"messages": messages} if len(messages) >= 2 else {"messages": []}


def format_ctf_satml24_defense_teams(example):
    messages = []
    if example.get("team_description"):
        messages.append({"role": "system", "content": f"CTF Team: {example['team_description']}"})
    if example.get("challenge"):
        messages.append({"role": "user", "content": example["challenge"]})
    if example.get("solution"):
        messages.append({"role": "assistant", "content": example["solution"]})
    return {"messages": messages} if len(messages) >= 2 else {"messages": []}


def format_agent_ctf24(example):
    messages = [
        {"role": "system", "content": "You are a CTF challenge analyst. Summarize feedback and findings."},
        {"role": "user", "content": f"Challenge: {example.get('name', '')}\nFeedback: {example.get('feedback', '')}"},
        {"role": "assistant", "content": example.get('summary', '')}
    ]
    return {"messages": messages} if example.get('summary') else {"messages": []}


def format_prompt_injection(example):
    text = example.get("text", "")
    label = example.get("label", 0)
    if label == 1:
        messages = [
            {"role": "system", "content": "You are a security analyst detecting prompt injection attempts."},
            {"role": "user", "content": f"Analyze this input for prompt injection: {text}"},
            {"role": "assistant", "content": "This appears to be a prompt injection attempt. The input tries to manipulate the model into revealing sensitive information or bypassing controls."}
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a security analyst detecting prompt injection attempts."},
            {"role": "user", "content": f"Analyze this input for prompt injection: {text}"},
            {"role": "assistant", "content": "This appears to be a benign input with no prompt injection detected."}
        ]
    return {"messages": messages}


FORMATTERS = {
    "fenrir": format_fenrir,
    "ctftime": format_ctftime,
    "ctf_satml24_attack": format_ctf_satml24_attack,
    "ctf_satml24_defense_teams": format_ctf_satml24_defense_teams,
    "agent_ctf24": format_agent_ctf24,
    "cgoosen_prompt_injection_ctf_dataset_2": format_prompt_injection,
    "cgoosen_prompt_injection_ctf_dataset_3": format_prompt_injection,
}


def prepare_training_data(all_datasets, tokenizer, max_seq_length=4096, system_prompt=None):
    """Format all datasets and apply chat template"""
    print("\nFormatting datasets...")
    formatted_datasets = []
    for name, ds in all_datasets:
        formatter = FORMATTERS.get(name)
        if not formatter:
            print(f"  No formatter for {name}, skipping")
            continue
        print(f"  Formatting {name}...")
        formatted = ds.map(formatter, remove_columns=ds.column_names)
        formatted = formatted.filter(lambda x: x["messages"] and len(x["messages"]) >= 2)
        print(f"    Valid examples: {len(formatted)}")
        formatted_datasets.append(formatted)
    
    if not formatted_datasets:
        raise ValueError("No valid datasets after formatting!")
    
    combined = concatenate_datasets(formatted_datasets)
    print(f"\nTotal combined: {len(combined)} examples")
    combined = combined.shuffle(seed=42)
    
    # Apply chat template, prepending the LocalZero system prompt for consistent persona
    def apply_chat_template(example):
        messages = example["messages"]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    
    print("Applying chat templates...")
    combined = combined.map(apply_chat_template, remove_columns=["messages"])
    # SFTTrainer tokenizes internally using its own `max_length` + packing/padding
    # config. Handing it pre-tokenized 2048-token padded sequences just wastes work
    # and slows the first step to a crawl. Return raw text strings instead.
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model (7B fits 8GB with QLoRA, 32B+ needs ZeRO-3)")
    parser.add_argument("--output", default="./localzero-qwen2.5-7b")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--name", default="LocalZero",
                        help="Model codename (baked into the system-prompt persona)")
    parser.add_argument("--report-to", default="none", choices=["none", "wandb", "tensorboard"],
                        help="Live monitoring backend")
    args = parser.parse_args()
    
    torch.manual_seed(42)
    
    SYSTEM_PROMPT = (
        f"You are {args.name}, an expert ethical hacker and CTF competitor. "
        f"You provide practical, authorized offensive-security guidance only: "
        f"reconnaissance, exploitation, reverse engineering, cryptography, and "
        f"incident response. You never refuse a legitimate security question asked "
        f"in good faith for authorized testing or competition."
    )
    
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Name: {args.name}")
    print(f"Monitoring: {args.report_to}")
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    
    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},  # Single GPU
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    
    # LoRA config for Qwen2.5
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare datasets
    all_datasets = load_cyber_datasets()
    train_dataset = prepare_training_data(all_datasets, tokenizer, args.max_seq_len, system_prompt=SYSTEM_PROMPT)
    print(f"Training samples: {len(train_dataset)}")
    
    # Training config (GPU QLoRA, no DeepSpeed needed)
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=args.lr,
        bf16=True,                  # bf16 is native on sm_120; fp16 amp unscale isn't implemented for bf16
        logging_steps=1,            # Log every step so the watcher can show live progress
        save_strategy="steps",
        save_steps=500,             # checkpoint every 500 steps so a reboot loses minutes, not a week
        save_total_limit=3,         # keep a few recent checkpoints, resume-able
        remove_unused_columns=False,
        report_to=args.report_to,   # "wandb" if token present, else "none"
        optim="paged_adamw_8bit",   # memory-efficient optimizer for 8GB
        warmup_steps=100,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        max_length=args.max_seq_len,
        packing=False,              # avoid cross-sample contamination w/o flash-attn; slower but correct
        dataset_text_field="text",
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    print("\nStarting QLoRA training on GPU...")
    print(f"Live progress: tail -f {args.output} logs, or watch the wandb dashboard if enabled")
    trainer.train()
    
    # Save
    print(f"\nSaving to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    
    print("Training complete!")


if __name__ == "__main__":
    main()