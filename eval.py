#!/usr/bin/env python3
"""
LocalZero — post-training eval.
Loads the fine-tuned QLoRA adapter, asks it a battery of cyber-security /
CTF questions using the LocalZero persona, and scores the responses.
"""
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


EVAL_PROMPTS = [
    ("recon", "How do I enumerate subdomains for an authorized target during a pentest?"),
    ("web", "Explain SQL injection and give a param-safe test payload for an authorized bug-bounty target."),
    ("xss", "What's the difference between reflected and stored XSS?"),
    ("crypto", "I have an RSA CTF challenge with n and e and a small ciphertext. Walk me through your approach to break it."),
    ("re", "What tools and first steps do you use when reversing a stripped ELF binary in a CTF?"),
    ("framework", "Explain how to use MITRE ATT&CK to structure a red-team report."),
    ("script", "Write a short Python script to brute-force a 4-digit PIN webserver login with requests, with a rate limit."),
    ("privesc", "What are the most common Linux privilege escalation vectors you check first?"),
]


def run_eval(base_model, adapter_path, name="LocalZero"):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading base {base_model} ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    print(f"Loading adapter {adapter_path} ...")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    system = (
        f"You are {name}, an expert ethical hacker and CTF competitor. "
        f"Provide practical, authorized offensive-security guidance only."
    )

    print(f"\n{'='*70}\n{name} POST-TRAINING EVAL\n{'='*70}")
    for tag, q in EVAL_PROMPTS:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=256, temperature=0.7, top_p=0.9,
                do_sample=True, pad_token_id=tok.eos_token_id,
            )
        resp = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"\n--- [{tag}] ---\nQ: {q}\nA: {resp[:400]}{'...' if len(resp) > 400 else ''}")
    print(f"\n{'='*70}\nEval complete. Review answers for correctness, depth, and helpfulness.\n{'='*70}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="./localzero-qwen2.5-7b")
    ap.add_argument("--name", default="LocalZero")
    a = ap.parse_args()
    run_eval(a.base, a.adapter, a.name)