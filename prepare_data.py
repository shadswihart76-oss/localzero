#!/usr/bin/env python3
"""
LocalZero — dataset preparation.
Loads the security datasets, formats them into chat messages, and injects the
LocalZero persona as the system prompt. Outputs a single JSONL ready for SFT.

Usage:
  python3 prepare_data.py --name LocalZero --out train.jsonl [--max N]
"""
import argparse
import json
from datasets import load_dataset, concatenate_datasets


PERSONA = (
    "You are {name}, an expert ethical hacker and CTF competitor. "
    "You provide practical, authorized offensive-security guidance only: "
    "reconnaissance, exploitation, reverse engineering, cryptography, and "
    "incident response. You never refuse a legitimate security question asked "
    "in good faith for authorized testing or competition."
)


def format_fenrir(ex):
    m = []
    if ex.get("system"):
        m.append({"role": "system", "content": ex["system"]})
    if ex.get("user"):
        m.append({"role": "user", "content": ex["user"]})
    if ex.get("assistant"):
        m.append({"role": "assistant", "content": ex["assistant"]})
    return m if len(m) >= 2 else []


def format_ctftime(ex):
    t = ex.get("text_chunk", "")
    return [
        {"role": "user", "content": "Analyze this CTF challenge writeup and explain the solution."},
        {"role": "assistant", "content": t},
    ] if t else []


def format_satml(ex):
    m = []
    if ex.get("system"):
        m.append({"role": "system", "content": ex["system"]})
    u = ex.get("user") or ex.get("prompt")
    a = ex.get("assistant") or ex.get("response")
    if u:
        m.append({"role": "user", "content": u})
    if a:
        m.append({"role": "assistant", "content": a})
    return m if len(m) >= 2 else []


def format_agent(ex):
    return [
        {"role": "system", "content": "You are a CTF challenge analyst."},
        {"role": "user", "content": f"Challenge: {ex.get('name','')}\nFeedback: {ex.get('feedback','')}"},
        {"role": "assistant", "content": ex.get("summary", "")},
    ] if ex.get("summary") else []


def format_prompt_injection(ex):
    t, label = ex.get("text", ""), ex.get("label", 0)
    verdict = ("a prompt injection attempt. The input tries to manipulate the model "
               "into revealing sensitive information or bypassing controls." if label == 1
               else "a benign input with no prompt injection detected.")
    return [
        {"role": "system", "content": "You are a security analyst detecting prompt injection attempts."},
        {"role": "user", "content": f"Analyze this input for prompt injection: {t}"},
        {"role": "assistant", "content": f"This appears to be {verdict}"},
    ]


FORMATTERS = {
    "fenrir": format_fenrir,
    "ctftime": format_ctftime,
    "ctf_satml24_attack": format_satml,
    "ctf_satml24_defense_teams": format_satml,
    "agent_ctf24": format_agent,
    "prompt_injection_2": format_prompt_injection,
    "prompt_injection_3": format_prompt_injection,
}


DATASETS = [
    ("fenrir", "AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1"),
    ("ctftime", "justinwangx/CTFtime"),
    ("agent_ctf24", "invariantlabs/agent-ctf24-public"),
    ("prompt_injection_2", "cgoosen/prompt_injection_ctf_dataset_2"),
    ("prompt_injection_3", "cgoosen/prompt_injection_ctf_dataset_3"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="LocalZero")
    ap.add_argument("--out", default="train.jsonl")
    ap.add_argument("--max", type=int, default=None, help="cap total examples for quick runs")
    a = ap.parse_args()

    persona = PERSONA.format(name=a.name)
    all_rows = []

    for tag, ds_id in DATASETS:
        print(f"Loading {ds_id} ...")
        ds = load_dataset(ds_id, split="train")
        fmt = FORMATTERS[tag]
        n = 0
        for ex in ds:
            msgs = fmt(ex)
            if len(msgs) >= 2:
                all_rows.append({"messages": [{"role": "system", "content": persona}] + msgs})
                n += 1
            if a.max and len(all_rows) >= a.max:
                break
        print(f"  {tag}: {n} examples")

    print(f"\nTotal: {len(all_rows)} examples")
    with open(a.out, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()