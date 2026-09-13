# LocalZero

An open-weight defensive-security language model — fine-tuned to reason about
vulnerability analysis, adversary emulation, and CTF-style security challenges.

Compute for the production runs is provided by the
[CloudRift AI Grant](https://cloudrift.ai/ai-grant).

---

## What it is

LocalZero is a LoRA/QLoRA fine-tune of an open base model (Qwen2.5-family) on
~137,000 structured security question/answer examples covering:

| Domain | Source |
|--------|--------|
| OWASP Top 10 web vulns (SQLi, XSS, injection classes) | AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1 |
| MITRE ATT&CK adversary TTPs | Fenrir v2.1 |
| CTF writeups (recon, RE, crypto) | justinwangx/CTFtime |
| SATML adversarial conversations | ethz-spylab/ctf-satml24 |
| Prompt-injection detection | cgoosen/prompt_injection_ctf_dataset_* |
| Agent-security scenario analysis | invariantlabs/agent-ctf24-public |

## Why

Defenders are short on realistic red-team capability. Offensive knowledge is the
most under-served part of the open-model ecosystem, and CTF-style training is the
standard way security teams practice detection and response. LocalZero makes that
knowledge available as an open-weight model anyone can audit, self-host, and improve.

## Repo layout

```
localzero/
  train.py               QLoRA fine-tuning (single-GPU, 7B; scales to 70B on A100)
  eval.py                post-training skill battery, calls the model by name
  prepare_data.py        dataset loading + formatting + LocalZero persona prompt
  model_card.md          model provenance + CloudRift credit
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers peft bitsandbytes trl datasets accelerate

# Train (7B, fits on ~8GB consumer GPU)
python3 train.py --model Qwen/Qwen2.5-7B-Instruct --name LocalZero --epochs 1

# Eval the resulting adapter
python3 eval.py --adapter ./localzero-qwen2.5-7b --name LocalZero
```

## Licensing

Model weights inherit the base model's license. Code in this repo is Apache-2.0.

## Acknowledgments

- Compute: CloudRift AI Grant (production runs on A100 80GB)
- Base model: Qwen2.5 (Apache-2.0)
- Data: HF datasets cited above