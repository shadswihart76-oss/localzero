# Model Card: LocalZero

## Model details
- **Name:** LocalZero
- **Base model:** Qwen2.5-7B-Instruct (scaling to 32B-70B with granted compute)
- **Method:** QLoRA (4-bit nf4, LoRA r=16 / alpha=32, target q/k/v/o + MLP)
- **Domain:** defensive security / CTF reasoning / adversary emulation

## Intended use
Authorized vulnerability analysis, CTF-style training, and defensive security
reasoning. NOT intended to automate attacks against systems without authorization.

## Training data
~137k examples from public HF datasets (Fenrir v2.1, CTFtime, ctf-satml24,
agent-ctf24-public, prompt_injection_ctf). See README for full citations.

## Compute
Local QLoRA on consumer GPU (8GB, RTX 5050) for v1; production 32B-70B runs on
A100 80GB provided by the CloudRift AI Grant.

## Evaluation
Prompt battery across recon, web, XSS, crypto, reverse engineering, ATT&CK,
scripting, and privilege escalation. Full results in eval.py.

## Ethics & safety
Built for defensive capability. The model provides authorized-only guidance;
it is a tool for understanding attacker technique, not a weapon.