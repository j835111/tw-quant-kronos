---
name: colab
description: "Run Kronos training on Google Colab GPU/TPU via the official Colab CLI (google-colab-cli)"
version: 1.0.0
---

# Colab CLI — Kronos Training on Remote GPU

## Prerequisites

```bash
# Install (already done)
uv tool install google-colab-cli

# Verify
colab version  # should print 0.6.0+

# First-time auth — prints a URL, user pastes the code back
colab sessions  # triggers OAuth on first run
```

> Authentication uses a remote copy-paste flow (same as `gcloud auth application-default login`).
> The CLI prints a URL → user opens in browser → copies the code → pastes at the prompt.
> Token is cached at `~/.config/colab-cli/token.json` after first login.

---

## GPU Options

| Flag | Hardware | Notes |
|------|----------|-------|
| *(none)* | CPU | Free |
| `--gpu T4` | NVIDIA T4 | Standard free-tier GPU |
| `--gpu L4` | NVIDIA L4 | Cost-effective modern GPU |
| `--gpu A100` | NVIDIA A100 | High-performance, paid |
| `--gpu H100` | NVIDIA H100 | Latest-gen, paid |

---

## Key Commands

```bash
# Session lifecycle
colab new -s kronos --gpu T4     # allocate GPU VM
colab sessions                    # list active sessions
colab status -s kronos            # hardware + status
colab stop -s kronos              # terminate and release

# Execute local scripts on remote VM
colab exec -s kronos -f train_tokenizer.py

# File transfer
colab upload -s kronos local/path /content/remote/path
colab download -s kronos /content/remote/path local/path

# Install packages on VM
colab install -s kronos -r requirements.txt
colab install -s kronos torch transformers

# One-shot (provision → run → auto-stop)
colab run --gpu T4 script.py
colab run --gpu T4 script.py --keep   # keep VM alive after run

# Logs
colab log -s kronos              # view execution history
colab log -s kronos -o run.ipynb # export as notebook
```

---

## Kronos finetune_tw Training Workflow

### Option A: Step-by-step (reusable session)

```bash
# 1. Provision VM
colab new -s kronos-trainer --gpu T4

# 2. Install dependencies
colab install -s kronos-trainer -r requirements.txt
colab install -s kronos-trainer yfinance pyyaml tqdm

# 3. Create remote directory structure
echo "import os; os.makedirs('finetune_tw/data', exist_ok=True); os.makedirs('finetune_tw/configs', exist_ok=True)" | colab exec -s kronos-trainer

# 4. Upload data and config
colab upload -s kronos-trainer finetune_tw/data/tw_stocks.db /content/finetune_tw/data/tw_stocks.db
colab upload -s kronos-trainer finetune_tw/configs/config_tw_daily.yaml /content/finetune_tw/configs/config_tw_daily.yaml

# 5. Upload training scripts
colab upload -s kronos-trainer finetune_tw/train_tokenizer.py /content/finetune_tw/train_tokenizer.py
colab upload -s kronos-trainer finetune_tw/train_predictor.py /content/finetune_tw/train_predictor.py

# 6. Run training
colab exec -s kronos-trainer -f finetune_tw/train_tokenizer.py
colab exec -s kronos-trainer -f finetune_tw/train_predictor.py

# 7. Download outputs
colab download -s kronos-trainer /content/finetune_tw/outputs ./finetune_tw/outputs

# 8. Clean up
colab stop -s kronos-trainer
```

### Option B: One-shot with `colab run` (auto-teardown)

Create `finetune_tw/colab_train.py` (a self-contained runner script), then:

```bash
colab run --gpu T4 finetune_tw/colab_train.py
```

---

## Agent Execution Limitations

These commands require human interaction and **cannot be run autonomously**:

- `colab auth` — requires user to click a URL and paste a code (interactive `input()`)
- `colab drivemount` — requires user to press Enter after browser OAuth
- `colab repl` / `colab console` — require a real TTY

Piped variants of `repl` and `console` work headlessly:
```bash
echo "import torch; print(torch.cuda.is_available())" | colab exec -s kronos
```

---

## Troubleshooting

```bash
# Check active credentials / scopes
colab whoami

# If auth fails, delete cached token and re-auth
rm ~/.config/colab-cli/token.json
colab sessions  # re-triggers OAuth

# Check for orphaned sessions (billed VMs)
colab sessions
```

---

## Usage in This Session

When the user invokes `/colab` or asks to run training on Colab:

1. Check if `colab` is installed: `colab version`
2. Check for active sessions: `colab sessions`
3. If data is already downloaded locally (`finetune_tw/data/tw_stocks.db` exists), upload it — don't re-download on Colab
4. Prefer `colab run --gpu T4` for one-shot jobs; prefer named sessions for multi-step workflows
5. Always `colab stop` when done to avoid billing orphaned VMs
6. Save training outputs locally via `colab download` before stopping
