# RTX Pro 6000 Training Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `finetune_tw/` tokenizer and predictor wall-clock training time on a MoLab host with 4 CPU cores, 32 GB RAM, and one RTX Pro 6000 96 GB GPU, focusing first on high-confidence changes that do not depend on local non-MoLab profiling.

**Architecture:** Keep the existing single-GPU `finetune_tw` workflow, add explicit per-epoch step caps, add bf16+TF32 execution for tokenizer training, and add a token-cache path so predictor training reuses frozen-tokenizer outputs instead of recomputing them every batch. DataLoader and dataset changes stay conservative by default and become follow-up work only if profiling on the actual MoLab machine proves they are a bottleneck.

**Tech Stack:** Python 3.12, PyTorch 2.12 CUDA 13, pandas, numpy, sqlite3 (stdlib), PyYAML, pytest

---

## Scope

- Target hardware: RTX Pro 6000 (96 GB VRAM), single GPU only
- Target host constraints: 4 CPU cores, 32 GB RAM
- Keep current CLI entrypoints:
  - `python -m finetune_tw.train_tokenizer --config finetune_tw/configs/config_tw_daily_rtx6000.yaml`
  - `python -m finetune_tw.train_predictor --config finetune_tw/configs/config_tw_daily_rtx6000.yaml`
- Preserve current model choices and checkpoint layout under `finetune_tw/outputs/<exp_name>/`
- Do not introduce DDP or `torchrun` in this iteration
- Do not treat local non-MoLab timing numbers as acceptance criteria for DataLoader/dataset changes

---

## Priorities

1. Limit total optimizer steps per epoch
2. Enable tokenizer bf16 + TF32
3. Remove repeated predictor `tokenizer.encode()` work via token cache
4. Profile on MoLab before changing DataLoader/dataset internals

---

## File Map

| File | Responsibility | Action |
| --- | --- | --- |
| `finetune_tw/config.py` | Add runtime knobs for capped epochs, TF32, and token cache | Modify |
| `finetune_tw/configs/config_tw_daily_rtx6000.yaml` | Tune defaults for MoLab RTX Pro 6000 host | Modify |
| `finetune_tw/train_tokenizer.py` | Add TF32/bf16 path and step caps | Modify |
| `finetune_tw/train_predictor.py` | Add TF32 path, token-cache build/load flow, and step caps | Modify |
| `tests/finetune_tw/test_train_predictor.py` | Add token-cache helper tests and runtime option tests | Modify |
| `tests/finetune_tw/test_train_tokenizer.py` | Add tokenizer runtime helper tests | Create |
| `finetune_tw/dataset.py` | Optional compact-index refactor if MoLab profiling shows startup/loader bottleneck | Modify (Optional) |
| `tests/finetune_tw/test_dataset.py` | Optional coverage for compact sample indexing behavior | Modify (Optional) |

---

### Task 1: Add MoLab runtime config and conservative RTX defaults

**Files:**
- Modify: `finetune_tw/config.py`
- Modify: `finetune_tw/configs/config_tw_daily_rtx6000.yaml`
- Modify: `tests/finetune_tw/test_train_predictor.py`

- [ ] **Step 1: Add failing tests for new runtime config fields**

```python
# tests/finetune_tw/test_train_predictor.py
from finetune_tw.config import Config


def test_config_accepts_training_control_fields():
    cfg = Config(
        train_steps_per_epoch=1000,
        val_steps_per_epoch=200,
        persistent_workers=True,
        prefetch_factor=2,
        enable_tf32=True,
        token_cache_enabled=True,
    )
    assert cfg.train_steps_per_epoch == 1000
    assert cfg.val_steps_per_epoch == 200
    assert cfg.persistent_workers is True
    assert cfg.prefetch_factor == 2
    assert cfg.enable_tf32 is True
    assert cfg.token_cache_enabled is True
```

- [ ] **Step 2: Run tests and confirm they fail**

Run: `pytest tests/finetune_tw/test_train_predictor.py -v`

Expected: FAIL because `Config` does not yet expose these fields.

- [ ] **Step 3: Extend `Config` with runtime controls**

```python
# finetune_tw/config.py
from dataclasses import dataclass
import yaml


@dataclass
class Config:
    # Data
    db_path: str = "finetune_tw/data/tw_stocks.db"
    lookback_window: int = 90
    predict_window: int = 10
    max_context: int = 512
    clip: float = 5.0
    train_end_date: str = "2023-12-31"
    val_end_date: str = "2024-06-30"

    # Training
    tokenizer_epochs: int = 30
    basemodel_epochs: int = 20
    batch_size: int = 16
    save_steps: int = 500
    log_interval: int = 50
    tokenizer_lr: float = 2e-4
    predictor_lr: float = 4e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 0.1
    num_workers: int = 2
    persistent_workers: bool = False
    prefetch_factor: int = 2
    train_steps_per_epoch: int = 0
    val_steps_per_epoch: int = 0
    amp_dtype: str = "bf16"
    enable_tf32: bool = True
    token_cache_enabled: bool = False
    token_cache_dtype: str = "uint16"
    seed: int = 42

    # Model paths
    pretrained_tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    pretrained_predictor: str = "NeoQuasar/Kronos-base"
    exp_name: str = "tw_daily"
    output_dir: str = "finetune_tw/outputs"

    # Backtest
    top_k: int = 20
    hold_days: int = 5
    pred_len: int = 10
    test_start_date: str = "2024-07-01"
    benchmark_symbol: str = "^TWII"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
```

- [ ] **Step 4: Tune RTX Pro 6000 config for a 4-core host**

```yaml
# finetune_tw/configs/config_tw_daily_rtx6000.yaml
db_path: "/home/james/kronos_data/tw_stocks.db"
lookback_window: 90
predict_window: 10
max_context: 512
clip: 5.0
train_end_date: "2023-12-31"
val_end_date: "2024-06-30"

tokenizer_epochs: 12
basemodel_epochs: 10
batch_size: 384
save_steps: 2000
log_interval: 50
tokenizer_lr: 0.0002
predictor_lr: 0.00004
adam_beta1: 0.9
adam_beta2: 0.95
adam_weight_decay: 0.1
num_workers: 4
persistent_workers: true
prefetch_factor: 2
train_steps_per_epoch: 1000
val_steps_per_epoch: 200
amp_dtype: "bf16"
enable_tf32: true
token_cache_enabled: true
token_cache_dtype: "uint16"
seed: 42

pretrained_tokenizer: "NeoQuasar/Kronos-Tokenizer-base"
pretrained_predictor: "NeoQuasar/Kronos-base"
exp_name: "tw_daily"
output_dir: "finetune_tw/outputs"

top_k: 20
hold_days: 5
pred_len: 10
test_start_date: "2024-07-01"
benchmark_symbol: "^TWII"
```

- [ ] **Step 5: Re-run tests**

Run: `pytest tests/finetune_tw/test_train_predictor.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add finetune_tw/config.py finetune_tw/configs/config_tw_daily_rtx6000.yaml tests/finetune_tw/test_train_predictor.py
git commit -m "feat(finetune_tw): add molab rtx runtime controls"
```

---

### Task 2: Add tokenizer bf16/TF32 runtime and capped epochs

**Files:**
- Modify: `finetune_tw/train_tokenizer.py`
- Create: `tests/finetune_tw/test_train_tokenizer.py`

- [ ] **Step 1: Write failing helper tests**

```python
# tests/finetune_tw/test_train_tokenizer.py
import torch

from finetune_tw.train_tokenizer import _resolve_runtime_flags, _steps_for_epoch


def test_resolve_runtime_flags_enables_bf16_tf32():
    flags = _resolve_runtime_flags("bf16", enable_tf32=True)
    assert flags["amp_enabled"] is True
    assert flags["amp_dtype"] == torch.bfloat16
    assert flags["enable_tf32"] is True


def test_resolve_runtime_flags_disables_amp_for_none():
    flags = _resolve_runtime_flags("none", enable_tf32=False)
    assert flags["amp_enabled"] is False
    assert flags["amp_dtype"] is None
    assert flags["enable_tf32"] is False


def test_steps_for_epoch_uses_full_loader_when_cap_is_zero():
    assert _steps_for_epoch(loader_len=300, step_cap=0) == 300


def test_steps_for_epoch_respects_config_cap():
    assert _steps_for_epoch(loader_len=300, step_cap=120) == 120
```

- [ ] **Step 2: Run tests and confirm they fail**

Run: `pytest tests/finetune_tw/test_train_tokenizer.py -v`

Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Add runtime helper functions**

```python
# finetune_tw/train_tokenizer.py
def _resolve_runtime_flags(amp_dtype: str, enable_tf32: bool) -> dict[str, object]:
    if amp_dtype == "bf16":
        dtype = torch.bfloat16
    elif amp_dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = None
    return {
        "amp_enabled": dtype is not None,
        "amp_dtype": dtype,
        "enable_tf32": enable_tf32,
    }


def _steps_for_epoch(loader_len: int, step_cap: int) -> int:
    return min(loader_len, step_cap) if step_cap > 0 else loader_len
```

- [ ] **Step 4: Enable TF32 and bf16 in tokenizer training**

```python
# finetune_tw/train_tokenizer.py inside run_training()
runtime = _resolve_runtime_flags(cfg.amp_dtype, cfg.enable_tf32)
if device.type == "cuda" and runtime["enable_tf32"]:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

```python
# finetune_tw/train_tokenizer.py inside epoch loop
steps_this_epoch = _steps_for_epoch(len(train_loader), cfg.train_steps_per_epoch)
for step_idx, (batch_x, _) in enumerate(train_loader):
    if step_idx >= steps_this_epoch:
        break

    batch_x = batch_x.to(device, non_blocking=True)
    with torch.autocast(
        device_type=device.type,
        dtype=runtime["amp_dtype"],
        enabled=bool(runtime["amp_enabled"] and device.type == "cuda"),
    ):
        (z_pre, z), bsq_loss, _, _ = tokenizer(batch_x)
        recon_loss = F.mse_loss(z_pre, batch_x) + F.mse_loss(z, batch_x)
        loss = (recon_loss + bsq_loss) / 2

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 3.0)
    optimizer.step()
    scheduler.step()
```

- [ ] **Step 5: Cap validation steps too**

```python
# finetune_tw/train_tokenizer.py
val_steps = _steps_for_epoch(len(val_loader), cfg.val_steps_per_epoch)
```

Use the same bounded-loop pattern in `_validate()`.

- [ ] **Step 6: Re-run tests**

Run: `pytest tests/finetune_tw/test_train_tokenizer.py -v`

Expected: PASS

- [ ] **Step 7: Smoke-check imports**

Run: `python3 -c "from finetune_tw.train_tokenizer import _resolve_runtime_flags, _steps_for_epoch; print(_resolve_runtime_flags('bf16', True)); print(_steps_for_epoch(300, 120))"`

Expected:
- First line prints a dict with `torch.bfloat16`
- Second line prints `120`

- [ ] **Step 8: Commit**

```bash
git add finetune_tw/train_tokenizer.py tests/finetune_tw/test_train_tokenizer.py
git commit -m "perf(finetune_tw): add tokenizer bf16 tf32 and capped epochs"
```

---

### Task 3: Add predictor token-cache build and load path

**Files:**
- Modify: `finetune_tw/train_predictor.py`
- Modify: `tests/finetune_tw/test_train_predictor.py`

- [ ] **Step 1: Add failing token-cache helper tests**

```python
# tests/finetune_tw/test_train_predictor.py
from pathlib import Path

from finetune_tw.train_predictor import _token_cache_paths


def test_token_cache_paths_are_split_specific(tmp_path):
    path = _token_cache_paths(Path(tmp_path), "train")
    assert path["data"].name == "train_token_cache.pt"
    assert path["meta"].name == "train_token_cache_meta.json"
```

- [ ] **Step 2: Run tests and confirm they fail**

Run: `pytest tests/finetune_tw/test_train_predictor.py -v`

Expected: FAIL because `_token_cache_paths` does not exist.

- [ ] **Step 3: Add token-cache dataset and path helper**

```python
# finetune_tw/train_predictor.py
from torch.utils.data import DataLoader, Dataset


class CachedTokenDataset(Dataset):
    def __init__(self, cache_file: Path) -> None:
        payload = torch.load(cache_file, map_location="cpu", weights_only=True)
        self.token_s1 = payload["token_s1"]
        self.token_s2 = payload["token_s2"]
        self.stamps = payload["stamps"]

    def __len__(self) -> int:
        return self.token_s1.shape[0]

    def __getitem__(self, idx: int):
        return self.token_s1[idx], self.token_s2[idx], self.stamps[idx]


def _token_cache_paths(cache_dir: Path, split: str) -> dict[str, Path]:
    return {
        "data": cache_dir / f"{split}_token_cache.pt",
        "meta": cache_dir / f"{split}_token_cache_meta.json",
    }
```

- [ ] **Step 4: Build cache once from the frozen tokenizer**

```python
# finetune_tw/train_predictor.py
def _build_token_cache(dataset, tokenizer, device, cache_dir: Path, split: str, batch_size: int) -> Path:
    paths = _token_cache_paths(cache_dir, split)
    if paths["data"].exists() and paths["meta"].exists():
        return paths["data"]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    token_s1_parts, token_s2_parts, stamp_parts = [], [], []

    tokenizer.eval()
    with torch.no_grad():
        for batch_x, batch_x_stamp in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            token_s1, token_s2 = tokenizer.encode(batch_x, half=True)
            token_s1_parts.append(token_s1.cpu().to(torch.int32))
            token_s2_parts.append(token_s2.cpu().to(torch.int32))
            stamp_parts.append(batch_x_stamp.to(torch.float32))

    payload = {
        "token_s1": torch.cat(token_s1_parts, dim=0),
        "token_s2": torch.cat(token_s2_parts, dim=0),
        "stamps": torch.cat(stamp_parts, dim=0),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, paths["data"])
    paths["meta"].write_text(json.dumps({"split": split, "rows": int(payload["token_s1"].shape[0])}, indent=2))
    return paths["data"]
```

- [ ] **Step 5: Re-run tests**

Run: `pytest tests/finetune_tw/test_train_predictor.py -v`

Expected: PASS

- [ ] **Step 6: Smoke-check cache helper**

Run: `python3 -c "from pathlib import Path; from finetune_tw.train_predictor import _token_cache_paths; print(_token_cache_paths(Path('tmp'), 'train')['data'])"`

Expected: prints `tmp/train_token_cache.pt`

- [ ] **Step 7: Commit**

```bash
git add finetune_tw/train_predictor.py tests/finetune_tw/test_train_predictor.py
git commit -m "perf(finetune_tw): add predictor token cache helpers"
```

---

### Task 4: Switch predictor training to cached tokens, TF32, and capped steps

**Files:**
- Modify: `finetune_tw/train_predictor.py`
- Modify: `tests/finetune_tw/test_train_predictor.py`

- [ ] **Step 1: Add failing tests for predictor step-cap helper**

```python
# tests/finetune_tw/test_train_predictor.py
from finetune_tw.train_predictor import _steps_for_epoch


def test_predictor_steps_for_epoch_uses_cap():
    assert _steps_for_epoch(500, 120) == 120
```

- [ ] **Step 2: Run tests and confirm they fail**

Run: `pytest tests/finetune_tw/test_train_predictor.py -v`

Expected: FAIL because `_steps_for_epoch` does not exist in predictor.

- [ ] **Step 3: Add predictor runtime helpers**

```python
# finetune_tw/train_predictor.py
def _steps_for_epoch(loader_len: int, step_cap: int) -> int:
    return min(loader_len, step_cap) if step_cap > 0 else loader_len


def _configure_cuda_runtime(device: torch.device, enable_tf32: bool) -> None:
    if device.type != "cuda" or not enable_tf32:
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

- [ ] **Step 4: Build or load token caches in `run_training()`**

```python
# finetune_tw/train_predictor.py inside run_training()
_configure_cuda_runtime(device, cfg.enable_tf32)
cache_dir = Path(cfg.output_dir) / cfg.exp_name / "token_cache"

if cfg.token_cache_enabled:
    train_cache = _build_token_cache(train_ds, tokenizer, device, cache_dir, "train", cfg.batch_size)
    val_cache = _build_token_cache(val_ds, tokenizer, device, cache_dir, "val", cfg.batch_size)
    train_loader = DataLoader(
        CachedTokenDataset(train_cache),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        CachedTokenDataset(val_cache),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
```

- [ ] **Step 5: Skip tokenizer encode in cached mode and cap loops**

```python
# finetune_tw/train_predictor.py inside epoch loop
steps_this_epoch = _steps_for_epoch(len(train_loader), cfg.train_steps_per_epoch)
for step_idx, batch in enumerate(train_loader):
    if step_idx >= steps_this_epoch:
        break

    if cfg.token_cache_enabled:
        token_s1, token_s2, batch_x_stamp = batch
        token_s1 = token_s1.to(device, non_blocking=True)
        token_s2 = token_s2.to(device, non_blocking=True)
        batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
    else:
        batch_x, batch_x_stamp = batch
        batch_x = batch_x.to(device, non_blocking=True)
        batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
        with torch.no_grad():
            token_s1, token_s2 = tokenizer.encode(batch_x, half=True)

    token_in = [token_s1[:, :-1], token_s2[:, :-1]]
    token_out = [token_s1[:, 1:], token_s2[:, 1:]]

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
        loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

    optimizer.zero_grad(set_to_none=True)
```

Apply the same branching and step cap to validation.

- [ ] **Step 6: Re-run targeted tests**

Run: `pytest tests/finetune_tw/test_train_predictor.py -v`

Expected: PASS

- [ ] **Step 7: Run non-GPU regression suite**

Run: `pytest tests/finetune_tw -v -k "not tokenizer_train"`

Expected: PASS with GPU smoke tests skipped on CPU-only machines.

- [ ] **Step 8: Commit**

```bash
git add finetune_tw/train_predictor.py tests/finetune_tw/test_train_predictor.py
git commit -m "perf(finetune_tw): use cached tokens and capped predictor epochs"
```

---

### Task 5: Optional DataLoader/dataset work gated by MoLab profiling

**Files:**
- Modify: `finetune_tw/dataset.py` only if MoLab profiling shows startup or worker starvation is materially limiting throughput
- Modify: `tests/finetune_tw/test_dataset.py` only if the dataset implementation changes

- [ ] **Step 1: Capture on-host profiling before changing dataset internals**

Run these on the MoLab machine during a short smoke run:

```bash
nvidia-smi dmon -s pucm
```

```bash
python -m finetune_tw.train_tokenizer --config finetune_tw/configs/config_tw_daily_rtx6000.yaml
```

```bash
python -m finetune_tw.train_predictor --config finetune_tw/configs/config_tw_daily_rtx6000.yaml
```

Record:
- GPU utilization
- GPU memory
- CPU utilization
- step/sec
- time spent before first batch

- [ ] **Step 2: Only continue if profiling shows loader/dataset bottleneck**

Continue to dataset changes only if at least one of these is true:
- GPU utilization is persistently low while CPU utilization is high
- first-batch latency is operationally significant for your workflow
- predictor token cache is already enabled and step/sec is still CPU-bound

If none are true:
- stop here
- keep the current dataset implementation

- [ ] **Step 3: If needed, add compact sample metadata indexing**

```python
# finetune_tw/dataset.py
# Replace fully materialized _samples with:
# - _meta: per-symbol row count and cumulative offsets
# - _resolve_index(idx): map global sample index to (symbol, start)
```

- [ ] **Step 4: Add or update dataset tests if implementation changes**

Run: `pytest tests/finetune_tw/test_dataset.py -v`

Expected: PASS

- [ ] **Step 5: Commit only if this task was actually needed**

```bash
git add finetune_tw/dataset.py tests/finetune_tw/test_dataset.py
git commit -m "perf(finetune_tw): optimize dataset indexing after molab profiling"
```

---

### Task 6: Validate the RTX Pro 6000 path end-to-end

**Files:**
- Modify: `finetune_tw/configs/config_tw_daily_rtx6000.yaml` only if runtime tuning from validation requires it

- [ ] **Step 1: Verify tokenizer launch uses capped epochs**

Run: `python -m finetune_tw.train_tokenizer --config finetune_tw/configs/config_tw_daily_rtx6000.yaml`

Expected:
- Logs show bf16 path active
- First epoch stops after 1000 train steps
- Validation stops after 200 steps

- [ ] **Step 2: Verify predictor cache build and reuse**

Run: `python -m finetune_tw.train_predictor --config finetune_tw/configs/config_tw_daily_rtx6000.yaml`

Expected:
- First run creates `finetune_tw/outputs/tw_daily/token_cache/train_token_cache.pt`
- Subsequent run reuses cache instead of rebuilding
- Predictor epoch stops after configured train/val step caps

- [ ] **Step 3: Compare throughput before/after on MoLab**

Run:
- Baseline branch: record `steps/sec` for 200 steps
- Optimized branch: record `steps/sec` for 200 steps

Expected:
- Tokenizer wall-clock epoch time materially lower due to bf16 + capped loops
- Predictor wall-clock epoch time materially lower due to cache reuse + capped loops

- [ ] **Step 4: Tune final RTX knobs conservatively for the 4-core host**

If GPU memory remains comfortably below 96 GB, adjust:

```yaml
batch_size: 512
num_workers: 4
prefetch_factor: 2
```

If host RAM or loader stability becomes a problem, revert to:

```yaml
batch_size: 384
num_workers: 2
prefetch_factor: 2
```

- [ ] **Step 5: Commit final config adjustments**

```bash
git add finetune_tw/configs/config_tw_daily_rtx6000.yaml
git commit -m "chore(finetune_tw): tune final molab rtx defaults"
```

---

## Self-Review

- Spec coverage:
  - Step-count reduction: Task 1, Task 2, Task 4
  - RTX bf16/TF32 for tokenizer: Task 2
  - Predictor token cache: Task 3, Task 4
  - DataLoader/dataset work is explicitly profiling-gated: Task 5
  - MoLab-specific config tuning: Task 1, Task 6
- Placeholder scan: no `TODO`, `TBD`, or unresolved implementation markers remain
- Type consistency:
  - `train_steps_per_epoch`, `val_steps_per_epoch`, `persistent_workers`, `prefetch_factor`, `enable_tf32`, `token_cache_enabled`, `token_cache_dtype` are used consistently across config and runtime tasks
