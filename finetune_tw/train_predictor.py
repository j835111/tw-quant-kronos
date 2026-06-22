"""
python finetune_tw/train_predictor.py --config finetune_tw/configs/config_tw_daily.yaml
Requires: tokenizer best_model saved by train_tokenizer.py
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

from model import Kronos, KronosTokenizer, KronosPredictor
from finetune_tw.config import Config
from finetune_tw.dataset import MultiStockDataset
from finetune_tw.db import list_symbols, query_symbol
from finetune_tw.ic_validation import (
    EarlyStopper,
    pick_val_dates,
    pick_val_universe,
    validate_predictor_ic,
    validate_predictor_ic_ir,
)
from finetune_tw.hf_utils import push_best_model, push_file, wait_for_pushes
from finetune_tw.train_tokenizer import _load_latest_checkpoint, _save_checkpoint


def _gdrive_sync(local_dir: Path, remote: str = "gdrive:Kronos/outputs") -> None:
    """Sync local_dir to Google Drive in background. Silently skips if rclone not found."""
    if shutil.which("rclone") is None:
        return
    rel = local_dir.name
    subprocess.Popen(
        ["rclone", "sync", str(local_dir), f"{remote}/{rel}", "--transfers=4"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"  [gdrive] sync started (background) → {remote}/{rel}")


def _gdrive_sync_logs(log_path: Path, remote: str) -> None:
    """Upload train_log.csv to Drive (fixes the lost-log gap)."""
    if shutil.which("rclone") is None or not log_path.exists():
        return
    subprocess.Popen(
        ["rclone", "copy", str(log_path), remote],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _gdrive_restore_checkpoints(ckpt_dir: Path, remote: str) -> None:
    """啟動時，若本地無 ckpt，從 Drive 拉回。"""
    if shutil.which("rclone") is None or list(ckpt_dir.glob("ckpt-*.pt")):
        return
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["rclone", "copy", remote, str(ckpt_dir), "--transfers=4"],
        capture_output=True, text=True, timeout=300,
    )
    n = len(list(ckpt_dir.glob("ckpt-*.pt")))
    if n:
        print(f"  [gdrive] restored {n} checkpoint(s) from {remote}")


def _gdrive_sync_checkpoint(ckpt_path: Path, remote_ckpt_dir: str) -> None:
    """每次存 checkpoint 後，背景上傳到 Drive（不阻塞訓練）。"""
    if shutil.which("rclone") is None:
        return
    subprocess.Popen(
        ["rclone", "copy", str(ckpt_path), remote_ckpt_dir, "--transfers=4"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _resolve_amp(amp_dtype: str) -> tuple[bool, "torch.dtype | None"]:
    """Map config amp_dtype to (autocast_enabled, dtype). Supports bf16 and fp16."""
    if amp_dtype == "bf16":
        return True, torch.bfloat16
    if amp_dtype == "fp16":
        return True, torch.float16
    return False, None


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


def _token_cache_storage_dtype(cache_dtype: str) -> torch.dtype:
    if cache_dtype == "uint16":
        return torch.uint16
    if cache_dtype == "int32":
        return torch.int32
    raise ValueError(f"Unsupported token cache dtype: {cache_dtype}")


def _build_token_cache(
    dataset,
    tokenizer,
    device,
    cache_dir: Path,
    split: str,
    batch_size: int,
    cache_dtype: str = "uint16",
) -> Path:
    paths = _token_cache_paths(cache_dir, split)
    if paths["data"].exists() and paths["meta"].exists():
        return paths["data"]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    storage_dtype = _token_cache_storage_dtype(cache_dtype)
    token_s1_parts, token_s2_parts, stamp_parts = [], [], []

    tokenizer.eval()
    with torch.no_grad():
        for batch_x, batch_x_stamp in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            token_s1, token_s2 = tokenizer.encode(batch_x, half=True)
            token_s1_parts.append(token_s1.cpu().to(storage_dtype))
            token_s2_parts.append(token_s2.cpu().to(storage_dtype))
            stamp_parts.append(batch_x_stamp.to(torch.float32))

    payload = {
        "token_s1": torch.cat(token_s1_parts, dim=0),
        "token_s2": torch.cat(token_s2_parts, dim=0),
        "stamps": torch.cat(stamp_parts, dim=0),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, paths["data"])
    paths["meta"].write_text(
        json.dumps(
            {
                "split": split,
                "rows": int(payload["token_s1"].shape[0]),
                "token_cache_dtype": cache_dtype,
            },
            indent=2,
        )
    )
    return paths["data"]


def _steps_for_epoch(loader_len: int, step_cap: int) -> int:
    return min(loader_len, step_cap) if step_cap > 0 else loader_len


def _configure_cuda_runtime(device: torch.device, enable_tf32: bool) -> None:
    if device.type != "cuda" or not enable_tf32:
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _build_ctx_for_date(cfg, sym, rebal_date):
    rebal_str = rebal_date.strftime("%Y-%m-%d")
    lookback_start = (rebal_date - pd.Timedelta(days=cfg.lookback_window * 2)).strftime("%Y-%m-%d")
    df = query_symbol(cfg.db_path, sym, start=lookback_start, end=rebal_str)
    if len(df) < cfg.lookback_window:
        return None
    ctx = df.iloc[-cfg.lookback_window:]
    ctx_df = ctx[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    if ctx_df.isnull().any().any():
        return None
    x_ts = pd.to_datetime(ctx["date"]).reset_index(drop=True)
    y_ts = pd.Series(pd.date_range(rebal_date, periods=cfg.pred_len, freq="B"))
    return ctx_df, x_ts, y_ts, x_ts.iloc[-1], float(ctx_df["close"].iloc[-1])


def _actual_close_lookup(cfg, cache, sym, ctx_last_date, n):
    ser = cache.get(sym)
    if ser is None:
        return np.array([], dtype=float)
    pos = ser.index.searchsorted(ctx_last_date, side="right")
    return ser.iloc[pos:pos + n].values.astype(float)


def _make_predict_batch_fn(predictor):
    def fn(df_list, x_timestamp_list, y_timestamp_list, pred_len):
        with torch.no_grad():
            return predictor.predict_batch(
                df_list=df_list,
                x_timestamp_list=x_timestamp_list,
                y_timestamp_list=y_timestamp_list,
                pred_len=pred_len,
                T=1.0,
                top_k=1,
                top_p=1.0,
                sample_count=1,
                verbose=False,
            )

    return fn


def run_training(cfg: Config, max_steps: int = -1) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    _configure_cuda_runtime(device, cfg.enable_tf32)

    tok_path = Path(cfg.output_dir) / cfg.exp_name / "tokenizer" / "best_model"
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}. Run train_tokenizer.py first.")

    tokenizer = KronosTokenizer.from_pretrained(str(tok_path)).to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    if cfg.hf_revision:
        from huggingface_hub import snapshot_download
        _snap = snapshot_download(
            cfg.pretrained_predictor,
            revision=cfg.hf_revision,
            allow_patterns=["predictor/best_model/*"],
            token=os.environ.get("HF_TOKEN"),
            local_files_only=False,
        )
        _pred_local = f"{_snap}/predictor/best_model"
        model = Kronos.from_pretrained(_pred_local).to(device)
        print(f"  Loaded predictor from {cfg.pretrained_predictor}@{cfg.hf_revision}/predictor/best_model")
    else:
        model = Kronos.from_pretrained(cfg.pretrained_predictor).to(device)
    amp_enabled, amp_dtype = _resolve_amp(cfg.amp_dtype)
    amp_enabled = amp_enabled and device.type == "cuda"
    scaler = GradScaler() if (amp_enabled and amp_dtype == torch.float16) else None

    save_dir = Path(cfg.output_dir) / cfg.exp_name / "predictor"
    ckpt_dir = save_dir / "checkpoints"
    remote_root = f"gdrive:Kronos/outputs/{cfg.exp_name}/predictor"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = MultiStockDataset(cfg.db_path, cfg.lookback_window, cfg.predict_window,
                                 "2015-01-01", cfg.train_end_date, cfg.clip, cfg.seed)
    val_ds   = MultiStockDataset(cfg.db_path, cfg.lookback_window, cfg.predict_window,
                                 cfg.train_end_date, cfg.val_end_date, cfg.clip, cfg.seed + 1)

    train_loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    val_loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": True,
    }
    if cfg.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = cfg.persistent_workers
        train_loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        val_loader_kwargs["persistent_workers"] = cfg.persistent_workers
        val_loader_kwargs["prefetch_factor"] = cfg.prefetch_factor

    cache_dir = Path(cfg.output_dir) / cfg.exp_name / "token_cache"
    if cfg.token_cache_enabled:
        train_cache = _build_token_cache(
            train_ds,
            tokenizer,
            device,
            cache_dir,
            "train",
            cfg.batch_size,
            cfg.token_cache_dtype,
        )
        val_cache = _build_token_cache(
            val_ds,
            tokenizer,
            device,
            cache_dir,
            "val",
            cfg.batch_size,
            cfg.token_cache_dtype,
        )
        train_loader = DataLoader(
            CachedTokenDataset(train_cache),
            batch_size=cfg.batch_size,
            shuffle=True,
            **train_loader_kwargs,
        )
        val_loader = DataLoader(
            CachedTokenDataset(val_cache),
            batch_size=cfg.batch_size,
            shuffle=False,
            **val_loader_kwargs,
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  **train_loader_kwargs)
        val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                  **val_loader_kwargs)

    train_steps_per_epoch = _steps_for_epoch(len(train_loader), cfg.train_steps_per_epoch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.predictor_lr,
                                  betas=(cfg.adam_beta1, cfg.adam_beta2),
                                  weight_decay=cfg.adam_weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.predictor_lr,
        steps_per_epoch=train_steps_per_epoch, epochs=cfg.basemodel_epochs,
        pct_start=0.03, div_factor=10,
    )
    _gdrive_restore_checkpoints(ckpt_dir, f"{remote_root}/checkpoints")
    start_epoch, global_step = _load_latest_checkpoint(ckpt_dir, model, optimizer, scheduler)
    log_path = save_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,step,train_loss,val_loss,val_ic,ic_ir_h5\n")

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=cfg.max_context)
    all_syms = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    val_universe = pick_val_universe(all_syms, cfg.ic_val_symbols, cfg.seed)
    val_dates = pick_val_dates(cfg.train_end_date, cfg.val_end_date, cfg.ic_val_dates)
    buffer_start = (pd.Timestamp(cfg.train_end_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    actual_cache = {}
    for sym in val_universe:
        df = query_symbol(
            cfg.db_path,
            sym,
            start=buffer_start,
            end=(pd.Timestamp(cfg.val_end_date) + pd.Timedelta(days=cfg.pred_len * 3)).strftime("%Y-%m-%d"),
        )
        if len(df):
            actual_cache[sym] = pd.Series(df["close"].values, index=pd.DatetimeIndex(df["date"]))
    stopper = EarlyStopper(patience=cfg.early_stop_patience, mode="max")

    for epoch in range(start_epoch, cfg.basemodel_epochs):
        model.train()
        steps_this_epoch = train_steps_per_epoch
        for step_idx, batch in enumerate(train_loader):
            if step_idx >= steps_this_epoch:
                break

            if cfg.token_cache_enabled:
                token_s1, token_s2, batch_x_stamp = batch
                token_s1 = token_s1.to(device=device, dtype=torch.long, non_blocking=True)
                token_s2 = token_s2.to(device=device, dtype=torch.long, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            else:
                batch_x, batch_x_stamp = batch
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
                with torch.no_grad():
                    token_s1, token_s2 = tokenizer.encode(batch_x, half=True)

            token_in  = [token_s1[:, :-1], token_s2[:, :-1]]
            token_out = [token_s1[:, 1:],  token_s2[:, 1:]]

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.bfloat16,
                enabled=amp_enabled,
            ):
                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % cfg.log_interval == 0:
                print(f"[epoch {epoch+1} step {global_step}] loss={loss.item():.4f}")

            if global_step % cfg.save_steps == 0:
                _save_checkpoint(ckpt_dir, global_step, epoch, model, optimizer, scheduler)
                _gdrive_sync_checkpoint(ckpt_dir / f"ckpt-{global_step}.pt", f"{remote_root}/checkpoints")

            if max_steps > 0 and global_step >= max_steps:
                return

        val_loss = _validate_predictor(
            model,
            tokenizer,
            val_loader,
            device,
            amp_enabled,
            amp_dtype,
            cfg.token_cache_enabled,
            cfg.val_steps_per_epoch,
        )
        model.eval()
        predict_fn = _make_predict_batch_fn(predictor)
        actual_fn = lambda sym, last, n: _actual_close_lookup(cfg, actual_cache, sym, last, n)
        ctx_fn = lambda sym, rebal_date: _build_ctx_for_date(cfg, sym, rebal_date)

        val_ic = validate_predictor_ic(predict_fn, actual_fn, val_universe, val_dates, cfg, ctx_fn)
        ic_ir_h5 = validate_predictor_ic_ir(predict_fn, actual_fn, val_universe, val_dates, cfg, ctx_fn,
                                             target_horizon=min(5, cfg.pred_len))

        ic_ir_str = f"{ic_ir_h5:.4f}" if not (ic_ir_h5 != ic_ir_h5) else "nan"
        print(f"  val_loss={val_loss:.4f}  val_ic={val_ic:.4f}  ic_ir_h5={ic_ir_str}")
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{global_step},{loss.item():.4f},{val_loss:.4f},{val_ic:.4f},{ic_ir_str}\n")

        # Use ic_ir_h5 for early stopping; fall back to val_ic if ic_ir_h5 is nan
        stop_metric = ic_ir_h5 if (ic_ir_h5 == ic_ir_h5) else val_ic
        is_best, should_stop = stopper.update(stop_metric)
        if is_best:
            model.save_pretrained(str(save_dir / "best_model"))
            print(f"  -> new best ic_ir_h5={ic_ir_str} val_ic={val_ic:.4f} (epoch {epoch+1}), saved.")
            _gdrive_sync(save_dir / "best_model", remote=remote_root)
            _gdrive_sync_logs(log_path, remote_root)
            if cfg.hf_repo and cfg.hf_revision_out:
                push_best_model(save_dir / "best_model", cfg.hf_repo,
                                "predictor/best_model", cfg.hf_revision_out)
                push_file(log_path, cfg.hf_repo, "predictor/train_log.csv", cfg.hf_revision_out)
        if should_stop:
            print(f"  -> early stop at epoch {epoch+1} (best ic_ir_h5={stopper.best:.4f})")
            break


def _validate_predictor(
    model,
    tokenizer,
    loader,
    device,
    amp_enabled=False,
    amp_dtype=None,
    token_cache_enabled=False,
    step_cap=0,
) -> float:
    model.eval()
    amp_enabled = amp_enabled and device.type == "cuda"
    total, count = 0.0, 0
    val_steps = _steps_for_epoch(len(loader), step_cap)
    with torch.no_grad():
        for step_idx, batch in enumerate(loader):
            if step_idx >= val_steps:
                break

            if token_cache_enabled:
                token_s1, token_s2, batch_x_stamp = batch
                token_s1 = token_s1.to(device=device, dtype=torch.long, non_blocking=True)
                token_s2 = token_s2.to(device=device, dtype=torch.long, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            else:
                batch_x, batch_x_stamp = batch
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
                token_s1, token_s2 = tokenizer.encode(batch_x, half=True)
            token_in  = [token_s1[:, :-1], token_s2[:, :-1]]
            token_out = [token_s1[:, 1:],  token_s2[:, 1:]]
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.bfloat16,
                enabled=amp_enabled,
            ):
                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
            total += loss.item() * batch_x_stamp.size(0)
            count += batch_x_stamp.size(0)
    return total / count if count else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    cfg = Config.from_yaml(parser.parse_args().config)
    run_training(cfg)
    wait_for_pushes()


if __name__ == "__main__":
    main()
