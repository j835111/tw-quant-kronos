"""
python finetune_tw/train_predictor.py --config finetune_tw/configs/config_tw_daily.yaml
Requires: tokenizer best_model saved by train_tokenizer.py
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from model import Kronos, KronosTokenizer
from finetune_tw.config import Config
from finetune_tw.dataset import MultiStockDataset
from finetune_tw.train_tokenizer import _load_latest_checkpoint, _save_checkpoint


def _gdrive_sync(local_dir: Path, remote: str = "gdrive:Kronos/outputs") -> None:
    """Sync local_dir to Google Drive. Silently skips if rclone not found."""
    if shutil.which("rclone") is None:
        return
    rel = local_dir.name
    r = subprocess.run(
        ["rclone", "sync", str(local_dir), f"{remote}/{rel}", "--transfers=4"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode == 0:
        print(f"  [gdrive] synced → {remote}/{rel}")
    else:
        print(f"  [gdrive] sync failed: {r.stderr[:200]}")


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
    """每次存 checkpoint 後，立即上傳該檔到 Drive。"""
    if shutil.which("rclone") is None:
        return
    subprocess.run(
        ["rclone", "copy", str(ckpt_path), remote_ckpt_dir, "--transfers=4"],
        capture_output=True, text=True, timeout=120,
    )


def _resolve_amp(amp_dtype: str) -> tuple[bool, "torch.dtype | None"]:
    """Map config amp_dtype to (autocast_enabled, dtype). Supports bf16 and fp16."""
    if amp_dtype == "bf16":
        return True, torch.bfloat16
    if amp_dtype == "fp16":
        return True, torch.float16
    return False, None


def run_training(cfg: Config, max_steps: int = -1) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    tok_path = Path(cfg.output_dir) / cfg.exp_name / "tokenizer" / "best_model"
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}. Run train_tokenizer.py first.")

    tokenizer = KronosTokenizer.from_pretrained(str(tok_path)).to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

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

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.predictor_lr,
                                  betas=(cfg.adam_beta1, cfg.adam_beta2),
                                  weight_decay=cfg.adam_weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.predictor_lr,
        steps_per_epoch=len(train_loader), epochs=cfg.basemodel_epochs,
        pct_start=0.03, div_factor=10,
    )
    _gdrive_restore_checkpoints(ckpt_dir, f"{remote_root}/checkpoints")
    start_epoch, global_step = _load_latest_checkpoint(ckpt_dir, model, optimizer, scheduler)
    best_val_loss = float("inf")
    log_path = save_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,step,train_loss,val_loss\n")

    for epoch in range(start_epoch, cfg.basemodel_epochs):
        model.train()
        for batch_x, batch_x_stamp in train_loader:
            batch_x       = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            with torch.no_grad():
                token_s1, token_s2 = tokenizer.encode(batch_x, half=True)

            token_in  = [token_s1[:, :-1], token_s2[:, :-1]]
            token_out = [token_s1[:, 1:],  token_s2[:, 1:]]

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            optimizer.zero_grad()
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

        val_loss = _validate_predictor(model, tokenizer, val_loader, device, amp_enabled, amp_dtype)
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{global_step},{loss.item():.4f},{val_loss:.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(str(save_dir / "best_model"))
            print(f"  -> new best val_loss={val_loss:.4f}, saved.")
            _gdrive_sync(save_dir / "best_model", remote=remote_root)


def _validate_predictor(model, tokenizer, loader, device, amp_enabled=False, amp_dtype=None) -> float:
    model.eval()
    amp_enabled = amp_enabled and device.type == "cuda"
    total, count = 0.0, 0
    with torch.no_grad():
        for batch_x, batch_x_stamp in loader:
            batch_x       = batch_x.to(device)
            batch_x_stamp = batch_x_stamp.to(device)
            token_s1, token_s2 = tokenizer.encode(batch_x, half=True)
            token_in  = [token_s1[:, :-1], token_s2[:, :-1]]
            token_out = [token_s1[:, 1:],  token_s2[:, 1:]]
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
            total += loss.item() * batch_x.size(0)
            count += batch_x.size(0)
    return total / count if count else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    cfg = Config.from_yaml(parser.parse_args().config)
    run_training(cfg)


if __name__ == "__main__":
    main()
