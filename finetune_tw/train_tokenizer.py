"""
python finetune_tw/train_tokenizer.py --config finetune_tw/configs/config_tw_daily.yaml
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import KronosTokenizer
from finetune_tw.config import Config
from finetune_tw.dataset import MultiStockDataset


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


def run_training(cfg: Config, max_steps: int = -1) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    save_dir = Path(cfg.output_dir) / cfg.exp_name / "tokenizer"
    ckpt_dir = save_dir / "checkpoints"
    remote_root = f"gdrive:Kronos/outputs/{cfg.exp_name}/tokenizer"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = KronosTokenizer.from_pretrained(cfg.pretrained_tokenizer).to(device)

    train_ds = MultiStockDataset(cfg.db_path, cfg.lookback_window, cfg.predict_window,
                                 "2015-01-01", cfg.train_end_date, cfg.clip, cfg.seed)
    val_ds   = MultiStockDataset(cfg.db_path, cfg.lookback_window, cfg.predict_window,
                                 cfg.train_end_date, cfg.val_end_date, cfg.clip, cfg.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=cfg.tokenizer_lr,
                                  betas=(cfg.adam_beta1, cfg.adam_beta2),
                                  weight_decay=cfg.adam_weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.tokenizer_lr,
        steps_per_epoch=len(train_loader), epochs=cfg.tokenizer_epochs,
        pct_start=0.03, div_factor=10,
    )
    # Resume from latest checkpoint
    _gdrive_restore_checkpoints(ckpt_dir, f"{remote_root}/checkpoints")
    start_epoch, global_step = _load_latest_checkpoint(ckpt_dir, tokenizer, optimizer, scheduler)
    best_val_loss = float("inf")
    log_path = save_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,step,train_loss,val_loss\n")

    for epoch in range(start_epoch, cfg.tokenizer_epochs):
        tokenizer.train()
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            (z_pre, z), bsq_loss, _, _ = tokenizer(batch_x)
            recon_loss = F.mse_loss(z_pre, batch_x) + F.mse_loss(z, batch_x)
            loss = (recon_loss + bsq_loss) / 2
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 3.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % cfg.log_interval == 0:
                print(f"[epoch {epoch+1} step {global_step}] loss={loss.item():.4f}")

            if global_step % cfg.save_steps == 0:
                _save_checkpoint(ckpt_dir, global_step, epoch, tokenizer, optimizer, scheduler)
                _gdrive_sync_checkpoint(ckpt_dir / f"ckpt-{global_step}.pt", f"{remote_root}/checkpoints")

            if max_steps > 0 and global_step >= max_steps:
                return

        val_loss = _validate(tokenizer, val_loader, device)
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{global_step},{loss.item():.4f},{val_loss:.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            tokenizer.save_pretrained(str(save_dir / "best_model"))
            print(f"  -> new best val_loss={val_loss:.4f}, saved.")
            _gdrive_sync(save_dir / "best_model", remote=remote_root)


def _validate(tokenizer: KronosTokenizer, loader: DataLoader, device: torch.device) -> float:
    tokenizer.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            (_, z), _, _, _ = tokenizer(batch_x)
            total += F.mse_loss(z, batch_x).item() * batch_x.size(0)
            count += batch_x.size(0)
    return total / count if count else 0.0


def _save_checkpoint(ckpt_dir: Path, step: int, epoch: int, model, optimizer, scheduler) -> None:
    torch.save({
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, ckpt_dir / f"ckpt-{step}.pt")


def _load_latest_checkpoint(ckpt_dir: Path, model, optimizer, scheduler):
    ckpts = sorted(ckpt_dir.glob("ckpt-*.pt"),
                   key=lambda p: int(p.stem.split("-")[1]))
    if not ckpts:
        return 0, 0
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    print(f"Resumed from {ckpts[-1].name} (step {state['step']})")
    return state.get("epoch", 0), state["step"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    cfg = Config.from_yaml(parser.parse_args().config)
    run_training(cfg)


if __name__ == "__main__":
    main()
