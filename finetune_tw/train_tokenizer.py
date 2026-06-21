"""
python finetune_tw/train_tokenizer.py --config finetune_tw/configs/config_tw_daily.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import KronosTokenizer
from finetune_tw.config import Config
from finetune_tw.dataset import MultiStockDataset
from finetune_tw.hf_utils import push_best_model, restore_best_model, wait_for_pushes


def run_training(cfg: Config, max_steps: int = -1) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    exp_dir  = Path(cfg.output_dir) / cfg.exp_name
    save_dir = exp_dir / "tokenizer"
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    restore_best_model(exp_dir, cfg.hf_repo, "tokenizer/best_model", cfg.hf_revision)

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

            if max_steps > 0 and global_step >= max_steps:
                return

        val_loss = _validate(tokenizer, val_loader, device)
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{global_step},{loss.item():.4f},{val_loss:.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            tokenizer.save_pretrained(str(save_dir / "best_model"))
            print(f"  -> new best val_loss={val_loss:.4f}, saved.")
            push_best_model(save_dir / "best_model", cfg.hf_repo, "tokenizer/best_model", cfg.hf_revision)

    wait_for_pushes()


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
