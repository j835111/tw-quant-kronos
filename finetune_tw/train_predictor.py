"""
python finetune_tw/train_predictor.py --config finetune_tw/configs/config_tw_daily.yaml
Requires: tokenizer best_model (local or on HuggingFace Hub).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from model import Kronos, KronosTokenizer, KronosPredictor
from finetune_tw.config import Config
from finetune_tw.dataset import MultiStockDataset
from finetune_tw.db import list_symbols, query_symbol
from finetune_tw.hf_utils import has_weights, push_best_model, restore_best_model, resolve_src, wait_for_pushes
from finetune_tw.ic_validation import (
    EarlyStopper,
    pick_val_dates,
    pick_val_universe,
    validate_predictor_ic,
)
from finetune_tw.train_tokenizer import _load_latest_checkpoint, _save_checkpoint


def _resolve_amp(amp_dtype: str) -> tuple[bool, "torch.dtype | None"]:
    """Map config amp_dtype to (autocast_enabled, dtype). Supports bf16 and fp16."""
    if amp_dtype == "bf16":
        return True, torch.bfloat16
    if amp_dtype == "fp16":
        return True, torch.float16
    return False, None


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

    exp_dir = Path(cfg.output_dir) / cfg.exp_name
    tok_path = exp_dir / "tokenizer" / "best_model"
    restore_best_model(exp_dir, cfg.hf_repo, "tokenizer/best_model", cfg.hf_revision)
    if not has_weights(tok_path):
        raise FileNotFoundError(
            f"Tokenizer weights not found at {tok_path} and could not be restored from HF. "
            "Run train_tokenizer.py first or set HF_TOKEN."
        )

    tok_src, tok_kwargs = resolve_src(tok_path, cfg.hf_repo, "tokenizer/best_model", cfg.hf_revision)
    tokenizer = KronosTokenizer.from_pretrained(tok_src, **tok_kwargs).to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    model = Kronos.from_pretrained(cfg.pretrained_predictor).to(device)
    amp_enabled, amp_dtype = _resolve_amp(cfg.amp_dtype)
    amp_enabled = amp_enabled and device.type == "cuda"
    scaler = GradScaler() if (amp_enabled and amp_dtype == torch.float16) else None

    save_dir = exp_dir / "predictor"
    ckpt_dir = save_dir / "checkpoints"
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
    start_epoch, global_step = _load_latest_checkpoint(ckpt_dir, model, optimizer, scheduler)
    log_path = save_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,step,train_loss,val_loss,val_ic\n")

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

            if max_steps > 0 and global_step >= max_steps:
                return

        val_loss = _validate_predictor(model, tokenizer, val_loader, device, amp_enabled, amp_dtype)
        model.eval()
        val_ic = validate_predictor_ic(
            _make_predict_batch_fn(predictor),
            lambda sym, last, n: _actual_close_lookup(cfg, actual_cache, sym, last, n),
            val_universe,
            val_dates,
            cfg,
            lambda sym, rebal_date: _build_ctx_for_date(cfg, sym, rebal_date),
        )
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{global_step},{loss.item():.4f},{val_loss:.4f},{val_ic:.4f}\n")

        is_best, should_stop = stopper.update(val_ic)
        if is_best:
            model.save_pretrained(str(save_dir / "best_model"))
            print(f"  -> new best val_ic={val_ic:.4f} (val_loss={val_loss:.4f}), saved.")
            push_best_model(save_dir / "best_model", cfg.hf_repo, "predictor/best_model", cfg.hf_revision)
        if should_stop:
            print(f"  -> early stop at epoch {epoch+1} (best val_ic={stopper.best:.4f})")
            break

    wait_for_pushes()


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
