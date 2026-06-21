"""
Tokenizer quality eval: reconstruction fidelity + codebook utilisation,
fine-tuned vs pretrained Kronos-Tokenizer-base. Settles whether re-finetuning
the tokenizer helped or whether the predictor should just reuse the pretrained
tokenizer (freeze).

Run:
    python -m finetune_tw.eval_tokenizer --config <cfg>
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import KronosTokenizer
from finetune_tw.config import Config
from finetune_tw.dataset import MultiStockDataset


def _usage_stats(counter: dict) -> dict:
    tot = sum(counter.values())
    ps = [v / tot for v in counter.values()]
    entropy = -sum(p * math.log2(p) for p in ps if p > 0)
    return {"unique_tokens": len(counter), "entropy_bits": float(entropy),
            "total_tokens": int(tot)}


def eval_one(cfg, src: str, tag: str, device) -> dict:
    tok = KronosTokenizer.from_pretrained(src).to(device)
    tok.eval()
    ds = MultiStockDataset(cfg.db_path, cfg.lookback_window, cfg.predict_window,
                           cfg.train_end_date, cfg.val_end_date, cfg.clip, cfg.seed + 1)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    tot_mse = tot_mse_pre = 0.0
    n = 0
    s1_counter: dict[int, int] = {}
    s2_counter: dict[int, int] = {}
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            (z_pre, z), _bsq, _, _ = tok(x)
            tot_mse += F.mse_loss(z, x).item() * x.size(0)
            tot_mse_pre += F.mse_loss(z_pre, x).item() * x.size(0)
            n += x.size(0)
            s1, s2 = tok.encode(x, half=True)
            for t in s1.flatten().tolist():
                s1_counter[t] = s1_counter.get(t, 0) + 1
            for t in s2.flatten().tolist():
                s2_counter[t] = s2_counter.get(t, 0) + 1
    return {
        "variant": tag, "source": src, "n_windows": n,
        "recon_mse": tot_mse / n,            # post-quantizer decode vs input
        "recon_mse_prequant": tot_mse_pre / n,  # pre-quant decode vs input
        "s1_usage": _usage_stats(s1_counter),
        "s2_usage": _usage_stats(s2_counter),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ft_src = str(Path(cfg.output_dir) / cfg.exp_name / "tokenizer" / "best_model")
    results = []
    print("evaluating fine-tuned tokenizer ...", flush=True)
    results.append(eval_one(cfg, ft_src, "finetuned", device))
    print("evaluating pretrained tokenizer ...", flush=True)
    results.append(eval_one(cfg, cfg.pretrained_tokenizer, "baseline", device))

    out_dir = Path(cfg.output_dir) / cfg.exp_name / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_tokenizer.json").write_text(json.dumps(results, indent=2))

    print("\n=== Tokenizer Eval (val set) ===")
    print(f"{'variant':>10} {'recon_mse':>10} {'mse_preq':>10} "
          f"{'s1_uniq':>8} {'s1_ent':>7} {'s2_uniq':>8} {'s2_ent':>7}")
    for r in results:
        print(f"{r['variant']:>10} {r['recon_mse']:>10.5f} {r['recon_mse_prequant']:>10.5f} "
              f"{r['s1_usage']['unique_tokens']:>8} {r['s1_usage']['entropy_bits']:>7.2f} "
              f"{r['s2_usage']['unique_tokens']:>8} {r['s2_usage']['entropy_bits']:>7.2f}")
    print(f"saved {out_dir / 'eval_tokenizer.json'}", flush=True)


if __name__ == "__main__":
    main()
