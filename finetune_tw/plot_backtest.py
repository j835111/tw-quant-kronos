"""
Plot cumulative return lines from backtest_returns_*.json files.

Usage (local, no GPU needed):
    python finetune_tw/plot_backtest.py \\
        finetune_tw/outputs/tw_daily/backtest_returns_pretrained.json \\
        finetune_tw/outputs/tw_daily/backtest_returns_round0.json \\
        finetune_tw/outputs/tw_daily/backtest_returns_round1.json \\
        --output finetune_tw/outputs/tw_daily/backtest_9way.png

    # Only specific hold variants:
    python finetune_tw/plot_backtest.py *.json --hold_days 5 10

    # Single file (still plots all hold variants in that file):
    python finetune_tw/plot_backtest.py backtest_returns_round0.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


# ── Colour / style maps ───────────────────────────────────────────────────────

MODEL_COLORS = {
    "pretrained": "tab:blue",
    "round0":     "tab:orange",
    "round1":     "tab:green",
}
HOLD_STYLES = {
    5:  "-",
    10: "--",
    15: ":",
    20: "-.",
}
DEFAULT_COLORS = ["tab:purple", "tab:red", "tab:brown", "tab:pink", "tab:cyan"]


def color_for(model_key: str, idx: int) -> str:
    return MODEL_COLORS.get(model_key, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])


def style_for(hold_days: int) -> str:
    return HOLD_STYLES.get(hold_days, (0, (3, 1, 1, 1)))


# ── Load & reconstruct ────────────────────────────────────────────────────────

def load_result(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def to_series(dates: list[str], values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.DatetimeIndex(dates))


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(
    result_files: list[Path],
    hold_days_filter: list[int] | None,
    output_path: Path,
) -> None:
    results = [load_result(p) for p in result_files]
    if not results:
        print("No input files found.")
        return

    fig, ax = plt.subplots(figsize=(16, 7))

    bm_plotted = False
    legend_models: list[Line2D] = []
    legend_holds:  list[Line2D] = []

    for idx, res in enumerate(results):
        model_key   = res["model_key"]
        model_label = res["model_label"]
        color       = color_for(model_key, idx)

        # Track which hold variants appear (for legend)
        variant_keys = [int(k) for k in res["hold_variants"]]
        if hold_days_filter:
            variant_keys = [h for h in variant_keys if h in hold_days_filter]

        for hd in sorted(variant_keys):
            v = res["hold_variants"][str(hd)]
            dr = to_series(v["dates"], v["daily_returns"])
            cum = (1 + dr).cumprod()
            m = v["metrics"]
            ls = style_for(hd)
            ax.plot(cum.index, cum.values,
                    color=color, linestyle=ls, linewidth=1.5,
                    label=f"{model_label} h={hd}d  "
                          f"Ann={m['annualised_return']:+.1%}  "
                          f"Sh={m['sharpe']:.2f}  "
                          f"DD={m['max_drawdown']:.1%}")

        # Model legend entry (one per file)
        legend_models.append(Line2D([0], [0], color=color, linewidth=2.5, label=model_label))

        # Benchmark — use first occurrence
        if not bm_plotted and "benchmark" in res:
            bm = res["benchmark"]
            bm_dr = to_series(bm["dates"], bm["daily_returns"])
            bm_cum = (1 + bm_dr).cumprod()
            bm_m = bm["metrics"]
            ax.plot(bm_cum.index, bm_cum.values,
                    color="black", linestyle="-.", linewidth=2.0,
                    label=f"^TWII  Ann={bm_m['annualised_return']:+.1%}  "
                          f"Sh={bm_m['sharpe']:.2f}  "
                          f"DD={bm_m['max_drawdown']:.1%}")
            bm_plotted = True

    # Hold-style legend entries
    all_holds = sorted({int(k)
                        for res in results
                        for k in res["hold_variants"]
                        if hold_days_filter is None or int(k) in hold_days_filter})
    for hd in all_holds:
        legend_holds.append(Line2D([0], [0], color="black", linestyle=style_for(hd),
                                   linewidth=1.5, label=f"hold={hd}d"))
    legend_holds.append(Line2D([0], [0], color="black", linestyle="-.", linewidth=2, label="Benchmark"))

    # Two-part legend: top = models (colour), bottom = hold styles
    legend1 = ax.legend(handles=legend_models + legend_holds,
                        loc="upper left", fontsize=8.5,
                        title="Color = Model    Style = Hold",
                        title_fontsize=8)
    ax.add_artist(legend1)

    # Axis formatting
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}x"))
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Return (NAV, start = 1×)", fontsize=11)
    ax.grid(alpha=0.2)

    # Title from metadata
    first = results[0]
    model_labels = " / ".join(r["model_label"] for r in results)
    top_k = first.get("top_k", "?")
    ax.set_title(
        f"Backtest Comparison — {model_labels} — top-{top_k}\n"
        f"{first['test_start']} → {first['test_end']}",
        fontsize=12,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved → {output_path}")

    # Print summary table
    print("\n── Summary ──────────────────────────────────────────────────")
    print(f"{'Model':<14} {'Hold':>6}  {'Ann Ret':>9}  {'Sharpe':>7}  {'Max DD':>8}")
    print("-" * 52)
    for res in results:
        label = res["model_label"]
        variant_keys = [int(k) for k in res["hold_variants"]]
        if hold_days_filter:
            variant_keys = [h for h in variant_keys if h in hold_days_filter]
        for hd in sorted(variant_keys):
            m = res["hold_variants"][str(hd)]["metrics"]
            print(f"{label:<14} {hd:>5}d  "
                  f"{m['annualised_return']:>+8.2%}  "
                  f"{m['sharpe']:>7.2f}  "
                  f"{m['max_drawdown']:>7.2%}")
    if bm_plotted:
        bm_m = results[0]["benchmark"]["metrics"]
        print("-" * 52)
        print(f"{'Benchmark (^TWII)':<14}         "
              f"{bm_m['annualised_return']:>+8.2%}  "
              f"{bm_m['sharpe']:>7.2f}  "
              f"{bm_m['max_drawdown']:>7.2%}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+",
                        help="backtest_returns_*.json files to plot")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: same dir as first input, backtest_comparison.png)")
    parser.add_argument("--hold_days", type=int, nargs="+", default=None,
                        help="Only plot these hold variants (default: all in files)")
    args = parser.parse_args()

    files = [Path(p) for p in args.inputs]
    missing = [p for p in files if not p.exists()]
    if missing:
        print(f"File(s) not found: {missing}")
        return

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = files[0].parent / "backtest_comparison.png"

    plot_results(files, args.hold_days, out_path)


if __name__ == "__main__":
    main()
