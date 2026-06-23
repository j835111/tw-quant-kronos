#!/usr/bin/env bash
# Compare mc_sample_count=3,5,10 in parallel (requires cached OOF).
# Usage: bash finetune_tw/mc_compare.sh [config]
set -euo pipefail

CONFIG="${1:-finetune_tw/configs/config_tw_daily_rtx6000.yaml}"
REPO="$(git rev-parse --show-toplevel)"
LOG_DIR="/tmp/stacking_mc_compare"
mkdir -p "$LOG_DIR"

echo "[mc_compare] launching 3 parallel test backtests (mc=3,5,10) ..."
echo "[mc_compare] config: $CONFIG"
echo "[mc_compare] logs: $LOG_DIR"

cd "$REPO"

python3 -m finetune_tw.stacking_backtest \
    --config "$CONFIG" --model round0 --mc 3 --suffix _mc3 \
    > "$LOG_DIR/mc3.log" 2>&1 &
PID3=$!

python3 -m finetune_tw.stacking_backtest \
    --config "$CONFIG" --model round0 --mc 5 --suffix _mc5 \
    > "$LOG_DIR/mc5.log" 2>&1 &
PID5=$!

python3 -m finetune_tw.stacking_backtest \
    --config "$CONFIG" --model round0 --mc 10 --suffix _mc10 \
    > "$LOG_DIR/mc10.log" 2>&1 &
PID10=$!

echo "[mc_compare] PIDs: mc3=$PID3  mc5=$PID5  mc10=$PID10"
echo "[mc_compare] waiting ..."
wait $PID3 $PID5 $PID10
echo "[mc_compare] all done. comparing results ..."

python3 - <<'PY'
import json, glob, sys
from pathlib import Path

out_dir = Path("finetune_tw/outputs/tw_daily")
results = {}
for suffix in ["_mc3", "_mc5", "_mc10"]:
    p = out_dir / f"backtest_stacking{suffix}.json"
    if not p.exists():
        print(f"MISSING: {p}")
        continue
    d = json.loads(p.read_text())
    mc = suffix.lstrip("_mc")
    results[f"mc={mc}"] = {
        "stacker":     d.get("stacker",     {}).get("metrics", {}),
        "kronos_only": d.get("kronos_only", {}).get("metrics", {}),
    }

if not results:
    print("No results found.")
    sys.exit(1)

print(f"\n{'':12} {'Sharpe':>8} {'Ann%':>8} {'MaxDD%':>8}")
print("-" * 42)
for label, r in sorted(results.items()):
    for key in ["stacker", "kronos_only"]:
        m = r[key]
        if not m:
            continue
        tag = f"{label}/{key[:7]}"
        print(f"{tag:20} {m.get('sharpe',0):8.2f} "
              f"{m.get('annualised_return',0):8.1%} "
              f"{m.get('max_drawdown',0):8.1%}")
    print()
PY
