#!/usr/bin/env bash
# run_signal_today_ensemble.sh — Run daily database update and output Kronos stock signals using static Z-Score ensemble blending.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="finetune_tw/configs/config_tw_daily.yaml"
TMP_PATH="$REPO_ROOT/.tmp"
MPLCONFIG_PATH="$REPO_ROOT/.cache/matplotlib"

mkdir -p "$TMP_PATH" "$MPLCONFIG_PATH"

export TMPDIR="$TMP_PATH"
export MPLCONFIGDIR="$MPLCONFIG_PATH"
export MPLBACKEND="Agg"

choose_python() {
  if [[ -n "${KRONOS_PYTHON:-}" ]]; then
    printf '%s\n' "$KRONOS_PYTHON"
    return 0
  fi

  local candidates=("$REPO_ROOT/.venv/bin/python")
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi

  local py
  for py in "${candidates[@]}"; do
    [[ -x "$py" ]] || continue
    if "$py" -c "import huggingface_hub, matplotlib, pandas, torch, yaml, xgboost" >/dev/null 2>&1; then
      printf '%s\n' "$py"
      return 0
    fi
  done

  echo "No usable Python interpreter found for Kronos signal flow." >&2
  echo "Checked: ${candidates[*]}" >&2
  exit 1
}

if [[ ! -f "$REPO_ROOT/$CONFIG_PATH" ]]; then
  echo "Missing config: $REPO_ROOT/$CONFIG_PATH" >&2
  exit 1
fi

PYTHON_BIN="$(choose_python)"

cd "$REPO_ROOT"

echo "==> Updating database..."
"$PYTHON_BIN" -m finetune_tw.download_data \
  --config "$CONFIG_PATH" \
  --update

echo "==> Running daily signal generation with Ensemble Blending..."
# Default models from Round 6 Batch 3c
XGB_FULL="finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/xgb_batch3c_full.json"
XGB_RAW="finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/xgb_batch3c_raw.json"

"$PYTHON_BIN" -m finetune_tw.signal_today_ensemble \
  --config "$CONFIG_PATH" \
  --xgb_model_full "$XGB_FULL" \
  --xgb_model_raw "$XGB_RAW" \
  --weight 0.6 \
  --top_k 10 \
  "$@"
