#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
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
    if "$py" -c "import huggingface_hub, matplotlib, pandas, torch, yaml" >/dev/null 2>&1; then
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

echo "==> Updating DB"
"$PYTHON_BIN" -m finetune_tw.download_data \
  --config "$CONFIG_PATH" \
  --update

echo "==> Running signal_today"
"$PYTHON_BIN" -m finetune_tw.signal_today \
  --config "$CONFIG_PATH" \
  --model round0 \
  --top_k 10 \
  --hold_days 3
