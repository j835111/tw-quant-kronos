#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH=""
STAGE="predictor"
REPO_URL=""
REPO_DIR="/marimo/Kronos"
STATE_DIR="/mnt/first/kronos_state"
BRANCH=""

usage() {
  cat <<'EOF' >&2
Usage: resume_molab_training.sh --config <path> [--stage tokenizer|predictor] [--repo-url <url>] [--repo-dir <path>] [--state-dir <path>] [--branch <name>]
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || usage
      CONFIG_PATH="$2"
      shift 2
      ;;
    --stage)
      [[ $# -ge 2 ]] || usage
      STAGE="$2"
      shift 2
      ;;
    --repo-url)
      [[ $# -ge 2 ]] || usage
      REPO_URL="$2"
      shift 2
      ;;
    --repo-dir)
      [[ $# -ge 2 ]] || usage
      REPO_DIR="$2"
      shift 2
      ;;
    --state-dir)
      [[ $# -ge 2 ]] || usage
      STATE_DIR="$2"
      shift 2
      ;;
    --branch)
      [[ $# -ge 2 ]] || usage
      BRANCH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      ;;
  esac
done

[[ -n "$CONFIG_PATH" ]] || {
  echo "--config is required" >&2
  usage
}

if [[ "$STAGE" != "tokenizer" && "$STAGE" != "predictor" ]]; then
  echo "stage must be tokenizer or predictor" >&2
  exit 1
fi

GIT_BIN="${KRONOS_GIT_BIN:-git}"
LAUNCH_PYTHON="${KRONOS_LAUNCH_PYTHON:-python3}"
LAUNCH_WAIT_SECONDS="${KRONOS_LAUNCH_WAIT_SECONDS:-2}"
STATE_DIR="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$STATE_DIR")"
ORIGINAL_CWD="$(pwd)"
CONFIG_DIR=""
resolve_config_path() {
  if [[ "$CONFIG_PATH" = /* ]]; then
    CONFIG_PATH="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$CONFIG_PATH")"
  elif [[ -f "$CONFIG_PATH" ]]; then
    CONFIG_PATH="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$CONFIG_PATH")"
  elif [[ -f "$REPO_DIR/$CONFIG_PATH" ]]; then
    CONFIG_PATH="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$REPO_DIR/$CONFIG_PATH")"
  elif [[ -f "$ORIGINAL_CWD/$CONFIG_PATH" ]]; then
    CONFIG_PATH="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$ORIGINAL_CWD/$CONFIG_PATH")"
  else
    echo "Missing config: $CONFIG_PATH" >&2
    exit 1
  fi
  CONFIG_DIR="$(dirname "$CONFIG_PATH")"
}

mkdir -p "$STATE_DIR/data" "$STATE_DIR/outputs" "$STATE_DIR/logs" "$STATE_DIR/run"

read_config_value() {
  local key="$1"
  python3 - "$CONFIG_PATH" "$key" <<'PY'
import ast
import sys

config_path, key = sys.argv[1], sys.argv[2]

with open(config_path, encoding="utf-8") as fh:
    for raw_line in fh:
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        current_key, value = line.split(":", 1)
        if current_key.strip() != key:
            continue
        value = value.strip()
        if not value:
            print("")
            raise SystemExit(0)
        if value[0] in "\"'" and value[-1] == value[0]:
            print(ast.literal_eval(value))
        else:
            print(value)
        raise SystemExit(0)

raise SystemExit(f"Missing config key: {key}")
PY
}

resolve_path() {
  python3 - "$1" "$CONFIG_DIR" <<'PY'
import os
import sys

target, base_dir = sys.argv[1], sys.argv[2]
if os.path.isabs(target):
    print(os.path.realpath(target))
else:
    print(os.path.realpath(os.path.join(base_dir, target)))
PY
}

path_within_dir() {
  python3 - "$1" "$2" <<'PY'
from pathlib import Path
import sys

target = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
try:
    target.relative_to(root)
except ValueError:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

stop_pidfile() {
  local pidfile="$1"
  local expected_substring="$2"
  if [[ ! -f "$pidfile" ]]; then
    return 0
  fi

  local pid
  pid="$(<"$pidfile")"
  if [[ ! "$pid" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid pid in $pidfile: $pid" >&2
    return 1
  fi
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    if [[ -r "/proc/$pid/cmdline" ]]; then
      local cmdline
      cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
      if [[ "$cmdline" != *"$expected_substring"* ]]; then
        echo "refusing to stop unexpected process $pid from $pidfile" >&2
        return 1
      fi
    else
      echo "unable to verify process identity for $pid from $pidfile" >&2
      return 1
    fi
    kill "$pid" >/dev/null 2>&1 || true
    for _ in $(seq 1 50); do
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        break
      fi
      sleep 0.1
    done
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" >/dev/null 2>&1 || true
      sleep 0.1
      if kill -0 "$pid" >/dev/null 2>&1; then
        echo "failed to stop existing process $pid from $pidfile" >&2
        return 1
      fi
    fi
  fi
  rm -f "$pidfile"
}

if ! "$GIT_BIN" -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  rm -rf "$REPO_DIR"
  [[ -n "$REPO_URL" ]] || {
    echo "repo checkout invalid and --repo-url not provided" >&2
    exit 1
  }
  "$GIT_BIN" clone "$REPO_URL" "$REPO_DIR"
fi

if [[ -n "$BRANCH" ]]; then
  "$GIT_BIN" -C "$REPO_DIR" fetch origin "$BRANCH" >/dev/null 2>&1
  "$GIT_BIN" -C "$REPO_DIR" checkout "$BRANCH" >/dev/null 2>&1
  "$GIT_BIN" -C "$REPO_DIR" reset --hard "origin/$BRANCH" >/dev/null 2>&1
fi

resolve_config_path
DB_PATH="$(resolve_path "$(read_config_value db_path)")"
OUTPUT_DIR="$(resolve_path "$(read_config_value output_dir)")"
EXP_NAME="$(read_config_value exp_name)"

if ! path_within_dir "$DB_PATH" "$STATE_DIR"; then
  echo "db_path must live under state-dir" >&2
  exit 1
fi

if ! path_within_dir "$OUTPUT_DIR" "$STATE_DIR"; then
  echo "output_dir must live under state-dir" >&2
  exit 1
fi

TRAIN_LOG="$STATE_DIR/logs/${STAGE}_train_stdout.log"
TRAIN_PIDFILE="$STATE_DIR/run/${STAGE}.pid"
MONITOR_LOG="$STATE_DIR/logs/${STAGE}_monitor.log"
MONITOR_PIDFILE="$STATE_DIR/run/${STAGE}_monitor.pid"
STAGE_DIR="$OUTPUT_DIR/$EXP_NAME/$STAGE"
MODULE_NAME="finetune_tw.train_${STAGE}"

stop_pidfile "$TRAIN_PIDFILE" "$MODULE_NAME"
stop_pidfile "$MONITOR_PIDFILE" "resume_molab_training.sh"

(
  cd "$REPO_DIR"
  exec "$LAUNCH_PYTHON" -m "$MODULE_NAME" --config "$CONFIG_PATH"
) >>"$TRAIN_LOG" 2>&1 &
TRAIN_PID="$!"
echo "$TRAIN_PID" > "$TRAIN_PIDFILE"
launch_checks="$(python3 - "$LAUNCH_WAIT_SECONDS" <<'PY'
import math
import sys
seconds = float(sys.argv[1])
print(max(1, math.ceil(seconds / 0.1)))
PY
)"
for _ in $(seq 1 "$launch_checks"); do
  if ! kill -0 "$TRAIN_PID" >/dev/null 2>&1; then
    rm -f "$TRAIN_PIDFILE"
    echo "training process failed to start" >&2
    exit 1
  fi
  sleep 0.1
done
if ! kill -0 "$TRAIN_PID" >/dev/null 2>&1; then
  rm -f "$TRAIN_PIDFILE"
  echo "training process failed to start" >&2
  exit 1
fi

if [[ "${KRONOS_SKIP_MONITOR:-0}" != "1" ]]; then
  (
    oneshot_retries=0
    max_oneshot_retries="${KRONOS_MONITOR_ONESHOT_RETRIES:-10}"
    while true; do
      snapshot="$(python3 - "$STAGE_DIR/checkpoints" "$STAGE_DIR/train_log.csv" <<'PY'
from pathlib import Path
import re
import sys

ckpt_dir = Path(sys.argv[1])
log_path = Path(sys.argv[2])

checkpoints = []
for path in ckpt_dir.glob("ckpt-*.pt"):
    if not path.is_file():
        continue
    if not re.fullmatch(r"ckpt-[0-9]+\.pt", path.name):
        continue
    try:
        step = int(path.stem.split("-", 1)[1])
    except (IndexError, ValueError):
        continue
    checkpoints.append((step, path.name))
checkpoints.sort()
latest_ckpt = checkpoints[-1][1] if checkpoints else "none"

if not log_path.exists():
    last_csv_line = "none"
else:
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    last_csv_line = lines[-1] if lines else "none"

print(f"{latest_ckpt}\t{last_csv_line}")
PY
)"
      latest_ckpt="${snapshot%%$'\t'*}"
      last_csv_line="${snapshot#*$'\t'}"
      if [[ "${KRONOS_MONITOR_ONESHOT:-0}" == "1" && "$latest_ckpt" == "none" && "$last_csv_line" == "none" ]]; then
        oneshot_retries=$((oneshot_retries + 1))
        if (( oneshot_retries >= max_oneshot_retries )); then
          printf '%s checkpoint=%s csv=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$latest_ckpt" "$last_csv_line" >> "$MONITOR_LOG"
          break
        fi
        sleep 0.3
        continue
      fi
      printf '%s checkpoint=%s csv=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$latest_ckpt" "$last_csv_line" >> "$MONITOR_LOG"
      if [[ "${KRONOS_MONITOR_ONESHOT:-0}" == "1" ]]; then
        break
      fi
      sleep 60
    done
  ) &
  echo "$!" > "$MONITOR_PIDFILE"
fi
