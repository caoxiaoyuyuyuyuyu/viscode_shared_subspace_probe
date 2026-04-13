#!/bin/bash
set -euo pipefail

# Negative Control (C1): Python non-visual code probe + CKA analysis
# 3 models × python × 252 samples × 7 layers, then CKA comparison

cd /root/autodl-tmp/viscode_shared_subspace_probe
PYTHON="/root/miniconda3/bin/python -u"
LOG_DIR=/root/autodl-tmp/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/negative_control.log"

# Default to nohup self-detach so SSH session drop does not kill the job.
# Set NEG_CTRL_FG=1 to run in foreground (for interactive debugging).
if [[ "${NEG_CTRL_DETACHED:-0}" != "1" && "${NEG_CTRL_FG:-0}" != "1" ]]; then
    export NEG_CTRL_DETACHED=1
    nohup bash "$0" "$@" >> "$LOG" 2>&1 &
    echo "PID=$!" >> "$LOG"
    echo "[launcher] detached nohup PID=$! log=$LOG"
    exit 0
fi

# Diagnosability trap: KILL cannot be trapped by POSIX, so we only catch TERM/HUP/INT.
# OOM-killer sends SIGKILL → not trappable; silent death is expected in that case.
trap 'echo "[TRAP] killed pid=$$ sig=TERM at $(date -Iseconds)" >> "$LOG"' TERM
trap 'echo "[TRAP] killed pid=$$ sig=HUP at $(date -Iseconds)" >> "$LOG"' HUP
trap 'echo "[TRAP] killed pid=$$ sig=INT at $(date -Iseconds)" >> "$LOG"' INT

{
  echo "[launcher] start $(date -Iseconds) commit=$(git rev-parse --short HEAD) pid=$$"
  echo "[diag] ---- memory ----"
  free -g || true
  echo "[diag] ---- disk ----"
  df -h /root/autodl-tmp || true
  echo "[diag] ---- gpu ----"
  nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv || true
} | tee -a "$LOG"

# Step 1: Prepare Python snippets
echo "[launcher] === Step 1: Prepare Python snippets ===" | tee -a "$LOG"
$PYTHON scripts/prepare_python_snippets.py 2>&1 | tee -a "$LOG"

# Verify snippets created
SNIPPETS=/root/autodl-tmp/cache/hidden_states/python_snippets.json
if [ ! -f "$SNIPPETS" ]; then
    echo "[FATAL] python_snippets.json not created" | tee -a "$LOG"
    exit 1
fi
N=$($PYTHON -c "import json; print(len(json.load(open('$SNIPPETS'))))")
echo "[launcher] snippets: $N" | tee -a "$LOG"
if [ "$N" -ne 252 ]; then
    echo "[FATAL] Expected 252 snippets, got $N" | tee -a "$LOG"
    exit 1
fi

# Step 2: Extract hidden states — 2 models parallel on 2 GPUs, then 1
echo "[launcher] === Step 2: Probe extraction ===" | tee -a "$LOG"

# Round 1: coder (GPU0) + viscoder2 (GPU1) in parallel
echo "[launcher] Round 1: coder + viscoder2 parallel" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/negative_control_probe.py --model coder --gpu 0 2>&1 | tee -a "$LOG" &
PID_CODER=$!
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/negative_control_probe.py --model viscoder2 --gpu 1 2>&1 | tee -a "$LOG" &
PID_VISC=$!
echo "[launcher] PIDs: coder=$PID_CODER viscoder2=$PID_VISC" | tee -a "$LOG"
wait $PID_CODER $PID_VISC
echo "[launcher] Round 1 done $(date -u +%FT%TZ)" | tee -a "$LOG"

# Round 2: qwen25 (GPU0)
echo "[launcher] Round 2: qwen25" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/negative_control_probe.py --model qwen25 --gpu 0 2>&1 | tee -a "$LOG"
echo "[launcher] Round 2 done $(date -u +%FT%TZ)" | tee -a "$LOG"

# Step 3: CKA analysis
echo "[launcher] === Step 3: CKA analysis ===" | tee -a "$LOG"
$PYTHON scripts/negative_control_cka.py 2>&1 | tee -a "$LOG"

echo "[launcher] === ALL DONE $(date -u +%FT%TZ) ===" | tee -a "$LOG"
echo "NEGATIVE_CONTROL_SENTINEL_DONE"
