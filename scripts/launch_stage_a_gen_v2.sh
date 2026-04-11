#!/bin/bash
# stage_a_gen launcher v2 — Reviewer 二审 critical fixes:
#   1. exec tee redirect (process-level, captures even bash startup failures)
#   2. set -euxo pipefail (fail-fast + trace)
#   3. conda base env (autodl-viscode-probe has no venv, only /root/miniconda3)
#   4. inline `git fetch + reset --hard origin/main` (sync_code framework only
#      fetches, does not reset working tree — proven via reproduce.code_commit
#      stuck at a49fc35 even with sync_code=true)
#   5. 4 offline env vars to prevent 7-min HF metadata retry per model load
#      (× 18 launches = 2.1h waste otherwise)
#   6. fail-fast preflight: vllm/torch import + offline patch presence + top_p absence
set -euxo pipefail

PROJECT=/root/autodl-tmp/viscode_shared_subspace_probe
LOG_DIR=$PROJECT/artifacts/stage_a/logs/gen_v2
OUT_BASE=$PROJECT/artifacts/stage_a/gen_v2

# mkdir BEFORE exec redirect — tee will silently fall back to stderr otherwise
mkdir -p "$LOG_DIR" "$OUT_BASE"

# Process-level stdout/stderr redirect — captures even early failures
exec > >(tee -a "$LOG_DIR/master.log") 2>&1

echo "[launcher v2] start $(date -u +%FT%TZ) host=$(hostname)"

# ── HuggingFace offline mode (Reviewer critical: avoid 7-min retry per load) ─
export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# ── conda activate base (server has no venv) ─────────────────────────────
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

cd "$PROJECT"

# ── Inline git resync (framework sync_code only fetches, doesn't reset) ──
git fetch origin main
git reset --hard origin/main
HEAD_HASH=$(git rev-parse HEAD)
HEAD_SUBJ=$(git log -1 --format=%s)
echo "[launcher v2] git HEAD=$HEAD_HASH subject='$HEAD_SUBJ'"

# ── Fail-fast preflight ──────────────────────────────────────────────────
python -c "import vllm, torch; print(f'vllm={vllm.__version__} torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')" || { echo "FAIL_99: vllm/torch import broken"; exit 99; }

if [ ! -f scripts/stage_a_gen.py ]; then
  echo "FAIL_97: scripts/stage_a_gen.py missing"
  exit 97
fi

if ! grep -q "TRANSFORMERS_OFFLINE" scripts/stage_a_gen.py; then
  echo "FAIL_96: stage_a_gen.py missing offline patch"
  exit 96
fi

if grep -q "top_p" scripts/stage_a_gen.py; then
  echo "FAIL_95: stage_a_gen.py still has top_p (Reviewer 二审 fix lost)"
  exit 95
fi

echo "[launcher v2] preflight PASSED, beginning generation matrix"

# ── Generation matrix: 3 models × 3 formats × 2 shards = 18 runs ─────────
# Models serial (each loads/unloads), formats serial within model,
# shards 0/1 parallel via CUDA_VISIBLE_DEVICES inside stage_a_gen.py
for MODEL in viscoder2 coder qwen25; do
  mkdir -p "$OUT_BASE/$MODEL"
  for FMT in svg tikz asy; do
    echo "[launcher v2] === $MODEL $FMT begin $(date -u +%FT%TZ) ==="
    python scripts/stage_a_gen.py \
      --model "$MODEL" --format "$FMT" --shard-id 0 --num-shards 2 \
      --gpu 0 --n-shots 2 --out-dir "$OUT_BASE/$MODEL" \
      > "$LOG_DIR/${MODEL}_${FMT}_s0.log" 2>&1 &
    PID0=$!
    python scripts/stage_a_gen.py \
      --model "$MODEL" --format "$FMT" --shard-id 1 --num-shards 2 \
      --gpu 1 --n-shots 2 --out-dir "$OUT_BASE/$MODEL" \
      > "$LOG_DIR/${MODEL}_${FMT}_s1.log" 2>&1 &
    PID1=$!
    echo "$(date -u +%FT%TZ) $MODEL $FMT shard0=$PID0 shard1=$PID1" >> "$LOG_DIR/pids.txt"
    wait $PID0 $PID1
    echo "[launcher v2] === $MODEL $FMT done $(date -u +%FT%TZ) ==="
  done
done

echo "[launcher v2] ALL DONE $(date -u +%FT%TZ)"
