#!/bin/bash
# stage_a_gen launcher v3 — D025
#
# v3 matrix: 3 models × 3 formats × 200 prompts × 2 shot regimes (0-shot + 3-shot)
#            × n=1 per prompt = 3,600 generations
#
# Key v3 changes vs v2:
#   - --shot-regime both            (was --n-shots 2, n=2 per prompt)
#   - --icl-exemplars-dir           (independent exemplar source, not in-pool)
#   - --eval-pool-dir               (eval pool dir, file per format)
#   - --max-prompts 200             (renamed from --limit)
#   - n=1 per prompt is hard-coded in stage_a_gen.py (not a CLI flag)
#   - out-dir is flat per model (filename includes model/format/shard)
#
# Inherits ALL v2 hardening:
#   1. exec tee redirect (process-level, captures early failures)
#   2. set -euxo pipefail
#   3. conda base env (no venv on autodl-viscode-probe)
#   4. inline `git fetch + reset --hard origin/main` (framework sync_code drift)
#   5. 4 offline env vars (avoid 7-min HF metadata retry per model load)
#   6. fail-fast preflight: vllm/torch import + offline patch + top_p absence
#   7. source /etc/network_turbo (AutoDL proxy for git fetch)
set -euxo pipefail

# ── AutoDL proxy (required for git fetch + apt; harmless to vLLM HF path) ──
source /etc/network_turbo

PROJECT=/root/autodl-tmp/viscode_shared_subspace_probe
LOG_DIR=$PROJECT/artifacts/stage_a/logs/gen_v3
OUT_BASE=$PROJECT/artifacts/stage_a/gen_v3
ICL_DIR=$PROJECT/artifacts/stage_a/icl_exemplars
EVAL_DIR=$PROJECT/artifacts/stage_a/eval_pool/v3_4

# mkdir BEFORE exec redirect — tee will silently fall back to stderr otherwise
mkdir -p "$LOG_DIR" "$OUT_BASE"

# Process-level stdout/stderr redirect — captures even early failures
exec > >(tee -a "$LOG_DIR/master.log") 2>&1

echo "[launcher v3] start $(date -u +%FT%TZ) host=$(hostname)"

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
echo "[launcher v3] git HEAD=$HEAD_HASH subject='$HEAD_SUBJ'"

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

# v3-specific preflight: required input dirs must exist and be non-empty
for f in svg tikz asymptote; do
  [ -s "$ICL_DIR/${f}.jsonl" ] || { echo "FAIL_94: ICL exemplar missing: $ICL_DIR/${f}.jsonl"; exit 94; }
  [ -s "$EVAL_DIR/${f}.jsonl" ] || { echo "FAIL_93: eval pool missing: $EVAL_DIR/${f}.jsonl"; exit 93; }
done

echo "[launcher v3] preflight PASSED, beginning generation matrix"

# ── Generation matrix: 3 models × 3 formats × 2 shards = 18 runs ─────────
# Models serial (each loads/unloads), formats serial within model,
# shards 0/1 parallel via CUDA_VISIBLE_DEVICES inside stage_a_gen.py.
# Each shard runs BOTH 0-shot + 3-shot regimes over its prompt subset.
for MODEL in viscoder2 coder qwen25; do
  mkdir -p "$OUT_BASE/$MODEL"
  for FMT in svg tikz asy; do
    echo "[launcher v3] === $MODEL $FMT begin $(date -u +%FT%TZ) ==="
    python scripts/stage_a_gen.py \
      --model "$MODEL" --format "$FMT" \
      --shard-id 0 --num-shards 2 --gpu 0 \
      --shot-regime both \
      --icl-exemplars-dir "$ICL_DIR" \
      --eval-pool-dir "$EVAL_DIR" \
      --out-dir "$OUT_BASE/$MODEL" \
      > "$LOG_DIR/${MODEL}_${FMT}_s0.log" 2>&1 &
    PID0=$!
    python scripts/stage_a_gen.py \
      --model "$MODEL" --format "$FMT" \
      --shard-id 1 --num-shards 2 --gpu 1 \
      --shot-regime both \
      --icl-exemplars-dir "$ICL_DIR" \
      --eval-pool-dir "$EVAL_DIR" \
      --out-dir "$OUT_BASE/$MODEL" \
      > "$LOG_DIR/${MODEL}_${FMT}_s1.log" 2>&1 &
    PID1=$!
    echo "$(date -u +%FT%TZ) $MODEL $FMT shard0=$PID0 shard1=$PID1" >> "$LOG_DIR/pids.txt"
    # fail-fast: wait each shard; `set -e` will abort the loop on non-zero
    wait $PID0
    wait $PID1
    echo "[launcher v3] === $MODEL $FMT done $(date -u +%FT%TZ) ==="
  done
done

echo "[launcher v3] ALL DONE $(date -u +%FT%TZ)"
