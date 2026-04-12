#!/bin/bash
# stage_a_gen_multimodel launcher — based on launch_stage_a_gen_v3.sh
#
# 3 new models (codestral, starcoder2, deepseek) × 3 formats × 200 prompts
# × 2 shot regimes (0-shot + 3-shot) = 3,600 generations
#
# Changes from v3 launcher:
#   - Models: codestral starcoder2 deepseek (was viscoder2 coder qwen25)
#   - python3 instead of python (AutoDL conda base has no python symlink)
#   - Output/log dirs separate from v3
#   - SVG max_tokens=2048 (D039, already in stage_a_gen.py commit b38963a)
set -euxo pipefail

# ── AutoDL proxy (required for git fetch) ──
source /etc/network_turbo

PROJECT=/root/autodl-tmp/viscode_shared_subspace_probe
LOG_DIR=$PROJECT/artifacts/stage_a/logs/gen_multimodel
OUT_BASE=$PROJECT/artifacts/stage_a/gen_multimodel
ICL_DIR=$PROJECT/artifacts/stage_a/icl_exemplars
EVAL_DIR=$PROJECT/artifacts/stage_a/eval_pool/v3_4

# mkdir BEFORE exec redirect
mkdir -p "$LOG_DIR" "$OUT_BASE"

# Process-level stdout/stderr redirect
exec > >(tee -a "$LOG_DIR/master.log") 2>&1

echo "[launcher multimodel] start $(date -u +%FT%TZ) host=$(hostname)"

# ── HuggingFace offline mode ─
export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export VLLM_USE_V1=0

# ── conda activate base ─────────────────────────────────────────────────
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

cd "$PROJECT"

# ── Inline git resync ──
git fetch origin main
git reset --hard origin/main
HEAD_HASH=$(git rev-parse HEAD)
HEAD_SUBJ=$(git log -1 --format=%s)
echo "[launcher multimodel] git HEAD=$HEAD_HASH subject='$HEAD_SUBJ'"

# ── Fail-fast preflight ──────────────────────────────────────────────────
python3 -c "import vllm, torch; print(f'vllm={vllm.__version__} torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')" || { echo "FAIL_99: vllm/torch import broken"; exit 99; }

if [ ! -f scripts/stage_a_gen.py ]; then
  echo "FAIL_97: scripts/stage_a_gen.py missing"
  exit 97
fi

# v3-specific preflight: required input dirs must exist and be non-empty
for f in svg tikz asymptote; do
  [ -s "$ICL_DIR/${f}.jsonl" ] || { echo "FAIL_94: ICL exemplar missing: $ICL_DIR/${f}.jsonl"; exit 94; }
  [ -s "$EVAL_DIR/${f}.jsonl" ] || { echo "FAIL_93: eval pool missing: $EVAL_DIR/${f}.jsonl"; exit 93; }
done

echo "[launcher multimodel] preflight PASSED, beginning generation matrix"

# ── Generation matrix: 3 models × 3 formats × 2 shards = 18 runs ─────────
# Models serial (each loads/unloads), formats serial within model,
# shards 0/1 parallel via CUDA_VISIBLE_DEVICES inside stage_a_gen.py.
# Each shard runs BOTH 0-shot + 3-shot regimes over its prompt subset.
for MODEL in codestral starcoder2 deepseek; do
  mkdir -p "$OUT_BASE/$MODEL"
  for FMT in svg tikz asy; do
    echo "[launcher multimodel] === $MODEL $FMT begin $(date -u +%FT%TZ) ==="
    python3 scripts/stage_a_gen.py \
      --model "$MODEL" --format "$FMT" \
      --shard-id 0 --num-shards 2 --gpu 0 \
      --shot-regime both \
      --icl-exemplars-dir "$ICL_DIR" \
      --eval-pool-dir "$EVAL_DIR" \
      --out-dir "$OUT_BASE/$MODEL" \
      --max-prompts 200 \
      --seed-base 20260412 \
      > "$LOG_DIR/${MODEL}_${FMT}_s0.log" 2>&1 &
    PID0=$!
    python3 scripts/stage_a_gen.py \
      --model "$MODEL" --format "$FMT" \
      --shard-id 1 --num-shards 2 --gpu 1 \
      --shot-regime both \
      --icl-exemplars-dir "$ICL_DIR" \
      --eval-pool-dir "$EVAL_DIR" \
      --out-dir "$OUT_BASE/$MODEL" \
      --max-prompts 200 \
      --seed-base 20260412 \
      > "$LOG_DIR/${MODEL}_${FMT}_s1.log" 2>&1 &
    PID1=$!
    echo "$(date -u +%FT%TZ) $MODEL $FMT shard0=$PID0 shard1=$PID1" >> "$LOG_DIR/pids.txt"
    # fail-fast: wait each shard
    wait $PID0
    wait $PID1
    echo "[launcher multimodel] === $MODEL $FMT done $(date -u +%FT%TZ) ==="
  done
done

# ── Final count ──
python3 -c "
import pathlib, sys
d = pathlib.Path('$OUT_BASE')
total = 0
for f in sorted(d.rglob('*.jsonl')):
    n = sum(1 for _ in open(f))
    print(f'  {f.relative_to(d)}: {n}')
    total += n
print(f'TOTAL_GENS={total}')
sys.exit(0 if total >= 3500 else 1)
"

echo "[launcher multimodel] ALL DONE $(date -u +%FT%TZ)"
