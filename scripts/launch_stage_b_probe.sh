#!/bin/bash
# Stage B: HF teacher-forcing probe — launcher
# 3 models × 3 formats × 252 triples, 2-GPU data-parallel (models parallel)
# Pre-req: sbert_triples.json + resolved_triples.json must exist
set -euxo pipefail

source /etc/network_turbo 2>/dev/null || true

# ── Conda ──
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
PYTHON=/root/miniconda3/bin/python

# ── Env ──
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=/root/autodl-tmp/.hf_cache

cd /root/autodl-tmp/viscode_shared_subspace_probe

# ── Git sync ──
source /etc/network_turbo 2>/dev/null || true
git fetch origin && git reset --hard origin/main
echo "=== Git synced to $(git rev-parse --short HEAD) ==="

# ── Preflight ──
$PYTHON -c "import torch, transformers; print(f'torch={torch.__version__} transformers={transformers.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')"
test -f scripts/stage_b_probe.py || { echo "FAIL: stage_b_probe.py not found"; exit 99; }

TRIPLES=outputs/stage_a/sbert_triples.json
test -f "$TRIPLES" || { echo "FAIL: sbert_triples.json not found"; exit 97; }

OUT_DIR=/root/autodl-tmp/cache/hidden_states

# ── Step 1: Resolve triples (if not cached) ──
RESOLVED="$(dirname "$TRIPLES")/resolved_triples.json"
if [ ! -f "$RESOLVED" ]; then
    echo "=== Resolving triples ==="
    $PYTHON scripts/stage_b_probe.py --resolve-only \
        --triples-path "$TRIPLES" \
        --out-dir "$OUT_DIR"
fi
test -f "$RESOLVED" || { echo "FAIL: resolved_triples.json not generated"; exit 96; }

# ── Step 2: Extract hidden states ──
# Strategy: 2 GPUs, round-robin model×format jobs (9 total)
# GPU 0: coder/svg, coder/asy, viscoder2/tikz, qwen25/svg, qwen25/asy
# GPU 1: coder/tikz, viscoder2/svg, viscoder2/asy, qwen25/tikz
# Each model loads once per format (~2min), processes 252 triples (~3-5min)

MODELS="coder viscoder2 qwen25"
FORMATS="svg tikz asy"

for model in $MODELS; do
    gpu_idx=0
    for fmt in $FORMATS; do
        echo "=== $model / $fmt on gpu $gpu_idx ==="
        $PYTHON scripts/stage_b_probe.py \
            --model "$model" \
            --format "$fmt" \
            --gpu "$gpu_idx" \
            --triples-path "$TRIPLES" \
            --out-dir "$OUT_DIR" &

        # Alternate GPUs within each model (2 parallel per model)
        gpu_idx=$(( (gpu_idx + 1) % 2 ))
    done
    # Wait for all 3 formats of this model to finish before loading next model
    # (to avoid OOM from 2 models on same GPU)
    wait
    echo "=== $model complete ==="
done

echo "=== All Stage B extractions complete ==="

# ── Verify ──
echo "=== Verification ==="
for model in $MODELS; do
    for fmt in $FORMATS; do
        n=$(find "$OUT_DIR/$model/$fmt" -name "*.pt" 2>/dev/null | wc -l)
        summary="$OUT_DIR/$model/$fmt/summary.json"
        if [ -f "$summary" ]; then
            echo "$model/$fmt: $n .pt files, summary exists"
        else
            echo "WARNING: $model/$fmt: $n .pt files, NO summary"
        fi
    done
done
