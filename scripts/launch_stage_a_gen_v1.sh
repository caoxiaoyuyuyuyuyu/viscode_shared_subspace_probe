#!/bin/bash
set -e
cd /root/autodl-tmp/viscode_shared_subspace_probe
source venv/bin/activate
export HF_HUB_OFFLINE=1
LOG_DIR=/root/autodl-tmp/viscode_shared_subspace_probe/artifacts/stage_a/logs/gen_v1
OUT_BASE=/root/autodl-tmp/viscode_shared_subspace_probe/artifacts/stage_a/gen_v1
echo "[launcher] start $(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD)" >> $LOG_DIR/master.log
for MODEL in viscoder2 coder qwen25; do
  mkdir -p $OUT_BASE/$MODEL
  for FMT in svg tikz asy; do
    echo "[launcher] === $MODEL $FMT begin $(date -u +%FT%TZ) ===" >> $LOG_DIR/master.log
    python scripts/stage_a_gen.py --model $MODEL --format $FMT --shard-id 0 --num-shards 2 --gpu 0 --n-shots 2 --out-dir $OUT_BASE/$MODEL > $LOG_DIR/${MODEL}_${FMT}_s0.log 2>&1 &
    PID0=$!
    python scripts/stage_a_gen.py --model $MODEL --format $FMT --shard-id 1 --num-shards 2 --gpu 1 --n-shots 2 --out-dir $OUT_BASE/$MODEL > $LOG_DIR/${MODEL}_${FMT}_s1.log 2>&1 &
    PID1=$!
    echo "$(date -u +%FT%TZ) $MODEL $FMT shard0=$PID0 shard1=$PID1" >> $LOG_DIR/pids.txt
    wait $PID0 $PID1
    echo "[launcher] === $MODEL $FMT done $(date -u +%FT%TZ) ===" >> $LOG_DIR/master.log
  done
done
echo "[launcher] ALL DONE $(date -u +%FT%TZ)" >> $LOG_DIR/master.log
