# F Audit: probe.py vs cka.py layer coverage

**Case 1 — 完全一致**（Director 2026-04-13 裁决，Reviewer 独立核实）

## 证据
- `scripts/negative_control_probe.py:33`: `LAYERS = [4, 8, 12, 16, 20, 24, 28]`（7 点）
- `scripts/negative_control_cka.py` (grep `^LAYERS =` 或 `LAYERS = \[4,`): `LAYERS = [4, 8, 12, 16, 20, 24, 28]`（7 点）
- 两脚本对 Qwen2.5-Coder 28 层模型的采样完全一致

## 结论
- 无语义分歧，无 bug
- 7 = `len(LAYERS)` 抽样子集大小；28 = `LAYERS[-1]` 子集最大 index，并非 `num_hidden_layers`
- 旧 CKPT 打印 `layer={layer}/{LAYERS[-1]}` 曾被误读为 "4 of 28 total"；已由 commit `e34b4bf` 的 cosmetic 修复消歧（`layer_idx={layer} li={li}/{len(LAYERS)-1}`）

## Cosmetic 修复
- commit `e34b4bf` 修改 CKPT 日志显示格式（不影响正确性）
- commit `23f591b` 后续增补 CKPT version=2 header 格式

## C1 覆盖度影响
**无影响**。v5 python-neg 按 7 层子集运行，C1 结论使用同一 `LAYERS` 索引，不存在漏算。
