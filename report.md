# Handover — 2026-04-15

> 由 Agent 生成，每次状态变更后自动更新。硬限 150 行。

## 运行中
（无）

## 待处理
- exp_006_eval_d031_v3_dpo_N8（准备中）
- exp_006_eval_d031_v3_dpo_N4（准备中）
- exp_006_eval_d031_v3_dpo_N2（准备中）
- d081_bh_fdr_correction: D081 Task 4: BH-FDR global correction. Collect all per-pair per-layer permutation p-values into one pool, apply Benjamini-Hochberg FDR at α=0.05. Replaces asymmetric correction approach.（准备中）
- d081_procrustes_k_sensitivity: D081 Task 3: Procrustes k sensitivity. Rerun Procrustes alignment at k=20,50,100,200 to verify robustness of conclusions to PCA dimensionality choice.（准备中）
- d081_stratified_permutation: D081 Task 2: Stratified permutation null using SBERT caption clusters. Permute format labels within semantic clusters to control for visual semantics. Tests whether cross-format CKA exceeds a semantics-controlled null.（准备中）
- d081_format_mean_residual_cka: D081 Task 1: Format-mean residualized CKA. Subtract per-format mean from hidden states, recompute CKA + 1000-perm test. Tests whether cross-format signal survives removal of format-specific mean activation.（准备中）
- v3_recompute_pid4356（准备中）
- p1_python_neg_pid9097（准备中）
- step2a_recompute_pid6497（准备中）
- …另有 3 项

## 近期结论
- **stage_a_gen** (killed): Stage A vLLM gen 全量：3 models × 3 formats × 200 prompts × n_shots=2 = 3600 sample。2-GPU data-parallel (prompt_id 切分)。新 stage_a_gen.py 脚本，T=0.3, top_p=default(1.0), per-format max_tokens (SVG/TikZ=1024, Asy=2048), stop tokens 来自 p3_coder_rerun.py，SamplingParams n=2 共享 prefix KV，双 seed 防退化。 ← 基于 stage_a_vllm
- **llama3_7layer_scan_v3** (success): Llama-3-8B 7-layer full scan (v3). Manifest cleared + code fix for .pt validation. Clean extraction from scratch. ← 基于 llama3_7layer_scan_v2
- **llama3_7layer_scan_v2** (failed): Llama-3-8B 7-layer full scan (v2, with checkpoint validation fix). Same as llama3_7layer_scan but with code fix: load_extract_done now validates .pt files exist before skipping. ← 基于 llama3_7layer_scan

## 趋势
成功 21 / 否定 0 / 失败 12 / 共 36

## 下一步
（无活跃 idea，请通过 ideas.md 添加）

## 资源状态
autodl-viscode-probe: 空闲 cuda:0, cuda:1

## 告警
- stage0_env_setup: 训练失败 — unknown
- stage0_dual_stack_sanity: 训练失败 — unknown
- stage_a_gen: 已被终止 — unknown
- stage_a_gen_v2: 训练失败 — unknown
- stage_b_probe: 训练失败 — unknown
- stage_b_probe_v2: 训练失败 — unknown
- stage_a_gen_multimodel: 已被终止 — unknown
- stage_a_gen_multimodel_v2: 训练失败 — unknown
- stage_a_gen_multimodel_v3: 训练失败 — unknown
- stage_a_gen_multimodel_v4: 训练失败 — unknown
- format_residualized_cka_full: 已被终止 — unknown
- p1_python_neg_pid4048_v3: 训练失败 — unknown
- v3_neg_pid4048: 训练失败 — unknown
- llama3_7layer_scan: 训练失败 — unknown
- llama3_7layer_scan_v2: 训练失败 — unknown
