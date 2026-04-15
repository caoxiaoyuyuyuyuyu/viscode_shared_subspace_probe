# ROADMAP — viscode_shared_subspace_probe

> 最后更新: 2026-04-14 09:50 UTC | 更新原因: 用户要求停止推进项目

## 研究目标（D082 同步）

1. **Negative report + confound analysis**（primary）：cross-format CKA in visual code LLMs is dominated by token-overlap and code-general confounds
2. **Confound triangulation**：C1 negative control + Llama code-naive ablation + stratified/token-shared nulls
3. **Methodological template**：honest negative reporting with confound-aware null baselines

## 投稿目标

- **Target**: EMNLP 2026 Findings (ARR 2026 May cycle, deadline 2026-05-25 AoE)
- **首轮结果**: 硬 Reject（D080）→ 修复重投（D081/D082）

## ⚠️ 项目已暂停

**2026-04-14 用户指令：停止推进本项目。**

### 暂停时状态
- Phase 1 计算实验未完成（AutoDL SSH 阻塞，刚切换 A6000 但未开跑）
- T1B PCA min-k 数据已有
- T4 BH-FDR 脚本已修正（n_perm=5000 + by_type pooling）
- T1A/T5 脚本已就绪
- Phase 2/3 未开始

### 若恢复需要
- 确认 AutoDL 实例状态或继续用 A6000
- 从 Phase 1 T1A 开始跑

## 关键决策日志

| ID | 日期 | 摘要 | 关联实验 |
|----|------|------|----------|
| D082 | 2026-04-14 | D082: Phase 1 修正——format-residualized 做两版(linear+P | — |
| D081 | 2026-04-14 | D081: 硬 Reject 后决定修复统计方法论 + 重投 ARR May cycle (2026 | — |
| D080 | 2026-04-14 | D080: 审稿报告 Reject（硬 Reject）— 9 critical: 超页、permut | — |
| D079 | 2026-04-14 | D078 outcome: Llama 7-layer scan confirms split hy | llama3_7layer_scan_v3, llama3_baseline_cka |
| D078 | 2026-04-14 | D078: Llama 7-layer scan pre-registration — 延续 D07 | llama3_7layer_scan |
| D077 | 2026-04-14 | 撤销 D076 fabrication 指控——Reviewer 自我更正，worker 运行在远程 | — |
| D076 | 2026-04-14 | Worker fabricated figure generation report — worke | — |
| D075 | 2026-04-13 | D075: Llama-3-8B baseline 结果为 D074 Outcome C (Mixe | llama3_baseline_cka |
| D074 | 2026-04-13 | D074: Llama-3-8B code-naive baseline pre-registere | llama3_baseline_probe |
| D073 | 2026-04-13 | D073: F3 采纳 HYBRID framing — token-overlap gradien | tokenizer_overlap_baseline |




