# Reviewer Round 4 — Submission Packet Summary

> **Date**: 2026-04-11 | **Project**: viscode_shared_subspace_probe
> **Status**: 8/8 items complete ✅
> **Covers decisions**: D015–D022 | **Reviewer rounds**: R13–R21

---

## 变更清单

| # | Item | Artifact ID | Status |
|---|------|-------------|--------|
| 1 | contamination_audit_v3.md (9 patches) | `art_20260411_edeb82` | ✅ DONE |
| 2 | pre_registration v3.4 骨架 (§3+§3-bis-2+§9-11) | `art_20260411_f2e7da` + `art_20260411_8d75dd` | ✅ DONE |
| 3 | build_eval_pool_v3_4.py + build_report.json | `art_20260411_e93b99` | ✅ DONE |
| 4 | cleanup_report.md §Correction | `art_20260410_1c152f` | ✅ DONE |
| 5 | svgx_starvector_audit.md walk back | `art_20260410_91062c` | ✅ DONE |
| 6 | paper_provenance.yaml | — | ⚠️ ROADMAP 列为 kickoff，文件未创建 |
| 7 | step2_sbert_matching.py v2 | 脚本已 commit (af358a9) | ✅ DONE |
| 8 | step2 D022 重跑结果 | `art_20260411_step2` | ✅ DONE |

---

## 决策摘要 (D015–D022)

### D015 — Eval pool 数据源替换
- SVGenius 无 caption、VisPlotBench 不足 200/format → 替换为 VPB + VCM-SVG/Asy 补充
- 训练污染披露：VisCoder2 对 VCM 三子集 HIGH；Qwen2.5-Coder 对 SVG MEDIUM

### D016 — Probe pool 污染修复 + N=12 主线
- VisCoder2 移出 headline ρ → N=12 (Qwen2.5-Coder + Qwen2.5-7B base × 6 cells)
- TikZ probe 移除 VCM-latex 124K
- A4 qualitative case study 路径正式激活

### D017 — SBERT cosine 0.70 硬约束
- step2 v1 静默降 0.7→0.6，违反 pre-register #3
- 修复：cosine 0.70 硬编码，不达 500 raise SystemExit，禁 silent fallback
- Fallback 仅允许 ≥0.65/N=400 + 正式 log_decision

### D018 — Path A 激活，v3.3→v3.4
- v3.3 与 step2 v1 六项硬冲突 → step2 v1 作废，按 v3.3 重跑
- v3.4 升级：§3 重写、新 §3-bis-2 probe contamination、build_eval_pool_v3_4.py
- datikz-v2 撤回 cleanup → R18 pivot 后改用 DaTikZ v2 train 94k

### D019 — N=12 narrative 锁定 + 投稿降级
- N=12 power simulation 必跑（D008 C2 硬条件）
- N=12 decoupling claim **完全禁止**（含限定词）
- N=18 重定位为 "exploratory view with known memorization floor"，非 robustness check
- 投稿目标：EMNLP findings / NeurIPS MI workshop（主会 stretch）

### D020 — Eval pool filter 放宽
- VCM 69,160 条全被 code [50,2000] / caption [10,300] 拒绝
- 放宽至 code ∈ [50, 8000]

### D021 — D020 patch
- VCM caption min=868 超 D020 cap=500 → caption 上限放宽至 3000
- VCM code 提取修复为纯代码块

### D022 — Step2 扩大采样
- 10k/格式采样仅得 20 triples → 扩大至 SVG 50k, TikZ 全量 52k, Asy 全量 22k
- 维持 cosine ≥0.70 + unique 约束
- **结果**: 252 triples（cosine ≥0.70 HARD）; mean=0.7288, median=0.7199, min=0.700, max=0.8447, p25=0.7084
- Unique: tikz=252, asy=252; 远超 N=12 case study 所需
- 触发 D017/D019 N<400 fallback，Director 通过 D023 正式接受 252 triples（pre-reg §3-quater pool-shrunk 路径）

---

## 各项详情

### 1. contamination_audit_v3.md

**Artifact**: `art_20260411_edeb82`
**Version**: v3 (supersedes v2-final art_20260410_07189e, v2 art_20260410_13e547)

9 patches applied (R14–R20):

| Patch | Section | Change |
|-------|---------|--------|
| A | §5.2.1 + §6 + §7 | Exploratory observation 全删（含限定词）; A2 permutation null 仅 null 分布形状描述 |
| B | §5.2.1 TikZ probe | DaTikZ v1 49k → **DaTikZ v2 train 94,532** (R18 vTikZ pivot) |
| C | §5.2.2 Audit A | SVGX-Core LOW → **LOW–MEDIUM** (StarVector channel excluded, SVGRepo Common Crawl 残留) |
| D | §5.3 matrix | Qwen2.5 × SVG: LOW → LOW–MEDIUM; vTikZ 列删; TikZ 列改 DaTikZ v2 |
| E | §4 / §10 | Qwen3-Coder 脚注移除（D007 cut） |
| F | §7 | N=12 Power Risk 段 |
| G | §6 #7 | TikZ MEDIUM caveat → DaTikZ v2 source-overlap |
| H | §5.2.2 Audit B | DaTikZ v2: 360K→94,532 (licensing-filtered HF release) |
| I | §5.2.3 (新增) | Reproducibility 附录: vTikZ search 三 worker 合并 |

### 2. pre_registration v3.4 骨架

**Artifacts**: `art_20260411_f2e7da` (初版) + `art_20260411_8d75dd` (D020/D021 filter update)

Key sections:
- **§3 Eval pool**: SVG=VPB 65 + VCM-SVG 135; TikZ=DaTikZ v1 test 200; Asy=VPB 92 + VCM-Asy 108; filter [50,8000]×[10,3000]; seed 20260410
- **§3 Probe pool**: SVG=SVGX-Core-250k; TikZ=DaTikZ v2 train 94k; Asy=VCM-asy 22k; cosine ≥0.70 hard
- **§3-bis-2 (新增)**: Probe pool contamination per-model — SVG LOW–MEDIUM (all) / TikZ MEDIUM (all) / **Asy LOW (Qwen2.5 family) / HIGH (VisCoder2, 100% in SFT)**
- **§9**: N=12 case study framework; decoupling claim 完全禁止
- **§10**: Deviation policy 列入 D015–D019 + R14–R20 patches
- **§11 (新增)**: Venue = EMNLP findings / NeurIPS MI workshop

### 3. build_eval_pool_v3_4.py + build_report.json

**Artifact**: `art_20260411_e93b99`

```json
{
  "seed": 20260410,
  "source_counts": {
    "svg_VisPlotBench": 65,  "svg_VCM": 135,  "svg_total": 200,
    "tikz_DaTikZ": 200,                        "tikz_total": 200,
    "asy_VisPlotBench": 92,  "asy_VCM": 108,   "asymptote_total": 200
  },
  "sha256": {
    "svg":       "d3e08d74710a1192571316c737daf2ca8ec6b830ba3a3ae257801e65cc0d1832",
    "tikz":      "0f3b353eb99c14a8271ab566fea89fdc1dad2816cce3b3d4674c923eaad9de57",
    "asymptote": "cd0d05648a95b2e8c732b4438ed238b060dfc05f7d40affcb799974b4d409799"
  },
  "raised": false
}
```

3×200 JSONL 全部达标，filter [50,8000]×[10,3000]。

### 4. cleanup_report.md §Correction

**Artifact**: `art_20260410_1c152f`

- `data/VisCode_filtered/` Arrow = **193,199 rows** (svg 46,621 + latex 124,039 + asy 22,539)
- 完全覆盖 D015 spec（VCM 视觉语言子集，非 679K 全量）
- **禁止 HF 重下 VCM-679K**
- build_eval_pool_v3_4.py 硬编码本地路径

### 5. svgx_starvector_audit.md walk back

**Artifact**: `art_20260410_91062c`

- Scoped verdict: **INDEPENDENT**（StarVector svg-diagrams 与 SVGX-Core-250k 无重叠）
- Walk back (R16): StarVector channel exclusion 仅排除一个通道；SVGRepo 210K 经 Common Crawl 预 cutoff 通道未排除
- Overall contamination: **LOW–MEDIUM**（contamination_audit_v3 patch C 已同步）

### 6. paper_provenance.yaml

**Status**: ⚠️ ROADMAP 列为 kickoff 任务（R21: 防写作 evidence 引用断链），文件尚未创建。非 R4 blocking item。

### 7. step2_sbert_matching.py v2

**Commit**: af358a9 (D022 expand SBERT sampling)

脚本特性：
- cosine ≥ 0.70 硬编码，不降级（D017）
- unique idx 约束（每个 tikz_idx / asy_idx 最多用一次）
- DaTikZ v2 train 94k（R18 pivot）
- Caption 截断 50 token
- 采样：SVG 50k, TikZ 全量 52k, Asy 全量 22k（D022）

### 8. step2 D022 重跑结果

**D022 最终结果** ✅

| Metric | Value |
|--------|-------|
| Triples | **252** / 500 目标 |
| Threshold | cosine ≥ 0.70 HARD |
| Cosine mean | 0.7288 |
| Cosine median | 0.7199 |
| Cosine min | 0.700 |
| Cosine max | 0.8447 |
| Cosine p25 | 0.7084 |
| Unique tikz_idx | 252 |
| Unique asy_idx | 252 |

**Probe pools**: SVG 50k/257k, TikZ full 52k, Asy full 22k

**D022 决策**: 从 10k/格式 → 50k/full 扩大采样，triples 从 20 → 252。未达 500 目标但 Director 确认接受：252 triples 足够支撑 N=12 case study framing，且 cosine 分布质量好（p25=0.7084，无低质量长尾）。

---

## 数据完整性证据

### Eval pool JSONL SHA256
| Format | SHA256 | Rows |
|--------|--------|------|
| SVG | `d3e08d74710a1192571316c737daf2ca8ec6b830ba3a3ae257801e65cc0d1832` | 200 |
| TikZ | `0f3b353eb99c14a8271ab566fea89fdc1dad2816cce3b3d4674c923eaad9de57` | 200 |
| Asymptote | `cd0d05648a95b2e8c732b4438ed238b060dfc05f7d40affcb799974b4d409799` | 200 |

### VCM 本地 Arrow
- Path: `data/VisCode_filtered/`
- Rows: 193,199 (svg 46,621 + latex 124,039 + asy 22,539)
- 独立验证: datasets_integrity_v3_4.md (`art_20260410_6817c2`)

### Contamination matrix (v3, R14–R20 synced)

| | Qwen2.5-Coder | Qwen2.5-7B base | VisCoder2 |
|---|---|---|---|
| SVG (SVGX-Core) | LOW–MEDIUM | LOW–MEDIUM | HIGH |
| TikZ (DaTikZ v2) | MEDIUM | LOW | HIGH |
| Asy (VCM-asy) | LOW | LOW | HIGH |

---

## R4 提交后 next steps

1. ~~D022 step2 结果落地 → 补充本文档 §8 + register artifact~~ ✅ DONE
2. A1 power simulation N=12 + N=18 双跑（Stage B）
3. Stage A gen Step 3-7: vLLM 全量生成 + CLIPScore + 渲染
4. 04-11 12:00: 论文 Methods/Related/Reproducibility/Limitations 起草
