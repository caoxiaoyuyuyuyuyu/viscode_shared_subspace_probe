# Pre-Registration — viscode_shared_subspace_probe

> **Version**: v3.4 | Locked: 2026-04-10 | Status: pending Director approval
> Any deviation from §1–§12 requires an explicit `log_decision` entry with rationale **before** any post-hoc analysis. Stage A pilot may not begin until this document is approved.
>
> **v3.4 changelog vs v3.3** (D016 N=12 pivot, D018 Path A, D019 decoupling ban, R18 TikZ probe pivot, R21 骨架先于 P4):
> - **§1 headline statistic**: N=24 → **N=12** (2 models × 6 format-pairs). N=18 (3 models × 6 format-pairs, including VisCoder2) demoted to auxiliary exploratory view. Decoupling claim **completely prohibited** at N=12 (D019 C2).
> - **§2 model pool**: Qwen3-Coder and LLM4SVG removed (D007 disk constraint). 3 models: Qwen2.5-Coder-7B-Instruct, Qwen2.5-7B base, VisCoder2-7B. VisCoder2 excluded from headline ρ (D016 contamination).
> - **§3 eval pool build script**: `build_eval_pool_v3_3.py` → `build_eval_pool_v3_4.py`. VCM data source = `data/VisCode_filtered/` local Arrow (193,199 rows, D015 spec FULL, **禁 HF 重下**).
> - **§3 probe pool TikZ**: vTikZ-train + DaTikZv2-train → **DaTikZ v2 train 94,532 rows only** (R18 pivot: vTikZ benchmark insufficient for probe pool). Fail-safe = DaTikZ v1 train 49k.
> - **§3-bis**: updated for 3-model pool (Qwen3-Coder and LLM4SVG rows removed).
> - **§3-bis-2 (new)**: probe pool contamination disclosure (R18 + D016 + D018). Per-model: SVG = LOW–MEDIUM (all); TikZ = MEDIUM (all); **Asy = LOW (Qwen2.5 family) / HIGH (VisCoder2, 100% in SFT)**. Explicitly disclosed.
> - **§6 Stage E**: removed entirely (D009 A7 — A4 activation cascades to Stage E skip; 8h returned to paper writing).
> - **§9 backup narrative**: rewritten for N=12 case study + framework framing. Decoupling claim banned in all regions (D019). N=18 repositioned as "exploratory view with known memorization floor" (not robustness check).
> - **§10 deviation policy**: added N=12 decoupling claim prohibition; N=18 must not be framed as N=12 robustness check.
> - **§11 statistical methods**: A1 power simulation N=12 + N=18 dual run; A2 permutation null = descriptive statistics of null distribution shape only; decoupling claim prohibited.
> - **No change** to: §3-quater (cosine ≥ 0.7), §4 CLIPScore gate, §5 attribute battery, §7 overhead, §8 budget (adjusted for 3 models + no Stage E), §12 tool-stack split.
> - Total locked items: **12** (was 11 in v3.3; +1 = §3-bis-2 probe pool contamination disclosure).
>
> **v3.3 changelog vs v3.2** (D015 Stage A eval-pool supplementation, contamination-aware):
> - §3 Eval pool source rewritten; §3-bis (new) Eval-pool training contamination disclosure; §9 VisCoder2 memorization-floor caveat; §10 deviation policy extended. See v3.3 header for details.
>
> **v3.2 changelog vs v3.1**: §1 HARKing-avoidance rule; §3 sbert model; §3-quater quality gate; §4 CLIPScore SNR gate; §5 attribute battery; §9 backup narrative regions.
>
> **v3.1 changelog vs v3**: Stage A vLLM + Stage B HF TF split; probe pool 500; mean-pool code tokens; Qwen2.5-7B base rehabilitated; tool-stack split.
>
> **v3.3 layer-set revision (2026-04-11, D026)**: LAYERS [8,16,20,24,28,32] → [4,8,12,16,20,24,28]. 因 Qwen2.5-7B / Qwen2.5-Coder-7B / VisCoder2-7B 均为 n_layers=28，原 spec layer 32 OOB 触发 silent clamp 到最后一层致 row 4==row 5 数据按位损坏。修订 rationale: 7 层等距对 28-layer 模型均匀覆盖 early(4,8,12)/mid(16,20)/late(24,28)，保持 7B 三模型受控对照 backbone 不变。Reviewer CRITICAL 独立定位 2026-04-11。

## 1. Headline statistical claim

- **Test**: Spearman ρ between cross-format **probe-transfer accuracy** (Stage B geometry) and **behavioral cross-format transfer** (Stage A).
- **Unit of analysis**: one (model, ordered format-pair) cell.
- **Headline N = 12** = 2 models (Qwen2.5-Coder-7B-Instruct, Qwen2.5-7B base) × 6 ordered format-pairs.
  - Format-pairs: SVG→TikZ, TikZ→SVG, SVG→Asymptote, Asymptote→SVG, TikZ→Asymptote, Asymptote→TikZ.
  - VisCoder2-7B excluded from headline ρ due to probe pool contamination (D016, §3-bis-2).
- **Auxiliary N = 18** = 3 models (+ VisCoder2-7B) × 6 format-pairs. Reported as **"exploratory view with known memorization floor on Asy/TikZ for VisCoder2"** (D019 W3). **Not** a robustness check for N=12.
- **Per-cell geometry score**: `max`-over-layer probe-transfer accuracy across layers {4,8,12,16,20,24,28}. One scalar per cell — layers are **not** stacked into N.
- **α = 0.05**; significance via **bootstrap 1000×** 95% CI, resampling the 12 (model, pair) cells.
- **Secondary (descriptive only, no p-value)**: per-layer ρ curves over the same 12 cells.
- **Forbidden**: collapsing 7 layers × 12 cells → 84 points (pseudo-replication).
- **N=12 decoupling claim prohibition (D019 C2)**: at N=12, bootstrap CI is extremely wide and permutation null variance is large — any "decoupling" finding would be a statistical artifact of insufficient power, not an empirical observation. **No decoupling claim of any kind is permitted at N=12**, including with hedging qualifiers (exploratory / preliminary / suggestive / qualitative). This prohibition is equivalent to a `power ≥ 0.5` hard gate on the decoupling claim pathway.
- **Pre-registered narrative rule**: the paper writes a **framework + Qwen2.5 family case study** narrative (§9). The headline ρ result selects the matching §9 Region outline. No post-hoc re-framing is permitted.

## 2. Model pool (v3.4 — 3 models, D007 + D016)

| Model | Role | In headline ρ (N=12)? | In auxiliary (N=18)? |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | code LLM | **yes** | yes |
| Qwen2.5-7B (base) | text-only control — rules out "visual-agnostic shared Qwen text subspace" null. Valid because Stage B uses teacher-forcing. | **yes** | yes |
| VisCoder2-7B | multi-format visual SFT; memorization-floor reference | **NO** (D016 contamination) | yes (exploratory, known contamination floor) |

**Removed (D007)**:
- Qwen3-Coder-7B: no 7B variant available; 30B-A3B exceeds disk budget.
- LLM4SVG-7B: snapshot zero weights; single-format specialist dropped.

## 3. Eval Pool vs Probe Pool — strict disjoint partition + tool-stack split (v3.4)

Two **independent generation passes on two independent tool stacks on two independent pools**.

### Eval pool (Stage A behavioral, **vLLM auto-regressive generation**)

- **Size**: 200 captions × 3 formats = 600 eval captions.
- **Per caption**: × 2 shot regimes (0-shot, 3-shot) × 3 models → 3 600 free generations total.
- **Outputs**: generated code → format-validity (binary) + render → hybrid eval (§4, D014).
- **Hidden states are NOT used from this pass.**

**Sources (locked v3.4)**:

| Format | Source mix | Per-source count | Total |
|---|---|---:|---:|
| SVG | VisPlotBench held-out test split | 65 | 200 |
| | VCM-SVG subset (from `data/VisCode_filtered/` local Arrow, random sample) | 135 | |
| TikZ | DaTikZ v1 test split (random sample) | 200 | 200 |
| Asymptote | VisPlotBench held-out test split | 92 | 200 |
| | VCM-Asy subset (from `data/VisCode_filtered/` local Arrow, random sample) | 108 | |

**VCM data source (locked, D015 + D018)**:
- Local Arrow dataset at `data/VisCode_filtered/` — **193,199 rows** (svg 46,621 + latex 124,039 + asy 22,539) = D015 spec FULL coverage.
- **禁止 HF 重下**. This is the sole authoritative copy; integrity verified by `datasets_integrity_v3_4.md`.

**Locked filter rules** (applied identically to all sources before random sampling):

1. Code length ∈ [50, 2000] characters (excludes one-liners and bloat).
2. Render-success required: cairosvg / pdflatex (with `tikz`) / asy must produce a valid PNG within 30 s wall-clock.
3. Caption / instruction length ∈ [10, 300] characters; instructions trimmed to the first user turn.
4. Normalized-code-hash dedup within each per-format pool **and** across the SVG-pool / Asy-pool merge boundary.
5. Random sample with `numpy.random.default_rng(seed=20260410)` after filtering, in that order.

**Build script (locked)**: `scripts/build_eval_pool_v3_4.py`. Outputs `artifacts/stage_a/eval_pool/v3_4/{svg,tikz,asymptote}.jsonl`. The resulting JSONL SHA256 hashes are recorded in `decisions.yaml` before any Stage A generation runs.

- SHA256 (SVG): [SHA256: TBD by build_eval_pool_v3_4.py]
- SHA256 (TikZ): [SHA256: TBD by build_eval_pool_v3_4.py]
- SHA256 (Asymptote): [SHA256: TBD by build_eval_pool_v3_4.py]

### §3-bis. Eval-pool training contamination disclosure (v3.4)

VisCode-Multi-679K is the SFT corpus of VisCoder2-7B. Using it as eval material against VisCoder2 measures memorization, not generalization. Disclosed up-front:

| Model ↓ / Subset → | SVG (mostly StarVector svg-diagrams) | LaTeX/TikZ (CoSyn-400K) | Asymptote (CoSyn-400K) |
|---|---|---|---|
| Qwen2.5-7B base | LOW | LOW | LOW |
| Qwen2.5-Coder-7B-Instruct | MEDIUM | LOW | LOW |
| **VisCoder2-7B** | **HIGH** | **HIGH** | **HIGH** |

**Reasoning** (full audit in `artifacts/_project/report/contamination_audit_v3.md`): Qwen2.5-Coder pretraining cuts off at Feb 2024 (GitHub data); CoSyn-400K was released Feb 2025 → LOW for the LaTeX/Asy subsets. StarVector svg-diagrams (Dec 2023, on HF) could have been swept via Common Crawl → MEDIUM for SVG. VisCoder2 was directly fine-tuned on VisCode-Multi-679K → HIGH for all three subsets.

**Mitigations** (locked):

1. **Disclosure footnote**: every Results table reporting VisCoder2 absolute scores carries a "potential memorization floor" footnote citing this section.
2. **Visual identification**: headline ρ scatter marks VisCoder2's 6 cells with a distinct symbol so reviewers can see the cluster.
3. **N=18 exploratory appendix**: reports ρ with VisCoder2 included as "exploratory view with known memorization floor" — **not** as a robustness check for the N=12 headline.
4. **§9 narrative anticipation**: all Region outlines anticipate VisCoder2 sitting at a memorization-inflated corner; the narrative does not rest on VisCoder2's absolute placement.
5. **No source swap**: replacing the SVG/Asy shortfall with a different corpus was considered and rejected (D015 rationale).

### §3-bis-2. Probe-pool training contamination disclosure (v3.4, new — D016 + D018 + R18)

Probe pool contamination is distinct from eval pool contamination (§3-bis). v3.3 §3-bis only covered eval pool; this section fills the gap identified by D018 and contamination_audit_v3.md §5.2.

| Format | Probe Source | Qwen2.5 family (N=12) | VisCoder2 (N=18 aux) |
|---|---|---|---|
| SVG | SVGX-Core-250k | LOW–MEDIUM | LOW–MEDIUM |
| TikZ | DaTikZ v2 train 94k (primary); DaTikZ v1 train 49k (fail-safe) | MEDIUM | MEDIUM |
| Asy | VCM-asy 22k | **LOW** (VCM post-cutoff, no SFT) | **HIGH** (100% in SFT) |

**Key implications**:

1. **VisCoder2 × Asy probe = memorization, not generalization**: VCM-asy is literally VisCoder2's SFT training data. Stage B probe-transfer accuracy involving VisCoder2 on Asy-related cells reflects memorized representations.
2. **VisCoder2 excluded from headline ρ (N=12)**: D016 removes VisCoder2's 6 cells from the headline statistic. This is the primary mitigation.
3. **N=18 auxiliary includes VisCoder2 with explicit caveat**: VisCoder2's cells are reported in N=18 as "exploratory view with known memorization floor on Asy/TikZ," not as a robustness check.
4. **N=12 vs N=18 ρ difference as contamination evidence**: the Discussion uses the delta between N=12 (clean models only) and N=18 (+ contaminated VisCoder2) as direct quantitative evidence of contamination effect on probe geometry.
5. **Asy probe pool = VCM-asy is disclosed, not hidden**: no independent Asymptote training corpus exists within the 48h budget. This limitation is reported in Limitations.

### Probe pool (Stage B hidden state, **HF transformers + teacher-forcing**)

- **Source (v3.4, R18 pivot)**:
  - SVG: SVGX-Core-250k (contamination: LOW–MEDIUM)
  - TikZ: **DaTikZ v2 train 94,532 rows** (R18 pivot from vTikZ benchmark; fail-safe DaTikZ v1 train 49k)
  - Asy: VCM-asy 22k (contamination: HIGH for VisCoder2, LOW for others; explicitly disclosed §3-bis-2)
- **Cross-format matching**: embed all candidate captions with `sentence-transformers/all-MiniLM-L6-v2`; greedy match into (svg_caption, tikz_caption, asymptote_caption) triples maximizing average pairwise cosine; keep top **500 triples** with min-pairwise cosine **≥ 0.7** (see §3-quater). Caption truncation: first **50 tokens** (D017 normalization fix).
- **Probe sample**: each triple yields 3 (caption, reference_code) pairs (one per format) → **500 × 3 = 1 500 pairs per model**, × 3 models = **4 500 single teacher-forced forwards** total.
- **Forward path**: prompt = caption (chat template); teacher-force the reference code tokens; capture residual stream **at code-token positions only**.
- **Aggregation**: **mean-pool over code-token positions** at each tracked layer ∈ {4, 8, 12, 16, 20, 24, 28}. One vector per (model, layer, sample).
- **Saved tensors**: `/root/autodl-tmp/cache/hidden_states/{model}/{format}/{triple_id}.pt` — shape `[7_layers, hidden_dim]` per sample.

### §3-quater. Cross-format triple-match quality gate (v3.2, unchanged)

Sentence-BERT cosine high ⇏ same visual concept. The three datasets have very different stylistic priors.

**Locked threshold**: triple-match cosine **≥ 0.7** (D017 hard gate). Triples below threshold are discarded; greedy matching continues until 500 valid triples are accumulated. If fewer than 500 triples meet the threshold from the available corpus, the pool is shrunk and recorded in `decisions.yaml` rather than the threshold lowered.

**Caption truncation (D017 fix)**: all captions truncated to first **50 tokens** by sbert tokenizer before embedding, normalizing for caption length heterogeneity across corpora.

**Fallback protocol (D017)**:
- cosine ≥ 0.70 hard gate; silent fallback **prohibited** — script must `raise SystemExit` if threshold not met.
- If 50k sample + caption normalization still < 500 triples ≥ 0.70: shrink pool (not lower threshold), log_decision required.
- Allowed fallback: ≥ 0.65 / N=400, but **must** formal `log_decision` + Limitations disclosure.
- N < 400 → escalate to Director as critical.

**Mandatory pilot human-eye sanity check**: from the 500 matched triples, sample **10 at random**, render the reference code in all three formats to PNG, human inspects each triple-row. **Pass criterion**: ≥ 8 / 10 triples judged concept-equivalent.

- **Pass** → open Stage B full run.
- **Fail (< 8 / 10)** → raise cosine threshold to **0.75**, re-match, repeat sanity.
- **Alternative escape (if 0.75 also fails)** → caption-template approach (logged deviation).

### Disjointness

- **eval pool ∩ probe pool = ∅** by construction (test-split vs train/dev-split boundary, plus the triple-match filter never re-uses test-split captions).
- **Strictly forbidden**: (a) eval pool and probe pool sharing any caption; (b) Stage B using free generation to harvest last-token residuals; (c) merging eval and probe passes into a single forward.

### Why teacher-forcing is the methodologically correct choice

Following standard representation-probing practice (Hewitt & Manning 2019; Belinkov 2022), free-generation hidden states confound **generation ability** with **representation content**. Teacher-forcing fixes the token sequence to a known-good reference, isolating "what the model encodes when it sees this concept in this format" from "whether the model can chain valid output tokens." This makes Qwen2.5-7B base a fair text-only control.

## 4. Behavioral metrics (two-tier, locked — v3.4 hybrid eval, D014)

- (i) **Format validity** — binary; code parses + renders without error/timeout (timeout = 30 s/sample). Failure → 0.
- (ii) **Content fidelity** — **hybrid eval** per D014 CLIPScore SNR results:
  - Color / Shape_count / Stroke_fill → **CLIPScore (ViT-L/14)** (SNR ≥ 1.0 for these dimensions)
  - Quadrant / Spatial → **programmatic attribute-match** (parse SVG/TikZ/Asy code to extract positional attributes; CLIPScore SNR < 1.0 for spatial dimensions, known CLIP limitation)

### CLIPScore signal-to-noise gate (v3.2, per-dimension result locked by D014)

Standard SNR baseline using a shuffle-permutation null. Per-dimension SNR results from D014 pilot:

| Dimension | SNR | Gate result |
|---|---|---|
| Color | 2.20 | PASS → CLIPScore |
| Shape_count | 1.08 | PASS → CLIPScore |
| Quadrant | 0.67 | FAIL → attribute-match |
| Spatial | 0.87 | FAIL → attribute-match |
| Stroke_fill | 1.82 | PASS → CLIPScore |

## 5. Probe targets and aggregation

**Primary attribute battery (5, used for headline ρ)**:

1. Dominant color bucket — 8-way
2. Shape count — 10-way (capped 0-9)
3. Layout quadrant — 4-way
4. Relative spatial relation of top-2 shapes — 5-way (above / below / left / right / inside)
5. Stroke/fill parity — binary

**Reference-only (appendix, NOT in headline ρ)**:

6. CLIP ViT-L/14 zero-shot 50-way scene class — appendix reference only (per-class training samples insufficient for stable probe).

The headline ρ argmax-over-layer is computed **only** over the 5 primary attributes.

**Aggregation (locked)**: at each tracked layer ∈ {4, 8, 12, 16, 20, 24, 28}, the per-sample probe input is the **mean of residual stream vectors over code-token positions of the teacher-forced reference code** (caption / chat-template / system tokens are masked out). One `[hidden_dim]` vector per (model, layer, format, triple_id).

**Probe head**: sklearn `LogisticRegression(C=1.0, max_iter=2000)` for categorical attributes, `Ridge` for any continuous future addition; CPU-fit only.

**Cross-format probe-transfer**: for each (model, layer, source_format → target_format) train on source-format vectors and auto-extracted attribute labels, test on target-format vectors of the **same triple_ids**.

**Note**: spatial relation and quadrant are statistically correlated; stroke-fill parity may degenerate on minimalist SVG. Reported as-is; no "independent attributes" claim.

## 6. Stage E — REMOVED (D009 A7)

Stage E (QLoRA pre vs post on Qwen3-Coder) is removed entirely. D007 dropped Qwen3-Coder; D016 activated A4 (qualitative case study); D009 A7 specifies that A4 activation cascades to Stage E skip. The 8h GPU budget is returned to paper writing (16→24h).

**Deviation**: restoring Stage E requires `log_decision` with model justification and GPU budget reallocation.

## 7. Model-switch + tool-stack-init overhead (v3.4)

- **vLLM stack** (Stage A): 3 model loads. ~2–3 min/load.
- **HF stack** (Stage B): 3 model loads + per-model hooks installed. ~1–2 min/load.
- **Reserved wall-clock budget**: ~0.5 h total init/switch.

## 8. Budget — GPU time vs CPU wall clock (v3.4, 3 models, no Stage E)

| Dim | Stage | Tool | Estimate |
|---|---|---|---|
| GPU | Stage A free generation (3 models × 3 formats × 200 captions × 2 shots) | vLLM fp16 | ~3 h |
| GPU | Stage B teacher-forced single forwards (3 models × 3 formats × 500 pairs) | HF transformers | ~2–3 h |
| GPU | Model-switch + init overhead (§7) | both | ~0.5 h |
| **GPU total** | | | **~6–7 h** |
| CPU | Rendering (parallel workers) | — | ~2–4 h |
| CPU | CLIP scoring + attribute-match | — | ~1 h |
| CPU | Probe fitting (5 attrs × 7 layers × 3 models × 6 transfer cells × sklearn) | — | ~2–4 h |
| CPU | CKA / Procrustes | — | ~1 h |
| **CPU total** | | | **~6–10 h** (partially overlaps GPU) |

**Wall-clock estimate**: recovery (5h) + gen (5h) + Stage B (4h) + probe+stats (6h) + paper (22h) ≈ **42 h** within deadline with **~10 h slack** (see ROADMAP v16 for precise budget).

## 9. Pre-registered backup narrative (v3.4 — N=12 case study framework, D019)

To prevent winner-of-the-data narrative drift, the headline ρ outcome is partitioned **in advance** into three regions. The paper writes a **framework + Qwen2.5 family case study** narrative regardless of region. **Decoupling claim is completely prohibited** at N=12 (D019 C2).

### Region 1 — "framework validates on case study" (ρ > 0 and bootstrap p < 0.05)

- **Method**: Frame as "training-free hidden-state geometry framework for analyzing cross-format behavioral transfer, demonstrated on a Qwen2.5 family case study (N=12)."
- **Results**: Lead with the 12-cell ρ scatter (geometry vs behavior) + bootstrap CI; per-layer ρ curve as supporting figure. Report N=18 exploratory view (with VisCoder2) in appendix with memorization-floor caveat.
- **Discussion**: Position as framework contribution validated on one model family; explicitly state that generalization to other families requires future replication with additional models. N=12 vs N=18 ρ difference reported as contamination effect evidence. **No decoupling claim.**

### Region 2 — "framework + existence finding" (|ρ| < 0.3 or p ≥ 0.05)

- **Method**: Same framework; contribution reframed as "a cleanly-designed test with a null result on the predictive question and a positive result on the existence question."
- **Results**: Lead with two findings: (a) cross-format probe transfer is non-trivially above chance — a shared subspace **does** exist; (b) behavioral transfer is poorly correlated with the geometric measure at N=12. Report N=18 exploratory view with memorization-floor caveat. **No decoupling claim of any kind** — the null result is reported as "insufficient evidence at N=12" (not "evidence of decoupling").
- **Discussion**: Frame as framework + existence finding. Cite power analysis (§11) showing N=12 is underpowered to detect moderate effects. The null ρ result constrains future work rather than establishing decoupling. **Prohibited language**: "decoupling," "dissociation," "independence" (even with hedging qualifiers).

### Region 3 — "anti-correlation case study" (ρ < -0.2)

- **Method**: Same framework; contribution becomes "evidence against the format-agnostic shared subspace hypothesis in a Qwen2.5 family case study."
- **Results**: Lead with negative ρ + bootstrap CI; show that high probe-transfer accuracy *anti-predicts* behavioral transfer. Report N=18 exploratory view with memorization-floor caveat.
- **Discussion**: Present as a case study finding requiring independent replication. **No decoupling claim.**

### Anti-drift commitment

- The Method paragraph and headline figure for the matching region are written **before** the ρ value is computed.
- After computing ρ, only the matching region's outline may be expanded; cross-region cherry-picking is a deviation under §10.
- If ρ + CI straddles two regions, the paper writes **both** outlines.

### v3.4 — VisCoder2 in N=18 auxiliary (applies to all three Regions)

Per §3-bis and §3-bis-2, VisCoder2-7B is HIGH-contamination on eval pool VCM subsets and probe pool VCM-asy. All three Region outlines include:

> "VisCoder2-7B is excluded from the headline N=12 statistic (§1) due to probe pool contamination on Asy (§3-bis-2). An exploratory N=18 view including VisCoder2 is reported in the appendix as a known memorization floor, not as a robustness check. The N=12 vs N=18 ρ difference serves as direct quantitative evidence of contamination effect on probe geometry."

## 10. Deviation policy (v3.4)

Any deviation from §1–§9 requires an explicit `log_decision` entry with rationale **before** any post-hoc analysis is run. The following all qualify as deviations:

- Changing N (12 headline cells; 18 auxiliary cells) or the model pool.
- Swapping the headline content-fidelity metric outside the §4 SNR trigger.
- Restoring eval/probe merging.
- Restoring Stage E (§6).
- Switching Stage B aggregation away from code-token mean-pool.
- Lowering the §3-quater cosine threshold below 0.7.
- Rewriting the §9 backup-narrative outlines after seeing data.
- **(v3.4)** Enlarging the VCM supplementation beyond the locked counts (135 SVG / 108 Asy), changing the §3 filter rules or seed (`20260410`), or swapping the supplementation source.
- **(v3.4)** Dropping the §3-bis appendix N=18 exploratory ρ or removing the per-table memorization-floor footnote on VisCoder2 results.
- **(v3.4, D019 C2)** Making any decoupling claim (including with hedging qualifiers: exploratory / preliminary / suggestive / qualitative) based on N=12 results.
- **(v3.4, D019 W3)** Framing the N=18 auxiliary as a "robustness check" for the N=12 headline.

## 11. Statistical methods (v3.4, D008 C2 + D019)

### A1. Power simulation — N=12 + N=18 dual run (mandatory)

Run detection power simulation for ρ_true ∈ {0.1, 0.3, 0.5}, 1000 trials each, for **both** N=12 and N=18. Results section must simultaneously report:
- Bootstrap 95% CI for the observed ρ
- Simulated power at ρ_true = 0.3 (the "moderate effect" benchmark)
- Known N=18 power from D012: 0.218 (ρ=0.3) / 0.435 (ρ=0.5)
- N=12 power expected to be lower (estimated ~0.13 / ~0.30 at ρ=0.3 / 0.5)

### A2. Permutation null — descriptive statistics only

Shuffle format-pair labels 1000× and recompute ρ to generate null distribution. Report:
- Null distribution shape (mean, median, std, 5th/95th percentiles)
- Where observed ρ falls relative to null distribution

**Prohibition**: A2 results must **not** be used to claim decoupling at N=12. The permutation null is reported for completeness and to characterize the null distribution shape, **not** as evidence for or against any hypothesis about representation-behavior relationship.

### Venue (D019 W2)

- **Primary target**: EMNLP 2026 findings / NeurIPS 2026 workshops on mechanistic interpretability
- **Stretch target**: NeurIPS / EMNLP main conference (not expected with N=12 case study)
- **Framing**: "framework + Qwen2.5 family case study"
- Post-deadline: may add 4th model (DeepSeek-Coder / StarCoder) for v2 submission to main conference

## 12. Fallback ladder (v3.4, replaces v3.1)

**Pre-flight check**: vLLM-gen + HF-TF dual sanity on Qwen2.5-Coder-7B-Instruct, ~5 min each:

1. **vLLM sanity**: serve fp16, generate 10 prompts × 100 tokens, report tok/s.
2. **HF teacher-forcing sanity**: load fp16, teacher-force 10 (caption, code) pairs, capture layer-16 residual stream, mean-pool over code tokens, report `[hidden_dim]` shape + dtype + tok/s.

**Fallback steps** (if sanity fails or wall-clock slips):

1. probe triples 500 → 300
2. layers {4,8,12,16,20,24,28} → {8, 16, 24}
3. drop Qwen2.5-7B base → headline N=6 (critical — requires Director escalation)

## 13. Tool-stack split (v3.1, appendix table)

| Stage | Tool stack | Why | Hidden state? |
|---|---|---|---|
| Stage A — behavioral generation | **vLLM fp16** | Free auto-regressive generation; high throughput | No |
| Stage B — probe hidden states | **HF transformers + `output_hidden_states=True` + teacher-forcing + mean-pool over code tokens** | Decouples generation ability from representation; native hidden-state access | Yes |
| Stage C — CKA / Procrustes | numpy / sklearn (CPU) | Operates on cached `[hidden_dim]` vectors from Stage B | (consumes Stage B output) |
| Stage D — Spearman ρ | numpy / scipy (CPU) | Pure analysis | — |

**Strict separation**: vLLM never sees probe pool prompts; HF never does free generation for behavioral scoring.
