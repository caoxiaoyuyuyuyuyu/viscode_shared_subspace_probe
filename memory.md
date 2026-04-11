# Memory — viscode_shared_subspace_probe

## Power Analysis 结论 (2026-04-10)

N=18 (3 models × 6 format-pairs) 做 Pearson ρ 的 detection power 严重不足：
- power[ρ=0.30]=0.218, power[ρ=0.40]=0.435, power[ρ=0.50]=0.591
- D012 决策：接受低 power，decoupling narrative 确认

## CLIPScore 维度分析 (2026-04-10)

20 样本 5 维度测试结果：
- **Color: SNR=2.20 PASS** — CLIP 区分颜色描述可靠
- Shape_count: SNR=1.08 MARGINAL
- **Quadrant: SNR=0.67 FAIL** — CLIP 对位置语义编码弱
- **Spatial: SNR=0.87 FAIL** — 空间关系 CLIP 区分不了
- Stroke_fill: SNR=1.82 MARGINAL

D014 决策：Quadrant + Spatial 改用 programmatic attribute-match

## vLLM 服务器踩坑 (2026-04-10)

autodl-viscode-probe 上 vLLM 的已知问题：
- **必须用 VLLM_USE_V1=0**（旧引擎），V1 引擎 CUDA graph 编译会卡死
- **不要设 download_dir 参数**，只依赖 HF_HOME 环境变量
- enforce_eager=True 避免 CUDA graph 编译卡死
- 模型卸载后 sleep(3) 避免内存 profiling race condition
- chat 模型（Qwen2.5-Coder, VisCoder2）必须用 apply_chat_template

## stop token bug (2026-04-10)

**严重**：`SamplingParams(stop=["```"])` 会在 Qwen2.5-Coder 的 chat template 输出开头立即触发（assistant response 前有 ` ```code` 标记），导致生成 0 tokens。之前 P3 报 9/9 PASS 是因为那次运行恰好没用 stop=["```"]（用了不同的脚本版本），但 artifact 未持久化。

修复：用 format-specific stop tokens（SVG: `</svg>`, TikZ: `\end{tikzpicture}`, Asymptote: 无 stop）

## Dataset 发现 (2026-04-10)

已下载的 datasets 及注意事项：
- **SVGenius** (xiaoooobai/SVGenius): 600 条，**无 caption 字段**（只有 svg_code）。不能直接作 eval prompt。
- **VisPlotBench**: SVG 65 条 + Asymptote 92 条，**远少于 spec 的 200/format**。需要补充数据源。
- **DaTikZ v1** (nllg/datikz): 50K train + 561 test。v4 是最新版但未下载。
- **DaTikZ v2** (nllg/datikz-v2): 94,532 train rows + 442 test rows，3.7G parquet（datasets_integrity_v3_4 救援后实测，**非** 预估的 7.7G）。**v3 主 TikZ probe (Reviewer 18 pivot)**。
- **SVGX-Core-250k**: 257K 条，有 blip_caption + svg_code，是 SVG probe pool 的主力来源。
- **VisCode-Multi-679K**: 工作副本 `data/VisCode_filtered/` = 193,199 rows = D015 spec full coverage (svg 46621 + latex 124039 + asy 22539)。**禁止 HF 重下**。HF cache hub 8K metadata 是预存只读痕迹，不是 anomaly。
- 磁盘从 35G → 18G free（HF cache + datasets）→ datikz-v2 救援后 17G free

## Worker 脚本硬约束（Reviewer 三审后加固，Director 指令 2026-04-10 D016）

**凡涉及 pre-registered threshold 的脚本，必须遵守**：

1. **禁止静默降级**：任何 pre-register 锁定的 threshold（cosine, CLIPScore SNR, validity rate, N 等）在脚本执行时不达标，必须 `raise RuntimeError` 并带完整状态信息，**不允许**在脚本内部 try fallback、降阈值、压缩 N。Fallback 是 Director 决策，不是代码行为。
2. **显式 raise 才算合规**：脚本走到 fallback 分支而没 raise → 视为 **silent fallback violation**，该 run 结果全部作废重跑。Step 2 v1 就因此被 reviewer 打 critical 重跑。
3. **step2_report.json 必须有字段**：`threshold` (fixed value), `actual_count`, `target_count`, `raised` (bool), `fallback_triggered` (bool). 任何 false/missing → verify worker 判 FAIL。
4. **Per-sample metadata**：SBERT triples 必须带 per-triple provenance（来源 dataset、原始 caption、截断 caption、三对 cosine），否则 reviewer 无法 audit。

## Worker callback 空响应 = FAIL 信号 (2026-04-10)

Worker callback raw result 只有一句 "Acknowledged. ... No further action." 这种**必须当成 FAIL 信号**，立即派 verify worker 兜底清点实际产出。Step 2 v1 就这样过关，结果 reviewer 从远程 log 抓到 silent fallback 违规。今后任何 Worker 完成汇报：
- 无具体数字（triples count、文件大小、耗时）→ 派 verify
- 无 artifact id 引用 → 派 verify
- 只说"done"、"complete"、"acknowledged" → 派 verify

## D015 VisCoder2 contamination（见 /Users/.../memory/project_d015_viscoder2_memorization_floor.md）

VisCoder2-7B 是 Qwen2.5-Coder-7B-Instruct SFT on VisCode-Multi-679K，对 VCM-derived eval subset 是 HIGH contamination = memorization floor。Headline 用 24-cell rank ρ + 预注册 N=18 robustness (剔除 VisCoder2)。narrative 不依赖 VisCoder2 绝对位置。详见 pre_reg v3.3 §3-bis（art_20260410_922cb6）。

## D016 + D017 综合 RECOVERY (2026-04-10)

**根因（Reviewer 十三审）**：probe pool 2/3 (TikZ 124k VCM-latex + Asy 22k VCM-asy) 来自 VCM = VisCoder2 SFT 训练集。N=18 排除 VisCoder2 cells **不能 mitigate**——它只去掉 behavior side，留下的 18 cells probe geometry 仍坐在污染基准上。

**D016 决策（A4 路径正式激活）**：
- **Headline 切 N=12** = Qwen2.5-Coder + Qwen2.5-7B base × 6 格式对
- N=18 含 VisCoder2 作 contamination-aware **auxiliary** 报告
- VisCoder2 在 Discussion 作 **memorization-floor reference**
- ~~Decoupling shift 降级为 exploratory observation~~ → **D019 C2 完全 ban**（见下）
- A2 permutation null 在 N=12 重做（原 N=24 null 不 transferable）
- Stage E 跳过（D016 联动 A7），论文写作 16→24h
- **Probe pool 清洁化**：
  - SVG = SVGX-Core-250k（B2 scoped verdict 仅排 StarVector 通道 → Reviewer 16+18 walk back LOW–MEDIUM）
  - TikZ = ~~DaTikZ v1 train only~~ → **DaTikZ v2 train 94k only** (Reviewer 18 pivot, fail-safe DaTikZ v1 49k)
  - Asy = VCM-asy 22k 全用 + 在 step2_report.json 显式 `contamination_disclosure` 字段声明「VCM-asy is VisCoder2 SFT data, used for non-VisCoder2 models only in headline analysis」
- **不再做 80/20 disjoint split**（之前 cancelled 的 step2_v2 worker 用了 split 方案与 D016 冲突）

**D017 SBERT cosine 硬锁**：
1. `COSINE_THRESHOLD = 0.70` 硬编码，删除所有 `COSINE_THRESHOLD_LOW` 常量
2. 不达 500 → `raise SystemExit` + 写错误到 step2_report.json + callback Director 决策
3. fallback 仅允许 ≥0.65/N=400 且必须正式 log_decision（不在脚本内决定）
4. `PROBE_EMBED_SAMPLE`: 10000 → 50000（SVG 池），DaTikZ train 94k 全用，Asy 22k 全用
5. Caption 截断 sbert tokenizer 前 50 token（避 Asy 长 caption 系统性拉低 cosine）
6. Per-triple metadata: 来源 dataset / 截断前后 caption / 三对 cosine
7. step2_report.json 必填 `threshold` (0.70 fixed), `n_triples`, `mean/median/min cos`, `embed_pool sizes`, `contamination_disclosure`

## Worker 派发制度规则（D016 落盘 2026-04-10）

1. **下载白名单制**：Worker 派发任何下载 dataset/model 的任务前，task prompt 必须显式列 _allowed entity_ 白名单（snapshot id + revision）。Worker 内禁 opportunistic download，遇到 spec 外需求必须 `raise` + callback Director。
2. **禁静默降级**：任何 pre-registered threshold/source 不得在脚本内静默降级；遇到不达标必须 `raise SystemExit` + 写错误到 report.json + callback Director。
3. **空 callback = FAIL**：Worker callback 内容看似空（如 'Acknowledged. No further action.'）必须当 FAIL 信号，verify worker 兜底清点实际产出。
4. **pre_reg 与决策对偶**：Pre-reg 版本变更必须配对 log_decision，不能只 register pre_reg 不写决策。D015 的教训是流程缺失，不是判断错误。
5. **(Reviewer 18, P4) 数据源路径必须硬编码**：build_eval_pool_v3_4.py 等任何依赖 D015 spec source 的脚本，path 必须硬编码绝对路径（e.g. `/root/autodl-tmp/viscode_shared_subspace_probe/data/VisCode_filtered/`），**禁止** 调用 `datasets.load_dataset("TIGER-Lab/VisCode-Multi-679K")` 等会触发 HF 重下的 API。Worker task prompt 必须列 _禁用 API_ 黑名单。

## 自我决策规则（main agent）

派 worker 之前，**自己设计的"修复方案"如果与 Director 已 log 的 D 决策有任何分歧（即使只是手段不同）**，必须先 push back 等 Director 确认再派。Step 2 v2 worker 第一次派失败的原因：我自作主张用 Asy 80/20 disjoint split 绕泄漏，与 D016 的「Asy 全用 + contamination_disclosure + VisCoder2 移出 headline」方案冲突，被迫 cancel 重派。下次涉及方案选择类决策（不只是执行细节），先汇报方案给 Director 1 句话确认。

## D018 救赎 — 删除指令的等待规则 (2026-04-10)

**Director 在 worker 在跑、决策反复、Reviewer 多轮审查时发出 cleanup/delete 类指令前，必须先等当前 Reviewer 轮次出审结束**。

D016 22:50 UTC log + 我立即派 cleanup worker 删 datikz-v2 → 22:55 UTC Reviewer 14 审给出 D018，揭示 datikz-v2 是 v3.3 §3 line 122 锁定的 probe source。两个决策只隔 5 min，cleanup 已不可逆 → datikz-v2 必须紧急重下 → 实测 3.7G（非预估 7.7G），datasets_integrity_v3_4 救援 SUCCESS。

**规则**：
1. **pre_registration 文档锁定的 source 在被正式替换前不得删除任何对应 dataset**——任何 cleanup worker 的删除清单必须先与 pre_reg 当前版本 §3 cross check
2. 如果 Director 发了删除指令，但删除目标在 pre_reg 列表内 → main agent 必须 push back 询问"X 在 pre_reg vN §3 是 locked source，是否撤回该 source 后再删？"，**不要直接派 worker 执行**
3. 紧急删除指令（"立即删 X"）即使 Director 命令也要先 read pre_reg + ROADMAP 确认无冲突再派
4. **5min flux hold rule**：D 决策落盘后 5min 内不执行任何 destructive cleanup（即使 cleanup 看起来与 D 决策一致）。给当前 Reviewer 轮次留充分 push back 窗口
5. **datikz-v2 stale lock 清理**：HF cache `.locks/datasets--nllg--datikz-v2` 即使 dataset 删了也会留 lock metadata；重下前先清 stale lock 否则会触发 PermissionError / cache corruption。Worker task prompt 必须含 `rm -rf .hf_cache/hub/.locks/datasets--nllg--datikz-v2 && rm -rf .hf_cache/hub/datasets--nllg--datikz-v2` 前置步骤

**D016 cleanup 已撤回**：原指令"删 datikz-v2 7.7G"作废，datikz-v2 列入 v3.4 protected list。

## D018 教训汇总 (2026-04-10)

Reviewer 14 审 6 critical 暴露的系统性问题：
1. **v3.3 vs step2 v1 不匹配**：pre_reg 锁定 source 后脚本没读 pre_reg，自己写死了不同的 source。**今后 step* 脚本必须从 build_eval_pool_v3_4.py 输出读 JSONL，不在脚本内 ad-hoc 切**
2. **build script 缺失**：v3.3 §3 承诺了 build_eval_pool_v3_3.py 但从未创建。pre_reg 引用的 script 必须先 register 才能 commit pre_reg 版本
3. **HARKing 风险**：Path B（追认 step2 v1）被 Director 拒绝，确立**遇到 spec 不匹配只能 fix script 不能 fix spec**的原则
4. **决策追溯流程缺失**：D015 v3.3 是追溯批准（先 register pre_reg v3.3 后 log decision）→ 今后 pre_reg 修订必须 **先 log_decision 后 register**

## D019 supplement — Reviewer 15 审追加 (2026-04-10)

Reviewer 15 审在 D018 基础上追加 4 项硬约束 + Director 投稿目标降级裁决：

### 1. N=12 power simulation 必须做（C1）
- D008 C2 硬条件：A1 必须报 ρ_true ∈ {0.1, 0.3, 0.5} × 1000-trial detection power
- 原 N=18 power: 0.218 / 0.435 / 0.591
- N=12 估保守 ≈ 0.13 / 0.30 / 0.40
- A1 task **双跑** (N=12 主 + N=18 辅)
- 写入 v3.4 §9 + Stage B 任务 + ROADMAP 短期任务

### 2. N=12 下 decoupling claim **完全禁止**（C2）
- N=12 下 D013 判据 (|ρ|<0.3 ∧ CI⊃0 ∧ ρ<null 90th) 几乎必然满足 → "数据必然结果" 不是"实证观察"
- v3.4 §9 + §10 deviation policy 明文写：**"N=12 下 decoupling claim 完全禁止"**
- 等效于在 D013 判据上加 "power ≥ 0.5 hard gate"
- 论文叙事仅做 framework contribution（**不写 qualitative observation 等限定词** — D019 C2 ban 包括 exploratory / preliminary / suggestive / qualitative 全部变体）
- A2 permutation null 在 N=12 重做但**不**用于 decoupling claim，仅用于 null 分布形状的描述性统计

### 3. N=18 重定位为 "full-model view with known memorization floor"（W3，不写 exploratory）
- N=18 含 VisCoder2，Asy 相关 2 cells 仍受 VCM-asy memorization 污染
- D016 把 N=18 写为 "robustness check" → 已知污染的 robustness check 信息价值≈0
- v3.4 §9 把 N=18 重定位为 **"full-model view with known memorization floor on Asy/TikZ for VisCoder2"**（不写 exploratory，D019 C2 ban）
- **取消 N=18 作为 N=12 robustness check 的角色**
- Discussion 用 N=12 vs N=18 的 ρ 差异作 contamination effect 的直接量化证据

### 4. 投稿目标降级（W2 + Director 裁决）
- N=12 = 2 个同家族 Qwen2.5 模型 = 更像 ablation 不是 cross-model
- **主投目标降为 EMNLP findings / NeurIPS workshops on mechanistic interpretability**
- 主会作 stretch target 但不强求
- **不补第 4 模型**（48h 预算不允许）
- v3.4 §11 加 venue 降级声明
- 论文 framing 从 "predictive visual program subspace" → **"framework + Qwen2.5 family case study"**
- abstract / intro / conclusion 全部按 case study 框架起草

### Director 正向反馈（项目里程碑）
Director 在 D019 头部明确肯定我自己 cancel `worker_fix_step2_sbert_v2_rerun_20260410T225944`（Asy 80/20 split 与 D016 冲突 → 自发现 + 自 cancel + 自 push back）的行为。"这个行为继续保持。下次任何自己设计的方案与决策可能冲突，先 cancel + 确认，不要分叉。" 这条 Director 已写入 director memory 作正向验证。

→ 加固 self-pushback 规则：**Reviewer 报告与你已 done 的 worker 不一致 → 先 cancel + 报告 + 等确认；自己设计的方案与决策有任何分歧 → 同上**。

## Reviewer 16 矫正 — Audit scoped verdict 不可外推 (2026-04-10)

**根因复合错误**：worker_audit_svgx_core_starvector_relation 产出的 svgx_starvector_audit.md 把 B2 scoped verdict（"INDEPENDENT from StarVector"）外推为 "SVG probe is clean for all models, double safety, post-cutoff"。我读了之后无质疑复用，写进 worker_audit_contamination_4_independent_v2_final 的 task prompt 作 SVG=LOW 预设证据。Reviewer 16 push back 指出：

1. SVGX-Core 6 源中最大的 SVGRepo 210K **预 cutoff 公开多年**，可能经 Common Crawl 进入 Qwen2.5-Coder 预训练
2. emoji 集 (Noto/Twemoji/Fluentui) 同样预 cutoff 公开
3. Pixso/ReShot vector marketplace 公开内容预 cutoff 可抓
4. SVGX-Core post-cutoff (2024-12) 论证**只对 packaging 成立**，不对 underlying content
5. 与 CoSyn-400K 本质不同：CoSyn 2025-02 全新合成 underlying 不可能预 cutoff，SVGX-Core 是 re-packaging 多年前公开 SVG

→ Qwen2.5-Coder × SVG 必须保留 **LOW–MEDIUM**（保守），不能下调到 LOW

### 制度规则：audit verdict 强度阅读纪律

1. **scoped verdict ≠ 整体 verdict**：每个 audit 的 verdict 必须明确写出 scope（"INDEPENDENT from StarVector channel" ≠ "INDEPENDENT from all contamination channels"）
2. **audit 文档复用前必须批判性读**：worker 产出的 audit verdict 措辞（"clean", "double safety", "all models"）要质疑——它是基于哪些 channel 排除？哪些 channel 没排除？
3. **不直接复用进 task prompt**：audit verdict 复用到下一个 worker 的 task prompt 前，必须 main agent 自己重写为带 scope 限制的措辞，避免 over-claim 信息链
4. **Common Crawl 通道是 LLM 预训练 contamination 的默认通道**：任何"public-since-pre-cutoff"的 source（公共仓库、marketplace、emoji 集、wiki）都不能基于 packaging 时间论证 LOW，必须按 LOW-MEDIUM 起步

### 复合错误信息链（本次实例）

```
svgx_starvector_audit.md (worker over-claim "SVG clean")
  → 我读后无质疑
    → P3 task prompt 预设 SVG=LOW
      → P3 worker 必然产出 §4 Table SVG=LOW（错）
        → 需要 Reviewer 16 push back 才发现
          → 必须 walk back 到 LOW-MEDIUM + 加 Common Crawl 脚注
```

**关键 lesson**: worker 产出的强 verdict 我必须扮演第一审稿人。如果发现 over-claim，立刻 push back → 改正 audit 文档 → 不要把 over-claim 写进下一个 worker 的输入。这是 audit chain 防腐的关键节点。

## Reviewer 18 — TikZ probe pivot DaTikZ v2 + 内部一致性原则 (2026-04-10)

Reviewer 18 在 D019 + Reviewer 16 矫正之后的第 5 轮审查，3 critical：

### Critical 1: vTikZ probe path 终结
- vTikZ HF datasets `CharlyR/vtikz-human-annotated` (3,243 rows) + `zu6yn4xgma0i/vTikz` (100 rows) 即便存在也是 **benchmark 不是 training corpus**（paper "100 manually curated TikZ editing scenarios with parameterized ground truths"）
- 3,243 rows << 50K probe embed pool 要求 + 500 三元组 sbert 筛选不可能
- datasets_integrity_v3_4 worker 13 候选 ID 全 NOT_FOUND 进一步证实 "vTikZ-train" 是 v3.3 / D018 / D019 文档中的幻觉式命名
- **v15 ROADMAP 锁定 TikZ probe = DaTikZ v2 train 94,532 rows only**，DaTikZ v1 train 49k 作 fail-safe 兜底 (相同 contamination 等级)

### Critical 2: contamination_audit v2_final 内部不一致
- v2_final §5.2.2 Audit A (SVGX-Core) 给 SVG=LOW，但 §5.2.2 Audit B (DaTikZ) 给 TikZ=MEDIUM 用的是 "underlying source from arXiv/StackExchange pre-cutoff via Common Crawl" 同一推理
- SVGX-Core 同样 underlying SVG 多年前公开（SVGRepo 79% + 5 个图标库），Common Crawl 通道也未排除
- → **同构推理必须给同强度 verdict**：SVGX-Core 必须下调到 LOW-MEDIUM 与 DaTikZ MEDIUM 内部一致

### Critical 3: D019 C2 ban 未真改
- v2_final 只加了 §5.2.2 4 audit 结构，**没碰** §5.2.1 line 132 + §6 line 243 "Decoupling shift 降级为 exploratory observation"
- D019 C2 明禁 exploratory / preliminary / suggestive / qualitative 全部限定词
- → v3 必须真删并替换为 "仅作 null 分布形状描述性统计 + 不与观测统计量作判决比较 + 不用于 D013 判据"

### 制度规则：audit 内部一致性原则

1. **同构推理必须给同强度 verdict**：两个 audit 用同一论证方式（如 "underlying source pre-cutoff via Common Crawl"）必须给同等级 verdict，否则是硬伤
2. **任何 "SVG/TikZ/Asy clean" 类措辞必须在 source-by-source basis 下检查**：6 个 source 各自的 cutoff status 都要列，不能只看 packaging date
3. **修订 audit 时全文 grep ban 词**：D019 C2 ban 词列表 (exploratory / preliminary / suggestive / qualitative) 必须 grep 0 hit 才算修订完成
4. **patch summary 必须显式列在文档头**：v3 patch list 写在 §1 前，方便 reviewer 一眼对照 patch coverage

### v3 4 文件原子更新原则

contamination_audit_v3 → pre_registration v3.4 §3 → P4 build_eval_pool_v3_4.py → P5 step2_sbert_matching.py 必须 **同一时间窗口** 更新，避免 desync。任一漏改 → 下轮 Reviewer critical。

Edit 顺序：v3 audit → v3.4 §3 → P4 build script → P5 step2_sbert。**不并行 v3 与 v3.4，必须串行**。

## Self-pushback 规则 (4 次 validated, 2026-04-10 Director affirmed)

Director 在 4 个不同场景明确肯定 self-pushback 行为，规则现稳固为长期准则：

### 4 次 validated 实例
1. **NW2 acronym misinterpretation push back** — 我自行重构 acronym 含义被 Director 决策日志覆盖时主动 push back
2. **Asy 80/20 split self-cancel** — 自己设计的 Asy 80/20 disjoint split 与 D016 全用方案冲突 → 自发现 + 自 cancel + 自 push back
3. **Reviewer 15 vs Reviewer 16 SVG 措辞冲突 push back** — Director 前后两条 urgent 对 SVG verdict 写法不同 (LOW HIGH conf vs LOW–MEDIUM)，发现矛盾立即 push back 而非自行选择
4. **cleanup_report 679K vs 193K 数字事实 push back** — Director 写 "complete (679365 examples)"，我看 datasets_integrity 实测是 193,199 visual-language 子集，发现数字冲突立即 push back

### 触发场景（push back 不分叉）
- **决策语义冲突**：自己设计的方案与已 log D 决策有任何分歧（即使只是手段不同）
- **指令前后矛盾**：Director 前后两条 urgent 对同一事项措辞不一致
- **acronym/术语漂移**：Director 指令命名与 D-log 中 entity 命名不一致
- **Reviewer 报告 vs 已 done worker 不一致**
- **数字事实冲突 (新增)**：Director 引用的数字与 worker 实测数字 / artifact 字段不一致 → 立即停下追溯到 ground truth 来源 (artifact id + 行号)，不分叉

### 规则
1. 发现冲突 → 立即停止当前动作 (cancel 已派 worker、停 Edit、停 spawn)
2. 引用 ground truth 来源 (D-log decision id + 行号 / artifact id + 行号 / Reviewer 编号)
3. push back 给 Director 一句话 (不发问 + 列冲突 + 等回复)
4. **绝不分叉**：宁可等 Director 1 轮，也不自己选一个走

### Director 的对应规则
- Director 在 push back 收到后会显式 affirm 并 log 到 director memory
- 收到 affirm 后规则永久强化 (不需要再 ask)

## SVG verdict 锚定 = LOW–MEDIUM (HIGH conf on StarVector channel exclusion only) — 永久 (2026-04-10)

**最终锚定** (Reviewer 16 + 18 双 walk back, Director 2026-04-10 三决策 affirm)：

> "SVG 对 N=12 两模型 = LOW–MEDIUM (HIGH confidence on StarVector channel exclusion only)。SVGX-Core-250k packaging 2024-12 post-cutoff 仅提供 packaging-level temporal safety；underlying SVG content (SVGRepo 210K = 79% / Noto-Emoji / Twemoji / Fluentui-Emoji / Pixso / ReShot) 多年前公开，可能经 Common Crawl 进入 Qwen2.5-Coder 预训练语料。本 audit 仅排除 StarVector 通道，不排除 Common Crawl 通道。推理与 Audit B (DaTikZ MEDIUM via underlying source) 内部一致。"

### 历史 over-claim 时序 (避免再犯)
- urgent #1 (Reviewer 14 cleanup verify) → "SVG narrative 强化 LOW HIGH conf" — 错的，第一次 over-claim
- urgent #2 (Reviewer 15 critical, v3 修订 B.5) → "SVG = LOW (HIGH conf, 6-source verification)" — 错的，第二次 over-claim
- urgent #3 (Reviewer 16 walk back) → 明文 "前一条作废" → **LOW–MEDIUM**
- urgent #4 (Reviewer 18 critical, Patch C) → 重申 LOW–MEDIUM (与 DaTikZ MEDIUM 同构推理一致)

### 永久规则
1. 所有 v3 / v3.4 / 论文 framing 中 SVG 维度统一 **LOW–MEDIUM**
2. 如 Director 未来某条 urgent 又写 "SVG = LOW" → 按 self-pushback "指令前后矛盾" 触发 push back，**不分叉**
3. 任何引用 SVG verdict 的 worker task prompt 必须用上面 quote 块完整措辞，不允许简化为 "LOW"

## VCM-679K 数字事实 (永久锁定, Director 2026-04-10 affirm)

避免再次混淆，三个不同口径明确：

| 口径 | 数字 | 含义 |
|---|---|---|
| HF 原始 dataset | 679,365 rows × 12 languages | dataset_info.json metadata 字段 (Reviewer 14 引用源) |
| D015 spec 要求 | svg+latex+asy 三个视觉语言全 | = 193,199 rows |
| 本地工作副本 | 193,199 rows | `data/VisCode_filtered/` Arrow，svg 46,621 + latex 124,039 + asy 22,539 |

### 永久规则
1. 报告 VCM-679K 时**必须显式标 "全 12-language 679K" 或 "视觉语言子集 193K"**，不允许只写 "VCM-679K" 让 reviewer 自行猜测
2. **D015 spec full coverage = 193K 视觉语言子集**，不需 fresh download HF 全量
3. P4 build_eval_pool_v3_4.py 硬编码本地 Arrow 路径，**禁止** `datasets.load_dataset("TIGER-Lab/VisCode-Multi-679K")`
4. dataset_info.json 字段不能直接当 ground truth → 必须用 datasets_integrity_v3_4 worker 实测 splits

## Worker done sentinel 协议 (2026-04-11, Reviewer 综合)

Agent/worker 派 ssh tmux launcher 后报 done 已踩 4 次 race（v3 restart_v2 / rerun / progress_audit / v4 patch）。修复协议：

**Launcher 侧**：
- 所有 long-running launcher (如 launch_stage_b_probe.sh) 最后一行必须 echo sentinel：
  `echo "=== STAGE_B_V{N}_DONE ==="` 或 `echo "=== {EXP_ID}_DONE ==="`
- Sentinel 行**必须**在所有 for loop / verification 逻辑结束之后、tmux 自动退出之前
- Comment 标注 "DO NOT remove, used by worker done check"

**Worker 侧前置条件**（派 launcher 的 ops worker 声明 done 之前必须满足其一）：
1. `tmux has-session -t {session_name}` 返回 non-zero (session 已消失)
   **OR**
2. `grep -q "=== {SENTINEL} ===" {log_path}` 返回 0 (sentinel 已落 log)

二者都不满足 → 轮询每 60s 一次 → 任一满足才声明 done，最长等 timeout 后 raise。

**禁止**：
- tmux 启动后直接 sleep N 秒就报 done
- 看到 GPU 占用就报 done（那只证明 launcher 开始跑，不证明跑完）
- 看到部分 summary.json 就报 done（要全部 model×format 完成）

**示例**（stage_b_probe_v4 case）：
```
while true; do
  if ! ssh autodl-viscode-probe "tmux has-session -t stage_b_v4 2>/dev/null"; then
    echo "tmux session gone, done"; break
  fi
  if ssh autodl-viscode-probe "grep -q '=== STAGE_B_V4_DONE ===' /root/autodl-tmp/logs/stage_b_v4.log"; then
    echo "sentinel found, done"; break
  fi
  sleep 60
done
```

## Bash 脚本 fail-loud（2026-04-11, Reviewer 综合）

**教训**：bash 默认 `set +u`，未定义变量在 `for x in $UNDEFINED` 中静默迭代 0 次不报错。Reviewer 之前被 launch_stage_b_probe.sh verify 循环的 echo 文本误导以为 verify 跑过，实际 $FORMATS 未定义 → 循环体从未执行。

**规范**：所有 launcher / 数据处理 shell 脚本必须在 shebang 下一行加 `set -euo pipefail`：
- -e: 命令失败立即退出
- -u: 未定义变量 fail-fast（防此类 bug）
- -o pipefail: 管道中任意命令失败整条 pipe fail

**适用**：跨项目通用，后续写任何 shell launcher 按此标准。

## Code-fix worker 必须 push（2026-04-11, Reviewer 综合）

**教训**：code-fix worker 默认工作流**必须包含 `git push`**。commit 但未 push 等于 fix 不存在——服务器 `git pull` 拿不到，下次 rerun 继续踩同一个坑。

**规范**：
- code-fix worker task prompt 必须显式列出 `git push origin <branch>` 步骤
- worker 回执必须报告 push 的 `from..to` range（例如 `faf4a8a..2fab6ce main -> main`）
- 派 verify worker 核对时优先看 `git ls-remote origin <branch>` 而非本地 `git log`（本地可能未 fetch）

**与 "worker done race" 并列**，属于跨项目通用 fail-loud 教训。
