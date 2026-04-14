"""Fig 3: PCA min-k distribution across 6 models x 7 layers (primary experiment).

Data source: registry experiment `format_residualized_cka_full_v2`
(metrics.pca_mink_full.*), aggregated per-model k_min breakdown over the
7 probed layers. Per-cell k_min assignments are taken verbatim from the
registry notes (Coder all k=2; DeepSeek L12 k=3 rest k=2; VisCoder2 all
k=2; Qwen2.5 all k=2; Codestral L8/L24 k=3, L16 k=4, rest k=2;
StarCoder2 L17/L23 k=3, rest k=2). Total = 36 x k=2 + 5 x k=3 + 1 x k=4
= 42 cells; k<=3 covers 41/42 (97.6%).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Per-model k_min counts over 7 probed layers (primary experiment).
# Columns = k in {2, 3, 4}.
MODELS = ["Qwen2.5-Coder", "DeepSeek-Coder", "VisCoder2",
          "Qwen2.5-7B", "Codestral", "StarCoder2"]
K2 = [7, 6, 7, 7, 4, 5]
K3 = [0, 1, 0, 0, 2, 2]
K4 = [0, 0, 0, 0, 1, 0]

assert sum(K2) + sum(K3) + sum(K4) == 42
assert sum(K2) + sum(K3) == 41   # k<=3 covers 41/42

C_K2 = "#0072B2"
C_K3 = "#56B4E9"
C_K4 = "#D55E00"

fig, ax = plt.subplots(figsize=(3.25, 2.3))
x = np.arange(len(MODELS))
w = 0.62

bottom = np.zeros(len(MODELS))
ax.bar(x, K2, w, bottom=bottom, color=C_K2, edgecolor="black", linewidth=0.4,
       label="$k{=}2$")
bottom = bottom + np.array(K2)
ax.bar(x, K3, w, bottom=bottom, color=C_K3, edgecolor="black", linewidth=0.4,
       label="$k{=}3$")
bottom = bottom + np.array(K3)
ax.bar(x, K4, w, bottom=bottom, color=C_K4, edgecolor="black", linewidth=0.4,
       label="$k{=}4$")

ax.axhline(7, color="black", linestyle=":", linewidth=0.7, zorder=1)
ax.text(len(MODELS) - 0.5, 7.18, "$7$ probed layers", color="black",
        fontsize=7, ha="right", va="bottom")

ax.set_xticks(x)
ax.set_xticklabels(MODELS, rotation=28, ha="right")
ax.set_ylim(0, 8.2)
ax.set_ylabel("$\\#$ (model, layer) cells")
ax.legend(loc="upper right", frameon=False, handlelength=1.0,
          columnspacing=0.6, borderaxespad=0.3, ncol=3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
OUT = Path(__file__).parent / "fig3_pca_mink.pdf"
fig.savefig(OUT)
print(f"Saved {OUT} ({OUT.stat().st_size} bytes)")
