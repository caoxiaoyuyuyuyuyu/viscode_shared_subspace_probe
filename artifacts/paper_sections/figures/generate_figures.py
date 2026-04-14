"""Generate publication-quality figures for the viscode shared subspace probe paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

# ── Style ──────────────────────────────────────────────────────────────
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
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,  # TrueType for editable text in PDF
    "ps.fonttype": 42,
})

# Colorblind-friendly palette (Okabe-Ito)
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_PURPLE = "#CC79A7"
C_CYAN = "#56B4E9"

MODEL_COLORS = {"coder": C_BLUE, "viscoder2": C_ORANGE, "qwen25": C_GREEN}
MODEL_LABELS = {"coder": "Qwen2.5-Coder", "viscoder2": "VisCoder2", "qwen25": "Qwen2.5-Base"}
PAIR_COLORS = {"tikz-asy": C_RED, "svg-asy": C_BLUE, "svg-tikz": C_CYAN}
PAIR_LABELS = {"tikz-asy": "TikZ\u2013Asy", "svg-asy": "SVG\u2013Asy", "svg-tikz": "SVG\u2013TikZ"}

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "stage_b_analysis_v1"
OUT_DIR = Path(__file__).resolve().parent

LAYERS = [4, 8, 12, 16, 20, 24, 28]
MODELS = ["coder", "viscoder2", "qwen25"]
DOUBLE_COL = 6.75  # inches


# ── Figure 1: CKA Layer-wise Trajectories ─────────────────────────────
def fig_cka_trajectories():
    # Use per-layer-per-pair data to compute mean CKA (no CI dependency)
    df_pair = pd.read_csv(DATA_DIR / "cka_per_layer_per_pair.csv")
    df = df_pair.groupby(["model", "layer"])["cka"].mean().reset_index(name="mean")

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.2), sharey=True)

    ymin = df["mean"].min() - 0.015
    ymax = df["mean"].max() + 0.015

    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model].sort_values("layer")
        color = MODEL_COLORS[model]
        ax.plot(sub["layer"], sub["mean"], "o-", color=color, markersize=3.5, zorder=3)
        ax.set_title(MODEL_LABELS[model], fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_xticks(LAYERS)
        ax.set_ylim(ymin, ymax)

        # Highlight qwen25 L28 washout
        if model == "qwen25":
            row28 = sub[sub["layer"] == 28].iloc[0]
            row24 = sub[sub["layer"] == 24].iloc[0]
            ax.annotate(
                "",
                xy=(28, row28["mean"]), xytext=(26, row24["mean"] + 0.012),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.0),
            )
            ax.text(25, row24["mean"] + 0.017, "washout", fontsize=7,
                    color=C_RED, ha="center")

    axes[0].set_ylabel("Mean CKA")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT_DIR / "fig_cka_trajectories.pdf")
    print("Saved fig_cka_trajectories.pdf")
    plt.close(fig)


# ── Figure 2: CKA by Format Pair ──────────────────────────────────────
def fig_cka_format_pairs():
    df = pd.read_csv(DATA_DIR / "cka_per_layer_per_pair.csv")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.2), sharey=True)

    ymin = df["cka"].min() - 0.01
    ymax = df["cka"].max() + 0.02

    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model]
        for pair in ["tikz-asy", "svg-asy", "svg-tikz"]:
            psub = sub[sub["format_pair"] == pair].sort_values("layer")
            ax.plot(psub["layer"], psub["cka"], "o-", color=PAIR_COLORS[pair],
                    label=PAIR_LABELS[pair], markersize=3, zorder=3)
        ax.set_title(MODEL_LABELS[model], fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_xticks(LAYERS)
        ax.set_ylim(ymin, ymax)

    axes[0].set_ylabel("CKA")
    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(top=0.85)
    fig.savefig(OUT_DIR / "fig_cka_format_pairs.pdf")
    print("Saved fig_cka_format_pairs.pdf")
    plt.close(fig)


# ── Figure 3: Permutation Null Distribution ───────────────────────────
def fig_permutation_null():
    data = {
        "coder":     {"observed": 0.1226, "null_mean": 0.0413, "null_std": 0.003035},
        "viscoder2": {"observed": 0.1243, "null_mean": 0.0443, "null_std": 0.003083},
        "qwen25":    {"observed": 0.1011, "null_mean": 0.0366, "null_std": 0.002847},
    }

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.0))

    for ax, model in zip(axes, MODELS):
        d = data[model]
        mu, sigma, obs = d["null_mean"], d["null_std"], d["observed"]

        # Bell curve
        x = np.linspace(mu - 4.5 * sigma, mu + 4.5 * sigma, 300)
        y = norm.pdf(x, mu, sigma)
        ax.fill_between(x, y, alpha=0.25, color=C_BLUE, linewidth=0)
        ax.plot(x, y, color=C_BLUE, linewidth=1.0)

        # Observed line
        ax.axvline(obs, color=C_RED, linewidth=1.2, linestyle="--", zorder=4)

        # z-score annotation
        z = (obs - mu) / sigma
        peak = norm.pdf(mu, mu, sigma)
        ax.text(obs + sigma * 0.5, peak * 0.55, f"z = {z:.1f}",
                fontsize=7, color=C_RED, va="bottom")

        ax.set_title(MODEL_LABELS[model], fontweight="bold")
        ax.set_xlabel("CKA")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    # Shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=C_BLUE, lw=1.0, label="Null distribution"),
        Line2D([0], [0], color=C_RED, lw=1.2, linestyle="--", label="Observed"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout(w_pad=1.5)
    fig.subplots_adjust(top=0.82)
    fig.savefig(OUT_DIR / "fig_permutation_null.pdf")
    print("Saved fig_permutation_null.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_cka_trajectories()
    fig_cka_format_pairs()
    fig_permutation_null()
    print("All figures generated.")
