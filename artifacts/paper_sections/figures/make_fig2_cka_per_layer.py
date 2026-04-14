"""Figure 2: Cross-format CKA per layer for all six models.

Bootstrap mean with 95% CI shaded band, plotted against relative network
depth. Legend placed outside the axes to avoid occluding curves.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    "savefig.pad_inches": 0.03,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Okabe-Ito palette
COLORS = {
    "coder":     "#0072B2",  # blue
    "viscoder2": "#E69F00",  # orange
    "qwen25":    "#009E73",  # green
    "deepseek":  "#D55E00",  # vermillion
    "codestral": "#CC79A7",  # reddish purple
    "starcoder2":"#56B4E9",  # sky blue
}
LABELS = {
    "coder":     "Qwen2.5-Coder-7B",
    "viscoder2": "VisCoder2-7B",
    "qwen25":    "Qwen2.5-7B",
    "deepseek":  "DeepSeek-Coder-V2-Lite",
    "codestral": "Codestral-22B",
    "starcoder2":"StarCoder2-15B",
}
ORDER = ["coder", "viscoder2", "qwen25", "deepseek", "codestral", "starcoder2"]

# Published null threshold (A2 permutation, shared across models ≈ 0.04)
NULL_LEVEL = 0.04

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent.parent  # .../artifacts
DATA = ARTIFACTS / "cross_family_summary_v2.json"
OUT = HERE / "fig_cka_layer_curves_6model.pdf"


def load_model_curves(summary):
    curves = {}
    for key in ORDER:
        md = summary["models"][key]
        n_layers = md["n_layers"]
        sampled = md["sampled_layers"]
        rel_depth = [100.0 * l / n_layers for l in sampled]
        mean, lo, hi = [], [], []
        for l in sampled:
            cell = md["all_layers"][f"L{l}"]
            mean.append(cell["bootstrap_mean"])
            lo.append(cell["ci_low"])
            hi.append(cell["ci_high"])
        curves[key] = {
            "rel": np.array(rel_depth),
            "mean": np.array(mean),
            "lo": np.array(lo),
            "hi": np.array(hi),
            "peak_layer": md.get("peak_layer"),
            "peak_depth_pct": md.get("peak_layer_depth_pct"),
        }
    return curves


def main():
    summary = json.loads(DATA.read_text())
    curves = load_model_curves(summary)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))

    for key in ORDER:
        c = curves[key]
        color = COLORS[key]
        ax.fill_between(c["rel"], c["lo"], c["hi"],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(c["rel"], c["mean"], "o-",
                color=color, markersize=3.5, linewidth=1.4,
                label=LABELS[key], zorder=3)

    # Permutation null reference
    ax.axhline(NULL_LEVEL, color="grey", linestyle="--",
               linewidth=0.9, zorder=1, label=f"Perm. null (\u2248{NULL_LEVEL:.2f})")

    # Qwen2.5-7B final-layer washout annotation (F4)
    q = curves["qwen25"]
    washout_idx = int(np.argmin(q["mean"][-2:])) + (len(q["mean"]) - 2)
    ax.annotate(
        "L28 washout",
        xy=(q["rel"][-1], q["mean"][-1]),
        xytext=(q["rel"][-1] - 18, q["mean"][-1] - 0.035),
        fontsize=7, color="#D55E00",
        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.8),
    )

    ax.set_xlabel("Relative network depth (\\%)")
    ax.set_ylabel("Cross-format linear CKA")
    ax.set_xlim(0, 105)
    ax.set_ylim(0.0, 0.22)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.6)

    # Legend OUTSIDE axes on the right — cannot occlude curves
    ax.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
        fontsize=7.5,
        handlelength=1.6,
        handletextpad=0.5,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"Saved {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
