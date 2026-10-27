"""Generate publication figures for paper.tex from this project's already-
collected experimental results (research_status.tex and session transcripts).
No experiments are re-run here; all numbers are the recorded outputs of
ood_severity_sweep.py, adaptive_k_crossover_study.py /
adaptive_k_single_point.py, the seed-batching verification script, and
mujoco_multiseed_study.py / mujoco_reacher_experiment.py.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})

COLOR_PRIMARY = "#2b6cb0"
COLOR_SECONDARY = "#c05621"
COLOR_TERTIARY = "#2f855a"
COLOR_GRID = "#d0d0d0"


def fig1_ood_severity_curve():
    shift = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    auc = np.array([0.520, 0.409, 0.458, 0.551, 0.654, 0.764, 0.895, 0.979, 0.999])
    auc_ci = np.array([0.011, 0.010, 0.014, 0.012, 0.008, 0.010, 0.006, 0.002, 0.000])

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.axhspan(0.0, 0.6, color=COLOR_SECONDARY, alpha=0.08, zorder=0)
    ax.axhspan(0.9, 1.0, color=COLOR_TERTIARY, alpha=0.10, zorder=0)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, zorder=1)
    ax.errorbar(
        shift, auc, yerr=auc_ci, marker="o", markersize=5, capsize=3,
        color=COLOR_PRIMARY, linewidth=1.8, zorder=3,
    )
    ax.text(0.05, 0.555, "near chance", fontsize=8, color=COLOR_SECONDARY, va="bottom")
    ax.text(0.05, 0.905, "reliable detection", fontsize=8, color=COLOR_TERTIARY, va="bottom")
    ax.set_xlabel("distribution-shift magnitude")
    ax.set_ylabel("OOD detection ROC AUC")
    ax.set_ylim(0.35, 1.03)
    ax.set_xlim(-0.15, 4.15)
    ax.grid(True, color=COLOR_GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_ood_severity.png"))
    plt.close(fig)


def fig2_crossover():
    k_max = np.array([100, 300, 1000, 3000, 8000, 12000, 13000, 14000, 15000, 16000, 20000, 26000])
    ratio = np.array([0.0434, 0.0391, 0.0773, 0.2052, 0.5577, 0.8104, 0.9006, 0.9512, 0.9708, 1.0165, 1.3224, 1.6546])

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.plot(k_max, ratio, marker="o", markersize=4.5, color=COLOR_PRIMARY, linewidth=1.8)
    ax.axvline(15600, color=COLOR_SECONDARY, linestyle="--", linewidth=1.3)
    ax.annotate(
        "crossover\n" + r"$K_{\max}\!\approx\!15{,}600$",
        xy=(15600, 1.0), xytext=(15600 * 1.35, 0.35),
        fontsize=8.5, color=COLOR_SECONDARY,
        arrowprops=dict(arrowstyle="->", color=COLOR_SECONDARY, linewidth=1),
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$K_{\max}$ (rollout budget, log scale)")
    ax.set_ylabel("latency ratio (fixed-high / adaptive)")
    ax.text(120, 1.42, "adaptive-K faster", fontsize=8, color=COLOR_TERTIARY)
    ax.text(120, 0.08, "fixed-high faster", fontsize=8, color=COLOR_SECONDARY)
    ax.grid(True, color=COLOR_GRID, linewidth=0.6, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_crossover.png"))
    plt.close(fig)


def fig3_seed_batching_speedup():
    s_values = np.array([2, 10, 30])
    speedup = np.array([2.11, 9.67, 33.95])
    ideal = s_values  # perfectly linear scaling would match S itself in this regime

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(s_values, ideal, linestyle="--", color="gray", linewidth=1.3, label="linear reference")
    ax.plot(s_values, speedup, marker="o", markersize=6, color=COLOR_PRIMARY, linewidth=2, label="measured speedup")
    for x, y in zip(s_values, speedup):
        ax.annotate(f"{y:.1f}x", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8.5)
    ax.set_xlabel("number of batched seeds $S$")
    ax.set_ylabel("wall-clock speedup vs. sequential loop")
    ax.set_xticks(s_values)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(True, color=COLOR_GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_seed_batching.png"))
    plt.close(fig)


def fig4_task_success_bars():
    tasks = ["MuJoCo\n(regulation, n=15)", "MuJoCo\n(reaching, n=10)"]
    methods = ["single", "ensemble", "adaptive"]
    colors = [COLOR_SECONDARY, COLOR_PRIMARY, COLOR_TERTIARY]

    success = {
        "single": [42.2, 16.7],
        "ensemble": [55.6, 33.3],
        "adaptive": [62.2, 33.3],
    }
    ci = {
        "single": [17.4, 9.7],
        "ensemble": [15.2, 13.8],
        "adaptive": [12.5, 12.9],
    }

    x = np.arange(len(tasks))
    width = 0.25
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for i, method in enumerate(methods):
        offset = (i - 1) * width
        ax.bar(
            x + offset, success[method], width, yerr=ci[method], capsize=3,
            color=colors[i], label=method, alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 85)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_task_success.png"))
    plt.close(fig)


def fig5_architecture():
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.axis("off")

    def box(x, y, w, h, text, color="#eef2f7", fontsize=13.5):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.4, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3, linespacing=1.6)
        return (x, y, w, h)

    def arrow(b1, b2, **kwargs):
        x1 = b1[0] + b1[2]
        y1 = b1[1] + b1[3] / 2
        x2 = b2[0]
        y2 = b2[1] + b2[3] / 2
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", linewidth=1.8, color="black"))

    b_feat = box(0.0, 1.6, 1.8, 0.9, "feature_fn\ntransition_fn\nresidual_fn\ncost_fn\n(env-specific)")
    b_ens = box(2.1, 1.6, 2.0, 0.9, "GenericEnsembleDynamics\n(stacked-parameter\neinsum, E members)", color="#dce8fb")
    b_mppi = box(4.4, 1.6, 2.1, 0.9, "SeedBatchedGenericMPPI\n(S seeds x K rollouts,\nper-seed softmax)", color="#dcf3e4")
    b_gate = box(2.1, 0.2, 2.0, 0.9, "OOD gate\n(pilot rollout,\ncalibrated threshold)", color="#fbe8dc")
    b_env = box(6.9, 1.6, 1.6, 0.9, "pendulum /\nMuJoCo arm", color="#f5f5f5")

    arrow(b_feat, b_ens)
    arrow(b_ens, b_mppi)
    arrow(b_mppi, b_env)
    ax.annotate("", xy=(3.1, 1.6), xytext=(3.1, 1.1), arrowprops=dict(arrowstyle="->", linewidth=1.8, color="black"))
    ax.text(3.25, 1.32, "uncertainty", fontsize=12)
    ax.annotate("", xy=(4.7, 1.6), xytext=(4.1, 1.1), arrowprops=dict(arrowstyle="->", linewidth=1.8, color="black"))
    ax.text(4.15, -0.15, "rollout budget / fallback trigger", fontsize=12, ha="left")

    ax.set_xlim(-0.2, 8.7)
    ax.set_ylim(-0.4, 2.7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_architecture.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig1_ood_severity_curve()
    fig2_crossover()
    fig3_seed_batching_speedup()
    fig4_task_success_bars()
    fig5_architecture()
    print(f"Figures written to {OUT_DIR}")
