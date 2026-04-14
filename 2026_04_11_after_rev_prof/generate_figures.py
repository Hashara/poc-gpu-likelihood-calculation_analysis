#!/usr/bin/env python3
"""
Figure generator for the 2026-04-11 Step 1 + Step 2 analysis.

Produces:
    fig01_cpu_vs_gpu_baseline.png   — CPU (1/10/48t) vs GPU baseline by length
    fig02_step1_vs_step2_walls.png  — Total wall, Step 1 vs Step 2, per length
    fig03_delta_percent.png         — Step 2 / Step 1 delta % with NONREV noise band
    fig04_gpu_rev_progression.png   — 2026-04-03 -> Step 1 -> Step 2, REV GPU
    fig05_phase_breakdown.png       — Fast ML / ModelFinder / Tree search stacks
    fig06_step2_vs_baseline.png     — 2026-04-03 baseline vs Step 2 per backend

Running this as a script regenerates all PNGs in the same directory.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Load data ---
BASE_COLS = ["len", "kernel", "backend", "fml_s", "mf_s", "ts_s", "total_s",
             "cpu_s", "lnl", "iters"]
STEP_COLS = ["len", "label", "fml_s", "mf_s", "ts_s", "total_s",
             "cpu_s", "lnl", "iters"]

baseline = pd.read_csv(os.path.join(HERE, "baseline_2026-04-03.csv"),
                       header=None, names=BASE_COLS)
step12 = pd.read_csv(os.path.join(HERE, "step12_metrics.csv"),
                     header=None, names=STEP_COLS)

# Numeric length (strip "len_" prefix)
baseline["nlen"] = baseline["len"].str.replace("len_", "").astype(int)
step12["nlen"] = step12["len"].str.replace("len_", "").astype(int)

# Split step12 label into kernel + step
step12[["kernel", "step"]] = step12["label"].str.split("_", expand=True)

LENGTHS = [100, 1000, 10000, 100000]
BACKENDS = ["CPU_1", "CPU_10", "CPU_48", "GPU_V100"]

# Consistent colour scheme
BACKEND_COLORS = {
    "CPU_1":    "#888888",
    "CPU_10":   "#4a90e2",
    "CPU_48":   "#1f3a6b",
    "GPU_V100": "#d44a4a",
}
STEP_COLORS = {"Step1": "#4a90e2", "Step2": "#d44a4a"}
KERNEL_HATCH = {"REV": "", "NONREV": "///"}


# ============================================================================
# Figure 1 — CPU vs GPU baseline (2026-04-03), by length, per kernel
# ============================================================================
def fig1_cpu_vs_gpu_baseline():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, kernel in zip(axes, ["REV", "NONREV"]):
        for backend in BACKENDS:
            sub = baseline[(baseline["kernel"] == kernel) &
                           (baseline["backend"] == backend)].sort_values("nlen")
            if sub.empty:
                continue
            ax.plot(sub["nlen"], sub["total_s"], "o-",
                    label=backend, color=BACKEND_COLORS[backend], linewidth=2, markersize=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Alignment length (sites)")
        ax.set_title(f"2026-04-03 baseline — {kernel}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Total wall-clock time (s), log scale")
    axes[0].legend(loc="best", frameon=True)
    fig.suptitle("DNA 100 taxa — CPU vs GPU total wall time (baseline)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig01_cpu_vs_gpu_baseline.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig01_cpu_vs_gpu_baseline.png")


# ============================================================================
# Figure 2 — Step 1 vs Step 2 grouped bars, per length, per kernel
# ============================================================================
def fig2_step1_vs_step2_walls():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    x = np.arange(len(LENGTHS))
    width = 0.35

    for ax, kernel in zip(axes, ["REV", "NONREV"]):
        s1_vals = []
        s2_vals = []
        for L in LENGTHS:
            s1 = step12[(step12["nlen"] == L) & (step12["kernel"] == kernel) &
                        (step12["step"] == "Step1")]["total_s"].values
            s2 = step12[(step12["nlen"] == L) & (step12["kernel"] == kernel) &
                        (step12["step"] == "Step2")]["total_s"].values
            s1_vals.append(s1[0] if len(s1) else np.nan)
            s2_vals.append(s2[0] if len(s2) else np.nan)

        ax.bar(x - width/2, s1_vals, width,
               label="Step 1", color=STEP_COLORS["Step1"], edgecolor="black")
        ax.bar(x + width/2, s2_vals, width,
               label="Step 2", color=STEP_COLORS["Step2"], edgecolor="black")

        # annotate delta %
        for i, (s1, s2) in enumerate(zip(s1_vals, s2_vals)):
            if not (np.isnan(s1) or np.isnan(s2)):
                delta_pct = (s2 - s1) / s1 * 100
                colour = "red" if delta_pct > 2 else ("green" if delta_pct < -2 else "black")
                ax.annotate(f"{delta_pct:+.1f}%",
                            xy=(i, max(s1, s2) + 15),
                            ha="center", fontsize=9, color=colour, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{L:,}" for L in LENGTHS])
        ax.set_xlabel("Alignment length (sites)")
        ax.set_title(f"{kernel} (GPU V100)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Total wall-clock time (s)")
    fig.suptitle(
        "Step 1 vs Step 2 — DNA 100 taxa, GPU V100, full tree search (2026-04-11)",
        fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig02_step1_vs_step2_walls.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig02_step1_vs_step2_walls.png")


# ============================================================================
# Figure 3 — Delta % (Step 2 / Step 1 - 1) with NONREV noise band
# ============================================================================
def fig3_delta_percent():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(LENGTHS))
    width = 0.35

    rev_delta = []
    nonrev_delta = []
    for L in LENGTHS:
        for kernel, bucket in [("REV", rev_delta), ("NONREV", nonrev_delta)]:
            s1 = step12[(step12["nlen"] == L) & (step12["kernel"] == kernel) &
                        (step12["step"] == "Step1")]["total_s"].values
            s2 = step12[(step12["nlen"] == L) & (step12["kernel"] == kernel) &
                        (step12["step"] == "Step2")]["total_s"].values
            if len(s1) and len(s2):
                bucket.append((s2[0] - s1[0]) / s1[0] * 100)
            else:
                bucket.append(np.nan)

    # Noise band from NONREV deltas (the control)
    noise = max(abs(d) for d in nonrev_delta if not np.isnan(d))
    ax.axhspan(-noise, noise, color="gray", alpha=0.18,
               label=f"NONREV noise band (±{noise:.1f}%)")
    ax.axhline(0, color="black", linewidth=0.8)

    bars_rev = ax.bar(x - width/2, rev_delta, width,
                      color=["#d44a4a" if d > noise else ("#2aa34e" if d < -noise else "#bfbfbf") for d in rev_delta],
                      edgecolor="black", label="REV")
    bars_nonrev = ax.bar(x + width/2, nonrev_delta, width,
                         color="#bfbfbf", edgecolor="black", hatch="///",
                         label="NONREV (control)")

    for bars, vals in [(bars_rev, rev_delta), (bars_nonrev, nonrev_delta)]:
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.annotate(f"{v:+.1f}%", xy=(b.get_x() + b.get_width()/2, v),
                        xytext=(0, 6 if v >= 0 else -14),
                        textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{L:,}" for L in LENGTHS])
    ax.set_xlabel("Alignment length (sites)")
    ax.set_ylabel("Δ wall-clock time  (Step 2 − Step 1) / Step 1  (%)")
    ax.set_title("Step 2 vs Step 1 — signal per length\n(NONREV is the noise control — both steps are identical for NONREV)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center")

    # Annotation explaining what the colours mean for REV
    ax.text(0.02, 0.98,
            "REV bar colors:\n"
            "  red   = regression outside noise\n"
            "  green = win outside noise\n"
            "  gray  = within noise band",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig03_delta_percent.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig03_delta_percent.png")


# ============================================================================
# Figure 4 — GPU REV progression: 2026-04-03 -> Step 1 -> Step 2
# ============================================================================
def fig4_gpu_rev_progression():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(LENGTHS))
    width = 0.25

    base_vals = []
    s1_vals = []
    s2_vals = []
    for L in LENGTHS:
        b = baseline[(baseline["kernel"] == "REV") &
                     (baseline["backend"] == "GPU_V100") &
                     (baseline["nlen"] == L)]["total_s"].values
        s1 = step12[(step12["nlen"] == L) & (step12["kernel"] == "REV") &
                    (step12["step"] == "Step1")]["total_s"].values
        s2 = step12[(step12["nlen"] == L) & (step12["kernel"] == "REV") &
                    (step12["step"] == "Step2")]["total_s"].values
        base_vals.append(b[0] if len(b) else np.nan)
        s1_vals.append(s1[0] if len(s1) else np.nan)
        s2_vals.append(s2[0] if len(s2) else np.nan)

    ax.bar(x - width, base_vals, width, label="2026-04-03 baseline",
           color="#9a9a9a", edgecolor="black")
    ax.bar(x,         s1_vals,  width, label="2026-04-11 Step 1",
           color=STEP_COLORS["Step1"], edgecolor="black")
    ax.bar(x + width, s2_vals,  width, label="2026-04-11 Step 2",
           color=STEP_COLORS["Step2"], edgecolor="black")

    for i, (b, s1, s2) in enumerate(zip(base_vals, s1_vals, s2_vals)):
        if not np.isnan(s2) and not np.isnan(b):
            pct = (s2 - b) / b * 100
            colour = "red" if pct > 2 else ("green" if pct < -2 else "black")
            ax.annotate(f"{pct:+.0f}% vs base", xy=(i + width, s2),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8, color=colour)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{L:,}" for L in LENGTHS])
    ax.set_xlabel("Alignment length (sites)")
    ax.set_ylabel("Total wall-clock time (s)")
    ax.set_title("GPU REV progression — DNA 100 taxa")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig04_gpu_rev_progression.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig04_gpu_rev_progression.png")


# ============================================================================
# Figure 5 — Phase breakdown (Fast ML / ModelFinder / Tree search) for REV
# ============================================================================
def fig5_phase_breakdown():
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    phases = ["fml_s", "mf_s", "ts_s"]
    phase_labels = ["Fast ML", "ModelFinder", "Tree search"]
    phase_colors = ["#a8d5a2", "#f4c16d", "#8b4fd4"]

    for ax, L in zip(axes, LENGTHS):
        sub = step12[(step12["nlen"] == L) & (step12["kernel"] == "REV")]
        if sub.empty:
            continue
        x = np.arange(2)  # Step 1, Step 2
        bottoms = np.zeros(2)
        for phase, label, colour in zip(phases, phase_labels, phase_colors):
            heights = [
                sub[sub["step"] == "Step1"][phase].values[0] if not sub[sub["step"] == "Step1"].empty else 0,
                sub[sub["step"] == "Step2"][phase].values[0] if not sub[sub["step"] == "Step2"].empty else 0,
            ]
            ax.bar(x, heights, 0.6, bottom=bottoms, color=colour,
                   edgecolor="black", label=label)
            # annotate if segment tall enough
            for i, (h, b) in enumerate(zip(heights, bottoms)):
                if h > max(heights) * 0.05:
                    ax.annotate(f"{h:.0f}", xy=(x[i], b + h/2), ha="center",
                                va="center", fontsize=8, color="black")
            bottoms += np.array(heights)
        # Total annotation
        for i, t in enumerate(bottoms):
            ax.annotate(f"Σ={t:.0f}s", xy=(x[i], t), xytext=(0, 4),
                        textcoords="offset points", ha="center",
                        fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Step 1", "Step 2"])
        ax.set_title(f"len_{L:,}")
        ax.grid(True, axis="y", alpha=0.3)
        if L == LENGTHS[0]:
            ax.set_ylabel("Wall-clock time (s)")
            ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "REV phase breakdown — Fast ML / ModelFinder / Tree search — Step 1 vs Step 2",
        fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig05_phase_breakdown.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig05_phase_breakdown.png")


# ============================================================================
# Figure 6 — Step 2 vs 2026-04-03 baseline, all backends, per length
# ============================================================================
def fig6_step2_vs_baseline():
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(LENGTHS))
    width = 0.13

    series = [
        ("CPU_1",             baseline, "CPU_1",    "#888888"),
        ("CPU_10",            baseline, "CPU_10",   "#4a90e2"),
        ("CPU_48",            baseline, "CPU_48",   "#1f3a6b"),
        ("GPU_V100 baseline", baseline, "GPU_V100", "#b8b8b8"),
    ]
    offsets = np.linspace(-2.5*width, 1.5*width, len(series) + 1)

    for (label, source, backend, colour), off in zip(series, offsets[:-1]):
        vals = []
        for L in LENGTHS:
            row = source[(source["kernel"] == "REV") &
                         (source["backend"] == backend) &
                         (source["nlen"] == L)]["total_s"].values
            vals.append(row[0] if len(row) else np.nan)
        ax.bar(x + off, vals, width, label=label, color=colour, edgecolor="black")

    # Step 2 REV on top
    step2_vals = []
    for L in LENGTHS:
        row = step12[(step12["kernel"] == "REV") &
                     (step12["step"] == "Step2") &
                     (step12["nlen"] == L)]["total_s"].values
        step2_vals.append(row[0] if len(row) else np.nan)
    ax.bar(x + offsets[-1], step2_vals, width,
           label="GPU_V100 Step 2 (2026-04-11)",
           color="#d44a4a", edgecolor="black")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{L:,}" for L in LENGTHS])
    ax.set_xlabel("Alignment length (sites)")
    ax.set_ylabel("Total wall-clock time (s), log scale")
    ax.set_title(
        "REV — where does Step 2 land in the CPU/GPU performance landscape?")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig06_step2_vs_baseline.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig06_step2_vs_baseline.png")


def main():
    fig1_cpu_vs_gpu_baseline()
    fig2_step1_vs_step2_walls()
    fig3_delta_percent()
    fig4_gpu_rev_progression()
    fig5_phase_breakdown()
    fig6_step2_vs_baseline()
    print("\nAll figures written to:", HERE)


if __name__ == "__main__":
    main()
