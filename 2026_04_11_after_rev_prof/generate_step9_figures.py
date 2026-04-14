#!/usr/bin/env python3
"""Generate Step 9 fix comparison figures. Adds to the existing analysis."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LENGTHS = [100, 1000, 10000, 100000]
STEP_COLORS = {"Step1": "#4a90e2", "Step2": "#d44a4a", "Step9fix": "#2aa34e"}

# Load all step data
cols = ["len", "label", "fml_s", "mf_s", "ts_s", "total_s", "cpu_s", "lnl", "iters"]
s12 = pd.read_csv(os.path.join(HERE, "step12_metrics.csv"), header=None, names=cols)
s9  = pd.read_csv(os.path.join(HERE, "step9fix_metrics.csv"), header=None, names=cols)
all_steps = pd.concat([s12, s9], ignore_index=True)
all_steps["nlen"] = all_steps["len"].str.replace("len_", "").astype(int)
all_steps[["kernel", "step"]] = all_steps["label"].str.split("_", n=1, expand=True)

# Also load baseline
bcols = ["len", "kernel", "backend", "fml_s", "mf_s", "ts_s", "total_s",
         "cpu_s", "lnl", "iters"]
baseline = pd.read_csv(os.path.join(HERE, "baseline_2026-04-03.csv"),
                       header=None, names=bcols)
baseline["nlen"] = baseline["len"].str.replace("len_", "").astype(int)


def get_val(df, nlen, kernel, step, col="total_s"):
    row = df[(df.nlen == nlen) & (df.kernel == kernel) & (df.step == step)]
    return row[col].values[0] if len(row) else np.nan


# ============================================================================
# Figure 7 — Step 1 vs Step 2 vs Step 9 fix (REV), grouped bars
# ============================================================================
def fig7():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(LENGTHS))
    width = 0.25
    for offset, step in zip([-width, 0, width], ["Step1", "Step2", "Step9fix"]):
        vals = [get_val(all_steps, L, "REV", step) for L in LENGTHS]
        ax.bar(x + offset, vals, width, label=step, color=STEP_COLORS[step],
               edgecolor="black")
    # annotate S9fix vs S1
    for i, L in enumerate(LENGTHS):
        s1 = get_val(all_steps, L, "REV", "Step1")
        s9 = get_val(all_steps, L, "REV", "Step9fix")
        if not (np.isnan(s1) or np.isnan(s9)):
            pct = (s9 / s1 - 1) * 100
            col = "red" if pct > 3 else ("green" if pct < -3 else "black")
            ax.annotate(f"S9/S1: {pct:+.1f}%", xy=(i + width, s9),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8, color=col, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{L:,}" for L in LENGTHS])
    ax.set_xlabel("Alignment length (sites)")
    ax.set_ylabel("Total wall-clock time (s)")
    ax.set_title("REV progression: Step 1 → Step 2 → Step 9 fix — DNA 100 taxa, GPU V100")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig07_step9fix_rev_progression.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig07_step9fix_rev_progression.png")


# ============================================================================
# Figure 8 — Delta % chart: S9fix/S1, S2/S1 side by side + NONREV noise
# ============================================================================
def fig8():
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(LENGTHS))
    width = 0.2

    # Compute deltas
    s2_rev, s9_rev, nonrev_noise = [], [], []
    for L in LENGTHS:
        s1r = get_val(all_steps, L, "REV", "Step1")
        s2r = get_val(all_steps, L, "REV", "Step2")
        s9r = get_val(all_steps, L, "REV", "Step9fix")
        s1n = get_val(all_steps, L, "NONREV", "Step1")
        s9n = get_val(all_steps, L, "NONREV", "Step9fix")
        s2_rev.append((s2r / s1r - 1) * 100 if s1r and s2r else np.nan)
        s9_rev.append((s9r / s1r - 1) * 100 if s1r and s9r else np.nan)
        nonrev_noise.append((s9n / s1n - 1) * 100 if s1n and s9n else np.nan)

    noise_max = max(abs(d) for d in nonrev_noise if not np.isnan(d))
    ax.axhspan(-noise_max, noise_max, color="gray", alpha=0.15,
               label=f"NONREV noise (±{noise_max:.1f}%)")
    ax.axhline(0, color="black", linewidth=0.8)

    def bar_colors(vals, noise):
        return ["#d44a4a" if v > noise else ("#2aa34e" if v < -noise else "#bfbfbf")
                for v in vals]

    ax.bar(x - width, s2_rev, width, color=bar_colors(s2_rev, noise_max),
           edgecolor="black", label="Step 2 / Step 1")
    ax.bar(x, s9_rev, width, color=bar_colors(s9_rev, noise_max),
           edgecolor="black", hatch="///", label="Step 9 fix / Step 1")
    ax.bar(x + width, nonrev_noise, width, color="#bfbfbf",
           edgecolor="black", hatch="...", label="NONREV noise (S9fix/S1)")

    for bars_x, vals in [(x - width, s2_rev), (x, s9_rev), (x + width, nonrev_noise)]:
        for bx, v in zip(bars_x, vals):
            if np.isnan(v): continue
            ax.annotate(f"{v:+.1f}%", xy=(bx, v),
                        xytext=(0, 6 if v >= 0 else -14),
                        textcoords="offset points", ha="center",
                        fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{L:,}" for L in LENGTHS])
    ax.set_xlabel("Alignment length (sites)")
    ax.set_ylabel("Δ total wall vs Step 1 (%)")
    ax.set_title("REV cumulative delta — Step 2 (vs S1) and Step 9 fix (vs S1)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig08_step9fix_delta_vs_step1.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig08_step9fix_delta_vs_step1.png")


# ============================================================================
# Figure 9 — Phase breakdown: Step 1 vs Step 2 vs Step 9 fix (REV only)
# ============================================================================
def fig9():
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.5), sharey=False)
    phases = ["fml_s", "mf_s", "ts_s"]
    phase_labels = ["Fast ML", "ModelFinder", "Tree search"]
    phase_colors = ["#a8d5a2", "#f4c16d", "#8b4fd4"]

    for ax, L in zip(axes, LENGTHS):
        x_positions = np.arange(3)
        for step_idx, step in enumerate(["Step1", "Step2", "Step9fix"]):
            sub = all_steps[(all_steps.nlen == L) & (all_steps.kernel == "REV") &
                            (all_steps.step == step)]
            if sub.empty: continue
            bottoms = 0.0
            for phase, plabel, pcol in zip(phases, phase_labels, phase_colors):
                h = sub[phase].values[0]
                ax.bar(step_idx, h, 0.6, bottom=bottoms, color=pcol,
                       edgecolor="black",
                       label=plabel if (L == LENGTHS[0] and step_idx == 0) else "")
                if h > 20:
                    ax.annotate(f"{h:.0f}", xy=(step_idx, bottoms + h / 2),
                                ha="center", va="center", fontsize=7)
                bottoms += h
            ax.annotate(f"Σ={bottoms:.0f}", xy=(step_idx, bottoms),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["S1", "S2", "S9fix"], fontsize=9)
        ax.set_title(f"len_{L:,}")
        ax.grid(True, axis="y", alpha=0.3)
        if L == LENGTHS[0]:
            ax.set_ylabel("Wall-clock time (s)")
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("REV phase breakdown — Step 1 → Step 2 → Step 9 fix", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig09_step9fix_phase_breakdown.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig09_step9fix_phase_breakdown.png")


# ============================================================================
# Figure 10 — Step 9 fix vs baseline landscape (CPU + GPU)
# ============================================================================
def fig10():
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(LENGTHS))
    width = 0.13
    series = [
        ("CPU_1",  "#888888"), ("CPU_10", "#4a90e2"),
        ("CPU_48", "#1f3a6b"), ("GPU baseline", "#b8b8b8"),
    ]
    offsets = np.linspace(-2.5 * width, 1.5 * width, len(series) + 1)

    for (label, colour), off in zip(series, offsets[:-1]):
        backend = label.replace(" baseline", "_V100") if "GPU" in label else label
        vals = []
        for L in LENGTHS:
            row = baseline[(baseline.kernel == "REV") &
                           (baseline.backend == backend) &
                           (baseline.nlen == L)].total_s.values
            vals.append(row[0] if len(row) else np.nan)
        ax.bar(x + off, vals, width, label=label, color=colour, edgecolor="black")

    s9_vals = [get_val(all_steps, L, "REV", "Step9fix") for L in LENGTHS]
    ax.bar(x + offsets[-1], s9_vals, width,
           label="GPU Step 9 fix", color="#2aa34e", edgecolor="black")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{L:,}" for L in LENGTHS])
    ax.set_xlabel("Alignment length (sites)")
    ax.set_ylabel("Total wall-clock time (s), log scale")
    ax.set_title("REV — Step 9 fix in the CPU/GPU landscape")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig10_step9fix_vs_baseline.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig10_step9fix_vs_baseline.png")


if __name__ == "__main__":
    fig7()
    fig8()
    fig9()
    fig10()
    print("\nAll Step 9 fix figures written to:", HERE)
