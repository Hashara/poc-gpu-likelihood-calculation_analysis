#!/usr/bin/env python3
"""Generate VANILA vs OPENACC comparison diagrams for v4 results."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# --- Configuration ---
CSV_PATH = os.path.join(os.path.dirname(__file__), "v4_results.csv")
OUT_DIR = os.path.dirname(__file__)

TREE_ORDER = [f"tree_{i}" for i in range(1, 11)]

GROUPS = [
    ("DNA", "rooted"),
    ("DNA", "unrooted"),
    ("AA", "rooted"),
    ("AA", "unrooted"),
]

OUTPUT_FILES = {
    ("DNA", "rooted"):   "DNA_rooted_vanila_openacc_comparison.png",
    ("DNA", "unrooted"): "DNA_unrooted_vanila_openacc_comparison.png",
    ("AA", "rooted"):    "AA_rooted_vanila_openacc_comparison.png",
    ("AA", "unrooted"):  "AA_unrooted_vanila_openacc_comparison.png",
}

# Difference thresholds
THRESH_GREEN = 1e-6
THRESH_YELLOW = 0.01


def classify_diff(diff):
    """Return color based on absolute difference magnitude."""
    ad = abs(diff)
    if ad < THRESH_GREEN:
        return "#2ca02c"  # green
    elif ad < THRESH_YELLOW:
        return "#f0ad4e"  # yellow/amber
    else:
        return "#d9534f"  # red


def generate_figure(df_group, data_type, tree_type):
    """Generate a comparison figure for one (data_type, tree_type) group."""
    models = sorted(df_group["model"].unique())
    n_models = len(models)

    # Layout: for each model, one bar-chart row + one diff row
    fig_height = max(4 * n_models + 2, 10)
    fig_width = 16

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.suptitle(
        f"{data_type} / {tree_type} — VANILA vs OPENACC Log-Likelihood Comparison",
        fontsize=16, fontweight="bold", y=0.995,
    )

    # Collect summary info
    summary_lines = []

    # Create a gridspec: 2 rows per model (bar + diff)
    gs = gridspec.GridSpec(
        n_models * 2, 1,
        height_ratios=[3, 1] * n_models,
        hspace=0.45,
    )

    for m_idx, model in enumerate(models):
        df_model = df_group[df_group["model"] == model]

        # Pivot to get VANILA and OPENACC values per tree
        vanila_vals = []
        openacc_vals = []
        diffs = []

        for tree in TREE_ORDER:
            v_row = df_model[(df_model["tree_num"] == tree) & (df_model["build"] == "VANILA")]
            o_row = df_model[(df_model["tree_num"] == tree) & (df_model["build"] == "OPENACC")]

            v_ll = v_row["log_likelihood"].values[0] if len(v_row) > 0 else np.nan
            o_ll = o_row["log_likelihood"].values[0] if len(o_row) > 0 else np.nan
            vanila_vals.append(v_ll)
            openacc_vals.append(o_ll)
            diffs.append(o_ll - v_ll if not (np.isnan(v_ll) or np.isnan(o_ll)) else np.nan)

        max_abs_diff = max(abs(d) for d in diffs if not np.isnan(d))

        # Summary
        if max_abs_diff < THRESH_GREEN:
            summary_lines.append(f"  {model}: MATCH (max |diff| = {max_abs_diff:.2e})")
        elif max_abs_diff < THRESH_YELLOW:
            summary_lines.append(f"  {model}: SMALL DIFF (max |diff| = {max_abs_diff:.2e})")
        else:
            summary_lines.append(f"  {model}: SIGNIFICANT DIFF (max |diff| = {max_abs_diff:.4f})")

        # --- Bar chart ---
        ax_bar = fig.add_subplot(gs[m_idx * 2])
        x = np.arange(len(TREE_ORDER))
        bar_width = 0.35

        bars_v = ax_bar.bar(x - bar_width / 2, vanila_vals, bar_width,
                            label="VANILA", color="#1f77b4", alpha=0.85)
        bars_o = ax_bar.bar(x + bar_width / 2, openacc_vals, bar_width,
                            label="OPENACC", color="#ff7f0e", alpha=0.85)

        ax_bar.set_ylabel("Log-Likelihood", fontsize=10)
        ax_bar.set_title(f"Model: {model}", fontsize=12, fontweight="bold")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(TREE_ORDER, fontsize=8, rotation=45, ha="right")
        ax_bar.legend(fontsize=9, loc="upper right")
        ax_bar.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar_set in [bars_v, bars_o]:
            for bar in bar_set:
                height = bar.get_height()
                if not np.isnan(height):
                    ax_bar.annotate(
                        f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, -12 if height < 0 else 3),
                        textcoords="offset points",
                        ha="center", va="bottom" if height >= 0 else "top",
                        fontsize=6, rotation=90,
                    )

        # --- Diff row ---
        ax_diff = fig.add_subplot(gs[m_idx * 2 + 1])
        diff_colors = [classify_diff(d) if not np.isnan(d) else "#cccccc" for d in diffs]
        ax_diff.bar(x, diffs, 0.6, color=diff_colors, edgecolor="black", linewidth=0.5)
        ax_diff.set_ylabel("OPENACC - VANILA", fontsize=8)
        ax_diff.set_xticks(x)
        ax_diff.set_xticklabels(TREE_ORDER, fontsize=8, rotation=45, ha="right")
        ax_diff.axhline(y=0, color="black", linewidth=0.5)
        ax_diff.grid(axis="y", alpha=0.3)

        # Add diff value labels
        for i, d in enumerate(diffs):
            if not np.isnan(d):
                label = f"{d:.4f}" if abs(d) >= 0.0001 else f"{d:.2e}"
                ax_diff.annotate(
                    label,
                    xy=(i, d),
                    xytext=(0, 3 if d >= 0 else -10),
                    textcoords="offset points",
                    ha="center", va="bottom" if d >= 0 else "top",
                    fontsize=6, fontweight="bold",
                )

        # Legend for diff colors
        if m_idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#2ca02c", edgecolor="black", label=f"|diff| < {THRESH_GREEN:.0e} (match)"),
                Patch(facecolor="#f0ad4e", edgecolor="black", label=f"{THRESH_GREEN:.0e} <= |diff| < {THRESH_YELLOW} (small)"),
                Patch(facecolor="#d9534f", edgecolor="black", label=f"|diff| >= {THRESH_YELLOW} (significant)"),
            ]
            ax_diff.legend(handles=legend_elements, fontsize=7, loc="upper right", ncol=3)

    # Add summary text at top
    summary_text = "Summary:\n" + "\n".join(summary_lines)
    fig.text(
        0.02, 0.99, summary_text,
        transform=fig.transFigure,
        fontsize=9, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.8),
    )

    fig.subplots_adjust(top=0.94 - 0.01 * n_models, bottom=0.04, left=0.06, right=0.97)
    return fig


def main():
    df = pd.read_csv(CSV_PATH)

    # Filter to non-verbose only
    df = df[df["is_verbose"] == False].copy()

    for data_type, tree_type in GROUPS:
        df_group = df[(df["data_type"] == data_type) & (df["tree_type"] == tree_type)]
        if df_group.empty:
            print(f"No data for {data_type}/{tree_type}, skipping.")
            continue

        fig = generate_figure(df_group, data_type, tree_type)
        out_path = os.path.join(OUT_DIR, OUTPUT_FILES[(data_type, tree_type)])
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
