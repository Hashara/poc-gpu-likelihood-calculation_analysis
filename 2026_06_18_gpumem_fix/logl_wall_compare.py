"""Side-by-side log-likelihood + wall-time check across the gpumem_fix reps
vs the 2026_06_13 baseline. Validates that the memory-saving build still
finds the same optimum (logL) and reports the wall delta.

Outputs:
  - logl_wall_table.csv  (machine-readable)
  - fig_logl_wall_compare.png (two-row panel: bar chart of wall + table-style
    logL listing)
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_CSV = Path("/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_13_morebenchmarks/runs.csv")
NEW_CSV  = Path("/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_18_gpumem_fix/runs.csv")
OUT_DIR  = NEW_CSV.parent

REP_RANK = {"test_3": 3, "test_2": 2, "test_1": 1}

# ---- gather: latest completed gpumem_fix per (datatype, sites, hardware) ---
new = pd.read_csv(NEW_CSV)
new = new[new["wall_s"].notna()].copy()
new["rep_rank"] = new["rep"].map(REP_RANK)
new = new.sort_values("rep_rank").groupby(
    ["datatype", "sites", "hardware"], as_index=False).tail(1)
new = new[["datatype", "sites", "hardware", "rep", "wall_s", "best_logL"]]
new = new.rename(columns={"wall_s": "fix_wall_s", "best_logL": "fix_logL",
                          "rep": "fix_rep"})

base = pd.read_csv(BASE_CSV)
base = base[(base["test"] == "MTEST")
            & base["hardware"].isin(["GPU_H200", "GPU_V100"])].copy()
base = base[["datatype", "sites", "hardware", "wall_total_s", "best_logL"]]
base = base.rename(columns={"wall_total_s": "base_wall_s",
                            "best_logL": "base_logL"})

cmp = base.merge(new, on=["datatype", "sites", "hardware"], how="outer")
cmp["wall_delta_s"]   = cmp["fix_wall_s"] - cmp["base_wall_s"]
cmp["wall_delta_pct"] = 100 * cmp["wall_delta_s"] / cmp["base_wall_s"]
cmp["logL_delta"]     = cmp["fix_logL"] - cmp["base_logL"]
cmp["match"] = np.where(
    cmp[["fix_logL", "base_logL"]].isna().any(axis=1),
    "—",
    np.where(np.isclose(cmp["fix_logL"], cmp["base_logL"], atol=1.0),
             "yes",
             np.where(np.isclose(cmp["fix_logL"], cmp["base_logL"], rtol=1e-9),
                      "yes", "diff")))

cmp = cmp.sort_values(["datatype", "sites", "hardware"]).reset_index(drop=True)
cmp.to_csv(OUT_DIR / "logl_wall_table.csv", index=False)

# pretty print
print("\n=== logL + wall-time agreement: gpumem_fix (latest rep) vs baseline ===")
show_cols = ["datatype","sites","hardware","fix_rep",
             "base_logL","fix_logL","logL_delta","match",
             "base_wall_s","fix_wall_s","wall_delta_pct"]
print(cmp[show_cols].to_string(index=False))

# ---- plot ------------------------------------------------------------------
plottable = cmp[cmp["base_wall_s"].notna() & cmp["fix_wall_s"].notna()].copy()
plottable["label"] = (plottable["datatype"] + " / "
                      + plottable["hardware"].str.replace("GPU_", "")
                      + " / " + plottable["sites"].astype(str).str.replace(
                          "1000000$", "1M", regex=True).str.replace(
                          "100000$", "100k", regex=True).str.replace(
                          "10000000$", "10M", regex=True))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# panel 1: wall-time bar
ax = axes[0]
x = np.arange(len(plottable))
w = 0.4
ax.bar(x - w/2, plottable["base_wall_s"] / 3600, w,
       label="2026_06_13 baseline", color="#888")
ax.bar(x + w/2, plottable["fix_wall_s"] / 3600, w,
       label="2026_06_18 gpumem_fix (latest rep)", color="#2ca02c")
ax.set_xticks(x); ax.set_xticklabels(plottable["label"], rotation=22, ha="right",
                                     fontsize=8)
ax.set_ylabel("Wall time (h)"); ax.set_yscale("log")
ax.set_title("Wall-clock time: gpumem_fix vs baseline (log scale)")
ax.grid(axis="y", which="both", alpha=0.3); ax.legend(fontsize=8)
for i, (b, f, pct) in enumerate(zip(plottable["base_wall_s"]/3600,
                                    plottable["fix_wall_s"]/3600,
                                    plottable["wall_delta_pct"])):
    ax.text(i, max(b, f) * 1.05, f"{pct:+.1f}%",
            ha="center", fontsize=7)

# panel 2: logL agreement (delta, signed, on linear axis)
ax = axes[1]
delta = plottable["logL_delta"].fillna(0)
colors = ["#2ca02c" if abs(d) < 1 else "#d62728" for d in delta]
ax.bar(x, delta, color=colors)
ax.set_xticks(x); ax.set_xticklabels(plottable["label"], rotation=22, ha="right",
                                     fontsize=8)
ax.set_ylabel("Δ logL (fix − base)")
ax.set_title("Log-likelihood agreement (delta vs baseline; |Δ|<1 → green)")
ax.grid(axis="y", alpha=0.3)
ax.axhline(0, color="k", linewidth=0.8)
for i, d in enumerate(delta):
    ax.text(i, d + (0.02 if d >= 0 else -0.05), f"{d:+.3f}",
            ha="center", fontsize=7)

fig.suptitle("gpumem_fix vs 2026_06_13 baseline — result correctness + run-time check")
fig.tight_layout()
out = OUT_DIR / "fig_logl_wall_compare.png"
fig.savefig(out, dpi=140); plt.close(fig)
print(f"\nWrote {out}, logl_wall_table.csv")
