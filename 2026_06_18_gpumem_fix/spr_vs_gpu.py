"""Intuitive three-way comparison: best-CPU (Intel SPR / OMP104) vs
OpenACC H200 (old build) vs OpenACC H200 (gpumem_fix latest rep).

Per cell (datatype × sites), shows:
  - Wall time (h), annotated with "Nx faster than SPR"
  - Peak memory: host RAM for SPR, VRAM for the GPUs
  - Total energy (MJ), annotated with "M% of SPR"

Outputs:
  - spr_vs_gpu_table.csv
  - fig_spr_vs_gpu.png  (3 panels)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_CSV = Path("/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_13_morebenchmarks/runs.csv")
NEW_CSV  = Path("/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_18_gpumem_fix/runs.csv")
OUT_DIR  = NEW_CSV.parent

CELLS = [  # (datatype, sites, tag) — only cells where SPR completed
    ("AA",  100_000,   "AA 100k"),
    ("DNA", 100_000,   "DNA 100k"),
    ("AA",  1_000_000, "AA 1M"),
    ("DNA", 1_000_000, "DNA 1M"),
    ("DNA", 10_000_000, "DNA 10M"),
]
REP_RANK = {"test_3": 3, "test_2": 2, "test_1": 1}

base = pd.read_csv(BASE_CSV)
base = base[base["test"] == "MTEST"]

new = pd.read_csv(NEW_CSV)
new = new[new["wall_s"].notna()].copy()
new["rep_rank"] = new["rep"].map(REP_RANK)

rows = []
for dt, sites, tag in CELLS:
    spr = base[(base["datatype"] == dt) & (base["sites"] == sites)
               & (base["hardware"] == "CPU_SPR_OMP104")].iloc[0]
    h200_old = base[(base["datatype"] == dt) & (base["sites"] == sites)
                    & (base["hardware"] == "GPU_H200")].iloc[0]
    fix = new[(new["datatype"] == dt) & (new["sites"] == sites)
              & (new["hardware"] == "GPU_H200")].sort_values("rep_rank")
    if fix.empty:
        fix_row, fix_rep = None, "—"
    else:
        fix_row = fix.iloc[-1]; fix_rep = fix_row["rep"]
    rows.append({
        "cell": tag, "datatype": dt, "sites": sites,
        "fix_rep": fix_rep,
        "spr_wall_h":       spr["wall_total_s"] / 3600,
        "h200_old_wall_h":  h200_old["wall_total_s"] / 3600,
        "h200_fix_wall_h":  fix_row["wall_s"] / 3600 if fix_row is not None else np.nan,
        "spr_mem_gb":       spr["cpu_mem_peak_MB"] / 1024,
        "h200_old_mem_gb":  h200_old["gpu_mem_peak_MB"] / 1024,
        "h200_fix_mem_gb":  fix_row["gpu_mem_peak_MB"] / 1024 if fix_row is not None else np.nan,
        "spr_energy_mj":      spr["energy_total_J"] / 1e6,
        "h200_old_energy_mj": h200_old["energy_total_J"] / 1e6,
        "h200_fix_energy_mj": (fix_row["energy_total_J"] / 1e6
                              if fix_row is not None else np.nan),
    })
df = pd.DataFrame(rows)
df["speedup_old"] = df["spr_wall_h"] / df["h200_old_wall_h"]
df["speedup_fix"] = df["spr_wall_h"] / df["h200_fix_wall_h"]
df["energy_pct_old"] = 100 * df["h200_old_energy_mj"] / df["spr_energy_mj"]
df["energy_pct_fix"] = 100 * df["h200_fix_energy_mj"] / df["spr_energy_mj"]
df["mem_pct_old"]    = 100 * df["h200_old_mem_gb"] / df["spr_mem_gb"]
df["mem_pct_fix"]    = 100 * df["h200_fix_mem_gb"] / df["spr_mem_gb"]
df.to_csv(OUT_DIR / "spr_vs_gpu_table.csv", index=False)

print("\n=== SPR vs H200 (old) vs H200 (gpumem_fix latest) — MTEST ===")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print(df[["cell","fix_rep",
          "spr_wall_h","h200_old_wall_h","h200_fix_wall_h","speedup_old","speedup_fix",
          "spr_mem_gb","h200_old_mem_gb","h200_fix_mem_gb",
          "spr_energy_mj","h200_old_energy_mj","h200_fix_energy_mj",
          "energy_pct_old","energy_pct_fix"]].to_string(index=False))

# ----- plot -----------------------------------------------------------------
CELL_LABELS = [f"{r['cell']}\n(fix={r['fix_rep']})" for _, r in df.iterrows()]
x = np.arange(len(df)); w = 0.27
C_SPR = "#55a868"; C_OLD = "#c44e52"; C_FIX = "#2ca02c"

fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

# ---- 1. wall (h) on log axis -----------------------------------------------
ax = axes[0]
ax.bar(x - w, df["spr_wall_h"],      w, label="CPU SPR (OMP104)",            color=C_SPR)
ax.bar(x,     df["h200_old_wall_h"], w, label="OpenACC H200 (old)",          color=C_OLD)
ax.bar(x + w, df["h200_fix_wall_h"], w, label="OpenACC H200 (gpumem_fix)",   color=C_FIX)
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(CELL_LABELS, fontsize=9)
ax.set_ylabel("Wall time (hours, log scale)")
ax.set_title("Wall time — and how much faster the GPU is than SPR")
ax.grid(axis="y", which="both", alpha=0.3); ax.legend(fontsize=8)
for i, r in df.iterrows():
    ax.text(i - w, r["spr_wall_h"] * 1.06,        f"{r['spr_wall_h']:.2f} h",
            ha="center", fontsize=7)
    ax.text(i,     r["h200_old_wall_h"] * 1.06,  f"{r['h200_old_wall_h']:.2f} h\n{r['speedup_old']:.1f}×",
            ha="center", fontsize=7)
    ax.text(i + w, r["h200_fix_wall_h"] * 1.06,  f"{r['h200_fix_wall_h']:.2f} h\n{r['speedup_fix']:.1f}×",
            ha="center", fontsize=7)

# ---- 2. memory (GB) – host vs VRAM, log axis -------------------------------
ax = axes[1]
ax.bar(x - w, df["spr_mem_gb"],      w, label="SPR host RAM",        color=C_SPR)
ax.bar(x,     df["h200_old_mem_gb"], w, label="H200 VRAM (old)",     color=C_OLD)
ax.bar(x + w, df["h200_fix_mem_gb"], w, label="H200 VRAM (fix)",     color=C_FIX)
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(CELL_LABELS, fontsize=9)
ax.set_ylabel("Peak memory (GB, log scale)")
ax.set_title("Peak memory — SPR uses host RAM, GPUs use VRAM")
ax.grid(axis="y", which="both", alpha=0.3); ax.legend(fontsize=8)
for i, r in df.iterrows():
    ax.text(i - w, r["spr_mem_gb"]      * 1.10, f"{r['spr_mem_gb']:.1f}",      ha="center", fontsize=7)
    ax.text(i,     r["h200_old_mem_gb"] * 1.10, f"{r['h200_old_mem_gb']:.1f}", ha="center", fontsize=7)
    ax.text(i + w, r["h200_fix_mem_gb"] * 1.10,
            f"{r['h200_fix_mem_gb']:.1f}\n({100*(r['h200_fix_mem_gb']-r['h200_old_mem_gb'])/r['h200_old_mem_gb']:+.0f}%)",
            ha="center", fontsize=7)

# ---- 3. energy (MJ), with "fraction of SPR" labels -------------------------
ax = axes[2]
ax.bar(x - w, df["spr_energy_mj"],      w, label="SPR total energy",         color=C_SPR)
ax.bar(x,     df["h200_old_energy_mj"], w, label="H200 total (old)",         color=C_OLD)
ax.bar(x + w, df["h200_fix_energy_mj"], w, label="H200 total (fix)",         color=C_FIX)
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(CELL_LABELS, fontsize=9)
ax.set_ylabel("Energy (MJ, log scale)")
ax.set_title("Energy — and what % of SPR's energy the GPU uses")
ax.grid(axis="y", which="both", alpha=0.3); ax.legend(fontsize=8)
for i, r in df.iterrows():
    ax.text(i - w, r["spr_energy_mj"] * 1.08,
            f"{r['spr_energy_mj']:.2f}", ha="center", fontsize=7)
    ax.text(i,     r["h200_old_energy_mj"] * 1.08,
            f"{r['h200_old_energy_mj']:.2f}\n{r['energy_pct_old']:.0f}%", ha="center", fontsize=7)
    ax.text(i + w, r["h200_fix_energy_mj"] * 1.08,
            f"{r['h200_fix_energy_mj']:.2f}\n{r['energy_pct_fix']:.0f}%", ha="center", fontsize=7)

fig.suptitle(
    "MTEST · 100 taxa · CPU Intel SPR (OMP104) vs OpenACC H200 (old) vs OpenACC H200 (gpumem_fix latest)\n"
    "Numbers under H200 bars: 'Nx' = speedup vs SPR · 'M%' = energy as fraction of SPR · "
    "memory bracket = change vs old GPU build", fontsize=11)
fig.tight_layout()
out = OUT_DIR / "fig_spr_vs_gpu.png"
fig.savefig(out, dpi=140); plt.close(fig)
print(f"\nWrote {out}, spr_vs_gpu_table.csv")
