"""Parse the 2026_06_18_gpumem_fix IQ-TREE logs and compare GPU/CPU memory and
energy against the 2026_06_13_morebenchmarks baseline for the same MTEST cells
(100 taxa, AA LG+I+G4 and DNA GTR+I+G4, on H200 and V100 OPENACC), at site
counts 100k and 1M.

Repetitions of the gpumem_fix build:
  - 100k: test_1 (all four cells) + test_2 (3 cells; no AA H200)
  - 1M:   test_2 (all four cells)

Produces (per site count):
  - runs.csv             : parsed metrics from gpumem_fix logs (both reps)
  - comparison.csv       : merged baseline + gpumem_fix rows
  - fig_memory_compare_{sites}.png : GPU peak + delta bars (baseline / test_1 / test_2)
  - fig_energy_compare_{sites}.png : CPU / GPU / total energy bars
  - fig_wall_compare_{sites}.png   : wall time bars
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

NEW_DIR = Path("/Users/u7826985/Projects/Nvidia/results/2026_06_18_gpumem_fix")
BASE_CSV = Path("/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_13_morebenchmarks/runs.csv")
OUT_DIR = Path(__file__).resolve().parent

RE_TOTAL_WALL = re.compile(r"Total wall-clock time used:\s+([\d.]+)\s+sec")
RE_BEST_LOGL  = re.compile(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)")
RE_E_CPU      = re.compile(r"Energy:\s*\n\s*CPU:\s+([\d.]+)\s+J", re.M)
RE_E_GPU      = re.compile(r"^\s*GPU:\s+([\d.]+)\s+J", re.M)
RE_GPU_MEM    = re.compile(r"GPU mem:\s+([\d.]+)\s*/\s*([\d.]+)\s+MB.*?\+([\d.]+)\s+MB")
RE_CPU_MEM    = re.compile(r"CPU mem:\s+([\d.]+)\s*/\s*([\d.]+)\s+MB")

def _f(rx, txt, grp=1):
    m = rx.search(txt)
    return float(m.group(grp)) if m else None

RE_SITES = re.compile(r"_tree_1_(\d+)_")

def classify(name: str):
    parts = name.split("_")
    hw = "GPU_H200" if "H200" in parts else "GPU_V100" if "V100" in parts else "GPU_unknown"
    dt = "AA" if "AA" in parts else "DNA" if "DNA" in parts else "unknown"
    rep = "test_2" if re.search(r"_test_2_", name) else "test_1"
    m = RE_SITES.search(name)
    sites = int(m.group(1)) if m else None
    return hw, dt, rep, sites

rows = []
for log in sorted(NEW_DIR.glob("*.log")):
    txt = log.read_text(errors="ignore")
    hw, dt, rep, sites = classify(log.name)
    gm = RE_GPU_MEM.search(txt)
    cm = RE_CPU_MEM.search(txt)
    rows.append({
        "file": log.name,
        "rep": rep,
        "hardware": hw,
        "datatype": dt,
        "test": "MTEST",
        "sites": sites,
        "wall_s": _f(RE_TOTAL_WALL, txt),
        "best_logL": _f(RE_BEST_LOGL, txt),
        "energy_cpu_J": _f(RE_E_CPU, txt),
        "energy_gpu_J": _f(RE_E_GPU, txt),
        "gpu_mem_peak_MB":  float(gm.group(1)) if gm else None,
        "gpu_mem_cap_MB":   float(gm.group(2)) if gm else None,
        "gpu_mem_delta_MB": float(gm.group(3)) if gm else None,
        "cpu_mem_peak_MB":  float(cm.group(1)) if cm else None,
        "cpu_mem_cap_MB":   float(cm.group(2)) if cm else None,
    })

new = pd.DataFrame(rows)
new["energy_total_J"] = new["energy_cpu_J"].fillna(0) + new["energy_gpu_J"].fillna(0)
new.to_csv(OUT_DIR / "runs.csv", index=False)

# --- baseline: 2026_06_13 MTEST for the same hw x datatype, both site counts --
SITES = sorted(s for s in new["sites"].dropna().unique())
base = pd.read_csv(BASE_CSV)
base = base[(base["test"] == "MTEST") & base["sites"].isin(SITES)
            & base["hardware"].isin(["GPU_H200", "GPU_V100"])].copy()
base = base[["hardware", "datatype", "sites", "wall_total_s", "energy_cpu_J",
             "energy_gpu_J", "energy_total_J", "gpu_mem_peak_MB",
             "gpu_mem_delta_MB", "cpu_mem_peak_MB"]].rename(
                 columns={"wall_total_s": "wall_s"})
base["rep"] = "baseline_06_13"
base["file"] = "(from 2026_06_13 runs.csv)"
base["test"] = "MTEST"

all_runs = pd.concat([base, new], ignore_index=True, sort=False)
all_runs["label"] = all_runs["datatype"] + "/" + all_runs["hardware"].str.replace("GPU_", "")
all_runs = all_runs.sort_values(["sites", "datatype", "hardware", "rep"]).reset_index(drop=True)
all_runs.to_csv(OUT_DIR / "comparison.csv", index=False)

key_cols = ["sites", "label", "rep", "gpu_mem_peak_MB", "gpu_mem_delta_MB",
            "wall_s", "energy_cpu_J", "energy_gpu_J", "energy_total_J"]
for s in SITES:
    print(f"\n=== MTEST 100taxa / {s} sites — across reps ===")
    print(all_runs[all_runs["sites"] == s][key_cols].to_string(index=False))

# ---------------------- plotting --------------------------------------------
REP_ORDER  = ["baseline_06_13", "test_1", "test_2"]
REP_COLORS = {"baseline_06_13": "#888", "test_1": "#1f77b4", "test_2": "#ff7f0e"}
REP_LABEL  = {"baseline_06_13": "2026_06_13 baseline",
              "test_1": "2026_06_18 gpumem_fix (rep 1)",
              "test_2": "2026_06_18 gpumem_fix (rep 2)"}

def grouped_bar3(ax, df, metric, ylabel, title):
    cells = sorted(df["label"].unique())
    x = np.arange(len(cells)); w = 0.27
    offsets = {"baseline_06_13": -w, "test_1": 0.0, "test_2": w}
    for rep in REP_ORDER:
        vals = [df[(df["label"] == c) & (df["rep"] == rep)][metric].mean()
                for c in cells]
        ax.bar(x + offsets[rep], vals, w,
               label=REP_LABEL[rep], color=REP_COLORS[rep])
    ax.set_xticks(x); ax.set_xticklabels(cells, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for i, c in enumerate(cells):
        b = df[(df["label"] == c) & (df["rep"] == "baseline_06_13")][metric].mean()
        if not np.isfinite(b) or b == 0:
            continue
        for rep, off in [("test_1", 0.0), ("test_2", w)]:
            v = df[(df["label"] == c) & (df["rep"] == rep)][metric].mean()
            if np.isfinite(v):
                ax.text(i + off, v * 1.01, f"{100*(v-b)/b:+.1f}%",
                        ha="center", fontsize=7)

def sites_tag(s):
    return f"{int(s/1_000_000)}M" if s >= 1_000_000 else f"{int(s/1000)}k"

for s in SITES:
    sub = all_runs[all_runs["sites"] == s]
    tag = sites_tag(s)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    grouped_bar3(axes[0], sub, "gpu_mem_peak_MB",
                 "GPU peak (MB)", "GPU peak memory")
    grouped_bar3(axes[1], sub, "gpu_mem_delta_MB",
                 "GPU Δ (MB)", "GPU memory delta (run-induced)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"MTEST 100taxa / {tag} sites — memory across reps")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig_memory_compare_{tag}.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    grouped_bar3(axes[0], sub, "energy_cpu_J",   "CPU energy (J)", "CPU energy")
    grouped_bar3(axes[1], sub, "energy_gpu_J",   "GPU energy (J)", "GPU energy")
    grouped_bar3(axes[2], sub, "energy_total_J", "Total energy (J)", "Total energy")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"MTEST 100taxa / {tag} sites — energy across reps")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig_energy_compare_{tag}.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    grouped_bar3(ax, sub, "wall_s", "Wall (s)", f"Total wall-clock ({tag} sites)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig_wall_compare_{tag}.png", dpi=140)
    plt.close(fig)

print(f"\nWrote runs.csv, comparison.csv, and figs for sites: "
      f"{[sites_tag(s) for s in SITES]}")
