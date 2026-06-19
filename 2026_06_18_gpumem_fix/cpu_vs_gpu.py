"""MTEST cross-platform comparison at one site count: CPU baselines
+ OpenACC H200/V100 (baseline 2026_06_13) vs OpenACC H200/V100 (gpumem_fix
2026_06_18). Run for AA and DNA, 1M and 10M sites.

Produces fig_{aa,dna}{tag}_cpu_vs_gpu.png with three panels: wall time,
peak memory (CPU + GPU), and energy (CPU + GPU stacked).
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_CSV = Path("/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_13_morebenchmarks/runs.csv")
NEW_DIR  = Path("/Users/u7826985/Projects/Nvidia/results/2026_06_18_gpumem_fix")
OUT_DIR  = Path(__file__).resolve().parent

LABEL = {
    "CPU_CLX_OMP48":   "CPU Intel CLX (OMP48)",
    "CPU_SPR_OMP104":  "CPU Intel SPR (OMP104)",
    "GPU_H200":        "OpenACC H200 (old)",
    "GPU_V100":        "OpenACC V100 (old)",
    "GPU_H200_fix":    "OpenACC H200 (gpumem_fix)",
    "GPU_V100_fix":    "OpenACC V100 (gpumem_fix)",
}
COLOR = {
    "CPU_CLX_OMP48":   "#4c72b0",
    "CPU_SPR_OMP104":  "#55a868",
    "GPU_H200":        "#c44e52",
    "GPU_V100":        "#8172b2",
    "GPU_H200_fix":    "#dd8452",
    "GPU_V100_fix":    "#937860",
}

def parse_new_log(p: Path) -> dict:
    txt = p.read_text(errors="ignore")
    def _f(pat, t=txt, g=1, flags=re.M):
        m = re.search(pat, t, flags)
        return float(m.group(g)) if m else np.nan
    hw = ("GPU_H200_fix" if "H200" in p.name.split("_")
          else "GPU_V100_fix" if "V100" in p.name.split("_") else "GPU_unknown_fix")
    return {
        "hardware": hw,
        "wall_s":          _f(r"Total wall-clock time used:\s+([\d.]+)\s+sec"),
        "energy_cpu_J":    _f(r"Energy:\s*\n\s*CPU:\s+([\d.]+)\s+J"),
        "energy_gpu_J":    _f(r"^\s*GPU:\s+([\d.]+)\s+J"),
        "gpu_mem_peak_MB": _f(r"GPU mem:\s+([\d.]+)\s*/"),
        "cpu_mem_peak_MB": _f(r"CPU mem:\s+([\d.]+)\s*/"),
    }

def build_table(sites: int, datatype: str) -> pd.DataFrame:
    base = pd.read_csv(BASE_CSV)
    base = base[(base["test"] == "MTEST") & (base["datatype"] == datatype)
                & (base["sites"] == sites)].copy()
    base = base[["hardware", "wall_total_s", "energy_cpu_J", "energy_gpu_J",
                 "energy_total_J", "gpu_mem_peak_MB", "cpu_mem_peak_MB"]
               ].rename(columns={"wall_total_s": "wall_s"})

    pattern = f"_tree_1_{sites}_"
    new_rows = [parse_new_log(p) for p in sorted(NEW_DIR.glob("*.log"))
                if datatype in p.name.split("_") and pattern in p.name]
    for r in new_rows:
        r["energy_total_J"] = (r.get("energy_cpu_J") or 0) + (r.get("energy_gpu_J") or 0)
    new = pd.DataFrame(new_rows)

    df = pd.concat([base, new], ignore_index=True, sort=False)
    order = [h for h in ["CPU_CLX_OMP48", "CPU_SPR_OMP104",
                         "GPU_H200", "GPU_H200_fix",
                         "GPU_V100", "GPU_V100_fix"]
             if h in df["hardware"].values]
    df = df.set_index("hardware").loc[order].reset_index()
    df["label"] = df["hardware"].map(LABEL)
    return df

def plot(df: pd.DataFrame, tag: str, datatype: str):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(df))
    colors = [COLOR[h] for h in df["hardware"]]

    # 1) wall (hours)
    ax = axes[0]
    wall_h = df["wall_s"] / 3600
    b = ax.bar(x, wall_h, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(df["label"], rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("Wall time (hours)")
    ax.set_title("Wall-clock time")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(b, wall_h):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width()/2, v * 1.01,
                    f"{v:.2f} h", ha="center", fontsize=8)

    # 2) memory: CPU vs GPU per cell (GB)
    ax = axes[1]
    w = 0.38
    cpu_gb = df["cpu_mem_peak_MB"] / 1024
    gpu_gb = df["gpu_mem_peak_MB"] / 1024
    b1 = ax.bar(x - w/2, cpu_gb, w, label="CPU peak (host RAM)", color="#999")
    b2 = ax.bar(x + w/2, gpu_gb, w, label="GPU peak (VRAM)",     color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(df["label"], rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("Peak memory (GB)")
    ax.set_title("Peak memory: host vs GPU")
    ax.grid(axis="y", alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    ymax = float(np.nanmax(pd.concat([cpu_gb, gpu_gb]).fillna(0))) or 1.0
    pad = ymax * 0.015
    for bar, v in zip(b1, cpu_gb):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + pad,
                    f"{v:.1f}", ha="center", fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, pad, "n/a",
                    ha="center", fontsize=8, color="#888")
    for bar, v in zip(b2, gpu_gb):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + pad,
                    f"{v:.1f}", ha="center", fontsize=8)

    # 3) energy stacked (MJ)
    ax = axes[2]
    cpu_mj = df["energy_cpu_J"] / 1e6
    gpu_mj = df["energy_gpu_J"].fillna(0) / 1e6
    ax.bar(x, cpu_mj, color="#7f7f7f", label="CPU energy")
    ax.bar(x, gpu_mj, bottom=cpu_mj,  color="#2ca02c", label="GPU energy")
    totals = cpu_mj + gpu_mj
    ax.set_xticks(x); ax.set_xticklabels(df["label"], rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("Energy (MJ)")
    ax.set_title("Energy (stacked: CPU + GPU)")
    ax.grid(axis="y", alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    for i, t in enumerate(totals):
        if np.isfinite(t):
            ax.text(i, t * 1.01, f"{t:.2f} MJ", ha="center", fontsize=8)

    model = "LG+I+G4" if datatype == "AA" else "GTR+I+G4"
    fig.suptitle(f"{datatype} / {model} / 100 taxa / {tag} sites / MTEST — CPU vs OpenACC, old vs new")
    fig.tight_layout()
    out = OUT_DIR / f"fig_{datatype.lower()}{tag}_cpu_vs_gpu.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"Wrote {out}")

for datatype in ("DNA", "AA"):
    for sites, tag in [(1_000_000, "1M"), (10_000_000, "10M")]:
        df = build_table(sites, datatype)
        if df.empty:
            continue
        print(f"\n=== {datatype} / {tag} MTEST cross-platform ===")
        print(df[["label", "wall_s", "cpu_mem_peak_MB", "gpu_mem_peak_MB",
                  "energy_cpu_J", "energy_gpu_J", "energy_total_J"]].to_string(index=False))
        plot(df, tag, datatype)
