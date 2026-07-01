"""Reference-style plots, modelled on
/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_13_morebenchmarks/analysis.py

Reads runs.csv produced by analysis.py and emits per-cell linear panels (wall
time in minutes), stacked CPU+GPU energy bars (Wh), stage-stacked energy
(ModelFinder + tree-search + finalization), a memory-headroom grid, scaling
curves vs alignment length, and a side-by-side speedup / energy-ratio heatmap
keyed to cpu_OMP_48 as in the reference.

Differences from the reference:
  - No `test` axis (the simulated runs are all the same workload).
  - 10 hardware variants instead of 4; bars are narrower and labels rotated.
  - Baseline = cpu_OMP_48 (CLX-equivalent).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
df = pd.read_csv(OUT / "runs.csv")

HW_ORDER = ["cpu_OMP_48", "cpu_OMP_104",
            "cudajolt_V100", "cudajolt_H200",
            "openACC_stable_V100",
            "openACC_stable_H200"]
HW_LABEL = {
    "cpu_OMP_48":                       "CPU 2nd Gen Xeon",
    "cpu_OMP_104":                      "CPU 4th Gen Xeon",
    "cudajolt_V100":                    "cudajolt V100",
    "cudajolt_H200":                    "cudajolt H200",
    "openACC_stable_V100":              "ACC V100",
    "openACC_stable_V100_nt12":         "ACC V100 nt12",
    "openACC_stable_H200":              "ACC H200",
    "openACC_stable_H200_nt12":         "ACC H200 nt12",
    "openACC_JOLT_h2tiling_H200":       "ACC JOLT H200",
    "openACC_JOLT_h2tiling_H200_nt12":  "ACC JOLT H200 nt12",
}
HW_COLOR = {
    "cpu_OMP_48":                       "#1f77b4",
    "cpu_OMP_104":                      "#e6611c",
    "cudajolt_V100":                    "#c7a3d9",   # light purple — distinct from CPU-blue
    "cudajolt_H200":                    "#54278f",   # deep purple
    "openACC_stable_V100":              "#a8d8a0",
    "openACC_stable_V100_nt12":         "#7fc97f",
    "openACC_stable_H200":              "#74c476",
    "openACC_stable_H200_nt12":         "#1f7a3a",
    "openACC_JOLT_h2tiling_H200":       "#bcbddc",
    "openACC_JOLT_h2tiling_H200_nt12":  "#54278f",
}

ok = df[df.status == "COMPLETE"].copy()
# derive avg_power and cpu_share (energy fields may be NaN for cudajolt)
ok["energy_total_J"] = ok["energy_cpu_total_J"].fillna(0) + ok["energy_gpu_total_J"].fillna(0)
ok.loc[ok["energy_total_J"] == 0, "energy_total_J"] = np.nan
ok["avg_power_W"]    = ok["energy_total_J"] / ok["wall_total_s"]
ok["cpu_share"]      = ok["energy_cpu_total_J"] / ok["energy_total_J"]

# ---- helpers ---------------------------------------------------------------
def _fmt_min(seconds):
    if pd.isna(seconds): return ""
    m = seconds / 60.0
    if m < 60: return f"{m:.0f}m" if m >= 1 else f"{m:.1f}m"
    h = int(m // 60); return f"{h}h{int(round(m - 60*h))}m"

def _fmt_wh(j):
    if pd.isna(j) or j is None: return ""
    wh = j / 3600.0
    if wh >= 1000: return f"{wh/1000:.1f}kWh"
    if wh >= 10:   return f"{wh:.0f}Wh"
    return f"{wh:.1f}Wh"

MIN_SITES = 100_000  # skip 1K / 10K panels — these were empty for ModelFinder etc.

def _cells():
    cells = []
    for dt in ("AA", "DNA"):
        for s in sorted(ok[ok.datatype == dt]["sites"].dropna().unique()):
            if s < MIN_SITES:
                continue
            if (ok[(ok.datatype == dt) & (ok.sites == s)]).empty:
                continue
            cells.append((dt, int(s)))
    return cells

def _sites_tag(s):
    if s >= 1_000_000: return f"{s//1_000_000}M"
    return f"{s//1000}K"

# ---------------------------------------------------------------------------
# 1. per-cell linear wall-time panels  (Total / Tree-search / ModelFinder)
# ---------------------------------------------------------------------------
def linear_panels(value_col: str, stage_label: str, fname: str):
    cells = [(dt, s) for (dt, s) in _cells()
             if ok[(ok.datatype == dt) & (ok.sites == s)][value_col].notna().any()]
    if not cells:
        print(f"[linear_panels] no data for {value_col}, skip"); return
    ncols = len(cells)
    fig, axes = plt.subplots(1, ncols, figsize=(3.6 * ncols, 5.6), squeeze=False)
    axes = axes[0]
    x = np.arange(len(HW_ORDER))
    for ax, (dt, s) in zip(axes, cells):
        slice_ = ok[(ok.datatype == dt) & (ok.sites == s)]
        vals = []
        for hw in HW_ORDER:
            row = slice_[slice_.hardware == hw]
            vals.append(row[value_col].iloc[0] if len(row) and pd.notna(row[value_col].iloc[0])
                        else np.nan)
        mins = [v / 60 if pd.notna(v) else np.nan for v in vals]
        colors = [HW_COLOR[h] for h in HW_ORDER]
        bars = ax.bar(x, mins, color=colors, edgecolor="black", linewidth=0.4, width=0.78)
        for b, raw in zip(bars, vals):
            if pd.notna(raw):
                ax.text(b.get_x() + b.get_width()/2, b.get_height(), _fmt_min(raw),
                        ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_title(f"{dt} — {_sites_tag(s)}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([HW_LABEL[h] for h in HW_ORDER],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("wall time (min)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        finite = [m for m in mins if pd.notna(m)]
        if finite: ax.set_ylim(0, max(finite) * 1.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=HW_COLOR[h]) for h in HW_ORDER]
    fig.legend(handles, [HW_LABEL[h] for h in HW_ORDER],
               loc="lower center", ncol=5, frameon=True, fontsize=8,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"{stage_label} (min) — per-cell linear scales",
                 fontsize=12, fontweight="bold")
    footnote = ("Each panel has its own linear y-axis scale.  "
                "CPU 2nd Gen Xeon = Intel Cascade Lake Xeon 8274 (48 threads).  "
                "CPU 4th Gen Xeon = Intel Sapphire Rapids Xeon 8480+ (104 threads).  "
                "cudajolt = IQTREE_GPU_SHARED build (no per-stage energy reporting).  "
                "ACC = openACC_stable build; ACC JOLT = openACC_bfgs_JOLT_h2tiling build.  "
                "'nt12' = host `-nt 12` thread pin.  Missing bar = run not COMPLETE.")
    fig.text(0.5, -0.07, footnote, ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUT / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[linear_panels] wrote {fname}")

linear_panels("wall_total_s", "Total wall-clock time",    "fig_wall_total_linear.png")
linear_panels("wall_tree_s",  "Tree-search wall-clock",   "fig_wall_treesearch_linear.png")
linear_panels("wall_mf_s",    "ModelFinder wall-clock",   "fig_wall_modelfinder_linear.png")

# ---------------------------------------------------------------------------
# 2. stacked CPU + GPU energy panels (per stage)
# ---------------------------------------------------------------------------
def stacked_energy_panels(cpu_col: str, gpu_col: str, stage_label: str, fname: str):
    cells = [(dt, s) for (dt, s) in _cells()
             if ok[(ok.datatype == dt) & (ok.sites == s)][cpu_col].notna().any()]
    if not cells:
        print(f"[stacked_energy] no data for {cpu_col}, skip"); return
    ncols = len(cells)
    fig, axes = plt.subplots(1, ncols, figsize=(3.6 * ncols, 5.6), squeeze=False)
    axes = axes[0]
    x = np.arange(len(HW_ORDER))
    CPU_BLUE  = "#1f77b4"
    GPU_GREEN = "#7fc97f"
    for ax, (dt, s) in zip(axes, cells):
        slice_ = ok[(ok.datatype == dt) & (ok.sites == s)]
        cpu_J, gpu_J = [], []
        for hw in HW_ORDER:
            row = slice_[slice_.hardware == hw]
            if len(row):
                cpu_J.append(row[cpu_col].iloc[0] if pd.notna(row[cpu_col].iloc[0]) else np.nan)
                gpu_J.append(row[gpu_col].iloc[0] if pd.notna(row[gpu_col].iloc[0]) else 0.0)
            else:
                cpu_J.append(np.nan); gpu_J.append(np.nan)
        cpu_wh = [c/3600 if pd.notna(c) else np.nan for c in cpu_J]
        gpu_wh = [g/3600 if pd.notna(g) else np.nan for g in gpu_J]
        ax.bar(x, cpu_wh, color=CPU_BLUE, edgecolor="black", linewidth=0.4,
               width=0.78, label="CPU (host)")
        ax.bar(x, gpu_wh, bottom=cpu_wh, color=GPU_GREEN, edgecolor="black",
               linewidth=0.4, width=0.78, label="GPU (accelerator)")
        for i, (c, g) in enumerate(zip(cpu_J, gpu_J)):
            if pd.notna(c):
                tot = (c if pd.notna(c) else 0) + (g if pd.notna(g) else 0)
                ax.text(x[i], tot/3600, _fmt_wh(tot),
                        ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_title(f"{dt} — {_sites_tag(s)}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([HW_LABEL[h] for h in HW_ORDER],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("energy (Wh)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        tops = [(c if pd.notna(c) else 0) + (g if pd.notna(g) else 0)
                for c, g in zip(cpu_J, gpu_J) if pd.notna(c)]
        if tops: ax.set_ylim(0, max(tops)/3600 * 1.30)
    handles = [plt.Rectangle((0,0),1,1, color=CPU_BLUE),
               plt.Rectangle((0,0),1,1, color=GPU_GREEN)]
    fig.legend(handles, ["CPU (host)", "GPU (accelerator)"],
               loc="lower center", ncol=2, frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Energy breakdown (Wh) — CPU host vs GPU stack — {stage_label} stage",
                 fontsize=12, fontweight="bold")
    footnote = ("CPU-only runs report `Energy: CPU X J` (no GPU); cudajolt build emits "
                "no `Energy:` block at all → bars absent.  openACC reports both.  "
                "Each panel has its own y-axis.")
    fig.text(0.5, -0.07, footnote, ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUT / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[stacked_energy] wrote {fname}")

stacked_energy_panels("energy_cpu_total_J", "energy_gpu_total_J",
                       "Total",        "fig_energy_stacked_total.png")
stacked_energy_panels("energy_cpu_ts_J",    "energy_gpu_ts_J",
                       "Tree-search",  "fig_energy_stacked_treesearch.png")
stacked_energy_panels("energy_cpu_mf_J",    "energy_gpu_mf_J",
                       "ModelFinder",  "fig_energy_stacked_modelfinder.png")

# ---------------------------------------------------------------------------
# 3. stage-stack: MF + tree-search + residual (Wh)
# ---------------------------------------------------------------------------
def energy_byStage_panels(fname: str):
    work = ok.copy()
    work["e_mf_total_J"] = work["energy_cpu_mf_J"].fillna(0) + work["energy_gpu_mf_J"].fillna(0)
    work["e_ts_total_J"] = work["energy_cpu_ts_J"].fillna(0) + work["energy_gpu_ts_J"].fillna(0)
    work["e_residual_J"] = (work["energy_total_J"].fillna(0)
                            - work["e_mf_total_J"] - work["e_ts_total_J"]).clip(lower=0)
    cells = [(dt, s) for (dt, s) in _cells()
             if ((work[(work.datatype == dt) & (work.sites == s)]["e_mf_total_J"] > 0).any()
                 or (work[(work.datatype == dt) & (work.sites == s)]["energy_total_J"] > 0).any())]
    if not cells:
        print(f"[byStage] no data, skip"); return
    ncols = len(cells)
    MF_COLOR, TS_COLOR, RS_COLOR = "#f4a261", "#1d8a3a", "#bdbdbd"
    fig, axes = plt.subplots(1, ncols, figsize=(3.6 * ncols, 5.6), squeeze=False)
    axes = axes[0]
    x = np.arange(len(HW_ORDER))
    for ax, (dt, s) in zip(axes, cells):
        slice_ = work[(work.datatype == dt) & (work.sites == s)]
        mf, ts, rs, tot = [], [], [], []
        for hw in HW_ORDER:
            row = slice_[slice_.hardware == hw]
            if len(row):
                rr = row.iloc[0]
                mf.append(rr["e_mf_total_J"] or 0)
                ts.append(rr["e_ts_total_J"] or 0)
                rs.append(rr["e_residual_J"] or 0)
                tot.append(rr["energy_total_J"] if pd.notna(rr["energy_total_J"])
                           else (rr["e_mf_total_J"] or 0) + (rr["e_ts_total_J"] or 0))
            else:
                mf.append(0); ts.append(0); rs.append(0); tot.append(np.nan)
        mf_wh = [v/3600 for v in mf]
        ts_wh = [v/3600 for v in ts]
        rs_wh = [v/3600 for v in rs]
        ax.bar(x, mf_wh, color=MF_COLOR, edgecolor="black", linewidth=0.4, width=0.78,
               label="ModelFinder")
        ax.bar(x, ts_wh, bottom=mf_wh, color=TS_COLOR, edgecolor="black", linewidth=0.4,
               width=0.78, label="Tree search")
        bot2 = [m + t for m, t in zip(mf_wh, ts_wh)]
        ax.bar(x, rs_wh, bottom=bot2, color=RS_COLOR, edgecolor="black", linewidth=0.4,
               width=0.78, label="Init + finalization")
        for i, top_J in enumerate(tot):
            if pd.notna(top_J) and top_J > 0:
                ax.text(x[i], top_J / 3600, _fmt_wh(top_J),
                        ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_title(f"{dt} — {_sites_tag(s)}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([HW_LABEL[h] for h in HW_ORDER],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("energy (Wh)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        fin = [v/3600 for v in tot if pd.notna(v) and v > 0]
        if fin: ax.set_ylim(0, max(fin) * 1.30)
    handles = [plt.Rectangle((0,0),1,1, color=MF_COLOR),
               plt.Rectangle((0,0),1,1, color=TS_COLOR),
               plt.Rectangle((0,0),1,1, color=RS_COLOR)]
    fig.legend(handles, ["ModelFinder", "Tree search", "Init + finalization"],
               loc="lower center", ncol=3, frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Total energy (Wh) — stage stack (MF + tree-search + finalization)",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, -0.07, "Top label = whole-run total from the `Energy:` block "
                          "(or MF+TS sum when block missing).",
             ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUT / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[byStage] wrote {fname}")

energy_byStage_panels("fig_energy_total_byStage.png")

# ---------------------------------------------------------------------------
# 4. memory-headroom grid (2×N panel, AA / DNA × sites)
# ---------------------------------------------------------------------------
MEM_CAP_GB = {  # advertised capacities — overridden by per-log values when present
    "cpu_OMP_48":  188,
    "cpu_OMP_104": 503,
    "cudajolt_V100":                    32,
    "cudajolt_H200":                    140,
    "openACC_stable_V100":              32,
    "openACC_stable_V100_nt12":         32,
    "openACC_stable_H200":              140,
    "openACC_stable_H200_nt12":         140,
    "openACC_JOLT_h2tiling_H200":       140,
    "openACC_JOLT_h2tiling_H200_nt12":  140,
}
COLOR_CLEAN, COLOR_MOD, COLOR_TIGHT = "#2ca02c", "#f5c518", "#f37020"
COLOR_EMPTY = "#e6e6e6"
def _mem_classify(frac):
    if pd.isna(frac): return None
    if frac < 0.50: return ("clean",    COLOR_CLEAN, "✓")
    if frac < 0.90: return ("moderate", COLOR_MOD,   "✓")
    return            ("tight",    COLOR_TIGHT, "△")

def memory_grid(fname):
    sites_present = sorted(ok["sites"].dropna().unique().astype(int))
    cols = [s for s in sites_present]
    dts = [dt for dt in ("AA", "DNA") if not ok[ok.datatype == dt].empty]
    nrows, ncols = len(dts), len(cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.0 * nrows),
                             squeeze=False)
    fig.suptitle("Memory headroom — host RAM (CPU) / VRAM (GPU) per backend",
                 fontsize=11)
    bar_h = 0.65
    for r, dt in enumerate(dts):
        for c, s in enumerate(cols):
            ax = axes[r][c]
            slice_ = ok[(ok.datatype == dt) & (ok.sites == s)]
            for i, hw in enumerate(HW_ORDER):
                y = len(HW_ORDER) - 1 - i
                row = slice_[slice_.hardware == hw]
                cap_gb = MEM_CAP_GB[hw]
                if len(row):
                    rr = row.iloc[0]
                    cap_col = "cpu_mem_cap_MB" if hw.startswith("cpu") else "gpu_mem_cap_MB"
                    cap_mb = rr.get(cap_col)
                    if pd.notna(cap_mb) and cap_mb:
                        cap_gb = cap_mb / 1024.0
                ax.barh(y, cap_gb, height=bar_h, color=COLOR_EMPTY,
                        edgecolor="black", linewidth=0.5)
                if not len(row):
                    ax.text(cap_gb * 1.02, y, " not run", va="center", ha="left",
                            fontsize=7, color="#888", style="italic"); continue
                used_mb = row["cpu_mem_peak_MB"].iloc[0] if hw.startswith("cpu") \
                    else row["gpu_mem_peak_MB"].iloc[0]
                if pd.isna(used_mb):
                    ax.text(cap_gb * 1.02, y, " no peak", va="center", ha="left",
                            fontsize=7, color="#888", style="italic"); continue
                used_gb = used_mb / 1024.0
                frac    = used_gb / cap_gb
                cls = _mem_classify(frac)
                ax.barh(y, min(used_gb, cap_gb), height=bar_h, color=cls[1],
                        edgecolor="black", linewidth=0.5)
                ax.text(min(used_gb, cap_gb) - cap_gb * 0.012, y, cls[2],
                        va="center", ha="right", fontsize=10, color="white",
                        fontweight="bold")
                ax.text(cap_gb * 1.02, y,
                        f" {used_gb:.0f}/{cap_gb:.0f} GB  ({frac*100:.0f}%)",
                        va="center", ha="left", fontsize=7)
            ax.set_yticks(range(len(HW_ORDER)))
            ax.set_yticklabels([HW_LABEL[h] for h in reversed(HW_ORDER)],
                               fontsize=7, fontweight="bold")
            ax.set_xlim(0, max(MEM_CAP_GB.values()) * 1.55)
            ax.set_xlabel("memory (GB)")
            ax.set_title(f"{dt} · {_sites_tag(s)} sites",
                         fontsize=10, fontweight="bold")
            ax.grid(axis="x", alpha=0.25, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    legend_items = [
        (COLOR_CLEAN, "✓ clean (<50%)"),
        (COLOR_MOD,   "moderate (50–90%)"),
        (COLOR_TIGHT, "△ tight (≥90%)"),
        (COLOR_EMPTY, "not run / no peak recorded"),
    ]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.4)
               for c, _ in legend_items]
    fig.legend(handles, [l for _, l in legend_items],
               loc="lower center", ncol=len(legend_items), frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUT / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[memory_grid] wrote {fname}")

memory_grid("fig_memory_grid.png")

# ---------------------------------------------------------------------------
# 5. scaling plots — y vs alignment length (log x, log y)
# ---------------------------------------------------------------------------
def scaling_plot(value, ylabel, title, fname, log_y=True):
    dts = ("AA", "DNA")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
    for ax, dt in zip(axes, dts):
        sub = ok[ok.datatype == dt]
        for hw in HW_ORDER:
            s = sub[sub.hardware == hw].dropna(subset=["sites", value]).sort_values("sites")
            if not len(s): continue
            ax.plot(s["sites"], s[value], "-o", label=HW_LABEL[hw],
                    color=HW_COLOR[hw], linewidth=1.6, markersize=5)
        ax.set_xscale("log")
        if log_y: ax.set_yscale("log")
        ax.set_title(f"{dt}  ({'LG+I+G4' if dt == 'AA' else 'GTR+I+G4'})")
        ax.set_xlabel("alignment length (sites)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].legend(fontsize=6, loc="best", ncol=2)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160)
    plt.close(fig)
    print(f"[scaling] wrote {fname}")

scaling_plot("wall_total_s",      "wall total (s, log)",
             "Wall-clock scaling vs alignment length", "fig_scaling_wall_total.png", True)
scaling_plot("wall_tree_s",       "tree-search wall (s, log)",
             "Tree-search wall scaling vs length",     "fig_scaling_wall_treesearch.png", True)
scaling_plot("energy_total_J",    "total energy (J, log)",
             "Energy scaling vs alignment length",     "fig_scaling_energy_total.png", True)
scaling_plot("avg_power_W",       "avg power (W)",
             "Average power vs alignment length",      "fig_scaling_avg_power.png", False)
scaling_plot("cpu_share",         "CPU energy share",
             "CPU share of total energy",              "fig_scaling_cpu_share.png", False)

# ---------------------------------------------------------------------------
# 6. side-by-side speedup + energy-ratio heatmap (baseline = cpu_OMP_48)
# ---------------------------------------------------------------------------
base = ok[ok.hardware == "cpu_OMP_48"][[
    "datatype", "sites", "wall_total_s", "energy_total_J"]].rename(
        columns={"wall_total_s": "base_wall", "energy_total_J": "base_energy"})
rel = ok.merge(base, on=["datatype", "sites"], how="inner")
rel["speedup"]      = rel["base_wall"]   / rel["wall_total_s"]
rel["energy_ratio"] = rel["energy_total_J"] / rel["base_energy"]
piv_sp = rel.pivot_table(index=["datatype", "sites"], columns="hardware",
                          values="speedup", aggfunc="mean")
piv_er = rel.pivot_table(index=["datatype", "sites"], columns="hardware",
                          values="energy_ratio", aggfunc="mean")
hw_cols = [h for h in HW_ORDER if h in piv_sp.columns]
piv_sp = piv_sp.reindex(columns=hw_cols)
piv_er = piv_er.reindex(columns=hw_cols)

fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.42 * len(piv_sp))))
for ax, mat, title, cmap, fmt_ in [
    (axes[0], piv_sp, "Speedup vs cpu_OMP_48 (higher = faster)", "Greens", "{:.2f}×"),
    (axes[1], piv_er, "Energy ratio vs cpu_OMP_48 (lower = greener)", "RdYlGn_r", "{:.2f}×"),
]:
    arr = mat.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([HW_LABEL[c] for c in mat.columns], rotation=35, ha="right",
                       fontsize=8)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([f"{a}/{_sites_tag(int(s))}" for a, s in mat.index], fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, fmt_.format(arr[i, j]), ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7)
fig.suptitle("Speedup and energy ratio vs cpu_OMP_48 baseline", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig_speedup_energy_heatmap.png", dpi=160)
plt.close(fig)
print("[heatmap] wrote fig_speedup_energy_heatmap.png")

# ---------------------------------------------------------------------------
# 7. logL agreement plot — Δ from best within (datatype, sites)
# ---------------------------------------------------------------------------
rows = []
for (dt, s), g in ok.groupby(["datatype", "sites"]):
    vals = g.dropna(subset=["best_logL"])
    if len(vals) < 2: continue
    best = vals["best_logL"].max()
    for _, r in vals.iterrows():
        rows.append({"datatype": dt, "sites": int(s), "hardware": r.hardware,
                     "logL": r.best_logL, "delta": r.best_logL - best})
agr = pd.DataFrame(rows)
if not agr.empty:
    agr.to_csv(OUT / "logL_agreement_v2.csv", index=False)
    labels = sorted(agr[["datatype", "sites"]].drop_duplicates().itertuples(index=False))
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(labels)), 5))
    label_idx = {lab: i for i, lab in enumerate(labels)}
    for _, r in agr.iterrows():
        i = label_idx[(r.datatype, r.sites)]
        ax.scatter(i, r.delta, color=HW_COLOR.get(r.hardware, "black"),
                   s=70, edgecolor="black", linewidth=0.4, label=HW_LABEL.get(r.hardware))
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{dt}/{_sites_tag(int(s))}" for dt, s in labels],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Δ log-likelihood from best (within cell)")
    ax.set_title("Log-likelihood agreement — Δ from best per (datatype, sites)")
    ax.grid(axis="y", alpha=0.3)
    handles, lbls = ax.get_legend_handles_labels()
    seen = {l: h for h, l in zip(handles, lbls)}
    ax.legend(seen.values(), seen.keys(), fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_logL_agreement_v2.png", dpi=160)
    plt.close(fig)
    print("[logL] wrote fig_logL_agreement_v2.png")
