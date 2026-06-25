"""Reference-style plots for the 2026_06_25 real-data run set, modelled on
2026_06_24_simulated_results/more_plots.py.

Reads runs.csv produced by analysis.py and emits:
  fig_wall_total_linear.png       per-cell linear wall panels (TOTAL)
  fig_wall_treesearch_linear.png  per-cell linear wall panels (TREE)
  fig_wall_modelfinder_linear.png per-cell linear wall panels (MODELFINDER)
  fig_energy_stacked_total.png    CPU+GPU energy stacks (Wh) — total
  fig_energy_stacked_treesearch.png   same — tree-search stage
  fig_energy_stacked_modelfinder.png  same — ModelFinder stage
  fig_energy_total_byStage.png    stage-stack MF + TS + finalization (Wh)
  fig_memory_grid.png             memory headroom grid (host RAM / GPU VRAM)

Differences from the simulated set:
  - Cells vary by (datatype, dataset, taxa, sites) — 11 cells, ≤9 with COMPLETE
    runs, so panels wrap into 2 rows.
  - No cudajolt or JOLT_h2tiling builds in this set; just CPU + openACC ×
    {V100, A100, H200} × {default, nt12}.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
df = pd.read_csv(OUT / "runs.csv")

HW_ORDER = [
    "cpu_OMP_48", "cpu_OMP_104",
    "openACC_V100",
    "openACC_A100",
    "openACC_H200",
]
HW_LABEL = {
    "cpu_OMP_48":        "CLX 48T",
    "cpu_OMP_104":       "SPR 104T",
    "openACC_V100":      "ACC V100",
    "openACC_V100_nt12": "ACC V100 nt12",
    "openACC_A100":      "ACC A100",
    "openACC_A100_nt12": "ACC A100 nt12",
    "openACC_H200":      "ACC H200",
    "openACC_H200_nt12": "ACC H200 nt12",
}
HW_COLOR = {
    "cpu_OMP_48":        "#1f77b4",  # CPU blue
    "cpu_OMP_104":       "#e6611c",  # CPU orange
    "openACC_V100":      "#b3d4e6",  # sky-light
    "openACC_V100_nt12": "#4292c6",  # sky-dark
    "openACC_A100":      "#bcbddc",  # purple-light
    "openACC_A100_nt12": "#54278f",  # purple-dark
    "openACC_H200":      "#a1d99b",  # green-light
    "openACC_H200_nt12": "#238b45",  # green-dark
}
NCOLS_PER_ROW = 5     # 9 cells → 2 rows × 5 (one empty)

# Only plot cells/runs from HW_ORDER (drop _nt12 variants entirely).
df = df[df["hardware"].isin(HW_ORDER)].copy()

ok = df[df.status == "COMPLETE"].copy()
ok["energy_total_J"] = ok["energy_cpu_total_J"].fillna(0) + ok["energy_gpu_total_J"].fillna(0)
ok.loc[ok["energy_total_J"] == 0, "energy_total_J"] = np.nan

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

def _sites_tag(s):
    s = int(s)
    if s >= 1_000_000:
        v = s / 1_000_000
        return f"{v:.1f}M" if v < 10 else f"{v:.0f}M"
    if s >= 10_000: return f"{s//1000}K"
    if s >= 1000:   return f"{s/1000:.1f}K"
    return str(s)

def _cell_title(dt, dataset, taxa, sites):
    return f"{dataset}\n{dt} · t={taxa} · L={_sites_tag(sites)}"

def _cells_for(col_or_cols):
    """List of (dt, dataset, taxa, sites) cells with at least one finite value
    in the requested column(s). Sorted DNA-then-AA, by descending sites."""
    cols = [col_or_cols] if isinstance(col_or_cols, str) else list(col_or_cols)
    keys = ok[["datatype", "dataset", "taxa", "sites"]].drop_duplicates()
    out = []
    for _, r in keys.iterrows():
        sub = ok[(ok.datatype == r.datatype) & (ok.dataset == r.dataset)
                 & (ok.taxa == r.taxa) & (ok.sites == r.sites)]
        if any(sub[c].notna().any() for c in cols):
            out.append((r.datatype, r.dataset, int(r.taxa), int(r.sites)))
    # group: DNA first then AA, biggest sites first
    out.sort(key=lambda t: (0 if t[0] == "DNA" else 1, -t[3], t[1]))
    return out

def _grid(n):
    nrows = (n + NCOLS_PER_ROW - 1) // NCOLS_PER_ROW
    return nrows, NCOLS_PER_ROW

# ---------------------------------------------------------------------------
# 1. per-cell linear wall-time panels
# ---------------------------------------------------------------------------
def linear_panels(value_col: str, stage_label: str, fname: str):
    cells = _cells_for(value_col)
    if not cells:
        print(f"[linear_panels] no data for {value_col}, skip"); return
    nrows, ncols = _grid(len(cells))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.6 * ncols, 5.4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    x = np.arange(len(HW_ORDER))
    for ax, (dt, dataset, taxa, sites) in zip(axes_flat, cells):
        slice_ = ok[(ok.datatype == dt) & (ok.dataset == dataset)
                    & (ok.taxa == taxa) & (ok.sites == sites)]
        vals = []
        for hw in HW_ORDER:
            row = slice_[slice_.hardware == hw]
            vals.append(row[value_col].iloc[0]
                        if len(row) and pd.notna(row[value_col].iloc[0])
                        else np.nan)
        mins = [v/60 if pd.notna(v) else np.nan for v in vals]
        colors = [HW_COLOR[h] for h in HW_ORDER]
        bars = ax.bar(x, mins, color=colors, edgecolor="black",
                      linewidth=0.4, width=0.78)
        for b, raw in zip(bars, vals):
            if pd.notna(raw):
                ax.text(b.get_x() + b.get_width()/2, b.get_height(), _fmt_min(raw),
                        ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_title(_cell_title(dt, dataset, taxa, sites), fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([HW_LABEL[h] for h in HW_ORDER],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("wall time (min)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        finite = [m for m in mins if pd.notna(m)]
        if finite:
            ax.set_ylim(0, max(finite) * 1.25)
    for ax in axes_flat[len(cells):]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=HW_COLOR[h]) for h in HW_ORDER]
    fig.legend(handles, [HW_LABEL[h] for h in HW_ORDER],
               loc="lower center", ncol=4, frameon=True, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{stage_label} (min) — per-cell linear scales",
                 fontsize=12, fontweight="bold")
    footnote = ("Each panel has its own linear y-axis scale.  "
                "CLX = Intel Cascade Lake (OMP48), SPR = Sapphire Rapids (OMP104).  "
                "ACC = openACC_stable build.  'nt12' = host `-nt 12` thread pin.  "
                "Missing bar = run not COMPLETE.")
    fig.text(0.5, -0.05, footnote, ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUT / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[linear_panels] wrote {fname}")

linear_panels("wall_total_s", "Total wall-clock time",   "fig_wall_total_linear.png")
linear_panels("wall_tree_s",  "Tree-search wall-clock",  "fig_wall_treesearch_linear.png")
linear_panels("wall_mf_s",    "ModelFinder wall-clock",  "fig_wall_modelfinder_linear.png")

# ---------------------------------------------------------------------------
# 2. stacked CPU + GPU energy panels (per stage)
# ---------------------------------------------------------------------------
def stacked_energy_panels(cpu_col: str, gpu_col: str, stage_label: str, fname: str):
    cells = _cells_for(cpu_col)
    if not cells:
        print(f"[stacked_energy] no data for {cpu_col}, skip"); return
    nrows, ncols = _grid(len(cells))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.6 * ncols, 5.4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    x = np.arange(len(HW_ORDER))
    CPU_BLUE  = "#1f77b4"
    GPU_GREEN = "#7fc97f"
    for ax, (dt, dataset, taxa, sites) in zip(axes_flat, cells):
        slice_ = ok[(ok.datatype == dt) & (ok.dataset == dataset)
                    & (ok.taxa == taxa) & (ok.sites == sites)]
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
        ax.set_title(_cell_title(dt, dataset, taxa, sites), fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([HW_LABEL[h] for h in HW_ORDER],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("energy (Wh)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        tops = [(c if pd.notna(c) else 0) + (g if pd.notna(g) else 0)
                for c, g in zip(cpu_J, gpu_J) if pd.notna(c)]
        if tops:
            ax.set_ylim(0, max(tops)/3600 * 1.30)
    for ax in axes_flat[len(cells):]:
        ax.axis("off")
    handles = [plt.Rectangle((0,0),1,1, color=CPU_BLUE),
               plt.Rectangle((0,0),1,1, color=GPU_GREEN)]
    fig.legend(handles, ["CPU (host)", "GPU (accelerator)"],
               loc="lower center", ncol=2, frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Energy breakdown (Wh) — CPU host vs GPU stack — {stage_label} stage",
                 fontsize=12, fontweight="bold")
    footnote = ("CPU-only runs report `Energy: CPU X J` (no GPU).  "
                "openACC reports both.  Each panel has its own y-axis.")
    fig.text(0.5, -0.05, footnote, ha="center", fontsize=7, style="italic")
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
# 3. stage-stack: MF + TS + residual (Wh)
# ---------------------------------------------------------------------------
def energy_byStage_panels(fname: str):
    work = ok.copy()
    work["e_mf_total_J"] = work["energy_cpu_mf_J"].fillna(0) + work["energy_gpu_mf_J"].fillna(0)
    work["e_ts_total_J"] = work["energy_cpu_ts_J"].fillna(0) + work["energy_gpu_ts_J"].fillna(0)
    work["e_residual_J"] = (work["energy_total_J"].fillna(0)
                            - work["e_mf_total_J"] - work["e_ts_total_J"]).clip(lower=0)
    cells = _cells_for(["energy_total_J", "e_mf_total_J"])
    if not cells:
        print(f"[byStage] no data, skip"); return
    nrows, ncols = _grid(len(cells))
    MF_COLOR, TS_COLOR, RS_COLOR = "#f4a261", "#1d8a3a", "#bdbdbd"
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.6 * ncols, 5.4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    x = np.arange(len(HW_ORDER))
    for ax, (dt, dataset, taxa, sites) in zip(axes_flat, cells):
        slice_ = work[(work.datatype == dt) & (work.dataset == dataset)
                      & (work.taxa == taxa) & (work.sites == sites)]
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
        ax.bar(x, mf_wh, color=MF_COLOR, edgecolor="black", linewidth=0.4,
               width=0.78, label="ModelFinder")
        ax.bar(x, ts_wh, bottom=mf_wh, color=TS_COLOR, edgecolor="black",
               linewidth=0.4, width=0.78, label="Tree search")
        bot2 = [m + t for m, t in zip(mf_wh, ts_wh)]
        ax.bar(x, rs_wh, bottom=bot2, color=RS_COLOR, edgecolor="black",
               linewidth=0.4, width=0.78, label="Init + finalization")
        for i, top_J in enumerate(tot):
            if pd.notna(top_J) and top_J > 0:
                ax.text(x[i], top_J / 3600, _fmt_wh(top_J),
                        ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_title(_cell_title(dt, dataset, taxa, sites), fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([HW_LABEL[h] for h in HW_ORDER],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("energy (Wh)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        fin = [v/3600 for v in tot if pd.notna(v) and v > 0]
        if fin: ax.set_ylim(0, max(fin) * 1.30)
    for ax in axes_flat[len(cells):]:
        ax.axis("off")
    handles = [plt.Rectangle((0,0),1,1, color=MF_COLOR),
               plt.Rectangle((0,0),1,1, color=TS_COLOR),
               plt.Rectangle((0,0),1,1, color=RS_COLOR)]
    fig.legend(handles, ["ModelFinder", "Tree search", "Init + finalization"],
               loc="lower center", ncol=3, frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Total energy (Wh) — stage stack (MF + tree-search + finalization)",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, -0.05, "Top label = whole-run total from the `Energy:` block "
                          "(or MF+TS sum when block missing).",
             ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUT / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[byStage] wrote {fname}")

energy_byStage_panels("fig_energy_total_byStage.png")

# ---------------------------------------------------------------------------
# 4. memory headroom grid (host RAM / VRAM)
# ---------------------------------------------------------------------------
MEM_CAP_GB = {
    "cpu_OMP_48":        188,
    "cpu_OMP_104":       503,
    "openACC_V100":      32,
    "openACC_V100_nt12": 32,
    "openACC_A100":      80,
    "openACC_A100_nt12": 80,
    "openACC_H200":      140,
    "openACC_H200_nt12": 140,
}
COLOR_CLEAN, COLOR_MOD, COLOR_TIGHT = "#2ca02c", "#f5c518", "#f37020"
COLOR_EMPTY = "#e6e6e6"

def _mem_classify(frac):
    if pd.isna(frac): return None
    if frac < 0.50: return ("clean",    COLOR_CLEAN, "✓")
    if frac < 0.90: return ("moderate", COLOR_MOD,   "✓")
    return            ("tight",    COLOR_TIGHT, "△")

def memory_grid(fname):
    cells = _cells_for(["cpu_mem_peak_MB", "gpu_mem_peak_MB"])
    if not cells:
        print(f"[memory_grid] no data, skip"); return
    nrows, ncols = _grid(len(cells))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.8 * ncols, 4.4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    fig.suptitle("Memory headroom — host RAM (CPU) / VRAM (GPU) per backend",
                 fontsize=12, fontweight="bold")
    bar_h = 0.65
    xlim = max(MEM_CAP_GB.values()) * 1.55
    for ax, (dt, dataset, taxa, sites) in zip(axes_flat, cells):
        slice_ = ok[(ok.datatype == dt) & (ok.dataset == dataset)
                    & (ok.taxa == taxa) & (ok.sites == sites)]
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
                    f" {used_gb:.1f}/{cap_gb:.0f} GB  ({frac*100:.0f}%)",
                    va="center", ha="left", fontsize=7)
        ax.set_yticks(range(len(HW_ORDER)))
        ax.set_yticklabels([HW_LABEL[h] for h in reversed(HW_ORDER)],
                           fontsize=7, fontweight="bold")
        ax.set_xlim(0, xlim)
        ax.set_xlabel("memory (GB)")
        ax.set_title(_cell_title(dt, dataset, taxa, sites), fontsize=9)
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes_flat[len(cells):]:
        ax.axis("off")
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
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUT / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[memory_grid] wrote {fname}")

memory_grid("fig_memory_grid.png")
