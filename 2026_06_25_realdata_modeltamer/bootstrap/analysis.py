"""Parse the CPU bootstrap runs from the 2026_06_25 real-data run set.

Source logs: /Users/u7826985/Projects/Nvidia/results/2026_06_25_realdata_modeltamer/bootstrap
Two variants are present per (dataset, cpu):
  BOOT100     — standard non-parametric bootstrap, -b 100
  UFBOOT1000  — ultrafast bootstrap, -bb 1000

CPU builds: OMP_104 (Sapphire Rapids, INTEL_VANILA) and OMP_48 (Cascade Lake, INTEL_VANILA_CLX).

Completion rules:
  BOOT100:   requires .contree + .boottrees + ≥101 "Total wall-clock" lines
             (1 for ML tree + 100 for bootstrap reps)
  UFBOOT1000: requires .splits.nex + .iqtree + ≥1 "Total wall-clock" line

Outputs:
  runs.csv                  per-run metrics
  status_summary.csv        (dataset,variant,cpu) → status
  fig_status_grid.png       completeness grid
  fig_wall_ufboot_vs_boot.png  UFBOOT vs BOOT100 wall-clock, per dataset
  fig_ml_logL_compare.png   ML logL agreement across variants
  fig_support_distribution.png bootstrap-support histogram per dataset
  SUMMARY.txt               text summary
"""
from __future__ import annotations
import os, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/Users/u7826985/Projects/Nvidia/results/2026_06_25_realdata_modeltamer/bootstrap")
OUT  = Path(__file__).resolve().parent

DATASETS = [
    "Green_plants", "Lassa_Virus", "Mammal_B", "Yeast_nc",
    "Birds", "Butterfly", "InsectsA", "InsectsB", "Mammal", "Plants",
    "Vertebrate",
]
DATATYPE_OF = {
    "Green_plants":"AA", "Vertebrate":"AA", "Yeast_nc":"AA",
    "Birds":"DNA", "Butterfly":"DNA", "InsectsA":"DNA", "InsectsB":"DNA",
    "Lassa_Virus":"DNA", "Mammal":"DNA", "Mammal_B":"DNA", "Plants":"DNA",
}

# ---------- regexes ----------
RE_BEST_LOGL  = re.compile(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)")
RE_BEST_MODEL = re.compile(r"Best-fit model:\s+(\S+)\s+chosen")
RE_MF_WALL    = re.compile(r"Wall-clock time for ModelFinder:\s+([\d.]+)\s+seconds")
RE_TREE_WALL  = re.compile(r"Wall-clock time used for tree search:\s+([\d.]+)\s+sec")
RE_TOTAL_WALL = re.compile(r"Total wall-clock time used:\s+([\d.]+)\s+sec")
RE_PATTERNS   = re.compile(r"Alignment has\s+(\d+)\s+sequences with\s+(\d+)\s+columns,\s+(\d+)\s+distinct patterns")
RE_HOST       = re.compile(r"^Host:\s+(\S+)", re.M)

def parse_name(name: str) -> dict:
    base = name[:-len(".log")] if name.endswith(".log") else name
    variant  = "UFBOOT1000" if "UFBOOT1000" in base else "BOOT100"
    cpu      = "OMP_104" if "OMP_104" in base else "OMP_48"
    dataset  = next((d for d in DATASETS if f"_{d}_" in base), "unknown")
    datatype = DATATYPE_OF.get(dataset, "unknown")
    m = re.search(r"taxa(\d+)", base);   taxa  = int(m.group(1)) if m else None
    m = re.search(r"(\d+)len", base);    sites = int(m.group(1)) if m else None
    return {
        "variant": variant, "cpu": cpu,
        "dataset": dataset, "datatype": datatype,
        "taxa": taxa, "sites": sites,
    }

def support_stats(contree_text: str) -> dict:
    """Count internal-branch bootstrap supports from a Newick contree.
    IQ-TREE writes ")NN:" or ")NN.N:" where NN is the support (0-100)."""
    supports = [int(x) for x in re.findall(r"\)(\d+):", contree_text) if 0 <= int(x) <= 100]
    if not supports: return {"n": 0}
    arr = np.array(supports)
    return {
        "n": len(arr),
        "median": float(np.median(arr)),
        "mean":   float(arr.mean()),
        "ge95":   int((arr >= 95).sum()),
        "ge70":   int((arr >= 70).sum()),
        "lt70":   int((arr <  70).sum()),
    }

# ---------- parse all logs ----------
rows = []
for log in sorted(ROOT.glob("output_cpu_bootstrap_*.log")):
    txt = log.read_text(errors="ignore")
    base = log.name[:-len(".log")]
    prefix = ROOT / base
    meta = parse_name(log.name)

    has_contree = prefix.with_suffix(".contree").exists()
    has_splits  = prefix.with_suffix(".splits.nex").exists()
    has_boot    = prefix.with_suffix(".boottrees").exists()
    has_iqtree  = prefix.with_suffix(".iqtree").exists()
    n_walls     = txt.count("Total wall-clock time used")
    has_crash   = "CRASHES WITH SIGNAL" in txt

    if has_crash:
        status = "CRASHED"
    elif meta["variant"] == "UFBOOT1000":
        status = "COMPLETE" if (has_splits and has_iqtree and n_walls >= 1) else "STILL_RUNNING"
    else:  # BOOT100
        status = "COMPLETE" if (has_contree and has_boot and n_walls >= 101) else "STILL_RUNNING"

    # timing: LAST "Total wall-clock time used" is the pipeline summary
    walls = RE_TOTAL_WALL.findall(txt)
    wall_total_s = float(walls[-1]) if walls else None
    mf_s   = float(RE_MF_WALL.search(txt).group(1))   if RE_MF_WALL.search(txt)   else None
    tree_s = float(RE_TREE_WALL.search(txt).group(1)) if RE_TREE_WALL.search(txt) else None

    m_logl = RE_BEST_LOGL.search(txt)
    logL   = float(m_logl.group(1)) if m_logl else None
    m_mod  = RE_BEST_MODEL.search(txt)
    model  = m_mod.group(1) if m_mod else None
    pat    = RE_PATTERNS.search(txt)
    host   = RE_HOST.search(txt)

    # bash sidecar walltime
    wt_path = prefix.with_suffix(".walltime")
    bash_walltime_s = None
    if wt_path.exists():
        try: bash_walltime_s = float(wt_path.read_text().strip())
        except Exception: pass

    # bootstrap support stats
    sup = {"n": 0}
    if has_contree:
        try: sup = support_stats(prefix.with_suffix(".contree").read_text())
        except Exception: pass

    rows.append({
        "file": log.name, **meta,
        "status": status,
        "n_wallclock_lines": n_walls,
        "wall_total_s":    wall_total_s,
        "wall_mf_s":       mf_s,
        "wall_tree_s":     tree_s,
        "bash_walltime_s": bash_walltime_s,
        "best_logL":       logL,
        "best_model":      model,
        "n_sequences":     int(pat.group(1)) if pat else None,
        "n_columns":       int(pat.group(2)) if pat else None,
        "distinct_patterns": int(pat.group(3)) if pat else None,
        "host":            host.group(1) if host else None,
        "has_contree":     has_contree,
        "has_splits_nex":  has_splits,
        "has_boottrees":   has_boot,
        "has_iqtree":      has_iqtree,
        "support_n":       sup.get("n"),
        "support_median":  sup.get("median"),
        "support_mean":    sup.get("mean"),
        "support_ge95":    sup.get("ge95"),
        "support_ge70":    sup.get("ge70"),
        "support_lt70":    sup.get("lt70"),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "runs.csv", index=False)
print(f"Parsed {len(df)} bootstrap logs → runs.csv")
sc = df["status"].value_counts().to_dict()
print(f"Status: {sc}")

# ---------- status_summary.csv ----------
grid = df.pivot_table(index=["datatype","dataset","taxa","sites"],
                      columns=["variant","cpu"], values="status", aggfunc="first")
grid.to_csv(OUT / "status_summary.csv")

ok = df[df.status == "COMPLETE"].copy()

# ---------- palette ----------
COLOR_BOOT100_104   = "#c72e29"
COLOR_BOOT100_48    = "#f4a582"
COLOR_UFBOOT100_104 = "#016c59"
COLOR_UFBOOT100_48  = "#a6dba0"
PALETTE = {
    ("BOOT100","OMP_104"): COLOR_BOOT100_104,
    ("BOOT100","OMP_48"):  COLOR_BOOT100_48,
    ("UFBOOT1000","OMP_104"): COLOR_UFBOOT100_104,
    ("UFBOOT1000","OMP_48"):  COLOR_UFBOOT100_48,
}
LABEL = {
    ("BOOT100","OMP_104"):   "BOOT100 · SPR 104T",
    ("BOOT100","OMP_48"):    "BOOT100 · CLX 48T",
    ("UFBOOT1000","OMP_104"):"UFBOOT · SPR 104T",
    ("UFBOOT1000","OMP_48"): "UFBOOT · CLX 48T",
}
BAR_ORDER = [("BOOT100","OMP_48"), ("BOOT100","OMP_104"),
             ("UFBOOT1000","OMP_48"), ("UFBOOT1000","OMP_104")]

# ---------- fig: status grid ----------
STATUS_COLOR = {"COMPLETE":"#2ca02c", "STILL_RUNNING":"#1f77b4",
                "CRASHED":"#d62728"}

cells = (df[["datatype","dataset","taxa","sites"]].drop_duplicates()
         .sort_values(["datatype","sites","taxa"], ascending=[True,False,False])
         .reset_index(drop=True))
cells["label"] = (cells.datatype + "/" + cells.dataset
                  + " t=" + cells.taxa.astype(str)
                  + " L=" + cells.sites.astype(str))

fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(cells))))
for j, (var, cpu) in enumerate(BAR_ORDER):
    for i, row in cells.iterrows():
        sub = df[(df.datatype==row.datatype) & (df.dataset==row.dataset)
                 & (df.taxa==row.taxa) & (df.sites==row.sites)
                 & (df.variant==var) & (df.cpu==cpu)]
        color = "#eeeeee" if sub.empty else STATUS_COLOR.get(sub.iloc[0].status, "#888")
        ax.add_patch(plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9,
                                   facecolor=color, edgecolor="#444"))
ax.set_xlim(-0.6, len(BAR_ORDER)-0.4); ax.set_ylim(len(cells)-0.5, -0.5)
ax.set_xticks(range(len(BAR_ORDER)))
ax.set_xticklabels([LABEL[k] for k in BAR_ORDER], rotation=35, ha="right", fontsize=8)
ax.set_yticks(range(len(cells))); ax.set_yticklabels(cells.label, fontsize=8)
ax.set_title("Bootstrap run status")
handles = [plt.Rectangle((0,0),1,1, color=c, label=k) for k,c in STATUS_COLOR.items()]
handles.append(plt.Rectangle((0,0),1,1, color="#eeeeee", label="not attempted"))
ax.legend(handles=handles, bbox_to_anchor=(1.02,1), loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_status_grid.png", dpi=140); plt.close(fig)

# ---------- fig: UFBOOT vs BOOT100 wall-clock (COMPLETE only) ----------
if not ok.empty:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ds_order = list(cells.dataset)
    x = np.arange(len(ds_order))
    w = 0.8 / len(BAR_ORDER)
    for k, (var, cpu) in enumerate(BAR_ORDER):
        ys = []
        for ds in ds_order:
            sub = ok[(ok.dataset==ds) & (ok.variant==var) & (ok.cpu==cpu)]
            ys.append(sub.wall_total_s.mean()/3600 if not sub.empty else np.nan)
        pos = x + (k - (len(BAR_ORDER)-1)/2) * w
        ax.bar(pos, np.nan_to_num(ys), w, color=PALETTE[(var,cpu)],
               edgecolor="black", label=LABEL[(var,cpu)])
        for xi, y in zip(pos, ys):
            if pd.notna(y):
                ax.text(xi, y, f"{y:.1f}h" if y >= 1 else f"{y*60:.0f}m",
                        ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(ds_order, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Total wall-clock (h, log)")
    ax.set_title("Bootstrap wall-clock — UFBOOT (ultrafast) vs BOOT100 (standard), by dataset")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(axis="y", which="both", alpha=0.3, linestyle="--")
    fig.tight_layout(); fig.savefig(OUT / "fig_wall_ufboot_vs_boot.png", dpi=140); plt.close(fig)

# ---------- fig: ML logL agreement ----------
if not ok.empty and ok.best_logL.notna().any():
    ref = ok[(ok.variant=="UFBOOT1000") & (ok.cpu=="OMP_104")][[
        "dataset","best_logL"]].rename(columns={"best_logL":"logL_ref"})
    agree = ok.merge(ref, on="dataset", how="inner")
    agree = agree[~((agree.variant=="UFBOOT1000") & (agree.cpu=="OMP_104"))]
    agree["delta_logL"] = agree.best_logL - agree.logL_ref
    agree.to_csv(OUT / "logL_agreement.csv", index=False)
    if not agree.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for (var, cpu), gr in agree.groupby(["variant","cpu"]):
            ax.scatter(gr.sites, gr.delta_logL, s=80, color=PALETTE[(var,cpu)],
                       edgecolor="black", linewidths=0.5, label=LABEL[(var,cpu)])
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("sites"); ax.set_ylabel("Δ ML logL vs UFBOOT1000 OMP_104 reference")
        ax.set_title("ML log-likelihood agreement across bootstrap variants")
        ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "fig_ml_logL_compare.png", dpi=140); plt.close(fig)

# ---------- fig: bootstrap support distribution ----------
supp = ok[ok.support_n.notna() & (ok.support_n > 0)].copy()
if not supp.empty:
    ds_list = sorted(supp.dataset.unique())
    fig, ax = plt.subplots(figsize=(max(9, 0.55*len(ds_list)*len(BAR_ORDER)), 5))
    x = np.arange(len(ds_list))
    w = 0.8 / len(BAR_ORDER)
    for k, (var, cpu) in enumerate(BAR_ORDER):
        y_ge95, y_lt70, y_n = [], [], []
        for ds in ds_list:
            sub = supp[(supp.dataset==ds) & (supp.variant==var) & (supp.cpu==cpu)]
            if sub.empty:
                y_ge95.append(np.nan); y_lt70.append(np.nan); y_n.append(0)
            else:
                r = sub.iloc[0]
                y_ge95.append(r.support_ge95 / r.support_n * 100)
                y_lt70.append(r.support_lt70 / r.support_n * 100)
                y_n.append(int(r.support_n))
        pos = x + (k - (len(BAR_ORDER)-1)/2) * w
        ax.bar(pos, np.nan_to_num(y_ge95), w, color=PALETTE[(var,cpu)],
               edgecolor="black", label=LABEL[(var,cpu)]+" (≥95%)")
        for xi, y, n in zip(pos, y_ge95, y_n):
            if pd.notna(y):
                ax.text(xi, y, f"n={n}", ha="center", va="bottom",
                        fontsize=6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(ds_list, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("% of internal branches with support ≥ 95")
    ax.set_ylim(0, 105)
    ax.set_title("Bootstrap support distribution — % branches with ≥95% support")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout(); fig.savefig(OUT / "fig_support_distribution.png", dpi=140); plt.close(fig)

# ---------- SUMMARY.txt ----------
with (OUT / "SUMMARY.txt").open("w") as fh:
    fh.write("Bootstrap runs — 2026_06_25 real-data set\n")
    fh.write("=" * 60 + "\n\n")
    fh.write(f"Source: {ROOT}\n")
    fh.write(f"Total logs parsed: {len(df)}\n")
    for k, v in sc.items():
        fh.write(f"  {k:<14} {v}\n")
    fh.write("\nCounts by (variant, cpu):\n")
    for (var, cpu), gr in df.groupby(["variant","cpu"]):
        n = len(gr); ok_n = (gr.status=="COMPLETE").sum()
        cr_n = (gr.status=="CRASHED").sum()
        run_n = (gr.status=="STILL_RUNNING").sum()
        fh.write(f"  {var:<11} {cpu:<8}  runs={n}  complete={ok_n}  crashed={cr_n}  running={run_n}\n")

    fh.write("\nCOMPLETE runs — wall time (h):\n")
    for _, r in ok.sort_values(["dataset","variant","cpu"]).iterrows():
        h = r.wall_total_s/3600 if pd.notna(r.wall_total_s) else None
        fh.write(f"  {r.dataset:<14} {r.variant:<11} {r.cpu:<8} {r.best_model:<20} "
                 f"wall={h:.2f}h  logL={r.best_logL:,.1f}\n" if h is not None
                 else f"  {r.dataset:<14} {r.variant:<11} {r.cpu:<8}  (no wall)\n")

    if not supp.empty:
        fh.write("\nBootstrap support (COMPLETE runs with contree/splits):\n")
        for _, r in supp.sort_values(["dataset","variant","cpu"]).iterrows():
            fh.write(f"  {r.dataset:<14} {r.variant:<11} {r.cpu:<8} "
                     f"n={int(r.support_n):3d}  median={r.support_median:5.1f}  "
                     f"≥95={int(r.support_ge95):3d}  <70={int(r.support_lt70):3d}\n")

    fh.write("\nFigures written:\n")
    for p in sorted(OUT.glob("fig_*.png")):
        fh.write(f"  {p.name}\n")

print("Done. Wrote:")
for p in sorted(OUT.iterdir()):
    if p.is_file():
        print(" ", p.name)
