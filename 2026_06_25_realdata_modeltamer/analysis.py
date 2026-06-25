"""Parse the 2026_06_25 real-data ModelFinder+tree-search IQ-TREE logs and
summarise wall-clock, energy, memory, and best log-likelihood across hardware
variants for empirical datasets pulled from NCI.

Source logs: /Users/u7826985/Projects/Nvidia/results/2026_06_25_realdata_modeltamer
Layout: flat — all output_*.log files live in one dir; everything is encoded
        in the filename.

Datasets observed (11):
  DNA:  Birds, Butterfly, InsectsA, InsectsB, Lassa_Virus, Mammal, Mammal_B, Plants
  AA:   Green_plants, Vertebrate, Yeast_nc

Variants observed:
  cpu_OMP_48, cpu_OMP_104
  openACC_A100, openACC_A100_nt12
  openACC_V100, openACC_V100_nt12
  openACC_H200, openACC_H200_nt12

A run is COMPLETE if its companion .treefile exists and the log shows
"Total wall-clock time used". Otherwise the tail is inspected for
CRASHED / OOM / STILL_RUNNING / ERROR.

Outputs:
  runs.csv                    per-log metrics
  status_summary.csv          per (datatype,dataset,taxa,sites,hardware) status
  fig_status_grid.png         completeness matrix across (cell × hardware)
  fig_wall_per_dataset.png    wall-clock bars per dataset, grouped by hardware
  fig_speedup_vs_cpu48.png    speedup heatmap vs cpu_OMP_48
  fig_modelfinder_vs_tree.png MF vs tree-search wall split, stacked, per cell
  fig_energy_stacked.png      CPU+GPU energy stacks, per cell
  fig_memory.png              CPU and GPU peak memory
  fig_logL_agreement.png      log-L delta vs cpu_OMP_104
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/Users/u7826985/Projects/Nvidia/results/2026_06_25_realdata_modeltamer")
OUT  = Path(__file__).resolve().parent

# Datasets present in the run set — used to anchor the filename parser so
# that dataset names with embedded underscores (Green_plants, Yeast_nc,
# Lassa_Virus, Mammal_B) survive splitting.
DATASETS = [
    "Green_plants", "Lassa_Virus", "Mammal_B", "Yeast_nc",
    "Birds", "Butterfly", "InsectsA", "InsectsB", "Mammal", "Plants",
    "Vertebrate",
]

# ---------- log regexes (same shape as 2026_06_24_simulated_results) ----------
RE_TOTAL_WALL = re.compile(r"Total wall-clock time used:\s+([\d.]+)\s+sec")
RE_TREE_WALL  = re.compile(r"Wall-clock time used for tree search:\s+([\d.]+)\s+sec")
RE_MF_WALL    = re.compile(r"Wall-clock time for ModelFinder:\s+([\d.]+)\s+seconds")
RE_BEST_LOGL  = re.compile(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)")
RE_BEST_MODEL = re.compile(r"Best-fit model:\s+(\S+)\s+chosen")
RE_E_TS       = re.compile(r"Energy used for tree search:\s+CPU\s+([\d.]+)\s+J(?:,\s*GPU\s+([\d.]+)\s+J)?")
RE_E_MF       = re.compile(r"Energy used for ModelFinder:\s+CPU\s+([\d.]+)\s+J(?:,\s*GPU\s+([\d.]+)\s+J)?")
RE_GPU_MEM    = re.compile(r"GPU mem:\s+([\d.]+)\s*/\s*([\d.]+)\s+MB.*?\+\s*([\d.]+)\s+MB")
RE_CPU_MEM    = re.compile(r"CPU mem:\s+([\d.]+)\s*/\s*([\d.]+)\s+MB")
RE_PATTERNS   = re.compile(r"Alignment has\s+(\d+)\s+sequences with\s+(\d+)\s+columns,\s+(\d+)\s+distinct patterns")
RE_HOST       = re.compile(r"^Host:\s+(\S+)", re.M)

def _f(rx, txt, grp=1):
    m = rx.search(txt)
    return float(m.group(grp)) if m else None

# ---------- filename → (dataset, hardware, taxa, sites, datatype) ----------
def parse_name(name: str) -> dict:
    base = name[:-len(".log")] if name.endswith(".log") else name

    datatype = "AA" if "_AA_" in base else ("DNA" if "_DNA_" in base else "unknown")

    dataset = next((d for d in DATASETS if f"_{d}_" in base), "unknown")

    m = re.search(r"taxa(\d+)", base)
    taxa = int(m.group(1)) if m else None

    m = re.search(r"(\d+)taxa", base)
    if taxa is None and m:
        taxa = int(m.group(1))

    m = re.search(r"(\d+)len", base)
    sites = int(m.group(1)) if m else None

    is_cpu     = base.startswith("output_cpu_bfgs_")
    is_openacc = base.startswith("output_openacc_stable_bfgs_")
    has_nt12   = "_nt12_" in base

    if is_cpu:
        if "OMP_104" in base: hw = "cpu_OMP_104"
        elif "OMP_48" in base: hw = "cpu_OMP_48"
        else: hw = "cpu_unknown"
    elif is_openacc:
        if   "_A100_" in base: gpu = "A100"
        elif "_V100_" in base: gpu = "V100"
        elif "_H200_" in base: gpu = "H200"
        else: gpu = "unknownGPU"
        hw = f"openACC_{gpu}" + ("_nt12" if has_nt12 else "")
    else:
        hw = "unknown"

    return {
        "datatype": datatype,
        "dataset":  dataset,
        "taxa":     taxa,
        "sites":    sites,
        "hardware": hw,
    }

def status_from_log(text: str, has_treefile: bool) -> str:
    if has_treefile and "Total wall-clock time used" in text:
        return "COMPLETE"
    tail = text[-4000:]
    if "CRASHES WITH SIGNAL" in tail:               return "CRASHED"
    if "Memory required exceeds your available RAM" in tail: return "OOM"
    if "ERROR" in tail and "Date and Time" not in tail: return "ERROR"
    return "STILL_RUNNING"

# ---------- walltime sidecar (bash-measured) ----------
def read_walltime(log_path: Path) -> float | None:
    wt = log_path.with_suffix(".walltime")
    if not wt.exists():
        return None
    try:
        return float(wt.read_text().strip())
    except Exception:
        return None

# ---------- parse all logs ----------
rows = []
for log in sorted(ROOT.glob("output_*.log")):
    txt = log.read_text(errors="ignore")
    base = log.name[:-len(".log")]
    treefile = (log.parent / (base + ".treefile")).exists()
    status   = status_from_log(txt, treefile)
    meta     = parse_name(log.name)

    gm = RE_GPU_MEM.search(txt)
    cm = RE_CPU_MEM.search(txt)
    pat = RE_PATTERNS.search(txt)
    e_ts = RE_E_TS.search(txt)
    e_mf = RE_E_MF.search(txt)
    host = RE_HOST.search(txt)
    model = RE_BEST_MODEL.search(txt)

    row = {
        "file": log.name,
        **meta,
        "status": status,
        "host": host.group(1) if host else None,
        "best_model": model.group(1) if model else None,
        "n_sequences":       int(pat.group(1)) if pat else None,
        "n_columns":         int(pat.group(2)) if pat else None,
        "distinct_patterns": int(pat.group(3)) if pat else None,
        "wall_total_s":      _f(RE_TOTAL_WALL, txt),
        "wall_tree_s":       _f(RE_TREE_WALL,  txt),
        "wall_mf_s":         _f(RE_MF_WALL,    txt),
        "best_logL":         _f(RE_BEST_LOGL,  txt),
        "energy_cpu_ts_J":   float(e_ts.group(1)) if e_ts else None,
        "energy_gpu_ts_J":   float(e_ts.group(2)) if e_ts and e_ts.group(2) else None,
        "energy_cpu_mf_J":   float(e_mf.group(1)) if e_mf else None,
        "energy_gpu_mf_J":   float(e_mf.group(2)) if e_mf and e_mf.group(2) else None,
        "gpu_mem_peak_MB":   float(gm.group(1)) if gm else None,
        "gpu_mem_cap_MB":    float(gm.group(2)) if gm else None,
        "gpu_mem_delta_MB":  float(gm.group(3)) if gm else None,
        "cpu_mem_peak_MB":   float(cm.group(1)) if cm else None,
        "cpu_mem_cap_MB":    float(cm.group(2)) if cm else None,
        "bash_walltime_s":   read_walltime(log),
    }
    row["energy_cpu_total_J"] = sum(
        x for x in (row["energy_cpu_ts_J"], row["energy_cpu_mf_J"]) if x is not None
    ) or None
    row["energy_gpu_total_J"] = sum(
        x for x in (row["energy_gpu_ts_J"], row["energy_gpu_mf_J"]) if x is not None
    ) or None
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT / "runs.csv", index=False)
print(f"Parsed {len(df)} logs → runs.csv")
print("Status counts:", df["status"].value_counts().to_dict())
print("Hardware counts:\n", df["hardware"].value_counts())

ok = df[df["status"] == "COMPLETE"].copy()
print(f"COMPLETE runs: {len(ok)}")

# ---------- canonical orderings ----------
HW_ORDER = [
    "cpu_OMP_48", "cpu_OMP_104",
    "openACC_V100",
    "openACC_A100",
    "openACC_H200",
]
HW_STYLE = {
    "cpu_OMP_48":        dict(color="#666666", marker="o", ls="-"),
    "cpu_OMP_104":       dict(color="#000000", marker="o", ls="-"),
    "openACC_V100":      dict(color="#3182bd", marker="s", ls="-"),
    "openACC_A100":      dict(color="#e6550d", marker="^", ls="-"),
    "openACC_H200":      dict(color="#238b45", marker="D", ls="-"),
}

# ---------- cell ordering: largest first, group by datatype ----------
cells = (df[["datatype", "dataset", "taxa", "sites"]]
         .drop_duplicates()
         .sort_values(["datatype", "sites", "taxa"], ascending=[True, False, False])
         .reset_index(drop=True))
cells["label"] = (cells["datatype"] + "/" + cells["dataset"]
                  + " t=" + cells["taxa"].astype(str)
                  + " L=" + cells["sites"].astype(str))

# ---------- status grid ----------
STATUS_COLOR = {"COMPLETE":"#2ca02c", "STILL_RUNNING":"#1f77b4",
                "CRASHED":"#d62728", "OOM":"#ff7f0e", "ERROR":"#7f7f7f"}

fig, ax = plt.subplots(figsize=(10, max(4, 0.36 * len(cells))))
for j, hw in enumerate(HW_ORDER):
    for i, row in cells.iterrows():
        sub = df[(df["datatype"]==row["datatype"]) & (df["dataset"]==row["dataset"])
                 & (df["taxa"]==row["taxa"]) & (df["sites"]==row["sites"])
                 & (df["hardware"]==hw)]
        if sub.empty:
            color = "#eeeeee"
        else:
            color = STATUS_COLOR.get(sub.iloc[0]["status"], "#888")
        ax.add_patch(plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9,
                                   facecolor=color, edgecolor="#444"))
ax.set_xlim(-0.6, len(HW_ORDER)-0.4); ax.set_ylim(len(cells)-0.5, -0.5)
ax.set_xticks(range(len(HW_ORDER))); ax.set_xticklabels(HW_ORDER, rotation=35, ha="right")
ax.set_yticks(range(len(cells))); ax.set_yticklabels(cells["label"], fontsize=8)
ax.set_title("Run status — real-data ModelFinder+treesearch (2026_06_25)")
handles = [plt.Rectangle((0,0),1,1, color=c, label=k) for k,c in STATUS_COLOR.items()]
handles.append(plt.Rectangle((0,0),1,1, color="#eeeeee", label="not attempted"))
ax.legend(handles=handles, bbox_to_anchor=(1.02,1), loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_status_grid.png", dpi=140); plt.close(fig)

status_grid = df.pivot_table(index=["datatype","dataset","taxa","sites"],
                             columns="hardware", values="status", aggfunc="first")
status_grid = status_grid.reindex(columns=[h for h in HW_ORDER if h in status_grid.columns])
status_grid.to_csv(OUT / "status_summary.csv")

# ---------- bars: wall_total per dataset, grouped by hardware ----------
if not ok.empty:
    cell_keys = list(cells.itertuples(index=False))
    labels = [c.label for c in cell_keys]
    hws = [h for h in HW_ORDER if h in ok["hardware"].unique()]
    x = np.arange(len(cell_keys))
    w = 0.8 / max(1, len(hws))
    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(cell_keys)), 6))
    for k, hw in enumerate(hws):
        ys = []
        for c in cell_keys:
            sub = ok[(ok["datatype"]==c.datatype) & (ok["dataset"]==c.dataset)
                     & (ok["taxa"]==c.taxa) & (ok["sites"]==c.sites)
                     & (ok["hardware"]==hw)]
            ys.append(sub["wall_total_s"].mean() if not sub.empty else np.nan)
        pos = x + (k - (len(hws)-1)/2) * w
        ax.bar(pos, np.nan_to_num(ys), w, color=HW_STYLE[hw]["color"],
               edgecolor="black", label=hw)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Total wall-clock (s, log)")
    ax.set_title("Total wall-clock per (dataset × hardware) — COMPLETE runs only")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_wall_per_dataset.png", dpi=140); plt.close(fig)

# ---------- speedup heatmap vs cpu_OMP_48 ----------
def speedup_table(metric="wall_total_s", baseline_hw="cpu_OMP_48"):
    base = ok[ok["hardware"]==baseline_hw][["datatype","dataset","taxa","sites",metric]]\
            .rename(columns={metric:"base"})
    m = ok.merge(base, on=["datatype","dataset","taxa","sites"], how="inner")
    m["speedup"] = m["base"] / m[metric]
    return m

m = speedup_table()
if not m.empty:
    pivot = m.pivot_table(
        index=["datatype","dataset","taxa","sites"],
        columns="hardware", values="speedup", aggfunc="mean")
    pivot = pivot[[h for h in HW_ORDER if h in pivot.columns]]
    pivot.to_csv(OUT / "speedup_vs_cpu48.csv")

    fig, ax = plt.subplots(figsize=(9, max(4, 0.36 * len(pivot))))
    arr = pivot.to_numpy(dtype=float)
    vmax = max(2.0, np.nanmax(arr) if np.isfinite(arr).any() else 2.0)
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns,
                                                                 rotation=35, ha="right")
    ylabels = [f"{a}/{b} t={c} L={d}" for a,b,c,d in pivot.index]
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels(ylabels, fontsize=7)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}×", ha="center", va="center",
                        fontsize=6, color="black" if v < 1.5 else "white")
    ax.set_title("Speedup vs cpu_OMP_48 — wall_total")
    fig.colorbar(im, ax=ax, label="speedup")
    fig.tight_layout(); fig.savefig(OUT / "fig_speedup_vs_cpu48.png", dpi=140); plt.close(fig)

# ---------- stacked ModelFinder vs treesearch wall split ----------
if not ok.empty:
    sub = ok.dropna(subset=["wall_mf_s","wall_tree_s"]).copy()
    if not sub.empty:
        sub["label_full"] = (sub["datatype"] + "/" + sub["dataset"]
                             + " t" + sub["taxa"].astype(str)
                             + " L" + sub["sites"].astype(str)
                             + "\n" + sub["hardware"])
        sub = sub.sort_values(["datatype","sites","dataset","hardware"])
        x = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(max(10, 0.32 * len(sub)), 6))
        ax.bar(x, sub["wall_mf_s"],   color="#a6cee3", label="ModelFinder")
        ax.bar(x, sub["wall_tree_s"], bottom=sub["wall_mf_s"],
               color="#1f78b4", label="Tree search")
        ax.set_xticks(x); ax.set_xticklabels(sub["label_full"], rotation=80,
                                              fontsize=6, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("Wall-clock (s, log)")
        ax.set_title("ModelFinder vs Tree-search wall split — COMPLETE runs")
        ax.legend(); ax.grid(axis="y", which="both", alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "fig_modelfinder_vs_tree.png", dpi=140); plt.close(fig)

# ---------- energy stacked: CPU + GPU per (cell × hardware) ----------
e = ok.dropna(subset=["energy_cpu_total_J"], how="all").copy()
if not e.empty:
    e["cell"] = (e["datatype"]+"/"+e["dataset"]+" t"+e["taxa"].astype(str)
                 +" L"+e["sites"].astype(str))
    hws = [h for h in HW_ORDER if h in e["hardware"].unique()]
    cells_e = sorted(e["cell"].unique())
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(cells_e) * len(hws) / 2), 5.5))
    x = np.arange(len(cells_e))
    w = 0.8 / max(1, len(hws))
    for k, hw in enumerate(hws):
        cpu = [e[(e["cell"]==c) & (e["hardware"]==hw)]["energy_cpu_total_J"].mean()
               for c in cells_e]
        gpu = [e[(e["cell"]==c) & (e["hardware"]==hw)]["energy_gpu_total_J"].mean()
               for c in cells_e]
        cpu_a = np.nan_to_num(cpu); gpu_a = np.nan_to_num(gpu)
        pos = x + (k - (len(hws)-1)/2) * w
        ax.bar(pos, cpu_a, w, color=HW_STYLE[hw]["color"], edgecolor="black",
               alpha=0.55, label=f"{hw} CPU")
        ax.bar(pos, gpu_a, w, bottom=cpu_a, color=HW_STYLE[hw]["color"],
               edgecolor="black", alpha=1.0, hatch="//", label=f"{hw} GPU")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(cells_e, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Energy (J, log)")
    ax.set_title("CPU + GPU energy by hardware (stacked) — COMPLETE runs")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_energy_stacked.png", dpi=140); plt.close(fig)

# ---------- memory ----------
mem = ok.copy()
if not mem.empty:
    mem["cell"] = (mem["datatype"]+"/"+mem["dataset"]+" t"+mem["taxa"].astype(str)
                   +" L"+mem["sites"].astype(str))
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.35*len(mem["cell"].unique()))))
    for ax, col, title in [
        (axes[0], "cpu_mem_peak_MB", "CPU peak resident (MB)"),
        (axes[1], "gpu_mem_peak_MB", "GPU peak (MB)"),
    ]:
        cells_m = sorted(mem["cell"].unique())
        hws = [h for h in HW_ORDER if h in mem["hardware"].unique()]
        y = np.arange(len(cells_m))
        w = 0.8 / max(1, len(hws))
        for k, hw in enumerate(hws):
            vals = [mem[(mem["cell"]==c) & (mem["hardware"]==hw)][col].mean()
                    for c in cells_m]
            pos = y + (k - (len(hws)-1)/2) * w
            ax.barh(pos, np.nan_to_num(vals), w,
                    color=HW_STYLE[hw]["color"], edgecolor="black", label=hw)
        ax.set_yticks(y); ax.set_yticklabels(cells_m, fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel(title)
        ax.invert_yaxis()
        ax.grid(axis="x", which="both", alpha=0.3)
    axes[0].legend(fontsize=6, loc="lower right")
    fig.suptitle("Peak memory usage by run (COMPLETE only)")
    fig.tight_layout(); fig.savefig(OUT / "fig_memory.png", dpi=140); plt.close(fig)

# ---------- logL agreement vs cpu_OMP_104 ----------
ref = ok[ok["hardware"]=="cpu_OMP_104"][["datatype","dataset","taxa","sites","best_logL"]]\
        .rename(columns={"best_logL":"logL_ref"})
agree = ok.merge(ref, on=["datatype","dataset","taxa","sites"], how="inner")
agree = agree[agree["hardware"] != "cpu_OMP_104"].copy()
agree["delta_logL"] = agree["best_logL"] - agree["logL_ref"]
agree["rel_diff"]   = (agree["best_logL"] - agree["logL_ref"]) / agree["logL_ref"].abs()
agree.to_csv(OUT / "logL_agreement.csv", index=False)

if not agree.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    for hw in [h for h in HW_ORDER if h in agree["hardware"].unique()]:
        s = agree[agree["hardware"]==hw]
        style = HW_STYLE[hw]
        ax.scatter(s["sites"], s["delta_logL"], label=hw,
                   color=style["color"], marker=style["marker"], s=55,
                   edgecolor="black", linewidths=0.5)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("sites"); ax.set_ylabel("Δ logL  vs cpu_OMP_104")
    ax.set_title("Best log-likelihood agreement vs CPU OMP_104 reference")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_logL_agreement.png", dpi=140); plt.close(fig)

# ---------- text summary ----------
status_counts = df["status"].value_counts().to_dict()
with (OUT / "SUMMARY.txt").open("w") as fh:
    fh.write("Real-data empirical IQ-TREE results — 2026_06_25 snapshot\n")
    fh.write("=" * 60 + "\n\n")
    fh.write(f"Source: {ROOT}\n")
    fh.write(f"Total logs parsed:  {len(df)}\n")
    for k, v in status_counts.items():
        fh.write(f"  {k:<14} {v}\n")
    fh.write("\nHardware variants (runs / complete):\n")
    for hw, c in df["hardware"].value_counts().items():
        ok_c = (df[(df["hardware"]==hw) & (df["status"]=="COMPLETE")]).shape[0]
        fh.write(f"  {hw:<22} runs={c:<3} complete={ok_c}\n")
    fh.write("\nCells (datatype/dataset/taxa/sites) with completion count:\n")
    cc = ok.groupby(["datatype","dataset","taxa","sites"]).size()\
           .reset_index(name="n_hw").sort_values(["datatype","sites","dataset"])
    for _, r in cc.iterrows():
        fh.write(f"  {r['datatype']:<3} {r['dataset']:<14} taxa={r['taxa']:<4} "
                 f"L={r['sites']:<8}  complete_HWs={r['n_hw']}\n")
    fh.write("\n")
    if not agree.empty:
        worst = agree.reindex(agree["delta_logL"].abs().sort_values(ascending=False).index).head(8)
        fh.write("Top 8 worst |Δ logL| vs cpu_OMP_104 (sanity):\n")
        for _, r in worst.iterrows():
            fh.write(f"  {r['hardware']:<22} {r['datatype']}/{r['dataset']:<14} "
                     f"taxa={r['taxa']:<4} L={r['sites']:<8} "
                     f"Δ={r['delta_logL']:+9.3f}  rel={r['rel_diff']:+.2e}\n")
    if not m.empty:
        fh.write("\nMedian speedup vs cpu_OMP_48 by hardware (across all cells):\n")
        med = m.groupby("hardware")["speedup"].median().sort_values(ascending=False)
        for hw, v in med.items():
            fh.write(f"  {hw:<22} {v:5.2f}×\n")
    fh.write("\nFigures written:\n")
    for p in sorted(OUT.glob("fig_*.png")):
        fh.write(f"  {p.name}\n")

print("Done. Wrote:")
for p in sorted(OUT.iterdir()):
    if p.is_file():
        print(" ", p.name)
