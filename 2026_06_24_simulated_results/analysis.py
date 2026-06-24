"""Parse the 2026_06_24 simulated complex_data IQ-TREE logs and summarise
wall-clock, energy, memory, and best log-likelihood across hardware variants.

Source logs: /Users/u7826985/Projects/Nvidia/results/2026_06_24_simulated_results
Layout: <DATATYPE>/<MODEL>/taxa_<N>/len_<L>/tree_1/output_*.log

Variants observed (8):
  cpu_OMP_104, cpu_OMP_48,
  cudajolt_H200, cudajolt_V100,
  openACC_H200, openACC_H200_nt12,
  openACC_V100, openACC_V100_nt12

A run is considered COMPLETE if its companion .treefile exists. Otherwise the
log tail is inspected to label STILL_RUNNING, CRASHED, or OOM.

Outputs:
  runs.csv               per-log metrics (all 46 runs)
  status_summary.csv     per (datatype,model,taxa,sites,hardware) status
  fig_status_grid.png    completeness matrix
  fig_wall_vs_sites_*.png    wall-clock scaling per (datatype,model)
  fig_speedup_vs_cpu48.png   speedup heatmap
  fig_energy_breakdown.png   CPU vs GPU energy (where measured)
  fig_logL_agreement.png     log-likelihood agreement vs CPU OMP_104
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/Users/u7826985/Projects/Nvidia/results/2026_06_24_simulated_results")
OUT = Path(__file__).resolve().parent

# ---------- regexes ----------
RE_TOTAL_WALL = re.compile(r"Total wall-clock time used:\s+([\d.]+)\s+sec")
RE_TREE_WALL  = re.compile(r"Wall-clock time used for tree search:\s+([\d.]+)\s+sec")
RE_BEST_LOGL  = re.compile(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)")
RE_FAST_ML    = re.compile(r"Time for fast ML tree search:\s+([\d.]+)\s+seconds")
RE_E_CPU      = re.compile(r"Energy:\s*\n\s*CPU:\s+([\d.]+)\s+J", re.M)
RE_E_GPU      = re.compile(r"^\s*GPU:\s+([\d.]+)\s+J", re.M)
RE_E_CPU_TS   = re.compile(r"Energy used for tree search:\s+CPU\s+([\d.]+)\s+J(?:,\s*GPU\s+([\d.]+)\s+J)?")
RE_E_CPU_MF   = re.compile(r"Energy used for ModelFinder:\s+CPU\s+([\d.]+)\s+J(?:,\s*GPU\s+([\d.]+)\s+J)?")
RE_GPU_MEM    = re.compile(r"GPU mem:\s+([\d.]+)\s*/\s*([\d.]+)\s+MB.*?\+([\d.]+)\s+MB")
RE_CPU_MEM    = re.compile(r"CPU mem:\s+([\d.]+)\s*/\s*([\d.]+)\s+MB")
RE_PATTERNS   = re.compile(r"Alignment has\s+(\d+)\s+sequences with\s+(\d+)\s+columns,\s+(\d+)\s+distinct patterns")
RE_HOST       = re.compile(r"^Host:\s+(\S+)", re.M)
RE_NT         = re.compile(r"-nt\s+(\d+)")

def _f(rx, txt, grp=1):
    m = rx.search(txt)
    return float(m.group(grp)) if m else None

# ---------- classify file name ----------
# Distinguishes openACC stable vs JOLT_h2tiling builds, and presence of _nt12 pin.
def hardware_of(name: str) -> str:
    if "cpu_bfgs" in name and "OMP_104" in name: return "cpu_OMP_104"
    if "cpu_bfgs" in name and "OMP_48"  in name: return "cpu_OMP_48"
    if "cudajolt" in name and "H200" in name:    return "cudajolt_H200"
    if "cudajolt" in name and "V100" in name:    return "cudajolt_V100"
    # JOLT_h2tiling build (different OpenACC implementation):
    if "openACC_bfgs_JOLT_h2tiling" in name and "H200" in name and "_nt12" in name:
        return "openACC_JOLT_h2tiling_H200_nt12"
    if "openACC_bfgs_JOLT_h2tiling" in name and "H200" in name:
        return "openACC_JOLT_h2tiling_H200"
    if "openACC_bfgs_JOLT_h2tiling" in name and "V100" in name and "_nt12" in name:
        return "openACC_JOLT_h2tiling_V100_nt12"
    if "openACC_bfgs_JOLT_h2tiling" in name and "V100" in name:
        return "openACC_JOLT_h2tiling_V100"
    # stable openACC build:
    if "openACC"  in name and "H200" in name and "_nt12_" in name: return "openACC_stable_H200_nt12"
    if "openACC"  in name and "H200" in name:    return "openACC_stable_H200"
    if "openACC"  in name and "V100" in name and "_nt12_" in name: return "openACC_stable_V100_nt12"
    if "openACC"  in name and "V100" in name:    return "openACC_stable_V100"
    return "unknown"

def status_from_log(text: str, has_treefile: bool) -> str:
    if has_treefile and "Total wall-clock time used" in text:
        return "COMPLETE"
    tail = text[-4000:]
    if "CRASHES WITH SIGNAL" in tail:
        return "CRASHED"
    if "Memory required exceeds your available RAM" in tail:
        return "OOM"
    if "ERROR" in tail and "Date and Time" not in tail:
        return "ERROR"
    return "STILL_RUNNING"

rows = []
for log in sorted(ROOT.rglob("output_*.log")):
    txt = log.read_text(errors="ignore")
    rel = log.relative_to(ROOT)
    # path: <DT>/<MODEL>/taxa_<N>/len_<L>/tree_1/<file>
    parts = rel.parts
    datatype = parts[0]
    model    = parts[1]
    taxa     = int(parts[2].split("_")[1])
    sites    = int(parts[3].split("_")[1])
    tree_dir = log.parent

    base = log.name[:-len(".log")]
    treefile = (tree_dir / (base + ".treefile")).exists()
    status   = status_from_log(txt, treefile)

    gm = RE_GPU_MEM.search(txt)
    cm = RE_CPU_MEM.search(txt)
    pat = RE_PATTERNS.search(txt)
    e_ts = RE_E_CPU_TS.search(txt)
    e_mf = RE_E_CPU_MF.search(txt)
    nt = RE_NT.search(txt)
    host = RE_HOST.search(txt)

    rows.append({
        "file": log.name,
        "datatype": datatype,
        "model": model,
        "taxa": taxa,
        "sites": sites,
        "hardware": hardware_of(log.name),
        "status": status,
        "nt": int(nt.group(1)) if nt else None,
        "host": host.group(1) if host else None,
        "distinct_patterns": int(pat.group(3)) if pat else None,
        "wall_total_s": _f(RE_TOTAL_WALL, txt),
        "wall_tree_s":  _f(RE_TREE_WALL, txt),
        "wall_fastml_s": _f(RE_FAST_ML, txt),
        "best_logL": _f(RE_BEST_LOGL, txt),
        "energy_cpu_total_J": _f(RE_E_CPU, txt),
        "energy_gpu_total_J": _f(RE_E_GPU, txt),
        "energy_cpu_ts_J":   float(e_ts.group(1)) if e_ts else None,
        "energy_gpu_ts_J":   float(e_ts.group(2)) if e_ts and e_ts.group(2) else None,
        "energy_cpu_mf_J":   float(e_mf.group(1)) if e_mf else None,
        "energy_gpu_mf_J":   float(e_mf.group(2)) if e_mf and e_mf.group(2) else None,
        "gpu_mem_peak_MB":  float(gm.group(1)) if gm else None,
        "gpu_mem_cap_MB":   float(gm.group(2)) if gm else None,
        "gpu_mem_delta_MB": float(gm.group(3)) if gm else None,
        "cpu_mem_peak_MB":  float(cm.group(1)) if cm else None,
        "cpu_mem_cap_MB":   float(cm.group(2)) if cm else None,
    })

df = pd.DataFrame(rows)
# derive wall_modelfinder = wall_total - wall_tree (when both present)
df["wall_mf_s"] = df["wall_total_s"] - df["wall_tree_s"]
df.to_csv(OUT / "runs.csv", index=False)
print(f"Parsed {len(df)} logs → runs.csv")

# ---------- status summary ----------
status_counts = df["status"].value_counts().to_dict()
print("Status counts:", status_counts)

cell_cols = ["datatype", "model", "taxa", "sites", "hardware"]
status_grid = df.pivot_table(index=["datatype", "model", "taxa", "sites"],
                             columns="hardware",
                             values="status", aggfunc="first")
status_grid.to_csv(OUT / "status_summary.csv")

# ---------- COMPLETE-only frame ----------
ok = df[df["status"] == "COMPLETE"].copy()
print(f"COMPLETE runs: {len(ok)}")

# ----- fig: status grid (qualitative) -----
HW_ORDER = ["cpu_OMP_48", "cpu_OMP_104",
            "cudajolt_V100", "cudajolt_H200",
            "openACC_stable_V100",
            "openACC_stable_H200",
            "openACC_JOLT_h2tiling_V100",
            "openACC_JOLT_h2tiling_H200"]
STATUS_COLOR = {"COMPLETE": "#2ca02c", "STILL_RUNNING": "#1f77b4",
                "CRASHED": "#d62728", "OOM": "#ff7f0e",
                "ERROR": "#7f7f7f"}

cells = df[["datatype", "model", "taxa", "sites"]].drop_duplicates().sort_values(
    ["datatype", "model", "taxa", "sites"]).reset_index(drop=True)
cells["cell"] = (cells["datatype"] + "/" + cells["model"]
                 + " taxa=" + cells["taxa"].astype(str)
                 + " L=" + cells["sites"].astype(str))
fig, ax = plt.subplots(figsize=(11, max(4, 0.28 * len(cells))))
for j, hw in enumerate(HW_ORDER):
    for i, row in cells.iterrows():
        sub = df[(df["datatype"] == row["datatype"]) & (df["model"] == row["model"])
                 & (df["taxa"] == row["taxa"]) & (df["sites"] == row["sites"])
                 & (df["hardware"] == hw)]
        if sub.empty:
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                        facecolor="#eeeeee", edgecolor="#bbbbbb"))
        else:
            color = STATUS_COLOR.get(sub.iloc[0]["status"], "#888")
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                        facecolor=color, edgecolor="#444"))
ax.set_xlim(-0.6, len(HW_ORDER) - 0.4)
ax.set_ylim(len(cells) - 0.5, -0.5)
ax.set_xticks(range(len(HW_ORDER))); ax.set_xticklabels(HW_ORDER, rotation=35, ha="right")
ax.set_yticks(range(len(cells))); ax.set_yticklabels(cells["cell"], fontsize=7)
ax.set_title("Run status across (cell × hardware)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in STATUS_COLOR.items()]
handles.append(plt.Rectangle((0, 0), 1, 1, color="#eeeeee", label="not attempted"))
ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig_status_grid.png", dpi=140)
plt.close(fig)

# ----- fig: wall vs sites per (datatype,model,taxa) -----
HW_STYLE = {
    "cpu_OMP_48":                       dict(color="#666666", marker="o", ls="-"),
    "cpu_OMP_104":                      dict(color="#000000", marker="o", ls="-"),
    "cudajolt_V100":                    dict(color="#9ecae1", marker="s", ls="--"),
    "cudajolt_H200":                    dict(color="#3182bd", marker="s", ls="--"),
    "openACC_stable_V100":              dict(color="#c476a8", marker="^", ls=":"),
    "openACC_stable_V100_nt12":         dict(color="#a01a78", marker="^", ls="-"),
    "openACC_stable_H200":              dict(color="#74c476", marker="D", ls=":"),
    "openACC_stable_H200_nt12":         dict(color="#238b45", marker="D", ls="-"),
    "openACC_JOLT_h2tiling_V100":       dict(color="#fdae6b", marker="v", ls=":"),
    "openACC_JOLT_h2tiling_V100_nt12":  dict(color="#e6550d", marker="v", ls="-"),
    "openACC_JOLT_h2tiling_H200":       dict(color="#bcbddc", marker="P", ls=":"),
    "openACC_JOLT_h2tiling_H200_nt12":  dict(color="#54278f", marker="P", ls="-"),
}

dm_pairs = sorted(ok[["datatype", "model"]].drop_duplicates().itertuples(index=False))
for dt, model in dm_pairs:
    sub_all = ok[(ok["datatype"] == dt) & (ok["model"] == model)]
    taxa_list = sorted(sub_all["taxa"].unique())
    n = len(taxa_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=False, squeeze=False)
    for k, taxa in enumerate(taxa_list):
        ax = axes[0, k]
        s = sub_all[sub_all["taxa"] == taxa]
        for hw in HW_ORDER:
            ss = s[s["hardware"] == hw].sort_values("sites")
            if ss.empty: continue
            style = HW_STYLE[hw]
            ax.plot(ss["sites"], ss["wall_total_s"], label=hw, **style)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("sites"); ax.set_ylabel("wall total (s)")
        ax.set_title(f"taxa={taxa}")
        ax.grid(True, which="both", alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{dt} / {model} — total wall-clock vs alignment length")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_wall_vs_sites_{dt}_{model.replace('+','-')}.png", dpi=140)
    plt.close(fig)

# ----- fig: speedup heatmap vs cpu_OMP_48 (only where both exist) -----
def speedup_table(metric="wall_total_s", baseline_hw="cpu_OMP_48"):
    base = ok[ok["hardware"] == baseline_hw][[
        "datatype", "model", "taxa", "sites", metric]].rename(
            columns={metric: "base"})
    m = ok.merge(base, on=["datatype", "model", "taxa", "sites"], how="inner")
    m["speedup"] = m["base"] / m[metric]
    return m

m = speedup_table()
if not m.empty:
    pivot = m.pivot_table(
        index=["datatype", "model", "taxa", "sites"],
        columns="hardware", values="speedup", aggfunc="mean")
    pivot = pivot[[h for h in HW_ORDER if h in pivot.columns]]
    pivot.to_csv(OUT / "speedup_vs_cpu48.csv")

    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(pivot))))
    arr = pivot.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=max(2, np.nanmax(arr)))
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns,
                                                                 rotation=35, ha="right")
    yticks = [f"{a}/{b} t={c} L={d}" for a, b, c, d in pivot.index]
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels(yticks, fontsize=7)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}×", ha="center", va="center",
                        fontsize=6, color="black" if v < 1.5 else "white")
    ax.set_title("Speedup vs cpu_OMP_48 (wall_total)")
    fig.colorbar(im, ax=ax, label="speedup")
    fig.tight_layout()
    fig.savefig(OUT / "fig_speedup_vs_cpu48.png", dpi=140)
    plt.close(fig)

# ----- fig: energy breakdown (any run with energy reported) -----
# CPU-only runs report `CPU:` (no GPU); cudajolt reports neither; openACC reports both.
egpu = ok[ok["energy_cpu_total_J"].notna() | ok["energy_gpu_total_J"].notna()].copy()
if not egpu.empty:
    egpu["cell"] = (egpu["datatype"] + "/" + egpu["model"] + "/t" + egpu["taxa"].astype(str)
                    + "/L" + egpu["sites"].astype(str))
    cells_e = sorted(egpu["cell"].unique())
    hws = [h for h in HW_ORDER if h in egpu["hardware"].unique()]
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(cells_e) * len(hws)), 5))
    x = np.arange(len(cells_e))
    w = 0.8 / max(1, len(hws))
    for k, hw in enumerate(hws):
        cpu = [egpu[(egpu["cell"] == c) & (egpu["hardware"] == hw)]["energy_cpu_total_J"].mean()
               for c in cells_e]
        gpu = [egpu[(egpu["cell"] == c) & (egpu["hardware"] == hw)]["energy_gpu_total_J"].mean()
               for c in cells_e]
        cpu_a = np.nan_to_num(cpu); gpu_a = np.nan_to_num(gpu)
        pos = x + (k - (len(hws) - 1) / 2) * w
        ax.bar(pos, cpu_a, w, color=HW_STYLE[hw]["color"], edgecolor="black",
               label=f"{hw} CPU", alpha=0.55)
        ax.bar(pos, gpu_a, w, bottom=cpu_a, color=HW_STYLE[hw]["color"],
               edgecolor="black", label=f"{hw} GPU", alpha=1.0, hatch="//")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(cells_e, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("energy (J, log)")
    ax.set_title("Energy breakdown (CPU + GPU stacks) by hardware")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_energy_breakdown.png", dpi=140)
    plt.close(fig)

# ----- fig: logL agreement (each completed run vs cpu_OMP_104 for same cell) -----
ref = ok[ok["hardware"] == "cpu_OMP_104"][[
    "datatype", "model", "taxa", "sites", "best_logL"]].rename(
        columns={"best_logL": "logL_ref"})
agree = ok.merge(ref, on=["datatype", "model", "taxa", "sites"], how="inner")
agree = agree[agree["hardware"] != "cpu_OMP_104"].copy()
agree["delta_logL"] = agree["best_logL"] - agree["logL_ref"]
agree["rel_diff"] = (agree["best_logL"] - agree["logL_ref"]) / agree["logL_ref"].abs()
agree.to_csv(OUT / "logL_agreement.csv", index=False)

if not agree.empty:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for hw in [h for h in HW_ORDER if h in agree["hardware"].unique()]:
        s = agree[agree["hardware"] == hw]
        ax.scatter(s["sites"], s["delta_logL"], label=hw, **{
            k: v for k, v in HW_STYLE[hw].items() if k != "ls"})
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xscale("symlog")
    ax.set_xlabel("sites"); ax.set_ylabel("Δ logL  vs cpu_OMP_104")
    ax.set_title("Best log-likelihood agreement vs CPU OMP_104 reference")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_logL_agreement.png", dpi=140)
    plt.close(fig)

# ---------- text summary ----------
with (OUT / "SUMMARY.txt").open("w") as fh:
    fh.write("Simulated complex_data — 2026_06_24 snapshot analysis\n")
    fh.write("=" * 60 + "\n\n")
    fh.write(f"Source: {ROOT}\n")
    fh.write(f"Total logs parsed:    {len(df)}\n")
    for k, v in status_counts.items():
        fh.write(f"  {k:<14} {v}\n")
    fh.write("\n")
    fh.write("Hardware variants present:\n")
    for hw, c in df["hardware"].value_counts().items():
        ok_c = (df[(df["hardware"] == hw) & (df["status"] == "COMPLETE")]).shape[0]
        fh.write(f"  {hw:<22} runs={c:<3} complete={ok_c}\n")
    fh.write("\n")
    fh.write("Cells (datatype/model/taxa/sites) per completion count:\n")
    cc = ok.groupby(["datatype", "model", "taxa", "sites"]).size().reset_index(name="n_hw")
    for _, r in cc.iterrows():
        fh.write(f"  {r['datatype']}/{r['model']} taxa={r['taxa']} L={r['sites']:<10} "
                 f"complete_HWs={r['n_hw']}\n")
    fh.write("\n")
    if not agree.empty:
        worst = agree.reindex(agree["delta_logL"].abs().sort_values(ascending=False).index).head(5)
        fh.write("Top 5 worst logL deltas vs cpu_OMP_104 (sanity):\n")
        for _, r in worst.iterrows():
            fh.write(f"  {r['hardware']:<20} {r['datatype']}/{r['model']} "
                     f"taxa={r['taxa']} L={r['sites']:<10} "
                     f"Δ={r['delta_logL']:+.3f}  rel={r['rel_diff']:.2e}\n")
    fh.write("\nFigures written:\n")
    for p in sorted(OUT.glob("fig_*.png")):
        fh.write(f"  {p.name}\n")

print("Done. Wrote:")
for p in sorted(OUT.iterdir()):
    if p.is_file():
        print(" ", p.name)
