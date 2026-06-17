"""Parse the 2026_06_13_morebenchmarks IQ-TREE logs and produce runtime + energy plots.

Run matrix: AA (LG+I+G4) and DNA (GTR+I+G4) on 100 taxa / 1000 sites, ninit2,
across four backends (Intel SPR OMP104, Intel CLX OMP48, V100 OPENACC, H200 OPENACC)
and three to four test modes (LGC10 / MTEST / MIXGTR / UFBOOT1000).
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("/Users/u7826985/Projects/Nvidia/results/2026_06_13_morebenchmarks")
OUT_DIR = Path(__file__).resolve().parent

# ---- regexes ---------------------------------------------------------------
RE_TOTAL_CPU      = re.compile(r"Total CPU time used:\s+([\d.]+)\s+sec")
RE_TOTAL_WALL     = re.compile(r"Total wall-clock time used:\s+([\d.]+)\s+sec")
RE_TS_CPU         = re.compile(r"CPU time used for tree search:\s+([\d.]+)\s+sec")
RE_TS_WALL        = re.compile(r"Wall-clock time used for tree search:\s+([\d.]+)\s+sec")
RE_MF_CPU         = re.compile(r"CPU time for ModelFinder:\s+([\d.]+)\s+seconds")
RE_MF_WALL        = re.compile(r"Wall-clock time for ModelFinder:\s+([\d.]+)\s+seconds")
# Resumed-run cumulative variants (preferred when present)
RE_TOTAL_CPU_PR   = re.compile(r"Total CPU time used \(including previous runs\):\s+([\d.]+)\s+sec")
RE_TOTAL_WALL_PR  = re.compile(r"Total wall-clock time used \(including previous runs\):\s+([\d.]+)\s+sec")
RE_TS_CPU_PR      = re.compile(r"CPU time used for tree search \(including previous runs\):\s+([\d.]+)\s+sec")
RE_TS_WALL_PR     = re.compile(r"Wall-clock time used for tree search \(including previous runs\):\s+([\d.]+)\s+sec")
RE_MF_CPU_PR      = re.compile(r"CPU time for ModelFinder \(including previous runs\):\s+([\d.]+)\s+seconds")
RE_MF_WALL_PR     = re.compile(r"Wall-clock time for ModelFinder \(including previous runs\):\s+([\d.]+)\s+seconds")

def _first(rxs, txt):
    """Return value from the first regex in `rxs` that matches; else None."""
    for rx in rxs:
        v = _f(rx, txt)
        if v is not None:
            return v
    return None
RE_E_TS           = re.compile(r"Energy used for tree search:\s+CPU\s+([\d.]+)\s+J(?:,\s+GPU\s+([\d.]+)\s+J)?")
RE_E_MF           = re.compile(r"Energy used for ModelFinder:\s+CPU\s+([\d.]+)\s+J(?:,\s+GPU\s+([\d.]+)\s+J)?")
RE_BEST_LOGL      = re.compile(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)")
RE_E_BLOCK_CPU    = re.compile(r"Energy:\s*\n\s*CPU:\s+([\d.]+)\s+J", re.M)
RE_E_BLOCK_GPU    = re.compile(r"^\s*GPU:\s+([\d.]+)\s+J", re.M)
RE_E_BLOCK_GPU_NA = re.compile(r"^\s*GPU:\s+not available", re.M)
RE_GPU_NAME       = re.compile(r"\[([^=]+)=[\d.]+\s+J\]")
RE_GPU_MEM        = re.compile(r"GPU mem:\s+([\d.]+)\s*/\s*([\d.]+)\s+MB.*?\+([\d.]+)\s+MB")
RE_CPU_MEM        = re.compile(r"CPU mem:\s+([\d.]+)\s*/\s*([\d.]+)\s+MB")
# Host RAM declared in the very first lines (works even for killed runs)
RE_HOST_RAM       = re.compile(r"Host:.*?\(.*?(\d+)\s+GB\s+RAM\)")
# GPU advertised VRAM ("NVIDIA H200, 143426 MB")
RE_GPU_CAP_HEAD   = re.compile(r"GPU:\s+Device 0 \(OpenACC[^)]*?,\s*(\d+)\s*MB\)")
# IQ-TREE's pre-run memory estimate
RE_MF_REQ_RAM     = re.compile(r"NOTE:\s+ModelFinder requires\s+(\d+)\s+MB RAM", re.I)
# memory-saving flag from the Command: line
RE_MEM_FLAG       = re.compile(r"^Command:.*?\s(-mem|--mem|-lmsave)\s+(\S+)", re.M)
# auto-enabled LM_MEM_SAVE notices written by IQ-TREE at runtime
RE_LM_GPU = re.compile(
    r"Switching to GPU memory-saving mode \(LM_MEM_SAVE\) using ([\d.]+) GB \(([\d.]+)% of free VRAM\)", re.I)
RE_LM_CPU = re.compile(
    r"Switching to memory saving mode using ([\d.]+) GB \(([\d.]+)% of normal mode\)", re.I)

def _f(rx, txt, grp=1, default=None):
    m = rx.search(txt)
    if not m:
        return default
    try:
        return float(m.group(grp))
    except (IndexError, TypeError):
        return default

def classify_hardware(name: str) -> str:
    """Classify by token boundaries — never substring match (see memory)."""
    parts = name.split("_")
    if "OPENACC" in parts:
        if "H200" in parts:
            return "GPU_H200"
        if "V100" in parts:
            return "GPU_V100"
        return "GPU_unknown"
    if "OMP104" in parts:
        return "CPU_SPR_OMP104"
    if "CLX" in parts and "OMP48" in parts:
        return "CPU_CLX_OMP48"
    return "CPU_unknown"

def classify_test(name: str) -> str:
    parts = name.split("_")
    for t in ("LGC10", "MTEST", "UFBOOT1000", "MIXGTR"):
        if t in parts:
            return t
    return "unknown"

def classify_datatype(name: str) -> str:
    parts = name.split("_")
    if "AA" in parts:
        return "AA"
    if "DNA" in parts:
        return "DNA"
    return "unknown"

RE_LEN = re.compile(r"(?:^|_)(\d+)len(?:_|$)")
def classify_length(name: str) -> int | None:
    m = RE_LEN.search(name)
    return int(m.group(1)) if m else None

def parse_log(path: Path) -> dict:
    txt = path.read_text(errors="ignore")
    cpu_b   = _f(RE_E_BLOCK_CPU, txt)
    gpu_b   = _f(RE_E_BLOCK_GPU, txt)
    gpu_na  = bool(RE_E_BLOCK_GPU_NA.search(txt))
    e_ts    = RE_E_TS.search(txt)
    e_mf    = RE_E_MF.search(txt)
    gpu_mem = RE_GPU_MEM.search(txt)
    gpu_name_m = RE_GPU_NAME.search(txt)
    cpu_mem = RE_CPU_MEM.search(txt)
    name = path.stem
    return {
        "run":              name,
        "hardware":         classify_hardware(name),
        "test":             classify_test(name),
        "datatype":         classify_datatype(name),
        "sites":            classify_length(name),
        "complete":         cpu_b is not None,
        "wall_total_s":     _first([RE_TOTAL_WALL_PR, RE_TOTAL_WALL], txt),
        "cpu_total_s":      _first([RE_TOTAL_CPU_PR,  RE_TOTAL_CPU],  txt),
        "wall_ts_s":        _first([RE_TS_WALL_PR,    RE_TS_WALL],    txt),
        "cpu_ts_s":         _first([RE_TS_CPU_PR,     RE_TS_CPU],     txt),
        "wall_mf_explicit_s": _first([RE_MF_WALL_PR, RE_MF_WALL], txt),
        "cpu_mf_explicit_s":  _first([RE_MF_CPU_PR,  RE_MF_CPU],  txt),
        "is_resumed":       bool(RE_TOTAL_WALL_PR.search(txt)),
        "best_logL":        _f(RE_BEST_LOGL, txt),
        "energy_cpu_J":     cpu_b,
        "energy_gpu_J":     None if gpu_na else gpu_b,
        "energy_total_J":   (cpu_b or 0) + (0 if gpu_na or gpu_b is None else gpu_b) if cpu_b else None,
        "e_ts_cpu_J":       float(e_ts.group(1)) if e_ts else None,
        "e_ts_gpu_J":       float(e_ts.group(2)) if e_ts and e_ts.group(2) else None,
        "e_mf_cpu_J":       float(e_mf.group(1)) if e_mf else None,
        "e_mf_gpu_J":       float(e_mf.group(2)) if e_mf and e_mf.group(2) else None,
        "gpu_name":         gpu_name_m.group(1).strip() if gpu_name_m else None,
        "gpu_mem_peak_MB":  float(gpu_mem.group(1)) if gpu_mem else None,
        "gpu_mem_cap_MB":   float(gpu_mem.group(2)) if gpu_mem
                            else (float(RE_GPU_CAP_HEAD.search(txt).group(1))
                                  if RE_GPU_CAP_HEAD.search(txt) else None),
        "gpu_mem_delta_MB": float(gpu_mem.group(3)) if gpu_mem else None,
        "cpu_mem_peak_MB":  float(cpu_mem.group(1)) if cpu_mem else None,
        "cpu_mem_cap_MB":   float(cpu_mem.group(2)) if cpu_mem
                            else (float(RE_HOST_RAM.search(txt).group(1)) * 1024
                                  if RE_HOST_RAM.search(txt) else None),
        "mf_required_MB":   _f(RE_MF_REQ_RAM, txt),
        "mem_flag":         (lambda m: f"{m.group(1)} {m.group(2)}" if m else None)(
                                RE_MEM_FLAG.search(txt)),
        # auto-LM_MEM_SAVE (last occurrence in case multiple, e.g. resumed runs)
        "lm_save_gpu_GB":   (lambda hits: float(hits[-1][0]) if hits else None)(
                                RE_LM_GPU.findall(txt)),
        "lm_save_gpu_pct":  (lambda hits: float(hits[-1][1]) if hits else None)(
                                RE_LM_GPU.findall(txt)),
        "lm_save_cpu_GB":   (lambda hits: float(hits[-1][0]) if hits else None)(
                                RE_LM_CPU.findall(txt)),
        "lm_save_cpu_pct":  (lambda hits: float(hits[-1][1]) if hits else None)(
                                RE_LM_CPU.findall(txt)),
    }

def collect() -> pd.DataFrame:
    rows = [parse_log(p) for p in sorted(RESULTS_DIR.glob("*.log"))]
    df = pd.DataFrame(rows)
    df["avg_power_W"] = df["energy_total_J"] / df["wall_total_s"]
    df["cpu_share"]   = df["energy_cpu_J"] / df["energy_total_J"]
    # ModelFinder wall: prefer the explicit log line, fall back to (total - tree_search)
    df["wall_mf_s"] = df["wall_mf_explicit_s"].fillna(df["wall_total_s"] - df["wall_ts_s"])
    df.loc[df["wall_mf_s"] < 0, "wall_mf_s"] = np.nan
    return df

# ---- plotting helpers ------------------------------------------------------
HW_ORDER = ["CPU_CLX_OMP48", "CPU_SPR_OMP104", "GPU_V100", "GPU_H200"]
HW_LABEL = {
    "CPU_CLX_OMP48":  "CPU CLX (48t)",
    "CPU_SPR_OMP104": "CPU SPR (104t)",
    "GPU_V100":       "GPU V100",
    "GPU_H200":       "GPU H200",
}
HW_COLOR = {
    "CPU_CLX_OMP48":  "#7f8c8d",
    "CPU_SPR_OMP104": "#2c3e50",
    "GPU_V100":       "#27ae60",
    "GPU_H200":       "#76b900",  # nvidia green
}
TEST_ORDER = ["LGC10", "MTEST", "MIXGTR", "UFBOOT1000"]

def grouped_bar(df, value, ylabel, title, fname, log=False, fmt="{:.0f}"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=log)
    for ax, dt in zip(axes, ("AA", "DNA")):
        sub = df[df.datatype == dt]
        tests = [t for t in TEST_ORDER if t in sub.test.unique()]
        hws   = [h for h in HW_ORDER if h in sub.hardware.unique()]
        x = np.arange(len(tests))
        w = 0.8 / max(len(hws), 1)
        for i, hw in enumerate(hws):
            vals = []
            for t in tests:
                row = sub[(sub.hardware == hw) & (sub.test == t)]
                vals.append(row[value].iloc[0] if len(row) and pd.notna(row[value].iloc[0]) else np.nan)
            offs = (i - (len(hws) - 1) / 2) * w
            bars = ax.bar(x + offs, vals, w, label=HW_LABEL[hw], color=HW_COLOR[hw], edgecolor="black", linewidth=0.4)
            for b, v in zip(bars, vals):
                if pd.notna(v):
                    ax.text(b.get_x() + b.get_width()/2, v, fmt.format(v),
                            ha="center", va="bottom", fontsize=7, rotation=0)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(tests)
        ax.set_title(f"{dt}  ({'LG+I+G4' if dt=='AA' else 'GTR+I+G4'})")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=160)
    plt.close(fig)

def stacked_energy(df, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, dt in zip(axes, ("AA", "DNA")):
        sub = df[df.datatype == dt]
        tests = [t for t in TEST_ORDER if t in sub.test.unique()]
        hws   = [h for h in HW_ORDER if h in sub.hardware.unique()]
        x = np.arange(len(tests))
        w = 0.8 / max(len(hws), 1)
        for i, hw in enumerate(hws):
            cpu_vals, gpu_vals = [], []
            for t in tests:
                row = sub[(sub.hardware == hw) & (sub.test == t)]
                if len(row):
                    cpu_vals.append(row.energy_cpu_J.iloc[0] or 0)
                    gpu_vals.append(row.energy_gpu_J.iloc[0] or 0)
                else:
                    cpu_vals.append(np.nan); gpu_vals.append(np.nan)
            offs = (i - (len(hws) - 1) / 2) * w
            ax.bar(x + offs, cpu_vals, w, color=HW_COLOR[hw], edgecolor="black", linewidth=0.4,
                   label=f"{HW_LABEL[hw]} CPU" if i == 0 else None)
            ax.bar(x + offs, gpu_vals, w, bottom=cpu_vals, color=HW_COLOR[hw], alpha=0.45,
                   edgecolor="black", linewidth=0.4, hatch="//",
                   label=f"{HW_LABEL[hw]} GPU" if i == 0 else None)
            for j, (c, g) in enumerate(zip(cpu_vals, gpu_vals)):
                if pd.notna(c):
                    tot = c + (g if pd.notna(g) else 0)
                    ax.text(x[j] + offs, tot, HW_LABEL[hw].split()[-1],
                            ha="center", va="bottom", fontsize=6, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels(tests)
        ax.set_title(f"{dt}  ({'LG+I+G4' if dt=='AA' else 'GTR+I+G4'})")
        ax.set_ylabel("Total energy (J)  [solid=CPU, hatched=GPU]")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Energy breakdown (CPU stacked with GPU)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=160)
    plt.close(fig)

def speedup_table(df) -> pd.DataFrame:
    """Speedup and energy ratio normalised to CPU_CLX_OMP48 within (datatype, test, sites)."""
    out = []
    base = (df[df.hardware == "CPU_CLX_OMP48"]
              .set_index(["datatype", "test", "sites"]))
    for _, row in df.iterrows():
        key = (row.datatype, row.test, row.sites)
        if key in base.index:
            b = base.loc[key]
            out.append({
                "run":             row.run,
                "hardware":        row.hardware,
                "test":            row.test,
                "datatype":        row.datatype,
                "sites":           row.sites,
                "wall_s":          row.wall_total_s,
                "speedup_vs_CLX":  (b.wall_total_s / row.wall_total_s) if row.wall_total_s else np.nan,
                "energy_J":        row.energy_total_J,
                "energy_ratio_vs_CLX": (row.energy_total_J / b.energy_total_J)
                                         if row.energy_total_J and b.energy_total_J else np.nan,
                "best_logL":       row.best_logL,
            })
    return pd.DataFrame(out)

def logL_agreement(df, fname):
    """Per (datatype, test, sites), check best logL agreement across hardware."""
    rows = []
    for (dt, t, s), g in df.groupby(["datatype", "test", "sites"]):
        vals = g.dropna(subset=["best_logL"])
        if len(vals) < 2:
            continue
        for _, r in vals.iterrows():
            rows.append({"datatype": dt, "test": t, "sites": int(s), "hardware": r.hardware,
                         "logL": r.best_logL,
                         "delta": r.best_logL - vals.best_logL.max(),
                         "range": vals.best_logL.max() - vals.best_logL.min()})
    agr = pd.DataFrame(rows)
    if agr.empty:
        return agr
    fig, ax = plt.subplots(figsize=(max(11, 0.5 * agr[["datatype","test","sites"]].drop_duplicates().shape[0]), 5))
    labels = []
    for i, (key, g) in enumerate(agr.groupby(["datatype", "test", "sites"], sort=False)):
        labels.append(f"{key[0]}/{key[1]}/{key[2]}")
        for _, r in g.iterrows():
            ax.scatter(i, r.delta, color=HW_COLOR.get(r.hardware, "k"),
                       s=70, edgecolor="black", linewidth=0.4,
                       label=HW_LABEL.get(r.hardware) if i == 0 else None)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Δ log-likelihood from best (within row)")
    ax.set_title("Log-likelihood agreement across hardware (per datatype/test/sites)")
    ax.grid(axis="y", alpha=0.3)
    handles, lbls = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, lbls):
        seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=160)
    plt.close(fig)
    return agr

def _fmt_min(seconds):
    if seconds is None or not np.isfinite(seconds):
        return ""
    m = seconds / 60.0
    if m < 60:
        return f"{m:.0f}m" if m >= 1 else f"{m:.1f}m"
    h = int(m // 60)
    return f"{h}h{int(round(m - 60*h))}m"

# Reference-style hardware ordering / labels (A100 omitted — no A100 data in this run set)
LINEAR_HW_ORDER  = ["CPU_CLX_OMP48", "CPU_SPR_OMP104", "GPU_V100", "GPU_H200"]
LINEAR_HW_LABEL  = {
    "CPU_CLX_OMP48":  "CLX 48T",
    "CPU_SPR_OMP104": "SPR 103T",
    "GPU_V100":       "GPU V100",
    "GPU_H200":       "GPU H200",
}
LINEAR_HW_COLOR  = {
    "CPU_CLX_OMP48":  "#1f77b4",   # blue
    "CPU_SPR_OMP104": "#e6611c",   # orange
    "GPU_V100":       "#a8d8a0",   # light green
    "GPU_H200":       "#1f7a3a",   # dark green
}

def _fmt_wh(joules):
    if joules is None or not np.isfinite(joules):
        return ""
    wh = joules / 3600.0
    if wh >= 1000:
        return f"{wh/1000:.1f}kWh"
    if wh >= 10:
        return f"{wh:.0f}Wh"
    return f"{wh:.1f}Wh"

def stacked_energy_panels(df, test_name, cpu_col, gpu_col, fname,
                          stage_label="Total", ninit=2):
    """Stacked CPU+GPU energy bars (Wh) per (datatype, sites) panel — one figure per test×stage."""
    sub = df[df.test == test_name].copy()
    cells = []
    for dt in ("AA", "DNA"):
        for s in sorted(sub[sub.datatype == dt]["sites"].dropna().unique()):
            if s < 100_000:
                continue
            slice_ = sub[(sub.datatype == dt) & (sub.sites == s)]
            if slice_[cpu_col].notna().any():
                cells.append((dt, int(s)))
    if not cells:
        print(f"[stacked_energy] no data for test={test_name} cpu_col={cpu_col}, skipping")
        return
    ncols = len(cells)
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 5.2), squeeze=False)
    axes = axes[0]
    x = np.arange(len(LINEAR_HW_ORDER))
    CPU_BLUE  = "#1f77b4"
    GPU_GREEN = "#7fc97f"
    for ax, (dt, s) in zip(axes, cells):
        slice_ = sub[(sub.datatype == dt) & (sub.sites == s)]
        cpu_J, gpu_J, lm_flags = [], [], []
        for hw in LINEAR_HW_ORDER:
            row = slice_[slice_.hardware == hw]
            if len(row):
                cpu_J.append(row[cpu_col].iloc[0] if pd.notna(row[cpu_col].iloc[0]) else np.nan)
                gpu_J.append(row[gpu_col].iloc[0] if pd.notna(row[gpu_col].iloc[0]) else 0.0)
                rr = row.iloc[0]
                lm_flags.append((hw.startswith("GPU") and pd.notna(rr.get("lm_save_gpu_GB")))
                                or (hw.startswith("CPU") and pd.notna(rr.get("lm_save_cpu_GB"))))
            else:
                cpu_J.append(np.nan); gpu_J.append(np.nan); lm_flags.append(False)
        cpu_wh = [c/3600 if pd.notna(c) else np.nan for c in cpu_J]
        gpu_wh = [g/3600 if pd.notna(g) else np.nan for g in gpu_J]
        hatches = ["////" if lm else None for lm in lm_flags]
        cpu_bars = ax.bar(x, cpu_wh, color=CPU_BLUE, edgecolor="black", linewidth=0.4, width=0.7,
                          hatch=hatches, label="CPU (host)")
        gpu_bars = ax.bar(x, gpu_wh, bottom=cpu_wh, color=GPU_GREEN, edgecolor="black", linewidth=0.4,
                          width=0.7, hatch=hatches, label="GPU (accelerator)")
        for i, (c, g, lm) in enumerate(zip(cpu_J, gpu_J, lm_flags)):
            if pd.notna(c):
                total = (c if pd.notna(c) else 0) + (g if pd.notna(g) else 0)
                top = total/3600
                label = _fmt_wh(total) + (" ★" if lm else "")
                ax.text(x[i], top, label, ha="center", va="bottom",
                        fontsize=8, rotation=90)
        title_sites = f"{s//1000}K" if s < 1_000_000 else f"{s//1_000_000}M"
        ax.set_title(f"{dt} — {title_sites}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([LINEAR_HW_LABEL[h] for h in LINEAR_HW_ORDER],
                           rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("energy (Wh)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        tops = [(c or 0)+(g or 0) for c, g in zip(cpu_J, gpu_J) if pd.notna(c)]
        if tops:
            ax.set_ylim(0, max(tops)/3600 * 1.25)
    handles = [plt.Rectangle((0,0),1,1, color=CPU_BLUE),
               plt.Rectangle((0,0),1,1, color=GPU_GREEN)]
    fig.legend(handles, ["CPU (host)", "GPU (accelerator)"],
               loc="lower center", ncol=2, frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Energy breakdown (Wh) — CPU host vs GPU stack — {stage_label} stage — {test_name}, ninit = {ninit}",
                 fontsize=12, fontweight="bold")
    footnote = ("Each panel has its own linear y-axis scale.  "
                "CLX = Cascade Lake Xeon 8274 (NCI `normal`, 48c).  "
                "SPR = Sapphire Rapids 8480+ (NCI `normalsr`, 104c, -nt 103).  "
                "GPU rows use the OpenACC build, host `-nt 1`.  "
                "Hatched (////) + ★ = LM_MEM_SAVE auto-engaged → ~1.5–3× slower per likelihood eval.  "
                "Missing bar = run didn't complete (no `Energy:` block) or wasn't submitted.")
    fig.text(0.5, -0.05, footnote, ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUT_DIR / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[stacked_energy] wrote {fname} ({ncols} panels, {cpu_col} + {gpu_col})")

# Memory capacities (GB) by backend — host RAM for CPU rows, VRAM for GPU rows
MEM_CAPACITY_GB = {
    "CPU_CLX_OMP48":  188,
    "CPU_SPR_OMP104": 503,
    "GPU_V100":        32,
    "GPU_H200":       140,
}
MEM_BAR_ORDER = ["CPU_CLX_OMP48", "CPU_SPR_OMP104", "GPU_V100", "GPU_H200"]
MEM_BAR_LABEL = {"CPU_CLX_OMP48": "CLX", "CPU_SPR_OMP104": "SPR",
                 "GPU_V100": "V100", "GPU_H200": "H200"}
COLOR_CLEAN   = "#2ca02c"  # green
COLOR_MOD     = "#f5c518"  # yellow
COLOR_TIGHT   = "#f37020"  # orange
COLOR_FAIL    = "#c1272d"  # red
COLOR_EMPTY   = "#e6e6e6"  # light grey (background)

def _mem_classify(frac):
    if frac is None or not np.isfinite(frac):
        return None
    if frac < 0.50: return ("clean",    COLOR_CLEAN, "✓")
    if frac < 0.90: return ("moderate", COLOR_MOD,   "✓")
    return            ("tight",    COLOR_TIGHT, "△")

def memory_grid(df, test_name, fname):
    """2×3 grid: AA on top row, DNA on bottom row; columns are 100K / 1M / 10M sites.
    Each panel has one horizontal bar per backend showing capacity vs peak-used."""
    sub = df[df.test == test_name].copy()
    sites_present = sorted(sub["sites"].dropna().unique())
    # filter to {100K, 1M, 10M} columns that have any data in this test
    cols = [s for s in (100_000, 1_000_000, 10_000_000) if s in sites_present]
    if not cols:
        print(f"[memory_grid] no data for test={test_name}, skipping")
        return
    dts = [dt for dt in ("AA", "DNA") if not sub[sub.datatype == dt].empty]
    nrows, ncols = len(dts), len(cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.0 * nrows),
                             squeeze=False)
    fig.suptitle(f"Memory headroom — {test_name}\n"
                 "each bar = one backend; full bar = capacity (host RAM for CPU, VRAM for GPU); "
                 "fill = peak used",
                 fontsize=11)
    bar_h = 0.65
    for r, dt in enumerate(dts):
        for c, s in enumerate(cols):
            ax = axes[r][c]
            slice_ = sub[(sub.datatype == dt) & (sub.sites == s)]
            for i, hw in enumerate(MEM_BAR_ORDER):
                y = len(MEM_BAR_ORDER) - 1 - i  # CLX at top
                row = slice_[slice_.hardware == hw]
                # default capacity if log missing
                cap_gb = MEM_CAPACITY_GB[hw]
                if len(row):
                    rr = row.iloc[0]
                    cap_col = "cpu_mem_cap_MB" if hw.startswith("CPU") else "gpu_mem_cap_MB"
                    cap_mb = rr.get(cap_col)
                    if pd.notna(cap_mb) and cap_mb:
                        cap_gb = cap_mb / 1024.0
                ax.barh(y, cap_gb, height=bar_h, color=COLOR_EMPTY,
                        edgecolor="black", linewidth=0.5)
                if not len(row):
                    ax.text(cap_gb*1.02, y, " not submitted", va="center", ha="left",
                            fontsize=8, color="#888", style="italic")
                    continue
                if hw.startswith("CPU"):
                    used_mb = rr.get("cpu_mem_peak_MB")
                else:
                    used_mb = rr.get("gpu_mem_peak_MB")
                if pd.isna(used_mb) or used_mb is None:
                    if rr["complete"]:
                        msg = "no peak recorded"
                    elif test_name == "LGC10":
                        msg = "killed in tree-search"   # LGC10 has no ModelFinder phase
                    elif pd.isna(rr["wall_mf_explicit_s"]):
                        msg = "killed in ModelFinder"
                    else:
                        msg = "killed in tree-search"
                    ax.text(cap_gb*1.02, y, f" {msg}", va="center", ha="left",
                            fontsize=8, color="#888", style="italic")
                    continue
                used_gb = used_mb / 1024.0
                frac    = used_gb / cap_gb
                cls = _mem_classify(frac)
                # If LM_MEM_SAVE auto-engaged, force the bar to the "tight" orange
                # so it's visually distinct from non-LM_SAVE bars of the same fill %.
                lm_on = (hw.startswith("GPU") and pd.notna(rr.get("lm_save_gpu_GB"))) or \
                        (hw.startswith("CPU") and pd.notna(rr.get("lm_save_cpu_GB")))
                color = COLOR_TIGHT if lm_on else cls[1]
                # fill rectangle
                ax.barh(y, min(used_gb, cap_gb), height=bar_h, color=color,
                        edgecolor="black", linewidth=0.5)
                # marker on fill
                ax.text(min(used_gb, cap_gb) - cap_gb*0.012, y, cls[2],
                        va="center", ha="right", fontsize=11, color="white",
                        fontweight="bold")
                lm_tag = ""
                if hw.startswith("GPU") and pd.notna(rr.get("lm_save_gpu_GB")):
                    lm_tag = f"  · LM_SAVE ({rr['lm_save_gpu_pct']:.0f}% VRAM)"
                elif hw.startswith("CPU") and pd.notna(rr.get("lm_save_cpu_GB")):
                    lm_tag = f"  · LM_SAVE ({rr['lm_save_cpu_pct']:.0f}% normal)"
                ax.text(cap_gb*1.02, y,
                        f" {used_gb:.0f}/{cap_gb:.0f} GB  ({frac*100:.0f}%){lm_tag}",
                        va="center", ha="left", fontsize=8)
            ax.set_yticks(range(len(MEM_BAR_ORDER)))
            ax.set_yticklabels([MEM_BAR_LABEL[h] for h in reversed(MEM_BAR_ORDER)],
                               fontsize=9, fontweight="bold")
            # x-axis scaled to the largest capacity actually plotted in this panel
            caps_here = []
            for hw in MEM_BAR_ORDER:
                rr = slice_[slice_.hardware == hw]
                cap_col = "cpu_mem_cap_MB" if hw.startswith("CPU") else "gpu_mem_cap_MB"
                if len(rr) and pd.notna(rr[cap_col].iloc[0]):
                    caps_here.append(rr[cap_col].iloc[0] / 1024.0)
                else:
                    caps_here.append(MEM_CAPACITY_GB[hw])
            ax.set_xlim(0, max(caps_here) * 1.45)
            ax.set_xlabel("memory (GB)")
            ax.set_title(f"{dt} · {s//1000}K sites" if s < 1_000_000
                         else f"{dt} · {s//1_000_000}M sites",
                         fontsize=11, fontweight="bold")
            ax.grid(axis="x", alpha=0.25, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    legend_items = [
        (COLOR_CLEAN, "✓ clean (<50%)"),
        (COLOR_MOD,   "moderate (50–90%)"),
        (COLOR_TIGHT, "△ tight (≥90%)"),
        (COLOR_EMPTY, "not submitted / no peak recorded"),
    ]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.4) for c, _ in legend_items]
    fig.legend(handles, [l for _, l in legend_items],
               loc="lower center", ncol=len(legend_items), frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    foot = ("capacities — CLX 188 GB · SPR 503 GB · V100 32 GB VRAM · H200 140 GB VRAM. "
            "CPU rows: peak resident host RAM from IQ-TREE `Energy:` block. "
            "GPU rows: peak VRAM. Empty bars = run was either not submitted or killed before "
            "the `Energy:` block was written.")
    fig.text(0.5, -0.06, foot, ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(OUT_DIR / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[memory_grid] wrote {fname} ({nrows}×{ncols} panels)")

def total_energy_byStage_panels(df, test_name, fname, ninit=2):
    """Per (datatype, sites) panel: each HW bar = total energy stacked by stage
    (ModelFinder bottom + tree-search middle + residual top)."""
    sub = df[df.test == test_name].copy()
    sub["e_mf_total_J"] = sub["e_mf_cpu_J"].fillna(0) + sub["e_mf_gpu_J"].fillna(0)
    sub["e_ts_total_J"] = sub["e_ts_cpu_J"].fillna(0) + sub["e_ts_gpu_J"].fillna(0)
    sub["e_residual_J"] = (sub["energy_total_J"].fillna(0)
                           - sub["e_mf_total_J"] - sub["e_ts_total_J"]).clip(lower=0)

    cells = []
    for dt in ("AA", "DNA"):
        for s in sorted(sub[sub.datatype == dt]["sites"].dropna().unique()):
            if s < 100_000:
                continue
            slice_ = sub[(sub.datatype == dt) & (sub.sites == s)]
            # plot the cell if any HW has either MF energy or total energy
            if (slice_["e_mf_total_J"] > 0).any() or (slice_["energy_total_J"] > 0).any():
                cells.append((dt, int(s)))
    if not cells:
        print(f"[total_energy_byStage] no data for test={test_name}, skipping")
        return

    MF_COLOR = "#f4a261"   # orange  — ModelFinder
    TS_COLOR = "#1d8a3a"   # green   — tree search
    RS_COLOR = "#bdbdbd"   # grey    — residual (init + finalization + bootstrap)
    ncols = len(cells)
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 5.2), squeeze=False)
    axes = axes[0]
    x = np.arange(len(LINEAR_HW_ORDER))
    for ax, (dt, s) in zip(axes, cells):
        slice_ = sub[(sub.datatype == dt) & (sub.sites == s)]
        mf, ts, rs, tot, lm_flags = [], [], [], [], []
        for hw in LINEAR_HW_ORDER:
            row = slice_[slice_.hardware == hw]
            if len(row):
                mf_J = row["e_mf_total_J"].iloc[0]
                ts_J = row["e_ts_total_J"].iloc[0]
                rs_J = row["e_residual_J"].iloc[0]
                tot_J = row["energy_total_J"].iloc[0]
                mf.append(mf_J if pd.notna(mf_J) else 0)
                ts.append(ts_J if pd.notna(ts_J) else 0)
                rs.append(rs_J if pd.notna(rs_J) else 0)
                tot.append(tot_J if pd.notna(tot_J) and tot_J > 0
                           else (mf_J or 0) + (ts_J or 0))
                rr = row.iloc[0]
                lm_flags.append((hw.startswith("GPU") and pd.notna(rr.get("lm_save_gpu_GB")))
                                or (hw.startswith("CPU") and pd.notna(rr.get("lm_save_cpu_GB"))))
            else:
                mf.append(0); ts.append(0); rs.append(0); tot.append(np.nan); lm_flags.append(False)
        mf_wh = [v/3600 for v in mf]
        ts_wh = [v/3600 for v in ts]
        rs_wh = [v/3600 for v in rs]
        hatches = ["////" if lm else None for lm in lm_flags]
        ax.bar(x, mf_wh, color=MF_COLOR, edgecolor="black", linewidth=0.4, width=0.7,
               hatch=hatches, label="ModelFinder")
        ax.bar(x, ts_wh, bottom=mf_wh, color=TS_COLOR, edgecolor="black", linewidth=0.4,
               width=0.7, hatch=hatches, label="Tree search")
        bot2 = [m + t for m, t in zip(mf_wh, ts_wh)]
        ax.bar(x, rs_wh, bottom=bot2, color=RS_COLOR, edgecolor="black", linewidth=0.4,
               width=0.7, hatch=hatches, label="Init + finalization")
        for i, (top_J, lm) in enumerate(zip(tot, lm_flags)):
            if pd.notna(top_J) and top_J > 0:
                top_wh = top_J / 3600
                label = _fmt_wh(top_J) + (" ★" if lm else "")
                ax.text(x[i], top_wh, label, ha="center", va="bottom", fontsize=8, rotation=90)
        title_sites = f"{s//1000}K" if s < 1_000_000 else f"{s//1_000_000}M"
        ax.set_title(f"{dt} — {title_sites}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([LINEAR_HW_LABEL[h] for h in LINEAR_HW_ORDER],
                           rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("energy (Wh)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        finite_tops = [v/3600 for v in tot if pd.notna(v) and v > 0]
        if finite_tops:
            ax.set_ylim(0, max(finite_tops) * 1.25)
    handles = [plt.Rectangle((0,0),1,1, color=MF_COLOR),
               plt.Rectangle((0,0),1,1, color=TS_COLOR),
               plt.Rectangle((0,0),1,1, color=RS_COLOR)]
    fig.legend(handles, ["ModelFinder", "Tree search", "Init + finalization"],
               loc="lower center", ncol=3, frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Total energy (Wh) — stage stack (MF + tree-search + finalization) — {test_name}, ninit = {ninit}",
                 fontsize=12, fontweight="bold")
    footnote = ("Each panel has its own linear y-axis scale.  "
                "CLX = Cascade Lake Xeon 8274 (NCI `normal`, 48c).  "
                "SPR = Sapphire Rapids 8480+ (NCI `normalsr`, 104c, -nt 103).  "
                "GPU rows use the OpenACC build, host `-nt 1`.  "
                "Top-label = whole-run total from the `Energy:` block (or MF+TS when run was killed).  "
                "Missing bar = run wasn't submitted.")
    fig.text(0.5, -0.05, footnote, ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUT_DIR / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[total_energy_byStage] wrote {fname} ({ncols} panels)")

def linear_panels(df, test_name, fname, value_col="wall_total_s",
                  stage_label="Total wall-clock", ninit=2):
    """Reference-style per-cell linear wall-clock plot for one (test, stage)."""
    sub = df[df.test == test_name].copy()
    cells = []
    for dt in ("AA", "DNA"):
        for s in sorted(sub[sub.datatype == dt]["sites"].dropna().unique()):
            if s < 100_000:   # skip 1k slice (matches reference family)
                continue
            slice_ = sub[(sub.datatype == dt) & (sub.sites == s)]
            if slice_[value_col].notna().any():
                cells.append((dt, int(s)))
    if not cells:
        print(f"[linear_panels] no data for test={test_name} value={value_col}, skipping")
        return
    ncols = len(cells)
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 5.2), squeeze=False)
    axes = axes[0]
    x = np.arange(len(LINEAR_HW_ORDER))
    for ax, (dt, s) in zip(axes, cells):
        slice_ = sub[(sub.datatype == dt) & (sub.sites == s)]
        vals, lm_flags = [], []
        for hw in LINEAR_HW_ORDER:
            row = slice_[slice_.hardware == hw]
            v = row[value_col].iloc[0] if len(row) else np.nan
            vals.append(v if pd.notna(v) else np.nan)
            if len(row):
                rr = row.iloc[0]
                lm_flags.append((hw.startswith("GPU") and pd.notna(rr.get("lm_save_gpu_GB")))
                                or (hw.startswith("CPU") and pd.notna(rr.get("lm_save_cpu_GB"))))
            else:
                lm_flags.append(False)
        mins = [v / 60 if pd.notna(v) else np.nan for v in vals]
        colors = [LINEAR_HW_COLOR[h] for h in LINEAR_HW_ORDER]
        bars = ax.bar(x, mins, color=colors, edgecolor="black", linewidth=0.4, width=0.7,
                      hatch=["////" if lm else None for lm in lm_flags])
        for b, raw, lm in zip(bars, vals, lm_flags):
            if pd.notna(raw):
                label = _fmt_min(raw) + (" ★" if lm else "")
                ax.text(b.get_x() + b.get_width()/2, b.get_height(), label,
                        ha="center", va="bottom", fontsize=8, rotation=90)
        title_sites = f"{s//1000}K" if s < 1_000_000 else f"{s//1_000_000}M"
        ax.set_title(f"{dt} — {title_sites}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([LINEAR_HW_LABEL[h] for h in LINEAR_HW_ORDER],
                           rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("wall time (min)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        finite = [m for m in mins if pd.notna(m)]
        if finite:
            ax.set_ylim(0, max(finite) * 1.20)
    handles = [plt.Rectangle((0, 0), 1, 1, color=LINEAR_HW_COLOR[h]) for h in LINEAR_HW_ORDER]
    fig.legend(handles, [LINEAR_HW_LABEL[h] for h in LINEAR_HW_ORDER],
               loc="lower center", ncol=len(LINEAR_HW_ORDER), frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{stage_label} (min) — per-cell linear scales — {test_name}, ninit = {ninit}",
                 fontsize=12, fontweight="bold")
    footnote = ("Each panel has its own linear y-axis scale.  "
                "CLX = Cascade Lake Xeon 8274 (NCI `normal`, 48c).  "
                "SPR = Sapphire Rapids 8480+ (NCI `normalsr`, 104c, -nt 103).  "
                "GPU rows use the OpenACC build, host `-nt 1`.  "
                "Hatched (////) + ★ = LM_MEM_SAVE auto-engaged → timing is 1.5–3× slower per likelihood eval.  "
                "Missing bar = run didn't complete (no `Energy:` block) or wasn't submitted.")
    fig.text(0.5, -0.05, footnote, ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUT_DIR / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[linear_panels] wrote {fname} ({ncols} panels, value={value_col})")

def scaling_plot(df, value, ylabel, title, fname, log_y=True):
    """One panel per (datatype, test); x = sites (log), y = value, line per hardware."""
    tests_by_dt = {dt: [t for t in TEST_ORDER if t in df[df.datatype == dt].test.unique()]
                   for dt in ("AA", "DNA")}
    rows = 2
    cols = max(len(tests_by_dt["AA"]), len(tests_by_dt["DNA"]))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows),
                             sharex=True, sharey=log_y, squeeze=False)
    for r, dt in enumerate(("AA", "DNA")):
        tests = tests_by_dt[dt]
        for c in range(cols):
            ax = axes[r][c]
            if c >= len(tests):
                ax.axis("off"); continue
            t = tests[c]
            sub = df[(df.datatype == dt) & (df.test == t)]
            for hw in HW_ORDER:
                s = sub[sub.hardware == hw].dropna(subset=["sites", value]).sort_values("sites")
                if not len(s):
                    continue
                ax.plot(s["sites"], s[value], "-o", label=HW_LABEL[hw],
                        color=HW_COLOR[hw], linewidth=1.6, markersize=5)
            ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")
            ax.set_title(f"{dt} / {t}")
            ax.set_xlabel("sites")
            ax.grid(True, which="both", alpha=0.3)
            if c == 0:
                ax.set_ylabel(ylabel)
    axes[0][-1].legend(loc="best", fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=160)
    plt.close(fig)

def main():
    df = collect()
    df.to_csv(OUT_DIR / "runs.csv", index=False)
    print(f"parsed {len(df)} runs, complete={df.complete.sum()}")
    print(f"lengths present: {sorted(df['sites'].dropna().unique().astype(int).tolist())}")
    sp = speedup_table(df)
    sp.to_csv(OUT_DIR / "speedup_vs_CLX.csv", index=False)

    # reference-style per-cell linear-scale wall-clock panels (one figure per test × stage)
    for test in ("MTEST", "UFBOOT1000", "LGC10"):
        linear_panels(df, test, f"fig_wall_total_linear_{test}.png",
                      value_col="wall_total_s", stage_label="Total wall-clock time")
        linear_panels(df, test, f"fig_wall_treesearch_linear_{test}.png",
                      value_col="wall_ts_s", stage_label="Tree-search wall-clock time")
        if test != "LGC10":   # LGC10 doesn't run ModelFinder
            linear_panels(df, test, f"fig_wall_modelfinder_linear_{test}.png",
                          value_col="wall_mf_s",
                          stage_label="ModelFinder wall-clock time")

    # stacked CPU+GPU energy per (test × stage)
    for test in ("MTEST", "UFBOOT1000", "LGC10"):
        stacked_energy_panels(df, test, "energy_cpu_J", "energy_gpu_J",
                              f"fig_energy_stacked_total_{test}.png",
                              stage_label="Total")
        stacked_energy_panels(df, test, "e_ts_cpu_J", "e_ts_gpu_J",
                              f"fig_energy_stacked_treesearch_{test}.png",
                              stage_label="Tree-search")
        if test != "LGC10":
            stacked_energy_panels(df, test, "e_mf_cpu_J", "e_mf_gpu_J",
                                  f"fig_energy_stacked_modelfinder_{test}.png",
                                  stage_label="ModelFinder")
        # total energy stacked by stage (MF + tree-search + finalization)
        total_energy_byStage_panels(df, test, f"fig_energy_total_byStage_{test}.png")
        # memory headroom grid
        memory_grid(df, test, f"fig_memory_grid_{test}.png")

    # length sweep (1k / 100k / 1M / 10M)
    scaling_plot(df, "wall_total_s",
                 "Wall-clock total (s, log)",
                 "Wall-clock scaling vs alignment length",
                 "fig_scaling_wall_total.png", log_y=True)
    scaling_plot(df, "wall_ts_s",
                 "Tree-search wall (s, log)",
                 "Tree-search wall scaling vs alignment length",
                 "fig_scaling_wall_treesearch.png", log_y=True)
    scaling_plot(df, "energy_total_J",
                 "Total energy (J, log)",
                 "Energy scaling vs alignment length",
                 "fig_scaling_energy_total.png", log_y=True)
    scaling_plot(df, "avg_power_W",
                 "Avg power (W)",
                 "Avg power vs alignment length",
                 "fig_scaling_avg_power.png", log_y=False)
    scaling_plot(df, "cpu_share",
                 "CPU energy share",
                 "CPU energy share vs alignment length",
                 "fig_scaling_cpu_share.png", log_y=False)

    # legacy single-length grouped bars (now only meaningful for the 1000-site slice;
    # restrict so labels remain readable)
    base_1k = df[df["sites"] == 1000]
    grouped_bar(base_1k, "wall_total_s",
                ylabel="Wall-clock (s, log)",
                title="1k sites: total wall-clock",
                fname="fig_1k_wall_total.png", log=True, fmt="{:.0f}")
    grouped_bar(base_1k, "energy_total_J",
                ylabel="Total energy (J)",
                title="1k sites: total energy",
                fname="fig_1k_energy_total.png", log=False, fmt="{:.0f}")
    stacked_energy(base_1k, "fig_1k_energy_breakdown.png")

    # speedup heatmap (rows = datatype/test/sites)
    pivot_sp = sp.pivot_table(index=["datatype", "test", "sites"], columns="hardware", values="speedup_vs_CLX")
    pivot_er = sp.pivot_table(index=["datatype", "test", "sites"], columns="hardware", values="energy_ratio_vs_CLX")
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, 0.35 * len(pivot_sp))))
    for ax, mat, title, cmap, fmt_ in [
        (axes[0], pivot_sp, "Speedup vs CLX OMP48 (higher = faster)", "Greens", "{:.2f}×"),
        (axes[1], pivot_er, "Energy ratio vs CLX OMP48 (lower = greener)", "RdYlGn_r", "{:.2f}×"),
    ]:
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([HW_LABEL.get(c, c) for c in mat.columns], rotation=20)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{a}/{b}/{int(s)}" for a, b, s in mat.index])
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat.values[r, c]
                if pd.notna(v):
                    ax.text(c, r, fmt_.format(v), ha="center", va="center", fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_speedup_energy_heatmap.png", dpi=160)
    plt.close(fig)

    logL_agreement(df, "fig_logL_agreement.png")

    # human summary
    summary = (df.groupby(["datatype", "test", "sites", "hardware"])
                 .agg(wall_s=("wall_total_s", "first"),
                      energy_J=("energy_total_J", "first"),
                      power_W=("avg_power_W", "first"),
                      cpu_share=("cpu_share", "first"),
                      best_logL=("best_logL", "first"))
                 .round(2))
    summary.to_csv(OUT_DIR / "summary.csv")
    print("\n== summary ==")
    print(summary.to_string())
    print(f"\nfigures written under {OUT_DIR}")

if __name__ == "__main__":
    main()
