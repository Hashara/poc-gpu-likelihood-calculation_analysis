"""Parse the modelwise (`-te <fixed-tree> -m <model>`) timing logs and compare
wall-clock + lnL across 3 builds × 4 model variants × 6 cells (datatype × sites).

These runs skip ModelFinder and tree search entirely — they evaluate one
specific model on a fixed topology.  Useful for isolating per-model GPU vs CPU
speedup without the search-space noise.

Source: /Users/u7826985/Projects/Nvidia/results/2026_06_24_simulated_results/modelwise/
Layout: <DATATYPE>/<MODEL_CELL>/taxa_<N>/len_<L>/tree_1/output_*modelwise*.log

Build / hardware variants observed:
  cpu_modelwisetest      → CPU SPR OMP_104  (Intel SPR)
  cudajolt_modelwise     → GPU H200         (IQTREE_GPU_SHARED build)
  openacc_stable_modelwise → GPU H200       (openACC stable build)

Model variants per cell:
  AA  cell (LG+I+G4 simulated data) :  LG / LG-G4 / LG-I / LG-I-G4
  DNA cell (GTR+I+G4 simulated data):  GTR / GTR-G4 / GTR-I / GTR-I-G4

Outputs:
  runs_modelwise.csv     all 72 parsed rows
  status_summary.csv     pivot view (cell × build × model → status)
  fig_wall_by_model.png  wall time per (datatype, sites) panel
  fig_speedup_heatmap.png  GPU vs CPU wall ratio per (cell × model)
  fig_logL_agreement.png   sanity check — all builds should hit same lnL per (cell × model)
  fig_wall_scaling.png     wall vs sites, line per (build × model)
  SUMMARY.txt
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/Users/u7826985/Projects/Nvidia/results/2026_06_24_simulated_results/modelwise")
OUT  = Path(__file__).resolve().parent

# ---------- regexes ----------
RE_TOTAL_WALL = re.compile(r"Total wall-clock time used:\s+([\d.]+)\s+sec")
RE_TREE_WALL  = re.compile(r"Wall-clock time used for tree search:\s+([\d.]+)\s+sec")
RE_BEST_LOGL  = re.compile(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)")
RE_E_CPU      = re.compile(r"Energy:\s*\n\s*CPU:\s+([\d.]+)\s+J", re.M)
RE_E_GPU      = re.compile(r"^\s*GPU:\s+([\d.]+)\s+J", re.M)
RE_GPU_MEM    = re.compile(r"GPU mem:\s+([\d.]+)\s*/\s*([\d.]+)\s+MB.*?\+([\d.]+)\s+MB")
RE_CPU_MEM    = re.compile(r"CPU mem:\s+([\d.]+)\s*/\s*([\d.]+)\s+MB")
RE_CMD_MODEL  = re.compile(r"-m\s+(\S+)")
RE_HOST       = re.compile(r"^Host:\s+(\S+)", re.M)

def _f(rx, txt, grp=1):
    m = rx.search(txt)
    return float(m.group(grp)) if m else None

def build_of(name: str) -> str:
    if "cpu_modelwisetest"     in name: return "cpu_OMP_104"
    if "cudajolt_modelwise"    in name: return "cudajolt_H200"
    if "openacc_stable_modelwise" in name: return "openACC_H200"
    return "unknown"

def model_of(cmd_model: str | None, name: str) -> str:
    """Resolve the model variant from -m on Command line; fall back to filename."""
    if cmd_model:
        return cmd_model
    # in the filename it appears as a single token like _LG_, _LG-G4_, _LG-I-G4_
    # right after the OMP_104 / H200 token, in the form _<MODEL>_100taxa_
    m = re.search(r"_(LG[\-A-Z0-9]*|GTR[\-A-Z0-9]*)_100taxa_", name)
    return m.group(1).replace("-", "+") if m else "unknown"

# ---------- parse ----------
rows = []
for log in sorted(ROOT.rglob("output_*modelwise*.log")):
    txt = log.read_text(errors="ignore")
    rel = log.relative_to(ROOT)
    parts = rel.parts                # e.g. ['AA','LG+I+G4','taxa_100','len_10000000','tree_1','output_...log']
    datatype = parts[0]
    sim_model = parts[1]             # data was simulated under this model
    taxa = int(parts[2].split("_")[1])
    sites = int(parts[3].split("_")[1])
    tree_dir = log.parent
    base = log.name[:-len(".log")]

    cmd_model_match = RE_CMD_MODEL.search(txt)
    inference_model = model_of(cmd_model_match.group(1) if cmd_model_match else None, log.name)
    has_wall = "Total wall-clock time used" in txt

    gm = RE_GPU_MEM.search(txt)
    cm = RE_CPU_MEM.search(txt)
    host = RE_HOST.search(txt)
    rows.append({
        "file": log.name,
        "datatype": datatype,
        "sim_model": sim_model,
        "taxa": taxa,
        "sites": sites,
        "build": build_of(log.name),
        "inference_model": inference_model,
        "status": "COMPLETE" if has_wall else "INCOMPLETE",
        "host": host.group(1) if host else None,
        "wall_total_s": _f(RE_TOTAL_WALL, txt),
        "wall_tree_s":  _f(RE_TREE_WALL,  txt),
        "best_logL":    _f(RE_BEST_LOGL,  txt),
        "energy_cpu_J": _f(RE_E_CPU, txt),
        "energy_gpu_J": _f(RE_E_GPU, txt),
        "gpu_mem_peak_MB":  float(gm.group(1)) if gm else None,
        "gpu_mem_delta_MB": float(gm.group(3)) if gm else None,
        "cpu_mem_peak_MB":  float(cm.group(1)) if cm else None,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "runs_modelwise.csv", index=False)
print(f"Parsed {len(df)} logs → runs_modelwise.csv")
print(f"Status: {df['status'].value_counts().to_dict()}")
print(f"Builds: {df['build'].value_counts().to_dict()}")
print(f"Models present: {sorted(df['inference_model'].unique())}")

# ---------- status pivot ----------
pivot = df.pivot_table(
    index=["datatype", "sim_model", "taxa", "sites"],
    columns=["build", "inference_model"],
    values="status", aggfunc="first")
pivot.to_csv(OUT / "status_summary.csv")

ok = df[df.status == "COMPLETE"].copy()
print(f"COMPLETE: {len(ok)} / {len(df)}")

# ---------- ordering / styles ----------
BUILD_ORDER = ["cpu_OMP_104", "cudajolt_H200", "openACC_H200"]
BUILD_LABEL = {"cpu_OMP_104": "CPU SPR 104T",
               "cudajolt_H200": "cudajolt H200",
               "openACC_H200":  "openACC H200"}
BUILD_COLOR = {"cpu_OMP_104":   "#e6611c",
               "cudajolt_H200": "#3182bd",
               "openACC_H200":  "#1f7a3a"}

def model_families(dt: str) -> "dict[str, list[str]]":
    """Return {family_name: [models]} in reading order (basic → R → I+R)."""
    if dt == "AA":
        return {
            "Basic (±I ±G4)":  ["LG", "LG+G4", "LG+I", "LG+I+G4"],
            "FreeRate +R":     [f"LG+R{i}"   for i in range(1, 8)],
            "Invar + FreeRate +I+R": [f"LG+I+R{i}" for i in range(1, 8)],
        }
    if dt == "DNA":
        return {
            "Basic (±I ±G4)":  ["GTR", "GTR+G4", "GTR+I", "GTR+I+G4"],
            "FreeRate +R":     [f"GTR+R{i}"   for i in range(1, 8)],
            "Invar + FreeRate +I+R": [f"GTR+I+R{i}" for i in range(1, 8)],
        }
    return {"All": sorted(ok[ok.datatype == dt]["inference_model"].unique())}

def model_order(dt: str) -> list[str]:
    return [m for fam in model_families(dt).values() for m in fam]

def _sites_tag(s):
    return f"{s//1_000_000}M" if s >= 1_000_000 else f"{s//1000}K"

# ---------- fig 1: wall-clock, reviewer-friendly design ----------
# Horizontal bars (models on y-axis) + log-x wall time.
# - Model families visually grouped with alternating background bands and family label.
# - Time landmarks (1s / 10s / 1min / 10min / 1h / 10h) as vertical reference lines
#   so the eye reads wall time without needing per-bar labels.
# - CPU bar is the reference; GPU bars are annotated with fold-change (2.5× faster / 3.0× slower).
# - GPU bar edge glows green (faster than CPU) or red (slower) — instant win/lose readout.
def _fmt_time(v):
    if pd.isna(v) or v <= 0: return ""
    if v < 60:   return f"{v:.0f}s"
    if v < 3600: return f"{v/60:.0f}m"
    return f"{v/3600:.1f}h"

TIME_LANDMARKS = [(1, "1 s"), (10, "10 s"), (60, "1 min"),
                  (600, "10 min"), (3600, "1 h"), (36000, "10 h")]

for dt in ("AA", "DNA"):
    sub_dt = ok[ok.datatype == dt]
    if sub_dt.empty: continue
    families = model_families(dt)
    families = {name: mods for name, mods in families.items()
                if any(m in sub_dt.inference_model.unique() for m in mods)}
    sites_list = sorted(sub_dt["sites"].unique())
    builds     = [b for b in BUILD_ORDER if b in sub_dt.build.unique()]

    # Ordered model list per datatype (basic → +R → +I+R) with family boundaries
    ordered_models, fam_boundaries = [], []
    row_cursor = 0
    for fam_name, fam_models in families.items():
        present = [m for m in fam_models if m in sub_dt.inference_model.unique()]
        if not present: continue
        ordered_models.extend(present)
        fam_boundaries.append((fam_name, row_cursor, row_cursor + len(present)))
        row_cursor += len(present)
    n_models = len(ordered_models)
    n_sites  = len(sites_list)

    # Wide-enough figure per site column; tall enough for every model row.
    # Extra left margin reserved for model labels on the leftmost panel.
    fig, axes = plt.subplots(1, n_sites,
                             figsize=(5.6 * n_sites, 0.42 * n_models + 2.4),
                             squeeze=False, sharey=True,
                             gridspec_kw={"wspace": 0.10, "left": 0.11,
                                          "right": 0.94, "top": 0.90, "bottom": 0.12})
    axes = axes[0]

    for ci, sites in enumerate(sites_list):
        ax = axes[ci]
        sub = sub_dt[sub_dt.sites == sites]
        cpu_wall_by_model = {m: sub[(sub.build == "cpu_OMP_104") & (sub.inference_model == m)]["wall_total_s"].iloc[0]
                             if not sub[(sub.build == "cpu_OMP_104") & (sub.inference_model == m)].empty else np.nan
                             for m in ordered_models}

        # y positions per (model, build) — 3 stacked bars per model
        n_builds = len(builds)
        h = 0.75 / n_builds
        y_model = np.arange(n_models)

        # alternate family background bands
        for fi, (fam_name, y0, y1) in enumerate(fam_boundaries):
            if fi % 2 == 0:
                ax.axhspan(y0 - 0.5, y1 - 0.5, color="#f6f6f6", zorder=0)
            # family label on far-right
            if ci == n_sites - 1:
                ax.text(1.02, (y0 + y1 - 1) / 2, fam_name,
                        rotation=270, va="center", ha="left", fontsize=8,
                        fontweight="bold", transform=ax.get_yaxis_transform(),
                        clip_on=False)

        for bi, b in enumerate(builds):
            offs = (bi - (n_builds - 1) / 2) * h
            for mi, m in enumerate(ordered_models):
                r = sub[(sub.build == b) & (sub.inference_model == m)]
                if r.empty: continue
                v = r["wall_total_s"].iloc[0]
                cpu = cpu_wall_by_model.get(m)
                # edge colour = win/lose vs CPU (only meaningful for GPU rows)
                if b == "cpu_OMP_104" or pd.isna(cpu) or cpu <= 0:
                    edge = "black"; lw = 0.5; halo = None
                else:
                    speedup = cpu / v
                    if speedup >= 1.10:
                        edge = "#1b7f3f"; lw = 1.3     # GPU wins
                    elif speedup <= 0.9:
                        edge = "#c73030"; lw = 1.3     # CPU wins
                    else:
                        edge = "#666"; lw = 0.7        # tie
                    halo = speedup
                ax.barh(y_model[mi] + offs, v, h,
                        color=BUILD_COLOR[b], edgecolor=edge, linewidth=lw,
                        label=BUILD_LABEL[b] if (ci == 0 and mi == 0) else None,
                        zorder=3)
                # bar label: absolute time + (for GPU) fold-change
                bar_label = _fmt_time(v)
                if halo is not None and np.isfinite(halo):
                    if halo >= 1.05:   bar_label += f"  ↓{halo:.1f}×"
                    elif halo <= 0.95: bar_label += f"  ↑{1/halo:.1f}×"
                ax.text(v * 1.05, y_model[mi] + offs, bar_label,
                        va="center", ha="left", fontsize=7, zorder=4)

        # time landmarks
        for tsec, tlbl in TIME_LANDMARKS:
            ax.axvline(tsec, color="#888", linewidth=0.5, linestyle=":", zorder=1)
            ax.text(tsec, -0.9, tlbl, ha="center", va="top",
                    fontsize=7, color="#666", clip_on=False)

        ax.set_yticks(y_model)
        # Show model names on every panel (not just leftmost) for scannability
        ax.set_yticklabels(ordered_models, fontsize=9, fontweight="bold")
        ax.tick_params(axis="y", pad=2)
        ax.set_ylim(n_models - 0.5, -0.5)   # invert so first model at top
        ax.set_xscale("log")
        # x range: 0.5s → widest bar × 3
        max_v = sub_dt[sub_dt.sites == sites]["wall_total_s"].max()
        if pd.notna(max_v) and max_v > 0:
            ax.set_xlim(0.5, max_v * 6)
        ax.set_xlabel("wall time (log)  •  ↓N× = GPU faster than CPU,  ↑N× = GPU slower", fontsize=8)
        ax.set_title(f"{_sites_tag(sites)} sites", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.25, which="major", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend: three build colours + two edge-colour meanings
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=BUILD_COLOR[b], edgecolor="black",
                          label=BUILD_LABEL[b]) for b in builds]
    legend_elems += [
        Patch(facecolor="white", edgecolor="#1b7f3f", linewidth=1.5, label="GPU faster than CPU"),
        Patch(facecolor="white", edgecolor="#c73030", linewidth=1.5, label="GPU slower than CPU"),
    ]
    fig.legend(handles=legend_elems, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=len(legend_elems),
               fontsize=9, frameon=True)
    fig.suptitle(f"{dt}  •  Modelwise wall-clock  ·  horizontal bars, log-scale time",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.savefig(OUT / f"fig_wall_by_model_{dt}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_wall_by_model_{dt}] written")

# ---------- fig 2: GPU vs CPU speedup heatmap ----------
base = ok[ok.build == "cpu_OMP_104"][[
    "datatype", "sites", "inference_model", "wall_total_s"]].rename(
        columns={"wall_total_s": "cpu_wall"})
rel = ok.merge(base, on=["datatype", "sites", "inference_model"], how="inner")
rel["speedup"] = rel["cpu_wall"] / rel["wall_total_s"]
rel = rel[rel.build != "cpu_OMP_104"]
# Pivot rows = (datatype, sites, model), cols = build
pv = rel.pivot_table(index=["datatype", "sites", "inference_model"],
                      columns="build", values="speedup", aggfunc="mean")
gpu_builds = [b for b in BUILD_ORDER if b in pv.columns]
pv = pv.reindex(columns=gpu_builds)
pv.to_csv(OUT / "speedup_vs_cpu.csv")

# One heatmap per datatype: rows = model (grouped by family), cols = (site × build)
# Colour = log2(speedup) so 1× is neutral, GPU wins are green, CPU wins are red.
from matplotlib.colors import TwoSlopeNorm
for dt in ("AA", "DNA"):
    pv_dt = pv[pv.index.get_level_values("datatype") == dt]
    if pv_dt.empty: continue
    families = model_families(dt)
    ordered_models = [m for fam in families.values() for m in fam
                      if m in pv_dt.index.get_level_values("inference_model")]
    sites_list = sorted(pv_dt.index.get_level_values("sites").unique())
    gpu_builds = list(pv_dt.columns)
    col_labels = []
    for s in sites_list:
        for b in gpu_builds:
            col_labels.append((s, b))

    mat = np.full((len(ordered_models), len(col_labels)), np.nan)
    for i, m in enumerate(ordered_models):
        for j, (s, b) in enumerate(col_labels):
            try:
                mat[i, j] = pv_dt.loc[(dt, s, m), b]
            except KeyError:
                pass
    log_mat = np.log2(mat)  # negative → CPU wins, positive → GPU wins
    span = max(0.5, np.nanmax(np.abs(log_mat)) if np.isfinite(log_mat).any() else 0.5)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-span, vmax=span)

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(col_labels)), max(3.5, 0.35 * len(ordered_models))))
    im = ax.imshow(log_mat, aspect="auto", cmap="RdYlGn", norm=norm)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([f"{_sites_tag(int(s))}\n{BUILD_LABEL[b]}" for s, b in col_labels],
                       fontsize=8)
    ax.set_yticks(range(len(ordered_models)))
    ax.set_yticklabels(ordered_models, fontsize=8)

    # Vertical separators between site groups
    for k in range(1, len(sites_list)):
        ax.axvline(k * len(gpu_builds) - 0.5, color="black", linewidth=1.0)

    # Horizontal separators between families
    row_cursor = 0
    for fam_name, mods in families.items():
        present = [m for m in mods if m in ordered_models]
        if not present: continue
        end = row_cursor + len(present)
        ax.axhline(end - 0.5, color="black", linewidth=1.0)
        # family label to the right of the plot
        mid = row_cursor + (len(present) - 1) / 2.0
        ax.text(len(col_labels) - 0.35, mid, fam_name, rotation=270,
                ha="left", va="center", fontsize=8, fontweight="bold",
                transform=ax.transData, clip_on=False)
        row_cursor = end

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                col = "black" if abs(np.log2(v)) < 0.7 else "white"
                ax.text(j, i, f"{v:.2f}×", ha="center", va="center", fontsize=7, color=col)

    ax.set_title(f"{dt}  •  GPU speedup vs CPU SPR 104T  (green > 1× = GPU faster)",
                 fontsize=11, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, label="log₂(speedup)   0 = tied,  +1 = 2× GPU,  −1 = 2× CPU")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_speedup_heatmap_{dt}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_speedup_heatmap_{dt}] written")

# ---------- fig 3: logL agreement ----------
ref = ok[ok.build == "cpu_OMP_104"][[
    "datatype", "sites", "inference_model", "best_logL"]].rename(
        columns={"best_logL": "logL_ref"})
agree = ok.merge(ref, on=["datatype", "sites", "inference_model"], how="inner")
agree["delta_logL"] = agree["best_logL"] - agree["logL_ref"]
agree["rel_diff"]   = agree["delta_logL"] / agree["logL_ref"].abs()
agree.to_csv(OUT / "logL_agreement.csv", index=False)

if not agree.empty:
    # Two-panel per datatype: (top) relative diff on symmetric log axis;
    # (bottom) same points on ±0.05 linear zoom to show the mass near zero.
    for dt in ("AA", "DNA"):
        a_dt = agree[agree.datatype == dt].copy()
        if a_dt.empty: continue
        families = model_families(dt)
        ordered = [(s, m) for s in sorted(a_dt.sites.unique())
                   for m in [x for fam in families.values() for x in fam
                             if x in a_dt.inference_model.unique()]]
        # deduplicate while preserving order
        seen_lab = set(); ordered = [x for x in ordered if not (x in seen_lab or seen_lab.add(x))]
        label_idx = {c: i for i, c in enumerate(ordered)}
        sites_list = sorted(a_dt.sites.unique())

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(max(12, 0.36 * len(ordered)), 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 2], "hspace": 0.05})
        for _, r in a_dt.iterrows():
            i = label_idx[(r.sites, r.inference_model)]
            for a in (ax_top, ax_bot):
                a.scatter(i, r.rel_diff, color=BUILD_COLOR.get(r.build, "k"),
                          s=45, edgecolor="black", linewidth=0.35,
                          label=BUILD_LABEL.get(r.build) if a is ax_top else None,
                          zorder=3)
        for a in (ax_top, ax_bot):
            a.axhline(0, color="grey", linewidth=0.5, zorder=1)
            a.grid(axis="y", alpha=0.3, zorder=0)
        # dashed vertical dividers between site groups
        counts_per_site = {s: sum(1 for x in ordered if x[0] == s) for s in sites_list}
        cursor = 0
        for s in sites_list[:-1]:
            cursor += counts_per_site[s]
            for a in (ax_top, ax_bot):
                a.axvline(cursor - 0.5, color="black", linewidth=0.6, alpha=0.4)

        ax_top.set_yscale("symlog", linthresh=1e-9)
        ax_top.set_ylabel("relative Δ (log/symlog)")
        ax_top.set_title(f"{dt}  •  logL agreement — same model, same data, all builds\n"
                         f"top: symlog full range   |   bottom: ±5×10⁻⁹ zoom",
                         fontsize=11, fontweight="bold")
        ax_bot.set_ylim(-5e-9, 5e-9)
        ax_bot.set_ylabel("relative Δ (linear zoom)")

        ax_bot.set_xticks(range(len(ordered)))
        ax_bot.set_xticklabels([f"{_sites_tag(s)}/{m}" for s, m in ordered],
                               rotation=90, ha="center", fontsize=7)
        # site range annotations above the top plot
        cursor = 0
        for s in sites_list:
            n = counts_per_site[s]
            mid = cursor + (n - 1) / 2.0
            ax_top.text(mid, ax_top.get_ylim()[1] * 1.05, f"{_sites_tag(s)}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold",
                        transform=ax_top.get_xaxis_transform(),
                        clip_on=False)
            cursor += n

        h, l = ax_top.get_legend_handles_labels()
        seen_l = {}
        for hh, ll in zip(h, l):
            seen_l[ll] = hh
        ax_top.legend(seen_l.values(), seen_l.keys(), fontsize=8,
                       loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_logL_agreement_{dt}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig_logL_agreement_{dt}] written")

# ---------- fig 4: scaling with sites — median line + min-max band per build ----------
# One subplot per model family, 3 lines per subplot (one per build). Shaded region
# = min-max envelope across the models in that family. Clean 9-line total.
for dt in ("AA", "DNA"):
    sub = ok[ok.datatype == dt]
    if sub.empty: continue
    families = model_families(dt)
    families = {n: mods for n, mods in families.items()
                if any(m in sub.inference_model.unique() for m in mods)}
    ncols = len(families)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 5.0),
                             squeeze=False, sharey=True)
    axes = axes[0]
    for ci, (fam_name, fam_models) in enumerate(families.items()):
        ax = axes[ci]
        models = [m for m in fam_models if m in sub.inference_model.unique()]
        for b in BUILD_ORDER:
            sb = sub[(sub.build == b) & (sub.inference_model.isin(models))]
            if sb.empty: continue
            g = sb.groupby("sites")["wall_total_s"].agg(["median", "min", "max"]).reset_index()
            g = g.sort_values("sites")
            ax.plot(g["sites"], g["median"], "-o",
                    color=BUILD_COLOR[b], linewidth=2.0, markersize=6,
                    label=BUILD_LABEL[b])
            # shaded envelope only if the family has >1 model
            if len(models) > 1 and (g["min"] < g["max"]).any():
                ax.fill_between(g["sites"], g["min"], g["max"],
                                color=BUILD_COLOR[b], alpha=0.18, linewidth=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("alignment length (sites, log)")
        if ci == 0: ax.set_ylabel("wall total (s, log)")
        subtitle = fam_name + (f"  (n={len(models)} models)" if len(models) > 1 else "")
        ax.set_title(subtitle, fontsize=10, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)
        if ci == 0:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle(f"{dt}  •  Modelwise wall scaling with alignment length  "
                 "(line=median across models in family, band=min–max)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_wall_scaling_{dt}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_wall_scaling_{dt}] written")

# ---------- text summary ----------
with (OUT / "SUMMARY.txt").open("w") as fh:
    fh.write("Modelwise analysis — 2026_06_24 subfolder\n")
    fh.write("=" * 60 + "\n\n")
    fh.write(f"Source: {ROOT}\n")
    fh.write(f"Total logs parsed:    {len(df)}\n")
    for k, v in df.status.value_counts().items():
        fh.write(f"  {k:<12} {v}\n")
    fh.write("\nBuilds × models per cell (COMPLETE only):\n")
    table = ok.groupby(["datatype", "sites", "build"])["inference_model"].apply(
        lambda s: ",".join(sorted(s))).reset_index()
    for _, r in table.iterrows():
        fh.write(f"  {r['datatype']}/{_sites_tag(r['sites']):<5}  {r['build']:<16}  {r['inference_model']}\n")
    fh.write("\nTop median GPU speedup vs CPU SPR 104T:\n")
    med = rel.groupby("build")["speedup"].agg(["median", "min", "max", "count"])
    fh.write(med.to_string() + "\n")
    fh.write("\nlogL agreement (Δ from CPU ref):\n")
    summ = agree.groupby("build")["delta_logL"].agg(["mean", "min", "max"])
    fh.write(summ.to_string() + "\n")
    fh.write("\nFigures:\n")
    for p in sorted(OUT.glob("fig_*.png")):
        fh.write(f"  {p.name}\n")

print("Done. Wrote:")
for p in sorted(OUT.iterdir()):
    if p.is_file():
        print(" ", p.name)
