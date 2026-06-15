"""
V100 single-thread vs OpenMP (-nt 12) comparison.

Parses the 4 IQ-TREE 3.1.2 OpenACC logs from
  /Users/u7826985/Projects/Nvidia/results/2026_05_25_v100_withOpenMP/
extracts wall-clock and energy, writes summary.csv and a 2x2 comparison plot.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(
    "/Users/u7826985/Projects/Nvidia/results/2026_05_25_v100_withOpenMP"
)
OUT_DIR = Path(__file__).resolve().parent


WALL_RE = re.compile(r"Total wall-clock time used:\s+([\d.]+)\s+sec")
CPU_TIME_RE = re.compile(r"Total CPU time used:\s+([\d.]+)\s+sec")
CPU_ENERGY_RE = re.compile(r"CPU:\s+([\d.]+)\s+J\s+\(avg\s+([\d.]+)\s+W\)")
GPU_ENERGY_RE = re.compile(r"GPU:\s+([\d.]+)\s+J\s+\(avg\s+([\d.]+)\s+W\)")
KERNEL_RE = re.compile(r"Kernel:\s+(\S+)\s+-\s+(\d+)\s+threads")
HOST_RE = re.compile(r"Host:\s+(\S+)")
COMMAND_RE = re.compile(r"Command:\s+(.+)")
BEST_SCORE_RE = re.compile(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)")
ITER_RE = re.compile(
    r"Iteration\s+(\d+)\s+/\s+LogL:\s+(-?[\d.]+)\s+/\s+Time:\s+(\d+)h:(\d+)m:(\d+)s"
)
BETTER_TREE_RE = re.compile(r"BETTER TREE FOUND at iteration\s+(\d+):\s+(-?[\d.]+)")
TREE_LENGTH_RE = re.compile(r"Total tree length:\s+([\d.]+)")


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    fname = path.name

    # data type from filename prefix (matches '_AA_' or '_DNA_' segment of OPENACC arg)
    datatype = "AA" if "_AA_" in fname else ("DNA" if "_DNA_" in fname else "?")
    # OpenMP variant: filenames with '_nt12_' use -nt 12
    variant = "openmp_nt12" if "_nt12_" in fname else "single"

    def first(rx):
        m = rx.search(text)
        return m.groups() if m else None

    kernel = first(KERNEL_RE)
    host = first(HOST_RE)
    cmd = first(COMMAND_RE)
    wall = first(WALL_RE)
    cputime = first(CPU_TIME_RE)
    cpu_e = first(CPU_ENERGY_RE)
    gpu_e = first(GPU_ENERGY_RE)
    best = first(BEST_SCORE_RE)
    tlen = first(TREE_LENGTH_RE)
    completed = "TREE SEARCH COMPLETED" in text and "Total wall-clock time used:" in text

    iters = [
        (int(m.group(1)), float(m.group(2)),
         int(m.group(3)) * 3600 + int(m.group(4)) * 60 + int(m.group(5)))
        for m in ITER_RE.finditer(text)
    ]
    better = [(int(m.group(1)), float(m.group(2))) for m in BETTER_TREE_RE.finditer(text)]

    return {
        "datatype": datatype,
        "variant": variant,
        "threads": int(kernel[1]) if kernel else None,
        "host": host[0] if host else None,
        "completed": completed,
        "wall_s": float(wall[0]) if wall else np.nan,
        "cpu_time_s": float(cputime[0]) if cputime else np.nan,
        "cpu_energy_J": float(cpu_e[0]) if cpu_e else np.nan,
        "cpu_avg_W": float(cpu_e[1]) if cpu_e else np.nan,
        "gpu_energy_J": float(gpu_e[0]) if gpu_e else np.nan,
        "gpu_avg_W": float(gpu_e[1]) if gpu_e else np.nan,
        "best_logL": float(best[0]) if best else np.nan,
        "tree_length": float(tlen[0]) if tlen else np.nan,
        "first_better_logL": better[0][1] if better else np.nan,
        "iters": iters,
        "log": path.name,
    }


def main() -> None:
    logs = sorted(RESULTS_DIR.glob("*_iqtree.log"))
    rows = [parse_log(p) for p in logs]
    df = pd.DataFrame(rows)
    df["total_energy_J"] = df["cpu_energy_J"] + df["gpu_energy_J"]
    df = df.sort_values(["datatype", "variant"]).reset_index(drop=True)
    df.drop(columns=["iters"]).to_csv(OUT_DIR / "summary.csv", index=False)
    print(df.drop(columns=["iters"]).to_string(index=False))

    # Build paired comparison (single vs openmp_nt12) per datatype.
    completed = df[df["completed"]].copy()
    print("\nCompleted runs:", len(completed), "of", len(df))

    pivot_wall = completed.pivot(index="datatype", columns="variant", values="wall_s")
    pivot_cpu_e = completed.pivot(index="datatype", columns="variant", values="cpu_energy_J")
    pivot_gpu_e = completed.pivot(index="datatype", columns="variant", values="gpu_energy_J")
    pivot_total_e = completed.pivot(index="datatype", columns="variant", values="total_energy_J")

    # Speedup of openmp_nt12 vs single (>1 means openmp faster).
    if "openmp_nt12" in pivot_wall.columns and "single" in pivot_wall.columns:
        pivot_wall["speedup_omp_vs_single"] = pivot_wall["single"] / pivot_wall["openmp_nt12"]
    print("\nWall-clock (s):")
    print(pivot_wall.to_string())

    # ----- Plot: 2x2 grid (wall, cpu energy, gpu energy, total energy) -----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    metrics = [
        ("Wall-clock time (s)", pivot_wall.drop(columns=["speedup_omp_vs_single"], errors="ignore")),
        ("CPU energy (J)", pivot_cpu_e),
        ("GPU energy (J)", pivot_gpu_e),
        ("Total energy CPU+GPU (J)", pivot_total_e),
    ]
    colors = {"single": "#4C78A8", "openmp_nt12": "#F58518"}
    variants_order = ["single", "openmp_nt12"]
    for ax, (title, pv) in zip(axes.ravel(), metrics):
        datatypes = list(pv.index)
        x = np.arange(len(datatypes))
        width = 0.38
        for i, v in enumerate(variants_order):
            if v not in pv.columns:
                continue
            vals = pv[v].values
            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width,
                label=v,
                color=colors[v],
                edgecolor="black",
                linewidth=0.5,
            )
            for xi, val in zip(x + (i - 0.5) * width, vals):
                if np.isfinite(val):
                    ax.text(
                        xi,
                        val,
                        f"{val:,.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(datatypes)
        ax.set_title(title)
        ax.set_ylabel(title.split(" (")[0])
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "V100 OpenACC — single-thread vs OpenMP (-nt 12)\n"
        "100 taxa, 100k sites, IQ-TREE 3.1.2 (May 17 2026 build)",
        fontsize=12,
    )
    fig.tight_layout()
    fig_path = OUT_DIR / "fig_single_vs_openmp.png"
    fig.savefig(fig_path, dpi=160)
    print(f"\nWrote {fig_path}")
    print(f"Wrote {OUT_DIR / 'summary.csv'}")

    # ----- Likelihood comparison -----
    print("\n=== Final log-likelihood comparison ===")
    ll_cols = ["datatype", "variant", "threads", "best_logL", "tree_length", "completed"]
    print(df[ll_cols].to_string(index=False))

    # Per-datatype Δ logL between single and openmp_nt12
    pivot_ll = completed.pivot(index="datatype", columns="variant", values="best_logL")
    if {"single", "openmp_nt12"}.issubset(pivot_ll.columns):
        print("\n=== Δ logL (openmp_nt12 - single) ===")
        for dt in pivot_ll.index:
            s_ll = pivot_ll.loc[dt, "single"]
            o_ll = pivot_ll.loc[dt, "openmp_nt12"]
            if np.isfinite(s_ll) and np.isfinite(o_ll):
                print(
                    f"  {dt}: single={s_ll:.6f}  openmp_nt12={o_ll:.6f}  "
                    f"ΔlogL={o_ll - s_ll:+.6f}  "
                    f"(identical: {abs(o_ll - s_ll) < 1e-3})"
                )

    # ----- Convergence plot: logL vs wall-clock time -----
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    datatypes_plot = ["AA", "DNA"]
    style = {"single": ("#4C78A8", "o-"), "openmp_nt12": ("#F58518", "s--")}
    for ax, dt in zip(axes2, datatypes_plot):
        plotted = False
        for _, row in df[df["datatype"] == dt].iterrows():
            iters = row["iters"]
            if not iters:
                continue
            its, lls, ts = zip(*iters)
            color, marker = style.get(row["variant"], ("gray", "x-"))
            ax.plot(
                np.array(ts) / 60.0,
                lls,
                marker,
                color=color,
                label=f"{row['variant']} ({row['threads']}t)",
                markersize=4,
            )
            plotted = True
            if np.isfinite(row["best_logL"]):
                ax.axhline(
                    row["best_logL"],
                    color=color,
                    linestyle=":",
                    alpha=0.5,
                    linewidth=0.8,
                )
        ax.set_xlabel("Wall-clock time (min)")
        ax.set_ylabel("LogL")
        ax.set_title(f"{dt} — tree-search convergence")
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=9)
    fig2.suptitle(
        "Likelihood convergence — V100 OpenACC, single vs OpenMP (-nt 12)",
        fontsize=12,
    )
    fig2.tight_layout()
    fig2_path = OUT_DIR / "fig_logL_convergence.png"
    fig2.savefig(fig2_path, dpi=160)
    print(f"\nWrote {fig2_path}")

    # Save the per-iteration trace as a long-form CSV for downstream use.
    rows_long = []
    for r in rows:
        for it, ll, t in r["iters"]:
            rows_long.append(
                {
                    "datatype": r["datatype"],
                    "variant": r["variant"],
                    "threads": r["threads"],
                    "iteration": it,
                    "logL": ll,
                    "wall_s": t,
                }
            )
    pd.DataFrame(rows_long).to_csv(OUT_DIR / "iteration_logL.csv", index=False)
    print(f"Wrote {OUT_DIR / 'iteration_logL.csv'}")

    # ----- Console-friendly summary table -----
    if {"single", "openmp_nt12"}.issubset(pivot_wall.columns):
        print("\n=== single vs openmp_nt12 (per datatype) ===")
        for dt in pivot_wall.index:
            s_wall = pivot_wall.loc[dt, "single"]
            o_wall = pivot_wall.loc[dt, "openmp_nt12"]
            if np.isfinite(s_wall) and np.isfinite(o_wall):
                delta = (o_wall - s_wall) / s_wall * 100
                s_e = pivot_total_e.loc[dt, "single"]
                o_e = pivot_total_e.loc[dt, "openmp_nt12"]
                e_delta = (o_e - s_e) / s_e * 100 if np.isfinite(s_e) else np.nan
                print(
                    f"  {dt}: wall {s_wall:,.0f}s -> {o_wall:,.0f}s ({delta:+.1f}% with OpenMP);  "
                    f"total energy {s_e:,.0f}J -> {o_e:,.0f}J ({e_delta:+.1f}%)"
                )


if __name__ == "__main__":
    main()
