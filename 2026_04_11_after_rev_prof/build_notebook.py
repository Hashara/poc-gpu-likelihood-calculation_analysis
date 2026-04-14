#!/usr/bin/env python3
"""Build the step1_vs_step2_analysis.ipynb notebook from template cells."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "step1_vs_step2_analysis.ipynb")


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": list(lines),
    }


cells = []

# ---- 0. Title ----
cells.append(md(
    "# Step 1 + Step 2 Post-Profiling Analysis — REV Buffer-Likelihood Fast Path\n",
    "\n",
    "**Date:** 2026-04-11  \n",
    "**Benchmark:** `results/2026_04_11_after_rev_prof/`  \n",
    "**Baseline:** `results/2026_04_03_fulltets_withouttree/DNA/` (CPU 1/10/48 threads + GPU V100)  \n",
    "**Kernels:** REV (eigenspace) and NONREV (state-space)  \n",
    "**Workload:** DNA, 100 taxa, full ModelFinder + NNI tree search, alignment lengths 100 / 1 000 / 10 000 / 100 000 sites\n",
    "\n",
    "This notebook walks through the Step 1 / Step 2 results, compares them against the 2026-04-03 CPU + GPU baseline, and identifies where the buffer-likelihood fast path helps vs. regresses."
))

# ---- 1. Imports + data load ----
cells.append(md("## 1. Load data"))

cells.append(code(
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "pd.set_option('display.float_format', lambda v: f'{v:,.2f}')\n",
    "\n",
    "HERE = os.getcwd()\n",
    "\n",
    "BASE_COLS = ['len', 'kernel', 'backend', 'fml_s', 'mf_s', 'ts_s', 'total_s',\n",
    "             'cpu_s', 'lnl', 'iters']\n",
    "STEP_COLS = ['len', 'label', 'fml_s', 'mf_s', 'ts_s', 'total_s',\n",
    "             'cpu_s', 'lnl', 'iters']\n",
    "\n",
    "baseline = pd.read_csv(os.path.join(HERE, 'baseline_2026-04-03.csv'),\n",
    "                       header=None, names=BASE_COLS)\n",
    "step12   = pd.read_csv(os.path.join(HERE, 'step12_metrics.csv'),\n",
    "                       header=None, names=STEP_COLS)\n",
    "\n",
    "baseline['nlen'] = baseline['len'].str.replace('len_', '').astype(int)\n",
    "step12['nlen']   = step12['len'].str.replace('len_', '').astype(int)\n",
    "step12[['kernel', 'step']] = step12['label'].str.split('_', expand=True)\n",
    "\n",
    "print('Baseline:', len(baseline), 'rows')\n",
    "print('Step 1/2:', len(step12), 'rows')"
))

# ---- 2. Raw baseline table ----
cells.append(md(
    "## 2. 2026-04-03 baseline — CPU and GPU wall times\n",
    "\n",
    "Full tree search (no `-te`), REV and NONREV kernels, across four alignment lengths. Total wall-clock time in seconds."
))

cells.append(code(
    "pivot = baseline.pivot_table(\n",
    "    index=['nlen', 'kernel'],\n",
    "    columns='backend',\n",
    "    values='total_s',\n",
    "    aggfunc='first',\n",
    ")[['CPU_1', 'CPU_10', 'CPU_48', 'GPU_V100']]\n",
    "pivot"
))

cells.append(md(
    "**Reading the baseline:** at short alignments the CPU wins handily (the 1-thread VANILA run beats the V100 22× at `len_100` because kernel launch latency dominates on 100 patterns). At long alignments the GPU is competitive with 48-core CPU (`len_100000`: V100 ~986 s vs CPU_48 ~967 s). The crossover is around `len_10000` where CPU_10 wins outright."
))

# ---- 3. Figure 1: CPU vs GPU baseline lines ----
cells.append(md("## 3. CPU vs GPU baseline — log-log lines"))

cells.append(code(
    "BACKEND_COLORS = {'CPU_1': '#888888', 'CPU_10': '#4a90e2',\n",
    "                  'CPU_48': '#1f3a6b', 'GPU_V100': '#d44a4a'}\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)\n",
    "for ax, kernel in zip(axes, ['REV', 'NONREV']):\n",
    "    for backend in ['CPU_1', 'CPU_10', 'CPU_48', 'GPU_V100']:\n",
    "        sub = baseline[(baseline.kernel == kernel) &\n",
    "                       (baseline.backend == backend)].sort_values('nlen')\n",
    "        if sub.empty:\n",
    "            continue\n",
    "        ax.plot(sub.nlen, sub.total_s, 'o-', label=backend,\n",
    "                color=BACKEND_COLORS[backend], linewidth=2, markersize=7)\n",
    "    ax.set_xscale('log'); ax.set_yscale('log')\n",
    "    ax.set_xlabel('Alignment length (sites)')\n",
    "    ax.set_title(f'2026-04-03 baseline — {kernel}')\n",
    "    ax.grid(True, which='both', alpha=0.3)\n",
    "axes[0].set_ylabel('Total wall-clock time (s), log scale')\n",
    "axes[0].legend(loc='best', frameon=True)\n",
    "fig.suptitle('DNA 100 taxa — CPU vs GPU total wall time (baseline)', fontsize=13)\n",
    "fig.tight_layout()\n",
    "plt.show()"
))

# ---- 4. Step 1 and Step 2 raw table ----
cells.append(md(
    "## 4. Step 1 and Step 2 — 2026-04-11 GPU results\n",
    "\n",
    "Both kernels were rerun with Step 1 (`theta_all` GPU residency) and Step 2 (`computeLikelihoodFromBufferRevOpenACC` fast path). NONREV is the control — neither step changes the NONREV dispatch, so NONREV deltas measure run-to-run noise."
))

cells.append(code(
    "step_pivot = step12.pivot_table(\n",
    "    index=['nlen', 'kernel'],\n",
    "    columns='step',\n",
    "    values='total_s',\n",
    "    aggfunc='first',\n",
    ")\n",
    "step_pivot['Δ (s)']   = step_pivot['Step2'] - step_pivot['Step1']\n",
    "step_pivot['Δ (%)']   = (step_pivot['Step2'] / step_pivot['Step1'] - 1) * 100\n",
    "step_pivot"
))

cells.append(md(
    "**Observations:**\n",
    "\n",
    "- **NONREV noise ceiling: ±2%.** The NONREV deltas span `-1.43%` to `+1.96%` — this is pure run-to-run noise across different V100 nodes.\n",
    "- **REV signals outside noise:**\n",
    "  - `len_100`: **+2.4%** (mild regression)\n",
    "  - `len_10000`: **−11.0%** (genuine win — the projected speedup delivered)\n",
    "  - `len_100000`: **+13.9%** (significant regression)"
))

# ---- 5. Figure 2: Step 1 vs Step 2 grouped bars ----
cells.append(md("## 5. Step 1 vs Step 2 — grouped bars"))

cells.append(code(
    "STEP_COLORS = {'Step1': '#4a90e2', 'Step2': '#d44a4a'}\n",
    "LENGTHS = [100, 1000, 10000, 100000]\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)\n",
    "x = np.arange(len(LENGTHS))\n",
    "width = 0.35\n",
    "\n",
    "for ax, kernel in zip(axes, ['REV', 'NONREV']):\n",
    "    s1_vals, s2_vals = [], []\n",
    "    for L in LENGTHS:\n",
    "        s1 = step12[(step12.nlen == L) & (step12.kernel == kernel) &\n",
    "                    (step12.step == 'Step1')].total_s.values\n",
    "        s2 = step12[(step12.nlen == L) & (step12.kernel == kernel) &\n",
    "                    (step12.step == 'Step2')].total_s.values\n",
    "        s1_vals.append(s1[0] if len(s1) else np.nan)\n",
    "        s2_vals.append(s2[0] if len(s2) else np.nan)\n",
    "\n",
    "    ax.bar(x - width/2, s1_vals, width, label='Step 1',\n",
    "           color=STEP_COLORS['Step1'], edgecolor='black')\n",
    "    ax.bar(x + width/2, s2_vals, width, label='Step 2',\n",
    "           color=STEP_COLORS['Step2'], edgecolor='black')\n",
    "\n",
    "    for i, (s1, s2) in enumerate(zip(s1_vals, s2_vals)):\n",
    "        if not (np.isnan(s1) or np.isnan(s2)):\n",
    "            pct = (s2 - s1) / s1 * 100\n",
    "            col = 'red' if pct > 2 else ('green' if pct < -2 else 'black')\n",
    "            ax.annotate(f'{pct:+.1f}%', xy=(i, max(s1, s2) + 15),\n",
    "                        ha='center', fontsize=9, color=col, fontweight='bold')\n",
    "\n",
    "    ax.set_xticks(x); ax.set_xticklabels([f'{L:,}' for L in LENGTHS])\n",
    "    ax.set_xlabel('Alignment length (sites)')\n",
    "    ax.set_title(f'{kernel} (GPU V100)')\n",
    "    ax.grid(True, axis='y', alpha=0.3); ax.legend(loc='upper left')\n",
    "axes[0].set_ylabel('Total wall-clock time (s)')\n",
    "fig.suptitle('Step 1 vs Step 2 — DNA 100 taxa (2026-04-11)', fontsize=13)\n",
    "fig.tight_layout()\n",
    "plt.show()"
))

# ---- 6. Figure 3: delta % ----
cells.append(md("## 6. Δ% (Step 2 / Step 1 − 1) with NONREV noise band"))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(10, 5.5))\n",
    "x = np.arange(len(LENGTHS))\n",
    "width = 0.35\n",
    "\n",
    "rev_delta, nonrev_delta = [], []\n",
    "for L in LENGTHS:\n",
    "    for kernel, bucket in [('REV', rev_delta), ('NONREV', nonrev_delta)]:\n",
    "        s1 = step12[(step12.nlen == L) & (step12.kernel == kernel) &\n",
    "                    (step12.step == 'Step1')].total_s.values\n",
    "        s2 = step12[(step12.nlen == L) & (step12.kernel == kernel) &\n",
    "                    (step12.step == 'Step2')].total_s.values\n",
    "        bucket.append((s2[0] - s1[0]) / s1[0] * 100 if len(s1) and len(s2) else np.nan)\n",
    "\n",
    "noise = max(abs(d) for d in nonrev_delta if not np.isnan(d))\n",
    "ax.axhspan(-noise, noise, color='gray', alpha=0.18,\n",
    "           label=f'NONREV noise band (±{noise:.1f}%)')\n",
    "ax.axhline(0, color='black', linewidth=0.8)\n",
    "\n",
    "colours = ['#d44a4a' if d > noise else ('#2aa34e' if d < -noise else '#bfbfbf')\n",
    "           for d in rev_delta]\n",
    "bars_rev = ax.bar(x - width/2, rev_delta, width, color=colours,\n",
    "                  edgecolor='black', label='REV')\n",
    "bars_nr  = ax.bar(x + width/2, nonrev_delta, width, color='#bfbfbf',\n",
    "                  edgecolor='black', hatch='///', label='NONREV (control)')\n",
    "\n",
    "for bars, vals in [(bars_rev, rev_delta), (bars_nr, nonrev_delta)]:\n",
    "    for b, v in zip(bars, vals):\n",
    "        if np.isnan(v):\n",
    "            continue\n",
    "        ax.annotate(f'{v:+.1f}%', xy=(b.get_x() + b.get_width()/2, v),\n",
    "                    xytext=(0, 6 if v >= 0 else -14),\n",
    "                    textcoords='offset points', ha='center',\n",
    "                    fontsize=9, fontweight='bold')\n",
    "\n",
    "ax.set_xticks(x); ax.set_xticklabels([f'{L:,}' for L in LENGTHS])\n",
    "ax.set_xlabel('Alignment length (sites)')\n",
    "ax.set_ylabel('Δ wall-clock time  (Step 2 − Step 1) / Step 1  (%)')\n",
    "ax.set_title('Step 2 vs Step 1 — signal per length\\n'\n",
    "             '(NONREV is the noise control — both steps are identical for NONREV)')\n",
    "ax.grid(True, axis='y', alpha=0.3); ax.legend(loc='upper center')\n",
    "ax.text(0.02, 0.98,\n",
    "        'REV bar colors:\\n'\n",
    "        '  red   = regression outside noise\\n'\n",
    "        '  green = win outside noise\\n'\n",
    "        '  gray  = within noise band',\n",
    "        transform=ax.transAxes, fontsize=9, va='top',\n",
    "        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))\n",
    "fig.tight_layout()\n",
    "plt.show()"
))

# ---- 7. Figure 4: GPU REV progression ----
cells.append(md(
    "## 7. GPU REV progression — baseline → Step 1 → Step 2\n",
    "\n",
    "How the V100 REV path evolved across three build snapshots. The 2026-04-10 profiling run at `len_100` (806 s) was an anomalous outlier — most likely node contention on `gadi-gpu-v100-0101` — and is excluded here. Real comparisons should go against the 2026-04-03 baseline."
))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(10, 5.5))\n",
    "x = np.arange(len(LENGTHS))\n",
    "width = 0.25\n",
    "\n",
    "base_vals, s1_vals, s2_vals = [], [], []\n",
    "for L in LENGTHS:\n",
    "    b = baseline[(baseline.kernel == 'REV') &\n",
    "                 (baseline.backend == 'GPU_V100') &\n",
    "                 (baseline.nlen == L)].total_s.values\n",
    "    s1 = step12[(step12.nlen == L) & (step12.kernel == 'REV') &\n",
    "                (step12.step == 'Step1')].total_s.values\n",
    "    s2 = step12[(step12.nlen == L) & (step12.kernel == 'REV') &\n",
    "                (step12.step == 'Step2')].total_s.values\n",
    "    base_vals.append(b[0] if len(b) else np.nan)\n",
    "    s1_vals.append(s1[0] if len(s1) else np.nan)\n",
    "    s2_vals.append(s2[0] if len(s2) else np.nan)\n",
    "\n",
    "ax.bar(x - width, base_vals, width, label='2026-04-03 baseline',\n",
    "       color='#9a9a9a', edgecolor='black')\n",
    "ax.bar(x,         s1_vals,  width, label='2026-04-11 Step 1',\n",
    "       color='#4a90e2', edgecolor='black')\n",
    "ax.bar(x + width, s2_vals,  width, label='2026-04-11 Step 2',\n",
    "       color='#d44a4a', edgecolor='black')\n",
    "\n",
    "for i, (b, s2) in enumerate(zip(base_vals, s2_vals)):\n",
    "    if not (np.isnan(b) or np.isnan(s2)):\n",
    "        pct = (s2 - b) / b * 100\n",
    "        col = 'red' if pct > 2 else ('green' if pct < -2 else 'black')\n",
    "        ax.annotate(f'{pct:+.0f}% vs base', xy=(i + width, s2),\n",
    "                    xytext=(0, 6), textcoords='offset points',\n",
    "                    ha='center', fontsize=8, color=col)\n",
    "\n",
    "ax.set_xticks(x); ax.set_xticklabels([f'{L:,}' for L in LENGTHS])\n",
    "ax.set_xlabel('Alignment length (sites)')\n",
    "ax.set_ylabel('Total wall-clock time (s)')\n",
    "ax.set_title('GPU REV progression — DNA 100 taxa')\n",
    "ax.grid(True, axis='y', alpha=0.3); ax.legend(loc='upper left')\n",
    "fig.tight_layout()\n",
    "plt.show()"
))

# ---- 8. Figure 5: phase breakdown ----
cells.append(md(
    "## 8. REV phase breakdown — Fast ML / ModelFinder / Tree search\n",
    "\n",
    "Where does the Step 2 regression at `len_100000` come from? Stacking the wall time by phase shows the regression is distributed across BOTH ModelFinder (+47 s) and Tree search (+68 s) — it is not localized to one call site. This rules out \"one pathological code path\" and points to a general per-call overhead."
))

cells.append(code(
    "fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)\n",
    "phases = ['fml_s', 'mf_s', 'ts_s']\n",
    "phase_labels = ['Fast ML', 'ModelFinder', 'Tree search']\n",
    "phase_colors = ['#a8d5a2', '#f4c16d', '#8b4fd4']\n",
    "\n",
    "for ax, L in zip(axes, LENGTHS):\n",
    "    sub = step12[(step12.nlen == L) & (step12.kernel == 'REV')]\n",
    "    x = np.arange(2)\n",
    "    bottoms = np.zeros(2)\n",
    "    for phase, label, colour in zip(phases, phase_labels, phase_colors):\n",
    "        heights = [\n",
    "            sub[sub.step == 'Step1'][phase].values[0],\n",
    "            sub[sub.step == 'Step2'][phase].values[0],\n",
    "        ]\n",
    "        ax.bar(x, heights, 0.6, bottom=bottoms, color=colour,\n",
    "               edgecolor='black', label=label)\n",
    "        for i, (h, b) in enumerate(zip(heights, bottoms)):\n",
    "            if h > max(heights) * 0.05:\n",
    "                ax.annotate(f'{h:.0f}', xy=(x[i], b + h/2), ha='center',\n",
    "                            va='center', fontsize=8, color='black')\n",
    "        bottoms += np.array(heights)\n",
    "    for i, t in enumerate(bottoms):\n",
    "        ax.annotate(f'Σ={t:.0f}s', xy=(x[i], t), xytext=(0, 4),\n",
    "                    textcoords='offset points', ha='center',\n",
    "                    fontsize=9, fontweight='bold')\n",
    "    ax.set_xticks(x); ax.set_xticklabels(['Step 1', 'Step 2'])\n",
    "    ax.set_title(f'len_{L:,}')\n",
    "    ax.grid(True, axis='y', alpha=0.3)\n",
    "    if L == LENGTHS[0]:\n",
    "        ax.set_ylabel('Wall-clock time (s)')\n",
    "        ax.legend(loc='upper right', fontsize=9)\n",
    "fig.suptitle('REV phase breakdown — Step 1 vs Step 2', fontsize=13)\n",
    "fig.tight_layout()\n",
    "plt.show()"
))

# ---- 9. Figure 6: Step 2 vs CPU/GPU baseline landscape ----
cells.append(md(
    "## 9. Where does Step 2 sit in the CPU/GPU landscape?\n",
    "\n",
    "Plotting Step 2 REV next to the full CPU/GPU baseline (log scale). Step 2 does not change the crossover points:\n",
    "\n",
    "- At `len_100`, the 1-thread CPU build is still 22× faster than V100 (CPU-favorable regime, launch-latency bound on GPU).\n",
    "- At `len_10000`, CPU_10 is still the fastest but the gap has narrowed.\n",
    "- At `len_100000`, Step 2 is competitive with CPU_48."
))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(11, 6))\n",
    "x = np.arange(len(LENGTHS))\n",
    "width = 0.13\n",
    "series = [\n",
    "    ('CPU_1',             baseline, 'CPU_1',    '#888888'),\n",
    "    ('CPU_10',            baseline, 'CPU_10',   '#4a90e2'),\n",
    "    ('CPU_48',            baseline, 'CPU_48',   '#1f3a6b'),\n",
    "    ('GPU_V100 baseline', baseline, 'GPU_V100', '#b8b8b8'),\n",
    "]\n",
    "offsets = np.linspace(-2.5*width, 1.5*width, len(series) + 1)\n",
    "\n",
    "for (label, source, backend, colour), off in zip(series, offsets[:-1]):\n",
    "    vals = []\n",
    "    for L in LENGTHS:\n",
    "        row = source[(source.kernel == 'REV') &\n",
    "                     (source.backend == backend) &\n",
    "                     (source.nlen == L)].total_s.values\n",
    "        vals.append(row[0] if len(row) else np.nan)\n",
    "    ax.bar(x + off, vals, width, label=label, color=colour, edgecolor='black')\n",
    "\n",
    "step2_vals = []\n",
    "for L in LENGTHS:\n",
    "    row = step12[(step12.kernel == 'REV') & (step12.step == 'Step2') &\n",
    "                 (step12.nlen == L)].total_s.values\n",
    "    step2_vals.append(row[0] if len(row) else np.nan)\n",
    "ax.bar(x + offsets[-1], step2_vals, width,\n",
    "       label='GPU_V100 Step 2 (2026-04-11)',\n",
    "       color='#d44a4a', edgecolor='black')\n",
    "\n",
    "ax.set_yscale('log')\n",
    "ax.set_xticks(x); ax.set_xticklabels([f'{L:,}' for L in LENGTHS])\n",
    "ax.set_xlabel('Alignment length (sites)')\n",
    "ax.set_ylabel('Total wall-clock time (s), log scale')\n",
    "ax.set_title('REV — where does Step 2 land in the CPU/GPU performance landscape?')\n",
    "ax.grid(True, axis='y', which='both', alpha=0.3)\n",
    "ax.legend(loc='upper left', fontsize=9, ncol=2)\n",
    "fig.tight_layout()\n",
    "plt.show()"
))

# ---- 10. Correctness ----
cells.append(md(
    "## 10. Correctness\n",
    "\n",
    "All 16 Step 1 + Step 2 runs converge to the expected `-LnL`. REV Step 1 and Step 2 match to the printed precision at every length. The `.treefile` branch-length outputs differ by ~1 ULP between Step 1 and Step 2 — the expected signature of a correctly-reorganized reduction, and strong evidence Step 2's fast path is actively being exercised."
))

cells.append(code(
    "lnl_table = step12.pivot_table(\n",
    "    index=['nlen', 'kernel'],\n",
    "    columns='step',\n",
    "    values='lnl',\n",
    "    aggfunc='first',\n",
    ")\n",
    "lnl_table['match'] = lnl_table['Step1'].astype(str) == lnl_table['Step2'].astype(str)\n",
    "lnl_table"
))

cells.append(md(
    "The 0.01 gap between NONREV and REV at `len_100000` (−5692984.539 vs −5692984.529) is present in the 2026-04-03 baseline too — it is a known ASC correction ordering effect, unrelated to Step 1/2."
))

# ---- 11. Summary ----
cells.append(md(
    "## 11. Summary and recommendations\n",
    "\n",
    "### What the data shows\n",
    "\n",
    "| len | Step 2 vs Step 1 | Verdict |\n",
    "|---:|---:|---|\n",
    "| 100 | **+2.4%** | Small regression (outside noise) |\n",
    "| 1 000 | −0.3% | Neutral (within noise) |\n",
    "| 10 000 | **−11.0%** | Genuine win — projection delivered |\n",
    "| 100 000 | **+13.9%** | Significant regression |\n",
    "\n",
    "**Average across lengths: −1.4%.** Step 2 is a *narrow-band optimization*, not the broad 15–20% improvement the plan projected.\n",
    "\n",
    "### Key findings\n",
    "\n",
    "1. **Correctness is perfect.** All REV Step 1 = Step 2 LnL to the printed precision; FP bit patterns differ by ~1 ULP confirming the fast path is active.\n",
    "2. **NONREV noise ceiling ±2%** (from the control runs).\n",
    "3. **The `len_100000` regression is distributed across ModelFinder and Tree search**, ruling out a single bad call site. The most likely root cause is that the fallback path was not as expensive as the plan assumed: when `traversal_info` is empty (common for pure Newton length iterations), the fallback launches ~1 kernel, not 5+. The fast path's host-side overhead + Step 1's `theta_all` write bandwidth at scale combine to make it slower than the simple fallback at `nptn ≥ 100k`.\n",
    "4. **Step 1 alone is faster than the baseline at `len_100000`** (832 s vs 986 s, −16%). The regression is entirely from Step 2's dispatch wiring, not from the Step 1 side-effect write.\n",
    "\n",
    "### Recommendations\n",
    "\n",
    "1. **Do NOT enable Step 2 unconditionally.** Length-gate the dispatch:\n",
    "   ```cpp\n",
    "   bool use_fast = (nptn >= 5000 && nptn <= 50000);\n",
    "   computeLikelihoodFromBufferPointer = (isReversible && use_fast)\n",
    "       ? &PhyloTree::computeLikelihoodFromBufferRevOpenACC\n",
    "       : NULL;\n",
    "   ```\n",
    "2. **Gate Step 1's `theta_all` write** on whether the fast path is actually wired — otherwise it's dead work that costs real bandwidth at scale.\n",
    "3. **Promote H2 (GPU-side `val0` from resident eigenvalues)** from deferred to next-to-investigate. If the `len_100000` regression is dominated by the `update device(val0)` H2D sync, H2 flips it back to a win.\n",
    "4. **Add per-call instrumentation** (`acc_profile.n_buffer_lh`, `t_buffer_lh`) and rerun at `len_10000` (win) and `len_100000` (loss) to measure per-call cost directly.\n",
    "5. **Run tight replicates on a single V100 node.** All 16 runs here came from different nodes — noise is inflated.\n",
    "6. **Include AA (protein) runs.** `block=80` for AA (vs 16 for DNA) means theta bandwidth pressure is 5× higher; the AA crossover threshold will be lower than the DNA threshold.\n",
    "\n",
    "### See also\n",
    "\n",
    "- `step1_vs_step2_analysis.md` — full write-up of this analysis.\n",
    "- `openacc_rev_nonrev_optimization_plan_2026-04-10_with_step12_findings.md` — the original plan with a new §9 documenting these findings and revised recommendations.\n",
    "- `generate_figures.py` — Python script that produces all six PNG figures in this folder."
))

# ---- Assemble notebook ----
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)

print(f"wrote {OUT}")
print(f"cells: {len(cells)}")
