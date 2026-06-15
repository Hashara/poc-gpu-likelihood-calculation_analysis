"""Build the analysis notebook (treesearch_opt_analysis.ipynb) for tree-search optimisation runs.

Run this once to (re)generate the notebook. Then open the notebook to execute the cells.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK_PATH = HERE / "treesearch_opt_analysis.ipynb"


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l if l.endswith("\n") else l + "\n" for l in lines][:-1]
        + ([lines[-1]] if lines else []),
    }


def code(*lines: str) -> dict:
    src = "\n".join(lines)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]],
    }


CELLS: list[dict] = []

CELLS.append(
    md(
        "# Tree-search optimisation analysis (2026-05-13)",
        "",
        "Compare tree-search runtimes and likelihood values across three implementations:",
        "",
        "- **baseline_openacc** — current OpenACC baseline (label `baseline` in filenames)",
        "- **t1_clang** — clang_vanila CPU build at t1 step (label `t1_clang`)",
        "- **t1_openacc** — OpenACC build with the t1 optimisation (label `t1_real`)",
        "",
        "All runs use 100 taxa, tree 1, run 1. We pull timing and log-likelihood from the IQ-TREE log files for both DNA (GTR+I+G4) and AA (LG+I+G4) datasets at lengths 10k, 100k, and 1M sites.",
        "",
        "**Note on completeness:** several long runs (1M sites and AA 100k clang) did not reach `BEST SCORE FOUND` before being terminated. Those rows are flagged `incomplete` and excluded from like-for-like runtime comparisons.",
    )
)

CELLS.append(
    code(
        "import re",
        "from pathlib import Path",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "",
        "pd.set_option('display.float_format', lambda v: f'{v:,.3f}')",
        "",
        "RESULTS_ROOT = Path('/Users/u7826985/Projects/Nvidia/results/2026_05_13_treesearch_opt').resolve()",
        "print('Looking under:', RESULTS_ROOT)",
    )
)

CELLS.append(md("## 1. Discover log files and parse them"))

CELLS.append(
    code(
        "FILENAME_RE = re.compile(",
        "    r'output_teesearch_(?P<impl>baseline|t1_clang|t1_real)_'",
        "    r'(?P<datatype>DNA|AA)_(?P<model>[^_]+)_'",
        "    r'(?P<compiler>OPENACC|CLANG_VANILA)_taxa(?P<taxa>\\d+)_run(?P<run>\\d+)_'",
        "    r'tree_(?P<tree>\\d+)_(?P<length>\\d+)_'",
        "    r'iqtree3.*?\\.log$'",
        ")",
        "",
        "IMPL_LABEL = {",
        "    'baseline': 'baseline_openacc',",
        "    't1_clang': 't1_clang_vanila',",
        "    't1_real':  't1_openacc',",
        "}",
        "",
        "PATTERNS = {",
        "    'total_wall_s':      r'Total wall-clock time used: ([0-9.]+) sec',",
        "    'total_cpu_s':       r'Total CPU time used: ([0-9.]+) sec',",
        "    'tree_search_wall_s':r'Wall-clock time used for tree search: ([0-9.]+) sec',",
        "    'tree_search_cpu_s': r'CPU time used for tree search: ([0-9.]+) sec',",
        "    'best_score':        r'BEST SCORE FOUND\\s*:\\s*(-?[0-9.]+)',",
        "    'rapidnj_ll':        r'Log-likelihood of RapidNJ tree:\\s*(-?[0-9.]+)',",
        "    'compute_initial_s': r'Computing log-likelihood of 98 initial trees \\.\\.\\. ([0-9.]+) seconds',",
        "    'iterations':        r'TREE SEARCH COMPLETED AFTER (\\d+) ITERATIONS',",
        "    'init_parsimony_s':  r'Create initial parsimony tree by phylogenetic likelihood library \\(PLL\\)\\.\\.\\. ([0-9.]+) seconds',",
        "}",
        "",
        "def parse_log(path: Path) -> dict:",
        "    text = path.read_text(errors='replace')",
        "    out = {'path': str(path), 'filename': path.name}",
        "    m = FILENAME_RE.match(path.name)",
        "    if m:",
        "        out.update(m.groupdict())",
        "        out['impl_label'] = IMPL_LABEL.get(out['impl'], out['impl'])",
        "        out['length'] = int(out['length'])",
        "    for key, pat in PATTERNS.items():",
        "        match = re.search(pat, text)",
        "        out[key] = float(match.group(1)) if match else np.nan",
        "    out['complete'] = not np.isnan(out['total_wall_s']) and not np.isnan(out['best_score'])",
        "    return out",
        "",
        "log_files = sorted(p for p in RESULTS_ROOT.rglob('output_teesearch_*.log'))",
        "print(f'Found {len(log_files)} log files')",
        "for p in log_files:",
        "    print(' -', p.relative_to(RESULTS_ROOT))",
    )
)

CELLS.append(
    code(
        "records = [parse_log(p) for p in log_files]",
        "df = pd.DataFrame(records)",
        "df = df[[",
        "    'datatype', 'length', 'impl_label',",
        "    'total_wall_s', 'total_cpu_s',",
        "    'tree_search_wall_s', 'tree_search_cpu_s',",
        "    'compute_initial_s', 'init_parsimony_s',",
        "    'iterations', 'best_score', 'rapidnj_ll', 'complete', 'filename', 'path',",
        "]].sort_values(['datatype', 'length', 'impl_label']).reset_index(drop=True)",
        "df.drop(columns=['path'])  # display without the long path column",
    )
)

CELLS.append(md("## 2. Runtime comparison",
                "",
                "Pivot wall-clock time (seconds) by datatype, length, and implementation. `NaN` means the run did not finish — see Section 4 for the partial progress those runs reached."))

CELLS.append(
    code(
        "wall_pivot = df.pivot_table(",
        "    index=['datatype', 'length'],",
        "    columns='impl_label',",
        "    values='total_wall_s',",
        "    aggfunc='first',",
        ")[['baseline_openacc', 't1_clang_vanila', 't1_openacc']]",
        "wall_pivot",
    )
)

CELLS.append(
    code(
        "ts_pivot = df.pivot_table(",
        "    index=['datatype', 'length'],",
        "    columns='impl_label',",
        "    values='tree_search_wall_s',",
        "    aggfunc='first',",
        ")[['baseline_openacc', 't1_clang_vanila', 't1_openacc']]",
        "ts_pivot",
    )
)

CELLS.append(md("### Speed-ups",
                "",
                "Ratios > 1 mean the optimisation is faster than the reference.",
                "",
                "- `t1_openacc_vs_baseline`: t1 OpenACC speed-up over the OpenACC baseline (this is the optimisation we're trying to land).",
                "- `t1_openacc_vs_clang`: how much t1 OpenACC beats the clang_vanila CPU build (proxy for GPU-vs-CPU benefit).",
                "- `baseline_vs_clang`: how the existing baseline already compares to the CPU build."))

CELLS.append(
    code(
        "speedup = pd.DataFrame(index=wall_pivot.index)",
        "speedup['baseline_openacc_s']  = wall_pivot['baseline_openacc']",
        "speedup['t1_clang_vanila_s']   = wall_pivot['t1_clang_vanila']",
        "speedup['t1_openacc_s']        = wall_pivot['t1_openacc']",
        "speedup['t1_openacc_vs_baseline'] = wall_pivot['baseline_openacc'] / wall_pivot['t1_openacc']",
        "speedup['t1_openacc_vs_clang']    = wall_pivot['t1_clang_vanila'] / wall_pivot['t1_openacc']",
        "speedup['baseline_vs_clang']      = wall_pivot['t1_clang_vanila'] / wall_pivot['baseline_openacc']",
        "speedup",
    )
)

CELLS.append(md("## 3. Cross-verify likelihoods",
                "",
                "All three implementations should converge to the same `BEST SCORE FOUND` log-likelihood, because the optimisation must not change numerical correctness. We tabulate the values per (datatype, length) and show the max pairwise absolute difference. A non-zero difference signals a numerical regression."))

CELLS.append(
    code(
        "ll_pivot = df.pivot_table(",
        "    index=['datatype', 'length'],",
        "    columns='impl_label',",
        "    values='best_score',",
        "    aggfunc='first',",
        ")[['baseline_openacc', 't1_clang_vanila', 't1_openacc']]",
        "ll_pivot",
    )
)

CELLS.append(
    code(
        "def row_max_abs_diff(row):",
        "    vals = row.dropna().values",
        "    if len(vals) < 2:",
        "        return np.nan",
        "    return float(np.max(vals) - np.min(vals))",
        "",
        "ll_check = ll_pivot.copy()",
        "ll_check['max_abs_diff'] = ll_check.apply(row_max_abs_diff, axis=1)",
        "ll_check['match'] = ll_check['max_abs_diff'].apply(lambda v: 'OK' if (np.isnan(v) or v == 0.0) else 'MISMATCH')",
        "ll_check",
    )
)

CELLS.append(md("Also cross-check the RapidNJ initial-tree log-likelihood. This is identical input across the three runs (same alignment), so it provides a sanity check that the likelihood kernel agrees at step 0."))

CELLS.append(
    code(
        "rnj_pivot = df.pivot_table(",
        "    index=['datatype', 'length'],",
        "    columns='impl_label',",
        "    values='rapidnj_ll',",
        "    aggfunc='first',",
        ")[['baseline_openacc', 't1_clang_vanila', 't1_openacc']]",
        "rnj_check = rnj_pivot.copy()",
        "rnj_check['max_abs_diff'] = rnj_pivot.apply(row_max_abs_diff, axis=1)",
        "rnj_check",
    )
)

CELLS.append(md("## 4. Incomplete runs",
                "",
                "These runs did not reach `BEST SCORE FOUND`. We show the final iteration / time recorded in the log so you can see how far they got."))

CELLS.append(
    code(
        "ITER_RE = re.compile(r'^Iteration (\\d+) / LogL: (-?[0-9.]+) / Time: ([0-9hms:]+)', re.MULTILINE)",
        "",
        "def last_iteration(path: Path):",
        "    text = path.read_text(errors='replace')",
        "    matches = list(ITER_RE.finditer(text))",
        "    if not matches:",
        "        return (np.nan, np.nan, '')",
        "    m = matches[-1]",
        "    return (int(m.group(1)), float(m.group(2)), m.group(3))",
        "",
        "incomplete = df[~df['complete']].copy()",
        "if incomplete.empty:",
        "    print('All runs completed.')",
        "    incomplete_view = None",
        "else:",
        "    last_iters = [last_iteration(Path(p)) for p in incomplete['path']]",
        "    incomplete[['last_iter', 'last_logL', 'last_time']] = last_iters",
        "    incomplete_view = incomplete[['datatype', 'length', 'impl_label',",
        "                                  'last_iter', 'last_logL', 'last_time']].reset_index(drop=True)",
        "incomplete_view",
    )
)

CELLS.append(md("## 5. Plots",
                "",
                "Wall-clock runtime per implementation, faceted by datatype and length (log scale). Incomplete runs are shown as hatched bars with their final recorded time so the comparison is honest about which runs ran to completion.",
                "",
                "Every figure is also written to `figures/*.png` next to the notebook."))

CELLS.append(
    code(
        "FIG_DIR = Path('figures')",
        "FIG_DIR.mkdir(exist_ok=True)",
        "",
        "def plot_runtime(metric='total_wall_s', title='Total wall-clock time (s)', savename=None):",
        "    pivot = df.pivot_table(",
        "        index=['datatype', 'length'],",
        "        columns='impl_label',",
        "        values=metric,",
        "        aggfunc='first',",
        "    )[['baseline_openacc', 't1_clang_vanila', 't1_openacc']]",
        "    complete_pivot = df.pivot_table(",
        "        index=['datatype', 'length'],",
        "        columns='impl_label',",
        "        values='complete',",
        "        aggfunc='first',",
        "    )[['baseline_openacc', 't1_clang_vanila', 't1_openacc']]",
        "    datatypes = sorted({i[0] for i in pivot.index})",
        "    fig, axes = plt.subplots(1, len(datatypes), figsize=(7 * len(datatypes), 5), sharey=False)",
        "    if len(datatypes) == 1:",
        "        axes = [axes]",
        "    for ax, dt in zip(axes, datatypes):",
        "        sub = pivot.xs(dt, level='datatype')",
        "        sub_complete = complete_pivot.xs(dt, level='datatype')",
        "        lengths = sub.index.tolist()",
        "        impls = ['baseline_openacc', 't1_clang_vanila', 't1_openacc']",
        "        x = np.arange(len(lengths))",
        "        width = 0.27",
        "        colors = {'baseline_openacc': '#4c72b0', 't1_clang_vanila': '#dd8452', 't1_openacc': '#55a868'}",
        "        for j, impl in enumerate(impls):",
        "            vals = sub[impl].values",
        "            done = sub_complete[impl].values",
        "            bars = ax.bar(x + (j - 1) * width, vals, width, label=impl, color=colors[impl],",
        "                          edgecolor='black')",
        "            for bar, d in zip(bars, done):",
        "                if not bool(d):",
        "                    bar.set_hatch('//')",
        "            for xi, v, d in zip(x + (j - 1) * width, vals, done):",
        "                if not np.isnan(v):",
        "                    ax.text(xi, v, f'{v:,.0f}{\"*\" if not bool(d) else \"\"}', ha='center', va='bottom', fontsize=8, rotation=0)",
        "        ax.set_xticks(x)",
        "        ax.set_xticklabels([f'{L:,}' for L in lengths])",
        "        ax.set_xlabel('alignment length (sites)')",
        "        ax.set_ylabel(title)",
        "        ax.set_title(f'{dt} — {title}')",
        "        ax.set_yscale('log')",
        "        ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.4)",
        "        ax.legend()",
        "    plt.suptitle(title + ' (hatched / *= incomplete run)')",
        "    plt.tight_layout()",
        "    if savename:",
        "        out = FIG_DIR / savename",
        "        fig.savefig(out, dpi=150, bbox_inches='tight')",
        "        print(f'Saved {out}')",
        "    return fig",
        "",
        "plot_runtime('total_wall_s', 'Total wall-clock time (s)', savename='total_wall_clock.png')",
        "plt.show()",
    )
)

CELLS.append(
    code(
        "plot_runtime('tree_search_wall_s', 'Tree-search wall-clock time (s)', savename='tree_search_wall_clock.png')",
        "plt.show()",
    )
)

CELLS.append(md("### Speed-up plot",
                "",
                "How much faster is `t1_openacc` than the OpenACC baseline and the clang CPU build? Bars > 1.0 mean the optimisation wins."))

CELLS.append(
    code(
        "su = speedup[['t1_openacc_vs_baseline', 't1_openacc_vs_clang', 'baseline_vs_clang']].copy()",
        "su = su.dropna(how='all')",
        "ax = su.plot(kind='bar', figsize=(10, 4), edgecolor='black')",
        "ax.axhline(1.0, color='red', linestyle='--', alpha=0.6, label='no speed-up')",
        "ax.set_ylabel('speed-up (×)')",
        "ax.set_title('Wall-clock speed-ups')",
        "ax.grid(True, axis='y', linestyle='--', alpha=0.4)",
        "ax.legend()",
        "plt.tight_layout()",
        "speedup_path = FIG_DIR / 'speedup.png'",
        "ax.get_figure().savefig(speedup_path, dpi=150, bbox_inches='tight')",
        "print(f'Saved {speedup_path}')",
        "plt.show()",
    )
)

CELLS.append(md("## 6. Summary",
                "",
                "Generate a single tidy table to drop into the writeup."))

CELLS.append(
    code(
        "summary = wall_pivot.copy()",
        "summary.columns = [f'{c}_wall_s' for c in summary.columns]",
        "summary = summary.join(ll_check[['baseline_openacc', 't1_clang_vanila', 't1_openacc', 'max_abs_diff', 'match']].rename(",
        "    columns={c: f'{c}_logL' for c in ['baseline_openacc', 't1_clang_vanila', 't1_openacc']}",
        "))",
        "summary['t1_openacc_vs_baseline'] = speedup['t1_openacc_vs_baseline']",
        "summary['t1_openacc_vs_clang']    = speedup['t1_openacc_vs_clang']",
        "summary",
    )
)

CELLS.append(
    code(
        "summary.to_csv('treesearch_opt_summary.csv')",
        "df.to_csv('treesearch_opt_all_runs.csv', index=False)",
        "print('Wrote treesearch_opt_summary.csv and treesearch_opt_all_runs.csv')",
    )
)

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(NOTEBOOK, indent=1))
print(f"Wrote {NOTEBOOK_PATH}")
