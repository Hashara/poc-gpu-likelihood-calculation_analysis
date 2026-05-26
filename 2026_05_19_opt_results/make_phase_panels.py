#!/usr/bin/env python
"""Build 5-panel comparison figures for IQ-TREE phase wall-clock times:
  - Tree-search wall time          (`Wall-clock time used for tree search:`)
  - ModelFinder wall time          (`Wall-clock time for ModelFinder:`)
  - Parsimony tree generation time (`Generating <N> parsimony trees... <X> second`)

Same per-cell linear-scale layout as fig_combined_wall_total_linear_5panel.png.
"""
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT_DIR = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_05_19_opt_results'

# Source roots
GPU_BASELINE_ROOT = '/Users/u7826985/Projects/Nvidia/results/2026_05_04_testing_gpus'
GPU_OPT_ROOT      = '/Users/u7826985/Projects/Nvidia/results/2026_05_19_opt_results'
MASTER_CPU_ROOT   = '/Users/u7826985/Projects/Nvidia/results/2026_05_10_master_cputests'
SPEED_CPU_ROOT    = '/Users/u7826985/Projects/Nvidia/results/2026_05_17_speed_cpus'

PATTERNS = {
    'wall_total_sec':       re.compile(r'Total wall-clock time used:\s+([\d.]+)\s+sec'),
    'wall_treesearch_sec':  re.compile(r'Wall-clock time used for tree search:\s+([\d.]+)\s+sec'),
    'modelfinder_wall_sec': re.compile(r'Wall-clock time for ModelFinder:\s+([\d.]+)\s+seconds'),
    'parsimony_wall_sec':   re.compile(r'Generating\s+\d+\s+parsimony trees\.\.\.\s+([\d.]+)\s+second'),
    'init_parsimony_sec':   re.compile(r'Create initial parsimony tree by phylogenetic likelihood library \(PLL\)\.\.\.\s+([\d.]+)\s+second'),
}

def parse_log(path):
    if not os.path.isfile(path):
        return {k: np.nan for k in PATTERNS}
    with open(path, 'r', errors='ignore') as fh:
        txt = fh.read()
    out = {}
    for k, pat in PATTERNS.items():
        m = pat.search(txt)
        out[k] = float(m.group(1)) if m else np.nan
    return out

# ---------------------------------------------------------------------------
# Build (data_type, length, config_label, log_path) tuples for every cell
# that appears in the 5-panel figure.
# ---------------------------------------------------------------------------

def first_glob(pattern):
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None

ROWS = []

def add(dt, length, config_label, path):
    ROWS.append((dt, length, config_label, path))

# CPU master Clang: CLX 48T + SPR 104T at 100K/1M
for dt, model in [('AA', 'LG+I+G4'), ('DNA', 'GTR+I+G4')]:
    for L in (100000, 1000000):
        suffix = 'aa' if dt == 'AA' else ''
        # CLX 48T
        p = first_glob(os.path.join(MASTER_CPU_ROOT, dt, f'len_{L}',
            f'output_CLANG_MASTER_{dt}_{model}_OMP_48_taxa100_run1_tree_1_{L}_iqtree3_*_OMP48_*_aa.log' if dt == 'AA'
            else f'output_CLANG_MASTER_{dt}_{model}_OMP_48_taxa100_run1_tree_1_{L}_iqtree3_*_OMP48_*.log'))
        add(dt, L, 'CLX 48T', p)
        # SPR 104T
        p = first_glob(os.path.join(MASTER_CPU_ROOT, dt, f'len_{L}',
            f'output_NORMALSR_MASTER_{dt}_{model}_OMP_104_taxa100_run1_tree_1_{L}_iqtree3_*_OMP104_*_aa.log' if dt == 'AA'
            else f'output_NORMALSR_MASTER_{dt}_{model}_OMP_104_taxa100_run1_tree_1_{L}_iqtree3_*_OMP104_*.log'))
        add(dt, L, 'SPR 104T', p)

# Speed CPU subdirs (Intel build, CLX 47T / SPR 103T) at 100K and 1M
SPEED_MAP = {
    ('AA',  100000,  'CLX 47T (speed)'):  'AA_100k_normal_seed1_168422809',
    ('AA',  100000,  'SPR 103T (speed)'): 'AA_100k_spr_seed1_168425673',
    ('AA',  1000000, 'CLX 47T (speed)'):  'AA_1m_normal_seed1_168425490',
    ('AA',  1000000, 'SPR 103T (speed)'): 'AA_1m_spr_seed1_168425491',
    ('DNA', 100000,  'CLX 47T (speed)'):  'DNA_100k_normal_seed1_168422811',
    ('DNA', 100000,  'SPR 103T (speed)'): 'DNA_100k_spr_seed1_168425674',
    ('DNA', 1000000, 'CLX 47T (speed)'):  'DNA_1m_normal_seed1_168422813',
    ('DNA', 1000000, 'SPR 103T (speed)'): 'DNA_1m_spr_seed1_168425675',
}
for (dt, L, lab), sub in SPEED_MAP.items():
    add(dt, L, lab, os.path.join(SPEED_CPU_ROOT, sub, 'iqtree_run.log'))

# Speed CPU 10M DNA SPR 103T (flat-file)
add('DNA', 10000000, 'SPR 103T (speed)', first_glob(os.path.join(
    SPEED_CPU_ROOT, 'output_intel_compiler_test_DNA_*OMP_104*10000000*.log')))

# Baseline GPU runs (May-04 test_cases) at 100K and 1M
for dt, model, tail in [('AA', 'LG+I+G4', 'aa_iqtree'), ('DNA', 'GTR+I+G4', 'iqtree')]:
    for L in (100000, 1000000):
        for gpu in ('V100', 'A100', 'H200'):
            p = first_glob(os.path.join(GPU_BASELINE_ROOT, dt, f'len_{L}',
                f'output_test_cases_{dt}_{model}_OPENACC_taxa100_run1_tree_1_{L}_iqtree3_{dt}_OPENACC_*-I-G4_{gpu}_100taxa_{L}len_OPENACC_run1_tree_1_{L}_{tail}.log'))
            # AA 1M H200 baseline: plain test_cases run was incomplete (no "Total wall-clock"
            # line). The canonical wall_total 12197.809 in combined_with_opt.csv comes from
            # the mem_full DEFAULT variant — fall back to it when the plain log is short.
            if p is not None and os.path.getsize(p) < 25000:
                alt = first_glob(os.path.join(GPU_BASELINE_ROOT, dt, f'len_{L}',
                    f'output_test_cases_mem_full_{dt}_{model}_OPENACC_taxa100_run1_tree_1_{L}_iqtree3_GPUMEM_{dt}_OPENACC_*-I-G4_{gpu}_DEFAULT_*len_OPENACC_run1_tree_1_{L}_{tail}.log'))
                if alt:
                    p = alt
            add(dt, L, f'GPU {gpu}', p)

# Baseline GPU 10M DNA H200 (NVHPC OpenACC), flat-file in May-17 speed_cpus
add('DNA', 10000000, 'GPU H200', first_glob(os.path.join(
    SPEED_CPU_ROOT, 'output_10M_DNA_*OPENACC*H200*10000000*.log')))

# Opt GPU runs (May-19) at 100K and 1M
for dt in ('AA', 'DNA'):
    for L in (100000, 1000000):
        for gpu in ('V100', 'A100', 'H200'):
            p = first_glob(os.path.join(GPU_OPT_ROOT, dt, f'len_{L}',
                f'output_{gpu.lower()}_test_after_opt_*OPENACC*{gpu}*_{L}len_OPENACC_*.log'))
            add(dt, L, f'GPU {gpu} opt', p)

# ---------------------------------------------------------------------------
# Parse logs and cross-check wall_total against the existing combined CSV
# ---------------------------------------------------------------------------
combined = pd.read_csv(os.path.join(OUT_DIR, 'combined_with_opt.csv'))

records = []
missing = []
for dt, L, lab, path in ROWS:
    if path is None or not os.path.isfile(path):
        missing.append((dt, L, lab, path))
        rec = {k: np.nan for k in PATTERNS}
    else:
        rec = parse_log(path)
    rec.update(data_type=dt, length=L, config_label=lab, log_path=path or '')
    records.append(rec)

phase = pd.DataFrame(records)

# Sanity cross-check: parsed wall_total vs existing combined CSV
check = phase.merge(
    combined[['data_type', 'length', 'config_label', 'wall_total_sec']]
        .rename(columns={'wall_total_sec': 'wall_total_sec_existing'}),
    on=['data_type', 'length', 'config_label'], how='left')
check['delta'] = (check['wall_total_sec'] - check['wall_total_sec_existing']).abs()
mismatches = check[(check['delta'] > 1.0) & check['wall_total_sec'].notna()
                   & check['wall_total_sec_existing'].notna()]
if len(mismatches):
    print('WARN: wall_total mismatches between parsed log and combined_with_opt.csv:')
    print(mismatches[['data_type', 'length', 'config_label',
                      'wall_total_sec', 'wall_total_sec_existing', 'delta']])

if missing:
    print('Missing logs (no file found):')
    for row in missing:
        print('  ', row)

# Convert to minutes for plotting
for col in ['wall_total_sec', 'wall_treesearch_sec',
            'modelfinder_wall_sec', 'parsimony_wall_sec', 'init_parsimony_sec']:
    phase[col.replace('_sec', '_min')] = phase[col] / 60.0

phase.to_csv(os.path.join(OUT_DIR, 'combined_with_opt_phases.csv'), index=False)
print(f'wrote {os.path.join(OUT_DIR, "combined_with_opt_phases.csv")}  ({len(phase)} rows)')

# ---------------------------------------------------------------------------
# Plotting — replicate make_wall_total_5panel.py layout
# ---------------------------------------------------------------------------
CONFIG_ORDER = [
    ('CLX 48T',         'CLX 48T',          '#1565C0', '#555', None),
    ('CLX 47T speed',   'CLX 47T (speed)',  '#0D47A1', '#000', None),
    ('SPR 104T',        'SPR 104T',         '#E65100', '#555', None),
    ('SPR 103T speed',  'SPR 103T (speed)', '#BF360C', '#000', None),
    ('GPU V100',        'GPU V100',         '#A5D6A7', '#1B5E20', None),
    ('GPU V100 opt',    'GPU V100 opt',     '#A5D6A7', '#000',    '//'),
    ('GPU A100',        'GPU A100',         '#66BB6A', '#1B5E20', None),
    ('GPU A100 opt',    'GPU A100 opt',     '#66BB6A', '#000',    '//'),
    ('GPU H200',        'GPU H200',         '#2E7D32', '#1B5E20', None),
    ('GPU H200 opt',    'GPU H200 opt',     '#2E7D32', '#000',    '//'),
]

CELLS_5 = [
    ('AA',  100000,  '100K'),
    ('AA',  1000000, '1M'),
    ('DNA', 100000,  '100K'),
    ('DNA', 1000000, '1M'),
    ('DNA', 10000000,'10M'),
]

def fmt_min(m):
    if pd.isna(m): return ''
    if m < 1:  return f'{m*60:.0f}s'
    if m < 60: return f'{m:.0f}m'
    h, mm = divmod(int(round(m)), 60)
    return f'{h}h{mm:02d}m' if mm else f'{h}h'

def get_val(dt, length, config_label, col):
    r = phase[(phase['data_type'] == dt)
              & (phase['length'] == length)
              & (phase['config_label'] == config_label)]
    return r[col].iloc[0] if len(r) and pd.notna(r[col].iloc[0]) else np.nan

legend_handles = [
    Patch(facecolor=c[2], edgecolor=c[3], hatch=c[4], label=c[0],
          linewidth=1.2 if ('speed' in c[0] or 'opt' in c[0]) else 0.5)
    for c in CONFIG_ORDER
]

FOOTNOTE_BASE = ('Each panel has its own linear y-axis scale.   '
                 'CLX = Cascade Lake Xeon 8274 (NCI `normal`, 48c)   |   '
                 'SPR = Sapphire Rapids Xeon 8470Q (NCI `normalsr`, 104c).   '
                 'speed_cpus = Intel build (CLX -nt 47, SPR -nt 103).   master = Clang build.   '
                 'H200 10M uses NVHPC OpenACC build.   Hatched bars = May-19 optimized OpenACC build.')

def render(cells, metric_col, title, out_name, footnote_tail='', figsize=(24, 6)):
    fig, axes = plt.subplots(1, len(cells), figsize=figsize)
    if len(cells) == 1:
        axes = [axes]

    for ax, (dt, L, Llab) in zip(axes, cells):
        labels  = [c[0] for c in CONFIG_ORDER]
        vals    = [get_val(dt, L, c[1], metric_col) for c in CONFIG_ORDER]
        colors  = [c[2] for c in CONFIG_ORDER]
        edges   = [c[3] for c in CONFIG_ORDER]
        hatches = [c[4] for c in CONFIG_ORDER]
        xp = np.arange(len(labels))

        for xi, v, col, ed, ht, lab in zip(xp, vals, colors, edges, hatches, labels):
            if np.isnan(v):
                continue
            lw = 1.2 if ('speed' in lab or 'opt' in lab) else 0.5
            ax.bar(xi, v, 0.8, color=col, edgecolor=ed, linewidth=lw, hatch=ht)
            ax.text(xi, v, fmt_min(v), ha='center', va='bottom', fontsize=7, rotation=90)

        finite = [v for v in vals if not np.isnan(v)]
        if finite:
            ax.set_ylim(0, max(finite) * 1.25)

        ax.set_xticks(xp)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(f'{title} (min)')
        ax.set_title(f'{dt} — {Llab}')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.02), ncol=10, fontsize=8, frameon=True)

    fig.suptitle(f'{title} (min) — per-cell linear scales')
    fig.tight_layout(rect=[0, 0.18, 1, 0.96])
    fig.text(0.5, 0.005, FOOTNOTE_BASE + (('   ' + footnote_tail) if footnote_tail else ''),
             ha='center', va='bottom', fontsize=7, style='italic', color='#444', wrap=True)

    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


# Tree-search
render(CELLS_5, 'wall_treesearch_min',
       'Tree-search wall-clock time',
       'fig_combined_wall_treesearch_linear_5panel.png')

# ModelFinder
render(CELLS_5, 'modelfinder_wall_min',
       'ModelFinder wall-clock time',
       'fig_combined_wall_modelfinder_linear_5panel.png')

# Parsimony tree generation (the "Generating N parsimony trees" step, right before tree search)
render(CELLS_5, 'parsimony_wall_min',
       'Parsimony tree generation wall-clock time (pre-tree-search)',
       'fig_combined_wall_parsimony_linear_5panel.png',
       footnote_tail='"Parsimony tree generation" = IQ-TREE `Generating N parsimony trees` step, '
                     'runs after ModelFinder and immediately before tree search '
                     '(not the single initial PLL parsimony tree that runs pre-ModelFinder).')

# Initial PLL parsimony tree step (single tree created up-front via PLL)
render(CELLS_5, 'init_parsimony_min',
       'Initial PLL parsimony tree wall-clock time',
       'fig_combined_wall_init_parsimony_linear_5panel.png',
       footnote_tail='"Initial PLL parsimony tree" = IQ-TREE `Create initial parsimony tree by '
                     'phylogenetic likelihood library (PLL)` step (single tree, runs before ModelFinder).')
