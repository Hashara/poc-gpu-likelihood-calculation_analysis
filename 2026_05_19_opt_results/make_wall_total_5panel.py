#!/usr/bin/env python
"""Break the linear wall_total figure into 5 sub-panels — one per (data_type, length),
each with its own y-axis scale so small cells aren't crushed by larger ones."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_05_19_opt_results'
combined = pd.read_csv(os.path.join(OUT_DIR, 'combined_with_opt.csv'))

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
CELLS_4 = [c for c in CELLS_5 if not (c[0] == 'DNA' and c[1] == 10000000)]

def fmt_min(m):
    if pd.isna(m): return ''
    if m < 1:  return f'{m*60:.0f}s'
    if m < 60: return f'{m:.0f}m'
    h, mm = divmod(int(round(m)), 60)
    return f'{h}h{mm:02d}m' if mm else f'{h}h'

def get_val(dt, length, config_label, col='wall_total_min'):
    r = combined[(combined['data_type'] == dt)
                 & (combined['length'] == length)
                 & (combined['config_label'] == config_label)]
    return r[col].iloc[0] if len(r) and pd.notna(r[col].iloc[0]) else np.nan

from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=c[2], edgecolor=c[3], hatch=c[4], label=c[0],
          linewidth=1.2 if ('speed' in c[0] or 'opt' in c[0]) else 0.5)
    for c in CONFIG_ORDER
]

FOOTNOTE = ('Each panel has its own linear y-axis scale.   '
            'CLX = Cascade Lake Xeon 8274 (NCI `normal`, 48c)   |   '
            'SPR = Sapphire Rapids Xeon 8470Q (NCI `normalsr`, 104c).   '
            'speed_cpus = Intel build (CLX -nt 47, SPR -nt 103).   master = Clang build.   '
            'H200 10M uses NVHPC OpenACC build.   Hatched bars = May-19 optimized OpenACC build.')


def render(cells, out_name, figsize):
    fig, axes = plt.subplots(1, len(cells), figsize=figsize)
    if len(cells) == 1:
        axes = [axes]

    for ax, (dt, L, Llab) in zip(axes, cells):
        labels  = [c[0] for c in CONFIG_ORDER]
        vals    = [get_val(dt, L, c[1]) for c in CONFIG_ORDER]
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
        ax.set_ylabel('wall_total (min)')
        ax.set_title(f'{dt} — {Llab}')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.02), ncol=10, fontsize=8, frameon=True)

    fig.suptitle('Total wall-clock time (min) — per-cell linear scales')
    fig.tight_layout(rect=[0, 0.18, 1, 0.96])
    fig.text(0.5, 0.005, FOOTNOTE, ha='center', va='bottom', fontsize=7,
             style='italic', color='#444', wrap=True)

    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


render(CELLS_5, 'fig_combined_wall_total_linear_5panel.png', figsize=(24, 6))
render(CELLS_4, 'fig_combined_wall_total_linear_4panel.png', figsize=(20, 6))
