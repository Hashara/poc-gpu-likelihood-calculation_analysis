"""Parse IQ-TREE3 master-branch CPU benchmark logs (2026-05-10) and produce CSV + plots.

Dataset layout: results/2026_05_10_master_cputests/{AA,DNA}/len_{10000,100000,1000000}/*.log
Two hardware platforms (both built with Clang) are present in the filenames:
  - CLANG_MASTER    -> NCI `normal`   queue, Intel Xeon Platinum 8274  (Cascade Lake, 2x24c, 3.2 GHz, DDR4)  -> labelled CLX
  - NORMALSR_MASTER -> NCI `normalsr` queue, Intel Xeon Platinum 8470Q (Sapphire Rapids, 2x52c, 2.1 GHz, DDR5) -> labelled SPR
Thread counts: 1 (VANILA), 10, 48 on CLX; 1, 10, 48, 104 on SPR.
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_ROOT = '/Users/u7826985/Projects/Nvidia/results/2026_05_10_master_cputests'
OUT_DIR = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_05_10_master_cputests'

SYSTEMS = ['CLANG_MASTER', 'NORMALSR_MASTER']
LENGTHS = [10000, 100000, 1000000]
LENGTH_LABEL = {10000: '10K', 100000: '100K', 1000000: '1M'}

# Map raw system tag (from filenames) -> hardware short-name used in plots
HW_LABEL = {'CLANG_MASTER': 'CLX', 'NORMALSR_MASTER': 'SPR'}
HW_LONG  = {'CLANG_MASTER': 'Cascade Lake (Xeon 8274, 2x24c)',
            'NORMALSR_MASTER': 'Sapphire Rapids (Xeon 8470Q, 2x52c)'}

FOOTNOTE = ('CLX = Cascade Lake (Intel Xeon Platinum 8274, NCI `normal` queue, 2x24c)   |   '
            'SPR = Sapphire Rapids (Intel Xeon Platinum 8470Q, NCI `normalsr` queue, 2x52c)')


def add_footnote(fig):
    fig.text(0.5, -0.02, FOOTNOTE, ha='center', va='top', fontsize=8, style='italic', color='#444')


# (label, system, threads, color)
CONFIG_SPECS = [
    ('CLX 1T',   'CLANG_MASTER',    1,   '#90CAF9'),
    ('CLX 10T',  'CLANG_MASTER',    10,  '#42A5F5'),
    ('CLX 48T',  'CLANG_MASTER',    48,  '#1565C0'),
    ('SPR 1T',   'NORMALSR_MASTER', 1,   '#FFCC80'),
    ('SPR 10T',  'NORMALSR_MASTER', 10,  '#FFA726'),
    ('SPR 48T',  'NORMALSR_MASTER', 48,  '#FB8C00'),
    ('SPR 104T', 'NORMALSR_MASTER', 104, '#E65100'),
]


def parse_log(path):
    with open(path, 'r') as f:
        content = f.read()
    rec = {'path': path, 'filename': os.path.basename(path)}

    def grab(pattern, cast=float):
        m = re.search(pattern, content)
        return cast(m.group(1)) if m else None

    rec['wall_total_sec']      = grab(r'Total wall-clock time used:\s+([\d.]+)\s+sec')
    rec['cpu_total_sec']       = grab(r'Total CPU time used:\s+([\d.]+)\s+sec')
    rec['wall_treesearch_sec'] = grab(r'Wall-clock time used for tree search:\s+([\d.]+)\s+sec')
    rec['cpu_treesearch_sec']  = grab(r'CPU time used for tree search:\s+([\d.]+)\s+sec')
    rec['best_lnl']            = grab(r'BEST SCORE FOUND\s*:\s*([-\d.]+)')
    rec['iterations']          = grab(r'Total number of iterations:\s+(\d+)', int)
    rec['completed']           = rec['wall_total_sec'] is not None
    return rec


def classify(filename):
    if 'CLANG_MASTER' in filename:
        system = 'CLANG_MASTER'
    elif 'NORMALSR_MASTER' in filename:
        system = 'NORMALSR_MASTER'
    else:
        system = 'UNKNOWN'

    if 'OMP_104' in filename or 'OMP104' in filename:
        threads = 104
    elif 'OMP_48' in filename or 'OMP48' in filename:
        threads = 48
    elif 'OMP_10' in filename or 'OMP10' in filename:
        threads = 10
    else:
        threads = 1
    return system, threads


def load_all():
    records = []
    for data_type in ['AA', 'DNA']:
        for length in LENGTHS:
            log_dir = os.path.join(RESULTS_ROOT, data_type, f'len_{length}')
            for logfile in sorted(glob.glob(os.path.join(log_dir, '*.log'))):
                rec = parse_log(logfile)
                system, threads = classify(rec['filename'])
                rec['data_type'] = data_type
                rec['length'] = length
                rec['length_label'] = LENGTH_LABEL[length]
                rec['system'] = system
                rec['threads'] = threads
                rec['model'] = 'LG+I+G4' if data_type == 'AA' else 'GTR+I+G4'
                rec['hw_label'] = HW_LABEL.get(system, system)
                rec['config_label'] = f"{rec['hw_label']} {threads}T"
                rec['wall_total_min'] = rec['wall_total_sec'] / 60 if rec['wall_total_sec'] else None
                rec['wall_treesearch_min'] = rec['wall_treesearch_sec'] / 60 if rec['wall_treesearch_sec'] else None
                records.append(rec)
    return pd.DataFrame(records)


def fmt_time(minutes):
    if minutes is None or pd.isna(minutes):
        return ''
    total_min = round(minutes)
    if total_min < 1:
        return f'{minutes*60:.0f}s'
    if total_min < 60:
        return f'{total_min}m'
    h, m = divmod(total_min, 60)
    return f'{h}h{m:02d}m' if m > 0 else f'{h}h'


# ---------- Figures ---------------------------------------------------------

def fig_wall_total(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    for ax, dt in zip(axes, ['AA', 'DNA']):
        sub = df[df['data_type'] == dt]
        n_cfg = len(CONFIG_SPECS)
        x = np.arange(len(LENGTHS))
        width = 0.8 / n_cfg
        for i, (label, system, threads, color) in enumerate(CONFIG_SPECS):
            vals = []
            for length in LENGTHS:
                row = sub[(sub['system'] == system) & (sub['threads'] == threads) & (sub['length'] == length)]
                vals.append(row['wall_total_min'].iloc[0] if len(row) and pd.notna(row['wall_total_min'].iloc[0]) else np.nan)
            offsets = x + (i - n_cfg / 2 + 0.5) * width
            bars = ax.bar(offsets, vals, width, label=label, color=color, edgecolor='black', linewidth=0.4)
            for bx, by in zip(offsets, vals):
                if not np.isnan(by):
                    ax.text(bx, by, fmt_time(by), ha='center', va='bottom', fontsize=7, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([LENGTH_LABEL[l] for l in LENGTHS])
        ax.set_xlabel('Sequence length')
        ax.set_ylabel('Total wall-clock time (min, log)')
        ax.set_yscale('log')
        ax.set_title(f'{dt} — Total wall-clock time')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4, which='both')
    axes[1].legend(loc='upper left', fontsize=8, ncol=2)
    fig.suptitle('IQ-TREE3 master (Clang build) — Cascade Lake (Xeon 8274, normal) vs Sapphire Rapids (Xeon 8470Q, normalsr) — total wall time (2026-05-10)')
    fig.tight_layout()
    add_footnote(fig)
    fig.savefig(os.path.join(OUT_DIR, 'fig01_wall_total.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_treesearch(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, dt in zip(axes, ['AA', 'DNA']):
        sub = df[df['data_type'] == dt]
        n_cfg = len(CONFIG_SPECS)
        x = np.arange(len(LENGTHS))
        width = 0.8 / n_cfg
        for i, (label, system, threads, color) in enumerate(CONFIG_SPECS):
            vals = []
            for length in LENGTHS:
                row = sub[(sub['system'] == system) & (sub['threads'] == threads) & (sub['length'] == length)]
                vals.append(row['wall_treesearch_min'].iloc[0] if len(row) and pd.notna(row['wall_treesearch_min'].iloc[0]) else np.nan)
            offsets = x + (i - n_cfg / 2 + 0.5) * width
            ax.bar(offsets, vals, width, label=label, color=color, edgecolor='black', linewidth=0.4)
            for bx, by in zip(offsets, vals):
                if not np.isnan(by):
                    ax.text(bx, by, fmt_time(by), ha='center', va='bottom', fontsize=7, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([LENGTH_LABEL[l] for l in LENGTHS])
        ax.set_xlabel('Sequence length')
        ax.set_ylabel('Tree-search wall-clock time (min, log)')
        ax.set_yscale('log')
        ax.set_title(f'{dt} — Tree-search wall time')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4, which='both')
    axes[1].legend(loc='upper left', fontsize=8, ncol=2)
    fig.suptitle('Tree-search wall time — Cascade Lake (8274) vs Sapphire Rapids (8470Q) per thread count')
    fig.tight_layout()
    add_footnote(fig)
    fig.savefig(os.path.join(OUT_DIR, 'fig02_treesearch.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_speedup_vs_1t(df):
    """Speedup vs same-system 1T baseline."""
    rows = []
    for system in SYSTEMS:
        for dt in ['AA', 'DNA']:
            for length in LENGTHS:
                base_row = df[(df['system'] == system) & (df['threads'] == 1) &
                              (df['data_type'] == dt) & (df['length'] == length)]
                if not len(base_row) or pd.isna(base_row['wall_total_sec'].iloc[0]):
                    continue
                base = base_row['wall_total_sec'].iloc[0]
                for thr in [1, 10, 48, 104]:
                    r = df[(df['system'] == system) & (df['threads'] == thr) &
                           (df['data_type'] == dt) & (df['length'] == length)]
                    if not len(r) or pd.isna(r['wall_total_sec'].iloc[0]):
                        continue
                    rows.append({
                        'system': system, 'data_type': dt, 'length': length,
                        'length_label': LENGTH_LABEL[length], 'threads': thr,
                        'speedup_total': base / r['wall_total_sec'].iloc[0],
                        'efficiency': (base / r['wall_total_sec'].iloc[0]) / thr,
                    })
    sp = pd.DataFrame(rows)
    sp.to_csv(os.path.join(OUT_DIR, 'speedup_vs_1t.csv'), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    color_sys = {'CLANG_MASTER': '#1565C0', 'NORMALSR_MASTER': '#E65100'}
    marker_len = {10000: 'o', 100000: 's', 1000000: '^'}
    for ax, dt in zip(axes, ['AA', 'DNA']):
        sub = sp[sp['data_type'] == dt]
        for system in SYSTEMS:
            for length in LENGTHS:
                s = sub[(sub['system'] == system) & (sub['length'] == length)].sort_values('threads')
                if not len(s):
                    continue
                ax.plot(s['threads'], s['speedup_total'],
                        marker=marker_len[length], color=color_sys[system],
                        label=f"{HW_LABEL[system]} {LENGTH_LABEL[length]}",
                        linewidth=1.5, markersize=8)
        max_thr = sub['threads'].max() if len(sub) else 104
        ax.plot([1, max_thr], [1, max_thr], 'k--', alpha=0.3, label='ideal')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Threads')
        ax.set_ylabel('Speedup vs 1T (same hardware)')
        ax.set_title(f'{dt} — OMP scaling')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
    fig.suptitle('OMP scaling vs same-hardware 1-thread baseline — CLX (8274) and SPR (8470Q)')
    fig.tight_layout()
    add_footnote(fig)
    fig.savefig(os.path.join(OUT_DIR, 'fig03_speedup_scaling.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return sp


def fig_normalsr_vs_clang(df):
    """Speedup of SPR (Sapphire Rapids) over CLX (Cascade Lake) at matched (threads, length, data_type)."""
    rows = []
    for dt in ['AA', 'DNA']:
        for length in LENGTHS:
            for thr in [1, 10, 48]:
                c = df[(df['system'] == 'CLANG_MASTER') & (df['threads'] == thr) &
                       (df['data_type'] == dt) & (df['length'] == length)]
                n = df[(df['system'] == 'NORMALSR_MASTER') & (df['threads'] == thr) &
                       (df['data_type'] == dt) & (df['length'] == length)]
                if not len(c) or not len(n):
                    continue
                cw = c['wall_total_sec'].iloc[0]
                nw = n['wall_total_sec'].iloc[0]
                cl = c['best_lnl'].iloc[0]
                nl = n['best_lnl'].iloc[0]
                if pd.isna(cw) or pd.isna(nw):
                    continue
                rows.append({
                    'data_type': dt, 'length': length, 'length_label': LENGTH_LABEL[length],
                    'threads': thr, 'clx_wall_sec': cw, 'spr_wall_sec': nw,
                    'speedup_spr_over_clx': cw / nw,
                    'clx_lnl': cl, 'spr_lnl': nl,
                    'lnl_diff': (nl - cl) if pd.notna(cl) and pd.notna(nl) else None,
                })
    cmp = pd.DataFrame(rows)
    cmp.to_csv(os.path.join(OUT_DIR, 'spr_vs_clx.csv'), index=False)
    if not len(cmp):
        return cmp

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, dt in zip(axes, ['AA', 'DNA']):
        sub = cmp[cmp['data_type'] == dt]
        x_labels = []
        vals = []
        colors = []
        thread_color = {1: '#90CAF9', 10: '#42A5F5', 48: '#1565C0'}
        for length in LENGTHS:
            for thr in [1, 10, 48]:
                r = sub[(sub['length'] == length) & (sub['threads'] == thr)]
                if not len(r):
                    continue
                x_labels.append(f"{LENGTH_LABEL[length]}\n{thr}T")
                vals.append(r['speedup_spr_over_clx'].iloc[0])
                colors.append(thread_color[thr])
        x = np.arange(len(x_labels))
        bars = ax.bar(x, vals, color=colors, edgecolor='black')
        ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
        for bx, by in zip(x, vals):
            ax.text(bx, by, f'{by:.2f}x', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel('SPR / CLX wall-time speedup')
        ax.set_title(f'{dt} — Sapphire Rapids vs Cascade Lake (same threads)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    fig.suptitle('Sapphire Rapids (Xeon 8470Q) vs Cascade Lake (Xeon 8274) — same thread count (>1 means SPR is faster)')
    fig.tight_layout()
    add_footnote(fig)
    fig.savefig(os.path.join(OUT_DIR, 'fig04_spr_vs_clx.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return cmp


def fig_lnl(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, dt in zip(axes, ['AA', 'DNA']):
        sub = df[(df['data_type'] == dt) & df['best_lnl'].notna()]
        # Use CLANG 1T as baseline if available, else min-config available
        base = {}
        for length in LENGTHS:
            base_row = sub[(sub['system'] == 'CLANG_MASTER') & (sub['threads'] == 1) & (sub['length'] == length)]
            if len(base_row):
                base[length] = base_row['best_lnl'].iloc[0]
        n_cfg = len(CONFIG_SPECS)
        x = np.arange(len(LENGTHS))
        width = 0.8 / n_cfg
        for i, (label, system, threads, color) in enumerate(CONFIG_SPECS):
            diffs = []
            for length in LENGTHS:
                r = sub[(sub['system'] == system) & (sub['threads'] == threads) & (sub['length'] == length)]
                if not len(r) or length not in base:
                    diffs.append(np.nan)
                else:
                    diffs.append(r['best_lnl'].iloc[0] - base[length])
            offsets = x + (i - n_cfg / 2 + 0.5) * width
            ax.bar(offsets, diffs, width, label=label, color=color, edgecolor='black', linewidth=0.4)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([LENGTH_LABEL[l] for l in LENGTHS])
        ax.set_xlabel('Sequence length')
        ax.set_ylabel('Δ log-likelihood vs CLX 1T')
        ax.set_title(f'{dt} — best LnL difference vs CLX 1T baseline')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    axes[1].legend(loc='upper right', fontsize=8, ncol=2)
    fig.suptitle('Correctness check — best log-likelihood difference vs Cascade Lake (CLX) 1T (same length)')
    fig.tight_layout()
    add_footnote(fig)
    fig.savefig(os.path.join(OUT_DIR, 'fig05_lnl_diff.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_all()
    df_sorted = df.sort_values(['data_type', 'length', 'system', 'threads'])
    cols = ['data_type', 'length', 'length_label', 'system', 'threads', 'model',
            'wall_total_sec', 'cpu_total_sec', 'wall_treesearch_sec', 'cpu_treesearch_sec',
            'wall_total_min', 'wall_treesearch_min',
            'best_lnl', 'iterations', 'completed', 'filename']
    df_sorted[cols].to_csv(os.path.join(OUT_DIR, 'all_results.csv'), index=False)

    print(f'Parsed {len(df)} log files. Completed: {df["completed"].sum()}, Incomplete: {(~df["completed"]).sum()}')
    print('\nIncomplete runs:')
    print(df[~df['completed']][['data_type', 'length', 'system', 'threads', 'filename']].to_string(index=False))

    fig_wall_total(df)
    fig_treesearch(df)
    sp = fig_speedup_vs_1t(df)
    cmp = fig_normalsr_vs_clang(df)
    fig_lnl(df)
    print('\nWrote figures and CSVs to', OUT_DIR)
    return df, sp, cmp


if __name__ == '__main__':
    main()
