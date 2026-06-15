#!/usr/bin/env python3
# 2026-06-08 benchmarks — 5 backends, AA + DNA, 100K / 1M / 10M sites
# Plus an AA SPR-only scaling sweep at 2.5M / 3.125M / 3.75M / 5M sites
#
# Data:   /Users/u7826985/Projects/Nvidia/results/2026_06_08_benchmarks/{AA,DNA}/*.log
# Out:    /Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_08_benchmarks/
#
# Mirrors the 2026_05_30 analysis: 5 backends (CLX 48T, SPR 103T, OpenACC V100/A100/H200),
# AA (LG+I+G4) and DNA (GTR+I+G4), 100 taxa. Energy from IQ-TREE's built-in `Energy:`
# block (CPU = RAPL, GPU = NVML), reported here in Wh.
#
# Differences from 2026_05_30:
#   * AA/DNA logs live in two subdirectories, not the data root.
#   * No Linaro Forge perf_report files present — Forge vs IQ-TREE sections omitted.
#   * AA SPR has additional scaling-sweep runs (2.5M, 3.125M, 3.75M, 5M sites);
#     2.5M and 3.125M completed, 3.75M and 5M crashed (SIGABRT).
#   * `output_energy_profile_test_fix2_*` log present — ignored (no ninit prefix).

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DATA_ROOT = '/Users/u7826985/Projects/Nvidia/results/2026_06_08_benchmarks'
OUT_DIR   = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_08_benchmarks'

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 11})

COLORS = {
    'Clang (CLX CPU, OMP=48)':  '#5DA5DA',
    'Intel (SPR CPU, OMP=103)': '#0071C5',
    'OpenACC (V100)':           '#F2A900',
    'OpenACC (A100)':           '#7BAFD4',
    'OpenACC (H200)':           '#76B900',
}
BACKENDS = list(COLORS.keys())
BACKEND_SHORT = {
    'Clang (CLX CPU, OMP=48)':  'CLX',
    'Intel (SPR CPU, OMP=103)': 'SPR',
    'OpenACC (V100)':           'V100',
    'OpenACC (A100)':           'A100',
    'OpenACC (H200)':           'H200',
}

# ---------------------------------------------------------------------------
# 1. Log parsing
# ---------------------------------------------------------------------------

WALL_RE     = re.compile(r'Total wall-clock time used:\s+([\d.]+)\s+sec')
CUM_WALL_RE = re.compile(r'Total wall-clock time used \(including previous runs\):\s+([\d.]+)\s+sec')
CPU_RE      = re.compile(r'Total CPU time used:\s+([\d.]+)\s+sec')
TREE_WALL_RE = re.compile(r'Wall-clock time used for tree search:\s+([\d.]+)\s+sec')
MF_WALL_RE   = re.compile(r'Wall-clock time for ModelFinder:\s+([\d.]+)\s+seconds')
BEST_LL_RE   = re.compile(r'BEST SCORE FOUND\s*:\s+(-?[\d.]+)')
HOST_RE      = re.compile(r'^Host:\s+(.+)$', re.M)
GPU_RE       = re.compile(r'^GPU:\s+(.+)$',  re.M)
KERNEL_RE    = re.compile(r'^Kernel:\s+(.+)$', re.M)

EN_CPU_RE     = re.compile(r'Energy:\s*\n\s*CPU:\s+([\d.]+)\s+J', re.M)
EN_GPU_RE     = re.compile(r'GPU:\s+([\d.]+)\s+J', re.M)
EN_GPU_NA_RE  = re.compile(r'GPU:\s+not available', re.M)
EN_GPU_MEM_RE = re.compile(r'GPU mem:\s+([\d.]+)\s*/\s*([\d.]+)\s+MB.*?\+([\d.]+)\s+MB', re.M)

def classify_backend(fname):
    # Match explicit tokens — avoid substring traps like 'a100' in 'taxa100'.
    if 'OPENACC_A100' in fname: return 'OpenACC (A100)'
    if 'OPENACC_V100' in fname: return 'OpenACC (V100)'
    if 'OPENACC_H200' in fname: return 'OpenACC (H200)'
    if 'INTEL_VANILA_CLX_OMP48' in fname: return 'Clang (CLX CPU, OMP=48)'
    if 'INTEL_VANILA_OMP104'    in fname: return 'Intel (SPR CPU, OMP=103)'
    return 'unknown'

def classify_datatype(fname):
    if '_AA_'  in fname: return 'AA'
    if '_DNA_' in fname: return 'DNA'
    return 'unknown'

def classify_length(fname):
    m = re.search(r'_(\d+)len_', fname)
    return int(m.group(1)) if m else None

def classify_ninit(fname):
    # `output_benchmark_ninit_1_*`, `output_benchmark_ninit_2_with_energy_*`.
    # Stray `energy_profile_test_fix2_*` has no ninit_ token → returns None and is dropped.
    if 'ninit_1' in fname: return 1
    if 'ninit_2' in fname: return 2
    return None

def parse_log(path):
    # For checkpoint-resumed runs, IQ-TREE prints two wall-time lines:
    #   "Total wall-clock time used: ..."                       — current segment only
    #   "Total wall-clock time used (including previous runs)"  — cumulative across resumes
    # The `Energy:` block reports the current segment only (verified by comparing
    # current J / current wall to non-resumed runs — both give ~691 W on SPR).
    # We use the cumulative wall when present and scale the energy by
    # (cumulative / current) to estimate the true total energy. We flag the cell
    # as `resumed=True` so the figures/CSV are auditable.
    txt = open(path).read()
    out = {}
    m_wall_cur = WALL_RE.search(txt)
    m_wall_cum = CUM_WALL_RE.search(txt)
    cur_wall = float(m_wall_cur.group(1)) if m_wall_cur else np.nan
    cum_wall = float(m_wall_cum.group(1)) if m_wall_cum else np.nan
    out['wall_current_s'] = cur_wall
    out['wall_cum_s']     = cum_wall
    out['resumed']        = bool(m_wall_cum)
    out['wall_s']         = cum_wall if not np.isnan(cum_wall) else cur_wall
    m = CPU_RE.search(txt);      out['cpu_s']       = float(m.group(1)) if m else np.nan
    m = TREE_WALL_RE.search(txt);out['tree_wall_s'] = float(m.group(1)) if m else np.nan
    m = MF_WALL_RE.search(txt);  out['mf_wall_s']   = float(m.group(1)) if m else np.nan
    m = BEST_LL_RE.search(txt);  out['best_logL']   = float(m.group(1)) if m else np.nan
    m = HOST_RE.search(txt);     out['host']        = m.group(1).strip() if m else ''
    m = GPU_RE.search(txt);      out['gpu_str']     = m.group(1).strip() if m else ''
    m = KERNEL_RE.search(txt);   out['kernel']      = m.group(1).strip() if m else ''
    m = EN_CPU_RE.search(txt);   out['cpu_J']       = float(m.group(1)) if m else np.nan
    if EN_GPU_NA_RE.search(txt):
        out['gpu_J'] = 0.0
    else:
        m = EN_GPU_RE.search(txt); out['gpu_J'] = float(m.group(1)) if m else np.nan
    m = EN_GPU_MEM_RE.search(txt)
    out['gpu_mem_used_MB']  = float(m.group(1)) if m else np.nan
    out['gpu_mem_total_MB'] = float(m.group(2)) if m else np.nan
    out['gpu_mem_delta_MB'] = float(m.group(3)) if m else np.nan

    # Preserve as-parsed energy from the Energy: block (current segment only on
    # resumed runs), then scale to cumulative by wall-time ratio. Assumes
    # average power was approximately constant across resumed segments — true
    # for sustained CPU/GPU work but worth flagging.
    out['cpu_J_segment'] = out['cpu_J']
    out['gpu_J_segment'] = out['gpu_J']
    if out['resumed'] and not np.isnan(cur_wall) and cur_wall > 0 and not np.isnan(cum_wall):
        scale = cum_wall / cur_wall
        if not np.isnan(out['cpu_J']): out['cpu_J'] = out['cpu_J'] * scale
        if not np.isnan(out['gpu_J']): out['gpu_J'] = out['gpu_J'] * scale
        out['energy_scale'] = scale
    else:
        out['energy_scale'] = 1.0
    return out

rows = []
log_paths = sorted(glob.glob(os.path.join(DATA_ROOT, '**/*.log'), recursive=True))
for path in log_paths:
    fname    = os.path.basename(path)
    backend  = classify_backend(fname)
    datatype = classify_datatype(fname)
    length   = classify_length(fname)
    ninit    = classify_ninit(fname)
    if backend == 'unknown' or datatype == 'unknown' or length is None or ninit is None:
        continue
    rec = dict(file=fname, datatype=datatype, length=length,
               backend=backend, ninit=ninit, **parse_log(path))
    rows.append(rec)

raw = pd.DataFrame(rows)
raw['complete'] = raw['cpu_J'].notna() & raw['wall_s'].notna()
print(f'parsed {len(raw)} logs, {raw["complete"].sum()} complete (have Energy: block + wall time)')

# ---------------------------------------------------------------------------
# 2. Coverage matrix
# ---------------------------------------------------------------------------

CANON_LENGTHS = [100_000, 1_000_000, 10_000_000]   # used for the headline 6-cell grid
SCALING_LENGTHS = [2_500_000, 3_125_000, 3_750_000, 5_000_000]  # AA SPR scaling sweep

def coverage(df, ninit, lengths=None):
    sub = df[df['ninit'] == ninit]
    if lengths is not None:
        sub = sub[sub['length'].isin(lengths)]
    pv = sub.pivot_table(index=['datatype','length'], columns='backend',
                         values='complete', aggfunc='any').reindex(columns=BACKENDS)
    pv = pv.fillna(False).applymap(lambda x: 'done' if x else '–')
    return pv

print('\n=== ninit = 1  (canonical 100K/1M/10M cells) ===')
print(coverage(raw, 1, CANON_LENGTHS).to_string())
print('\n=== ninit = 2  (canonical 100K/1M/10M cells) ===')
print(coverage(raw, 2, CANON_LENGTHS).to_string())
print('\n=== ninit = 1  (AA SPR scaling sweep — extra lengths) ===')
print(coverage(raw, 1, SCALING_LENGTHS).to_string())

# ---------------------------------------------------------------------------
# 3. Summary table — one row per (datatype, length, backend), prefer ninit=1
# ---------------------------------------------------------------------------

good = raw[raw['complete']].copy()
good['ninit_rank'] = good['ninit'].map({1: 0, 2: 1})
summary = (good.sort_values(['datatype','length','backend','ninit_rank'])
                .drop_duplicates(['datatype','length','backend'], keep='first')
                .copy())

summary['cpu_Wh']    = summary['cpu_J'] / 3600.0
summary['gpu_Wh']    = summary['gpu_J'] / 3600.0
summary['energy_Wh'] = summary['cpu_Wh'] + summary['gpu_Wh']
summary['cpu_pct']   = 100.0 * summary['cpu_Wh'] / summary['energy_Wh']
summary['acc_pct']   = 100.0 * summary['gpu_Wh'] / summary['energy_Wh']
summary['avg_power_W'] = summary['energy_Wh'] * 3600.0 / summary['wall_s']

cols = ['datatype','length','backend','ninit',
        'wall_s','wall_current_s','wall_cum_s','resumed','energy_scale',
        'cpu_s','tree_wall_s','mf_wall_s',
        'energy_Wh','cpu_Wh','gpu_Wh','cpu_pct','acc_pct',
        'avg_power_W','best_logL',
        'gpu_mem_used_MB','gpu_mem_total_MB','gpu_mem_delta_MB',
        'gpu_str','host','kernel','file']
summary = summary[cols].sort_values(['datatype','length','backend']).reset_index(drop=True)
summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)

# ---------------------------------------------------------------------------
# 4. Speedup and energy-ratio tables
# ---------------------------------------------------------------------------

# Restrict ratios to the canonical 100K/1M/10M cells (the AA SPR scaling cells
# don't have GPU counterparts, so ratio rows would all be NaN there).
summary_canon = summary[summary['length'].isin(CANON_LENGTHS)]

pv = summary_canon.pivot_table(index=['datatype','length'], columns='backend',
                                values=['wall_s','energy_Wh','best_logL'])
SPR  = 'Intel (SPR CPU, OMP=103)'
CLX  = 'Clang (CLX CPU, OMP=48)'
A100 = 'OpenACC (A100)'
V100 = 'OpenACC (V100)'
H200 = 'OpenACC (H200)'

def col(metric, b):
    if (metric, b) in pv.columns: return pv[(metric, b)]
    return pd.Series(index=pv.index, dtype=float)

ratios = pd.DataFrame(index=pv.index)
for cpu_short, cpu in [('SPR', SPR), ('CLX', CLX)]:
    for gpu_short, gpu in [('A100', A100), ('V100', V100), ('H200', H200)]:
        ratios[f'speedup_{gpu_short}vs{cpu_short}']        = col('wall_s', cpu) / col('wall_s', gpu)
        ratios[f'energy_ratio_{cpu_short}over{gpu_short}'] = col('energy_Wh', cpu) / col('energy_Wh', gpu)
        ratios[f'logL_absdiff_{cpu_short}vs{gpu_short}']   = (col('best_logL', cpu) - col('best_logL', gpu)).abs()
ratios['speedup_SPRvsCLX']        = col('wall_s', CLX) / col('wall_s', SPR)
ratios['energy_ratio_CLXoverSPR'] = col('energy_Wh', CLX) / col('energy_Wh', SPR)
ratios.to_csv(os.path.join(OUT_DIR, 'ratios.csv'))

# ---------------------------------------------------------------------------
# 5. Grouped-bar runtime / energy / power figures
# ---------------------------------------------------------------------------

def fmt_hms(seconds):
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    if h: return f'{h}h {m}m'
    if m: return f'{m}m {s}s'
    return f'{s}s'

def grouped_bar(metric, ylabel, fname, log_y=True, label_fmt=None,
                df=None, lengths=None, title_extra=''):
    if df is None: df = summary
    if label_fmt is None: label_fmt = lambda v: f'{v:,.0f}'
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    n = len(BACKENDS); width = 0.8 / n
    offsets = [(i - (n - 1) / 2) * width for i in range(n)]
    for ax, dt in zip(axes, ['AA', 'DNA']):
        sub = df[df['datatype'] == dt]
        if lengths is not None: sub = sub[sub['length'].isin(lengths)]
        Ls = sorted(sub['length'].unique())
        x = np.arange(len(Ls))
        for i, backend in enumerate(BACKENDS):
            vals, positions = [], []
            for j, L in enumerate(Ls):
                row = sub[(sub['length'] == L) & (sub['backend'] == backend)]
                if row.empty: continue
                vals.append(row[metric].iloc[0]); positions.append(x[j] + offsets[i])
            if not vals: continue
            bars = ax.bar(positions, vals, width, label=BACKEND_SHORT[backend],
                          color=COLORS[backend], edgecolor='black', linewidth=0.4)
            for b, v in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, v, label_fmt(v),
                        ha='center', va='bottom', fontsize=7, rotation=0)
        ax.set_xticks(x); ax.set_xticklabels([f'{L:,}' for L in Ls])
        ax.set_xlabel('alignment length (sites)')
        ax.set_title(f'{dt} — {ylabel}{title_extra}')
        if log_y: ax.set_yscale('log')
        ax.grid(axis='y', linestyle=':', alpha=0.5)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc='upper left', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), bbox_inches='tight')
    plt.close(fig)

grouped_bar('wall_s',      'wall-clock runtime (s, log)', 'runtime_comparison.png',
            df=summary_canon, lengths=CANON_LENGTHS, label_fmt=fmt_hms)
grouped_bar('energy_Wh',   'total energy (Wh, log)',      'energy_comparison.png',
            df=summary_canon, lengths=CANON_LENGTHS, label_fmt=lambda v: f'{v:,.0f}')
grouped_bar('avg_power_W', 'average system power (W)',    'avg_power.png',
            df=summary_canon, lengths=CANON_LENGTHS, log_y=False,
            label_fmt=lambda v: f'{v:.0f}')

# ---------------------------------------------------------------------------
# 6. Stacked CPU vs GPU energy split + GPU-run 100% stack
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), sharey=True)
for ax, dt in zip(axes, ['AA', 'DNA']):
    sub = summary_canon[summary_canon['datatype'] == dt]
    labels, cpu_e, acc_e = [], [], []
    for L in sorted(sub['length'].unique()):
        for backend in BACKENDS:
            row = sub[(sub['length'] == L) & (sub['backend'] == backend)]
            if row.empty: continue
            r = row.iloc[0]
            short_L = f'{L//1000}k' if L < 1_000_000 else f'{L//1_000_000}M'
            labels.append(f'{short_L}\n{BACKEND_SHORT[backend]}')
            cpu_e.append(r['cpu_Wh']); acc_e.append(r['gpu_Wh'])
    x = np.arange(len(labels))
    ax.bar(x, cpu_e, color='#0071C5', label='CPU (host) energy', edgecolor='black', linewidth=0.4)
    ax.bar(x, acc_e, bottom=cpu_e, color='#76B900', label='GPU (accelerator) energy',
           edgecolor='black', linewidth=0.4)
    for xi, (c, a) in enumerate(zip(cpu_e, acc_e)):
        ax.text(xi, c+a, f'{c+a:,.0f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f'{dt} — energy breakdown')
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle=':', alpha=0.5)
axes[0].set_ylabel('energy (Wh, log)')
axes[0].legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'energy_breakdown.png'), bbox_inches='tight')
plt.close(fig)

gpu_share = summary_canon[summary_canon['backend'].isin(
    ['OpenACC (A100)','OpenACC (V100)','OpenACC (H200)'])].copy()
gpu_share = gpu_share.sort_values(['datatype','length','backend']).reset_index(drop=True)
gpu_share['label'] = (gpu_share['datatype'] + '\n'
                      + gpu_share['length'].apply(lambda L: f'{L//1000}k' if L < 1_000_000 else f'{L//1_000_000}M')
                      + '\n' + gpu_share['backend'].map(BACKEND_SHORT))

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(gpu_share))
ax.bar(x, gpu_share['cpu_pct'], color='#0071C5', label='host CPU share',  edgecolor='black', linewidth=0.4)
ax.bar(x, gpu_share['acc_pct'], bottom=gpu_share['cpu_pct'], color='#76B900',
       label='accelerator share', edgecolor='black', linewidth=0.4)
ax.axhline(50, color='red', linestyle='--', linewidth=0.8, label='50 % crossover')
for xi, (cpu, acc) in enumerate(zip(gpu_share['cpu_pct'], gpu_share['acc_pct'])):
    if cpu > 6: ax.text(xi, cpu/2, f'{cpu:.0f}%', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    if acc > 6: ax.text(xi, cpu + acc/2, f'{acc:.0f}%', ha='center', va='center', color='black', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(gpu_share['label'], fontsize=8)
ax.set_ylim(0, 105)
ax.set_ylabel('share of GPU-run energy (%)')
ax.set_title('Energy split per GPU run (100 % stack) — A100 vs V100 vs H200')
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cpu_acc_share.png'), bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Heatmaps — speedup, energy ratio, logL agreement
# ---------------------------------------------------------------------------

def heatmap(cols_prefix, exclude, vmin, vmax_floor, title, fname, cmap='RdYlGn', fmt='{:.2f}x', log10_color=False):
    sel = [c for c in ratios.columns if c.startswith(cols_prefix) and c != exclude]
    M = ratios[sel].copy()
    fig, ax = plt.subplots(figsize=(11, max(3, 0.6*len(M)+1)))
    if log10_color:
        plot_vals = np.where(M.values <= 0, 1e-6, M.values)
        im = ax.imshow(np.log10(plot_vals), cmap=cmap, aspect='auto')
    else:
        vmax = max(vmax_floor, np.nanmax(M.values)*1.05) if np.isfinite(np.nanmax(M.values)) else vmax_floor
        im = ax.imshow(M.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels([c.replace(cols_prefix,'') for c in sel], rotation=30, ha='right')
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([f'{d} {L:,}' for d, L in M.index])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isnan(v):
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=8,
                        color='gray' if log10_color else 'black')
            elif v == 0 and log10_color:
                ax.text(j, i, '0', ha='center', va='center', fontsize=8, color='white')
            else:
                txt = fmt.format(v) if not log10_color else f'{v:.3g}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                        color='white' if log10_color else 'black')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                 label='log10 |Δ logL|' if log10_color else ('speedup (x)' if 'speedup' in cols_prefix else 'energy ratio (x)'))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), bbox_inches='tight')
    plt.close(fig)

heatmap('speedup_',      'speedup_SPRvsCLX',       0, 2.5,
        'Speedup = CPU wall / GPU wall  (>1 = GPU is faster)',
        'speedup_heatmap.png')
heatmap('energy_ratio_', 'energy_ratio_CLXoverSPR', 0, 2.5,
        'Energy ratio = CPU Wh / GPU Wh  (>1 = GPU uses less energy)',
        'energy_ratio_heatmap.png')
heatmap('logL_absdiff_', '__none__', None, None,
        '|logL_CPU - logL_GPU|  (colour = log10)',
        'logL_agreement.png', cmap='magma_r', log10_color=True)

# ---------------------------------------------------------------------------
# 8. Throughput and energy per site
# ---------------------------------------------------------------------------

summary['energy_per_site_mWh'] = summary['energy_Wh'] * 1000.0 / summary['length']
summary['sites_per_sec']       = summary['length'] / summary['wall_s']

MARKERS = {'Clang (CLX CPU, OMP=48)': 'D',
           'Intel (SPR CPU, OMP=103)': 'o',
           'OpenACC (V100)': '^',
           'OpenACC (A100)': 'v',
           'OpenACC (H200)': 's'}

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, metric, ylabel in [
    (axes[0], 'energy_per_site_mWh', 'energy per site (mWh)'),
    (axes[1], 'sites_per_sec',       'throughput (sites / s)'),
]:
    for backend in BACKENDS:
        for dt, ls in [('AA','-'), ('DNA','--')]:
            sub = summary[(summary['backend']==backend) & (summary['datatype']==dt)].sort_values('length')
            if sub.empty: continue
            ax.plot(sub['length'], sub[metric], marker=MARKERS[backend], linestyle=ls,
                    color=COLORS[backend], label=f'{BACKEND_SHORT[backend]} — {dt}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('alignment length (sites, log)')
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'normalised_metrics.png'), bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------------------
# 9. ninit=1 vs ninit=2 sanity-check
# ---------------------------------------------------------------------------

good_pair = raw[raw['complete']].copy()
good_pair['energy_Wh'] = (good_pair['cpu_J'].fillna(0) + good_pair['gpu_J'].fillna(0)) / 3600.0
p_w = good_pair.pivot_table(index=['datatype','length','backend'], columns='ninit', values='wall_s')
p_e = good_pair.pivot_table(index=['datatype','length','backend'], columns='ninit', values='energy_Wh')
cmp = pd.DataFrame({
    'wall_s_n1':              p_w.get(1),
    'wall_s_n2':              p_w.get(2),
    'wall_ratio_n2_over_n1':  p_w.get(2) / p_w.get(1),
    'energy_Wh_n1':           p_e.get(1),
    'energy_Wh_n2':           p_e.get(2),
    'energy_ratio_n2_over_n1': p_e.get(2) / p_e.get(1),
})
cmp_complete = cmp.dropna(subset=['wall_s_n1','wall_s_n2'])
cmp_complete.to_csv(os.path.join(OUT_DIR, 'ninit1_vs_ninit2.csv'))

# ---------------------------------------------------------------------------
# 10. 5-panel phase-wall and energy figures, per ninit
# ---------------------------------------------------------------------------

PHASE_BACKENDS = [
    ('Clang (CLX CPU, OMP=48)',  'CLX 48T',  '#1565C0', '#000'),
    ('Intel (SPR CPU, OMP=103)', 'SPR 103T', '#E65100', '#000'),
    ('OpenACC (V100)',           'GPU V100', '#A5D6A7', '#1B5E20'),
    ('OpenACC (A100)',           'GPU A100', '#66BB6A', '#1B5E20'),
    ('OpenACC (H200)',           'GPU H200', '#2E7D32', '#1B5E20'),
]

CELLS_6 = [
    ('AA',  100000,   '100K'),
    ('AA',  1000000,  '1M'),
    ('AA',  10000000, '10M'),
    ('DNA', 100000,   '100K'),
    ('DNA', 1000000,  '1M'),
    ('DNA', 10000000, '10M'),
]
CELLS_5 = [c for c in CELLS_6 if not (c[0] == 'AA' and c[1] == 10000000)]

FOOTNOTE = ('Each panel has its own linear y-axis scale.   '
            'CLX = Cascade Lake Xeon 8274 (NCI `normal`, 48c).   '
            'SPR = Sapphire Rapids 8480+ (NCI `normalsr`, 104c, -nt 103).   '
            'GPU rows use the OpenACC build, host `-nt 1`.   '
            "Missing bar = run didn't complete (no Energy: block) or wasn't submitted.")

def fmt_min(m):
    if pd.isna(m) or m == 0: return ''
    if m < 1:   return f'{m*60:.0f}s'
    if m < 60:  return f'{m:.0f}m'
    h, mm = divmod(int(round(m)), 60)
    return f'{h}h{mm:02d}m' if mm else f'{h}h'

def get_phase(df_ninit, dt, length, backend_label, seconds_col):
    r = df_ninit[(df_ninit['datatype'] == dt)
                 & (df_ninit['length']   == length)
                 & (df_ninit['backend']  == backend_label)
                 & (df_ninit['complete'])]
    if r.empty: return np.nan
    secs = r[seconds_col].iloc[0]
    return np.nan if pd.isna(secs) else secs / 60.0

def render_phase_panel(df_ninit, seconds_col, title, out_name, ninit_label, cells):
    fig, axes = plt.subplots(1, len(cells), figsize=(4.3*len(cells), 6))
    if len(cells) == 1: axes = [axes]
    for ax, (dt, L, Llab) in zip(axes, cells):
        labels = [b[1] for b in PHASE_BACKENDS]
        colors = [b[2] for b in PHASE_BACKENDS]
        edges  = [b[3] for b in PHASE_BACKENDS]
        vals   = [get_phase(df_ninit, dt, L, b[0], seconds_col) for b in PHASE_BACKENDS]
        xp = np.arange(len(labels))
        for xi, v, col_, ed in zip(xp, vals, colors, edges):
            if np.isnan(v): continue
            ax.bar(xi, v, 0.78, color=col_, edgecolor=ed, linewidth=0.8)
            ax.text(xi, v, fmt_min(v), ha='center', va='bottom', fontsize=8, rotation=90)
        finite = [v for v in vals if not np.isnan(v)]
        if finite: ax.set_ylim(0, max(finite) * 1.25)
        else:
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, 'no completed runs', ha='center', va='center',
                    transform=ax.transAxes, color='#888', fontsize=10, style='italic')
        ax.set_xticks(xp); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('wall time (min)')
        ax.set_title(f'{dt} — {Llab}')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    legend_handles = [Patch(facecolor=b[2], edgecolor=b[3], linewidth=0.8, label=b[1])
                      for b in PHASE_BACKENDS]
    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.02), ncol=len(PHASE_BACKENDS),
               fontsize=9, frameon=True)
    fig.suptitle(f'{title}  —  ninit = {ninit_label}', fontsize=13)
    fig.tight_layout(rect=[0, 0.14, 1, 0.95])
    fig.text(0.5, 0.005, FOOTNOTE, ha='center', va='bottom', fontsize=7,
             style='italic', color='#444', wrap=True)
    fig.savefig(os.path.join(OUT_DIR, out_name), dpi=300, bbox_inches='tight')
    plt.close(fig)

PHASE_METRICS = [
    ('wall_s',      'Total wall-clock time (min) — per-cell linear scales', 'wall_total'),
    ('tree_wall_s', 'Tree-search wall time (min) — per-cell linear scales', 'wall_treesearch'),
    ('mf_wall_s',   'ModelFinder wall time (min) — per-cell linear scales', 'wall_modelfinder'),
]

for ninit_val in (1, 2):
    df_n = raw[raw['ninit'] == ninit_val].copy()
    for col_, title_, slug in PHASE_METRICS:
        render_phase_panel(df_n, col_, title_,
                           f'fig_{slug}_linear_5panel_ninit{ninit_val}.png',
                           ninit_label=ninit_val, cells=CELLS_6)
        render_phase_panel(df_n, col_, title_,
                           f'fig_{slug}_linear_5cell_ninit{ninit_val}.png',
                           ninit_label=ninit_val, cells=CELLS_5)

# ---------------------------------------------------------------------------
# 11. Energy 5-panel + 5-cell (total and stacked)
# ---------------------------------------------------------------------------

def fmt_wh(v):
    if pd.isna(v) or v == 0: return ''
    if v >= 1000: return f'{v/1000:.1f}kWh'
    if v >= 10:   return f'{v:.0f}Wh'
    return f'{v:.1f}Wh'

def get_energy(df_ninit, dt, length, backend_label):
    r = df_ninit[(df_ninit['datatype'] == dt)
                 & (df_ninit['length']   == length)
                 & (df_ninit['backend']  == backend_label)
                 & (df_ninit['complete'])]
    if r.empty: return (np.nan, np.nan)
    cpu_wh = r['cpu_J'].iloc[0] / 3600.0 if pd.notna(r['cpu_J'].iloc[0]) else np.nan
    gpu_wh = r['gpu_J'].iloc[0] / 3600.0 if pd.notna(r['gpu_J'].iloc[0]) else 0.0
    return cpu_wh, gpu_wh

def render_energy_panel(df_ninit, out_name, ninit_label, cells, stacked=False):
    fig, axes = plt.subplots(1, len(cells), figsize=(4.3*len(cells), 6))
    if len(cells) == 1: axes = [axes]
    for ax, (dt, L, Llab) in zip(axes, cells):
        labels = [b[1] for b in PHASE_BACKENDS]
        colors = [b[2] for b in PHASE_BACKENDS]
        edges  = [b[3] for b in PHASE_BACKENDS]
        cpu_vals, gpu_vals = [], []
        for b in PHASE_BACKENDS:
            c, g = get_energy(df_ninit, dt, L, b[0])
            cpu_vals.append(c); gpu_vals.append(g)
        xp = np.arange(len(labels))
        for xi, c, g, col_, ed in zip(xp, cpu_vals, gpu_vals, colors, edges):
            if np.isnan(c): continue
            total = c + g
            if stacked:
                ax.bar(xi, c, 0.78, color='#0071C5', edgecolor=ed, linewidth=0.8)
                if g > 0:
                    ax.bar(xi, g, 0.78, bottom=c, color=col_, edgecolor=ed, linewidth=0.8)
            else:
                ax.bar(xi, total, 0.78, color=col_, edgecolor=ed, linewidth=0.8)
            ax.text(xi, total, fmt_wh(total), ha='center', va='bottom', fontsize=8, rotation=90)
        totals = [c+g for c, g in zip(cpu_vals, gpu_vals) if not np.isnan(c)]
        if totals: ax.set_ylim(0, max(totals) * 1.25)
        else:
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, 'no completed runs', ha='center', va='center',
                    transform=ax.transAxes, color='#888', fontsize=10, style='italic')
        ax.set_xticks(xp); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('energy (Wh)')
        ax.set_title(f'{dt} — {Llab}')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    if stacked:
        legend_handles = [
            Patch(facecolor='#0071C5', edgecolor='#000', linewidth=0.8, label='CPU (host)'),
            Patch(facecolor='#76B900', edgecolor='#1B5E20', linewidth=0.8, label='GPU (accelerator)'),
        ]
        suptitle = f'Energy breakdown (Wh) — CPU host vs GPU stack  —  ninit = {ninit_label}'
    else:
        legend_handles = [Patch(facecolor=b[2], edgecolor=b[3], linewidth=0.8, label=b[1])
                          for b in PHASE_BACKENDS]
        suptitle = f'Total energy (Wh) — per-cell linear scales  —  ninit = {ninit_label}'

    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.02), ncol=len(legend_handles),
               fontsize=9, frameon=True)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0.14, 1, 0.95])
    fig.text(0.5, 0.005, FOOTNOTE, ha='center', va='bottom', fontsize=7,
             style='italic', color='#444', wrap=True)
    fig.savefig(os.path.join(OUT_DIR, out_name), dpi=300, bbox_inches='tight')
    plt.close(fig)

for ninit_val in (1, 2):
    df_n = raw[raw['ninit'] == ninit_val].copy()
    render_energy_panel(df_n, f'fig_energy_total_linear_5panel_ninit{ninit_val}.png',
                        ninit_label=ninit_val, cells=CELLS_6, stacked=False)
    render_energy_panel(df_n, f'fig_energy_stacked_linear_5panel_ninit{ninit_val}.png',
                        ninit_label=ninit_val, cells=CELLS_6, stacked=True)
    render_energy_panel(df_n, f'fig_energy_total_linear_5cell_ninit{ninit_val}.png',
                        ninit_label=ninit_val, cells=CELLS_5, stacked=False)
    render_energy_panel(df_n, f'fig_energy_stacked_linear_5cell_ninit{ninit_val}.png',
                        ninit_label=ninit_val, cells=CELLS_5, stacked=True)

# ---------------------------------------------------------------------------
# 11b. Per-datatype 3-panel linear variants
#      Same render functions as §10/§11, but each (datatype, ninit) gets its own
#      3-panel figure (100K, 1M, 10M). Useful for slides that focus on either AA
#      or DNA alone.
# ---------------------------------------------------------------------------

CELLS_AA  = [c for c in CELLS_6 if c[0] == 'AA']
CELLS_DNA = [c for c in CELLS_6 if c[0] == 'DNA']

for ninit_val in (1, 2):
    df_n = raw[raw['ninit'] == ninit_val].copy()
    for dt_tag, cells_dt in [('aa', CELLS_AA), ('dna', CELLS_DNA)]:
        # Phase walls — total / tree-search / ModelFinder
        for col_, title_, slug in PHASE_METRICS:
            render_phase_panel(df_n, col_, title_,
                               f'fig_{dt_tag}_{slug}_linear_3panel_ninit{ninit_val}.png',
                               ninit_label=ninit_val, cells=cells_dt)
        # Energy — total + stacked
        render_energy_panel(df_n, f'fig_{dt_tag}_energy_total_linear_3panel_ninit{ninit_val}.png',
                            ninit_label=ninit_val, cells=cells_dt, stacked=False)
        render_energy_panel(df_n, f'fig_{dt_tag}_energy_stacked_linear_3panel_ninit{ninit_val}.png',
                            ninit_label=ninit_val, cells=cells_dt, stacked=True)

# ---------------------------------------------------------------------------
# 12. AA SPR scaling sweep — wall and energy vs alignment length
# ---------------------------------------------------------------------------

SCAN_LENGTHS_ALL = sorted(set(CANON_LENGTHS) | set(SCALING_LENGTHS))  # 100K..10M
scan_ninit = 1
scan = raw[(raw['datatype'] == 'AA') & (raw['backend'] == SPR) & (raw['ninit'] == scan_ninit)].copy()
scan = scan[scan['length'].isin(SCAN_LENGTHS_ALL)].sort_values('length')
scan['energy_Wh'] = (scan['cpu_J'].fillna(0) + scan['gpu_J'].fillna(0)) / 3600.0

# CSV for the scaling sweep — useful for later regression / projection work.
scan_view = scan[['length','wall_s','wall_current_s','wall_cum_s','resumed','energy_scale',
                  'tree_wall_s','mf_wall_s','cpu_J','energy_Wh','best_logL','complete','file']]
scan_view.to_csv(os.path.join(OUT_DIR, 'aa_spr_scaling_sweep_ninit1.csv'), index=False)

if not scan.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    completed = scan[scan['complete']]
    crashed   = scan[~scan['complete']]

    axes[0].plot(completed['length'], completed['wall_s']/60.0, marker='o',
                 color=COLORS[SPR], linewidth=1.6, markersize=7, label='completed')
    for _, r in crashed.iterrows():
        axes[0].axvline(r['length'], color='#c00', linestyle=':', alpha=0.7)
        axes[0].text(r['length'], axes[0].get_ylim()[1]*0.05 if axes[0].get_ylim()[1] > 0 else 1,
                     ' crashed', color='#c00', fontsize=8, rotation=90, va='bottom')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('alignment length (sites, log)')
    axes[0].set_ylabel('total wall-clock (min)')
    axes[0].set_title('AA · SPR 103T · ninit=1 — wall vs length')
    axes[0].grid(True, which='both', linestyle=':', alpha=0.5)
    axes[0].legend(fontsize=8)

    axes[1].plot(completed['length'], completed['energy_Wh'], marker='o',
                 color=COLORS[SPR], linewidth=1.6, markersize=7, label='completed')
    for _, r in crashed.iterrows():
        axes[1].axvline(r['length'], color='#c00', linestyle=':', alpha=0.7)
        axes[1].text(r['length'], axes[1].get_ylim()[1]*0.05 if axes[1].get_ylim()[1] > 0 else 1,
                     ' crashed', color='#c00', fontsize=8, rotation=90, va='bottom')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('alignment length (sites, log)')
    axes[1].set_ylabel('total energy (Wh)')
    axes[1].set_title('AA · SPR 103T · ninit=1 — energy vs length')
    axes[1].grid(True, which='both', linestyle=':', alpha=0.5)
    axes[1].legend(fontsize=8)

    fig.suptitle('AA SPR scaling sweep — 100K to 10M sites (3.75M & 5M SIGABRT)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, 'aa_spr_scaling_sweep.png'), bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------
# 13. Console summary
# ---------------------------------------------------------------------------

print('\n==== speedups (canonical cells) ====')
for c in ['speedup_A100vsSPR','speedup_V100vsSPR','speedup_H200vsSPR',
         'speedup_A100vsCLX','speedup_V100vsCLX','speedup_H200vsCLX',
         'speedup_SPRvsCLX']:
    print(c); print(ratios[c].round(2).to_string()); print()

print('==== energy ratios (>1 ⇒ GPU uses less) ====')
for c in ['energy_ratio_SPRoverA100','energy_ratio_SPRoverV100','energy_ratio_SPRoverH200',
         'energy_ratio_CLXoverA100','energy_ratio_CLXoverV100','energy_ratio_CLXoverH200',
         'energy_ratio_CLXoverSPR']:
    print(c); print(ratios[c].round(2).to_string()); print()

print('\nWrote outputs to', OUT_DIR)
