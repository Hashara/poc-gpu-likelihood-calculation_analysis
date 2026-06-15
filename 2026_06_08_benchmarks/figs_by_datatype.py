"""Per-datatype (AA-only / DNA-only) versions of the key benchmark figures.

The notebook's existing figures put AA and DNA side-by-side as two panels in one
PNG. For slide use it's often cleaner to drop just the AA or just the DNA story.

Reads:  summary.csv, ratios.csv  (both produced by analysis.py)
Writes (300 DPI):
  fig_aa_runtime.png          fig_dna_runtime.png
  fig_aa_energy.png           fig_dna_energy.png
  fig_aa_avg_power.png        fig_dna_avg_power.png
  fig_aa_energy_breakdown.png fig_dna_energy_breakdown.png
  fig_aa_speedup_heatmap.png  fig_dna_speedup_heatmap.png
  fig_aa_energy_ratio.png     fig_dna_energy_ratio.png
  fig_aa_logL_agreement.png   fig_dna_logL_agreement.png
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

OUT_DIR = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_08_benchmarks'
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 11})

summary = pd.read_csv(os.path.join(OUT_DIR, 'summary.csv'))
ratios  = pd.read_csv(os.path.join(OUT_DIR, 'ratios.csv'))

# Match analysis.py styling
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
CANON_LENGTHS = [100_000, 1_000_000, 10_000_000]
summary_canon = summary[summary['length'].isin(CANON_LENGTHS)].copy()

def fmt_hms(seconds):
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h: return f'{h}h {m}m'
    if m: return f'{m}m {s}s'
    return f'{s}s'

# ---------------------------------------------------------------------------
# Grouped-bar metrics (runtime / energy / avg power) — one PNG per datatype
# ---------------------------------------------------------------------------

def grouped_bar_single(dt, metric, ylabel, fname, log_y=True, label_fmt=None):
    if label_fmt is None: label_fmt = lambda v: f'{v:,.0f}'
    sub = summary_canon[summary_canon['datatype'] == dt]
    Ls = sorted(sub['length'].unique())
    x = np.arange(len(Ls))
    n = len(BACKENDS); width = 0.8 / n
    offsets = [(i - (n - 1) / 2) * width for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 6))
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
            ax.text(b.get_x() + b.get_width()/2, v, label_fmt(v),
                    ha='center', va='bottom', fontsize=8, rotation=0)
    ax.set_xticks(x); ax.set_xticklabels([f'{L:,}' for L in Ls])
    ax.set_xlabel('alignment length (sites)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{dt} — {ylabel}')
    if log_y: ax.set_yscale('log')
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), bbox_inches='tight')
    plt.close(fig)

for dt in ('AA', 'DNA'):
    tag = dt.lower()
    grouped_bar_single(dt, 'wall_s',      'wall-clock runtime (s, log)',
                        f'fig_{tag}_runtime.png',     label_fmt=fmt_hms)
    grouped_bar_single(dt, 'energy_Wh',   'total energy (Wh, log)',
                        f'fig_{tag}_energy.png',      label_fmt=lambda v: f'{v:,.0f}')
    grouped_bar_single(dt, 'avg_power_W', 'average system power (W)',
                        f'fig_{tag}_avg_power.png',   log_y=False,
                        label_fmt=lambda v: f'{v:.0f}')

# ---------------------------------------------------------------------------
# Stacked CPU vs GPU energy breakdown — one PNG per datatype
# ---------------------------------------------------------------------------

def energy_breakdown_single(dt, fname):
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
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, cpu_e, color='#0071C5', label='CPU (host) energy',
           edgecolor='black', linewidth=0.4)
    ax.bar(x, acc_e, bottom=cpu_e, color='#76B900',
           label='GPU (accelerator) energy', edgecolor='black', linewidth=0.4)
    for xi, (c, a) in enumerate(zip(cpu_e, acc_e)):
        ax.text(xi, c+a, f'{c+a:,.0f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('energy (Wh, log)')
    ax.set_yscale('log')
    ax.set_title(f'{dt} — energy breakdown (host CPU vs accelerator)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), bbox_inches='tight')
    plt.close(fig)

for dt in ('AA', 'DNA'):
    energy_breakdown_single(dt, f'fig_{dt.lower()}_energy_breakdown.png')

# ---------------------------------------------------------------------------
# Heatmaps — speedup / energy ratio / logL — split per datatype
# ---------------------------------------------------------------------------

def heatmap_single(dt, prefix, exclude, fname, title, cmap='RdYlGn',
                   fmt='{:.2f}x', log10_color=False, cbar_label=None):
    sel_cols = [c for c in ratios.columns if c.startswith(prefix) and c != exclude]
    sub = ratios[ratios['datatype'] == dt][['length'] + sel_cols].set_index('length')
    if sub.empty: return
    M = sub.values.astype(float)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.6*M.shape[0]+1.5)))
    if log10_color:
        plot_vals = np.where(M <= 0, 1e-6, M)
        im = ax.imshow(np.log10(plot_vals), cmap=cmap, aspect='auto')
    else:
        vmax = max(2.5, np.nanmax(M)*1.05) if np.isfinite(np.nanmax(M)) else 2.5
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(sel_cols)))
    ax.set_xticklabels([c.replace(prefix, '') for c in sel_cols],
                        rotation=30, ha='right')
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([f'{int(L):,}' for L in sub.index])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=9,
                        color='gray' if log10_color else 'black')
            elif v == 0 and log10_color:
                ax.text(j, i, '0', ha='center', va='center', fontsize=9, color='white')
            else:
                txt = fmt.format(v) if not log10_color else f'{v:.3g}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                        color=('white' if log10_color else 'black'))
    ax.set_title(f'{dt} — {title}')
    ax.set_ylabel('alignment length (sites)')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=cbar_label or '')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), bbox_inches='tight')
    plt.close(fig)

for dt in ('AA', 'DNA'):
    tag = dt.lower()
    heatmap_single(dt, 'speedup_',      'speedup_SPRvsCLX',
                   f'fig_{tag}_speedup_heatmap.png',
                   'Speedup = CPU wall / GPU wall  (>1 = GPU is faster)',
                   cbar_label='speedup (×)')
    heatmap_single(dt, 'energy_ratio_', 'energy_ratio_CLXoverSPR',
                   f'fig_{tag}_energy_ratio.png',
                   'Energy ratio = CPU Wh / GPU Wh  (>1 = GPU uses less)',
                   cbar_label='energy ratio (×)')
    heatmap_single(dt, 'logL_absdiff_', '__none__',
                   f'fig_{tag}_logL_agreement.png',
                   '|logL_CPU − logL_GPU|  (colour = log10)',
                   cmap='magma_r', log10_color=True, cbar_label='log10 |Δ logL|')

print('Wrote per-datatype figures to', OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith('fig_aa_') or f.startswith('fig_dna_'):
        print(' ', f)
