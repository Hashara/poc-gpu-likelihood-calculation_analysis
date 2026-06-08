"""Memory figures from memory_per_log.csv.

Produces three figures in the analysis folder:

  fig_gpu_vram_usage.png   — GPU VRAM total / used / LM_MEM_SAVE cap per (dt, length, backend)
  fig_host_ram_usage.png   — declared ModelFinder RAM vs host total RAM per (dt, length, backend)
  fig_memory_regime.png    — heatmap of memory regime (clean / LM_SAVE / host_fail / vram_fail)

Run *after* mem_scan.py has written memory_per_log.csv into this folder.
"""
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT_DIR = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_08_benchmarks'
CSV     = os.path.join(OUT_DIR, 'memory_per_log.csv')

df = pd.read_csv(CSV)
# Prefer ninit=1 for the canonical view; if missing, fall back to ninit=2.
df['ninit_rank'] = df['ninit'].map({1: 0, 2: 1}).fillna(2)
df = (df.sort_values(['dt','length','backend','ninit_rank'])
        .drop_duplicates(['dt','length','backend'], keep='first')
        .copy())

# Hardware-block colours match the benchmark notebook.
COLOR = {'CLX':'#5DA5DA','SPR':'#0071C5','V100':'#F2A900','A100':'#7BAFD4','H200':'#76B900'}

# ---------------------------------------------------------------------------
# Figure 1 — GPU VRAM usage (total / used / LM_MEM_SAVE cap) per cell
# ---------------------------------------------------------------------------
GPU_BACKENDS = ['V100','A100','H200']
gpu = df[df['backend'].isin(GPU_BACKENDS) & df['gpu_total_MB'].notna()].copy()
gpu['gpu_total_GB'] = gpu['gpu_total_MB'] / 1024.0
gpu['vram_used_GB'] = gpu['gpu_used_MB'] / 1024.0

# Order: by datatype, length, then V100/A100/H200.
gpu['backend_rank'] = gpu['backend'].map({'V100':0,'A100':1,'H200':2})
gpu = gpu.sort_values(['dt','length','backend_rank']).reset_index(drop=True)

def short_len(L):
    L = int(L)
    if L < 1_000_000:
        return f'{L//1000}K'
    Mn = L / 1_000_000
    return f'{Mn:.0f}M' if abs(Mn - round(Mn)) < 1e-9 else f'{Mn:g}M'

gpu['label'] = gpu.apply(lambda r: f"{r['dt']}\n{short_len(r['length'])}\n{r['backend']}", axis=1)

fig, ax = plt.subplots(figsize=(15, 6.5))
x = np.arange(len(gpu))
width = 0.28

ax.bar(x - width, gpu['gpu_total_GB'], width, color='#e0e0e0',
       edgecolor='black', linewidth=0.6, label='total VRAM (card capacity)')
ax.bar(x,         gpu['vram_used_GB'], width,
       color=[COLOR[b] for b in gpu['backend']], edgecolor='black', linewidth=0.6,
       label='peak VRAM used  (end-of-run, completed only)')
ax.bar(x + width, gpu['lm_save_GB'].fillna(0), width,
       color='none', edgecolor='#c00', linewidth=1.4, hatch='///',
       label='LM_MEM_SAVE cap  (forced auto-cap)')

# Annotate bars
for i, r in gpu.iterrows():
    ax.text(i - width, r['gpu_total_GB'] + 2, f"{r['gpu_total_GB']:.0f}",
            ha='center', va='bottom', fontsize=7, color='#555')
    if pd.notna(r['vram_used_GB']):
        ax.text(i, r['vram_used_GB'] + 2, f"{r['vram_used_GB']:.0f}",
                ha='center', va='bottom', fontsize=7)
    if pd.notna(r['lm_save_GB']):
        ax.text(i + width, r['lm_save_GB'] + 2, f"{r['lm_save_GB']:.0f}",
                ha='center', va='bottom', fontsize=7, color='#c00')
    # VRAM hard-fail marker
    if pd.notna(r.get('vram_fail_GB')):
        ax.text(i, 150, f'✗ need {r["vram_fail_GB"]:.0f} GB',
                ha='center', va='top', fontsize=8, color='#c00', rotation=0,
                bbox=dict(facecolor='white', edgecolor='#c00', boxstyle='round,pad=0.2'))

ax.set_xticks(x)
ax.set_xticklabels(gpu['label'], fontsize=8)
ax.set_ylabel('VRAM (GB)')
ax.set_title('GPU VRAM — card capacity vs peak used vs LM_MEM_SAVE cap '
             '(ninit=1 preferred; ninit=2 used if ninit=1 missing)')
ax.set_ylim(0, max(160, gpu['gpu_total_GB'].max() * 1.15))
ax.grid(axis='y', linestyle=':', alpha=0.5)
ax.legend(loc='upper left', fontsize=9, framealpha=0.95)

# Vertical separators between (dt, length) cells
boundaries = []
for i in range(1, len(gpu)):
    if (gpu.loc[i,'dt'] != gpu.loc[i-1,'dt']) or (gpu.loc[i,'length'] != gpu.loc[i-1,'length']):
        boundaries.append(i - 0.5)
for b in boundaries:
    ax.axvline(b, color='#bbb', linewidth=0.6, linestyle=':')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_gpu_vram_usage.png'), bbox_inches='tight', dpi=300)
plt.close(fig)
print('wrote fig_gpu_vram_usage.png')

# ---------------------------------------------------------------------------
# Figure 2 — Host RAM: declared ModelFinder RAM vs host total
# ---------------------------------------------------------------------------
host = df[df['mf_ram_MB'].notna() & df['host_GB'].notna()].copy()
host['mf_ram_GB']  = host['mf_ram_MB']  / 1024.0
host['run_ram_GB'] = host['run_ram_MB'] / 1024.0
host['backend_rank'] = host['backend'].map({'CLX':0,'SPR':1,'V100':2,'A100':3,'H200':4})
host = host.sort_values(['dt','length','backend_rank']).reset_index(drop=True)
host['label'] = host.apply(lambda r: f"{r['dt']}\n{short_len(r['length'])}\n{r['backend']}", axis=1)

fig, ax = plt.subplots(figsize=(18, 6.5))
x = np.arange(len(host))
width = 0.30

ax.bar(x - width, host['host_GB'], width, color='#e0e0e0',
       edgecolor='black', linewidth=0.6, label='host RAM total')
ax.bar(x,         host['mf_ram_GB'], width,
       color=[COLOR[b] for b in host['backend']], edgecolor='black', linewidth=0.6,
       label='declared ModelFinder RAM')
ax.bar(x + width, host['run_ram_GB'].fillna(0), width,
       color='none', edgecolor='#444', linewidth=0.8, hatch='..',
       label='declared tree-search RAM')

for i, r in host.iterrows():
    ax.text(i - width, r['host_GB'] + 10, f"{r['host_GB']:.0f}",
            ha='center', va='bottom', fontsize=6, color='#555')
    if pd.notna(r['mf_ram_GB']):
        ax.text(i, r['mf_ram_GB'] + 10, f"{r['mf_ram_GB']:.0f}",
                ha='center', va='bottom', fontsize=6)
    if r.get('host_hit'):
        ax.text(i, r['mf_ram_GB'] + 30, '✗ host RAM\nexceeded',
                ha='center', va='bottom', fontsize=7, color='#c00',
                bbox=dict(facecolor='white', edgecolor='#c00', boxstyle='round,pad=0.2'))
    if isinstance(r.get('mem_arg'), str) and r['mem_arg']:
        ax.text(i, r['mf_ram_GB'] + 30, f'-mem {r["mem_arg"]}',
                ha='center', va='bottom', fontsize=7, color='#0a0',
                bbox=dict(facecolor='white', edgecolor='#0a0', boxstyle='round,pad=0.2'))

ax.set_xticks(x)
ax.set_xticklabels(host['label'], fontsize=7, rotation=0)
ax.set_ylabel('host RAM (GB)')
ax.set_title('Host RAM — node total vs declared peak (ModelFinder + tree-search)\n'
             "red ✗ = SIGABRT on 'Memory required exceeds your computer RAM size'   "
             "green = explicit -mem cap (only 2 SPR AA 10M runs)")
ax.grid(axis='y', linestyle=':', alpha=0.5)
ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax.set_yscale('log')
ax.set_ylim(0.5, max(host['host_GB'].max(), host['mf_ram_GB'].max()) * 1.8)

boundaries = []
for i in range(1, len(host)):
    if (host.loc[i,'dt'] != host.loc[i-1,'dt']) or (host.loc[i,'length'] != host.loc[i-1,'length']):
        boundaries.append(i - 0.5)
for b in boundaries:
    ax.axvline(b, color='#bbb', linewidth=0.6, linestyle=':')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_host_ram_usage.png'), bbox_inches='tight', dpi=300)
plt.close(fig)
print('wrote fig_host_ram_usage.png')

# ---------------------------------------------------------------------------
# Figure 3 — Memory regime heatmap
# ---------------------------------------------------------------------------
# Regime per (dt, length, backend):
#   0 = clean (no cap, no fail)
#   1 = LM_MEM_SAVE auto-cap   (degraded GPU perf)
#   2 = host_RAM SIGABRT
#   3 = VRAM SIGABRT
# Multiple flags can apply — encode by precedence: vram_fail > host_hit > lm_save > clean.

def regime(r):
    if pd.notna(r.get('vram_fail_GB')): return 3
    if r.get('host_hit'):               return 2
    if pd.notna(r.get('lm_save_GB')):   return 1
    return 0

df['regime'] = df.apply(regime, axis=1)

BACKEND_ORDER = ['CLX','SPR','V100','A100','H200']
LENGTHS       = sorted(df['length'].dropna().unique())
DTS           = ['AA','DNA']

# Build a grid (DT × length) rows × (backend) cols. AA SPR scaling sweep lengths
# (2.5M, 3.125M, 3.75M, 5M) only exist for AA SPR — they show up as a single
# coloured cell with the other backends as blanks.
row_labels, mat = [], []
for dt in DTS:
    for L in LENGTHS:
        sub = df[(df['dt'] == dt) & (df['length'] == L)]
        if sub.empty: continue
        row_labels.append(f'{dt}  {short_len(L)}')
        row = []
        for b in BACKEND_ORDER:
            cell = sub[sub['backend'] == b]
            if cell.empty: row.append(np.nan)
            else:          row.append(cell['regime'].iloc[0])
        mat.append(row)
mat = np.array(mat, dtype=float)

# Discrete colourmap: -1=missing, 0=clean, 1=LM_SAVE, 2=host_fail, 3=vram_fail
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap = ListedColormap(['#f0f0f0', '#d8f3d8', '#fff3a8', '#f3b0b0', '#9e3a3a'])
bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, cmap.N)
display = np.where(np.isnan(mat), -1, mat)

fig, ax = plt.subplots(figsize=(9, max(4, 0.45*len(row_labels)+1)))
im = ax.imshow(display, cmap=cmap, norm=norm, aspect='auto')
ax.set_xticks(range(len(BACKEND_ORDER)))
ax.set_xticklabels(BACKEND_ORDER)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=9)
ax.set_title('Memory regime per (datatype, length, backend) — ninit=1 preferred')

# Cell annotations
LABELS = {-1: 'n/a', 0: 'clean', 1: 'LM_SAVE', 2: 'host✗', 3: 'VRAM✗'}
for i in range(display.shape[0]):
    for j in range(display.shape[1]):
        v = int(display[i, j])
        ax.text(j, i, LABELS[v], ha='center', va='center', fontsize=8,
                color=('white' if v == 3 else 'black'))

legend_handles = [
    Patch(facecolor='#f0f0f0', edgecolor='black', label='n/a (not submitted)'),
    Patch(facecolor='#d8f3d8', edgecolor='black', label='clean'),
    Patch(facecolor='#fff3a8', edgecolor='black', label='LM_MEM_SAVE auto-cap (1.5–3× slower)'),
    Patch(facecolor='#f3b0b0', edgecolor='black', label='host RAM SIGABRT'),
    Patch(facecolor='#9e3a3a', edgecolor='black', label='VRAM SIGABRT'),
]
ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          ncol=3, fontsize=8, frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_memory_regime.png'), bbox_inches='tight', dpi=300)
plt.close(fig)
print('wrote fig_memory_regime.png')

print('\nAll memory figures saved to', OUT_DIR)
