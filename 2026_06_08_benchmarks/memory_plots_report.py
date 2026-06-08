"""Report-ready memory figures: 'fuel-gauge' style horizontal bars.

Each run is one row. The full bar = device capacity. Filled colour = used (peak)
or declared. LM_MEM_SAVE caps are drawn as a red dashed marker; hard failures
get a red ✗ badge. Aimed at slide / report use — readable in 2 seconds.

Writes:
  fig_report_gpu_vram.png    — GPU VRAM headroom (all GPU runs)
  fig_report_host_ram.png    — host RAM headroom (all runs, declared MF RAM)
  fig_report_regime.png      — regime heatmap (polished version of earlier figure)
"""
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, FancyBboxPatch
from matplotlib.colors import ListedColormap, BoundaryNorm

OUT_DIR = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_08_benchmarks'
CSV     = os.path.join(OUT_DIR, 'memory_per_log.csv')

df = pd.read_csv(CSV)
df['ninit_rank'] = df['ninit'].map({1: 0, 2: 1}).fillna(2)
df = (df.sort_values(['dt','length','backend','ninit_rank'])
        .drop_duplicates(['dt','length','backend'], keep='first')
        .copy())

COLOR = {'CLX':'#5DA5DA','SPR':'#0071C5','V100':'#F2A900','A100':'#7BAFD4','H200':'#76B900'}

def short_len(L):
    L = int(L)
    if L < 1_000_000: return f'{L//1000}K'
    Mn = L / 1_000_000
    return f'{Mn:.0f}M' if abs(Mn - round(Mn)) < 1e-9 else f'{Mn:g}M'

def cell_label(dt, L):
    return f'{dt} · {short_len(L)}'

# ============================================================================
# Figure 1 — GPU VRAM headroom ("fuel gauge")
# ============================================================================

GPU_ORDER = ['V100','A100','H200']
gpu = df[df['backend'].isin(GPU_ORDER) & df['gpu_total_MB'].notna()].copy()
gpu['gpu_total_GB'] = gpu['gpu_total_MB'] / 1024.0
gpu['vram_used_GB'] = gpu['gpu_used_MB'] / 1024.0
gpu['backend_rank'] = gpu['backend'].map({b:i for i,b in enumerate(GPU_ORDER)})
gpu = gpu.sort_values(['dt','length','backend_rank']).reset_index(drop=True)
gpu['row_label'] = gpu.apply(lambda r: cell_label(r['dt'], r['length']) + f"   {r['backend']}", axis=1)

n = len(gpu)
fig, ax = plt.subplots(figsize=(13, 0.42*n + 1.8))
max_cap = max(140, gpu['gpu_total_GB'].max() * 1.05)

bar_h = 0.62
for i, r in gpu.iterrows():
    y = n - 1 - i  # top-down
    cap = r['gpu_total_GB']
    used = r['vram_used_GB']
    lm_cap = r['lm_save_GB']
    has_vram_fail = pd.notna(r['vram_fail_GB'])

    # Capacity frame (light grey background)
    ax.add_patch(Rectangle((0, y - bar_h/2), cap, bar_h,
                            facecolor='#f1f1f1', edgecolor='#888', linewidth=0.7))
    # Used fill
    if pd.notna(used) and used > 0:
        ax.add_patch(Rectangle((0, y - bar_h/2), used, bar_h,
                                facecolor=COLOR[r['backend']], edgecolor='black',
                                linewidth=0.4, alpha=0.9))
    # LM_MEM_SAVE cap marker
    if pd.notna(lm_cap):
        ax.plot([lm_cap, lm_cap], [y - bar_h/2 - 0.05, y + bar_h/2 + 0.05],
                color='#c00', linewidth=1.6, linestyle='--', zorder=5)
        ax.text(lm_cap, y + bar_h/2 + 0.07, f'LM_SAVE {lm_cap:.0f}GB',
                color='#c00', fontsize=7, ha='center', va='bottom')

    # Right-side annotation
    pct_used  = (100.0 * used / cap) if pd.notna(used) and cap else None
    if has_vram_fail:
        tag = f'  ✗ needs {r["vram_fail_GB"]:.0f} GB, card only {cap:.0f} GB'
        tag_color = '#c00'; tag_weight = 'bold'
    elif pd.notna(used):
        tag = f'  used {used:.0f} GB / {cap:.0f} GB ({pct_used:.0f}%)'
        tag_color = 'black'; tag_weight = 'normal'
    elif pd.notna(lm_cap):
        tag = f'  no Energy block (run crashed/timeout); cap {lm_cap:.0f} GB / {cap:.0f} GB'
        tag_color = '#666'; tag_weight = 'normal'
    else:
        tag = f'  {cap:.0f} GB capacity, no usage recorded'
        tag_color = '#666'; tag_weight = 'normal'
    ax.text(max_cap, y, tag, fontsize=9, va='center', ha='left',
            color=tag_color, fontweight=tag_weight)

# Cell separators (between (dt, length) blocks)
seps = []
for i in range(1, n):
    if (gpu.loc[i,'dt'] != gpu.loc[i-1,'dt']) or (gpu.loc[i,'length'] != gpu.loc[i-1,'length']):
        seps.append(n - i - 0.5)
for s_y in seps:
    ax.axhline(s_y, color='#bbb', linewidth=0.5, linestyle=':')

ax.set_xlim(0, max_cap * 1.55)  # leave room for right-side text
ax.set_ylim(-0.7, n - 0.3)
ax.set_yticks([n - 1 - i for i in range(n)])
ax.set_yticklabels(gpu['row_label'], fontsize=9)
ax.set_xlabel('VRAM (GB) — full bar = card capacity, fill = peak used')
ax.set_title('GPU VRAM headroom per run\n'
             'Yellow = LM_MEM_SAVE triggered (1.5–3× slower per likelihood eval);   '
             'Red ✗ = VRAM-fail SIGABRT',
             fontsize=12)

# Hardware reference box
ref = ('Card capacities:\n'
       '   V100-SXM2-32GB:  31.7 GB\n'
       '   A100-SXM4-80GB:  79.4 GB\n'
       '   H200:                  140.1 GB')
ax.text(0.01, 1.02, ref, transform=ax.transAxes, fontsize=8, va='bottom', ha='left',
        bbox=dict(facecolor='#f8f8f8', edgecolor='#aaa', boxstyle='round,pad=0.3'))

ax.grid(False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', length=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_report_gpu_vram.png'), bbox_inches='tight', dpi=300)
plt.close(fig)
print('wrote fig_report_gpu_vram.png')

# ============================================================================
# Figure 2 — Host RAM headroom ("fuel gauge")
# ============================================================================

host = df[df['mf_ram_MB'].notna() & df['host_GB'].notna()].copy()
host['mf_ram_GB']  = host['mf_ram_MB']  / 1024.0
host['backend_rank'] = host['backend'].map({'CLX':0,'SPR':1,'V100':2,'A100':3,'H200':4})
host = host.sort_values(['dt','length','backend_rank']).reset_index(drop=True)
host['row_label'] = host.apply(lambda r: cell_label(r['dt'], r['length']) + f"   {r['backend']}", axis=1)

n = len(host)
fig, ax = plt.subplots(figsize=(13, 0.32*n + 1.8))
max_cap = max(host['host_GB'].max(), host['mf_ram_GB'].max()) * 1.05

bar_h = 0.62
for i, r in host.iterrows():
    y = n - 1 - i
    cap = r['host_GB']
    used = r['mf_ram_GB']
    has_host_fail = bool(r['host_hit'])

    # Capacity frame
    ax.add_patch(Rectangle((0, y - bar_h/2), cap, bar_h,
                            facecolor='#f1f1f1', edgecolor='#888', linewidth=0.7))
    # Declared use (clip to capacity for visual, annotate true value)
    visible = min(used, cap)
    if visible > 0:
        ax.add_patch(Rectangle((0, y - bar_h/2), visible, bar_h,
                                facecolor=COLOR[r['backend']], edgecolor='black',
                                linewidth=0.4, alpha=0.9))
    # Overflow visualisation
    if used > cap:
        ax.add_patch(Rectangle((cap, y - bar_h/2), used - cap, bar_h,
                                facecolor='none', edgecolor='#c00', linewidth=1.2,
                                hatch='///'))
    # -mem cap marker
    mem_arg = r['mem_arg'] if isinstance(r['mem_arg'], str) else ''
    if mem_arg:
        # parse 300G, 100M etc.
        try:
            v = mem_arg.upper()
            mult = 1.0
            if v.endswith('G'): v = float(v[:-1]); mult = 1.0
            elif v.endswith('M'): v = float(v[:-1]); mult = 1/1024
            elif v.endswith('T'): v = float(v[:-1]); mult = 1024
            else: v = float(v); mult = 1/1024
            mem_GB = v * mult
            ax.plot([mem_GB, mem_GB], [y - bar_h/2 - 0.05, y + bar_h/2 + 0.05],
                    color='#0a0', linewidth=1.6, linestyle='--', zorder=5)
            ax.text(mem_GB, y + bar_h/2 + 0.07, f'-mem {mem_arg}',
                    color='#0a0', fontsize=7, ha='center', va='bottom')
        except Exception:
            pass

    pct = 100.0 * used / cap if cap else None
    if has_host_fail:
        tag = f'  ✗ needs {used:.0f} GB, host has {cap:.0f} GB  ({pct:.0f}%)'
        col, w = '#c00', 'bold'
    elif pct is not None and pct > 70:
        tag = f'  {used:.0f} / {cap:.0f} GB  ({pct:.0f}%, tight)'
        col, w = '#a40', 'normal'
    else:
        tag = f'  {used:.0f} / {cap:.0f} GB  ({pct:.0f}%)'
        col, w = 'black', 'normal'
    ax.text(max_cap, y, tag, fontsize=9, va='center', ha='left',
            color=col, fontweight=w)

# Cell separators
seps = []
for i in range(1, n):
    if (host.loc[i,'dt'] != host.loc[i-1,'dt']) or (host.loc[i,'length'] != host.loc[i-1,'length']):
        seps.append(n - i - 0.5)
for s_y in seps:
    ax.axhline(s_y, color='#bbb', linewidth=0.5, linestyle=':')

ax.set_xlim(0, max_cap * 1.55)
ax.set_ylim(-0.7, n - 0.3)
ax.set_yticks([n - 1 - i for i in range(n)])
ax.set_yticklabels(host['row_label'], fontsize=8)
ax.set_xlabel('host RAM (GB) — full bar = node capacity, fill = declared ModelFinder peak')
ax.set_title('Host RAM headroom per run (declared ModelFinder peak vs node total)\n'
             'Red ✗ = SIGABRT (Memory required exceeds RAM);   '
             'Green = explicit -mem cap (used to rescue SPR AA 10M)',
             fontsize=12)

ref = ('Node host RAM:\n'
       '   CLX (gadi-cpu-clx):    188 GB\n'
       '   SPR (gadi-cpu-spr):    503 GB\n'
       '   V100 host:                       376 GB\n'
       '   H200 host:                     1 007 GB\n'
       '   A100 host (DGX):       2 015 GB')
ax.text(0.01, 1.02, ref, transform=ax.transAxes, fontsize=7.5, va='bottom', ha='left',
        bbox=dict(facecolor='#f8f8f8', edgecolor='#aaa', boxstyle='round,pad=0.3'))

ax.grid(False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', length=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_report_host_ram.png'), bbox_inches='tight', dpi=300)
plt.close(fig)
print('wrote fig_report_host_ram.png')

# ============================================================================
# Figure 3 — Memory regime heatmap (polished)
# ============================================================================

def regime(r):
    if pd.notna(r.get('vram_fail_GB')): return 3
    if r.get('host_hit'):               return 2
    if pd.notna(r.get('lm_save_GB')):   return 1
    return 0

df['regime'] = df.apply(regime, axis=1)
BACKEND_ORDER = ['CLX','SPR','V100','A100','H200']
LENGTHS = sorted(df['length'].dropna().unique())

row_labels, mat = [], []
for dt in ['AA','DNA']:
    for L in LENGTHS:
        sub = df[(df['dt'] == dt) & (df['length'] == L)]
        if sub.empty: continue
        row_labels.append(cell_label(dt, L))
        row = []
        for b in BACKEND_ORDER:
            cell = sub[sub['backend'] == b]
            row.append(np.nan if cell.empty else cell['regime'].iloc[0])
        mat.append(row)
mat = np.array(mat, dtype=float)

cmap = ListedColormap(['#f0f0f0', '#cdebcd', '#ffe98a', '#f3a3a3', '#8a2424'])
bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, cmap.N)
display = np.where(np.isnan(mat), -1, mat)

fig, ax = plt.subplots(figsize=(9.5, 0.55*len(row_labels)+1.4))
im = ax.imshow(display, cmap=cmap, norm=norm, aspect='auto')
ax.set_xticks(range(len(BACKEND_ORDER)))
ax.set_xticklabels(BACKEND_ORDER, fontsize=11)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=10)
ax.set_title('Memory regime at a glance\n'
             'each cell shows the worst memory state for that (datatype, length, backend)',
             fontsize=12)

LABELS = {-1: '–', 0: '✓ clean', 1: '⚠ LM_SAVE\n(slower)',
          2: '✗ host RAM\nSIGABRT', 3: '✗ VRAM\nSIGABRT'}
for i in range(display.shape[0]):
    for j in range(display.shape[1]):
        v = int(display[i, j])
        ax.text(j, i, LABELS[v], ha='center', va='center', fontsize=9,
                color=('white' if v == 3 else 'black'))

legend_handles = [
    Patch(facecolor='#f0f0f0', edgecolor='black', label='– not submitted'),
    Patch(facecolor='#cdebcd', edgecolor='black', label='✓ clean — no memory throttling'),
    Patch(facecolor='#ffe98a', edgecolor='black', label='⚠ LM_MEM_SAVE auto-cap (1.5–3× slower per LL eval)'),
    Patch(facecolor='#f3a3a3', edgecolor='black', label='✗ host RAM SIGABRT'),
    Patch(facecolor='#8a2424', edgecolor='black', label='✗ VRAM SIGABRT'),
]
ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.07),
          ncol=3, fontsize=9, frameon=True)
ax.tick_params(axis='both', length=0)
for spine in ax.spines.values(): spine.set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_report_regime.png'), bbox_inches='tight', dpi=300)
plt.close(fig)
print('wrote fig_report_regime.png')

print('\nAll report figures saved to', OUT_DIR)
