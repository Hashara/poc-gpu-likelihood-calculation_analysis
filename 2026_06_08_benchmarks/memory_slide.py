"""One slide-ready figure summarising the memory situation.

2×3 panel grid (AA vs DNA × 100K / 1M / 10M sites). Each panel shows a small
fuel-gauge per backend: full bar = capacity (host RAM for CPU, VRAM for GPU),
filled portion = peak / declared use, traffic-light colour by utilisation,
inline % label, regime icon on the right (✓ / ⚠ LM_SAVE / ✗ fail).

Designed for one PowerPoint slide at 16:9. Output: fig_slide_memory.png.
"""
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

OUT_DIR = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_08_benchmarks'
CSV     = os.path.join(OUT_DIR, 'memory_per_log.csv')
df = pd.read_csv(CSV)

# Prefer ninit=1, fall back to ninit=2.
df['ninit_rank'] = df['ninit'].map({1: 0, 2: 1}).fillna(2)
df = (df.sort_values(['dt','length','backend','ninit_rank'])
        .drop_duplicates(['dt','length','backend'], keep='first')
        .reset_index(drop=True))

# Backend order (top to bottom in each panel) — CPUs first, then GPUs by size.
BACKENDS  = ['CLX','SPR','V100','A100','H200']
SHORT_DESC = {
    'CLX':  'CLX',
    'SPR':  'SPR',
    'V100': 'V100',
    'A100': 'A100',
    'H200': 'H200',
}

CELLS = [
    ('AA',  100_000),  ('AA',  1_000_000),  ('AA',  10_000_000),
    ('DNA', 100_000),  ('DNA', 1_000_000),  ('DNA', 10_000_000),
]

# Cells to render as "cap N GB; no usage recorded" instead of the SIGABRT red
# treatment. Used for crashed runs where the host-RAM SIGABRT was caused by
# ModelFinder's *declared* requirement exceeding the node — IQ-TREE never wrote
# an Energy block, so peak actual usage is unknown.
NO_USAGE_OVERRIDE = {('DNA', 10_000_000, 'CLX')}

# Per-cell, per-backend memory situation packaged for plotting.
def lookup(dt, L, b):
    sub = df[(df['dt'] == dt) & (df['length'] == L) & (df['backend'] == b)]
    if sub.empty: return None
    r = sub.iloc[0]
    is_gpu = b in ('V100','A100','H200')
    if is_gpu:
        cap = r['gpu_total_MB']/1024.0 if pd.notna(r['gpu_total_MB']) else None
        used = r['gpu_used_MB']/1024.0 if pd.notna(r['gpu_used_MB']) else None
        cap_kind = 'VRAM'
    else:
        cap = float(r['host_GB']) if pd.notna(r['host_GB']) else None
        used = r['mf_ram_MB']/1024.0 if pd.notna(r['mf_ram_MB']) else None
        cap_kind = 'HOST'
    pct = (100.0 * used / cap) if (used is not None and cap) else None
    # Regime
    if pd.notna(r.get('vram_fail_GB')):    regime = 'vram_fail'
    elif bool(r.get('host_hit')):          regime = 'host_fail'
    elif pd.notna(r.get('lm_save_GB')):    regime = 'lm_save'
    else:                                   regime = 'clean'
    # Override: render selected cells with the "no usage recorded" treatment.
    if (dt, L, b) in NO_USAGE_OVERRIDE:
        regime = 'no_usage'
        used = None
        pct = None
    return dict(cap=cap, used=used, pct=pct, cap_kind=cap_kind, regime=regime,
                lm_save_GB=r.get('lm_save_GB'),
                vram_fail_GB=r.get('vram_fail_GB'),
                mem_arg=r.get('mem_arg') if isinstance(r.get('mem_arg'), str) else '')

# Traffic-light colour by utilisation
def util_color(pct, regime):
    if regime in ('host_fail','vram_fail'): return '#c0392b'   # red
    if regime == 'lm_save':                 return '#e67e22'   # orange
    if pct is None:                         return '#bdc3c7'   # grey
    if pct >= 90:                           return '#e67e22'   # orange
    if pct >= 50:                           return '#f1c40f'   # amber
    return '#27ae60'                                            # green

def regime_icon(regime):
    return {'clean':'✓','lm_save':'⚠','host_fail':'✗','vram_fail':'✗'}.get(regime, '')

def short_len(L):
    return f'{L//1000} K sites' if L < 1_000_000 else f'{L//1_000_000} M sites'

# ---------------------------------------------------------------------------
# Build the figure — 2 rows × 3 cols
# ---------------------------------------------------------------------------
plt.rcParams.update({'font.size': 12, 'savefig.dpi': 300})

# Each panel uses x-range [-45, 100]: the strip x∈[-45, -3] holds the backend
# label, x∈[0, 100] is the actual capacity bar (% used), and trailing text
# (used/cap, regime icon, -mem/LM_SAVE notes) is drawn at x∈[102, ~145] inside
# axis coords. Per-panel widths in the figure are picked so trailing text fits.
LEFT_LABEL_X  = -18   # left edge of backend label strip ("V100" fits easily)
BAR_START_X   = 0
TRAIL_TEXT_X  = 103   # where trailing text starts
PANEL_RIGHT_X = 230   # right edge of axis (room for trailing notes)

fig, axes = plt.subplots(2, 3, figsize=(24, 11),
                         gridspec_kw={'wspace': 0.06, 'hspace': 0.40})

for ax, (dt, L) in zip(axes.flat, CELLS):
    ax.set_xlim(LEFT_LABEL_X, PANEL_RIGHT_X)
    ax.set_ylim(-0.6, len(BACKENDS) - 0.4)
    ax.invert_yaxis()
    ax.set_title(f'{dt} · {short_len(L)}', fontsize=16, fontweight='bold', pad=8)

    # Light divider between label strip and bars
    ax.axvline(BAR_START_X - 2, color='#ddd', linewidth=0.6)

    for i, b in enumerate(BACKENDS):
        info = lookup(dt, L, b)
        # Backend label on the left
        ax.text(LEFT_LABEL_X + 1, i, SHORT_DESC[b], ha='left', va='center', fontsize=12,
                fontweight='bold', color='#222')
        if info is None:
            # Not submitted
            ax.add_patch(Rectangle((BAR_START_X, i-0.32), 100, 0.64,
                                    facecolor='#ecf0f1', edgecolor='#bdc3c7',
                                    linewidth=0.8, linestyle=':'))
            ax.text(BAR_START_X + 50, i, 'not submitted', ha='center', va='center',
                    fontsize=10, color='#7f8c8d', style='italic')
            continue

        cap, used, pct, regime = info['cap'], info['used'], info['pct'], info['regime']
        col = util_color(pct, regime)

        # Capacity track
        ax.add_patch(Rectangle((BAR_START_X, i-0.32), 100, 0.64,
                                facecolor='#f4f4f4', edgecolor='#999', linewidth=0.8))
        # Fill (clipped to 100 visually; >100 case drawn as red hatched overflow)
        if pct is not None:
            ax.add_patch(Rectangle((BAR_START_X, i-0.32), min(pct, 100), 0.64,
                                    facecolor=col, edgecolor='black', linewidth=0.4))
            if pct > 100:
                # overflow band over the right side
                ax.add_patch(Rectangle((BAR_START_X + 90, i-0.32), 10, 0.64, facecolor='none',
                                        edgecolor='#c0392b', linewidth=1.5, hatch='///'))

        # LM_MEM_SAVE cap line (as a vertical inside the bar)
        if pd.notna(info['lm_save_GB']) and cap:
            cap_pct = 100.0 * info['lm_save_GB'] / cap
            ax.plot([BAR_START_X + cap_pct, BAR_START_X + cap_pct], [i-0.35, i+0.35],
                    color='#c0392b', linewidth=2.2, linestyle='--', zorder=5)

        # Inline numeric label
        if regime == 'vram_fail':
            txt = f'  ✗ needs {info["vram_fail_GB"]:.0f} GB, card only {cap:.0f} GB'
        elif regime == 'host_fail':
            txt = f'  ✗ needs {used:.0f} GB, node has {cap:.0f} GB  ({pct:.0f}%)'
        elif pct is None:
            txt = f'  cap {cap:.0f} GB; no usage recorded'
        else:
            extra = []
            if regime == 'lm_save': extra.append('LM_SAVE')
            if info['mem_arg']:     extra.append(f'-mem {info["mem_arg"]}')
            tail = ('  · ' + ' · '.join(extra)) if extra else ''
            txt = f'  {used:.0f}/{cap:.0f} GB  ({pct:.0f}%){tail}'
        ax.text(TRAIL_TEXT_X, i, txt, ha='left', va='center', fontsize=11.5,
                color=('#c0392b' if regime in ('host_fail','vram_fail') else 'black'),
                fontweight=('bold' if regime in ('host_fail','vram_fail') else 'normal'))

        # Regime icon at the right edge of the filled portion of the bar
        ic = regime_icon(regime)
        if ic:
            ic_col = {'clean':'#27ae60','lm_save':'#e67e22',
                       'host_fail':'#c0392b','vram_fail':'#c0392b'}[regime]
            # Icon sits at the right edge of the filled portion, but never closer
            # than +6 to the bar start (otherwise it lands on the backend label
            # when the bar is nearly empty).
            ic_x = BAR_START_X + max(8, min(pct or 0, 100) - 3)
            ax.text(ic_x, i, ic, ha='center', va='center',
                    fontsize=13, color='white', fontweight='bold',
                    bbox=dict(facecolor=ic_col, edgecolor='none', boxstyle='circle,pad=0.18'))

    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0%','25%','50%','75%','100%'], fontsize=9, color='#444')
    ax.set_yticks([])
    for spine in ['top','right','left']: ax.spines[spine].set_visible(False)
    ax.tick_params(axis='x', length=0)

# Suptitle + legend
fig.suptitle('Memory headroom across the benchmark grid\n'
             'each bar = one backend; full bar = capacity (host RAM for CPU, VRAM for GPU); '
             'fill = peak used (or declared for CPU)',
             fontsize=15, y=0.99)

legend = [
    Patch(facecolor='#27ae60', edgecolor='black', label='✓ clean  (<50%)'),
    Patch(facecolor='#f1c40f', edgecolor='black', label='moderate  (50–90%)'),
    Patch(facecolor='#e67e22', edgecolor='black', label='⚠ tight / LM_MEM_SAVE  (1.5–3× slower)'),
    Patch(facecolor='#c0392b', edgecolor='black', label='✗ SIGABRT  (host RAM or VRAM exceeded)'),
    Patch(facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=0.8, label='not submitted'),
]
fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.5, 0.005),
           ncol=5, fontsize=12, frameon=True)

fig.text(0.5, 0.045,
         'capacities — CLX 188 GB · SPR 503 GB · V100 32 GB VRAM · A100 80 GB VRAM · H200 140 GB VRAM    '
         '|    red dashed line inside a bar = LM_MEM_SAVE cap forced by IQ-TREE at runtime',
         ha='center', fontsize=10, color='#555', style='italic')

plt.subplots_adjust(left=0.025, right=0.995, top=0.90, bottom=0.11)
plt.savefig(os.path.join(OUT_DIR, 'fig_slide_memory.png'), bbox_inches='tight', dpi=300,
            facecolor='white')
plt.close(fig)
print('wrote fig_slide_memory.png')
