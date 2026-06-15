"""AA · SPR · ninit=1 scaling sweep — intermediate (non-power-of-10) lengths.

The 2026_06_08 benchmark batch added four AA SPR (Sapphire Rapids, OMP=103) runs
at alignment lengths that fall between the canonical 100K / 1M / 10M points:

    2 500 000  · completed
    3 125 000  · completed
    3 750 000  · SIGABRT — host RAM
    5 000 000  · SIGABRT — host RAM

Together with the canonical anchors (100K, 1M, 10M) they form a 7-point
SPR-only scaling sweep on the LG+I+G4 protein model.

This script:

  1. Parses every relevant log (always using the 06_08 copy since 3.125M only
     completed there; the 05_30 copy is identical for the other lengths).
  2. Captures cumulative wall + scaled energy for the AA SPR 10M cell, which
     was checkpoint-resumed (see 2026_06_08_benchmarks/analysis.py).
  3. Fits power-law scaling models:
        wall_s         ~ a * L^k_wall
        cpu_J          ~ a * L^k_energy
        ModelFinder_MB ~ a * L^k_mf      (linear in L by construction)
  4. Projects ModelFinder's host-RAM demand against the SPR 503 GB ceiling to
     identify where it crashes; predicts the -mem cap that would have rescued
     the 3.75M and 5M runs (the cap that worked for 10M was -mem 300G).
  5. Plots and CSVs go in this folder.
"""
import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_08_benchmarks/aa_spr_scaling_intermediate_lengths'

# Always use the 06_08 batch copy of each log (3.125M completed only there;
# the rest of the lengths have byte-identical 05_30 copies).
DATA_DIR = '/Users/u7826985/Projects/Nvidia/results/2026_06_08_benchmarks/AA'
LENGTHS  = [100_000, 1_000_000, 2_500_000, 3_125_000, 3_750_000, 5_000_000, 10_000_000]

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 11})

# ---------------------------------------------------------------------------
# 1. Parse logs
# ---------------------------------------------------------------------------

WALL_RE      = re.compile(r'Total wall-clock time used:\s+([\d.]+)\s+sec')
CUM_WALL_RE  = re.compile(r'Total wall-clock time used \(including previous runs\):\s+([\d.]+)\s+sec')
CPU_TIME_RE  = re.compile(r'Total CPU time used:\s+([\d.]+)\s+sec')
TREE_WALL_RE = re.compile(r'Wall-clock time used for tree search:\s+([\d.]+)\s+sec')
MF_WALL_RE   = re.compile(r'Wall-clock time for ModelFinder:\s+([\d.]+)\s+seconds')
BEST_LL_RE   = re.compile(r'BEST SCORE FOUND\s*:\s+(-?[\d.]+)')
CPU_J_RE     = re.compile(r'Energy:\s*\n\s*CPU:\s+([\d.]+)\s+J', re.M)
HOST_GB_RE   = re.compile(r'^Host:\s+\S+\s+\([^,]+,\s+[^,]+,\s+(\d+)\s+GB RAM\)', re.M)
MF_RAM_RE    = re.compile(r'NOTE:\s+ModelFinder requires\s+(\d+)\s+MB RAM', re.M)
EXC_RAM_RE   = re.compile(r'Memory required exceeds your computer RAM size', re.M)
CMD_MEM_RE   = re.compile(r'-mem\s+(\S+)')

def parse(L):
    pattern = f'output_benchmark_ninit_1_AA_LG+I+G4_OMP_104_taxa100_run1_tree_1_{L}_*_{L}len_ninit1_aa.log'
    candidates = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    assert candidates, f'no log for length {L}'
    path = candidates[0]
    txt = open(path).read()
    def f(rx, cast=float):
        m = rx.search(txt); return cast(m.group(1)) if m else None
    cur_wall = f(WALL_RE)
    cum_wall = f(CUM_WALL_RE)
    cpu_J    = f(CPU_J_RE)
    # For resumed runs (10M only) IQ-TREE's Energy: block reports the final
    # segment only. Scale by wall ratio to estimate cumulative energy.
    cpu_J_scaled = cpu_J
    scale = 1.0
    if cum_wall and cur_wall and cur_wall > 0 and cpu_J:
        scale = cum_wall / cur_wall
        cpu_J_scaled = cpu_J * scale
    cmd_mem = ''
    cmd_line = next((ln for ln in txt.splitlines() if ln.startswith('Command:')), '')
    m = CMD_MEM_RE.search(cmd_line);  cmd_mem = m.group(1) if m else ''
    return dict(
        length         = L,
        wall_current_s = cur_wall,
        wall_cum_s     = cum_wall,
        wall_s         = (cum_wall if cum_wall else cur_wall),
        resumed        = cum_wall is not None,
        energy_scale   = scale,
        cpu_time_s     = f(CPU_TIME_RE),
        tree_wall_s    = f(TREE_WALL_RE),
        mf_wall_s      = f(MF_WALL_RE),    # first match = original full run
        best_logL      = f(BEST_LL_RE),
        cpu_J_segment  = cpu_J,
        cpu_J          = cpu_J_scaled,
        energy_Wh      = cpu_J_scaled / 3600.0 if cpu_J_scaled else None,
        avg_power_W    = (cpu_J / cur_wall) if (cpu_J and cur_wall) else None,
        mf_ram_MB      = f(MF_RAM_RE, int),
        host_GB        = f(HOST_GB_RE, int),
        host_hit       = bool(EXC_RAM_RE.search(txt)),
        mem_arg        = cmd_mem,
        complete       = (cpu_J_scaled is not None) and ((cum_wall or cur_wall) is not None),
        file           = os.path.basename(path),
    )

rows = [parse(L) for L in LENGTHS]
df = pd.DataFrame(rows).sort_values('length').reset_index(drop=True)
df.to_csv(os.path.join(OUT_DIR, 'aa_spr_scaling.csv'), index=False)
print(df[['length','wall_s','resumed','energy_Wh','avg_power_W','mf_ram_MB',
          'host_hit','mem_arg','complete']].to_string(index=False))

# ---------------------------------------------------------------------------
# 2. Power-law fits — fit only completed cells
# ---------------------------------------------------------------------------

ok = df[df['complete']].copy()

def fit_powerlaw(L, y, label):
    valid = pd.notna(y)
    L = np.asarray(L)[valid]
    y = np.asarray(y, dtype=float)[valid]
    if len(L) < 2: return None
    # y = a * L^k  ⇒  log y = log a + k log L
    logL, logy = np.log(L), np.log(y)
    k, log_a = np.polyfit(logL, logy, 1)
    a = np.exp(log_a)
    r2 = 1 - np.sum((logy - (k*logL + log_a))**2) / np.sum((logy - logy.mean())**2)
    print(f'  {label:<14} y = {a:.3g} · L^{k:.3f}    R² = {r2:.4f}    (n={len(L)})')
    return dict(a=a, k=k, r2=r2)

print('\n=== power-law fits ===')
fit_wall   = fit_powerlaw(ok['length'], ok['wall_s'],    'wall_s')
fit_energy = fit_powerlaw(ok['length'], ok['cpu_J'],     'cpu_J')
# ModelFinder RAM is declared at startup (visible even in the SIGABRT logs),
# so we fit using EVERY length except the -mem-capped 10M point. This gives
# a much cleaner fit because the declared demand scales near-linearly with L.
mf_rows = df[(df['mem_arg'] == '') & df['mf_ram_MB'].notna()]
fit_mf_ram = fit_powerlaw(mf_rows['length'], mf_rows['mf_ram_MB'], 'mf_ram_MB')
print(f'  (mf_ram_MB fit uses every length except 10M — 10M ran with -mem 300G, '
      f'which forced MF to a smaller model.)')

# ---------------------------------------------------------------------------
# 3. Predict the crash threshold and the -mem cap that would have rescued
#    3.75M and 5M (the SPR scaling sweep crashes were both ModelFinder host-
#    RAM overruns).
# ---------------------------------------------------------------------------

SPR_HOST_GB = 503  # node total host RAM
SPR_HEADROOM_GB = 60  # rough OS/IO headroom — IQ-TREE realistically can claim ~440 GB
SAFE_MF_MB  = (SPR_HOST_GB - SPR_HEADROOM_GB) * 1024  # ~453 GB

print('\n=== crash threshold projection ===')
if fit_mf_ram:
    # length at which MF_RAM equals the safe ceiling
    a, k = fit_mf_ram['a'], fit_mf_ram['k']
    L_crit = (SAFE_MF_MB / a) ** (1.0 / k)
    print(f'  Safe MF budget ≈ {SAFE_MF_MB/1024:.0f} GB  (={SPR_HOST_GB} - {SPR_HEADROOM_GB} GB OS/IO headroom)')
    print(f'  Projected crash threshold (MF_RAM = ceiling) ≈ {L_crit/1e6:.2f} M sites')
print(f'  Observed: 3.125 M completes, 3.75 M crashes ⇒ threshold sits between them.')

print('\n=== suggested -mem cap to rescue 3.75M / 5M ===')
# Empirically, -mem 300G already rescued AA SPR 10M (the largest crash-risk
# point). For 3.75M and 5M — both smaller than 10M — -mem 300G is the safe,
# pre-validated answer. We additionally print the theoretical cap derived from
# the MF-RAM fit, clipped to the safe budget.
SAFE_GB = (SPR_HOST_GB - SPR_HEADROOM_GB)
for L in (3_750_000, 5_000_000):
    pred = fit_mf_ram['a'] * (L ** fit_mf_ram['k']) / 1024  # GB
    # Theoretical: pick the smaller of (60% of predicted demand) and (safe budget).
    theoretical = min(SAFE_GB, max(150, int(round(pred * 0.6 / 50) * 50)))
    print(f'  L = {L/1e6:.2f} M ⇒ predicted MF demand ≈ {pred:.0f} GB; '
          f'theoretical safe cap ≈ -mem {theoretical}G; '
          f'recommended -mem 300G (already proven on AA 10M).')

# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

def annotate_status(ax, df, x_col, y_col, dy_factor=1.10):
    for _, r in df.iterrows():
        x = r[x_col]
        y_lim = ax.get_ylim()[1]
        if r['complete']:
            continue
        # plot a red X at the highest visible y for crashed cells
        ax.scatter([x], [y_lim*0.6], marker='x', color='#c0392b', s=80, zorder=5)
        ax.text(x, y_lim*0.65, ' SIGABRT', color='#c0392b', fontsize=9,
                ha='left', va='bottom', rotation=90)

def plot_loglog(ax, df, y_col, ylabel, title, fit=None,
                fit_extrapolate_to=None, crash_line=None, crash_label=None,
                marker='o', color='#0071C5'):
    ok = df[df['complete']]
    ax.plot(ok['length'], ok[y_col], marker=marker, color=color, linewidth=1.6,
            markersize=8, label='completed runs')
    # Crashed points: draw at the y the model predicts, with red X
    crashed = df[~df['complete']]
    if fit is not None and not crashed.empty:
        pred_y = fit['a'] * crashed['length'].astype(float)**fit['k']
        ax.scatter(crashed['length'], pred_y, marker='x', color='#c0392b', s=110,
                   linewidths=2.5, zorder=6, label='SIGABRT (predicted y)')
        for _, r in crashed.iterrows():
            pred = fit['a'] * float(r['length'])**fit['k']
            ax.text(r['length'], pred*1.18, f' L={int(r["length"])/1e6:.2f}M ✗',
                    color='#c0392b', fontsize=9, ha='center', va='bottom')
    # Power-law extrapolation curve
    if fit is not None:
        L_grid = np.geomspace(ok['length'].min(), (fit_extrapolate_to or ok['length'].max()), 200)
        y_grid = fit['a'] * L_grid**fit['k']
        ax.plot(L_grid, y_grid, color=color, linewidth=0.8, linestyle='--', alpha=0.6,
                label=f'fit  y = {fit["a"]:.2g}·L$^{{{fit["k"]:.2f}}}$  (R²={fit["r2"]:.3f})')
    if crash_line is not None:
        ax.axhline(crash_line, color='#c0392b', linewidth=1.1, linestyle=':',
                   label=crash_label)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('alignment length (sites)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', linestyle=':', alpha=0.45)
    ax.legend(fontsize=9, loc='lower right')

# Figure 1: wall vs length
fig, ax = plt.subplots(figsize=(9, 6))
plot_loglog(ax, df, 'wall_s', 'total wall-clock time (s)',
            'AA · SPR · ninit=1 — wall time vs alignment length',
            fit=fit_wall, fit_extrapolate_to=11_000_000)
ax.annotate('AA 10M:  resumed run\nactual wall = 220 k s', xy=(10_000_000, 219981),
            xytext=(2_500_000, 220_000), fontsize=9, color='#555',
            arrowprops=dict(arrowstyle='->', color='#888', lw=0.6))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_wall_vs_length.png'), bbox_inches='tight')
plt.close(fig)

# Figure 2: energy vs length
fig, ax = plt.subplots(figsize=(9, 6))
plot_loglog(ax, df, 'cpu_J', 'CPU energy (J, scaled to cumulative for 10M)',
            'AA · SPR · ninit=1 — CPU energy vs alignment length',
            fit=fit_energy, fit_extrapolate_to=11_000_000)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_energy_vs_length.png'), bbox_inches='tight')
plt.close(fig)

# Figure 3: ModelFinder RAM vs length — the binding constraint
fig, ax = plt.subplots(figsize=(9, 6))
plot_loglog(ax, df, 'mf_ram_MB', 'ModelFinder declared peak RAM (MB)',
            'AA · SPR — ModelFinder RAM demand vs alignment length',
            fit=fit_mf_ram, fit_extrapolate_to=11_000_000,
            crash_line=SAFE_MF_MB,
            crash_label=f'SPR safe budget ≈ {SAFE_MF_MB/1024:.0f} GB ({SPR_HOST_GB} GB node − {SPR_HEADROOM_GB} GB OS/IO)')

# Mark the 10M outlier — it used -mem 300G so MF was forced to a smaller model.
ten_M_mf = float(df[df['length']==10_000_000]['mf_ram_MB'].iloc[0])
ax.annotate(f'  10M with -mem 300G\n  → MF capped at {ten_M_mf/1024:.0f} GB',
             xy=(10_000_000, ten_M_mf), xytext=(2_500_000, ten_M_mf*1.2),
             fontsize=9, color='#0a0',
             arrowprops=dict(arrowstyle='->', color='#0a0', lw=0.6))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_mfram_vs_length.png'), bbox_inches='tight')
plt.close(fig)

# Figure 4: ModelFinder phase share — phase split per length
fig, ax = plt.subplots(figsize=(9, 5.5))
ok = df[df['complete']]
labels = [f'{L/1e6:.2f}M' if L >= 1_000_000 else f'{L/1000:.0f}K' for L in ok['length']]
mf  = ok['mf_wall_s'].fillna(0).values / 60.0
tre = ok['tree_wall_s'].fillna(0).values / 60.0
# For 10M (resumed) tree-wall is segment-only; estimate cumulative as
# cum_wall - mf_wall. Otherwise mf+tree usually ≈ wall.
for i, r in ok.reset_index(drop=True).iterrows():
    if r['resumed']:
        tre[i] = max(0, r['wall_cum_s']/60.0 - mf[i])
x = np.arange(len(labels))
ax.bar(x, mf,  color='#7f4ca5', label='ModelFinder phase')
ax.bar(x, tre, bottom=mf, color='#0071C5', label='tree-search phase')
for i, (m, t) in enumerate(zip(mf, tre)):
    total = m + t
    ax.text(i, total*1.01, f'{total/60:.1f} h', ha='center', va='bottom', fontsize=9)
    if m > 5: ax.text(i, m/2,         f'{m/60:.1f} h', ha='center', va='center', color='white', fontsize=8)
    if t > 5: ax.text(i, m + t/2,     f'{t/60:.1f} h', ha='center', va='center', color='white', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel('alignment length')
ax.set_ylabel('wall time (minutes)')
ax.set_title('AA · SPR — phase split (ModelFinder + tree search)\n'
             '10M tree-search estimated as cumulative_wall − ModelFinder_wall')
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_phase_split.png'), bbox_inches='tight')
plt.close(fig)

# Figure 5: avg power vs length — sanity check
fig, ax = plt.subplots(figsize=(9, 5.5))
ok = df[df['complete']]
ax.plot(ok['length'], ok['avg_power_W'], marker='o', color='#E65100', linewidth=1.6, markersize=8)
ax.set_xscale('log')
ax.set_xlabel('alignment length (sites, log)')
ax.set_ylabel('average CPU power (W)')
ax.set_title('AA · SPR — average CPU power vs alignment length\n'
             '(should stay near socket TDP; deviations point to underutilisation)')
ax.grid(True, which='both', linestyle=':', alpha=0.45)
for _, r in ok.iterrows():
    ax.text(r['length'], r['avg_power_W']+8, f'{r["avg_power_W"]:.0f} W',
            ha='center', va='bottom', fontsize=8)
ax.set_ylim(550, 800)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_avg_power.png'), bbox_inches='tight')
plt.close(fig)

print('\nWrote:')
for f in sorted(os.listdir(OUT_DIR)):
    print('  ', f)
