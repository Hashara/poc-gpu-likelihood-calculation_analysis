#!/usr/bin/env python3
# Single consolidated diagram of the H200 SM-scaling sweep (reads sm_sweep_summary.csv).
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_13_sm_sweep'
GREEN='#76B900'; BLUE='#0071C5'; ORANGE='#F2A900'; GREY='#888888'; RED='#C0392B'

df = pd.read_csv(os.path.join(OUT, 'sm_sweep_summary.csv')).sort_values('sm')
X  = df['sm'].values
plat = df.loc[df['mps_plateau'], 'sm'].tolist()
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 10})

def setx(ax):
    ax.set_xscale('log', base=2); ax.set_xticks(X); ax.set_xticklabels([str(s) for s in X])
    ax.set_xlabel('SMs requested via MPS (of 132 on H200)'); ax.grid(True, which='both', alpha=0.25)
def shade(ax):
    if len(plat) >= 2:
        ax.axvspan(plat[0]*0.9, plat[-1]*1.1, color=RED, alpha=0.07, zorder=0)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('H200 SM-scaling sweep — IQ-TREE3 OpenACC  (AA LG+I+G4, 100 taxa × 100k sites, -ninit 2 -seed 1)',
             fontsize=13, fontweight='bold')

# (a) runtime
ax = axes[0,0]
ax.plot(X, df['wall_s']/60, 'o-', color=GREEN, lw=2, label='Total wall')
ax.plot(X, df['ts_wall_s']/60, 's--', color=BLUE, label='Tree search')
ax.plot(X, df['mf_wall_s']/60, '^--', color=ORANGE, label='ModelFinder')
shade(ax); setx(ax); ax.set_ylabel('Wall-clock time (min)'); ax.set_title('(a) Runtime'); ax.legend(fontsize=8.5)
ax.text(np.sqrt(plat[0]*plat[-1]), ax.get_ylim()[1]*0.97, 'MPS floor\n(1/2/4/8 = same ~8 SMs)',
        ha='center', va='top', fontsize=8, color=RED)

# (b) speedup
ax = axes[0,1]
ax.plot(X, X/X.min(), ':', color=GREY, label='Ideal linear')
ax.plot(X, df['speedup'], 'o-', color=GREEN, lw=2, label='Total wall speedup')
for x,y in zip(X, df['speedup']):
    ax.annotate(f'{y:.1f}×',(x,y),textcoords='offset points',xytext=(4,5),fontsize=8)
ax.set_yscale('log'); shade(ax); setx(ax)
ax.set_ylabel('Speedup vs 1-SM request'); ax.set_title('(b) Speedup (real scaling starts at 16)'); ax.legend(fontsize=8.5)

# (c) GPU power — MPS evidence
ax = axes[1,0]
ax.plot(X, df['measured_W'], 'o-', color=GREEN, lw=2, label='GPU board power (sampled)')
for x,y in zip(X, df['measured_W']):
    ax.annotate(f'{y:.0f}W',(x,y),textcoords='offset points',xytext=(3,5),fontsize=7.5)
shade(ax); setx(ax); ax.set_ylabel('Avg GPU power (W)')
ax.set_title('(c) GPU power — flat 1→8 = MPS gives same allocation'); ax.legend(fontsize=8.5)

# (d) energy
ax = axes[1,1]
ax.plot(X, df['gpu_J']/3600, 'o-', color=GREEN, lw=2, label='GPU (NVML)')
ax.plot(X, df['cpu_J']/3600, 's--', color=BLUE, label='CPU (RAPL, noisy)')
ax.plot(X, (df['gpu_J']+df['cpu_J'])/3600, '^-', color='black', lw=1.5, label='Total')
shade(ax); setx(ax); ax.set_ylabel('Energy per run (Wh)')
ax.set_title('(d) Energy to solution (min at 132 SM)'); ax.legend(fontsize=8.5)

fig.text(0.5, 0.005,
         'logL = -7541976.86 identical across all SM counts (bit-exact). '
         'Full GPU = 9.6× over floor, fastest AND lowest energy. Peak 321 W ≈ 46% TDP → HBM-bound.',
         ha='center', fontsize=9, style='italic')
fig.tight_layout(rect=[0, 0.02, 1, 0.97])
fig.savefig(os.path.join(OUT, 'fig_summary_dashboard.png'))
print('wrote', os.path.join(OUT, 'fig_summary_dashboard.png'))
