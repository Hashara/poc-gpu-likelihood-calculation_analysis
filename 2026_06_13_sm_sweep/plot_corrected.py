#!/usr/bin/env python3
# CORRECTED H200 SM-scaling diagram for IQ-TREE.
#
# Why corrected: on the H200 the minimum schedulable SM partition is 8 SMs
# (confirmed two ways: MPS active-thread-% collapses 1/2/4/8 to one allocation,
# and cuDevSmResourceSplitByCount grants 8 for any request <=8). So the MPS
# "1/2/4/8 SM" runs are NOT distinct points — they are 4 samples of the same
# 8-SM floor. The real operating points are 8, 16, 33, 66, 132 SMs, and speedup
# / efficiency must be baselined at the 8-SM floor, not a non-existent 1-SM run.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_13_sm_sweep'
GREEN='#76B900'; BLUE='#0071C5'; ORANGE='#F2A900'; GREY='#888888'; RED='#C0392B'
FLOOR = 8

df = pd.read_csv(os.path.join(OUT, 'sm_sweep_summary.csv')).sort_values('sm').reset_index(drop=True)

# the four floor requests (1/2/4/8) = same physical 8-SM allocation
floor_samples = df[df['sm'] <= FLOOR].copy()
real          = df[df['sm'] >  FLOOR].copy()

def floor_mean(col): return float(floor_samples[col].mean())
floor_row = {
    'sm': FLOOR, 'wall_s': floor_mean('wall_s'),
    'mf_wall_s': floor_mean('mf_wall_s'), 'ts_wall_s': floor_mean('ts_wall_s'),
    'gpu_J': floor_mean('gpu_J'), 'cpu_J': floor_mean('cpu_J'),
    'measured_W': floor_mean('measured_W'), 'best_logL': floor_samples['best_logL'].iloc[0],
}
pts = pd.concat([pd.DataFrame([floor_row]),
                 real[['sm','wall_s','mf_wall_s','ts_wall_s','gpu_J','cpu_J','measured_W','best_logL']]],
                ignore_index=True).sort_values('sm').reset_index(drop=True)

base = pts.loc[pts['sm'] == FLOOR].iloc[0]
pts['speedup']    = base['wall_s'] / pts['wall_s']
pts['ideal']      = pts['sm'] / FLOOR
pts['efficiency'] = pts['speedup'] / pts['ideal']
pts['gpu_Wh'] = pts['gpu_J']/3600; pts['cpu_Wh'] = pts['cpu_J']/3600
pts['total_Wh'] = pts['gpu_Wh'] + pts['cpu_Wh']
pts.to_csv(os.path.join(OUT, 'sm_sweep_corrected.csv'), index=False)
print(pts[['sm','wall_s','speedup','ideal','efficiency','gpu_Wh','measured_W']].to_string(index=False))

X = pts['sm'].values
plt.rcParams.update({'figure.dpi':110,'savefig.dpi':300,'font.size':10})
def setx(ax):
    ax.set_xscale('log', base=2); ax.set_xticks(X); ax.set_xticklabels([str(int(s)) for s in X])
    ax.set_xlabel('SMs actually used on H200 (min partition = 8)'); ax.grid(True, which='both', alpha=0.25)
def floor_note(ax):
    yt = ax.get_ylim()[1]
    ax.annotate('MPS requests 1/2/4/8\nall resolve here (8-SM floor)', xy=(FLOOR, base_y(ax)),
                xytext=(FLOOR*1.15, yt*0.62), fontsize=7.5, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))
def base_y(ax): return None

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('H200 SM scaling — IQ-TREE3 OpenACC (CORRECTED: 8-SM hardware floor)\n'
             'AA LG+I+G4, 100 taxa × 100k sites, -ninit 2 -seed 1', fontsize=13, fontweight='bold')

# (a) runtime — show the 4 floor samples overlapping at x=8
ax = axes[0,0]
ax.plot(X, pts['wall_s']/60, 'o-', color=GREEN, lw=2, ms=7, label='Total wall (5 real points)')
ax.plot(X, pts['ts_wall_s']/60, 's--', color=BLUE, label='Tree search')
ax.plot(X, pts['mf_wall_s']/60, '^--', color=ORANGE, label='ModelFinder')
ax.scatter([FLOOR]*len(floor_samples), floor_samples['wall_s']/60, s=70, facecolors='none',
           edgecolors=RED, lw=1.3, zorder=6, label='1/2/4/8 requests → same 8 SMs')
setx(ax); ax.set_ylabel('Wall-clock time (min)'); ax.set_title('(a) Runtime'); ax.legend(fontsize=8)

# (b) speedup vs 8-SM floor
ax = axes[0,1]
ax.plot(X, pts['ideal'], ':', color=GREY, label='Ideal linear (from 8 SM)')
ax.plot(X, pts['speedup'], 'o-', color=GREEN, lw=2, ms=7, label='Measured speedup')
for x,y in zip(X, pts['speedup']):
    ax.annotate(f'{y:.2f}×',(x,y),textcoords='offset points',xytext=(5,5),fontsize=8)
ax.set_xscale('log', base=2); ax.set_yscale('log', base=2)
ax.set_xticks(X); ax.set_xticklabels([str(int(s)) for s in X]); ax.grid(True, which='both', alpha=0.25)
ax.set_xlabel('SMs actually used on H200 (min partition = 8)')
ax.set_ylabel('Speedup vs 8-SM floor'); ax.set_title('(b) Speedup (baseline = 8 SM)'); ax.legend(fontsize=8.5)

# (c) efficiency vs 8-SM floor — the honest one
ax = axes[1,0]
ax.plot(X, pts['efficiency']*100, 'o-', color=GREEN, lw=2, ms=7)
for x,y in zip(X, pts['efficiency']*100):
    ax.annotate(f'{y:.0f}%',(x,y),textcoords='offset points',xytext=(5,5),fontsize=8.5)
ax.axhline(100, ls=':', color=GREY, label='100% (perfect scaling)')
ax.set_ylim(0, 110); setx(ax); ax.set_ylabel('Parallel efficiency vs 8-SM floor (%)')
ax.set_title('(c) Efficiency — ~84% to 33 SM, 58% at full GPU'); ax.legend(fontsize=8.5)

# (d) energy
ax = axes[1,1]
ax.plot(X, pts['gpu_Wh'], 'o-', color=GREEN, lw=2, ms=7, label='GPU (NVML)')
ax.plot(X, pts['cpu_Wh'], 's--', color=BLUE, label='CPU (RAPL, noisy)')
ax.plot(X, pts['total_Wh'], '^-', color='black', lw=1.5, label='Total')
setx(ax); ax.set_ylabel('Energy per run (Wh)'); ax.set_title('(d) Energy (min at 132 SM)'); ax.legend(fontsize=8.5)

fig.text(0.5, 0.005,
         'H200 min SM partition = 8 (MPS + green-context confirmed) → 1/2/4 SM are unreachable for the real workload. '
         'logL identical (bit-exact). Full GPU = 9.6× over the 8-SM floor, fastest AND lowest energy.',
         ha='center', fontsize=9, style='italic')
fig.tight_layout(rect=[0,0.02,1,0.96])
fig.savefig(os.path.join(OUT, 'fig_corrected_dashboard.png'))
print('\nwrote', os.path.join(OUT, 'fig_corrected_dashboard.png'))
