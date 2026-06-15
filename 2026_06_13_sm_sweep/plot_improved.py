#!/usr/bin/env python3
# Improved, publication-quality H200 SM-scaling figures for IQ-TREE.
# Honest 8-SM-floor framing (1/2/4/8 MPS requests = same 8-SM hardware partition).
# Reads sm_sweep_corrected.csv (5 real points) + sm_sweep_summary.csv (floor samples).
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

OUT = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_13_sm_sweep'
G='#76B900'; GD='#4e7a00'; B='#0071C5'; O='#E8930C'; GREY='#9aa0a6'; RED='#C0392B'; INK='#202124'

plt.rcParams.update({
    'figure.dpi':120,'savefig.dpi':300,'font.size':12,
    'axes.titlesize':13,'axes.titleweight':'bold','axes.labelsize':12,
    'axes.labelcolor':INK,'text.color':INK,'xtick.labelsize':11,'ytick.labelsize':11,
    'legend.fontsize':10,'legend.frameon':False,'axes.grid':True,'grid.alpha':0.25,
    'grid.linewidth':0.7,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#666','font.family':'DejaVu Sans','figure.facecolor':'white',
})

p = pd.read_csv(os.path.join(OUT,'sm_sweep_corrected.csv')).sort_values('sm').reset_index(drop=True)
s = pd.read_csv(os.path.join(OUT,'sm_sweep_summary.csv')).sort_values('sm')
floor = s[s['sm'] <= 8]                              # the 4 collapsed requests
X = p['sm'].values
FLOOR = 8

def logx(ax):
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_locator(FixedLocator(X))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(v)) for v in X]))
    ax.set_xlabel('SMs used on H200  (hardware floor = 8)')
    ax.set_xlim(X.min()*0.82, X.max()*1.18)

def label_pts(ax, xs, ys, fmt, dy=9, dx=0, fs=10, color=INK):
    for x,y in zip(xs,ys):
        ax.annotate(fmt(y),(x,y),textcoords='offset points',xytext=(dx,dy),
                    ha='center',fontsize=fs,color=color,fontweight='bold')

# ---------- standalone: RUNTIME ----------
fig, ax = plt.subplots(figsize=(8,5.2))
ax.plot(X, p['wall_s']/60,'o-',color=G,lw=2.6,ms=9,zorder=5,label='Total wall')
ax.plot(X, p['ts_wall_s']/60,'s--',color=B,lw=1.8,ms=6,label='Tree search')
ax.plot(X, p['mf_wall_s']/60,'^--',color=O,lw=1.8,ms=6,label='ModelFinder')
ax.scatter([FLOOR]*len(floor), floor['wall_s']/60, s=80, facecolors='none',
           edgecolors=RED, lw=1.4, zorder=6)
ax.annotate('1/2/4/8 SM requests →\nsame 8-SM partition',
            xy=(FLOOR, floor['wall_s'].mean()/60), xytext=(FLOOR*1.5, 150),
            fontsize=9.5, color=RED, arrowprops=dict(arrowstyle='->',color=RED,lw=1))
label_pts(ax, X, p['wall_s']/60, lambda v:f'{v:.0f}', dy=10, color=GD)
logx(ax); ax.set_ylabel('Wall-clock time  (minutes)')
ax.set_title('IQ-TREE on H200 — runtime vs SM count')
ax.legend(loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig2_runtime.png')); plt.close(fig)

# ---------- standalone: SPEEDUP (with lost-scaling shading) ----------
fig, ax = plt.subplots(figsize=(8,5.2))
ax.fill_between(X, p['speedup'], p['ideal'], color=GREY, alpha=0.18, label='scaling lost to HBM bound')
ax.plot(X, p['ideal'],':',color=GREY,lw=2,label='ideal linear (from 8 SM)')
ax.plot(X, p['speedup'],'o-',color=G,lw=2.6,ms=9,zorder=5,label='measured')
ax.set_xscale('log',base=2); ax.set_yscale('log',base=2)
ax.xaxis.set_major_locator(FixedLocator(X)); ax.xaxis.set_major_formatter(FixedFormatter([str(int(v)) for v in X]))
ax.yaxis.set_major_locator(FixedLocator([1,2,4,8,16])); ax.yaxis.set_major_formatter(FixedFormatter(['1×','2×','4×','8×','16×']))
ax.set_xlim(X.min()*0.82, X.max()*1.18); ax.set_xlabel('SMs used on H200  (hardware floor = 8)')
label_pts(ax, X, p['speedup'], lambda v:f'{v:.2f}×', dy=-16, color=GD)
ax.set_ylabel('Speedup vs 8-SM floor')
ax.set_title('IQ-TREE on H200 — speedup (baseline = 8-SM floor)')
ax.legend(loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig2_speedup.png')); plt.close(fig)

# ---------- standalone: EFFICIENCY ----------
fig, ax = plt.subplots(figsize=(8,5.2))
ax.axhspan(80,110,color=G,alpha=0.08)
ax.text(X.max(), 104, 'near-linear (>80%)', ha='right', fontsize=9, color=GD)
ax.plot(X, p['efficiency']*100,'o-',color=G,lw=2.6,ms=9,zorder=5)
ax.axhline(100, ls=':', color=GREY, lw=1.5)
label_pts(ax, X, p['efficiency']*100, lambda v:f'{v:.0f}%', dy=10, color=GD)
ax.set_ylim(40,112); logx(ax); ax.set_ylabel('Parallel efficiency vs 8-SM floor  (%)')
ax.set_title('IQ-TREE on H200 — efficiency: ~84% to 33 SM, 58% at full GPU')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig2_efficiency.png')); plt.close(fig)

# ---------- standalone: PARETO (energy vs time — the "no trade-off" figure) ----------
fig, ax = plt.subplots(figsize=(8,5.6))
xs = p['wall_s']/60; ys = p['gpu_Wh']
ax.plot(xs, ys, '-', color=GREY, lw=1.4, zorder=2)
sc = ax.scatter(xs, ys, c=X, cmap='viridis', s=160, zorder=5, edgecolors='white', lw=1.2)
for x,y,n in zip(xs, ys, X):
    ax.annotate(f'{int(n)} SM',(x,y),textcoords='offset points',xytext=(8,6),fontsize=10,fontweight='bold')
ax.annotate('better', xy=(0.10,0.12), xytext=(0.34,0.30), xycoords='axes fraction',
            fontsize=12, color=GD, fontweight='bold',
            arrowprops=dict(arrowstyle='-|>', color=GD, lw=2.2))
cb = fig.colorbar(sc, ax=ax, label='SMs used'); cb.outline.set_visible(False)
ax.set_xlabel('Wall-clock time  (minutes)'); ax.set_ylabel('GPU energy to solution  (Wh, NVML)')
ax.set_title('IQ-TREE on H200 — energy vs runtime: 132 SM dominates both')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig2_pareto.png')); plt.close(fig)

# ---------- improved DASHBOARD (2×2) ----------
fig, axes = plt.subplots(2,2, figsize=(13.5,9.4))
fig.suptitle('H200 SM scaling — IQ-TREE3 OpenACC   (8-SM hardware floor; AA LG+I+G4, 100 taxa × 100k sites)',
             fontsize=14, fontweight='bold', y=0.985)

ax=axes[0,0]
ax.plot(X,p['wall_s']/60,'o-',color=G,lw=2.4,ms=8,label='Total wall')
ax.plot(X,p['ts_wall_s']/60,'s--',color=B,lw=1.6,ms=5,label='Tree search')
ax.plot(X,p['mf_wall_s']/60,'^--',color=O,lw=1.6,ms=5,label='ModelFinder')
ax.scatter([FLOOR]*len(floor),floor['wall_s']/60,s=70,facecolors='none',edgecolors=RED,lw=1.3,zorder=6,
           label='1/2/4/8 req → 8 SMs')
label_pts(ax,X,p['wall_s']/60,lambda v:f'{v:.0f}',dy=9,fs=9,color=GD)
logx(ax); ax.set_ylabel('Wall time (min)'); ax.set_title('(a) Runtime'); ax.legend(fontsize=8.5)

ax=axes[0,1]
ax.fill_between(X,p['speedup'],p['ideal'],color=GREY,alpha=0.18)
ax.plot(X,p['ideal'],':',color=GREY,lw=1.8,label='ideal linear')
ax.plot(X,p['speedup'],'o-',color=G,lw=2.4,ms=8,label='measured')
ax.set_xscale('log',base=2); ax.set_yscale('log',base=2)
ax.xaxis.set_major_locator(FixedLocator(X)); ax.xaxis.set_major_formatter(FixedFormatter([str(int(v)) for v in X]))
ax.yaxis.set_major_locator(FixedLocator([1,2,4,8,16])); ax.yaxis.set_major_formatter(FixedFormatter(['1×','2×','4×','8×','16×']))
ax.set_xlim(X.min()*0.82,X.max()*1.18); ax.set_xlabel('SMs used on H200  (hardware floor = 8)')
label_pts(ax,X,p['speedup'],lambda v:f'{v:.1f}×',dy=-15,fs=9,color=GD)
ax.set_ylabel('Speedup vs 8-SM'); ax.set_title('(b) Speedup (shaded = lost to HBM bound)'); ax.legend(fontsize=8.5,loc='upper left')

ax=axes[1,0]
ax.axhspan(80,112,color=G,alpha=0.08)
ax.plot(X,p['efficiency']*100,'o-',color=G,lw=2.4,ms=8)
ax.axhline(100,ls=':',color=GREY,lw=1.4)
label_pts(ax,X,p['efficiency']*100,lambda v:f'{v:.0f}%',dy=9,fs=9.5,color=GD)
ax.set_ylim(40,112); logx(ax); ax.set_ylabel('Efficiency vs 8-SM (%)'); ax.set_title('(c) Parallel efficiency')

ax=axes[1,1]
xs=p['wall_s']/60; ys=p['gpu_Wh']
ax.plot(xs,ys,'-',color=GREY,lw=1.3,zorder=2)
sc=ax.scatter(xs,ys,c=X,cmap='viridis',s=130,zorder=5,edgecolors='white',lw=1)
for x,y,n in zip(xs,ys,X):
    ax.annotate(f'{int(n)}',(x,y),textcoords='offset points',xytext=(7,5),fontsize=9,fontweight='bold')
ax.annotate('better',xy=(0.10,0.12),xytext=(0.36,0.30),xycoords='axes fraction',
            fontsize=11,color=GD,fontweight='bold',arrowprops=dict(arrowstyle='-|>',color=GD,lw=2))
ax.set_xlabel('Wall time (min)'); ax.set_ylabel('GPU energy (Wh)'); ax.set_title('(d) Energy vs runtime — 132 SM wins both')

fig.text(0.5,0.005,'H200 min SM partition = 8 (MPS + green-context confirmed). logL bit-exact across all runs. '
         'Full GPU = 9.6× over the 8-SM floor, fastest AND lowest energy.',ha='center',fontsize=9.5,style='italic')
fig.tight_layout(rect=[0,0.02,1,0.965])
fig.savefig(os.path.join(OUT,'fig_corrected_dashboard.png'))   # overwrite canonical with improved
plt.close(fig)
print('wrote: fig_corrected_dashboard.png (improved), fig2_runtime/speedup/efficiency/pareto.png')
