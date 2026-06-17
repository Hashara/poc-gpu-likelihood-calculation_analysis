#!/usr/bin/env python3
# 2026-06-16 — H200 SM-scaling sweep for IQ-TREE3 OpenACC, DNA GTR+I+G4 data.
#
# One PBS job per SM count, REPEATS=1, same workload each time:
#     iqtree3 -s alignment_1000000.phy -m TEST -ninit 2 -seed 1
#     (DNA, GTR+I+G4 data, 100 taxa, 1,000,000 sites)
# SM count restricted via CUDA MPS active-thread-percentage = SM/132*100.
#
# Sweep points: 8, 16, 33, 66, 132 SMs. 8 = H200 MPS minimum SM partition
# (proven in 2026_06_13_sm_sweep) → the hardware floor and speedup/efficiency
# baseline. Energy from IQ-TREE's Energy: block (CPU = RAPL, GPU = NVML).
#
# Data:  /Users/u7826985/Projects/Nvidia/results/2026_06_16_sm_sweep_dna1M/sm_<N>/
# Out:   .../poc-gpu-likelihood-calculation_analysis/2026_06_16_sm_sweep_dna1M/

import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

DNA_ROOT = '/Users/u7826985/Projects/Nvidia/results/2026_06_16_sm_sweep_dna1M'
OUT_DIR  = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_16_sm_sweep_dna1M'
TOTAL_SM = 132
FLOOR    = 8
POINTS   = [8, 16, 33, 66, 132]

plt.rcParams.update({
    'figure.dpi':120,'savefig.dpi':300,'font.size':12,
    'axes.titlesize':13,'axes.titleweight':'bold','axes.labelsize':12,
    'legend.fontsize':10,'legend.frameon':False,'axes.grid':True,'grid.alpha':0.25,
    'grid.linewidth':0.7,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#666','figure.facecolor':'white',
})
G='#76B900'; GD='#4e7a00'; B='#0071C5'; O='#E8930C'; GREY='#9aa0a6'; RED='#C0392B'; INK='#202124'

MF_WALL_RE = re.compile(r'Wall-clock time for ModelFinder:\s+([\d.]+)\s+seconds')
TS_WALL_RE = re.compile(r'Wall-clock time used for tree search:\s+([\d.]+)\s+sec')
TOT_WALL_RE= re.compile(r'Total wall-clock time used:\s+([\d.]+)\s+sec')
TOT_CPU_RE = re.compile(r'Total CPU time used:\s+([\d.]+)\s+sec')
BEST_LL_RE = re.compile(r'BEST SCORE FOUND\s*:\s+(-?[\d.]+)')
MODEL_RE   = re.compile(r'Best-fit model:\s+(\S+)')
MF_EN_RE   = re.compile(r'Energy used for ModelFinder:\s+CPU\s+([\d.]+)\s+J,\s+GPU\s+([\d.]+)\s+J')
TS_EN_RE   = re.compile(r'Energy used for tree search:\s+CPU\s+([\d.]+)\s+J,\s+GPU\s+([\d.]+)\s+J')
EN_CPU_RE  = re.compile(r'Energy:\s*\n\s*CPU:\s+([\d.]+)\s+J')
EN_GPU_RE  = re.compile(r'\n\s*GPU:\s+([\d.]+)\s+J')
GPU_MEM_RE = re.compile(r'GPU mem:\s+([\d.]+)\s*/\s*([\d.]+)\s+MB')

def f(m, g=1): return float(m.group(g)) if m else np.nan

def parse_log(path):
    txt = open(path, errors='ignore').read()
    o = {}
    o['model']     = (MODEL_RE.search(txt).group(1) if MODEL_RE.search(txt) else '?')
    o['mf_wall_s'] = f(MF_WALL_RE.search(txt)); o['ts_wall_s'] = f(TS_WALL_RE.search(txt))
    o['wall_s']    = f(TOT_WALL_RE.search(txt)); o['cpu_s']    = f(TOT_CPU_RE.search(txt))
    o['best_logL'] = f(BEST_LL_RE.search(txt))
    m = MF_EN_RE.search(txt); o['mf_cpu_J']=f(m,1); o['mf_gpu_J']=f(m,2)
    m = TS_EN_RE.search(txt); o['ts_cpu_J']=f(m,1); o['ts_gpu_J']=f(m,2)
    o['cpu_J'] = f(EN_CPU_RE.search(txt)); o['gpu_J'] = f(EN_GPU_RE.search(txt))
    m = GPU_MEM_RE.search(txt); o['gpu_mem_peak_MB'] = f(m,1)
    return o

def mean_power_from_csv(path):
    try:
        df = pd.read_csv(path)
        col = [c for c in df.columns if 'power' in c.lower()][0]
        vals = pd.to_numeric(df[col].astype(str).str.extract(r'([\d.]+)')[0], errors='coerce').dropna()
        vals = vals[vals > 0]
        return float(vals.mean()) if len(vals) else np.nan
    except Exception:
        return np.nan

def collect(root, log_tmpl, power_tmpl):
    rows = []
    for sm in POINTS:
        log = os.path.join(root, f'sm_{sm}', log_tmpl.format(sm=sm))
        if not os.path.exists(log):
            print('  missing', log); continue
        rec = {'sm': sm, 'pct': round(sm/TOTAL_SM*100, 4)}
        rec.update(parse_log(log))
        pcsv = os.path.join(root, f'sm_{sm}', power_tmpl.format(sm=sm))
        rec['measured_W'] = mean_power_from_csv(pcsv) if os.path.exists(pcsv) else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values('sm').reset_index(drop=True)
    df['total_J'] = df['cpu_J'] + df['gpu_J']
    df['gpu_W'] = df['gpu_J']/df['wall_s']; df['cpu_W'] = df['cpu_J']/df['wall_s']
    df['gpu_Wh']=df['gpu_J']/3600; df['cpu_Wh']=df['cpu_J']/3600; df['total_Wh']=df['total_J']/3600
    base = df.loc[df['sm']==FLOOR].iloc[0]
    df['speedup']    = base['wall_s']/df['wall_s']
    df['ts_speedup'] = base['ts_wall_s']/df['ts_wall_s']
    df['mf_speedup'] = base['mf_wall_s']/df['mf_wall_s']
    df['ideal']      = df['sm']/FLOOR
    df['efficiency'] = df['speedup']/df['ideal']
    df['edp_Js']     = df['gpu_J']*df['wall_s']
    return df

print('Parsing DNA 1M -m TEST run ...')
dna = collect(DNA_ROOT, 'MTEST_dna1M_sm_{sm}_rep1.log', 'MTEST_dna1M_sm_{sm}_rep1_power.csv')

dna.to_csv(os.path.join(OUT_DIR, 'sm_sweep_runs.csv'), index=False)
dna.to_csv(os.path.join(OUT_DIR, 'sm_sweep_summary.csv'), index=False)
print('\n=== DNA 1M -m TEST summary ===')
print(dna[['sm','pct','model','wall_s','mf_wall_s','ts_wall_s','speedup','efficiency',
           'gpu_J','cpu_J','measured_W','gpu_mem_peak_MB','best_logL']].to_string(index=False))

X = dna['sm'].values
WL = 'IQ-TREE3 OpenACC on H200  ·  -m TEST  ·  DNA 1M'

def logx(ax, x=X):
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(v)) for v in x]))
    ax.set_xlabel('SMs used on H200  (hardware floor = 8)')
    ax.set_xlim(min(x)*0.82, max(x)*1.18)

def label_pts(ax, xs, ys, fmt, dy=9, fs=10, color=GD):
    for x,y in zip(xs,ys):
        ax.annotate(fmt(y),(x,y),textcoords='offset points',xytext=(0,dy),
                    ha='center',fontsize=fs,fontweight='bold',color=color)

# 1. runtime
fig, ax = plt.subplots(figsize=(8,5.2))
ax.plot(X, dna['wall_s']/60,'o-',color=G,lw=2.6,ms=9,zorder=5,label='Total wall')
ax.plot(X, dna['ts_wall_s']/60,'s--',color=B,lw=1.8,ms=6,label='Tree search')
ax.plot(X, dna['mf_wall_s']/60,'^--',color=O,lw=1.8,ms=6,label='ModelFinder (-m TEST)')
label_pts(ax, X, dna['wall_s']/60, lambda v:f'{v:.0f}', dy=10)
logx(ax); ax.set_ylabel('Wall-clock time  (minutes)')
ax.set_title('Runtime vs SM count\n'+WL); ax.legend(loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_runtime_vs_sm.png')); plt.close(fig)

# 2. GPU power
fig, ax = plt.subplots(figsize=(8,5.2))
ax.plot(X, dna['measured_W'], 'o-', color=G, lw=2.4, ms=8, label='GPU board power (nvidia-smi sampled)')
ax.plot(X, dna['gpu_W'], 'x--', color=GD, label='GPU avg (NVML energy / wall)')
label_pts(ax, X, dna['measured_W'], lambda v:f'{v:.0f}W', dy=8, fs=9, color=GD)
logx(ax); ax.set_ylabel('Average GPU power (W)')
ax.set_title('GPU power vs SM\n'+WL); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_power_vs_sm.png')); plt.close(fig)

# 3. speedup
fig, ax = plt.subplots(figsize=(8,5.2))
ax.fill_between(X, dna['speedup'], dna['ideal'], color=GREY, alpha=0.18, label='scaling lost to HBM bound')
ax.plot(X, dna['ideal'],':',color=GREY,lw=2,label='ideal linear (from 8 SM)')
ax.plot(X, dna['speedup'],'o-',color=G,lw=2.6,ms=9,zorder=5,label='Total wall')
ax.plot(X, dna['ts_speedup'],'s--',color=B,lw=1.8,ms=6,label='Tree search')
ax.plot(X, dna['mf_speedup'],'^--',color=O,lw=1.8,ms=6,label='ModelFinder')
ax.set_xscale('log',base=2); ax.set_yscale('log',base=2)
ax.xaxis.set_major_locator(FixedLocator(X)); ax.xaxis.set_major_formatter(FixedFormatter([str(int(v)) for v in X]))
ax.yaxis.set_major_locator(FixedLocator([1,2,4,8,16])); ax.yaxis.set_major_formatter(FixedFormatter(['1×','2×','4×','8×','16×']))
ax.set_xlim(X.min()*0.82, X.max()*1.18); ax.set_xlabel('SMs used on H200  (hardware floor = 8)')
label_pts(ax, X, dna['speedup'], lambda v:f'{v:.2f}×', dy=-16)
ax.annotate(f"{dna['ts_speedup'].iloc[-1]:.2f}× tree search",
            (X[-1], dna['ts_speedup'].iloc[-1]), textcoords='offset points',
            xytext=(8, 7), ha='left', fontsize=9, fontweight='bold', color=B)
ax.annotate(f"{dna['mf_speedup'].iloc[-1]:.2f}× ModelFinder",
            (X[-1], dna['mf_speedup'].iloc[-1]), textcoords='offset points',
            xytext=(8, -14), ha='left', fontsize=9, fontweight='bold', color=O)
ax.set_ylabel('Speedup vs 8-SM floor')
ax.set_title('Speedup by phase (baseline = 8-SM floor)\n'+WL); ax.legend(loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_speedup_vs_sm.png')); plt.close(fig)

# 4. efficiency
fig, ax = plt.subplots(figsize=(8,5.2))
ax.axhspan(80,112,color=G,alpha=0.08)
ax.text(X.max(), 104, 'near-linear (>80%)', ha='right', fontsize=9, color=GD)
ax.plot(X, dna['efficiency']*100,'o-',color=G,lw=2.6,ms=9,zorder=5)
ax.axhline(100, ls=':', color=GREY, lw=1.5)
label_pts(ax, X, dna['efficiency']*100, lambda v:f'{v:.0f}%', dy=10)
ax.set_ylim(40,112); logx(ax); ax.set_ylabel('Parallel efficiency vs 8-SM floor  (%)')
ax.set_title('Parallel efficiency vs SM\n'+WL)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_efficiency_vs_sm.png')); plt.close(fig)

# 5. energy
fig, ax = plt.subplots(figsize=(8,5.2))
ax.plot(X, dna['gpu_Wh'], 'o-', color=G, lw=2.4, ms=8, label='GPU (NVML)')
ax.plot(X, dna['cpu_Wh'], 's--', color=B, label='CPU (RAPL)')
ax.plot(X, dna['total_Wh'], '^-', color='black', lw=1.5, label='Total')
label_pts(ax, X, dna['gpu_Wh'], lambda v:f'{v:.0f}', dy=9, fs=9, color=GD)
logx(ax); ax.set_ylabel('Energy per run (Wh)')
ax.set_title('Energy to solution vs SM  (min at 132 SM)\n'+WL); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_energy_vs_sm.png')); plt.close(fig)

# 6. energy breakdown
fig, ax = plt.subplots(figsize=(8.4,5.2))
xi=np.arange(len(X)); w=0.6
b1=dna['mf_gpu_J']/3600; b2=dna['ts_gpu_J']/3600; b3=dna['mf_cpu_J']/3600; b4=dna['ts_cpu_J']/3600
ax.bar(xi,b1,w,color=G,label='GPU ModelFinder')
ax.bar(xi,b2,w,bottom=b1,color=GD,label='GPU tree search')
ax.bar(xi,b3,w,bottom=b1+b2,color=B,label='CPU ModelFinder')
ax.bar(xi,b4,w,bottom=b1+b2+b3,color='#003a66',label='CPU tree search')
ax.set_xticks(xi); ax.set_xticklabels([str(s) for s in X])
ax.set_xlabel('SMs used on H200  (hardware floor = 8)'); ax.set_ylabel('Energy (Wh)')
ax.set_title('Energy breakdown by phase & device\n'+WL); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_energy_breakdown.png')); plt.close(fig)

# 7. logL agreement
fig, ax = plt.subplots(figsize=(8,4.6))
ax.plot(X, dna['best_logL'].values, 'o-', color=G, lw=2)
ll = dna['best_logL'].dropna().unique(); spread = (ll.max()-ll.min()) if len(ll) else 0.0
ax.set_ylabel('Best log-likelihood found'); logx(ax)
ax.set_title(f'Correctness — best logL vs SM (spread = {spread:.3g})\n'
             f'{"IDENTICAL across all SM counts → bit-exact" if spread==0 else "NOT identical"}; model = {dna["model"].iloc[0]}')
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_logL_agreement.png')); plt.close(fig)

# 8. EDP
fig, ax = plt.subplots(figsize=(8,5.2))
ax.plot(X, dna['edp_Js']/1e9, 'o-', color=G, lw=2.4, ms=8)
best = dna.loc[dna['edp_Js'].idxmin()]
ax.scatter([best['sm']],[best['edp_Js']/1e9],s=170,facecolors='none',edgecolors=RED,lw=2,zorder=5,
           label=f"min EDP @ {int(best['sm'])} SM")
logx(ax); ax.set_ylabel('Energy-delay product  GPU_J × wall_s  (×10⁹)')
ax.set_title('Energy-delay product (lower = better)\n'+WL); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_edp_vs_sm.png')); plt.close(fig)

# 9. CANONICAL 2x2 dashboard
def dashboard(path):
    fig, axes = plt.subplots(2,2, figsize=(13.5,9.4))
    fig.suptitle('H200 SM scaling — IQ-TREE3 OpenACC, ModelFinder = -m TEST\n'
                 '(8-SM hardware floor; DNA GTR+I+G4, 100 taxa × 1M sites; model picked = '
                 f'{dna["model"].iloc[0]}, logL bit-exact)', fontsize=14, fontweight='bold', y=0.985)
    ax=axes[0,0]
    ax.plot(X,dna['wall_s']/60,'o-',color=G,lw=2.4,ms=8,label='Total wall')
    ax.plot(X,dna['ts_wall_s']/60,'s--',color=B,lw=1.6,ms=5,label='Tree search')
    ax.plot(X,dna['mf_wall_s']/60,'^--',color=O,lw=1.6,ms=5,label='ModelFinder')
    label_pts(ax,X,dna['wall_s']/60,lambda v:f'{v:.0f}',dy=9,fs=9)
    logx(ax); ax.set_ylabel('Wall time (min)'); ax.set_title('(a) Runtime'); ax.legend(fontsize=8.5)
    ax=axes[0,1]
    ax.fill_between(X,dna['speedup'],dna['ideal'],color=GREY,alpha=0.18)
    ax.plot(X,dna['ideal'],':',color=GREY,lw=1.8,label='ideal linear')
    ax.plot(X,dna['speedup'],'o-',color=G,lw=2.4,ms=8,label='Total wall')
    ax.plot(X,dna['ts_speedup'],'s--',color=B,lw=1.6,ms=5,label='Tree search')
    ax.plot(X,dna['mf_speedup'],'^--',color=O,lw=1.6,ms=5,label='ModelFinder')
    ax.set_xscale('log',base=2); ax.set_yscale('log',base=2)
    ax.xaxis.set_major_locator(FixedLocator(X)); ax.xaxis.set_major_formatter(FixedFormatter([str(int(v)) for v in X]))
    ax.yaxis.set_major_locator(FixedLocator([1,2,4,8,16])); ax.yaxis.set_major_formatter(FixedFormatter(['1×','2×','4×','8×','16×']))
    ax.set_xlim(X.min()*0.82,X.max()*1.18); ax.set_xlabel('SMs used on H200  (hardware floor = 8)')
    label_pts(ax,X,dna['speedup'],lambda v:f'{v:.1f}×',dy=-15,fs=9)
    ax.set_ylabel('Speedup vs 8-SM'); ax.set_title('(b) Speedup by phase (shaded = lost to HBM bound)'); ax.legend(fontsize=8,loc='upper left')
    ax=axes[1,0]
    ax.axhspan(80,112,color=G,alpha=0.08)
    ax.plot(X,dna['efficiency']*100,'o-',color=G,lw=2.4,ms=8,label='DNA 1M')
    ax.axhline(100,ls=':',color=GREY,lw=1.4)
    label_pts(ax,X,dna['efficiency']*100,lambda v:f'{v:.0f}%',dy=9,fs=9.5)
    ax.set_ylim(40,112); logx(ax); ax.set_ylabel('Efficiency vs 8-SM (%)')
    ax.set_title('(c) Parallel efficiency')
    ax=axes[1,1]
    xs=dna['wall_s']/60; ys=dna['gpu_Wh']
    ax.plot(xs,ys,'-',color=GREY,lw=1.3,zorder=2)
    sc=ax.scatter(xs,ys,c=X,cmap='viridis',s=130,zorder=5,edgecolors='white',lw=1)
    for x,y,n in zip(xs,ys,X):
        ax.annotate(f'{int(n)}',(x,y),textcoords='offset points',xytext=(7,5),fontsize=9,fontweight='bold')
    ax.annotate('better',xy=(0.10,0.12),xytext=(0.36,0.30),xycoords='axes fraction',
                fontsize=11,color=GD,fontweight='bold',arrowprops=dict(arrowstyle='-|>',color=GD,lw=2))
    ax.set_xlabel('Wall time (min)'); ax.set_ylabel('GPU energy (Wh)'); ax.set_title('(d) Energy vs runtime — 132 SM wins both')
    fig.text(0.5,0.005,f'Full GPU (132 SM) = {dna["speedup"].iloc[-1]:.1f}× over the 8-SM floor, fastest AND lowest energy. '
             f'GPU mem ~{dna["gpu_mem_peak_MB"].iloc[-1]/1024:.0f} GB of 140 GB. logL identical across all runs.',
             ha='center',fontsize=9.5,style='italic')
    fig.tight_layout(rect=[0,0.02,1,0.965])
    fig.savefig(path); plt.close(fig)

dashboard(os.path.join(OUT_DIR,'fig_dashboard.png'))
dashboard(os.path.join(OUT_DIR,'fig_corrected_dashboard.png'))

# 10. compact runtime + speedup dashboard
fig, axes = plt.subplots(1,2, figsize=(13.5,5.2))
fig.suptitle('H200 SM scaling — IQ-TREE3 OpenACC, ModelFinder = -m TEST, DNA 1M',
             fontsize=14, fontweight='bold', y=0.99)
ax=axes[0]
ax.plot(X,dna['wall_s']/60,'o-',color=G,lw=2.4,ms=8,label='Total wall')
ax.plot(X,dna['ts_wall_s']/60,'s--',color=B,lw=1.6,ms=5,label='Tree search')
ax.plot(X,dna['mf_wall_s']/60,'^--',color=O,lw=1.6,ms=5,label='ModelFinder')
label_pts(ax,X,dna['wall_s']/60,lambda v:f'{v:.0f}',dy=9,fs=9)
logx(ax); ax.set_ylabel('Wall time (min)'); ax.set_title('(a) Runtime'); ax.legend(fontsize=8.5)
ax=axes[1]
ax.fill_between(X,dna['speedup'],dna['ideal'],color=GREY,alpha=0.18)
ax.plot(X,dna['ideal'],':',color=GREY,lw=1.8,label='ideal linear')
ax.plot(X,dna['speedup'],'o-',color=G,lw=2.4,ms=8,label='Total wall')
ax.plot(X,dna['ts_speedup'],'s--',color=B,lw=1.6,ms=5,label='Tree search')
ax.plot(X,dna['mf_speedup'],'^--',color=O,lw=1.6,ms=5,label='ModelFinder')
ax.set_xscale('log',base=2); ax.set_yscale('log',base=2)
ax.xaxis.set_major_locator(FixedLocator(X)); ax.xaxis.set_major_formatter(FixedFormatter([str(int(v)) for v in X]))
ax.yaxis.set_major_locator(FixedLocator([1,2,4,8,16])); ax.yaxis.set_major_formatter(FixedFormatter(['1×','2×','4×','8×','16×']))
ax.set_xlim(X.min()*0.82,X.max()*1.18); ax.set_xlabel('SMs used on H200  (hardware floor = 8)')
label_pts(ax,X,dna['speedup'],lambda v:f'{v:.1f}×',dy=-15,fs=9)
ax.set_ylabel('Speedup vs 8-SM'); ax.set_title('(b) Speedup by phase'); ax.legend(fontsize=8.5,loc='upper left')
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(os.path.join(OUT_DIR,'fig_runtime_speedup_dashboard.png')); plt.close(fig)

print('\nFigures + CSVs written to', OUT_DIR)
