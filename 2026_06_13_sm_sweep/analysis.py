#!/usr/bin/env python3
# 2026-06-13 — H200 SM-scaling sweep for IQ-TREE3 OpenACC (FULL, all 8 points)
#
# One PBS job per SM count, REPEATS=1, same workload each time:
#     iqtree3 -s alignment_100000.phy -ninit 2 -seed 1   (AA, LG+I+G4, 100 taxa, 100k sites)
# SM count restricted via CUDA MPS active-thread-percentage = SM/132*100.
#
# Data:  /Users/u7826985/Projects/Nvidia/results/2026_06_13_sn_sweep/sm_sweep/sm_<N>/
# Out:   .../poc-gpu-likelihood-calculation_analysis/2026_06_13_sm_sweep/
#
# HEADLINE: SM = 1,2,4,8 are NOT distinct hardware allocations. MPS has a coarse
# minimum SM partition on H200 (~8 SMs), so 0.76%-6.06% thread requests all map to
# the same floor -> identical wall time, GPU energy AND GPU power (~147 W). Real
# scaling is only observable from 16 SMs up. Energy from IQ-TREE's Energy: block
# (CPU=RAPL, GPU=NVML).

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_ROOT = '/Users/u7826985/Projects/Nvidia/results/2026_06_13_sn_sweep/sm_sweep'
OUT_DIR   = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_13_sm_sweep'
TOTAL_SM  = 132

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 11})
GREEN='#76B900'; BLUE='#0071C5'; ORANGE='#F2A900'; GREY='#888888'; RED='#C0392B'

MF_WALL_RE = re.compile(r'Wall-clock time for ModelFinder:\s+([\d.]+)\s+seconds')
TS_WALL_RE = re.compile(r'Wall-clock time used for tree search:\s+([\d.]+)\s+sec')
TOT_WALL_RE= re.compile(r'Total wall-clock time used:\s+([\d.]+)\s+sec')
TOT_CPU_RE = re.compile(r'Total CPU time used:\s+([\d.]+)\s+sec')
BEST_LL_RE = re.compile(r'BEST SCORE FOUND\s*:\s+(-?[\d.]+)')
MF_EN_RE   = re.compile(r'Energy used for ModelFinder:\s+CPU\s+([\d.]+)\s+J,\s+GPU\s+([\d.]+)\s+J')
TS_EN_RE   = re.compile(r'Energy used for tree search:\s+CPU\s+([\d.]+)\s+J,\s+GPU\s+([\d.]+)\s+J')
EN_CPU_RE  = re.compile(r'Energy:\s*\n\s*CPU:\s+([\d.]+)\s+J')
EN_GPU_RE  = re.compile(r'\n\s*GPU:\s+([\d.]+)\s+J')
GPU_MEM_RE = re.compile(r'GPU mem:\s+([\d.]+)\s*/\s*([\d.]+)\s+MB')

def f(m, g=1): return float(m.group(g)) if m else np.nan

def parse_log(path):
    txt = open(path, errors='ignore').read()
    o = {}
    o['mf_wall_s'] = f(MF_WALL_RE.search(txt)); o['ts_wall_s'] = f(TS_WALL_RE.search(txt))
    o['wall_s']    = f(TOT_WALL_RE.search(txt)); o['cpu_s']    = f(TOT_CPU_RE.search(txt))
    o['best_logL'] = f(BEST_LL_RE.search(txt))
    m = MF_EN_RE.search(txt); o['mf_cpu_J']=f(m,1); o['mf_gpu_J']=f(m,2)
    m = TS_EN_RE.search(txt); o['ts_cpu_J']=f(m,1); o['ts_gpu_J']=f(m,2)
    o['cpu_J'] = f(EN_CPU_RE.search(txt)); o['gpu_J'] = f(EN_GPU_RE.search(txt))
    m = GPU_MEM_RE.search(txt); o['gpu_mem_peak_MB'] = f(m,1)
    o['complete'] = (not np.isnan(o['wall_s']) and not np.isnan(o['gpu_J'])
                     and not np.isnan(o['best_logL']))
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

rows = []
for d in sorted(glob.glob(os.path.join(DATA_ROOT, 'sm_*')),
                key=lambda p: int(re.search(r'sm_(\d+)$', p).group(1)) if re.search(r'sm_(\d+)$', p) else 1e9):
    msm = re.search(r'sm_(\d+)$', d)
    if not msm: continue
    sm = int(msm.group(1))
    for log in sorted(glob.glob(os.path.join(d, f'sm_{sm}_rep[0-9].log'))):
        rep = int(re.search(r'_rep(\d+)\.log$', log).group(1))
        rec = {'sm': sm, 'rep': rep, 'pct': round(sm/TOTAL_SM*100, 4), 'log': log}
        rec.update(parse_log(log))
        pcsv = log.replace('.log', '_power.csv')
        rec['measured_W'] = mean_power_from_csv(pcsv) if os.path.exists(pcsv) else np.nan
        rows.append(rec)

runs = pd.DataFrame(rows).sort_values(['sm','rep']).reset_index(drop=True)
runs['total_J'] = runs['cpu_J'] + runs['gpu_J']
runs['gpu_W'] = runs['gpu_J']/runs['wall_s']; runs['cpu_W'] = runs['cpu_J']/runs['wall_s']
runs.to_csv(os.path.join(OUT_DIR, 'sm_sweep_runs.csv'), index=False)

done = runs[runs['complete']].copy()
agg = (done.groupby('sm').agg(
        reps=('rep','count'), pct=('pct','first'),
        wall_s=('wall_s','mean'), mf_wall_s=('mf_wall_s','mean'), ts_wall_s=('ts_wall_s','mean'),
        cpu_J=('cpu_J','mean'), gpu_J=('gpu_J','mean'), total_J=('total_J','mean'),
        mf_gpu_J=('mf_gpu_J','mean'), ts_gpu_J=('ts_gpu_J','mean'),
        mf_cpu_J=('mf_cpu_J','mean'), ts_cpu_J=('ts_cpu_J','mean'),
        gpu_W=('gpu_W','mean'), cpu_W=('cpu_W','mean'), measured_W=('measured_W','mean'),
        best_logL=('best_logL','first'), gpu_mem_peak_MB=('gpu_mem_peak_MB','mean')).reset_index())

base = agg.loc[agg['sm'].idxmin()]
agg['speedup']    = base['wall_s']/agg['wall_s']
agg['ts_speedup'] = base['ts_wall_s']/agg['ts_wall_s']
agg['efficiency'] = agg['speedup']/(agg['sm']/base['sm'])
agg['edp_Js']     = agg['gpu_J']*agg['wall_s']
agg['energy_Wh']  = agg['total_J']/3600; agg['gpu_Wh']=agg['gpu_J']/3600; agg['cpu_Wh']=agg['cpu_J']/3600

# --- detect the MPS minimum-partition plateau (flat GPU power at the low end) ---
agg = agg.sort_values('sm').reset_index(drop=True)
pmin = agg['measured_W'].iloc[0]
plateau = agg.loc[(agg['measured_W'] <= pmin*1.05), 'sm'].tolist()
# keep only the contiguous low-SM run
plat = []
for s in agg['sm']:
    if s in plateau: plat.append(s)
    else: break
agg['mps_plateau'] = agg['sm'].isin(plat)
agg.to_csv(os.path.join(OUT_DIR, 'sm_sweep_summary.csv'), index=False)

print(agg[['sm','pct','wall_s','speedup','efficiency','gpu_J','cpu_J','gpu_W','measured_W','mps_plateau','best_logL']].to_string(index=False))
print('\nMPS plateau (collapsed to one allocation):', plat)

X = agg['sm'].values
def shade_plateau(ax):
    if len(plat) >= 2:
        ax.axvspan(plat[0]*0.9, plat[-1]*1.1, color=RED, alpha=0.07, zorder=0)
        ax.text(np.sqrt(plat[0]*plat[-1]), ax.get_ylim()[1],
                'MPS min-partition\n(1/2/4/8 → same ~8 SMs)', ha='center', va='top',
                fontsize=8.5, color=RED)
def smx(ax):
    ax.set_xscale('log', base=2); ax.set_xticks(X); ax.set_xticklabels([str(s) for s in X])
    ax.set_xlabel('SMs requested via MPS (of 132 on H200)'); ax.grid(True, which='both', alpha=0.25)

# 1. runtime
fig, ax = plt.subplots(figsize=(7.8,5))
ax.plot(X, agg['wall_s']/60, 'o-', color=GREEN, lw=2, label='Total wall')
ax.plot(X, agg['ts_wall_s']/60, 's--', color=BLUE, label='Tree search')
ax.plot(X, agg['mf_wall_s']/60, '^--', color=ORANGE, label='ModelFinder')
ax.set_ylabel('Wall-clock time (minutes)')
ax.set_title('H200 SM scaling — runtime vs SM\nAA LG+I+G4, 100 taxa × 100k sites, -ninit 2 -seed 1')
smx(ax); shade_plateau(ax); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_runtime_vs_sm.png')); plt.close(fig)

# 2. GPU power — the evidence for MPS quantization
fig, ax = plt.subplots(figsize=(7.8,5))
ax.plot(X, agg['measured_W'], 'o-', color=GREEN, lw=2, label='GPU board power (nvidia-smi sampled)')
ax.plot(X, agg['gpu_W'], 'x--', color='darkgreen', label='GPU avg (NVML energy / wall)')
for x,y in zip(X, agg['measured_W']):
    ax.annotate(f'{y:.0f}W',(x,y),textcoords='offset points',xytext=(3,6),fontsize=8)
ax.set_ylabel('Average GPU power (W)')
ax.set_title('H200 SM scaling — GPU power vs SM\nflat 1→8 SM = MPS gives the same physical allocation')
smx(ax); shade_plateau(ax); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_power_vs_sm.png')); plt.close(fig)

# 3. speedup
fig, ax = plt.subplots(figsize=(7.8,5))
ax.plot(X, X/base['sm'], ':', color=GREY, label='Ideal linear (×SM)')
ax.plot(X, agg['speedup'], 'o-', color=GREEN, lw=2, label='Total wall speedup')
for x,y in zip(X, agg['speedup']):
    ax.annotate(f'{y:.1f}×',(x,y),textcoords='offset points',xytext=(4,6),fontsize=8.5)
ax.set_yscale('log'); ax.set_ylabel('Speedup vs 1-SM request')
ax.set_title('H200 SM scaling — speedup (log–log)')
smx(ax); shade_plateau(ax); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_speedup_vs_sm.png')); plt.close(fig)

# 4. efficiency
fig, ax = plt.subplots(figsize=(7.8,5))
ax.plot(X, agg['efficiency']*100, 'o-', color=GREEN, lw=2)
for x,y in zip(X, agg['efficiency']*100):
    ax.annotate(f'{y:.0f}%',(x,y),textcoords='offset points',xytext=(4,6),fontsize=8.5)
ax.axhline(100, ls=':', color=GREY, label='100%')
ax.set_ylabel('Parallel efficiency = speedup / SM-ratio (%)')
ax.set_title('H200 SM scaling — parallel efficiency (nominal SM, 1-SM baseline)')
smx(ax); shade_plateau(ax); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_efficiency_vs_sm.png')); plt.close(fig)

# 5. energy
fig, ax = plt.subplots(figsize=(7.8,5))
ax.plot(X, agg['gpu_Wh'], 'o-', color=GREEN, lw=2, label='GPU (NVML)')
ax.plot(X, agg['cpu_Wh'], 's--', color=BLUE, label='CPU (RAPL)')
ax.plot(X, agg['energy_Wh'], '^-', color='black', lw=1.5, label='Total')
ax.set_ylabel('Energy per run (Wh)')
ax.set_title('H200 SM scaling — energy to solution vs SM')
smx(ax); shade_plateau(ax); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_energy_vs_sm.png')); plt.close(fig)

# 6. energy breakdown
fig, ax = plt.subplots(figsize=(8.2,5))
xi=np.arange(len(X)); w=0.6
b1=agg['mf_gpu_J']/3600; b2=agg['ts_gpu_J']/3600; b3=agg['mf_cpu_J']/3600; b4=agg['ts_cpu_J']/3600
ax.bar(xi,b1,w,color=GREEN,label='GPU ModelFinder')
ax.bar(xi,b2,w,bottom=b1,color='#3f6600',label='GPU tree search')
ax.bar(xi,b3,w,bottom=b1+b2,color=BLUE,label='CPU ModelFinder')
ax.bar(xi,b4,w,bottom=b1+b2+b3,color='#003a66',label='CPU tree search')
ax.set_xticks(xi); ax.set_xticklabels([str(s) for s in X])
ax.set_xlabel('SMs requested via MPS (of 132)'); ax.set_ylabel('Energy (Wh)')
ax.set_title('H200 SM scaling — energy breakdown by phase & device'); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_energy_breakdown.png')); plt.close(fig)

# 7. logL agreement
fig, ax = plt.subplots(figsize=(7.8,4.5))
ax.plot(X, done.groupby('sm')['best_logL'].first().values, 'o-', color=GREEN, lw=2)
ll = done['best_logL'].dropna().unique(); spread = (ll.max()-ll.min()) if len(ll) else 0.0
ax.set_ylabel('Best log-likelihood found')
ax.set_title(f'Correctness — best logL vs SM (spread = {spread:.3g})\n'
             f'{"IDENTICAL across all SM counts → bit-exact" if spread==0 else "NOT identical"}')
smx(ax)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_logL_agreement.png')); plt.close(fig)

# 8. EDP
fig, ax = plt.subplots(figsize=(7.8,5))
ax.plot(X, agg['edp_Js']/1e9, 'o-', color=GREEN, lw=2)
best = agg.loc[agg['edp_Js'].idxmin()]
ax.scatter([best['sm']],[best['edp_Js']/1e9],s=160,facecolors='none',edgecolors=RED,lw=2,zorder=5,
           label=f"min EDP @ {int(best['sm'])} SM")
ax.set_ylabel('Energy-delay product  GPU_J × wall_s  (×10⁹)')
ax.set_title('H200 SM scaling — energy-delay product (lower = better)')
smx(ax); shade_plateau(ax); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'fig_edp_vs_sm.png')); plt.close(fig)

print('\nFigures + CSVs written to', OUT_DIR)
