#!/usr/bin/env python3
# 2026-06-12 — H200 SM-scaling sweep for IQ-TREE3 OpenACC
#
# Question: how does wall time / energy scale as we restrict the H200 (132 SMs)
# to N SMs via CUDA MPS active-thread-percentage? One PBS job per SM count,
# each running the SAME workload:
#     iqtree3 -s alignment_100000.phy -ninit 2 -seed 1   (AA, LG+I+G4, 100 taxa, 100k sites)
#
# Data:  /Users/u7826985/Projects/Nvidia/results/sm_sweep/sm_<N>/sm_<N>_rep<r>.log
# Out:   .../poc-gpu-likelihood-calculation_analysis/2026_06_12_sm_sweep/
#
# IMPORTANT caveat (see README): the per-SM PBS walltimes were too tight, so the
# low/mid SM jobs were KILLED before finishing all repeats. Completed runs:
#   SM=1 (1 rep), 2 (1), 33 (1), 66 (2), 132 (3).  SM=4, 8, 16 produced NO
#   complete run and are absent here. Energy from IQ-TREE's built-in Energy:
#   block (CPU=RAPL, GPU=NVML).

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_ROOT = '/Users/u7826985/Projects/Nvidia/results/sm_sweep'
OUT_DIR   = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_06_12_sm_sweep'
TOTAL_SM  = 132

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 11})
GREEN = '#76B900'   # NVIDIA green (H200)
BLUE  = '#0071C5'
ORANGE= '#F2A900'
GREY  = '#888888'

# ---------------------------------------------------------------------------
# 1. Parse logs
# ---------------------------------------------------------------------------
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
    o['mf_wall_s']  = f(MF_WALL_RE.search(txt))
    o['ts_wall_s']  = f(TS_WALL_RE.search(txt))
    o['wall_s']     = f(TOT_WALL_RE.search(txt))
    o['cpu_s']      = f(TOT_CPU_RE.search(txt))
    o['best_logL']  = f(BEST_LL_RE.search(txt))
    m = MF_EN_RE.search(txt); o['mf_cpu_J'] = f(m,1); o['mf_gpu_J'] = f(m,2)
    m = TS_EN_RE.search(txt); o['ts_cpu_J'] = f(m,1); o['ts_gpu_J'] = f(m,2)
    o['cpu_J'] = f(EN_CPU_RE.search(txt))
    o['gpu_J'] = f(EN_GPU_RE.search(txt))
    m = GPU_MEM_RE.search(txt); o['gpu_mem_peak_MB'] = f(m,1)
    # complete run == reached the final Energy block AND printed a best score
    o['complete'] = (not np.isnan(o['wall_s']) and not np.isnan(o['gpu_J'])
                     and not np.isnan(o['best_logL']))
    return o

def mean_power_from_csv(path):
    """Mean board power (W) from the nvidia-smi sampling CSV, if usable."""
    try:
        df = pd.read_csv(path)
        col = [c for c in df.columns if 'power' in c.lower()][0]
        vals = pd.to_numeric(df[col].astype(str).str.extract(r'([\d.]+)')[0], errors='coerce')
        vals = vals.dropna()
        return float(vals.mean()) if len(vals) else np.nan
    except Exception:
        return np.nan

rows = []
for d in sorted(glob.glob(os.path.join(DATA_ROOT, 'sm_*')),
                key=lambda p: int(re.search(r'sm_(\d+)$', p).group(1)) if re.search(r'sm_(\d+)$', p) else 1e9):
    msm = re.search(r'sm_(\d+)$', d)
    if not msm:
        continue
    sm = int(msm.group(1))
    for log in sorted(glob.glob(os.path.join(d, f'sm_{sm}_rep[0-9].log'))):
        rep = int(re.search(r'_rep(\d+)\.log$', log).group(1))
        rec = {'sm': sm, 'rep': rep, 'pct': round(sm / TOTAL_SM * 100, 4), 'log': log}
        rec.update(parse_log(log))
        pcsv = log.replace('.log', '_power.csv')
        rec['measured_W'] = mean_power_from_csv(pcsv) if os.path.exists(pcsv) else np.nan
        rows.append(rec)

runs = pd.DataFrame(rows).sort_values(['sm', 'rep']).reset_index(drop=True)
runs['total_J'] = runs['cpu_J'] + runs['gpu_J']
runs['gpu_W']   = runs['gpu_J'] / runs['wall_s']
runs['cpu_W']   = runs['cpu_J'] / runs['wall_s']
runs.to_csv(os.path.join(OUT_DIR, 'sm_sweep_runs.csv'), index=False)

done = runs[runs['complete']].copy()

# Aggregate per SM (mean over completed reps)
agg = (done.groupby('sm')
       .agg(reps=('rep', 'count'),
            pct=('pct', 'first'),
            wall_s=('wall_s', 'mean'),
            wall_std=('wall_s', 'std'),
            mf_wall_s=('mf_wall_s', 'mean'),
            ts_wall_s=('ts_wall_s', 'mean'),
            cpu_J=('cpu_J', 'mean'),
            gpu_J=('gpu_J', 'mean'),
            total_J=('total_J', 'mean'),
            mf_gpu_J=('mf_gpu_J', 'mean'),
            ts_gpu_J=('ts_gpu_J', 'mean'),
            mf_cpu_J=('mf_cpu_J', 'mean'),
            ts_cpu_J=('ts_cpu_J', 'mean'),
            gpu_W=('gpu_W', 'mean'),
            cpu_W=('cpu_W', 'mean'),
            measured_W=('measured_W', 'mean'),
            best_logL=('best_logL', 'first'),
            gpu_mem_peak_MB=('gpu_mem_peak_MB', 'mean'))
       .reset_index())

base = agg.loc[agg['sm'].idxmin()]            # 1-SM baseline
agg['speedup']     = base['wall_s'] / agg['wall_s']
agg['ts_speedup']  = base['ts_wall_s'] / agg['ts_wall_s']
agg['efficiency']  = agg['speedup'] / (agg['sm'] / base['sm'])
agg['edp_Js']      = agg['gpu_J'] * agg['wall_s']        # energy-delay product
agg['energy_Wh']   = agg['total_J'] / 3600.0
agg['gpu_Wh']      = agg['gpu_J'] / 3600.0
agg['cpu_Wh']      = agg['cpu_J'] / 3600.0
agg.to_csv(os.path.join(OUT_DIR, 'sm_sweep_summary.csv'), index=False)

print(runs[['sm','rep','complete','wall_s','mf_wall_s','ts_wall_s','cpu_J','gpu_J','best_logL']].to_string(index=False))
print('\n--- per-SM summary (completed runs only) ---')
print(agg[['sm','pct','reps','wall_s','speedup','efficiency','gpu_J','cpu_J','total_J','gpu_W','best_logL']].to_string(index=False))

X = agg['sm'].values
MISSING = [s for s in (4, 8, 16) if s not in set(X)]

# ---------------------------------------------------------------------------
# 2. Figures
# ---------------------------------------------------------------------------
def smx(ax):
    ax.set_xscale('log', base=2)
    ax.set_xticks(X)
    ax.set_xticklabels([str(s) for s in X])
    ax.set_xlabel('SMs enabled (of 132 on H200)')
    ax.grid(True, which='both', alpha=0.25)
    if MISSING:
        ax.text(0.5, -0.22, f"missing (walltime-killed): SM = {', '.join(map(str, MISSING))}",
                transform=ax.transAxes, ha='center', va='top', fontsize=8.5, color='firebrick')

# --- 2a. runtime breakdown vs SM -------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(X, agg['wall_s']/60, 'o-', color=GREEN, lw=2, label='Total wall')
ax.plot(X, agg['ts_wall_s']/60, 's--', color=BLUE, label='Tree search')
ax.plot(X, agg['mf_wall_s']/60, '^--', color=ORANGE, label='ModelFinder')
ax.set_ylabel('Wall-clock time (minutes)')
ax.set_title('H200 SM scaling — runtime vs SM count\nAA LG+I+G4, 100 taxa × 100k sites, -ninit 2 -seed 1')
smx(ax); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'fig_runtime_vs_sm.png')); plt.close(fig)

# --- 2b. speedup vs SM (with ideal) ----------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))
ideal = X / base['sm']
ax.plot(X, ideal, ':', color=GREY, label='Ideal linear (×SM)')
ax.plot(X, agg['speedup'], 'o-', color=GREEN, lw=2, label='Total wall speedup')
ax.plot(X, agg['ts_speedup'], 's--', color=BLUE, label='Tree-search speedup')
for x, y in zip(X, agg['speedup']):
    ax.annotate(f'{y:.1f}×', (x, y), textcoords='offset points', xytext=(4, 6), fontsize=8.5)
ax.set_yscale('log'); ax.set_ylabel('Speedup vs 1 SM')
ax.set_title('H200 SM scaling — speedup vs 1 SM (log–log)')
smx(ax); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'fig_speedup_vs_sm.png')); plt.close(fig)

# --- 2c. parallel efficiency ------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(X, agg['efficiency']*100, 'o-', color=GREEN, lw=2)
for x, y in zip(X, agg['efficiency']*100):
    ax.annotate(f'{y:.0f}%', (x, y), textcoords='offset points', xytext=(4, 6), fontsize=8.5)
ax.axhline(100, ls=':', color=GREY, label='100% (perfect scaling)')
ax.set_ylabel('Parallel efficiency = speedup / SM-ratio  (%)')
ax.set_title('H200 SM scaling — parallel efficiency (1-SM baseline)')
smx(ax); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'fig_efficiency_vs_sm.png')); plt.close(fig)

# --- 2d. energy vs SM -------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(X, agg['gpu_Wh'], 'o-', color=GREEN, lw=2, label='GPU (NVML)')
ax.plot(X, agg['cpu_Wh'], 's--', color=BLUE, label='CPU (RAPL)')
ax.plot(X, agg['energy_Wh'], '^-', color='black', lw=1.5, label='Total')
ax.set_ylabel('Energy per run (Wh)')
ax.set_title('H200 SM scaling — energy to solution vs SM count')
smx(ax); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'fig_energy_vs_sm.png')); plt.close(fig)

# --- 2e. average power vs SM ------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(X, agg['gpu_W'], 'o-', color=GREEN, lw=2, label='GPU avg (NVML energy / wall)')
if agg['measured_W'].notna().any():
    ax.plot(X, agg['measured_W'], 'x--', color='darkgreen', label='GPU avg (nvidia-smi sampled)')
ax.plot(X, agg['cpu_W'], 's--', color=BLUE, label='CPU avg (RAPL)')
ax.set_ylabel('Average power (W)')
ax.set_title('H200 SM scaling — average power vs SM count')
smx(ax); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'fig_power_vs_sm.png')); plt.close(fig)

# --- 2f. energy breakdown (stacked: phase × device) ------------------------
fig, ax = plt.subplots(figsize=(8, 5))
xi = np.arange(len(X)); w = 0.6
b1 = agg['mf_gpu_J']/3600; b2 = agg['ts_gpu_J']/3600
b3 = agg['mf_cpu_J']/3600; b4 = agg['ts_cpu_J']/3600
ax.bar(xi, b1, w, color=GREEN, label='GPU ModelFinder')
ax.bar(xi, b2, w, bottom=b1, color='#3f6600', label='GPU tree search')
ax.bar(xi, b3, w, bottom=b1+b2, color=BLUE, label='CPU ModelFinder')
ax.bar(xi, b4, w, bottom=b1+b2+b3, color='#003a66', label='CPU tree search')
ax.set_xticks(xi); ax.set_xticklabels([str(s) for s in X])
ax.set_xlabel('SMs enabled (of 132)'); ax.set_ylabel('Energy (Wh)')
ax.set_title('H200 SM scaling — energy breakdown by phase & device')
ax.legend(fontsize=9)
if MISSING:
    ax.text(0.5, -0.16, f"missing (walltime-killed): SM = {', '.join(map(str, MISSING))}",
            transform=ax.transAxes, ha='center', va='top', fontsize=8.5, color='firebrick')
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'fig_energy_breakdown.png')); plt.close(fig)

# --- 2g. logL agreement (correctness: SM-limiting must not change results) --
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(X, done.groupby('sm')['best_logL'].first().values, 'o-', color=GREEN, lw=2)
ll = done['best_logL'].dropna().unique()
ax.set_ylabel('Best log-likelihood found')
spread = (ll.max() - ll.min()) if len(ll) else 0.0
ax.set_title(f'Correctness — best logL vs SM (spread = {spread:.3g})\n'
             f'{"IDENTICAL across all SM counts → bit-exact" if spread == 0 else "NOT identical — investigate"}')
smx(ax)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'fig_logL_agreement.png')); plt.close(fig)

# --- 2h. energy-delay product (find the efficient operating point) ----------
fig, ax = plt.subplots(figsize=(7.5, 5))
edp = agg['edp_Js'] / 1e9
ax.plot(X, edp, 'o-', color=GREEN, lw=2)
best = agg.loc[agg['edp_Js'].idxmin()]
ax.scatter([best['sm']], [best['edp_Js']/1e9], s=160, facecolors='none', edgecolors='firebrick',
           lw=2, zorder=5, label=f"min EDP @ {int(best['sm'])} SM")
ax.set_ylabel('Energy-delay product  GPU_J × wall_s  (×10⁹)')
ax.set_title('H200 SM scaling — energy-delay product (lower = better)')
smx(ax); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'fig_edp_vs_sm.png')); plt.close(fig)

print('\nFigures + CSVs written to', OUT_DIR)
print('Missing SM points (no complete run):', MISSING or 'none')
