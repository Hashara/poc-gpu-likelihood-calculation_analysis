#!/usr/bin/env python3
"""
Analysis for R_v3 results: VANILA vs OPENACC across all models.
Generates:
  1. ll_comparison.csv — raw data
  2. correctness_summary.csv — per-model summary
  3. correctness_matrix.png — heatmap of |LL diff| per model/tree
  4. ll_scatter.png — VANILA vs OPENACC scatter
  5. Per-model +R4 comparison (styled like v2's 3-way plots)
  6. v2_vs_v3_R4_comparison.png — did the fix help?
  7. all_models_overview.png — all models at a glance
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

results_v3 = "/Users/u7826985/Projects/Nvidia/results/2026_03_19_complexdata_bugfixes_R_v3"
results_v2 = "/Users/u7826985/Projects/Nvidia/results/2026_03_19_complexdata_bugfixes_R_v2"
out_dir = "/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_03_19_complexdata_bugfixes_R_v3"

plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 150, 'font.size': 10})

COLORS = {
    'VANILA': '#4C72B0',
    'OPENACC': '#DD8452',
    'v2_OPENACC': '#C44E52',
    'v3_OPENACC': '#55A868',
}

iqtree_ll_pattern = re.compile(r'Log-likelihood of the tree:\s+([-\d.]+)')
iqtree_rates_pattern = re.compile(r'Site proportion and rates:\s+(.*)')
iqtree_time_pattern = re.compile(r'Total wall-clock time used:\s+([\d.]+)\s+seconds')
iqtree_underflow_pattern = re.compile(r'WARNING: (\d+) .* numerical underflow')


def classify_backend(filename):
    if 'VANILA_NONREV' in filename or 'VANILA__NONREV' in filename:
        return 'VANILA_NONREV'
    elif 'OPENACC' in filename:
        return 'OPENACC'
    elif 'VANILA' in filename:
        return 'VANILA'
    return None


def parse_results(base_path, tag=''):
    """Parse all .iqtree results under base_path."""
    rows = []
    for data_type in ['DNA', 'AA']:
        data_dir = os.path.join(base_path, data_type)
        if not os.path.isdir(data_dir):
            continue
        for topology in ['rooted', 'unrooted']:
            topo_dir = os.path.join(data_dir, topology)
            if not os.path.isdir(topo_dir):
                continue
            for model in sorted(os.listdir(topo_dir)):
                model_dir = os.path.join(topo_dir, model)
                if not os.path.isdir(model_dir):
                    continue
                for tree in sorted(os.listdir(model_dir)):
                    tree_dir = os.path.join(model_dir, tree)
                    if not os.path.isdir(tree_dir):
                        continue
                    iqtree_files = [f for f in os.listdir(tree_dir) if f.endswith('.iqtree')]

                    for f in iqtree_files:
                        backend = classify_backend(f)
                        if backend is None:
                            continue
                        content = open(os.path.join(tree_dir, f)).read()

                        row = {
                            'data_type': data_type, 'topology': topology,
                            'model': model, 'tree': tree,
                            'backend': backend, 'tag': tag,
                        }

                        m = iqtree_ll_pattern.search(content)
                        if m:
                            row['ll'] = float(m.group(1))
                        m = iqtree_rates_pattern.search(content)
                        if m:
                            row['rates'] = m.group(1).strip()
                        m = iqtree_time_pattern.search(content)
                        if m:
                            row['time'] = float(m.group(1))
                        m = iqtree_underflow_pattern.search(content)
                        row['underflows'] = int(m.group(1)) if m else 0

                        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# 1. Parse v3 results
# ============================================================
print("Parsing v3 results...")
df_v3 = parse_results(results_v3, tag='v3')
print(f"  {len(df_v3)} rows")

# Also parse v2 for comparison
print("Parsing v2 results...")
df_v2 = parse_results(results_v2, tag='v2')
print(f"  {len(df_v2)} rows")

# ============================================================
# 2. Build comparison table (wide format) for v3
# ============================================================
def pivot_and_diff(df, tag_filter=None):
    """Pivot to wide format with VANILA/OPENACC columns and compute diffs."""
    d = df if tag_filter is None else df[df['tag'] == tag_filter]

    vanila = d[d['backend'] == 'VANILA'][['data_type','topology','model','tree','ll','time','underflows','rates']].rename(
        columns={'ll':'ll_VANILA','time':'time_VANILA','underflows':'uf_VANILA','rates':'rates_VANILA'})
    openacc = d[d['backend'] == 'OPENACC'][['data_type','topology','model','tree','ll','time','underflows','rates']].rename(
        columns={'ll':'ll_OPENACC','time':'time_OPENACC','underflows':'uf_OPENACC','rates':'rates_OPENACC'})

    merged = pd.merge(vanila, openacc, on=['data_type','topology','model','tree'], how='outer')
    merged['ll_diff'] = (merged['ll_VANILA'] - merged['ll_OPENACC']).abs()
    return merged

wide_v3 = pivot_and_diff(df_v3, 'v3')
wide_v2 = pivot_and_diff(df_v2, 'v2')

# Save CSV
wide_v3.to_csv(os.path.join(out_dir, 'll_comparison.csv'), index=False)
print(f"Saved ll_comparison.csv ({len(wide_v3)} rows)")

# ============================================================
# 3. Correctness summary
# ============================================================
summary_rows = []
for (dt, topo, model), grp in wide_v3.groupby(['data_type','topology','model']):
    diffs = grp['ll_diff'].dropna()
    n = len(diffs)
    n_exact = (diffs < 1e-4).sum()
    summary_rows.append({
        'data_type': dt, 'topology': topo, 'model': model,
        'n_trees': n,
        'n_exact': int(n_exact),
        'max_diff': diffs.max(),
        'mean_diff': diffs.mean(),
        'median_diff': diffs.median(),
        'status': '✅ PASS' if diffs.max() < 0.01 else ('⚠️ WARN' if diffs.max() < 1.0 else '❌ FAIL'),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(out_dir, 'correctness_summary.csv'), index=False)
print(f"Saved correctness_summary.csv")
print(summary_df.to_string(index=False))

# ============================================================
# 4. Correctness heatmap (all models)
# ============================================================
def make_correctness_heatmap(wide, save_path, title_suffix=''):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    for idx, (dt, topo) in enumerate([('AA','rooted'),('AA','unrooted'),
                                       ('DNA','rooted'),('DNA','unrooted')]):
        ax = axes[idx//2][idx%2]
        subset = wide[(wide['data_type']==dt) & (wide['topology']==topo)].copy()
        if len(subset) == 0:
            ax.set_title(f'{dt}/{topo} — no data')
            continue

        # Sort trees numerically
        subset['tree_num'] = subset['tree'].str.extract(r'(\d+)').astype(int)
        subset = subset.sort_values(['model','tree_num'])

        models = sorted(subset['model'].unique())
        trees = sorted(subset['tree_num'].unique())

        matrix = np.full((len(models), len(trees)), np.nan)
        for i, model in enumerate(models):
            for j, tn in enumerate(trees):
                row = subset[(subset['model']==model) & (subset['tree_num']==tn)]
                if len(row) > 0 and pd.notna(row.iloc[0]['ll_diff']):
                    matrix[i,j] = max(row.iloc[0]['ll_diff'], 1e-10)

        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r',
                        norm=LogNorm(vmin=1e-10, vmax=max(100, np.nanmax(matrix)+1)))
        ax.set_xticks(range(len(trees)))
        ax.set_xticklabels([f'T{t}' for t in trees], fontsize=7)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=8)
        ax.set_title(f'{dt} / {topo}')
        plt.colorbar(im, ax=ax, label='|LL diff|', shrink=0.8)

        # Annotate
        for i in range(len(models)):
            for j in range(len(trees)):
                v = matrix[i,j]
                if np.isnan(v):
                    continue
                txt = '0.0' if v < 1e-6 else f'{v:.2f}' if v >= 0.01 else f'{v:.4f}'
                color = 'black' if v < 1.0 else 'white'
                ax.text(j, i, txt, ha='center', va='center', fontsize=5, color=color)

    fig.suptitle(f'Correctness Matrix: |VANILA − OPENACC| {title_suffix}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

make_correctness_heatmap(wide_v3, os.path.join(out_dir, 'correctness_matrix.png'), '(v3)')

# ============================================================
# 5. LL Scatter plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for idx, dt in enumerate(['AA', 'DNA']):
    ax = axes[idx]
    subset = wide_v3[wide_v3['data_type'] == dt].dropna(subset=['ll_VANILA','ll_OPENACC'])
    r4 = subset[subset['model'].str.contains('R4')]
    other = subset[~subset['model'].str.contains('R4')]

    ax.scatter(other['ll_VANILA'], other['ll_OPENACC'], c='#4C72B0', s=30, alpha=0.7,
               label='Other models', edgecolors='gray', linewidth=0.3)
    ax.scatter(r4['ll_VANILA'], r4['ll_OPENACC'], c='#e74c3c', s=50, alpha=0.9,
               marker='^', label='+R4', edgecolors='black', linewidth=0.5)

    all_ll = np.concatenate([subset['ll_VANILA'].values, subset['ll_OPENACC'].values])
    mn, mx = all_ll.min() - 2, all_ll.max() + 2
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_xlabel('VANILA LL'); ax.set_ylabel('OPENACC LL')
    ax.set_title(f'{dt}: VANILA vs OPENACC')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

    # Label outliers
    for _, r in subset.iterrows():
        if r['ll_diff'] > 0.5:
            ax.annotate(f"{r['model']}\nT{r['tree'].replace('tree_','')}",
                        (r['ll_VANILA'], r['ll_OPENACC']),
                        textcoords='offset points', xytext=(8,-8), fontsize=6, color='red')

plt.suptitle('VANILA vs OPENACC Log-Likelihood (v3)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'll_scatter.png'), bbox_inches='tight')
plt.close()
print("Saved: ll_scatter.png")

# ============================================================
# 6. V2 vs V3 comparison for +R4
# ============================================================
def make_v2_v3_comparison(save_path):
    """Compare R4 diffs between v2 and v3 to see if the fix helped."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, (dt, topo) in enumerate([('AA','rooted'),('AA','unrooted'),
                                       ('DNA','rooted'),('DNA','unrooted')]):
        ax = axes[idx//2][idx%2]

        r4_v2 = wide_v2[(wide_v2['data_type']==dt) & (wide_v2['topology']==topo) &
                          (wide_v2['model'].str.contains('R4'))].copy()
        r4_v3 = wide_v3[(wide_v3['data_type']==dt) & (wide_v3['topology']==topo) &
                          (wide_v3['model'].str.contains('R4'))].copy()

        if len(r4_v2) == 0 and len(r4_v3) == 0:
            ax.set_title(f'{dt}/{topo}/+R4 — no data')
            continue

        for d, label, color in [(r4_v2, 'v2 (before fix)', COLORS['v2_OPENACC']),
                                 (r4_v3, 'v3 (after fix)', COLORS['v3_OPENACC'])]:
            if len(d) == 0:
                continue
            d = d.sort_values('tree', key=lambda x: x.str.extract(r'(\d+)')[0].astype(int))
            trees = [t.replace('tree_','') for t in d['tree'].values]
            x = np.arange(len(trees))
            diffs = d['ll_diff'].values
            ax.bar(x + (0.2 if 'v3' in label else -0.2), diffs, 0.35,
                   label=label, color=color, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([f'T{t}' for t in trees], fontsize=8)

        ax.axhline(y=0.01, color='green', linestyle='--', alpha=0.4, label='0.01 threshold')
        ax.set_ylabel('|LL Difference|')
        ax.set_title(f'{dt} / {topo} / +R4')
        ax.legend(fontsize=7)
        ax.set_yscale('symlog', linthresh=1e-4)

    fig.suptitle('v2 vs v3: +R4 |VANILA − OPENACC| Improvement', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

make_v2_v3_comparison(os.path.join(out_dir, 'v2_vs_v3_R4_comparison.png'))

# ============================================================
# 7. Per-combo R4 detail plots (styled like v2's 3-way)
# ============================================================
def make_r4_detail(data_type, topology, model, save_path):
    """Create detailed R4 comparison: VANILA vs OPENACC."""
    subset = wide_v3[(wide_v3['data_type']==data_type) & (wide_v3['topology']==topology) &
                      (wide_v3['model']==model)].copy()
    subset = subset.sort_values('tree', key=lambda x: x.str.extract(r'(\d+)')[0].astype(int))

    if len(subset) == 0:
        print(f"  No data for {data_type}/{topology}/{model}")
        return

    trees = subset['tree'].values
    tree_labels = [t.replace('tree_','') for t in trees]
    n = len(trees)
    x = np.arange(n)

    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 1.3],
                           hspace=0.35, wspace=0.3)

    # Panel 1: LL per tree
    ax1 = fig.add_subplot(gs[0, :])
    width = 0.35
    ax1.bar(x - width/2, subset['ll_VANILA'].values, width,
            label='VANILA', color=COLORS['VANILA'], alpha=0.85)
    ax1.bar(x + width/2, subset['ll_OPENACC'].values, width,
            label='OPENACC', color=COLORS['OPENACC'], alpha=0.85)
    ax1.set_xlabel('Tree')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Final Log-Likelihood per Tree')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'tree_{l}' for l in tree_labels])
    ax1.legend()

    # Panel 2: Diff magnitude
    ax2 = fig.add_subplot(gs[1, 0])
    colors_bar = ['#d5f5e3' if d < 1e-4 else '#fdebd0' if d < 1.0 else '#fadbd8'
                   for d in subset['ll_diff'].values]
    ax2.bar(x, subset['ll_diff'].values, color=colors_bar, edgecolor='gray', linewidth=0.5)
    ax2.axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='0.01 threshold')
    ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='1.0 threshold')
    ax2.set_yscale('symlog', linthresh=1e-6)
    ax2.set_xlabel('Tree')
    ax2.set_ylabel('|LL Difference|')
    ax2.set_title('|VANILA − OPENACC| per Tree')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'T{l}' for l in tree_labels])
    ax2.legend(fontsize=7)

    # Panel 3: Rate categories
    ax3 = fig.add_subplot(gs[1, 1])

    def parse_rates(rate_str):
        if pd.isna(rate_str):
            return []
        matches = re.findall(r'\(([\d.]+),([\d.]+)\)', rate_str)
        return [(float(p), float(r)) for p, r in matches]

    for backend, label, color in [('VANILA', 'VANILA', COLORS['VANILA']),
                                   ('OPENACC', 'OPENACC', COLORS['OPENACC'])]:
        rates_col = f'rates_{backend}'
        if rates_col not in subset.columns:
            continue
        max_rates = []
        for _, r in subset.iterrows():
            parsed = parse_rates(r.get(rates_col))
            if parsed:
                max_rates.append(max(rate for _, rate in parsed))
            else:
                max_rates.append(np.nan)
        if any(not np.isnan(v) for v in max_rates):
            offset = -0.2 if backend == 'VANILA' else 0.2
            ax3.bar(x + offset, max_rates, 0.35, label=label, color=color, alpha=0.85)

    ax3.set_xlabel('Tree')
    ax3.set_ylabel('Max Rate Category')
    ax3.set_title('EM-Estimated Max Rate by Kernel Type')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'T{l}' for l in tree_labels])
    ax3.legend(fontsize=7)

    # Panel 4: Scatter
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(subset['ll_VANILA'], subset['ll_OPENACC'],
                c=COLORS['OPENACC'], s=60, alpha=0.8, edgecolors='gray', linewidth=0.5)
    all_ll = np.concatenate([subset['ll_VANILA'].dropna().values, subset['ll_OPENACC'].dropna().values])
    ll_min, ll_max = all_ll.min() - 2, all_ll.max() + 2
    ax4.plot([ll_min, ll_max], [ll_min, ll_max], 'r--', alpha=0.5, linewidth=1, label='Perfect match')
    ax4.set_xlim(ll_min, ll_max); ax4.set_ylim(ll_min, ll_max)
    ax4.set_xlabel('VANILA LL'); ax4.set_ylabel('OPENACC LL')
    ax4.set_title('VANILA vs OPENACC (should be on diagonal)')
    ax4.legend(fontsize=8)
    ax4.set_aspect('equal')
    for _, r in subset.iterrows():
        if r['ll_diff'] >= 0.5:
            ax4.annotate(f"T{r['tree'].replace('tree_','')} (Δ={r['ll_diff']:.2f})",
                         (r['ll_VANILA'], r['ll_OPENACC']),
                         textcoords='offset points', xytext=(8,-8), fontsize=7,
                         color='red', fontweight='bold')

    # Panel 5: Summary table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    headers = ['Tree', 'VANILA', 'OPENACC', '|Diff|']
    table_data = []
    for _, r in subset.iterrows():
        d = r['ll_diff'] if pd.notna(r['ll_diff']) else 0
        table_data.append([
            r['tree'],
            f"{r['ll_VANILA']:.4f}" if pd.notna(r['ll_VANILA']) else 'N/A',
            f"{r['ll_OPENACC']:.4f}" if pd.notna(r['ll_OPENACC']) else 'N/A',
            f"{d:.6f}",
        ])
    max_d = subset['ll_diff'].max() if 'll_diff' in subset.columns else 0
    table_data.append(['MAX |Δ|', '', '', f"{max_d:.6f}"])

    table = ax5.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.2)

    for i in range(len(table_data)):
        if i < len(subset):
            val = subset.iloc[i]['ll_diff'] if pd.notna(subset.iloc[i]['ll_diff']) else 0
        else:
            val = max_d
        color = '#d5f5e3' if val < 1e-4 else '#fdebd0' if val < 1.0 else '#fadbd8'
        table[i+1, 3].set_facecolor(color)

    ax5.set_title('Summary Table', fontsize=10, pad=10)

    fig.suptitle(f'{data_type} / {topology} / {model} — VANILA vs OPENACC (v3)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# Generate R4 detail plots
for dt, model in [('DNA','GTR+R4'), ('AA','LG+R4')]:
    for topo in ['rooted', 'unrooted']:
        fname = f'vanila_openacc_comparison_{dt}_{model.replace("+","_")}_{topo}.png'
        make_r4_detail(dt, topo, model, os.path.join(out_dir, fname))

# ============================================================
# 8. All-models overview
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
for idx, (dt, topo) in enumerate([('AA','rooted'),('AA','unrooted'),
                                   ('DNA','rooted'),('DNA','unrooted')]):
    ax = axes[idx//2][idx%2]
    subset = wide_v3[(wide_v3['data_type']==dt) & (wide_v3['topology']==topo)].copy()
    if len(subset) == 0:
        ax.set_title(f'{dt}/{topo} — no data')
        continue

    models = sorted(subset['model'].unique())
    model_diffs = []
    for m in models:
        ms = subset[subset['model']==m]
        diffs = ms['ll_diff'].dropna()
        model_diffs.append({
            'model': m,
            'max': diffs.max() if len(diffs) > 0 else 0,
            'mean': diffs.mean() if len(diffs) > 0 else 0,
            'n_exact': (diffs < 1e-4).sum() if len(diffs) > 0 else 0,
            'n_total': len(diffs),
        })
    md = pd.DataFrame(model_diffs)
    x = np.arange(len(md))
    colors_bar = ['#d5f5e3' if d < 0.01 else '#fdebd0' if d < 1.0 else '#fadbd8'
                   for d in md['max'].values]
    ax.bar(x, md['max'].values, color=colors_bar, edgecolor='gray', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(md['model'].values, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Max |LL Diff|')
    ax.set_title(f'{dt} / {topo}')
    ax.set_yscale('symlog', linthresh=1e-6)
    ax.axhline(y=0.01, color='green', linestyle='--', alpha=0.5, linewidth=0.8)

    # Label bars
    for i, row in md.iterrows():
        v = row['max']
        txt = '0.0' if v < 1e-6 else f'{v:.4f}' if v < 1 else f'{v:.1f}'
        ax.text(i, v * 1.3 if v > 1e-6 else 1e-7, txt, ha='center', va='bottom', fontsize=6)

fig.suptitle('Max |VANILA − OPENACC| by Model (v3)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'all_models_overview.png'), bbox_inches='tight')
plt.close()
print("Saved: all_models_overview.png")

# ============================================================
# Print summary
# ============================================================
print("\n" + "=" * 100)
print("V3 ANALYSIS SUMMARY")
print("=" * 100)

for dt in ['AA', 'DNA']:
    for topo in ['rooted', 'unrooted']:
        subset = wide_v3[(wide_v3['data_type']==dt) & (wide_v3['topology']==topo)]
        if len(subset) == 0:
            continue
        print(f"\n--- {dt} / {topo} ---")
        for model in sorted(subset['model'].unique()):
            ms = subset[subset['model']==model]
            diffs = ms['ll_diff'].dropna()
            if len(diffs) == 0:
                continue
            n_exact = (diffs < 1e-4).sum()
            status = '✅' if diffs.max() < 0.01 else '⚠️' if diffs.max() < 1.0 else '❌'
            print(f"  {status} {model:12s}: max={diffs.max():.6f}  mean={diffs.mean():.6f}  exact={n_exact}/{len(diffs)}")

# V2 vs V3 comparison for R4
print("\n" + "=" * 100)
print("V2 vs V3 COMPARISON (+R4 only)")
print("=" * 100)
for dt, model in [('DNA','GTR+R4'), ('AA','LG+R4')]:
    for topo in ['rooted', 'unrooted']:
        r4_v2 = wide_v2[(wide_v2['data_type']==dt) & (wide_v2['topology']==topo) &
                          (wide_v2['model']==model)]
        r4_v3 = wide_v3[(wide_v3['data_type']==dt) & (wide_v3['topology']==topo) &
                          (wide_v3['model']==model)]
        v2_max = r4_v2['ll_diff'].max() if len(r4_v2) > 0 else float('nan')
        v3_max = r4_v3['ll_diff'].max() if len(r4_v3) > 0 else float('nan')
        change = ''
        if pd.notna(v2_max) and pd.notna(v3_max):
            if v3_max < v2_max * 0.1:
                change = '🎉 FIXED'
            elif v3_max < v2_max:
                change = '📉 improved'
            elif v3_max > v2_max:
                change = '📈 regression'
            else:
                change = '→ same'
        print(f"  {dt:3s}/{topo:9s}/{model:8s}: v2 max={v2_max:.6f}  v3 max={v3_max:.6f}  {change}")

print("\nDone!")
