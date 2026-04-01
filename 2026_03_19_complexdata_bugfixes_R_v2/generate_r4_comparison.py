#!/usr/bin/env python3
"""Generate VANILA vs OPENACC comparison diagrams for +R4 models,
styled like the +I+G4 vanila_rev_nonrev_openacc_comparison.png reference."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# Read the already-parsed data
df = pd.read_csv('/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/'
                 '2026_03_19_complexdata_bugfixes_R_v2/ll_comparison.csv')

plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 150, 'font.size': 10})

COLORS = {
    'VANILA': '#4C72B0',
    'OPENACC': '#DD8452',
}


def make_comparison_figure(data_type, topology, model, save_path):
    """Create a multi-panel comparison figure for a single model/topology combination."""
    subset = df[(df['data_type'] == data_type) & (df['topology'] == topology) &
                (df['model'] == model)].copy()
    subset = subset.sort_values('tree', key=lambda x: x.str.extract(r'(\d+)')[0].astype(int))

    if len(subset) == 0:
        print(f"  No data for {data_type}/{topology}/{model}")
        return

    trees = subset['tree'].values
    tree_labels = [t.replace('tree_', '') for t in trees]
    n = len(trees)
    x = np.arange(n)

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 1.2],
                           hspace=0.35, wspace=0.3)

    # --- Panel 1: Final Log-Likelihood per Tree (top, full width) ---
    ax1 = fig.add_subplot(gs[0, :])
    width = 0.35
    bars_van = ax1.bar(x - width/2, subset['ll_VANILA'].values, width,
                       label='VANILA', color=COLORS['VANILA'], alpha=0.85)
    bars_acc = ax1.bar(x + width/2, subset['ll_OPENACC'].values, width,
                       label='OPENACC', color=COLORS['OPENACC'], alpha=0.85)
    ax1.set_xlabel('Tree')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Final Log-Likelihood per Tree')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'tree_{l}' for l in tree_labels])
    ax1.legend(loc='upper left')

    # --- Panel 2: LL Diff per Tree (middle-left) ---
    ax2 = fig.add_subplot(gs[1, 0])
    diffs = subset['ll_diff'].values
    bar_colors = ['#2ecc71' if d < 1e-4 else '#f39c12' if d < 1.0 else '#e74c3c' for d in diffs]
    ax2.bar(x, diffs, color=bar_colors, alpha=0.85, edgecolor='gray', linewidth=0.5)
    ax2.set_yscale('symlog', linthresh=0.001)
    ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=0.0001, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Tree')
    ax2.set_ylabel('|LL Diff|')
    ax2.set_title('|LL_OPENACC − LL_VANILA| per Tree')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'T{l}' for l in tree_labels])
    # Add value labels
    for i, d in enumerate(diffs):
        if d >= 0.001:
            ax2.text(i, d * 1.3, f'{d:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
        elif d < 1e-4:
            ax2.text(i, 0.0005, '0', ha='center', va='bottom', fontsize=7)

    legend_elements = [
        Line2D([0], [0], color='#2ecc71', lw=8, label='Exact (<0.0001)'),
        Line2D([0], [0], color='#f39c12', lw=8, label='Small (0.0001–1.0)'),
        Line2D([0], [0], color='#e74c3c', lw=8, label='Large (≥1.0)'),
    ]
    ax2.legend(handles=legend_elements, fontsize=7, loc='upper right')

    # --- Panel 3: Who Found Better LL (middle-right) ---
    ax3 = fig.add_subplot(gs[1, 1])
    ll_diff_signed = (subset['ll_OPENACC'] - subset['ll_VANILA']).values  # positive = ACC better
    bar_colors_signed = ['#DD8452' if d > 0.01 else '#4C72B0' if d < -0.01 else '#95a5a6' for d in ll_diff_signed]
    ax3.bar(x, ll_diff_signed, color=bar_colors_signed, alpha=0.85, edgecolor='gray', linewidth=0.5)
    ax3.axhline(y=0, color='black', linewidth=0.8)
    ax3.set_xlabel('Tree')
    ax3.set_ylabel('LL_OPENACC − LL_VANILA')
    ax3.set_title('Signed LL Difference (positive = OPENACC better)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'T{l}' for l in tree_labels])
    for i, d in enumerate(ll_diff_signed):
        if abs(d) > 0.01:
            ax3.text(i, d + (0.05 if d > 0 else -0.05), f'{d:.3f}',
                     ha='center', va='bottom' if d > 0 else 'top', fontsize=6, rotation=45)

    legend_signed = [
        Line2D([0], [0], color='#DD8452', lw=8, label='OPENACC better'),
        Line2D([0], [0], color='#4C72B0', lw=8, label='VANILA better'),
        Line2D([0], [0], color='#95a5a6', lw=8, label='~Equal (<0.01)'),
    ]
    ax3.legend(handles=legend_signed, fontsize=7, loc='lower right')

    # --- Panel 4: Scatter + Summary Table (bottom) ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(subset['ll_VANILA'], subset['ll_OPENACC'],
                c=[COLORS['OPENACC']], s=60, alpha=0.8, edgecolors='gray', linewidth=0.5, zorder=3)
    # Perfect match line
    all_ll = np.concatenate([subset['ll_VANILA'].values, subset['ll_OPENACC'].values])
    ll_min, ll_max = all_ll.min() - 2, all_ll.max() + 2
    ax4.plot([ll_min, ll_max], [ll_min, ll_max], 'r--', alpha=0.5, linewidth=1, label='Perfect match')
    ax4.set_xlim(ll_min, ll_max)
    ax4.set_ylim(ll_min, ll_max)
    ax4.set_xlabel('VANILA Log-Likelihood')
    ax4.set_ylabel('OPENACC Log-Likelihood')
    ax4.set_title('Same Kernel: VANILA vs OPENACC\n(should be on diagonal = identical)')
    ax4.legend(fontsize=8)
    ax4.set_aspect('equal')
    # Label outliers
    for _, r in subset.iterrows():
        if r['ll_diff'] >= 1.0:
            ax4.annotate(f"T{r['tree'].replace('tree_', '')} (Δ={r['ll_diff']:.2f})",
                         (r['ll_VANILA'], r['ll_OPENACC']),
                         textcoords="offset points", xytext=(8, -8), fontsize=7,
                         color='red', fontweight='bold')

    # --- Panel 5: Summary Table (bottom-right) ---
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    table_data = []
    headers = ['Tree', 'VANILA', 'OPENACC', '|Diff|', 'V_rev→\nV_nrev*', 'V_nrev→\nOPENACC*']
    for _, r in subset.iterrows():
        tree_label = r['tree'].replace('tree_', 'tree_')
        table_data.append([
            tree_label,
            f"{r['ll_VANILA']:.4f}",
            f"{r['ll_OPENACC']:.4f}",
            f"{r['ll_diff']:.4f}",
            'N/A',
            f"{r['ll_diff']:.4f}",
        ])

    # Add MAX row
    table_data.append([
        'MAX |Δ|', '', '',
        f"{subset['ll_diff'].max():.4f}",
        '',
        f"{subset['ll_diff'].max():.4f}",
    ])

    table = ax5.table(cellText=table_data, colLabels=headers,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.2)

    # Color cells by diff magnitude
    for i, row in enumerate(table_data):
        diff_val = subset.iloc[i]['ll_diff'] if i < len(subset) else subset['ll_diff'].max()
        if diff_val < 1e-4:
            color = '#d5f5e3'  # green
        elif diff_val < 1.0:
            color = '#fdebd0'  # orange
        else:
            color = '#fadbd8'  # red
        for j in [3, 5]:
            table[i + 1, j].set_facecolor(color)

    ax5.set_title('Summary Table', fontsize=10, pad=10)
    ax5.text(0.5, -0.02, '*Only 2 backends available for +R4 (no rev/nonrev split)',
             transform=ax5.transAxes, fontsize=7, ha='center', style='italic', color='gray')

    fig.suptitle(f'{data_type} / {topology} / {model} — VANILA vs OPENACC',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# Generate for all +R4 combinations
out_dir = '/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_03_19_complexdata_bugfixes_R_v2'

for data_type, model in [('DNA', 'GTR+R4'), ('AA', 'LG+R4')]:
    for topology in ['rooted', 'unrooted']:
        fname = f'vanila_openacc_comparison_{data_type}_{model.replace("+", "_")}_{topology}.png'
        make_comparison_figure(data_type, topology, model,
                               f'{out_dir}/{fname}')

print("\nDone! All comparison diagrams generated.")
