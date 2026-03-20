#!/usr/bin/env python3
"""Generate VANILA vs OPENACC comparison diagrams for +R4 models (v4 - tree_lh fix).
2-way comparison (no VANILA_NONREV in v4). Also generates verbose-specific R4 plots.
Styled like the v3 3-way comparison reference."""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

results_path = "/Users/u7826985/Projects/Nvidia/results/2026_03_19_complexdata_bugfixes_R_v4"
out_dir = "/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_03_19_complexdata_bugfixes_R_v4"

plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 150, 'font.size': 10})

COLORS = {
    'VANILA': '#4C72B0',
    'OPENACC': '#DD8452',
}

iqtree_ll_pattern = re.compile(r'Log-likelihood of the tree:\s+([-\d.]+)')
iqtree_rates_pattern = re.compile(r'Site proportion and rates:\s+(.*)')


def parse_r4_results(base_path, model_filter, verbose_mode=False):
    """Parse +R4 results. verbose_mode=True parses only verbose files, False parses only non-verbose."""
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
                if model not in model_filter:
                    continue
                model_dir = os.path.join(topo_dir, model)
                if not os.path.isdir(model_dir):
                    continue
                for tree in sorted(os.listdir(model_dir)):
                    tree_dir = os.path.join(model_dir, tree)
                    if not os.path.isdir(tree_dir):
                        continue
                    iqtree_files = [f for f in os.listdir(tree_dir) if f.endswith('.iqtree')]

                    row = {
                        'data_type': data_type, 'topology': topology,
                        'model': model, 'tree': tree,
                    }

                    for f in iqtree_files:
                        is_verbose = 'verbose' in f
                        if verbose_mode != is_verbose:
                            continue

                        # Classify backend
                        if 'OPENACC' in f:
                            backend = 'OPENACC'
                        elif 'VANILA' in f:
                            backend = 'VANILA'
                        else:
                            continue

                        content = open(os.path.join(tree_dir, f)).read()
                        m = iqtree_ll_pattern.search(content)
                        if m:
                            row[f'll_{backend}'] = float(m.group(1))
                        m = iqtree_rates_pattern.search(content)
                        if m:
                            row[f'rates_{backend}'] = m.group(1).strip()

                    rows.append(row)

    df = pd.DataFrame(rows)
    for la, lb in [('ll_OPENACC', 'll_VANILA')]:
        if la in df.columns and lb in df.columns:
            df['diff_acc_vanila'] = (df[la] - df[lb]).abs()
    return df


def make_2way_comparison(df, data_type, topology, model, save_path):
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

    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 1.3],
                           hspace=0.35, wspace=0.3)

    # --- Panel 1: Final Log-Likelihood per Tree (top, full width) ---
    ax1 = fig.add_subplot(gs[0, :])
    width = 0.35
    if 'll_VANILA' in subset.columns:
        ax1.bar(x - width/2, subset['ll_VANILA'].values, width,
                label='VANILA', color=COLORS['VANILA'], alpha=0.85)
    if 'll_OPENACC' in subset.columns:
        ax1.bar(x + width/2, subset['ll_OPENACC'].values, width,
                label='OPENACC', color=COLORS['OPENACC'], alpha=0.85)
    ax1.set_xlabel('Tree')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Final Log-Likelihood per Tree')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'tree_{l}' for l in tree_labels])
    ax1.legend(loc='upper left')

    # --- Panel 2: Diff magnitude (middle-left) ---
    ax2 = fig.add_subplot(gs[1, 0])

    if 'diff_acc_vanila' in subset.columns and subset['diff_acc_vanila'].notna().any():
        vals = subset['diff_acc_vanila'].values
        plot_vals = np.where(vals < 1e-10, 1e-10, vals)
        colors_bar = ['#d5f5e3' if d < 1e-4 else '#fdebd0' if d < 1.0 else '#fadbd8'
                       for d in vals]
        ax2.bar(x, plot_vals, color=colors_bar, edgecolor='gray', linewidth=0.5)

    ax2.set_yscale('symlog', linthresh=1e-6)
    ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.4, linewidth=1, label='1.0 threshold')
    ax2.axhline(y=0.0001, color='green', linestyle='--', alpha=0.4, linewidth=1, label='0.0001 threshold')
    ax2.set_xlabel('Tree')
    ax2.set_ylabel('|LL Difference|')
    ax2.set_title('|VANILA - OPENACC| per Tree')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'T{l}' for l in tree_labels])
    ax2.legend(fontsize=7, loc='upper right')

    # --- Panel 3: EM-Estimated Rate Categories (middle-right) ---
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

    # --- Panel 4: Scatter VANILA vs OPENACC (bottom-left) ---
    ax4 = fig.add_subplot(gs[2, 0])

    if 'll_VANILA' in subset.columns and 'll_OPENACC' in subset.columns:
        ax4.scatter(subset['ll_VANILA'], subset['ll_OPENACC'],
                    c=COLORS['OPENACC'], s=60, alpha=0.8, edgecolors='gray', linewidth=0.5, zorder=3)
        all_ll = np.concatenate([subset['ll_VANILA'].dropna().values,
                                 subset['ll_OPENACC'].dropna().values])
        ll_min, ll_max = all_ll.min() - 2, all_ll.max() + 2
        ax4.plot([ll_min, ll_max], [ll_min, ll_max], 'r--', alpha=0.5, linewidth=1, label='Perfect match')
        ax4.set_xlim(ll_min, ll_max)
        ax4.set_ylim(ll_min, ll_max)
        ax4.set_xlabel('VANILA LL')
        ax4.set_ylabel('OPENACC LL')
        ax4.set_title('VANILA vs OPENACC\n(should be on diagonal = identical)')
        ax4.legend(fontsize=8)
        ax4.set_aspect('equal')
        if 'diff_acc_vanila' in subset.columns:
            for _, r in subset.iterrows():
                d = r.get('diff_acc_vanila', 0)
                if pd.notna(d) and d >= 0.5:
                    ax4.annotate(f"T{r['tree'].replace('tree_', '')} (d={d:.2f})",
                                 (r['ll_VANILA'], r['ll_OPENACC']),
                                 textcoords="offset points", xytext=(8, -8), fontsize=7,
                                 color='red', fontweight='bold')

    # --- Panel 5: Summary Table (bottom-right) ---
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    headers = ['Tree', 'VANILA', 'OPENACC', '|Diff|']

    table_data = []
    for _, r in subset.iterrows():
        tree_label = r['tree']
        ll_vanila = r.get('ll_VANILA', np.nan)
        ll_acc = r.get('ll_OPENACC', np.nan)
        d = r.get('diff_acc_vanila', 0)

        table_data.append([
            tree_label,
            f"{ll_vanila:.4f}" if pd.notna(ll_vanila) else 'N/A',
            f"{ll_acc:.4f}" if pd.notna(ll_acc) else 'N/A',
            f"{d:.6f}" if pd.notna(d) else 'N/A',
        ])

    # MAX row
    max_d = subset['diff_acc_vanila'].max() if 'diff_acc_vanila' in subset.columns else 0
    if pd.isna(max_d):
        max_d = 0
    table_data.append(['MAX |d|', '', '', f"{max_d:.6f}"])

    table = ax5.table(cellText=table_data, colLabels=headers,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.2)

    for i in range(len(table_data)):
        if i < len(subset):
            val = subset.iloc[i].get('diff_acc_vanila', 0)
        else:
            val = max_d
        if pd.isna(val):
            val = 0
        if val < 1e-4:
            color = '#d5f5e3'
        elif val < 1.0:
            color = '#fdebd0'
        else:
            color = '#fadbd8'
        table[i + 1, 3].set_facecolor(color)

    ax5.set_title('Summary Table', fontsize=10, pad=10)

    fig.suptitle(f'{data_type} / {topology} / {model} -- VANILA vs OPENACC (v4)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# Parse non-verbose R4 results
# ============================================================
print("Parsing non-verbose R4 results...")
df = parse_r4_results(results_path, ['GTR+R4', 'LG+R4'], verbose_mode=False)
print(f"Parsed {len(df)} rows (non-verbose)")
for col in ['ll_VANILA', 'll_OPENACC']:
    n = df[col].notna().sum() if col in df.columns else 0
    print(f"  {col}: {n} values")

# Generate non-verbose R4 plots
for data_type, model in [('DNA', 'GTR+R4'), ('AA', 'LG+R4')]:
    for topology in ['rooted', 'unrooted']:
        fname = f'vanila_openacc_comparison_{data_type}_{model.replace("+", "_")}_{topology}.png'
        make_2way_comparison(df, data_type, topology, model, f'{out_dir}/{fname}')

# ============================================================
# Parse verbose R4 results
# ============================================================
print("\nParsing verbose R4 results...")
df_verbose = parse_r4_results(results_path, ['GTR+R4', 'LG+R4'], verbose_mode=True)
print(f"Parsed {len(df_verbose)} rows (verbose)")
for col in ['ll_VANILA', 'll_OPENACC']:
    n = df_verbose[col].notna().sum() if col in df_verbose.columns else 0
    print(f"  {col}: {n} values")

# Generate verbose R4 plots
for data_type, model in [('DNA', 'GTR+R4'), ('AA', 'LG+R4')]:
    for topology in ['rooted', 'unrooted']:
        fname = f'vanila_openacc_comparison_{data_type}_{model.replace("+", "_")}_{topology}_verbose.png'
        make_2way_comparison(df_verbose, data_type, topology, model, f'{out_dir}/{fname}')

# ============================================================
# Print summary analysis
# ============================================================
print("\n" + "=" * 100)
print("2-WAY COMPARISON SUMMARY (v4 - tree_lh fix): VANILA vs OPENACC for +R4")
print("=" * 100)

for label, frame in [('NON-VERBOSE', df), ('VERBOSE', df_verbose)]:
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for data_type, model in [('DNA', 'GTR+R4'), ('AA', 'LG+R4')]:
        for topology in ['rooted', 'unrooted']:
            subset = frame[(frame['data_type'] == data_type) & (frame['topology'] == topology) &
                           (frame['model'] == model)]
            if len(subset) == 0:
                continue

            print(f"\n--- {data_type} / {topology} / {model} ---")
            if 'diff_acc_vanila' in subset.columns:
                vals = subset['diff_acc_vanila'].dropna()
                if len(vals) > 0:
                    n_exact = (vals < 1e-4).sum()
                    print(f"  OPENACC vs VANILA              : max={vals.max():.6f}  mean={vals.mean():.6f}  "
                          f"exact={n_exact}/{len(vals)}  median={vals.median():.6f}")

print("\n--- INTERPRETATION ---")
print("v4 fixes tree_lh synchronization for +R4 models.")
print("If diffs dropped to ~0: the tree_lh fix resolved the remaining +R4 divergence.")
print("Verbose runs use -v flag for additional IQ-TREE output.")

print("\nDone!")
