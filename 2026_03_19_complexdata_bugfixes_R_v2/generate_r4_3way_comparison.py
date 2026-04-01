#!/usr/bin/env python3
"""Generate VANILA(rev) vs VANILA(nonrev) vs OPENACC comparison diagrams for +R4 models,
styled like the +I+G4 vanila_rev_nonrev_openacc_comparison.png reference."""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

results_path = "/Users/u7826985/Projects/Nvidia/results/2026_03_19_complexdata_bugfixes_R_v2"
out_dir = "/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_03_19_complexdata_bugfixes_R_v2"

plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 150, 'font.size': 10})

COLORS = {
    'VANILA (rev)': '#4C72B0',
    'VANILA (nonrev)': '#55A868',
    'OPENACC': '#DD8452',
}

# --- Parse all 3 backends from .iqtree files ---
iqtree_ll_pattern = re.compile(r'Log-likelihood of the tree:\s+([-\d.]+)')
iqtree_rates_pattern = re.compile(r'Site proportion and rates:\s+(.*)')


def classify_backend(filename):
    """Classify backend from filename, handling both single and double underscore."""
    if 'VANILA_NONREV' in filename or 'VANILA__NONREV' in filename:
        return 'VANILA_NONREV'
    elif 'OPENACC' in filename:
        return 'OPENACC'
    elif 'VANILA' in filename:
        return 'VANILA_REV'
    return None


def parse_r4_results(base_path, model_filter):
    """Parse R4 results with all 3 backends."""
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
                        backend = classify_backend(f)
                        if backend is None:
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
    # Compute pairwise diffs
    for a, b, col in [('OPENACC', 'VANILA_REV', 'diff_acc_rev'),
                       ('OPENACC', 'VANILA_NONREV', 'diff_acc_nonrev'),
                       ('VANILA_NONREV', 'VANILA_REV', 'diff_nonrev_rev')]:
        la, lb = f'll_{a}', f'll_{b}'
        if la in df.columns and lb in df.columns:
            df[col] = (df[la] - df[lb]).abs()
    return df


df = parse_r4_results(results_path, ['GTR+R4', 'LG+R4'])
print(f"Parsed {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# Check which backends are present
for col in ['ll_VANILA_REV', 'll_VANILA_NONREV', 'll_OPENACC']:
    n = df[col].notna().sum() if col in df.columns else 0
    print(f"  {col}: {n} values")


def make_3way_comparison(data_type, topology, model, save_path):
    """Create a multi-panel 3-way comparison figure."""
    subset = df[(df['data_type'] == data_type) & (df['topology'] == topology) &
                (df['model'] == model)].copy()
    subset = subset.sort_values('tree', key=lambda x: x.str.extract(r'(\d+)')[0].astype(int))

    if len(subset) == 0:
        print(f"  No data for {data_type}/{topology}/{model}")
        return

    has_nonrev = 'll_VANILA_NONREV' in subset.columns and subset['ll_VANILA_NONREV'].notna().any()

    trees = subset['tree'].values
    tree_labels = [t.replace('tree_', '') for t in trees]
    n = len(trees)
    x = np.arange(n)

    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 1.3],
                           hspace=0.35, wspace=0.3)

    # --- Panel 1: Final Log-Likelihood per Tree (top, full width) ---
    ax1 = fig.add_subplot(gs[0, :])
    if has_nonrev:
        width = 0.25
        ax1.bar(x - width, subset['ll_VANILA_REV'].values, width,
                label='VANILA (rev)', color=COLORS['VANILA (rev)'], alpha=0.85)
        ax1.bar(x, subset['ll_VANILA_NONREV'].values, width,
                label='VANILA (nonrev)', color=COLORS['VANILA (nonrev)'], alpha=0.85)
        ax1.bar(x + width, subset['ll_OPENACC'].values, width,
                label='OPENACC', color=COLORS['OPENACC'], alpha=0.85)
    else:
        width = 0.35
        ax1.bar(x - width/2, subset['ll_VANILA_REV'].values, width,
                label='VANILA (rev)', color=COLORS['VANILA (rev)'], alpha=0.85)
        ax1.bar(x + width/2, subset['ll_OPENACC'].values, width,
                label='OPENACC', color=COLORS['OPENACC'], alpha=0.85)
    ax1.set_xlabel('Tree')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Final Log-Likelihood per Tree')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'tree_{l}' for l in tree_labels])
    ax1.legend(loc='upper left')

    # --- Panel 2: Source of Remaining Diffs (middle-left) ---
    ax2 = fig.add_subplot(gs[1, 0])

    # Show all 3 pairwise diffs
    diff_pairs = [
        ('diff_acc_rev', 'ACC vs REV', '#e74c3c', 'o'),
        ('diff_nonrev_rev', 'NONREV vs REV\n(kernel_nonrev effect)', '#55A868', 's'),
        ('diff_acc_nonrev', 'ACC vs NONREV\n(GPU vs CPU effect)', '#DD8452', '^'),
    ]

    for col, label, color, marker in diff_pairs:
        if col in subset.columns and subset[col].notna().any():
            vals = subset[col].values
            # Replace zeros with small value for log scale
            plot_vals = np.where(vals < 1e-10, 1e-10, vals)
            ax2.scatter(x, plot_vals, c=color, s=50, marker=marker, alpha=0.85,
                       edgecolors='gray', linewidth=0.5, label=label, zorder=3)

    ax2.set_yscale('symlog', linthresh=1e-6)
    ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.4, linewidth=1)
    ax2.axhline(y=0.0001, color='green', linestyle='--', alpha=0.4, linewidth=1)
    ax2.set_xlabel('Tree')
    ax2.set_ylabel('|LL Difference|')
    ax2.set_title('Source of Remaining Diffs')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'T{l}' for l in tree_labels])
    ax2.legend(fontsize=7, loc='upper right')

    # --- Panel 3: EM-Estimated Rate Categories (middle-right) ---
    ax3 = fig.add_subplot(gs[1, 1])

    # Parse rate categories for visual comparison
    def parse_rates(rate_str):
        """Parse '(prop,rate) (prop,rate) ...' into list of (prop, rate) tuples."""
        if pd.isna(rate_str):
            return []
        import re as _re
        matches = _re.findall(r'\(([\d.]+),([\d.]+)\)', rate_str)
        return [(float(p), float(r)) for p, r in matches]

    # Show max rate per tree for each backend (indicates convergence point)
    for backend, label, color in [('VANILA_REV', 'VANILA (rev)', COLORS['VANILA (rev)']),
                                   ('VANILA_NONREV', 'VANILA (nonrev)', COLORS['VANILA (nonrev)']),
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
            offset = {'VANILA_REV': -0.2, 'VANILA_NONREV': 0, 'OPENACC': 0.2}.get(backend, 0)
            ax3.bar(x + offset, max_rates, 0.2, label=label, color=color, alpha=0.85)

    ax3.set_xlabel('Tree')
    ax3.set_ylabel('Max Rate Category')
    ax3.set_title('EM-Estimated Max Rate by Kernel Type')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'T{l}' for l in tree_labels])
    ax3.legend(fontsize=7)

    # --- Panel 4: Scatter VANILA(nonrev) vs OPENACC (bottom-left) ---
    ax4 = fig.add_subplot(gs[2, 0])
    if has_nonrev:
        scatter_x_col = 'll_VANILA_NONREV'
        scatter_x_label = 'VANILA (nonrev) LL'
    else:
        scatter_x_col = 'll_VANILA_REV'
        scatter_x_label = 'VANILA (rev) LL'

    ax4.scatter(subset[scatter_x_col], subset['ll_OPENACC'],
                c=COLORS['OPENACC'], s=60, alpha=0.8, edgecolors='gray', linewidth=0.5, zorder=3)
    all_ll = np.concatenate([subset[scatter_x_col].dropna().values,
                             subset['ll_OPENACC'].dropna().values])
    ll_min, ll_max = all_ll.min() - 2, all_ll.max() + 2
    ax4.plot([ll_min, ll_max], [ll_min, ll_max], 'r--', alpha=0.5, linewidth=1, label='Perfect match')
    ax4.set_xlim(ll_min, ll_max)
    ax4.set_ylim(ll_min, ll_max)
    ax4.set_xlabel(scatter_x_label)
    ax4.set_ylabel('OPENACC LL')
    ax4.set_title(f'Same Kernel: {scatter_x_label.split(" LL")[0]} vs OPENACC\n(should be on diagonal = identical)')
    ax4.legend(fontsize=8)
    ax4.set_aspect('equal')
    # Label outliers
    diff_col = 'diff_acc_nonrev' if has_nonrev else 'diff_acc_rev'
    for _, r in subset.iterrows():
        d = r.get(diff_col, 0)
        if pd.notna(d) and d >= 0.5:
            ax4.annotate(f"T{r['tree'].replace('tree_', '')} (Δ={d:.2f})",
                         (r[scatter_x_col], r['ll_OPENACC']),
                         textcoords="offset points", xytext=(8, -8), fontsize=7,
                         color='red', fontweight='bold')

    # --- Panel 5: Summary Table (bottom-right) ---
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    if has_nonrev:
        headers = ['Tree', 'VANILA\n(rev)', 'VANILA\n(nonrev)', 'OPENACC',
                   'V_rev→\nV_nrev', 'V_nrev→\nACC']
    else:
        headers = ['Tree', 'VANILA\n(rev)', 'OPENACC', '|Diff|', '', '']

    table_data = []
    for _, r in subset.iterrows():
        tree_label = r['tree']
        ll_rev = r.get('ll_VANILA_REV', np.nan)
        ll_nonrev = r.get('ll_VANILA_NONREV', np.nan)
        ll_acc = r.get('ll_OPENACC', np.nan)
        d_nr = r.get('diff_nonrev_rev', 0)
        d_an = r.get('diff_acc_nonrev', r.get('diff_acc_rev', 0))

        if has_nonrev:
            table_data.append([
                tree_label,
                f"{ll_rev:.4f}" if pd.notna(ll_rev) else 'N/A',
                f"{ll_nonrev:.4f}" if pd.notna(ll_nonrev) else 'N/A',
                f"{ll_acc:.4f}" if pd.notna(ll_acc) else 'N/A',
                f"{d_nr:.4f}" if pd.notna(d_nr) else 'N/A',
                f"{d_an:.4f}" if pd.notna(d_an) else 'N/A',
            ])
        else:
            d_ar = r.get('diff_acc_rev', 0)
            table_data.append([
                tree_label,
                f"{ll_rev:.4f}" if pd.notna(ll_rev) else 'N/A',
                f"{ll_acc:.4f}" if pd.notna(ll_acc) else 'N/A',
                f"{d_ar:.4f}" if pd.notna(d_ar) else 'N/A',
                '', '',
            ])

    # MAX row
    if has_nonrev:
        max_nr = subset['diff_nonrev_rev'].max() if 'diff_nonrev_rev' in subset.columns else 0
        max_an = subset['diff_acc_nonrev'].max() if 'diff_acc_nonrev' in subset.columns else 0
        table_data.append(['MAX |Δ|', '', '', '', f"{max_nr:.4f}", f"{max_an:.4f}"])
    else:
        max_ar = subset['diff_acc_rev'].max() if 'diff_acc_rev' in subset.columns else 0
        table_data.append(['MAX |Δ|', '', '', f"{max_ar:.4f}", '', ''])

    table = ax5.table(cellText=table_data, colLabels=headers,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.2)

    # Color cells by diff magnitude
    for i in range(len(table_data)):
        if has_nonrev:
            # Color columns 4 and 5 (diff_nonrev_rev and diff_acc_nonrev)
            for j, col_name in [(4, 'diff_nonrev_rev'), (5, 'diff_acc_nonrev')]:
                if i < len(subset):
                    val = subset.iloc[i].get(col_name, 0)
                else:
                    val = subset[col_name].max() if col_name in subset.columns else 0
                if pd.isna(val):
                    val = 0
                if val < 1e-4:
                    color = '#d5f5e3'
                elif val < 1.0:
                    color = '#fdebd0'
                else:
                    color = '#fadbd8'
                table[i + 1, j].set_facecolor(color)
        else:
            j = 3
            if i < len(subset):
                val = subset.iloc[i].get('diff_acc_rev', 0)
            else:
                val = subset['diff_acc_rev'].max() if 'diff_acc_rev' in subset.columns else 0
            if pd.isna(val):
                val = 0
            if val < 1e-4:
                color = '#d5f5e3'
            elif val < 1.0:
                color = '#fdebd0'
            else:
                color = '#fadbd8'
            table[i + 1, j].set_facecolor(color)

    ax5.set_title('Summary Table', fontsize=10, pad=10)

    backends_str = "VANILA(rev) vs VANILA(nonrev) vs OPENACC" if has_nonrev else "VANILA(rev) vs OPENACC"
    fig.suptitle(f'{data_type} / {topology} / {model} — {backends_str}',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# Generate for all +R4 combinations
for data_type, model in [('DNA', 'GTR+R4'), ('AA', 'LG+R4')]:
    for topology in ['rooted', 'unrooted']:
        fname = f'vanila_rev_nonrev_openacc_comparison_{data_type}_{model.replace("+", "_")}_{topology}.png'
        make_3way_comparison(data_type, topology, model, f'{out_dir}/{fname}')

# --- Print summary analysis ---
print("\n" + "=" * 100)
print("3-WAY COMPARISON SUMMARY: VANILA(rev) vs VANILA(nonrev) vs OPENACC for +R4")
print("=" * 100)

for data_type, model in [('DNA', 'GTR+R4'), ('AA', 'LG+R4')]:
    for topology in ['rooted', 'unrooted']:
        subset = df[(df['data_type'] == data_type) & (df['topology'] == topology) &
                    (df['model'] == model)]
        if len(subset) == 0:
            continue

        print(f"\n--- {data_type} / {topology} / {model} ---")
        for col, label in [('diff_nonrev_rev', 'NONREV vs REV (kernel effect)'),
                           ('diff_acc_nonrev', 'ACC vs NONREV (GPU effect)'),
                           ('diff_acc_rev', 'ACC vs REV (total)')]:
            if col in subset.columns:
                vals = subset[col].dropna()
                n_exact = (vals < 1e-4).sum()
                print(f"  {label:40s}: max={vals.max():.4f}  mean={vals.mean():.4f}  "
                      f"exact={n_exact}/{len(vals)}  median={vals.median():.4f}")

print("\n--- INTERPRETATION ---")
print("If NONREV vs REV diffs are ~0: kernel_nonrev is NOT the source of remaining R4 diffs")
print("If ACC vs NONREV ≈ ACC vs REV: the diff is entirely GPU-side (OpenACC EM optimization path)")
print("If NONREV vs REV diffs > 0: kernel_nonrev introduces FP divergence that EM amplifies")

print("\nDone!")
