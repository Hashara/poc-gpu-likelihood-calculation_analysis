"""
Plot the Phase 2 + Phase 1 transpose optimization journey per test case.

Parses all result logs for baseline, aferprof1-5, and VANILA (CPU) and plots:
  1. Wall-clock time progression per test case (absolute)
  2. Parameter-optimization time progression (the GPU-specific metric)
  3. Speedup vs baseline, per experiment stage
  4. Correctness check (lnL exact match)
  5. GPU vs CPU speedup at final state

Saves all figures to the same folder as this script.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = "/Users/u7826985/Projects/Nvidia/results/2026_04_20_more_opt"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Experiment stage labels in chronological order
STAGES = ['baseline', 'aferprof1', 'aferprof2', 'aferprof3', 'aferprof4', 'aferprof5', 'aferprof6', 'aferprof7']
STAGE_PRETTY = {
    'baseline':   'Baseline\n(pre-opt)',
    'aferprof1':  'aferprof1\nPh2 hoist',
    'aferprof2':  'aferprof2\n+Ph2 transpose',
    'aferprof3':  'aferprof3\n+maxreg:96',
    'aferprof4':  'aferprof4\n+Ph1 hoist',
    'aferprof5':  'aferprof5\n+Ph1 transpose',
    'aferprof6':  'aferprof6\n+TipTip batch',
    'aferprof7':  'aferprof7\n(REVERTED)',
}
STAGE_SHORT = {
    'baseline':   'base',
    'aferprof1':  'aP1',
    'aferprof2':  'aP2',
    'aferprof3':  'aP3',
    'aferprof4':  'aP4',
    'aferprof5':  'aP5',
    'aferprof6':  'aP6',
    'aferprof7':  'aP7 [X]',
}

# Long description of each stage (for the legend)
STAGE_DESCRIPTIONS = {
    'baseline':   'Original REV partial-LH kernels before any profiling-driven tuning.',
    'aferprof1':  'Phase 2 inner-loop hoist: precompute inv_evec row pointer outside x-loop, replace s%NSTATES with s-cat*NSTATES. Cleaner code + 6 regs/thread saved.',
    'aferprof2':  'Phase 2 inv_evec transpose: persistent column-major mirror so warp threads (different k, same x) read adjacent memory. Coalesced access, 2.27x Phase-2 kernel speedup.',
    'aferprof3':  'maxregcount:96 compiler flag: lifted REV occupancy 22.6 to 28.0%. REVERTED - NonRev kernels in same .cpp spilled and regressed 3.8%. Will re-enable once NonRev is optimized.',
    'aferprof4':  'Phase 1 forward-transform hoist: same pattern as Ph2 hoist, for the eigenspace-multiply loop. Foundation for Ph1 transpose.',
    'aferprof5':  'Phase 1 echildren transpose: persistent GPU mirror + Phase 0 transpose helper kernel before Phase 1. Coalesced reads, 2.7x kernel, -15.7% end-to-end.',
    'aferprof6':  'TipTip_Rev fully batched (committed ec5b576b): collapse(3) over (op,p,s). TipTip 13.7x faster per-kernel. Safe because block=80-100 keeps per-op slice under L2 capacity.',
    'aferprof7':  'REVERTED. Tried the aP6 recipe on TipInt/IntInt. Kernel-level: TipInt 2.64x, IntInt 2.15x. BUT deriv kernel 2x slower because block=1000 on LG+I+G4 makes per-op slice 32 MB; with 35 ops per level the live eigen_prod footprint hits 1.1 GB (180x V100 L2 cache) and pollutes cache for downstream deriv kernels. Net end-to-end: neutral.',
}

# ==========================================================================
# Log parsing
# ==========================================================================

def parse_log(path):
    """Extract lnL, total wall time, and parameter-optimization time."""
    lnl = wall_time = opt_time = None
    with open(path) as f:
        for line in f:
            m = re.search(r'BEST SCORE FOUND\s*:\s*([\-\d\.]+)', line)
            if m:
                lnl = float(m.group(1))
            m = re.search(r'Total wall-clock time used:\s*([\d\.]+)\s*sec', line)
            if m:
                wall_time = float(m.group(1))
            m = re.search(r'Parameters optimization took\s+\d+\s+rounds\s+\(([\d\.]+)\s*sec\)', line)
            if m:
                opt_time = float(m.group(1))
    return lnl, wall_time, opt_time


def parse_filename(fname):
    """
    Determine stage, kernel, data_type, exec_type, test_name from filename.

    Patterns:
      baseline/opt2:       output_reduction_kernel_{kernel}_tests_{version}_{data}_...
      aferprofN:           output_reduction_kernel_{kernel}_tests_aferprofN_{data}_...
      VANILA has version == 'baseline' always.
    """
    # Stage: baseline / opt2 / aferprofN
    stage = None
    for s in STAGES + ['opt2']:
        if f'_{s}_' in fname:
            stage = s
            break
    if stage is None:
        return None

    # Kernel: rev or nonrev
    kernel = 'nonrev' if 'nonrev_tests' in fname else 'rev'

    # Data type
    data_type = 'AA' if '_AA_' in fname else 'DNA'

    # Exec type
    m = re.search(r'_(GTR[^_]+|LG[^_]+)_(OPENACC|VANILA)_taxa', fname)
    exec_type = m.group(2) if m else None

    # Test variant (unique_name)
    m = re.search(r'iqtree3_(.*?)_(OPENACC|VANILA)_run1', fname)
    unique_name = m.group(1) if m else fname
    unique_name = re.sub(r'_(OPENACC|VANILA)_', '_', unique_name)

    return stage, kernel, data_type, exec_type, unique_name


def load_all():
    records = []
    for subdir in ['AA', 'DNA']:
        folder = os.path.join(RESULTS_DIR, subdir)
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.log'):
                continue
            # Skip NCU-instrumented runs (their timing is polluted by ncu overhead)
            if fname.startswith('outputncu_') or 'ncu_OPENACC_PROFILE' in fname:
                continue
            parsed = parse_filename(fname)
            if parsed is None:
                continue
            stage, kernel, data_type, exec_type, unique_name = parsed
            if exec_type is None:
                continue
            lnl, wall_time, opt_time = parse_log(os.path.join(folder, fname))
            records.append({
                'stage':        stage,
                'kernel':       kernel,
                'data_type':    data_type,
                'exec_type':    exec_type,
                'unique_name':  unique_name,
                'lnl':          lnl,
                'wall_time_s':  wall_time,
                'opt_time_s':   opt_time,
            })
    return pd.DataFrame(records)


# ==========================================================================
# Main
# ==========================================================================

def main():
    df = load_all()
    print(f"Parsed {len(df)} log files")

    # Build test-case label
    df['test_case'] = df['data_type'] + ' ' + df['kernel'].str.upper() + ' | ' + df['unique_name']

    # Split GPU vs CPU
    gpu = df[df['exec_type'] == 'OPENACC'].copy()
    cpu = df[df['exec_type'] == 'VANILA'].copy()

    # Unique test cases (based on GPU rows)
    test_cases = sorted(gpu['test_case'].unique())
    print(f"\nGPU test cases ({len(test_cases)}):")
    for tc in test_cases:
        print(f"  - {tc}")

    # ----------------------------------------------------------------------
    # Figure 1: Opt-time progression per test case — one subplot per test
    # ----------------------------------------------------------------------
    n_tests = len(test_cases)
    ncols = 3
    nrows = (n_tests + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten() if n_tests > 1 else [axes]

    for ax, tc in zip(axes, test_cases):
        rows = gpu[gpu['test_case'] == tc]
        rows_by_stage = rows.set_index('stage').reindex(STAGES)
        opt_times = rows_by_stage['opt_time_s'].values

        stages_present = [s for s, t in zip(STAGES, opt_times) if pd.notna(t)]
        vals = [t for t in opt_times if pd.notna(t)]

        if not vals:
            ax.set_visible(False)
            continue

        baseline = vals[0] if stages_present[0] == 'baseline' else None
        colors = []
        for s, v in zip(stages_present, vals):
            if baseline and v < baseline * 0.98:
                colors.append('#2E7D32')   # green = improvement
            elif baseline and v > baseline * 1.02:
                colors.append('#C62828')   # red = regression
            else:
                colors.append('#757575')   # grey = neutral

        x = np.arange(len(stages_present))
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

        # Baseline reference line
        if baseline:
            ax.axhline(baseline, color='black', linestyle='--', linewidth=0.8, alpha=0.6)

        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.2f}s', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([STAGE_SHORT[s] for s in stages_present], fontsize=9)
        ax.set_ylabel('Parameter opt time (s)')
        ax.set_title(tc, fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # Hide unused axes
    for ax in axes[len(test_cases):]:
        ax.set_visible(False)

    fig.suptitle('Parameter-optimization time per test case — experiment progression',
                 fontsize=14, y=1.00)

    # Legend footer describing each stage (inline with the figure)
    legend_lines = []
    for s in STAGES:
        legend_lines.append(f'{STAGE_SHORT[s]:<5} {s:<10}  {STAGE_DESCRIPTIONS[s]}')
    legend_text = '\n'.join(legend_lines)
    fig.text(0.02, -0.01, 'Stage legend:\n' + legend_text,
             fontsize=8, family='monospace', va='top',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5',
                       edgecolor='#BDBDBD', linewidth=0.8))
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'progression_opt_time_per_test.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")

    # ----------------------------------------------------------------------
    # Figure 2: Speedup vs baseline, grouped bars
    # ----------------------------------------------------------------------
    # Pivot to (test_case × stage)
    pivot = gpu.pivot_table(index='test_case', columns='stage', values='opt_time_s')
    pivot = pivot.reindex(columns=[s for s in STAGES if s in pivot.columns])

    if 'baseline' in pivot.columns:
        speedup = pivot['baseline'].values[:, None] / pivot.values
        stages_plotted = list(pivot.columns)

        fig, ax = plt.subplots(figsize=(14, 6))
        n_tc = len(pivot.index)
        n_st = len(stages_plotted)
        bar_w = 0.8 / n_st
        x = np.arange(n_tc)

        # Simple color map per stage
        stage_colors = {
            'baseline':  '#90A4AE',
            'aferprof1': '#BDBDBD',
            'aferprof2': '#42A5F5',
            'aferprof3': '#FF9800',
            'aferprof4': '#BDBDBD',
            'aferprof5': '#2E7D32',
            'aferprof6': '#6A1B9A',
            'aferprof7': '#BDBDBD',  # grey = reverted, not part of cumulative stack
        }
        for j, s in enumerate(stages_plotted):
            offset = (j - (n_st - 1) / 2) * bar_w
            vals = speedup[:, j]
            bars = ax.bar(x + offset, vals, bar_w,
                          label=STAGE_SHORT[s],
                          color=stage_colors.get(s, '#888'),
                          alpha=0.9, edgecolor='black', linewidth=0.3)
            for bar, v in zip(bars, vals):
                if pd.notna(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f'{v:.2f}×', ha='center', va='bottom', fontsize=7)

        ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Speedup vs baseline\n(baseline / stage)')
        ax.set_title('Speedup vs baseline — per test case per experiment stage')
        ax.legend(loc='upper left', fontsize=9, ncol=n_st)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, 'speedup_vs_baseline_per_stage.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    # ----------------------------------------------------------------------
    # Figure 3: Correctness check (lnL diff vs baseline)
    # ----------------------------------------------------------------------
    pivot_lnl = gpu.pivot_table(index='test_case', columns='stage', values='lnl')
    pivot_lnl = pivot_lnl.reindex(columns=[s for s in STAGES if s in pivot_lnl.columns])

    if 'baseline' in pivot_lnl.columns:
        diff = pivot_lnl.subtract(pivot_lnl['baseline'], axis=0).abs()
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(diff.values, aspect='auto', cmap='RdYlGn_r',
                       vmin=0, vmax=max(0.01, diff.values.max() or 0.01))
        ax.set_xticks(range(len(diff.columns)))
        ax.set_xticklabels([STAGE_SHORT[s] for s in diff.columns], fontsize=9)
        ax.set_yticks(range(len(diff.index)))
        ax.set_yticklabels(diff.index, fontsize=9)
        # Annotate cells
        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                v = diff.values[i, j]
                if pd.notna(v):
                    label = '0.000' if v == 0 else f'{v:.3f}'
                    ax.text(j, i, label, ha='center', va='center', fontsize=8,
                            color='white' if v > 0.01 else 'black')
        ax.set_title('Correctness: |lnL diff from baseline| per stage (should be 0.000)')
        plt.colorbar(im, ax=ax, label='|lnL diff|')
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, 'correctness_lnl_diff.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    # ----------------------------------------------------------------------
    # Figure 4: GPU vs CPU (VANILA) speedup — final stage
    # ----------------------------------------------------------------------
    # Pair each GPU row at the latest present stage with its CPU VANILA baseline.
    # We pick the latest stage actually present per test case so the figure
    # always reflects the most recent experiment available. Restrict to STAGES.
    gpu_in_stages = gpu[gpu['stage'].isin(STAGES)].dropna(subset=['opt_time_s'])
    latest_stage_per_tc = (gpu_in_stages
                              .groupby('test_case')['stage']
                              .apply(lambda s: max(s, key=lambda x: STAGES.index(x))))
    final_gpu = (gpu_in_stages.set_index(['test_case', 'stage'])
                   .loc[[(tc, st) for tc, st in latest_stage_per_tc.items()]]
                   .reset_index()[['test_case', 'opt_time_s', 'kernel',
                                   'data_type', 'unique_name', 'stage']].copy())
    cpu_base = cpu[['test_case', 'opt_time_s']].rename(columns={'opt_time_s': 'cpu_opt_time_s'})
    merged = final_gpu.merge(cpu_base, on='test_case', how='left')
    merged = merged.dropna(subset=['cpu_opt_time_s', 'opt_time_s'])
    merged['gpu_speedup'] = merged['cpu_opt_time_s'] / merged['opt_time_s']

    if not merged.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(merged))
        bars = ax.bar(x, merged['gpu_speedup'], color='#1E88E5', alpha=0.85,
                      edgecolor='black', linewidth=0.5)
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1, label='no speedup')
        ax.set_xticks(x)
        ax.set_xticklabels(merged['test_case'], rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Speedup (CPU single-thread / GPU latest stage)')
        ax.set_title('GPU latest-stage vs CPU baseline speedup')
        for bar, v in zip(bars, merged['gpu_speedup']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.2f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, 'gpu_vs_cpu_final.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    # ----------------------------------------------------------------------
    # Figure 5: Line plot - opt time vs stage, one line per test case
    # ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    for tc in test_cases:
        rows = gpu[gpu['test_case'] == tc].set_index('stage').reindex(STAGES)
        vals = rows['opt_time_s'].values
        stages_present = [s for s, v in zip(STAGES, vals) if pd.notna(v)]
        vals_present = [v for v in vals if pd.notna(v)]
        x = np.arange(len(stages_present))
        ax.plot(x, vals_present, marker='o', label=tc, linewidth=2, markersize=7)
        for xi, yi in zip(x, vals_present):
            ax.annotate(f'{yi:.2f}', (xi, yi), textcoords='offset points',
                        xytext=(0, 6), ha='center', fontsize=7, alpha=0.75)

    # x-ticks based on the longest stage sequence present
    all_stages = [s for s in STAGES if s in gpu['stage'].unique()]
    ax.set_xticks(range(len(all_stages)))
    ax.set_xticklabels([STAGE_PRETTY[s] for s in all_stages], fontsize=9)
    ax.set_ylabel('Parameter optimization time (s, log scale)')
    ax.set_yscale('log')
    ax.set_title('Optimization time progression — log scale')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'progression_line_plot.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ----------------------------------------------------------------------
    # CSV export
    # ----------------------------------------------------------------------
    df.to_csv(os.path.join(OUTPUT_DIR, 'all_results.csv'), index=False)
    pivot.to_csv(os.path.join(OUTPUT_DIR, 'opt_time_per_stage.csv'))
    if 'baseline' in pivot.columns:
        speedup_df = pd.DataFrame(speedup, index=pivot.index, columns=pivot.columns)
        speedup_df.to_csv(os.path.join(OUTPUT_DIR, 'speedup_vs_baseline.csv'))
    print(f"\nCSVs saved: all_results.csv, opt_time_per_stage.csv, speedup_vs_baseline.csv")

    # ----------------------------------------------------------------------
    # Text summary — baseline vs latest stage present per test case
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY (baseline → latest stage present)")
    print("=" * 80)
    if 'baseline' in pivot.columns:
        for tc in pivot.index:
            bl = pivot.loc[tc, 'baseline']
            row_stages = [s for s in STAGES if s in pivot.columns
                          and pd.notna(pivot.loc[tc, s])]
            if not row_stages or 'baseline' not in row_stages:
                continue
            final_stage = row_stages[-1]
            final = pivot.loc[tc, final_stage]
            delta = (bl - final) / bl * 100
            print(f"  {tc:<55}  {bl:>7.2f}s → {final:>7.2f}s  "
                  f"[{final_stage}] ({delta:+.1f}%, {bl/final:.2f}× faster)")

    # ----------------------------------------------------------------------
    # Figure 6: Legend explaining each optimization stage (separate figure)
    # ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 0.75 + 0.7 * len(STAGES)))
    ax.axis('off')
    stage_colors = {
        'baseline':  '#90A4AE',
        'aferprof1': '#BDBDBD',
        'aferprof2': '#42A5F5',
        'aferprof3': '#FF9800',
        'aferprof4': '#BDBDBD',
        'aferprof5': '#2E7D32',
        'aferprof6': '#6A1B9A',
        'aferprof7': '#BDBDBD',  # grey = reverted, not part of cumulative stack
    }
    y = 0.94
    ax.text(0.005, y, 'REV Kernel Optimization Journey — Stage Legend',
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    y -= 0.10
    for s in STAGES:
        color = stage_colors.get(s, '#888')
        # Colored swatch
        ax.add_patch(plt.Rectangle((0.005, y - 0.035), 0.035, 0.055,
                                    transform=ax.transAxes,
                                    facecolor=color, edgecolor='black', linewidth=0.5))
        # Short label (bold)
        ax.text(0.052, y, f'{STAGE_SHORT[s]:<5} {s:<10}',
                fontsize=10, fontweight='bold', transform=ax.transAxes,
                family='monospace', va='center')
        # Long description (wrapped)
        desc = STAGE_DESCRIPTIONS.get(s, '')
        ax.text(0.22, y, desc, fontsize=9, transform=ax.transAxes,
                va='center', wrap=True)
        y -= 0.105
    path = os.path.join(OUTPUT_DIR, 'stage_legend.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
