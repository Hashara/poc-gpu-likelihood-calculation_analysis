# H200 SM-scaling sweep - IQ-TREE3 OpenACC, DNA 1M, `-m TEST` (2026-06-16)

5-point sweep (`8 16 33 66 132` SMs), one repeat per SM count. Each run used the
same workload, with SM count restricted via CUDA MPS active-thread-percentage:

```bash
iqtree3 -s alignment_1000000.phy -m TEST -ninit 2 -seed 1
```

- Data: `/Users/u7826985/Projects/Nvidia/results/2026_06_16_sm_sweep_dna1M/sm_<N>/` (prefix `MTEST_dna1M_`)
- Energy source: IQ-TREE `Energy:` block, with CPU from RAPL and GPU from NVML.
- Regenerate: `python3 analysis.py`
- Baseline: 8 SM, because the H200 MPS minimum partition is the practical hardware floor.

## Summary

`-m TEST` picks `F81+F+G4` for every SM count, and the best log-likelihood is
bit-exact across the sweep: `-59208019.245`.

| SMs | total wall (min) | tree search (min) | ModelFinder (min) | total speedup | tree-search speedup | ModelFinder speedup | efficiency | GPU energy (Wh) | total energy (Wh) | GPU mem (GB) |
|----:|-----------------:|------------------:|------------------:|--------------:|--------------------:|--------------------:|-----------:|----------------:|------------------:|-------------:|
| 8   | 274.5 | 160.5 | 113.0 | 1.00x | 1.00x | 1.00x | 100% | 699 | 1934 | 30.2 |
| 16  | 162.4 | 94.6  | 67.2  | 1.69x | 1.70x | 1.68x | 85%  | 473 | 1166 | 30.2 |
| 33  | 78.3  | 45.0  | 32.8  | 3.51x | 3.57x | 3.44x | 85%  | 306 | 643  | 30.3 |
| 66  | 44.5  | 25.1  | 19.0  | 6.17x | 6.38x | 5.94x | 75%  | 217 | 418  | 30.4 |
| 132 | 29.2  | 16.2  | 12.7  | 9.39x | 9.91x | 8.89x | 57%  | 183 | 346  | 30.7 |

## Figures

**Canonical:** `fig_dashboard.png` - DNA-only 2x2 dashboard:
runtime, speedup by phase, parallel efficiency, and energy-vs-runtime.

**Compact dashboard:** `fig_runtime_speedup_dashboard.png` - runtime and
speedup by phase only.

`fig_corrected_dashboard.png` is retained as an alias of the same DNA-only
dashboard.

**Speedup:** `fig_speedup_vs_sm.png` shows total wall time, tree-search, and
ModelFinder speedups, each relative to the 8-SM floor.

**Per-metric figures:** `fig_runtime_vs_sm.png`, `fig_efficiency_vs_sm.png`,
`fig_energy_vs_sm.png`, `fig_energy_breakdown.png`, `fig_power_vs_sm.png`,
`fig_edp_vs_sm.png`, and `fig_logL_agreement.png`.

## CSVs

- `sm_sweep_runs.csv` - parsed and derived per-SM metrics.
- `sm_sweep_summary.csv` - same 5-point DNA-only summary, including
  `speedup`, `ts_speedup`, and `mf_speedup`.
