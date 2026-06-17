# H200 SM-scaling sweep - IQ-TREE3 OpenACC, ModelFinder = `-m TEST` (2026-06-16)

5-point sweep (`8 16 33 66 132` SMs), one repeat per SM count. Each run used the
same workload, with SM count restricted via CUDA MPS active-thread-percentage:

```bash
iqtree3 -s alignment_100000.phy -m TEST -ninit 2 -seed 1
```

- Data: `/Users/u7826985/Projects/Nvidia/results/2026_06_16_sm_sweep_m_test/sm_<N>/` (prefix `MTEST_`)
- Energy source: IQ-TREE `Energy:` block, with CPU from RAPL and GPU from NVML.
- Regenerate: `python3 analysis.py`
- Baseline: 8 SM, because the H200 MPS minimum partition is the practical hardware floor.

## Summary

`-m TEST` picks `LG+G4` for every SM count, and the best log-likelihood is
bit-exact across the sweep: `-7541976.86`.

| SMs | total wall (min) | tree search (min) | ModelFinder (min) | total speedup | tree-search speedup | ModelFinder speedup | efficiency | GPU energy (Wh) |
|----:|-----------------:|------------------:|------------------:|--------------:|--------------------:|--------------------:|-----------:|----------------:|
| 8   | 170.3 | 132.6 | 36.9 | 1.00x | 1.00x | 1.00x | 100% | 415 |
| 16  | 101.7 | 79.0  | 22.2 | 1.67x | 1.68x | 1.66x | 84%  | 280 |
| 33  | 48.8  | 37.7  | 10.9 | 3.49x | 3.52x | 3.40x | 85%  | 167 |
| 66  | 27.6  | 21.1  | 6.3  | 6.17x | 6.28x | 5.84x | 75%  | 117 |
| 132 | 17.8  | 13.5  | 4.2  | 9.57x | 9.84x | 8.79x | 58%  | 94  |

## Figures

**Canonical:** `fig_dashboard.png` - TEST-only 2x2 dashboard:
runtime, speedup by phase, parallel efficiency, and energy-vs-runtime.

**Compact dashboard:** `fig_runtime_speedup_dashboard.png` - runtime and
speedup by phase only.

**Speedup:** `fig_speedup_vs_sm.png` now shows all requested speedups:
total wall time, tree search, and ModelFinder, each relative to the 8-SM floor.

**Per-metric figures:** `fig_runtime_vs_sm.png`, `fig_efficiency_vs_sm.png`,
`fig_energy_vs_sm.png`, `fig_energy_breakdown.png`, `fig_power_vs_sm.png`,
`fig_edp_vs_sm.png`, and `fig_logL_agreement.png`.

`fig_corrected_dashboard.png` is retained as an alias of the TEST-only dashboard.

## CSVs

- `sm_sweep_runs.csv` - parsed and derived per-SM metrics.
- `sm_sweep_summary.csv` - same 5-point TEST-only summary, including
  `speedup`, `ts_speedup`, and `mf_speedup`.
