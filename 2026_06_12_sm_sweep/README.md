# H200 SM-scaling sweep — IQ-TREE3 OpenACC (2026-06-12)

**Question.** On an H200 (132 SMs), how do wall time and energy scale as we
restrict the GPU to *N* SMs via CUDA MPS (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`)?
One PBS job per SM count, identical workload each time:

```
iqtree3 -s alignment_100000.phy -ninit 2 -seed 1     # AA, LG+I+G4, 100 taxa × 100k sites
```

- Data: `/Users/u7826985/Projects/Nvidia/results/sm_sweep/sm_<N>/`
- Energy from IQ-TREE's built-in `Energy:` block — CPU = RAPL, GPU = NVML.
- Regenerate: `python3 analysis.py`

## ⚠️ Data is partial — the knee is missing

The per-SM PBS walltimes in the original `submit_sweep.sh` were too tight, so
**every job except `sm_132` was killed mid-run** (PBS exit −29). Completed runs:

| SM | % of GPU | complete reps | status |
|----|----------|---------------|--------|
| 1 | 0.76 | 1 | rep2 killed at 6 h |
| 2 | 1.52 | 1 | rep2 killed at 4 h |
| **4** | 3.03 | **0** | killed at 10838 s (mid rep1) |
| **8** | 6.06 | **0** | killed at 7268 s (mid rep1) |
| **16** | 12.12 | **0** | killed at 5443 s (mid rep1) |
| 33 | 25 | 1 | rep2 killed at 1.5 h |
| 66 | 50 | 2 | rep3 killed at 1.5 h |
| 132 | 100 | 3 | ✅ clean exit |

So **SM = 4, 8, 16 produced no usable data** — and that's exactly the knee of the
curve (we jump straight from 2 SM to 33 SM). ~1855 SU were spent, ~590 of them on
the three jobs that yielded nothing. `benchmarks/submit_sweep.sh` has since been
fixed to size walltime from measured per-rep runtimes.

## Results (completed runs)

| SM | wall (min) | speedup vs 1 SM | parallel eff. | GPU energy (Wh) | CPU energy (Wh) | total (Wh) | GPU avg (W) |
|----|-----------|-----------------|---------------|-----------------|-----------------|-----------|-------------|
| 1 | 187.2 | 1.00× | 100% | 457 | 882 | 1340 | 147 |
| 2 | 187.5 | 1.00× | 50% | 457 | 1125 | 1582 | 146 |
| 33 | 53.8 | 3.48× | 10.5% | 178 | 296 | 474 | 199 |
| 66 | 30.5 | 6.15× | 9.3% | 130 | 132 | 262 | 255 |
| 132 | 19.5 | 9.60× | 7.3% | 104 | 87 | 191 | 320 |

(`sm_sweep_summary.csv` = per-SM means, `sm_sweep_runs.csv` = every run.)

## Findings

1. **Bit-exact across all SM counts.** Every run — all SM counts, all reps —
   gives `logL = -7541976.86`. MPS SM-limiting does not perturb results
   (`fig_logL_agreement.png`). Good correctness signal for the partitioning approach.

2. **Strongly sublinear scaling.** The full 132 SMs deliver only **9.6×** over a
   single SM; parallel efficiency falls from 50% (2 SM) to **7.3%** at 132 SM
   (`fig_speedup_vs_sm.png`, `fig_efficiency_vs_sm.png`). The H200 is badly
   under-fed by this workload — consistent with HBM-bandwidth-bound kernels plus
   the launch-bound gradient path (the dry-run showed the deriv kernels launch
   ~58k times with only **5 gangs** each, so they can't use more than ~5 SMs).

3. **1→2 SM plateau.** 2 SMs give *zero* speedup over 1 (0.998×); both ModelFinder
   and tree-search phases are byte-for-byte the same wall time. Below a few SMs the
   runtime is pinned by something a second SM doesn't relieve.

4. **More SMs = faster *and* lower energy — no trade-off.** Total energy drops
   **7.0×** (1340 → 191 Wh) from 1 → 132 SM; min energy-delay product is at 132 SM
   (`fig_energy_vs_sm.png`, `fig_edp_vs_sm.png`). Idle SMs still draw power, so the
   9.6× time saving dominates. CPU energy falls hardest (10×): at 1 SM the 3.1-hour
   runtime accrues ~882 Wh of CPU "wall-clock tax" even though the CPU isn't the
   bottleneck. **Takeaway: for this workload, always run the full GPU — partitioning
   buys nothing on either axis.**

5. **Power utilisation stays low.** Even at 132 SM the board averages **320 W**
   (~46% of the H200's ~700 W TDP); at 1 SM it's 147 W (`fig_power_vs_sm.png`). The
   unused power headroom is the memory-bound stall — the same headroom the kernel
   optimisation roadmap targets.

## Figures

`fig_runtime_vs_sm` · `fig_speedup_vs_sm` · `fig_efficiency_vs_sm` ·
`fig_energy_vs_sm` · `fig_power_vs_sm` · `fig_energy_breakdown` ·
`fig_logL_agreement` · `fig_edp_vs_sm`

## Recommended re-run

Fill the missing knee. You already have 1, 2, 33, 66, 132 and run-to-run variance
is tiny (sm_132 reps within 1 s, sm_66 within 2 s), so **1 repeat is enough**:

```bash
# in submit_sweep.sh:  SM_LIST="4 8 16"   REPEATS=1
bash submit_sweep.sh
```

That fills the gap for ~680 SU instead of re-running everything (~4200 SU at
REPEATS=3). Adding 6, 12, 24 would sharpen the knee further.
