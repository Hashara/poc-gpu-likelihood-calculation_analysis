# H200 SM-scaling sweep — IQ-TREE3 OpenACC (2026-06-13, full)

Full 8-point sweep, **all jobs exit 0**, 1 repeat each (run-to-run variance was
measured <0.2% in the prior run, so 1 is enough). One PBS job per SM count,
identical workload, SM count restricted via CUDA MPS active-thread-percentage:

```
iqtree3 -s alignment_100000.phy -ninit 2 -seed 1     # AA, LG+I+G4, 100 taxa × 100k sites
```

- Data: `/Users/u7826985/Projects/Nvidia/results/2026_06_13_sn_sweep/sm_sweep/`
- Energy from IQ-TREE's `Energy:` block — CPU = RAPL, GPU = NVML.
- Regenerate: `python3 analysis.py`

## 🔑 Headline: SM = 1/2/4/8 are NOT distinct allocations (MPS artifact)

`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` has a **coarse minimum SM partition on H200
(~8 SMs)**. Every request from 0.76% to 6.06% lands on the *same* physical
allocation, so those four "points" are identical hardware:

| SM req. | wall (s) | GPU power | GPU energy | → interpretation |
|--------|----------|-----------|------------|------------------|
| 1 | 11232 | 146 W | 457 Wh | all four are the |
| 2 | 11244 | 149 W | 464 Wh | **same ~8-SM** |
| 4 | 11222 | 149 W | 465 Wh | partition — flat |
| 8 | 11258 | 147 W | 460 Wh | time, power, energy |

Flat GPU **power** (146–149 W) is the proof — if MPS had actually given 1 vs 8
SMs, power would scale even if time didn't (`fig_power_vs_sm.png`). Real scaling
only appears from **16 SMs** up. The nominal "efficiency" column for 1/2/4/8
(100/50/25/12.5%) is therefore meaningless — ignore it below 16.

## Results — corrected to the 5 real operating points

**Canonical figure: `fig_corrected_dashboard.png`** (data: `sm_sweep_corrected.csv`).
Because 8 SMs is the hardware floor, the 1/2/4/8 MPS requests are one point sampled
4× — they are collapsed below into a single **8-SM floor** row (mean of the 4),
and speedup/efficiency are baselined at 8 SM (not a non-existent 1-SM run).

| SMs (actual) | wall (min) | speedup vs 8-SM | efficiency vs 8-SM | GPU power (W) | GPU energy (Wh) |
|----|-----------|------|------|------|------|
| **8 (floor = 1/2/4/8 reqs)** | 187.3 | 1.00× | 100% | 148 | 461 |
| 16 | 111.6 | 1.68× | 84% | 160 | 297 |
| 33 | 53.8 | 3.48× | 84% | 207 | 186 |
| 66 | 30.4 | 6.16× | 75% | 258 | 131 |
| 132 | 19.5 | **9.62×** | **58%** | 321 | 104 |

The earlier `fig_summary_dashboard.png` / per-`fig_*` plots use the nominal MPS
request (1/2/4/8 shown separately) and a 1-SM baseline — kept for the audit
trail but **superseded by the corrected diagram**; their low-SM efficiency
numbers (100/50/25/12.5%) are artifacts of the floor, not real.

CPU energy (RAPL, whole-socket) is noisy (811–1155 Wh for identical work at the
floor) — trust the GPU/NVML column.

## Findings

1. **Bit-exact.** `logL = -7541976.86` for all 8 SM counts — MPS limiting never
   changes the result (`fig_logL_agreement.png`).

2. **Real scaling region (8→132) is good early, then tapers.** Two views, both
   baselined at the 8-SM floor:

   *Marginal* (each doubling) — what the next chunk of SMs buys:

   | step | SM ratio | speedup | step efficiency |
   |------|----------|---------|------------|
   | 8→16 | 2.0× | 1.68× | 84% |
   | 16→33 | 2.06× | 2.07× | ~100% (linear) |
   | 33→66 | 2.0× | 1.77× | 88% |
   | 66→132 | 2.0× | 1.56× | 78% |

   *Cumulative* vs the 8-SM floor (panel c of the corrected dashboard):
   100% (8) → 84% (16) → 84% (33) → 75% (66) → **58% (132)**.

   So the H200 scales near-linearly up to ~33 SMs and then runs into diminishing
   returns — the last doubling (66→132) only buys 1.56×, and at the full GPU you
   keep only 58% of ideal. Full GPU = **9.6× over the floor**
   (`fig_corrected_dashboard.png`).

3. **More SMs = faster *and* lower energy — no trade-off.** Total energy falls
   8× (≈1600 → 199 Wh) and GPU energy 4.4× (457 → 104 Wh) from floor to full GPU;
   minimum energy-delay product is at **132 SM** (`fig_energy_vs_sm.png`,
   `fig_edp_vs_sm.png`). Always run the full GPU — partitioning helps neither
   speed nor energy.

4. **Memory-bound, big headroom.** Even at 132 SM the board averages **321 W**,
   ~46% of the H200's ~700 W TDP (`fig_power_vs_sm.png`). The tapering scaling +
   low power = HBM-bandwidth-bound kernels, the same headroom the kernel
   optimisation roadmap targets.

## Figures

**Publication-quality (8-SM-floor framing, `plot_improved.py`):**
- `fig_corrected_dashboard.png` — canonical 2×2 (runtime · speedup · efficiency · energy-vs-time Pareto)
- `fig2_runtime.png` · `fig2_speedup.png` (shaded = scaling lost to HBM bound) ·
  `fig2_efficiency.png` · **`fig2_pareto.png`** (energy vs runtime — 132 SM dominates both)

**Evidence / audit (`analysis.py`):** `fig_power_vs_sm` (MPS-quantization proof:
flat power 1→8) · `fig_logL_agreement` · `fig_energy_breakdown`. The remaining
`fig_*_vs_sm` (nominal-request, 1-SM baseline) are superseded by the corrected set.

## Note for future sub-16-SM runs

MPS thread-percentage **cannot** resolve below ~8 SMs on H200 — it floors. To
measure 1/2/4-SM behaviour you'd need **MIG** slices (1g/2g/3g…) or **CUDA green
contexts** for exact small partitions. That said, the low-SM region isn't where
the project's interest lies; the actionable conclusions (full GPU is
Pareto-optimal; kernels are memory-bound) stand on the 16→132 data. To sharpen
the knee, add 24/48/96 SM.
