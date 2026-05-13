# REV Kernel Optimization Journey — 2026-04-20 to 2026-04-22

A week of profile-driven optimization of IQ-TREE3's OpenACC REV partial-LH kernels.
Full data: `all_results.csv`, `opt_time_per_stage.csv`, `speedup_vs_baseline.csv`.
Plots: `progression_opt_time_per_test.png`, `stage_legend.png`, `speedup_vs_baseline_per_stage.png`.

## Stage Summary

| Stage | Short | Experiment | Outcome |
|---|---|---|---|
| baseline  | base | Original REV partial-LH kernels pre-profiling | Reference |
| aferprof1 | aP1  | Phase 2 inner-loop hoist (pre-compute row pointers) | Code cleanup, 6 regs/thread saved, runtime neutral |
| aferprof2 | aP2  | Phase 2 `inv_evec` transpose (column-major mirror) | **2.27× Phase-2 kernel, -15.7% end-to-end AA REV** |
| aferprof3 | aP3  | `maxregcount:96` compiler flag | REV occupancy 22.6 → 28.0%, but NonRev regressed 3.8% → **REVERTED** |
| aferprof4 | aP4  | Phase 1 forward-transform hoist | Foundation, neutral |
| aferprof5 | aP5  | Phase 1 `echildren` transpose (Phase 0 helper kernel) | **2.7× Phase-1 kernel, -15.7% additional end-to-end** |
| aferprof6 | aP6  | TipTip_Rev fully batched across ops | **TipTip 13.7× faster** (kernel), end-to-end neutral at 100-taxa. **Committed `ec5b576b`** |
| aferprof7 | aP7  | TipInternal_Rev + InternalInternal_Rev batched (same recipe) | **Reverted.** TipInt 2.64× / IntInt 2.15× faster (kernel), **but deriv kernel 2× slower** due to 1.1 GB L2-cache pollution from enlarged `eigen_prod` working set (block=1000 on LG+I+G4 × ~35 ops per level). Net end-to-end: noise. |

## Per-Test Results (parameter-optimization time, seconds)

**Final state = aferprof6 (TipTip batched, Int-Int/Tip-Int unchanged from aP5).** aP7 was attempted and reverted — see next section.

| Test case | base | aP1 | aP2 | aP3 | aP4 | aP5 | **aP6 (final)** | Latest vs base |
|---|---|---|---|---|---|---|---|---|
| AA NONREV 100taxa (LG+G4)     | 1.38 | 1.37 | 1.37 | 1.42 | 1.39 | 1.36 | 1.38 | **1.00×** (neutral, control) |
| AA REV 100taxa LG+G4          | 1.44 | 1.43 | 1.29 | 1.24 | 1.25 | 1.06 | **1.07** | **1.35×** faster |
| AA REV 100taxa LG+I+G4        | 18.75 | 18.96 | 15.81 | 15.76 | 16.26 | 12.24 | **12.14** | **1.55×** faster |
| DNA NONREV 100taxa (GTR+I+G4) | 0.13 | 0.13 | 0.14 | 0.14 | 0.13 | 0.13 | 0.14 | **0.98×** (within noise) |
| DNA REV 100taxa GTR+I+G4      | 0.27 | 0.28 | 0.28 | 0.28 | 0.27 | 0.28 | 0.28 | **0.96×** (within noise) |

**Observations:**
- **AA REV LG+I+G4 cumulative: 1.55× faster** vs baseline
- **NonRev controls flat** — no leaks into unrelated paths
- **DNA REV flat** — stride-4 work per pattern already tiny

## aferprof7 — Attempted and Reverted

**What was tried:** Apply the aP6 batching recipe (`collapse(3) + per-op eigen_prod slice`) to `batchedInternalInternal_Rev` and `batchedTipInternal_Rev`.

**Kernel-level results (NCU, AA LG+I+G4):**
- TIP-INT kernels: **47.524s → 18.035s (2.64×)**
- INT-INT kernels: **83.081s → 38.656s (2.15×)**

**But the net end-to-end was neutral (+2% noise)** because the derivative kernel path slowed by 2×:
- GPU deriv kernel: **35.9s → 74.5s**
- Partial-LH traversal+stale recomp: **19.9s → 40.1s**

**Root cause — cache pollution from enlarged live `eigen_prod` footprint:**
- On LG+I+G4, `block = nstates × ncat_mix = 20 × 50 = 1000`
- Per-op slice = `nptn × block × 8 B` ≈ **32 MB**
- Max Int-Int ops per level (100-taxa) ≈ 35
- Live working set = **~1.1 GB** vs V100 L2 cache = 6 MB (180× over capacity)
- DRAM writeback saturation blew the cache for subsequent deriv kernels

**Why aP6 TipTip batching was OK but aP7 wasn't:**
- TipTip `block = 80-100` → per-op slice ~2.5 MB, live set under L2 → safe
- Int-Int / Tip-Int at `block = 1000` → per-op slice 32 MB, live set 180× L2 → unsafe

**Lesson for future batching:**
The per-op slice batching pattern is safe only when `num_ops × nptn × block × 8B` is L2-friendly (< ~6 MB on V100). For mixture models with large `block`, alternative approaches needed:
- Shared-memory staging of the per-op slice
- Tile the op dimension into small chunks (e.g., 4 ops per launch)
- Fuse Phase 1 + Phase 2 (avoids intermediate buffer entirely)

**Revert action:** `git checkout HEAD -- tree/phylokernel_openacc.cpp` — restored to aP6 commit `ec5b576b`. aP7 changes never committed.

## What aferprof6 Proved

Captured from the **NCU-instrumented** AA LG+I+G4 run (where we can compare kernel timing across runs with same capture window):

| Call # | Nodes | TipTip time aP5 | TipTip time aP6 | **Speedup** |
|---|---|---|---|---|
| 10 | 264  | 11.451 s | **0.180 s** | **63.6×** |
| 20 | 594  | 11.676 s | **0.400 s** | **29.2×** |
| 30 | 858  | 11.856 s | **0.577 s** | **20.5×** |
| 40 | 1122 | 12.036 s | **0.754 s** | **16.0×** |
| 50 | 1320 | 12.171 s | **0.886 s** | **13.7×** |

- **TipTip share of GPU time: 9.9% → 0.6%**
- **Correctness: lnL = -837336.222 in both runs (exact match)**

## Correctness (all stages)

All 5 test cases produce **exact lnL match** with the baseline across every stage (see `correctness_lnl_diff.png`). No numerical drift was introduced at any stage.

## What's Next

The aferprof6 profile shows the new bottleneck:
- **TipTip kernels: 0.6%** ✅ done
- **TIP-INT kernels: 34%** — still serial `for(op)` loop
- **INT-INT kernels: 60%** — still serial `for(op)` loop
- **Derivative kernels: separate path** — also needs batching

Applying the **same batching recipe** (remove outer `for(op)`, collapse(3) with parallel `(op,p,s)`) to `batchedTipInternal_Rev` and `batchedInternalInternal_Rev` should give the biggest remaining end-to-end win.

## Deferred / On Hold

- `maxregcount:96`: blocked on NonRev kernel optimization (shares the .cpp file)
- Shared-memory staging of `inv_evec_T`: may become unnecessary once batching spreads warps across more ops
- Derivative kernel transpose + batch: not yet measured, likely big win
