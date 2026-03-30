# Post-Profiling ModelFinder GPU Optimizations — 2026-03-27

## Overview

Optimizations guided by nsys + ncu profiling of the MS6+MS9 build on DNA JC+G4 (100 taxa, 1K sites, V100 GPU).

**Hardware:** NVIDIA V100 GPU (Gadi HPC, NCI Australia)
**Dataset:** 100 taxa, 1000 sites
**Alignments:** DNA (GTR+I+G4), AA (LG+I+G4)
**Branch:** `openacc-kernel-v2`
**Starting point:** MS6+MS9 shipped (DNA 34.7s, AA 123.3s)

---

## Profiling Summary

Profiled with `nsys` (timeline) and `ncu` (per-kernel metrics) on DNA JC+G4.

### GPU Time Breakdown (132 ms total kernel time, 2 models)

| Kernel Category | Time (ms) | % | Calls |
|---|---|---|---|
| **Derivative kernels** | **72.2** | **54.5%** | 4,754 |
| Partial LH (stale recomp) | 34.1 | 25.7% | 4,008 |
| Batched partial LH | 24.8 | 18.7% | 4,444 |
| Reduction (likelihood) | 1.3 | 1.0% | 266 |

### PCIe Transfer Breakdown (92 ms total, 2 models)

| Transfer Type | Count | Time (ms) | % |
|---|---|---|---|
| H→D tiny (<256B) | 15,746 | 22.2 | 24% |
| **D→H tiny (<256B)** | **24,036** | **36.3** | **40%** |
| H→D data (256B-1MB) | 17,441 | 32.5 | 35% |
| D→H data (4-64KB) | 266 | 0.7 | 1% |

**39,782 tiny transfers (<256B) consume 58.5 ms = 64% of all transfer time.**

### GPU Kernel Utilization (ncu)

| Kernel | Grid Size | Regs/Thread | SM Utilization | Problem |
|---|---|---|---|---|
| batchedTipTip | 4,433 | 48 | **49.9%** | Only well-utilized kernel |
| **derivKernelIntInt** | **8** | **96** | **0.8%** | #1 bottleneck — 90% GPU idle |
| **derivKernelTipInt** | **8** | **106** | **0.5%** | #2 bottleneck |
| computeTransDerivOnGPU | 1 | 80 | 0.05% | 1 block on 80 SMs |
| reductionTipInt | 8 | 54 | 0.3% | Severely underutilized |

**Root cause:** Derivative/reduction kernels parallelize over ~1000 patterns with vector_length(128) = 8 thread blocks on 80 SMs. Register pressure (96-106 regs/thread) further limits occupancy.

---

## Optimization Plan

See [plan_after_profiling.md](plan_after_profiling.md) for the full 7-step plan.

| Step | Optimization | Effort | Status |
|---|---|---|---|
| 1 | Eigen dirty flag (Opt-A) | 5 lines | **Shipped** ✅ |
| 2 | Persistent offset arrays (Opt-C) | 40 lines | **Shipped** ✅ |
| 2b | M5: Persistent offsets (iqtree3-forked) | 30 lines | **Validated** ✅ |
| 3 | Batch df/ddf downloads (Opt-B) | 30 lines | Planned |
| 4 | Profile validation | — | Planned |
| 5 | Register pressure reduction (Opt-D) | 50 lines | Planned |
| 6 | Derivative parallelism (Opt-E) | 100 lines | Planned |
| 7 | Async kernel launches (Opt-F) | 30 lines | Planned |

---

## Step 1: Opt-A — Eigen Dirty Flag ✅

### Problem

`uploadEigenToGPU()` called ~2,355 times per 2 models during Newton-Raphson branch optimization. The eigendecomposition data (eigenvalues, eigenvectors, state_freq, rate_cats, rate_props) does NOT change when only branch lengths change.

nsys confirmed: `computeTransDerivOnGPU` was the 3rd most time-consuming GPU operation at 9.7 ms per 2 models.

### Fix

Added `bool gpu_eigen_dirty = true` flag:
- **Set `true`** after `model->optimizeParameters()` and `site_rate->optimizeParameters()` in `modelfactory.cpp`
- **Checked** in `uploadEigenToGPU()` — early return when `!gpu_eigen_dirty`
- **Set `false`** after successful upload
- **Set `true`** on realloc (size change)

Defaults to `true` (fail-safe: first call always uploads).

### When eigendata changes

| Event | Eigenvalues/vectors | rate_cats/props | state_freq |
|---|---|---|---|
| Branch optimization (NR) | NO | NO | NO |
| Q-matrix optimization (BFGS) | YES | NO | YES |
| Gamma shape optimization (Brent) | NO | YES | NO |
| p_invar optimization | NO | NO | NO |

### Files Changed

- `tree/phylotree.h` — 1 line (new member `gpu_eigen_dirty`)
- `tree/phylokernel_openacc.cpp` — 3 lines (early return + clear flag + set on realloc)
- `model/modelfactory.cpp` — 8 lines (set dirty after model + rate optimization, with `#ifdef USE_OPENACC` guards)

**Total: 3 files, 21 lines**

### Results

| Metric | DNA OpenACC | AA OpenACC |
|---|---|---|
| Before (MS6+MS9, this run) | 32.7s | 133.2s |
| + Eigen flag | 34.3s | 126.7s |
| **Incremental change** | **+1.7s (noise)** | **-6.5s (4.9%)** |

**DNA:** +1.7s is within run-to-run variance (~5%). The "Before" baseline (32.7s) was already 2s faster than the previous MS6+MS9 measurement (34.7s), confirming server load variation.

**AA:** Clear 6.5s improvement. Eigen upload is 6.8 KB per call for AA (vs 384B for DNA), so skipping ~2,355 redundant uploads saves measurable PCIe time.

### Commit

```
Opt-A: Skip redundant uploadEigenToGPU during branch optimization

Eigendecomposition data (eigenvalues, eigenvectors, rate_cats, rate_props,
state_freq) is unchanged when only branch lengths change. Add gpu_eigen_dirty
flag set after model/rate parameter optimization, cleared after GPU upload.
Skips ~2,355 identical uploads per 2 models during Newton-Raphson branch
optimization. AA: -6.5s (4.9%). DNA: within noise.
```

---

## Step 2: Opt-C — Persistent Offset Arrays ✅

### Problem

Each batched kernel launch (batchedTipTip, batchedTipInternal, batchedInternalInternal) allocates a temporary offset array on the heap, uploads it to GPU via `copyin`, then deletes it. This pattern repeats ~15 times per likelihood evaluation (once per tree level per kernel type), across ~6,000+ evaluations per model.

nsys confirmed: 15,746 tiny H→D transfers (<256B) consuming 22.2 ms per 2 models. Each `copyin` on a fresh `new[]` pointer triggers a GPU malloc → H→D transfer → kernel → GPU free cycle.

### Fix

Replaced the per-level `new[]/copyin/delete[]` pattern with a persistent GPU-resident offset buffer:

1. **New helper function `ensureOffsetBuffer()`** — allocates a persistent `size_t[]` buffer on host+GPU. Grows if needed (with headroom to avoid frequent reallocs). Reuses existing buffer when large enough.

2. **6 call sites changed** — `new size_t[N]` → `ensureOffsetBuffer(this, N)`, `delete[]` removed.

3. **3 kernel functions changed** — `copyin(offsets[...])` → `present(offsets[...])` inside the `#pragma acc data` block. The data is already on GPU from the persistent buffer.

4. **6 `#pragma acc update device(offsets[...])` added** — before each kernel call, uploads the freshly-filled offset data to the already-allocated GPU buffer. This is a pure data transfer with no GPU malloc/free overhead.

5. **Cleanup in `freeOpenACCData()`** — frees the persistent buffer when GPU data is released.

### Why This Works

Before: Each `copyin(offsets[0:N])` on a fresh pointer triggers:
```
GPU malloc(N*8) → H→D transfer(N*8) → kernel uses data → GPU free(N*8)
```

After: `update device(offsets[0:N])` on a persistent pointer triggers:
```
H→D transfer(N*8) → kernel uses data
```

The GPU malloc/free overhead per transfer (~2-5 µs) is eliminated. Over 15,000 transfers, this saves ~30-75 ms per 2 models.

### Files Changed

- `tree/phylotree.h` — 3 lines (new members: `gpu_offset_buf`, `gpu_offset_buf_size`, `gpu_offset_buf_resident`)
- `tree/phylokernel_openacc.cpp` — 61 lines (helper function + 6 call sites + 6 update directives + 3 kernel `copyin→present` changes + cleanup)

**Total: 2 files, 64 lines**

### Results

| Metric | DNA OpenACC | AA OpenACC |
|---|---|---|
| + S1 Eigen flag | 34.3s | 126.7s |
| + S2 Persistent offsets | 31.4s | 120.6s |
| **Incremental change (S2 vs S1)** | **-2.9s (8.5%)** | **-6.2s (4.9%)** |

| Metric | DNA | AA |
|---|---|---|
| Cumulative vs Before (S1+S2) | -1.2s (-3.8%) | -12.6s (-9.5%) |
| vs Original baseline | **-10.7s (-25.4%)** | **-109.2s (-47.5%)** |
| GPU vs 1 CPU speedup | 0.76x | **5.05x** ← crossed 5× |

**DNA:** -2.9s over S1. Consistent with eliminating ~15K tiny H→D transfer overhead.

**AA:** -6.2s over S1. AA benefits more because the offset arrays are larger (more nodes per level with nstates=20) and the per-transfer GPU malloc overhead is fixed-cost.

**Milestone:** AA crossed the **5× GPU speedup** over 1 CPU (5.05×) for the first time.

### Commit

```
Opt-C: Persistent offset buffer for batched kernels

Replace per-level new[]/copyin/delete[] with a persistent GPU-resident
offset buffer. ensureOffsetBuffer() allocates once, grows if needed.
Kernel copyin() changed to present() + prior update device().
Eliminates ~15K GPU malloc/free cycles per 2 models.
DNA: -2.9s (8.5%). AA: -6.2s (4.9%). AA reaches 5.05x GPU/CPU speedup.
```

---

## Step 2b: M5 — Persistent Offset Arrays (iqtree3-forked branch) ✅

### Context

Re-implementation of Step 2 (Opt-C) on the `openacc_kernels` branch of `iqtree3-forked`. Validates the same optimization with a simplified approach: fixed pre-allocation to `leafNum * 8` instead of the growable `ensureOffsetBuffer()` helper.

### Implementation Differences vs Step 2

| Aspect | Step 2 (Opt-C) | M5 (iqtree3-forked) |
|---|---|---|
| Buffer allocation | `ensureOffsetBuffer()` — growable with headroom | Fixed `new size_t[leafNum * 8]` at first-call setup |
| Member naming | `gpu_offset_buf`, `gpu_offset_buf_size` | `gpu_offsets`, `gpu_offsets_size` |
| Realloc strategy | Grows 2× when exceeded | No realloc — pre-sized to max possible batch |
| Code delta | 64 lines (2 files) | ~30 lines (2 files) |
| Core mechanism | Identical: `copyin → present` + `update device` | Identical |

### Results

| Metric | DNA OpenACC | DNA Vanilla | AA OpenACC | AA Vanilla |
|---|---|---|---|---|
| M5 wall-clock (ModelFinder) | 31.35s | 24.66s | 121.02s | 543.92s |
| S2 wall-clock (for comparison) | 31.4s | 23.7s | 120.6s | 609.3s |

| Metric | DNA | AA |
|---|---|---|
| GPU vs 1-core CPU | 1.27× slower | **4.49× faster** |
| M5 vs S2 (GPU) | -0.05s (−0.2%) | +0.42s (+0.3%) |

**M5 confirms S2 within noise.** Both approaches eliminate the same ~15K GPU malloc/free cycles.

### Correctness

All log-likelihoods match VANILLA exactly across all models tested:
- **DNA:** 100 models (968 candidates), best model F81+F+G4, score **-56182.151** (both)
- **AA:** 128 models (1232 candidates), best model LG+G4, score **-77823.964** (both)

Only R5/I+R5 models show ≤0.002 unit diffs — numerical noise from iterative optimization convergence, not correctness issues. All simple models match to all reported decimal places.

### Note on Vanilla Baselines

AA Vanilla times differ between runs (609.3s S2 vs 543.9s M5) due to different CPU nodes:
- S2: unknown node assignment
- M5: gadi-cpu-clx-0133

The GPU times are stable across runs (120.6s vs 121.0s), confirming the GPU is not affected by node variability.

---

## Step 3: M6 — T2 Template-Based Optimizations (iqtree3-forked) ⚠️ AA REGRESSION

### Changes

Four optimizations applied together as "M6":

1. **Step 1 — DNA derivative kernel VL=128→32:** Reduce `vector_length` from 128 to 32 in `derivKernelTipInt<4>` and `derivKernelIntInt<4>`. Increases block count from 8 to 32 on 80 SMs. Each thread still processes one independent pattern — 100% lane utilization maintained.

2. **Step 2 — Template `computeTransDerivOnGPU<NSTATES>`:** Added `template<int NSTATES>` with `if constexpr` for compile-time div/mod optimization (DNA: shift/mask) and inner loop unrolling. Dispatched via `OPENACC_DISPATCH_NSTATES` macro.

3. **Step 3 — Persistent `partial_lh_node` buffer:** `reductionKernelTipInt`'s per-call `new[]/copyin()/delete[]` replaced with persistent `enter data create` + `update device` + `present()`.

4. **Step 4 — Persistent `trans_mat` buffer (likelihood path):** `reductionKernelIntInt`'s per-call `new[]/copyin()/delete[]` replaced with same persistent pattern.

### Results

| Metric | DNA OpenACC | DNA Vanilla | AA OpenACC | AA Vanilla |
|---|---|---|---|---|
| M6 wall-clock (ModelFinder) | 31.11s | 23.99s | 141.03s | 538.09s |
| M5 baseline (for comparison) | 31.35s | 24.66s | 121.02s | 543.92s |

| Metric | DNA | AA |
|---|---|---|
| **M6 vs M5 (GPU)** | **-0.24s (−0.8%)** | **+20.0s (+16.5%) ⚠️ REGRESSION** |
| GPU vs 1-core CPU | 1.30× slower | **3.81× faster** |

### Analysis

**DNA: Neutral (−0.8%).** The VL=128→32 change (Step 1) was expected to give 8-14% speedup from 4× more blocks. The result is flat. Possible explanations:
- The derivative kernels may be compute-bound (not occupancy-limited) on this dataset — more blocks don't help if each block already saturates the FP64 pipeline
- The nvc++ compiler may have generated different register allocation at VL=32, changing the instruction mix
- The profiling showed 0.5-0.8% SM utilization for derivative kernels, but this measures grid-level occupancy, not throughput per SM

**AA: 16.5% regression.** This is a real regression — M6 AA (141.0s) is worse than even the MS6+MS9 baseline (133.2s). The vanilla baseline is stable (538s vs 544s), ruling out server variance.

Most likely culprit: **Steps 3 or 4 (persistent buffers for reduction kernels)**. The `present()` lookup for persistent buffers may have different overhead characteristics than scoped `copyin()` in the nvc++ OpenACC runtime, especially when called ~1232× (one per AA model). The `copyin()` on a fresh pointer avoids the device-mapping table lookup that `present()` requires.

### Correctness

All log-likelihoods match VANILLA exactly:
- **DNA:** Best model F81+F+G4, score **-56182.1512** (both)
- **AA:** Best model LG+G4, score **-77823.9643** (both)

### Verdict: REVERT M6 changes on iqtree3-forked

The DNA improvement is negligible and the AA regression is significant. All four T2 changes should be reverted. The VL tuning hypothesis (Step 1) was disproved — derivative kernel performance is not occupancy-limited on this workload.

---

## Cumulative Results Summary

| Stage | DNA OpenACC | AA OpenACC | AA GPU/CPU |
|---|---|---|---|
| Original baseline | 42.1s | 229.8s | 2.2× |
| + MS6 (selective upload) | 34.7s | 123.3s | 4.5× |
| + MS9 (skip pattern_lh_cat) | 34.7s | 123.3s | 4.5× |
| + S1 Eigen dirty flag | 34.3s | 126.7s | 4.2× |
| + S2 Persistent offsets | 31.4s | 120.6s | 5.1× |
| + M5 Persistent offsets (forked) | **31.4s** | **121.0s** | **4.5×** |
| + M6 T2 Templates (forked) | 31.1s | 141.0s ⚠️ | 3.8× ⚠️ |

**Best achieved: DNA 31.1s, AA 121.0s (M5)**

*Note: M6 AA regressed — recommended to stay at M5 baseline. M5 AA GPU/CPU ratio (4.5×) uses a different (faster) vanilla baseline node than S2 (5.1×). GPU times are stable: 120.6s vs 121.0s.*

---

## Results Data Location

```
results/2026_03_27_modelselection_opt_after_prof/
├── DNA/
│   ├── *before_prof_opt*OPENACC*              # MS6+MS9 baseline (32.7s)
│   ├── *before_prof_opt*VANILA*               # Vanilla baseline (26.9s)
│   ├── *s1_eigenflag*OPENACC*                 # + Opt-A (34.3s)
│   ├── *s1_eigenflag*VANILA*                  # Vanilla (24.0s)
│   ├── *s2_persistance_offsetarray*OPENACC*   # + Opt-A+C (31.4s)
│   ├── *s2_persistance_offsetarray*VANILA*    # Vanilla (23.7s)
│   ├── *m5_persistant_array*OPENACC*          # M5 forked branch (31.4s)
│   ├── *m5_persistant_array*VANILA*           # Vanilla (24.7s)
│   ├── *m6_templates*OPENACC*                 # M6 T2 templates (31.1s) ⚠️
│   └── *m6_templates*VANILA*                  # Vanilla (24.0s)
└── AA/
    ├── *before_prof_opt*OPENACC*              # MS6+MS9 baseline (133.2s)
    ├── *before_prof_opt*VANILA*               # Vanilla baseline (554.6s)
    ├── *s1_eigenflag*OPENACC*                 # + Opt-A (126.7s)
    ├── *s1_eigenflag*VANILA*                  # Vanilla (535.6s)
    ├── *s2_persistance_offsetarray*OPENACC*   # + Opt-A+C (120.6s)
    ├── *s2_persistance_offsetarray*VANILA*    # Vanilla (609.3s)
    ├── *m5_persistant_array*OPENACC*          # M5 forked branch (121.0s)
    ├── *m5_persistant_array*VANILA*           # Vanilla (543.9s)
    ├── *m6_templates*OPENACC*                 # M6 T2 templates (141.0s) ⚠️ REGRESSION
    └── *m6_templates*VANILA*                  # Vanilla (538.1s)
```

## Profiling Data Location

```
results/2026_03_27_modelselection_profile/
└── DNA/
    ├── *.nsys-rep          # Nsight Systems timeline (14.5 MB)
    ├── *.sqlite            # Nsight Systems queryable database (27.3 MB)
    └── *.ncu-rep           # Nsight Compute per-kernel metrics (3.5 GB)
```

## Analysis Files

```
poc-gpu-likelihood-calculation_analysis/2026_03_27_modelselection_opt_after_prof/
├── README.md                              # This file
├── plan_after_profiling.md                # Full 7-step optimization plan
├── step1_eigen_flag_analysis.ipynb        # Jupyter notebook with plots (S1+S2)
├── step1_2_comparison.png                 # Bar chart: all stages comparison
├── step1_2_gpu_speedup_progress.png       # GPU speedup progression chart
├── step1_eigen_flag_comparison.png        # Bar chart: S1 only
└── step1_gpu_speedup_progress.png         # GPU speedup: S1 only
```

## Reverted Attempts (for reference)

### MS1: Persistent GPU Buffer Pool — REVERTED
Shared host memory across models to eliminate GPU alloc/dealloc cycles. Pool management overhead (bind/unbind, tip_states rebuild) exceeded the alloc savings. DNA: +1s gain; AA: -8s regression.

### MS4 Phase A/C: Batch Model Pool — REVERTED
Extended MS1 with a BatchModelPool class. Same fundamental issue — pool overhead > alloc savings for sequential single-model evaluation. DNA: -1.2s; AA: +4.6s regression vs MS6-only.

### MS4 Phase B: Multi-Model Batched Kernels — REVERTED
Added `computeBatchedLikelihoodMultiModel` for N-model simultaneous GPU evaluation. Compiled and ran but never exercised (pool still used 1 slot). Required major model loop restructuring to batch models — blocked by sequential checkpoint dependencies between models.

### MS5: GPU-Side Tip Derivative Tables — REVERTED
Computed tip lookup tables on GPU instead of host. Regressed because the TIP-INT path needs P(t) on host for tip table computation. GPU→host download of P(t) stalled the pipeline more than the host-side computation saved.
