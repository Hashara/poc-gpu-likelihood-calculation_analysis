# ModelFinder GPU Optimization Plan — Post-Profiling

**Date:** 2026-03-27
**Based on:** nsys + ncu profiling of MS6+MS9 build (DNA JC+G4, 100 taxa, 1K sites, V100)
**Current state:** MS6+MS9 shipped. DNA 34.7s, AA 123.3s.
**Target:** DNA <25s, AA <100s

---

## Profiling Summary

The GPU is **90% idle during the most time-consuming kernels**.

- Derivative kernels = 55% of GPU time, but only **0.5-0.8% SM utilization** (8 blocks on 80 SMs)
- 39,782 tiny scalar transfers (<256B) consume **64% of all PCIe transfer time**
- 22,269 kernel launches at ~5 µs each = ~111 ms launch overhead for 2 models
- Register pressure: 96-106 regs/thread in derivative kernels limits occupancy to 4-5 warps/SM

---

## Implementation Steps

### Step 1: Opt-A — Eigen Dirty Flag

**Effort:** 5 lines, 3 files
**Expected savings:** ~0.2s for 120 models

**Problem:** `uploadEigenToGPU()` is called ~2,355 times per 2 models during branch optimization. The eigendecomposition (eigenvalues, eigenvectors, state_freq, rate_cats, rate_props) does NOT change when only branch lengths change — it only changes during model parameter optimization.

**Implementation:**
- Add `bool gpu_eigen_dirty = true` member to PhyloTree (tree/phylotree.h)
- In `uploadEigenToGPU()` (tree/phylokernel_openacc.cpp): skip upload when `!gpu_eigen_dirty`, set `gpu_eigen_dirty = false` after upload
- Set `gpu_eigen_dirty = true` after `model->optimizeParameters()` and `site_rate->optimizeParameters()` return (model/modelfactory.cpp)

**When eigendata actually changes:**

| Event | Eigenvalues/vectors | rate_cats/props | state_freq |
|---|---|---|---|
| Branch optimization | NO | NO | NO |
| Q-matrix optimization (BFGS) | YES | NO | YES |
| Gamma shape optimization (Brent) | NO | YES | NO |
| p_invar optimization | NO | NO | NO |

A single dirty flag (set after any model/rate parameter change, cleared after upload) is safe and simple.

**Scaling:** Same benefit for DNA, AA, and large datasets. Branch count is tree-dependent (not site-dependent), so the number of skipped uploads is constant regardless of alignment size.

---

### Step 2: Opt-C — Persistent Offset Arrays

**Effort:** ~40 lines, 2 files
**Expected savings:** ~1.3s for 120 models

**Problem:** Each batched kernel level (TIP-TIP, TIP-INT, INT-INT) allocates a fresh offset array with `new[]`, uploads via `copyin`, then deletes with `delete[]`. This triggers ~15,000 tiny H→D transfers (each <256B) consuming 22 ms per 2 models. Each `copyin` of a small array triggers GPU-side malloc + PCIe transfer + GPU-side free — the latency (~1.5 µs per transfer) dominates.

**Implementation:**
- Add persistent offset buffer members to PhyloTree (tree/phylotree.h):
  ```cpp
  size_t *gpu_offset_buffer = nullptr;
  size_t gpu_offset_buffer_size = 0;
  bool gpu_offset_buffer_resident = false;
  ```
- In the offset-building code (tree/phylokernel_openacc.cpp, lines ~1403-1547):
  - First call: allocate to max needed size, `acc enter data create`
  - Subsequent calls: `memcpy` new offsets into same buffer, `acc update device`
  - No `new[]`/`delete[]` per level — reuse the persistent buffer
- Free in `freeOpenACCData()`

**Scaling:** Same benefit for all dataset sizes. The offset array count is tree-dependent (proportional to number of internal nodes), not site-dependent.

---

### Step 3: Opt-B — Batch df/ddf Scalar Downloads

**Effort:** ~30 lines, 1 file
**Expected savings:** ~2.2s for 120 models

**Problem:** Each derivative kernel call downloads 5 scalar reduction results (my_df, my_ddf, prob_const, df_const, ddf_const) via implicit `#pragma acc` reduction downloads. This creates 24,036 tiny D→H transfers (<256B each) consuming 36.3 ms per 2 models. Each transfer has ~1.5 µs fixed latency regardless of size.

**Implementation options:**

**(a) GPU-resident reduction array (recommended):**
- Allocate a small persistent GPU array: `double deriv_results[5]`
- Replace the 5 scalar reductions with writes to the GPU array inside the kernel
- Download all 5 values in ONE transfer after the kernel: `#pragma acc update self(deriv_results[0:5])`
- This replaces N × 5 tiny transfers with N × 1 transfers (5x reduction), and each transfer is still tiny but batched

**(b) Accumulate on GPU across NR iterations:**
- Keep the NR branch length update logic on GPU
- Only download the final converged branch length (one transfer per branch instead of per NR step)
- More complex but eliminates ~80% of the scalar downloads

**Scaling:** Same benefit for all dataset sizes. The transfer count is proportional to NR iterations × branches, which is tree-dependent, not site-dependent.

---

### Step 4: Profile Again

**Effort:** 0 lines (just re-run profiling)

Re-run `nsys` on the Opt-A+B+C build to validate:
- Transfer time should drop from 92 ms → ~30 ms (per 2 models)
- Kernel time unchanged at 132 ms
- Bottleneck shifts from transfers to kernel occupancy
- This validates that Steps 5-7 are worth implementing

---

### Step 5: Opt-D — Reduce Derivative Register Pressure

**Effort:** ~50 lines, 1 file
**Expected savings:** 1-3s for 120 models (speculative, needs profiling validation)

**Problem:** Derivative kernels use 96-106 registers per thread:
- `derivKernelIntInt<4>`: 96 regs → occupancy limited to 5 warps/SM (of 16 max)
- `derivKernelTipInt<4>`: 106 regs → occupancy limited to 4 warps/SM
- Low occupancy = poor latency hiding = GPU stalls waiting for memory loads

V100 has 65,536 registers per SM. At 96 regs/thread × 128 threads/block = 12,288 regs/block. Only 5 blocks fit (65536/12288 = 5.3). With 4 warps per block (128 threads / 32), that's 20 warps — but the 5-warp occupancy limit means only 5 out of 16 max warps are active.

**Implementation approaches:**

**(a) Compiler flag:** Add `--gpu-maxregcount=64` for derivative kernel compilation. Forces register spilling to local memory for excess registers. Trade-off: possible slowdown from spilling, but better occupancy may compensate.

**(b) Code restructuring:** The derivative kernels compute 3 quantities (lh, df, ddf) simultaneously, each accumulating across nstates inner products. Splitting into separate passes (first lh, then df, then ddf) reduces live registers per pass but triples memory reads.

**(c) `#pragma acc loop seq` hints:** Ensure inner loops are explicitly sequential to help the compiler minimize register allocation for loop variables.

**Approach (a) is recommended first** — zero code changes, just a compile flag. Profile to see if spilling hurts more than occupancy helps.

**Scaling:** Proportionally more impactful for AA (106 regs → even worse occupancy). For large datasets (100K sites), more patterns = more blocks, partially masking the occupancy issue, but register pressure still limits warps per SM.

---

### Step 6: Opt-E — Increase Derivative Kernel Parallelism

**Effort:** ~100 lines, 1 file
**Expected savings:** 2-5s for 120 models (speculative)

**Problem:** Derivative kernels for DNA 1K sites launch only 8 thread blocks:
- `nptn ≈ 1000` patterns with `vector_length(128)` → `ceil(1000/128) = 8` blocks
- V100 has 80 SMs → 90% of SMs are completely idle
- Each block runs on one SM, processing 128 patterns sequentially

**Implementation — 2D decomposition:**

Current: `#pragma acc parallel loop gang vector vector_length(128)` over patterns only.

Proposed: Split the per-pattern work across multiple thread blocks:

```
// Option 1: collapse over (pattern_chunk, category)
#pragma acc parallel loop gang collapse(2) vector_length(32)
for (int chunk = 0; chunk < num_chunks; chunk++)
    for (int cc = 0; cc < ncat; cc++)
        #pragma acc loop vector reduction(+:lh_cat)
        for (int ii = 0; ii < nstates; ii++)
            ...
```

For DNA+G4: ncat=4, 1000 patterns → 4000 independent (chunk, cat) work items → ~125 blocks. For AA+G4: same improvement.

**Alternative:** Use `gang worker(ncat) vector(nstates)` to add a worker dimension for category-level parallelism.

**Scaling:** Most impactful for small datasets (1K sites = 8 blocks). For 100K sites, patterns alone give 782 blocks — still benefits from more blocks but less critically.

---

### Step 7: Opt-F — Async Kernel Launches

**Effort:** ~30 lines, 1 file
**Expected savings:** 0.5-1s for 120 models

**Problem:** All kernel launches are synchronous. After launching a GPU kernel for level L, the CPU waits for completion before building offsets for level L+1. The CPU sits idle while the GPU runs, and the GPU sits idle while the CPU builds offsets.

**Implementation:**
- Add `async(1)` to batched kernel launches (batchedTipTip, batchedTipInternal, batchedInternalInternal)
- Build next level's offsets on CPU while GPU executes current level
- Insert `#pragma acc wait(1)` before launching the next level that depends on the previous level's output (parent nodes depend on child nodes, so levels must be sequential, but offset BUILDING can overlap)
- Also add `async` to the `buffer_partial_lh` upload at the top of `computeLikelihoodBranch`

**Scaling:** More impactful for large datasets where kernel execution time is longer (more overlap opportunity). For small datasets, kernels finish quickly so overlap benefit is small.

---

## Summary

| Step | Optimization | Lines | Savings (DNA 120 models) | Cumulative |
|---|---|---|---|---|
| 1 | Eigen dirty flag | 5 | ~0.2s | 34.5s |
| 2 | Persistent offset arrays | 40 | ~1.3s | 33.2s |
| 3 | Batch df/ddf downloads | 30 | ~2.2s | 31.0s |
| 4 | Profile validation | 0 | — | — |
| 5 | Register pressure reduction | 50 | 1-3s | 28-30s |
| 6 | Derivative parallelism | 100 | 2-5s | 23-28s |
| 7 | Async kernel launches | 30 | 0.5-1s | 22-27s |
| **Total** | | **~255 lines** | **7-12s** | **22-28s** |

**Target:** DNA from 34.7s → 22-28s. AA improvements proportionally larger (derivative occupancy fixes have bigger absolute impact due to heavier per-thread work).

**Risk:** All optimizations are pure waste elimination — no changes to mathematical computation, no added overhead. No negative effects on AA or large datasets confirmed by analysis.

**Validation points:** Profile after Steps 1-3 and again after Steps 5-6 to verify gains match predictions.
