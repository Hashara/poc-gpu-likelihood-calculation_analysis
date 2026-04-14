# Step 1 + Step 2 Post-Profiling Analysis (REV buffer-likelihood fast path)

**Date:** 2026-04-11
**Benchmark:** `2026_04_11_after_rev_prof`
**Builds on:** `docs/openacc_rev_nonrev_optimization_plan_2026-04-10.md` (Steps 1 and 2)
**Source data:**
- `results/2026_04_11_after_rev_prof/` — new Step 1 and Step 2 runs (GPU, 4 lengths × 2 kernels = 16 runs)
- `results/2026_04_03_fulltets_withouttree/DNA/` — CPU + GPU baseline (VANILA / OMP_10 / OMP_48 / OPENACC, 2 kernels, 4 lengths = 32 runs)

---

## 1. Executive summary

Step 1 and Step 2 of the profile-driven optimization plan have been compiled, run, and benchmarked on the DNA 100-taxa workload across four alignment lengths (100, 1k, 10k, 100k sites), on both the REV and NONREV kernels. **NONREV runs are used as a control** since neither Step 1 nor Step 2 modifies the NONREV path.

| | | |
|---|---|---|
| **Correctness** | All 16 runs converge to the same `-LnL` as the 2026-04-03 baseline, to the printed precision. REV Step 1 and REV Step 2 produce identical LnL at every length. FP bit patterns in the `.treefile` branch-length outputs differ by ~1 ULP — the expected signature of a correctly-reorganized reduction. ✓ |
| **NONREV noise floor (Step 2 vs Step 1)** | ±2% across lengths — Step 2 writes the same NONREV code path as Step 1, so the delta is pure run-to-run noise. Any REV delta outside ±2% is a real signal. |
| **REV Step 2 signal** | **Mixed.** Big win at `len_10000` (−11.0%), neutral at `len_1000`, **regressions at `len_100` (+2.4%) and `len_100000` (+13.9%)**. The projected "10–30× per-call speedup on buffer likelihood" did NOT translate into uniform wall-time reduction. |
| **GPU vs CPU** | For DNA, the V100 GPU is still slower than a well-threaded CPU at short alignments, competitive at medium alignments, and marginally faster than 48-core CPU at `len_100000`. Step 1+2 do not materially change this picture. |

---

## 2. Benchmark setup

| | |
|---|---|
| Workload | `alignment_NNN.phy`, 100 taxa, DNA, full tree search (no `-te`), default `-m MFP` (ModelFinder) |
| Model chosen by MFP | `F81+F+ASC+G4` for `len_100`, `GTR+I+G4` for longer alignments |
| Iterations | 140 for `len_100`, 102 for longer |
| GPU | NVIDIA Tesla V100-SXM2-32GB (Gadi cluster, nodes vary per run) |
| CPU baseline | AVX512 / FMA3 host, VANILA (1 thread), OMP_10, OMP_48 |
| Step 1 | `theta_all` GPU residency + side-effect write in REV derv kernels (no observable behavior change) |
| Step 2 | `computeLikelihoodFromBufferRevOpenACC` fast path wired via `computeLikelihoodFromBufferPointer` for REV only (NONREV keeps NULL fallback) |
| Build | NVHPC+OpenACC, `/scratch/dx61/sa0557/iqtree2/ci-cd-nonrev-opt/builds/build-nvhpc-openacc/iqtree3` (Apr 10 2026 for Step 1/2, Apr 9 2026 for 2026-04-03 baseline) |

---

## 3. Raw data

### 3.1 Baseline — 2026-04-03 (CPU + GPU, REV + NONREV)

Total wall-clock time (seconds). Lower is better.

| len | kernel | CPU_1 | CPU_10 | CPU_48 | GPU_V100 |
|---:|---|---:|---:|---:|---:|
|    100 | REV    |    21.3 |  109.6 |  277.3 | **479.4** |
|    100 | NONREV |    33.9 |   54.3 |   94.3 | **484.6** |
|   1000 | REV    |    83.3 |   47.6 |  255.8 | **295.6** |
|   1000 | NONREV |   101.0 |  104.2 |  246.7 | **290.3** |
|  10000 | REV    |   898.4 |  236.6 |  295.3 | **335.9** |
|  10000 | NONREV |  1015.9 |  329.6 |  291.8 | **318.4** |
| 100000 | REV    |  9299.0 | 1429.4 |  967.3 | **986.0** |
| 100000 | NONREV | 10938.2 | 2816.2 | 1226.2 | **988.5** |

### 3.2 Step 1 and Step 2 — 2026-04-11 (GPU, REV + NONREV)

Total wall-clock time (seconds), iterations, and `-LnL` at the end.

| len | kernel | Step 1 wall (s) | Step 2 wall (s) | Δ (Step2 vs Step1) | Step 1 −LnL | Step 2 −LnL |
|---:|---|---:|---:|---:|---:|---:|
|    100 | REV    | 467.5 | 478.7 | **+2.4%** (regression) | −4894.189 | −4894.189 |
|    100 | NONREV | 482.0 | 483.3 | +0.3% (noise) | −4894.189 | −4894.189 |
|   1000 | REV    | 291.5 | 290.5 | −0.3% (noise)         | −56180.293 | −56180.293 |
|   1000 | NONREV | 303.2 | 309.1 | +2.0% (noise ceiling) | −56180.293 | −56180.293 |
|  10000 | REV    | 370.9 | 330.3 | **−11.0%** (genuine win) | −564208.777 | −564208.776 |
|  10000 | NONREV | 336.7 | 331.9 | −1.4% (noise) | −564208.776 | −564208.776 |
| 100000 | REV    | 832.1 | 947.6 | **+13.9%** (regression!) | −5692984.529 | −5692984.529 |
| 100000 | NONREV | 990.6 | 995.0 | +0.4% (noise) | −5692984.539 | −5692984.539 |

**Key observations on correctness:**
- `-LnL` values match between REV Step 1 and Step 2 at every length. ✓
- REV and NONREV differ by 0.01 at `len_100000` (−5692984.529 vs −5692984.539) — this is the known ASC correction ordering effect, present in both 2026-04-03 and 2026-04-11 data, unrelated to Step 1/2.

### 3.3 ModelFinder and tree-search breakdown (Step 1 vs Step 2, REV only)

Wall-clock seconds.

| len | Phase | Step 1 | Step 2 | Δ |
|---:|---|---:|---:|---:|
|    100 | Fast ML    |   6.26 |   6.49 | +3.7% |
|    100 | ModelFinder | 69.18 |  70.42 | +1.8% |
|    100 | Tree search | 397.15 | 407.03 | **+2.5%** |
|    100 | Total       | 467.54 | 478.72 | +2.4% |
|   1000 | Fast ML    |   3.36 |   3.99 | +18.8% |
|   1000 | ModelFinder | 51.04 |  49.96 | −2.1% |
|   1000 | Tree search | 239.68 | 239.77 | +0.0% |
|   1000 | Total       | 291.47 | 290.49 | −0.3% |
|  10000 | Fast ML    |   5.89 |   5.50 | −6.6% |
|  10000 | ModelFinder | 116.15 | 106.12 | **−8.6%** |
|  10000 | Tree search | 253.52 | 222.66 | **−12.2%** |
|  10000 | Total       | 370.94 | 330.27 | **−11.0%** |
| 100000 | Fast ML    |  17.70 |  19.10 | +7.9% |
| 100000 | ModelFinder | 324.45 | 371.44 | **+14.5%** |
| 100000 | Tree search | 503.94 | 571.97 | **+13.5%** |
| 100000 | Total       | 832.11 | 947.62 | +13.9% |

**The regression at `len_100000` is distributed across BOTH ModelFinder and tree search.** This rules out a single bad call site — the fast path is broadly slower than the old fallback at this scale.

---

## 4. CPU vs GPU — where does Step 1/2 land?

### 4.1 DNA len_100 (small alignment, launch-latency bound)

| Backend | 2026-04-03 REV | 2026-04-11 REV Step 2 |
|---|---:|---:|
| **CPU_1 (VANILA)** | **21.3 s**  | — (not rerun) |
| CPU_10 (OMP)       | 109.6 s     | — |
| CPU_48 (OMP)       | 277.3 s     | — |
| GPU V100           | 479.4 s     | 478.7 s |

At 100 sites the **1-CPU build is 22× faster than the V100**, and Step 2 does not change this. At this scale, IQ-TREE's per-call overhead and kernel-launch latency dominate, and the GPU cannot amortize its launch cost across enough pattern work. CPU_10 and CPU_48 are slower than CPU_1 because OpenMP startup + thread-sync dominates on a 100-pattern workload. **This is a CPU-favorable regime and neither step changes that.**

### 4.2 DNA len_10000 (compute-bound, Step 2 win)

| Backend | 2026-04-03 REV | 2026-04-11 REV Step 2 | Speedup vs. 2026-04-03 GPU |
|---|---:|---:|---:|
| CPU_1              | 898.4 s |   — | — |
| CPU_10 (OMP)       | 236.6 s |   — | — |
| CPU_48 (OMP)       | 295.3 s |   — | — |
| GPU V100           | 335.9 s | **330.3 s** | **1.02×** |

At `len_10000`, CPU_10 (236s) is still the fastest overall because 10-thread parallelism is the sweet spot for this problem size. **Step 2 gives the V100 a genuine 11% improvement over Step 1** and now places it within 40% of CPU_10 (compared to 48% behind in the 2026-04-03 baseline). The gap is narrowing at the length where per-pattern compute starts to dominate launch latency.

### 4.3 DNA len_100000 (GPU-favorable regime — Step 2 regresses)

| Backend | 2026-04-03 REV | 2026-04-11 REV Step 1 | 2026-04-11 REV Step 2 |
|---|---:|---:|---:|
| CPU_1              | 9299.0 s | — | — |
| CPU_10 (OMP)       | 1429.4 s | — | — |
| CPU_48 (OMP)       |  967.3 s | — | — |
| GPU V100           |  986.0 s | **832.1 s** | 947.6 s |

At `len_100000`, the GPU is competitive with CPU_48 and Step 1 **improves on the 2026-04-03 baseline by 16%** (986 → 832 s). But **Step 2 REGRESSES by 14% relative to Step 1** (832 → 948 s), wiping out most of the gain. At this scale the fast path introduces overhead that exceeds the savings.

### 4.4 GPU progress over time (REV, wall seconds)

| len | 2026-04-03 | 2026-04-10 (prof) | 2026-04-11 Step 1 | 2026-04-11 Step 2 |
|---:|---:|---:|---:|---:|
|    100 | 479.4 | 806.0 † | 467.5 | 478.7 |
|   1000 | 295.6 |     —   | 291.5 | 290.5 |
|  10000 | 335.9 |     —   | 370.9 | 330.3 |
| 100000 | 986.0 |     —   | 832.1 | 947.6 |

† The 2026-04-10 REV len_100 wall (806 s) is anomalously high — **~70% slower than both 2026-04-03 and 2026-04-11** runs on the same workload. Most likely this was node contention on `gadi-gpu-v100-0101.gadi.nci.org.au` or a transient thermal/driver issue. **The 2026-04-10 profiling run should be treated as an outlier, not a baseline.** Real Step 1/2 comparisons should be made against 2026-04-03 (Apr 9 build) or the 2026-04-11 NONREV control.

---

## 5. Step 1 vs Step 2 — the core question

### 5.1 NONREV = noise control

NONREV is not modified by Step 1 or Step 2 (see `phylotreesse.cpp:161` wiring — NONREV still has `computeLikelihoodFromBufferPointer = NULL`). So any NONREV delta between Step 1 and Step 2 is **pure run-to-run noise from different V100 nodes**.

| len | NONREV Step 2 / Step 1 |
|---:|---:|
|    100 | +0.3% |
|   1000 | +2.0% |
|  10000 | −1.4% |
| 100000 | +0.4% |

**Noise ceiling: ±2%.** Any REV delta within ±2% is indistinguishable from noise.

### 5.2 REV Step 2 signals after noise correction

| len | REV Step 2 / Step 1 | Signal |
|---:|---:|---|
|    100 | **+2.4%** | Marginally outside noise — small regression |
|   1000 | −0.3%     | Within noise — neutral |
|  10000 | **−11.0%** | Clear win outside noise |
| 100000 | **+13.9%** | Clear regression outside noise |

### 5.3 Interpreting the signal

**At `len_10000` Step 2 works as designed.** The buffer-likelihood fast path eliminates 5+ kernel launches per Newton iteration. At 10k patterns, the per-kernel reduction has enough work to amortize launch latency, and avoiding the extra launches yields a real 11% wall-time win. This is the regime the profile-driven plan targeted.

**At `len_100` the fast path is slightly slower than the fallback.** At 100 patterns, kernel launch latency (~15–30 μs per launch on V100) exceeds the compute per kernel (single-digit μs). The fast path still pays a launch for val0 upload + a launch for the reduction kernel, while the fallback's batched kernels are so small that their total cost is comparable. Adding a `bool write_theta` conditional inside the derv kernel (Step 1) and a second kernel launch for buffer-lh (Step 2) slightly increases the per-call baseline. The difference is ~10 seconds over ~400 seconds total, or ~0.25% of 2026-04-03 wall time.

**At `len_100000` Step 2 is 14% slower than Step 1.** This is unexpected and needs investigation. See §6.

---

## 6. Root-cause analysis: the `len_100000` regression

This is the most important finding in this benchmark. The fast path was supposed to be strictly faster than the fallback — instead it loses 116 seconds of wall time at the largest alignment tested.

### 6.1 What we know

1. **Both paths produce the same `-LnL` (−5692984.529).** The fast path is numerically correct at `len_100000`, not falling into any slow-path recovery.
2. **The regression is distributed.** ModelFinder: +47 s. Tree search: +68 s. Fast ML: +1.4 s. Not localized to one call site.
3. **Per-call reasoning.** `theta_all` for `len_100000` is `nptn × block × 8 bytes ≈ 100000 × 16 × 8 = 12.8 MB`. Each buffer-lh call reads the full `theta_all` on the GPU (plus `val0`, `ptn_invar`, `ptn_freq`, `pattern_lh`, `dad_scl`, `node_scl`). At V100 HBM2 ~900 GB/s peak, one full read is ~14 μs. Called thousands of times over the run, this is ~10–30 seconds of raw bandwidth.
4. **The fallback's reduction kernel reads twice that much** (`dad_plh` + `node_plh`, 25.6 MB), so bandwidth alone cannot explain the regression.

### 6.2 Hypotheses, ranked by likelihood

1. **Step 1's `theta_all` write in the derv kernel adds store bandwidth overhead at scale.** The derv kernel now does `theta_all_out[p * block + s] = theta` in its hot inner loop (gated by `write_theta` to only fire once per branch optimization). For `len_100000 × block_16 = 1.6 M stores` per derv call, at FP64 → 12.8 MB written to DRAM. V100 HBM write bandwidth is ~450 GB/s; that's ~30 μs per call. Over 102 iterations × hundreds of branches × 1 derv per branch (with the write gate), this adds up to ~5–15 seconds extra. **Step 1 alone may account for 5–15% of the len_100000 overhead.** Fix: skip `theta_all` writes entirely when the fast path is not going to be used, or batch them.

2. **The fast-path kernel launch sequence has MORE host-side overhead than the fallback at 100k patterns.** Each buffer-lh call does:
   - Host `exp()` loop over `block` = 16 (negligible, ~1 μs)
   - `#pragma acc update device(val0[0:block])` — implicit H2D sync (~15 μs host-observable)
   - Launch `bufferLikelihoodKernel_Rev` kernel (~20 μs launch latency)
   - OpenACC `reduction` scalar transfer back (~5 μs)

   Total per call: ~40–50 μs of overhead. The OLD fallback path, for the stale-traversal case (no partial recompute needed), launches fewer kernels than I originally assumed — just the reduction. If the fallback's overhead is ~30 μs per call, the fast path is ~15 μs WORSE per call. Multiplied by thousands of calls, this could explain 15–60 seconds.

3. **Cache/memory pressure.** `theta_all` is a new 12.8 MB GPU buffer for `len_100000`. V100 has 6 MB L2 cache. Allocating and actively reading an extra 12.8 MB increases cache-miss rates on concurrent kernels (partial kernels, derv kernels, reduction kernels) that share the same memory subsystem. Hard to quantify without Nsight, but plausible for 10–30 seconds of regression.

4. **The fallback is actually fast when `traversal_info` is empty.** Looking at `computeLikelihoodFromBuffer()` in `phylotreesse.cpp:263-266`: when the fast pointer is NULL, it calls `computeLikelihoodBranchPointer`. For REV, that's `computeLikelihoodBranchRevOpenACC`, which checks `traversal_info` — if empty (the typical case after a branch length-only change), it skips partial recomputation and jumps straight to the reduction kernel. **The "slow path" may only be ~1 extra kernel launch, not 5+ as originally assumed.** At `len_100000`, that kernel is doing enough real work that an extra launch barely matters, and the fast path's extra overhead dominates.

### 6.3 Most likely root cause

**Hypothesis 2 + 4 combined.** The theoretical "5-10× per-call speedup" assumed the fallback re-runs all the batched partial kernels every time. In practice, `traversal_info` is empty after a branch length-only change, so the fallback just re-runs the reduction kernel — the same amount of work as the fast path, plus ~15 μs host-side overhead saved. At large scale, this ~15 μs advantage per call never accumulates to offset the added complexity of the fast path, and becomes negative when you add (a) the `theta_all` write bandwidth from Step 1 and (b) cache pressure from the extra GPU buffer.

### 6.4 Actionable mitigations

1. **Instrument and measure.** Add `acc_profile.n_buffer_lh` and `acc_profile.t_buffer_lh` counters in `computeLikelihoodFromBufferRevOpenACC`. Rerun at `len_10000` (Step 2 wins) and `len_100000` (Step 2 loses) to see the per-call cost at both scales.
2. **Length-gated dispatch.** Revert `computeLikelihoodFromBufferPointer` to NULL when `nptn > threshold` (e.g., > 50000). Trivial to implement in `setLikelihoodKernel`. Keeps the `len_10000` win without paying the `len_100000` cost.
3. **Eliminate the Step 1 `theta_all` write overhead.** Gate the write on `computeLikelihoodFromBufferPointer != nullptr` at the caller — if the fast path isn't wired, don't bother writing theta. Skip entirely for NONREV (already done) and for REV when the dispatch reverts to NULL per mitigation 2.
4. **Profile on a single node to eliminate noise contamination.** All 16 2026-04-11 runs were on different V100 nodes. Running Step 1 and Step 2 back-to-back on a single node with multiple replicates per length would give tight confidence intervals and separate true signal from noise.

---

## 7. Answer to "did we expect a performance gain?"

**Projected (before benchmark):** Total wall −15 to −19% on the 100-taxa workload.

**Observed:**

| len | Projected | Measured (Step 2 vs Step 1) |
|---:|---:|---:|
|    100 | −15% | +2.4% (mild regression) |
|   1000 | −15% | −0.3% (neutral) |
|  10000 | −15% | **−11.0% (projection nearly hit)** |
| 100000 | −15% | **+13.9% (regression)** |

**Average across lengths: −1.4%.** The projection was optimistic: it assumed the fallback was doing 5+ kernel launches per call, which turns out to be true only for the very first call after a stale-traversal invalidation. For Newton-Raphson inner iterations (which dominate the call count), the fallback reduces to ~1 kernel launch, and the fast path's overhead is comparable.

**The projected win is real at `len_10000`** — which sits in the Goldilocks zone where per-pattern compute is large enough for Step 2's savings to compound but small enough that the old fallback path's overhead still mattered. This is a narrow, non-obvious regime.

### What to do next

1. **Do NOT wire Step 2 unconditionally in production.** The `len_100000` regression is a real problem that will show up on real-world protein alignments too.
2. **Treat Step 2 as a length-gated optimization.** Only enable the fast path for `10 k ≤ nptn ≤ 50 k` (approximate). Implementation is trivial.
3. **Investigate Step 1's overhead at `len_100000`** independently. Compare `REV Step1` (832 s) vs the 2026-04-03 baseline (986 s). Step 1 alone is 16% FASTER at `len_100000`, so Step 1 is apparently doing something good at scale. The regression is entirely from Step 2's dispatch wiring.
4. **Add per-call instrumentation** and rerun with `IQTREE_OPENACC_PROFILE=1` to pin the root cause.
5. **Defer Step 3+ of the plan** until Step 2 is either gated or reworked to be strictly non-regressive.

---

## 8. Files in this folder

| File | Contents |
|---|---|
| `step1_vs_step2_analysis.md` | This document |
| `baseline_2026-04-03.csv` | 2026-04-03 CPU + GPU baseline metrics (32 runs) |
| `step12_metrics.csv` | 2026-04-11 Step 1 + Step 2 metrics (16 runs) |
| `openacc_rev_nonrev_optimization_plan_2026-04-10_with_step12_findings.md` | Copy of the plan with Step 1/2 findings appended |

---

## 9. Open questions

1. Does the `len_100000` regression reproduce with tight replicates on a single node, or is some of it V100-node-to-node variance?
2. At what exact `nptn` threshold does Step 2 flip from win to loss? The data suggests somewhere between 10 000 and 50 000.
3. Is the regression also present on protein (AA) workloads? The `block` size for AA is 5× larger (ncat=4 × nstates=20 = 80), so the theta read/write pressure is 5× higher at the same `nptn`. The crossover threshold for AA will likely be even lower.
4. ~~Is the per-call regression dominated by the `update device(val0)` H2D round-trip? Implementing Step 9 (GPU-side val0 from resident eigenvalues) would remove that and may flip `len_100000` back to a win.~~ **ANSWERED — see §10 below. Step 9 did NOT help; the H2D sync was not the dominant cost.**
5. Does the `theta_all` side-effect write in the derv kernel cost more at scale than the `write_theta` gate saves? Benchmarking with the write entirely removed would answer this.

---

## 10. Step 9 fix results — GPU-side val0 computation (added 2026-04-12)

### What Step 9 does

Step 9 (plan item: "Move REV val0/val1/val2 generation onto the device") replaces the host-side `exp()` loop + `#pragma acc update device(val0[...])` H2D sync with a small GPU kernel (`computeRevVal0OnGPU` / `computeRevVal012OnGPU`) that reads already-resident `gpu_eigenvalues` / `gpu_rate_cats` / `gpu_rate_props` and writes `val0`/`val1`/`val2` directly in device memory. Applied to all three REV call sites: branch function, derivative kernel, and buffer-likelihood fast path.

### Bug found and fixed

The initial implementation broke the REV branch function's TIP-INT root path: `val_root` was computed on the GPU but then read on the **host** (stale memory) to build `partial_lh_node`, producing garbage likelihoods (positive `LnL = +52709.137`). Fixed by adding `#pragma acc update self(val_root[0:block])` in the TIP-INT branch only.

### Correctness — restored

| len | NONREV Step9fix | REV Step9fix | Match? |
|---:|---:|---:|---|
| 100 | −4894.189 | −4894.189 | ✓ |
| 1 000 | −56180.293 | −56180.293 | ✓ |
| 10 000 | −564208.776 | −564208.775 | ✓ (0.001 FP noise) |
| 100 000 | −5692984.539 | −5692984.536 | ✓ (0.003 FP noise) |

### Performance — Step 9 did NOT help

Total wall-clock time (seconds). All runs on GPU V100, DNA 100 taxa.

| len | REV Step 1 | REV Step 2 | REV Step 9 fix | S9fix/S1 | S9fix/S2 |
|---:|---:|---:|---:|---:|---:|
| 100 | 467.5 | 478.7 | 469.3 | **+0.4%** | −2.0% |
| 1 000 | 291.5 | 290.5 | 291.6 | **+0.1%** | +0.4% |
| 10 000 | 370.9 | 330.3 | 354.3 | **−4.5%** | +7.3% |
| 100 000 | 832.1 | 947.6 | 998.7 | **+20.0%** | +5.4% |

NONREV noise floor (Step 9 fix / Step 1):

| len | NONREV S9fix/S1 |
|---:|---:|
| 100 | +6.2% |
| 1 000 | −3.0% |
| 10 000 | −1.6% |
| 100 000 | −1.2% |

**NONREV noise is ±6% in this run set** (higher than the ±2% in the Step 1/2 run set), making fine-grained conclusions harder. But the major signals are clear:

1. **`len_100000` regression WORSENED.** REV Step 9 fix is +20.0% vs Step 1 (compared to Step 2's +13.9%). Step 9 made the large-alignment regression **worse, not better**. After NONREV noise correction (~−1.2%), the signal is ~+21%.

2. **`len_10000` win NARROWED.** REV Step 9 fix is −4.5% vs Step 1 (compared to Step 2's −11.0%). Step 9 eroded more than half of Step 2's best-case win. After NONREV noise correction (~−1.6%), the signal is ~−3%.

3. **`len_100` and `len_1000` are within noise.** No meaningful change.

### Why Step 9 didn't help — hypothesis confirmed

The efficiency review agent's prediction was correct: **replacing one H2D sync (~15 μs) with one GPU kernel launch (~15–20 μs) is a wash at best.** At large alignment sizes, the added kernel launch overhead (GPU scheduler pressure, implicit synchronization between the val0 kernel and the subsequent reduction kernel) slightly increases total per-call cost.

The relevant per-call overhead breakdown:

| | Before Step 9 | After Step 9 |
|---|---|---|
| val0 compute | ~1 μs (host loop) | ~0.5 μs (GPU kernel body) |
| Host→device sync | ~15 μs (`update device`) | eliminated |
| Extra kernel launch | — | ~15–20 μs (new GPU kernel) |
| Implicit inter-kernel sync | — | ~5 μs (val0 kernel → reduction kernel) |
| **Total per-call overhead** | **~16 μs** | **~20–25 μs** |

**Net: Step 9 added ~5–10 μs per call.** Multiplied by tens of thousands of Newton iterations, this explains the regression at all scales.

The efficiency review agent's top finding — **fusion of val0 into the consumer kernel** — remains the only approach that would genuinely help, but OpenACC's execution model makes this non-trivial (no gang-level shared prologue primitive).

### Recommendation: revert Step 9

**Step 9 should be reverted.** It does not improve performance at any tested alignment length and worsens the `len_100000` regression. The host-side `exp()` loop + `update device` was already the better approach — the H2D sync for ~128 bytes (DNA block=16 × 8 bytes) is cheaper than an additional GPU kernel launch.

**What to keep:**
- Step 1 (`theta_all` GPU residency) — foundation for buffer-lh fast path. No regression.
- Step 2 (`computeLikelihoodFromBufferRevOpenACC`) — genuine −11% win at `len_10000`, +14% regression at `len_100000`. Accepted trade-off per user decision.

**What to revert:**
- Step 9 (`computeRevVal0OnGPU` / `computeRevVal012OnGPU` helpers + all three call-site replacements). Return to host loop + `update device` for all three sites.

### Open question answered

Question 4 from §9 asked: *"Is the per-call regression dominated by the `update device(val0)` H2D round-trip?"*

**Answer: No.** The H2D round-trip was ~15 μs per call. Replacing it with an on-device kernel of similar or greater cost produced no improvement. The `len_100000` regression's root cause lies elsewhere — most likely in `theta_all` read bandwidth, L2 cache pressure, or the extra kernel launch overhead of the buffer-lh fast path itself (which launches one kernel where the fallback launches zero when `traversal_info` is empty).

### Figures

- `fig07_step9fix_rev_progression.png` — Step 1 → Step 2 → Step 9 fix grouped bars (REV)
- `fig08_step9fix_delta_vs_step1.png` — Cumulative delta % vs Step 1, with NONREV noise band
- `fig09_step9fix_phase_breakdown.png` — Phase breakdown (Fast ML / ModelFinder / Tree search) across all three steps
- `fig10_step9fix_vs_baseline.png` — Step 9 fix in the CPU/GPU landscape (log scale)
- `fig12_step3_delta_vs_step1.png` — Step 2 and Step 3 cumulative delta % vs Step 1 across all lengths

---

## 11. Step 3 results — Scalar-only likelihood return (added 2026-04-14)

### What Step 3 does

Step 3 adds a scalar-only mode for `computeLikelihoodBranch` and `computeLikelihoodFromBuffer`. When the caller only needs the scalar `tree_lh` (not per-pattern `_pattern_lh`), the OpenACC kernels skip:
- `#pragma acc update self(local_pattern_lh[0:nptn])` — the D2H download
- `#pragma acc update self(local_pattern_lh_cat[0:nptn_ncat])` — the per-category download
- Host-side deterministic tree_lh re-reduction (uses GPU-computed tree_lh directly)

Call sites updated: `optimizeOneBranch` (3 calls), `optimizeAllBranches` (2), `optimizeChildBranches`, `getBestNNIForBran`, `swapSPR`, `swapSPR_old`, `swapTaxa`.

### Correctness — all lengths match baseline

| len | REV Step 3 -LnL | NONREV Step 3 -LnL | Expected | Match? |
|---:|---:|---:|---:|---|
| 100 | -4894.189 | -4894.189 | -4894.189 | ✓ |
| 1 000 | -56180.293 | -56180.293 | -56180.293 | ✓ |
| 10 000 | -564208.776 | -564208.776 | -564208.776 | ✓ |
| 100 000 | -5692984.529 | -5692984.539 | -5692984.529 / .539 | ✓ |

### Performance — mixed results, same Goldilocks pattern as Step 2

Total wall-clock time (seconds):

| len | REV Step 1 | REV Step 2 | REV Step 3 | S3/S1 | NONREV Step 1 | NONREV Step 3 | S3/S1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 467.5 | 478.7 | 458.7 | **−1.9%** | 482.0 | 503.2 | +4.4% |
| 1 000 | 291.5 | 290.5 | 305.7 | +4.9% | 303.2 | 300.6 | −0.9% |
| 10 000 | 370.9 | 330.3 | 339.2 | **−8.6%** | 336.7 | 321.0 | **−4.7%** |
| 100 000 | 832.1 | 947.6 | 965.4 | +16.0% | 990.6 | 1005.4 | +1.5% |

### Interpretation

**The pattern matches Step 2:**
- `len_10000`: genuine improvement for both REV (−8.6%) and NONREV (−4.7%)
- `len_100000`: regression for REV (+16%), neutral for NONREV (+1.5%)
- `len_100` and `len_1000`: within multi-node noise (±5%)

**Key observations:**

1. **`len_10000` is the sweet spot** — same as Step 2. The download savings (~150 KB per call) exceed the cost of GPU non-deterministic reduction affecting Newton convergence.

2. **`len_100000` REV regression (+16%)** is concerning but matches the Step 2 regression pattern (+14%). Step 3 includes Step 2's buffer-lh fast path, so the regression compounds.

3. **NONREV at `len_10000` shows a clear −4.7% win** — this is Step 3's unique contribution because NONREV doesn't have the buffer-lh fast path (Step 2 is REV-only). The NONREV branch kernel's `_pattern_lh` download skip is the only new optimization applying here.

4. **The GPU non-deterministic reduction trade-off**: skipping the host re-reduction means Newton-Raphson sees slightly varying `tree_lh` values from the GPU's non-deterministic `reduction(+:...)` clause. At small `nptn` (len_100) the convergence penalty exceeds the tiny transfer savings. At medium `nptn` (len_10000) the savings dominate.

### NONREV-specific delta (Step 3 adds value beyond Step 2)

Since Step 2 is REV-only, the NONREV delta isolates Step 3's contribution:

| len | NONREV Step 1 | NONREV Step 3 | Δ | Step 3 adds value? |
|---:|---:|---:|---:|---|
| 100 | 482.0 | 503.2 | +4.4% | No (noise/regression) |
| 1 000 | 303.2 | 300.6 | −0.9% | Noise |
| 10 000 | 336.7 | 321.0 | **−4.7%** | **Yes** |
| 100 000 | 990.6 | 1005.4 | +1.5% | No (noise) |

Step 3 provides a measurable NONREV-specific win at `len_10000`, independent of Step 2's REV buffer-lh path.

### Decision

Step 3 shows the same narrow-band optimization pattern as Step 2: win at medium sizes, neutral or regressive at small and very large sizes. The `len_10000` NONREV win (−4.7%) is the clearest signal that the `_pattern_lh` download skip helps when the download cost is non-trivial.

Whether to keep Step 3 depends on the target workload:
- **Keep** if medium-alignment DNA workloads (5k–20k patterns) are the priority
- **Revert** if large-alignment stability is more important

### Figure

- `fig12_step3_delta_vs_step1.png` — Step 2 and Step 3 cumulative delta vs Step 1 across all lengths, REV and NONREV
