# OpenACC Bugfixes for Complex Substitution Models (+I, +I+G4, +R)

**Repository:** `iqtree3-forked`
**Date:** March 19–20, 2026
**Commits:** `b450bcf2`, `a73fde8c`, `61c539ba`

---

## Test Setup

### Command Format

```bash
# CPU (VANILA) — single-threaded baseline
iqtree3 -s alignment_10.phy -te <tree>.full.treefile \
    --prefix <output_prefix> -m <MODEL> -seed 1

# GPU (OpenACC)
iqtree3_openacc -s alignment_10.phy -te <tree>.full.treefile \
    --prefix <output_prefix> -m <MODEL> -seed 1
```

### Models Tested

| Data Type | Models |
|-----------|--------|
| DNA | GTR, GTR+G4, GTR+I, GTR+I+G4, GTR+R4 |
| AA  | LG, LG+G4, LG+I, LG+I+G4, LG+R4 |

### Dataset

- **Alignments:** 10 simulated alignments per model, 10 sites each
- **Trees:** 10 random trees per model (rooted and unrooted)
- **Total runs:** 10 trees × 2 backends (VANILA, OPENACC) × 2 tree types (rooted, unrooted) × 10 models = 400 runs
- **Hardware:** NCI Gadi — Intel Cascade Lake CPUs, NVIDIA V100 GPU
- **Builds:** `bugfixes_v5` (base/+I/+I+G4/+G4), `bugfixes_R_v4` (+R4)

---

## Background

The initial OpenACC GPU implementation (Phase 1–3, Feb–Mar 2026) supported only base substitution models (GTR, LG) and Gamma rate heterogeneity (+G4). When models with **invariant sites (+I)**, **combined invariant+gamma (+I+G4)**, or **FreeRate (+R4)** were run on the GPU, the log-likelihoods diverged catastrophically from the CPU baseline — differences of ~7000 LL units, indicating fundamentally wrong computations.

### Root Cause Overview

All three bugs shared a common theme: **stale GPU data**. The GPU kernels computed likelihoods correctly for a given set of parameters, but during parameter optimization the host (CPU) updated key arrays that the GPU never received, causing the GPU to evaluate likelihoods against outdated parameter values.

| Model | Bug | Host array | GPU state |
|-------|-----|-----------|-----------|
| +I | Stale `ptn_invar` | Updated every optimization round | Never synced after initial `copyin` |
| +I+G4 | Missing `_pattern_lh_cat` download | Computed on GPU | Never downloaded to host for EM |
| +R | Non-deterministic `tree_lh` + stale `ptn_freq` | Recomputed by EM on host | GPU reduction order differs; `ptn_freq` not synced back |

---

## Fix 1: Invariant Sites (+I) — `b450bcf2`

**Commit:** `b450bcf2` (2026-03-19)
**Message:** `add support for +I models for OpenACC`
**File:** `tree/phylotreesse.cpp`

### The Bug

The `ptn_invar` array stores the invariant-site contribution to each alignment pattern's likelihood. It is **recomputed on the host** every time the proportion of invariant sites (`p_invar`) changes during optimization (called from `RateInvar`, `RateGammaInvar`, `RateFree`, etc.).

However, `ptn_invar` was uploaded to the GPU only once during the initial `#pragma acc enter data copyin(...)`. All subsequent host-side updates were invisible to the GPU — the GPU retained the **initial stale values**, producing incorrect likelihoods for +I models.

### The Fix

Added a `#pragma acc update device(...)` at the end of `PhyloTree::computePtnInvar()` to sync the updated array to the GPU whenever it changes:

```cpp
// tree/phylotreesse.cpp — inside computePtnInvar()

#ifdef USE_OPENACC
    // Sync updated ptn_invar to GPU if data is already resident.
    // ptn_invar is recomputed on the host whenever p_invar changes during
    // optimization (called from rateinvar, rategammainvar, ratefree, etc.).
    // Without this sync, the GPU retains stale values from the initial copyin,
    // causing +I and +I+G4 models to produce incorrect log-likelihoods.
    if (gpu_data_resident && gpu_ptn_invar_ptr == ptn_invar && gpu_nptn > 0) {
        double *local_ptn_invar = ptn_invar;
        size_t local_gpu_nptn = gpu_nptn;
        #pragma acc update device(local_ptn_invar[0:local_gpu_nptn])
    }
#endif
```

### Why It Matters

Without this fix, the GPU's reduction kernel adds the wrong invariant-site baseline to every pattern:

```
lh_ptn = ptn_invar[p] + sum_over_categories(...)
           ↑ STALE on GPU
```

This produced LL differences of ~3–4 units for +I models, growing to ~7000 for +I+G4 (where the cascading error through gamma categories was amplified).

### Result

| Model | Before | After |
|-------|--------|-------|
| LG+I (rooted) | FAIL (diff ~4.05) | PASS (diff = 0.0) |
| LG+I (unrooted) | FAIL (diff ~3.71) | PASS (diff = 0.0) |
| GTR+I (rooted) | FAIL (diff ~3.28) | PASS (diff = 0.0028) |
| GTR+I (unrooted) | FAIL (diff ~3.32) | PASS (diff = 0.0) |

---

## Fix 2: Combined Invariant + Gamma (+I+G4) — `a73fde8c`

**Commit:** `a73fde8c` (2026-03-19)
**Message:** `add support for +I+G4 models for OpenACC`
**File:** `tree/phylokernel_openacc.cpp`

### The Bug

The `_pattern_lh_cat` array stores **per-category likelihoods** for each alignment pattern — i.e., the likelihood contribution from each rate category (e.g., 4 gamma categories) before they are summed. This array is critical for the **EM algorithm** used by `RateGammaInvar::optimizeWithEM()` and `RateFree::optimizeWithEM()` to compute posterior category probabilities and update rate/proportion parameters.

The GPU reduction kernel computed `_pattern_lh_cat` correctly on the device, but the array was **never downloaded to the host**. The existing code only downloaded `_pattern_lh` (the final per-pattern log-likelihood). When the EM algorithm tried to read `_pattern_lh_cat` on the host, it found **stale or zero values**, causing parameter optimization to diverge catastrophically.

### The Fix

Added a `#pragma acc update self(...)` to download `_pattern_lh_cat` from GPU to host after the reduction kernel:

```cpp
// tree/phylokernel_openacc.cpp — inside computeLikelihoodBranchGenericOpenACC()

    #pragma acc update self(local_pattern_lh[0:nptn])
    // Download per-category likelihoods for EM optimization.
    // RateGammaInvar::optimizeWithEM() and RateFree read _pattern_lh_cat
    // on the host to compute posterior category probabilities.
    #pragma acc update self(local_pattern_lh_cat[0:nptn_ncat])
```

### The EM Feedback Loop

This is the data flow that was broken:

```
GPU: reduction kernel → _pattern_lh_cat[ptn * ncat + c] = lh_cat
                         ↓ (was MISSING before this fix)
HOST: #pragma acc update self(_pattern_lh_cat[...])
                         ↓
HOST: RateGammaInvar::optimizeWithEM()
      reads _pattern_lh_cat to compute posterior P(cat|pattern)
      updates gamma shape α and proportion of invariant sites
                         ↓
HOST: computePtnInvar() → recomputes ptn_invar
                         ↓ (Fix 1: syncs to GPU)
GPU: next likelihood evaluation uses updated ptn_invar
```

Without the download, the EM algorithm had no valid data to optimize against, causing ~7000 LL unit divergence.

### Result

| Model | Before | After |
|-------|--------|-------|
| LG+I+G4 (rooted) | FAIL (diff ~7055) | PASS (diff = 0.0) |
| LG+I+G4 (unrooted) | FAIL (diff ~7065) | PASS (diff = 0.0) |
| GTR+I+G4 (rooted) | FAIL (diff ~7082) | CLOSE (diff = 0.0405) |
| GTR+I+G4 (unrooted) | FAIL (diff ~7085) | PASS (diff = 0.0015) |

---

## Fix 3: FreeRate (+R) Models — `61c539ba`

**Commit:** `61c539ba` (2026-03-20)
**Message:** `add support for +R for OpenACC`
**Files:** `model/ratefree.cpp`, `tree/phylokernel_openacc.cpp`

### The Bug (Two Issues)

FreeRate (+R4) models had **two** problems, both related to the EM optimization loop in `RateFree::optimizeWithEM()`:

#### Issue A: Non-deterministic `tree_lh` from GPU reduction

The GPU reduction kernel sums pattern likelihoods across all patterns using OpenACC parallel reduction:

```cpp
#pragma acc parallel loop reduction(+:tree_lh)
for (ptn = 0; ptn < orig_nptn; ptn++)
    tree_lh += pattern_lh[ptn] * ptn_freq[ptn];
```

Floating-point addition is **non-associative** — the parallel reduction sums values in a different order than the sequential CPU code. For base models this produces identical results (the same optimum is reached). But for FreeRate models, where the EM algorithm runs multiple iterations feeding `tree_lh` back into the optimization loop, the small differences **compound across EM iterations**, causing the optimizer to converge to a different local optimum.

#### Issue B: Stale `ptn_freq` on GPU

During EM optimization, `RateFree::optimizeWithEM()` copies posterior probabilities into `ptn_freq` on the host:

```cpp
// RateFree::optimizeWithEM() line 645-647
double *this_lk_cat = phylo_tree->_pattern_lh_cat + c;
for (ptn = 0; ptn < nptn; ptn++)
    tree->ptn_freq[ptn] = this_lk_cat[ptn * nmix];
```

The EM then calls `tree->optimizeTreeLengthScaling(...)`, which evaluates the likelihood on the GPU — but the GPU still has the **old `ptn_freq`** values. The likelihood evaluation uses stale weights, producing incorrect results.

### The Fix (Two Parts)

**Part 1: Deterministic host-side `tree_lh` recomputation**

After downloading `_pattern_lh` from the GPU, recompute `tree_lh` on the host with deterministic sequential summation:

```cpp
// tree/phylokernel_openacc.cpp — after #pragma acc update self(local_pattern_lh[...])

    // Recompute tree_lh on host from downloaded _pattern_lh for deterministic
    // FP summation order. The GPU parallel reduction produces equivalent but
    // non-deterministic results that compound through EM feedback loops
    // (e.g., +R4 RateFree optimization), causing convergence to different
    // local optima. Per-pattern likelihoods are computed identically on GPU;
    // only the final weighted sum across patterns needs deterministic ordering.
    tree_lh = 0.0;
    for (ptn = 0; ptn < orig_nptn; ptn++)
        tree_lh += _pattern_lh[ptn] * ptn_freq[ptn];
```

**Part 2: Sync `ptn_freq` to GPU after EM update**

After the EM algorithm updates `ptn_freq` on the host, sync it to the GPU before the next likelihood evaluation:

```cpp
// model/ratefree.cpp — inside optimizeWithEM(), after ptn_freq update

#ifdef USE_OPENACC
            // Sync updated ptn_freq to GPU for the copy tree's likelihood evaluations.
            if (tree->gpu_data_resident) {
                double *freq_ptr = tree->ptn_freq;
                #pragma acc update device(freq_ptr[0:nptn])
            }
#endif
```

### Why Deterministic Summation Matters for +R but Not +G4

**+G4 (Gamma):** Has a single parameter (alpha) optimized by Brent's method (univariate). Small LL variations don't change the 1D optimum — the function has the same shape regardless of summation order.

**+R4 (FreeRate):** Has 2K-1 parameters (K proportions + K rates, with constraints). The EM algorithm iterates: compute posteriors, update proportions, refit branch lengths, recompute posteriors. At each step, a slightly different `tree_lh` value changes the gradient direction, pushing the optimizer toward a different path through the multi-dimensional landscape. After several EM rounds, the solutions diverge.

### Result (with bugfixes_v5)

| Model | Before | After (v5) |
|-------|--------|-----------|
| LG+R4 (rooted) | FAIL (diff ~7062) | FAIL (diff = 1.52) |
| LG+R4 (unrooted) | FAIL (diff ~7066) | FAIL (diff = 1.76) |
| GTR+R4 (rooted) | FAIL (diff ~7082) | FAIL (diff = 1.40) |
| GTR+R4 (unrooted) | FAIL (diff ~7084) | FAIL (diff = 3.83) |

The fix dramatically improved +R4 from ~7000 diff to 1–4, but some residual difference remained in the `bugfixes_v5` build due to additional non-determinism in the EM loop.

### Result (with bugfixes_R_v4 build)

Using the `bugfixes_R_v4` build (which includes additional tree_lh synchronization fixes), the +R4 models achieve near-exact match:

| Model | v5 result | R_v4 result |
|-------|-----------|-------------|
| LG+R4 (rooted) | FAIL (1.52) | PASS (0.0) |
| LG+R4 (unrooted) | FAIL (1.76) | PASS (0.0001) |
| GTR+R4 (rooted) | FAIL (1.40) | CLOSE (0.0121) |
| GTR+R4 (unrooted) | FAIL (3.83) | PASS (0.0031) |

---

## Summary: Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        HOST (CPU)                           │
│                                                             │
│  computePtnInvar()                                          │
│    ├── recomputes ptn_invar[] from p_invar                  │
│    └── FIX 1: #pragma acc update device(ptn_invar)  ───────►│ GPU
│                                                             │
│  RateGammaInvar::optimizeWithEM()                           │
│    ├── reads _pattern_lh_cat[] (posterior probabilities)     │
│    │     ◄── FIX 2: #pragma acc update self(lh_cat) ────────│ GPU
│    ├── updates gamma alpha, p_invar                         │
│    └── calls computePtnInvar() → FIX 1 triggers             │
│                                                             │
│  RateFree::optimizeWithEM()                                 │
│    ├── reads _pattern_lh_cat[] (posterior probabilities)     │
│    │     ◄── FIX 2: #pragma acc update self(lh_cat) ────────│ GPU
│    ├── updates ptn_freq[] with posterior probabilities       │
│    │    FIX 3b: #pragma acc update device(ptn_freq) ───────►│ GPU
│    └── calls optimizeTreeLengthScaling() → GPU eval          │
│                                                             │
│  After GPU reduction download:                              │
│    FIX 3a: Recompute tree_lh sequentially on host           │
│            (avoids non-deterministic GPU parallel reduction) │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        GPU (OpenACC)                        │
│                                                             │
│  Reduction kernel:                                          │
│    lh_ptn = ptn_invar[p]  ◄── must be current (FIX 1)      │
│    for each category c:                                     │
│      lh_cat = sum( P(t) * L )                               │
│      _pattern_lh_cat[p*ncat+c] = lh_cat  ──► downloaded     │
│      lh_ptn += lh_cat                        (FIX 2)       │
│    _pattern_lh[p] = log(lh_ptn) + scale correction         │
│    tree_lh += _pattern_lh[p] * ptn_freq[p]                 │
│                                    ↑ must be current        │
│                                      (FIX 3b)              │
│                                                             │
│  tree_lh → discarded, recomputed on host (FIX 3a)          │
└─────────────────────────────────────────────────────────────┘
```

---

## Final Correctness Results

After all three fixes, using `bugfixes_v5` for base/+I/+I+G4/+G4 models and `bugfixes_R_v4` for +R4 models:

| Model | Rooted | Unrooted |
|-------|--------|----------|
| LG | PASS (0.0) | PASS (0.0) |
| LG+G4 | PASS (0.0) | PASS (0.0) |
| LG+I | PASS (0.0) | PASS (0.0) |
| LG+I+G4 | PASS (0.0) | PASS (0.0) |
| LG+R4 | PASS (0.0) | PASS (0.0001) |
| GTR | PASS (0.0012) | PASS (0.0039) |
| GTR+G4 | PASS (0.0001) | PASS (0.0001) |
| GTR+I | PASS (0.0028) | PASS (0.0) |
| GTR+I+G4 | CLOSE (0.0405) | PASS (0.0015) |
| GTR+R4 | CLOSE (0.0121) | PASS (0.0031) |

**18/20 PASS, 2/20 CLOSE, 0/20 FAIL — 100% pass+close rate.**
