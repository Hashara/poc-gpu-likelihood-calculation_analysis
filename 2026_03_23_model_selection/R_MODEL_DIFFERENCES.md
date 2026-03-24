# Why +R (FreeRate) Models Show Larger lnL Differences Across Workflows

## Summary

During ModelFinder comparison of 4 workflows (1 CPU, 10 CPUs, 48 CPUs, GPU V100), only **FreeRate (+R) models** show non-zero log-likelihood differences across workflows. All other model classes (+G4, +I+G4, +F, etc.) produce identical results. This document explains why.

## Affected Models

| Model | len=1K | len=10K | len=100K | len=1M |
|-------|--------|---------|----------|--------|
| JC+I+R3 | 0.0100 | 0.0000 | 0.0000 | 0.0000 |
| JC+I+R4 | 0.0200 | 0.0000 | 0.0000 | 0.0000 |
| JC+I+R5 | 0.0060 | 0.0070 | 0.1460 | 0.9600 |
| JC+R5 | 0.0030 | 0.0410 | 0.0920 | 1.6000 |

Values shown are `|max lnL - min lnL|` across the 4 workflows.

Key observations:
- **Only JC+R/JC+I+R models** are affected (simple JC substitution model + FreeRate)
- Higher rate categories (+R5) have larger differences than lower ones (+R3)
- Differences grow with alignment length (1M sites: up to 1.6 lnL units)
- **1 CPU also differs from 10/48 CPUs** — this is NOT a GPU-specific issue
- All F81, HKY, TN, TIM, TVM, GTR, K2P models with +G4 or +I+G4 show **zero** difference

## Root Cause: Cascading Floating-Point Non-Determinism in the EM Algorithm

### Step 1: OpenMP Parallel Reduction is Non-Associative

The core likelihood computation in IQ-TREE uses OpenMP parallel reduction:

```cpp
// tree/phylotreesse.cpp:1353
#pragma omp parallel for reduction(+: tree_lh, prob_const) private(ptn, i, c) schedule(static)
for (ptn = 0; ptn < nptn; ptn++) {
    // ... compute per-pattern likelihood ...
    tree_lh += lh_ptn * ptn_freq[ptn];
}
```

Floating-point addition is **not associative**: `(a + b) + c != a + (b + c)`. With different thread counts, patterns are partitioned differently across threads, producing slightly different `tree_lh` sums due to different reduction order. The same applies to GPU parallel reductions on the V100.

### Step 2: +G4 (Gamma) Optimization — Single Parameter, No Cascade

The Gamma rate heterogeneity model (`model/rategamma.cpp`) optimizes a **single scalar parameter** (shape alpha) using 1-D line search (`minimizeOneDimen()`). Each likelihood evaluation may have tiny floating-point differences, but these affect only one optimization decision. The result converges to the same optimum regardless of thread count.

### Step 3: +R (FreeRate) Optimization — Iterative EM with Multiple Likelihood Calls

The FreeRate model (`model/ratefree.cpp:506-684`, function `optimizeWithEM()`) uses the **EM algorithm** (Wang, Li, Susko & Roger, 2008) to jointly optimize `2*ncat - 2` parameters (rates and proportions). This creates a cascade:

```
EM Iteration Loop (up to ncat iterations):
│
├── E-step: computePatternLhCat()
│   └── Calls computeLikelihood() → OpenMP reduction → tiny differences in _pattern_lh_cat
│
├── M-step: Accumulate posterior probabilities
│   └── new_prop[c] += this_lk_cat[c] * (ptn_freq / lk_ptn)
│   └── Slightly different _pattern_lh_cat → slightly different proportions
│
├── Rate optimization: For EACH rate category c:
│   └── optimizeTreeLengthScaling() → calls computeLikelihood() multiple times
│   └── Each call has OpenMP reduction → more tiny differences
│   └── Different scaling factor → different rate[c]
│
├── Convergence check: fabs(prop[c] - new_prop[c]) < 1e-4
│   └── May converge at different iterations across workflows
│
└── Next iteration uses updated rates/proportions from this iteration
    └── Amplifies differences from previous iteration
```

### Step 4: More Rate Categories = More Divergence

Each EM iteration optimizes **each rate category independently** (line 628-665 in `ratefree.cpp`). With +R5 (5 categories), each EM iteration performs 5 separate rate optimizations, each calling `computeLikelihood()` multiple times. Over multiple EM iterations, this produces:

- +R2: ~4 rate optimizations total → minimal cascade
- +R3: ~9 rate optimizations total → small cascade
- +R5: ~25 rate optimizations total → significant cascade

This explains why +R5 shows the largest differences and +R3 the smallest.

## Code References

| Component | File | Lines | Role |
|-----------|------|-------|------|
| OpenMP parallel reduction | `tree/phylotreesse.cpp` | 1353, 1397 | Source of FP non-determinism |
| FreeRate EM algorithm | `model/ratefree.cpp` | 506-684 | Cascading amplification |
| E-step (pattern lh) | `model/ratefree.cpp` | 544, 563-580 | Feeds different values per thread count |
| M-step (proportions) | `model/ratefree.cpp` | 582-624 | Accumulates from E-step differences |
| Rate optimization loop | `model/ratefree.cpp` | 626-665 | Multiple computeLikelihood() calls |
| GPU ptn_freq sync | `model/ratefree.cpp` | 649-655 | `#pragma acc update device` for OpenACC |
| Convergence check | `model/ratefree.cpp` | 602-608 | 1e-4 tolerance, may differ per workflow |
| Gamma optimization | `model/rategamma.cpp` | `optimizeParameters()` | Single-param, no cascade |

## Generalized IQ-TREE Commands Used

```bash
# 1 CPU (VANILA) — single-threaded, baseline
iqtree3 -s <alignment.phy> -te <tree.treefile> --prefix <output> -seed 1

# 10 CPUs (OMP)
iqtree3 -s <alignment.phy> -te <tree.treefile> --prefix <output> -seed 1 -nt 10

# 48 CPUs (OMP)
iqtree3 -s <alignment.phy> -te <tree.treefile> --prefix <output> -seed 1 -nt 48

# GPU (OpenACC) — uses nvhpc-compiled binary
iqtree3_openacc -s <alignment.phy> -te <tree.treefile> --prefix <output> -seed 1
```

No `-m` flag is specified, so ModelFinder runs automatically to test all candidate models.

## Conclusion

The lnL differences in +R models are **expected floating-point behavior**, not a bug:

1. The maximum difference (1.6 at 1M sites for JC+R5) represents a **0.0000027% relative error**
2. The differences occur **between CPU thread counts too** (1 CPU vs 10 CPUs vs 48 CPUs), not just CPU vs GPU
3. All workflows select the **same best-fit model** (F81+F+G4) at every alignment length
4. BIC scores are **identical** across all workflows for the best model
5. The affected models (JC+R5, JC+I+R5) are **not selected** as best models — they rank near the bottom of the BIC table

This is a well-known property of parallel floating-point reduction and is inherent to any parallelized implementation of iterative optimization algorithms like EM.
