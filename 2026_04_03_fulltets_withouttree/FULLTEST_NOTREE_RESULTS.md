# Full Test Without Starting Tree: CPU vs GPU + Kernel Comparison (2026-04-03)

## Overview

Full IQ-TREE ModelFinder + tree reconstruction benchmark **without providing a starting tree**. IQ-TREE builds the tree from scratch via BIONJ + NNI search.

**Workflows:**
- **1 CPU** (VANILA) — single-threaded baseline
- **10 CPUs** (OMP_10) — 10 OpenMP threads
- **48 CPUs** (OMP_48) — 48 OpenMP threads (full node)
- **GPU (V100)** (OPENACC) — NVIDIA V100 via OpenACC

**Kernel variants:** Each workflow tested with both kernel-rev (default) and kernel-nonrev.

**Configuration**: 100 taxa, alignment lengths 100 to 1M sites, seed=1, no starting tree

## Data Completeness

| Data Type | len_100 | len_1K | len_10K | len_100K | len_1M |
|-----------|---------|--------|---------|----------|--------|
| **DNA** (GTR+I+G4) | 8/8 | 8/8 | 8/8 | 8/8 | 6/8 (VANILA missing both kernels) |
| **AA** (LG+I+G4) | 8/8 | 8/8 | 8/8 | 6/8 (VANILA missing) | 0/8 |

## Key Findings

### 1. Model Selection Agreement

**100% agreement across all workflows** at all alignment lengths for both DNA and AA (Kernel-Rev):
- DNA: F81+F+ASC+G4 (100 sites), F81+F+G4 (1K+ sites) selected consistently
- AA: LG+G4 selected consistently (all lengths)

### 2. Log-Likelihood Accuracy

Log-likelihoods match within floating-point precision across all workflows:
- DNA max |diff|: 0.0004 (at 100 sites), essentially zero at 10K+ sites
- AA: identical across workflows at all lengths

### 3. Wall-Clock Time (Kernel-Rev)

#### DNA (GTR+I+G4, 100 taxa, no starting tree)

| Sites | 1 CPU | 10 CPUs | 48 CPUs | GPU (V100) |
|------:|------:|--------:|--------:|-----------:|
| 100 | 21.3s | 1.8m | 4.6m | 8.0m |
| 1,000 | 1.4m | 47.6s | 4.3m | 4.9m |
| 10,000 | 15.0m | 3.9m | 4.9m | 5.6m |
| 100,000 | 2.6h | 23.8m | 16.1m | 16.4m |
| 1,000,000 | N/A | 8.1h | 4.7h | 2.7h |

#### AA (LG+I+G4, 100 taxa, no starting tree)

| Sites | 1 CPU | 10 CPUs | 48 CPUs | GPU (V100) |
|------:|------:|--------:|--------:|-----------:|
| 100 | 2.6m | 3.0m | 1.4m | 9.5m |
| 1,000 | 20.1m | 5.4m | 4.4m | 8.3m |
| 10,000 | 3.2h | 26.2m | 11.0m | 17.1m |
| 100,000 | N/A | 3.4h | 1.3h | 1.6h |

### 4. GPU Speedup vs 48 CPUs (Kernel-Rev)

| Sites | DNA | AA |
|------:|----:|---:|
| 100 | 0.58x (slower) | 0.15x (slower) |
| 1,000 | 0.87x (slower) | 0.52x (slower) |
| 10,000 | 0.88x (slower) | 0.64x (slower) |
| 100,000 | 0.98x (~equal) | 0.81x (slower) |
| 1,000,000 | 1.77x faster | N/A |

### 5. GPU Speedup vs 1 CPU (Kernel-Rev)

| Sites | DNA | AA |
|------:|----:|---:|
| 100 | 0.04x (slower) | 0.27x (slower) |
| 1,000 | 0.28x (slower) | 2.42x faster |
| 10,000 | 2.67x faster | 11.1x faster |
| 100,000 | 9.43x faster | N/A |

### 6. GPU Crossover Point (No Starting Tree)

- **DNA**: GPU becomes faster than 48 CPUs at ~1M sites (much later than with starting tree)
- **AA**: GPU does NOT surpass 48 CPUs at any tested length (up to 100K)
- **DNA**: GPU becomes faster than 1 CPU at ~10K sites
- **AA**: GPU becomes faster than 1 CPU at ~1K sites

### 7. Kernel-Rev vs Kernel-NonRev Comparison

#### DNA: Rev/NonRev Wall-Clock Ratio (ratio < 1 = Rev faster)

| Sites | 1 CPU | 10 CPUs | 48 CPUs | GPU (V100) |
|------:|------:|--------:|--------:|-----------:|
| 100 | 0.63 | 2.02 | 2.94 | 0.99 |
| 1,000 | 0.83 | 0.46 | 1.04 | 1.02 |
| 10,000 | 0.88 | 0.72 | 1.01 | 1.06 |
| 100,000 | 0.85 | 0.51 | 0.79 | 1.00 |
| 1,000,000 | N/A | 0.80 | 0.90 | 1.00 |

**DNA Summary:** Kernel-Rev is faster on 1 CPU (15-37% faster) and 10 CPUs (28-50% faster) at medium-large alignments. On GPU, both kernels perform identically (~1.0x ratio). On 48 CPUs, Rev is slightly slower at small sizes (100 sites: 2.94x slower) but faster at large sizes.

#### AA: Rev/NonRev Wall-Clock Ratio (ratio < 1 = Rev faster)

| Sites | 1 CPU | 10 CPUs | 48 CPUs | GPU (V100) |
|------:|------:|--------:|--------:|-----------:|
| 100 | 0.23 | 0.39 | 0.18 | 1.03 |
| 1,000 | 0.47 | 0.51 | 0.45 | 1.00 |
| 10,000 | 0.53 | 0.60 | 0.56 | 1.01 |
| 100,000 | N/A | 0.55 | 0.73 | 1.00 |

**AA Summary:** Kernel-Rev is dramatically faster across all CPU configurations (40-80% faster). On GPU, both kernels perform identically. The Rev advantage is strongest for protein data (c=20 states) on single-threaded execution.

## Observations

1. **No starting tree significantly changes the performance landscape.** Without a starting tree, IQ-TREE must build the tree from scratch, adding substantial overhead from BIONJ construction and extensive NNI search. This makes GPU crossover points much later compared to the with-tree tests.

2. **GPU advantage is diminished without starting tree.** GPU only wins at 1M DNA sites (1.77x vs 48 CPUs). For AA, GPU never surpasses 48 CPUs up to 100K sites. The tree-building phase is sequential and cannot benefit from GPU parallelism.

3. **Small alignments are dominated by overhead.** At 100 sites, GPU takes 8 minutes for DNA vs 21 seconds for 1 CPU — a 22x slowdown. Similarly 48 CPUs takes 4.6 minutes — thread synchronization overhead completely dominates.

4. **Kernel-Rev advantage is more pronounced for AA (c=20).** The O(c) derivative speedup from the eigenspace kernel matters more with 20 amino acid states than 4 DNA states. AA sees 40-80% speedups on CPU, while DNA sees 15-50%.

5. **GPU neutralizes kernel differences.** On GPU, kernel-rev and kernel-nonrev perform identically (ratio ~1.0x). The fused GPU kernel's redundant computation cancels the O(c) derivative advantage, consistent with the 2026-04-02 kernel-rev analysis.

6. **Thread contention at tiny data.** 48 CPUs at 100 DNA sites takes 4.6 minutes vs 21 seconds for 1 CPU — a 13x slowdown.

## Comparison with Prior Tests (2026-03-30, with starting tree)

| Aspect | With Tree (03-30) | Without Tree (04-03) |
|--------|-------------------|---------------------|
| GPU crossover (DNA, vs 48 CPU) | ~100K sites | ~1M sites |
| GPU crossover (AA, vs 48 CPU) | ~100K sites | Not reached (up to 100K) |
| GPU max speedup (DNA, vs 48 CPU) | 3.37x at 1M | 1.77x at 1M |
| DNA 100K: GPU vs 48 CPU | 1.77x faster | 0.98x (~equal) |
| AA 100K: GPU vs 48 CPU | 1.45x faster | 0.81x (slower) |

The tree-building phase adds sequential overhead that GPU cannot parallelize, pushing the crossover to larger alignment sizes.

## Files

- **Notebook**: `fulltest_notree_analysis.ipynb`
- **Results**: `/Users/u7826985/Projects/Nvidia/results/2026_04_03_fulltets_withouttree/`
- **Plots**: `*.png` files in this directory (21 plots generated by notebook)
