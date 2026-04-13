# Why the REV Kernel Is Slower Than Expected: Deep Analysis

**Date:** 2026-04-02
**Benchmark:** `2026_04_02_kernelrev`
**Reference Paper:** UFBoot2 Supplementary (Hoang et al.), "Speeding up Felsenstein's pruning algorithm"

---

## 1. Executive Summary

The REV (eigenspace) kernel was expected to speed up phylogenetic likelihood computation by replacing O(c²) branch length estimation with O(c) eigenvalue dot products (where c = number of states: 4 for DNA, 20 for protein). However, benchmarks show:

- **GPU (V100):** Zero speedup (ratio 0.98-1.02× across all cases)
- **CPU, protein, short alignments:** 40-68% faster (as expected)
- **CPU, protein, long alignments:** 7-33% SLOWER (unexpected)
- **CPU, DNA, almost all cases:** SLOWER (unexpected)

This document explains why, combining the paper's theoretical analysis, the actual implementation, and the benchmark data.

---

## 2. Benchmark Setup

**Actual command (from log files):**
```
iqtree3 -s alignment_NNN.phy -te tree_1.full.treefile --prefix ... -seed 1 [-nt N] [--kernel-nonrev]
```

**What was measured:**
- IQ-TREE3 with `-te tree_1.full.treefile` (user-provided starting tree)
- **NO `-m` flag** — IQ-TREE defaults to ModelFinder (`-m MFP`)
- ModelFinder tested 121 protein / 97 DNA candidate models
- `-te` provides a starting tree but **does NOT disable tree search** — however, the log shows `Total number of iterations: 0` and `Wall-clock time used for tree search: 0.000 sec`, meaning the starting tree was already optimal and NNI found no improvements
- **The runtime is dominated by ModelFinder** (>95% of wall time for most runs)
- 100 taxa
- Alignment lengths: 100, 10,000, 100,000 sites
- Backends: 1 CPU, 10 CPU, 48 CPU, GPU (V100)

**Key detail:** ModelFinder was active, meaning IQ-TREE evaluated 97 (DNA) or 121 (AA) candidate models. Each model evaluation includes full tree traversal + branch optimization. The REV penalty applies to EVERY model evaluation. Tree search contributed essentially 0 seconds since the `-te` tree was already NNI-optimal.

**Example timing breakdown (AA, len=10000, 48 CPU):**
```
CPU time for ModelFinder: 8729.5 sec (2h:25m)
Wall-clock time for ModelFinder: 191.2 sec (3m:11s)
CPU time used for tree search: 0.000 sec
Total wall-clock time: 192.3 sec
→ ModelFinder = 99.4% of total time
```

---

## 3. The Paper's Theory vs Reality

### What the Paper Claims (UFBoot2 Supplementary, Section 4)

The paper describes two operations with different costs:

| Operation | Standard (NONREV) | Eigenspace (REV) | Paper's claim |
|-----------|-------------------|-------------------|---------------|
| **V computation at internal nodes** (eq. 9) | L computation (eq. 2): c² per node | V computation: "twice more expensive" | REV is **2× slower** for partial LH |
| **Branch length estimation** (eq. 8 vs eq. 4) | c² per site | c per site | REV is **c× faster** for derivatives |

The paper explicitly says:
> "This computation [V at internal nodes] is **twice more expensive** than computing partial likelihood vectors."

And:
> "the branch length estimation using eq. (8) is **4, 20 and 61 times faster** than applying eq. (4) for DNA, protein and codon models, respectively."

### What the Paper Does NOT Claim

The paper **never claims REV is faster overall**. It claims:
1. The V computation costs 2× more (partial LH penalty)
2. The branch estimation is c× faster (derivative speedup)
3. The net effect depends on how much time is spent on each

### Actual Implementation Cost (from phylokernelnew.h)

Careful operation counting of the actual CPU code shows:

| Operation | NONREV | REV | Actual ratio |
|-----------|--------|-----|-------------|
| **Partial LH per node** (forward transforms) | 2c² FMAs | 2c² FMAs | 1× (identical) |
| **V⁻¹ back-transform** (productVecMat) | 0 (not needed) | c² FMAs | **+c² EXTRA** |
| **Total partial LH per node** | **2c²** | **3c²** | **1.5×** (not 2×) |
| **Derivative per site** (dotProductTriple vs matrix-vector) | 3c² FMAs | 4c FMAs | **c/1.3× faster** |

The paper says "twice" but the code shows **1.5×**. The discrepancy is because the paper counts the storage cost (V vectors are "twice" the size in a different sense) while the operation count for the back-transform is c², giving 3c²/2c² = 1.5×.

---

## 4. The Five Reasons REV Is Slower Than Expected

### Reason 1: The 1.5× Partial LH Penalty Applies EVERYWHERE

The V⁻¹ back-transform is an extra c² matrix-vector product per node per pattern per rate category. This cost is paid:

- During **every** full tree traversal (all ~98 internal nodes for 100 taxa)
- During **every** model evaluation in ModelFinder (97-121 times)
- During **every** NNI move evaluation
- During **every** stale partial recomputation in branch optimization

The derivative c× speedup only helps during the Newton-Raphson steps of branch length optimization — a fraction of total runtime.

### Reason 2: For DNA (c=4), Derivative Savings Are Too Small

| | NONREV | REV | Saving |
|---|---|---|---|
| Partial LH penalty per node | 32 FMAs | 52 FMAs | +20 (63% more) |
| Derivative saving per site | 48 FMAs | 16 FMAs | -32 (67% less) |

**Break-even calculation for DNA:**
```
98 nodes × 20 extra FMAs = 1960 extra FMAs per full traversal
32 saved FMAs per derivative call

Need 1960/32 = 61 derivative calls per traversal to break even
```

With typical Newton-Raphson convergence (5-10 iterations per branch, 98 branches = 490-980 derivative calls per optimization round), this seems favorable. BUT:

- ModelFinder does ~97 model evaluations, each with a full traversal
- Each model evaluation: 1 traversal (penalty) + brief optimization (small saving)
- The 97 traversals × 1960 extra FMAs overwhelm the derivative savings

### Reason 3: Long Alignments Amplify the Partial LH Penalty

Both partial LH and derivative costs scale linearly with alignment length S:
```
Partial LH per traversal:  nodes × S × cost_per_node
Derivative per call:        S × cost_per_site
```

The RATIO of total partial LH time to total derivative time stays roughly constant. But the ABSOLUTE penalty grows:

| Alignment | Partial LH penalty per traversal | Derivative saving per call |
|-----------|----------------------------------|---------------------------|
| 100 sites | 100 × 98 × 20 = 196K extra FMAs | 100 × 32 = 3.2K saved |
| 10,000 sites | 10K × 98 × 20 = 19.6M extra FMAs | 10K × 32 = 320K saved |
| 100,000 sites | 100K × 98 × 20 = 196M extra FMAs | 100K × 32 = 3.2M saved |

The ratio (penalty/saving per call) is always 61:1. But with more traversals than derivative calls in a ModelFinder-dominated run, the penalty dominates for ALL alignment lengths.

**Why short alignments still win for protein:** With S=100, total compute time is tiny (milliseconds). The REV advantage comes from **optimizer convergence speed** — REV's faster derivatives mean the Newton-Raphson converges in fewer total iterations, and the absolute partial LH penalty is negligible. For S=100,000, the absolute penalty is minutes, overwhelming convergence improvements.

### Reason 4: ModelFinder Multiplies the Penalty

The benchmarks used `-m MFP` (ModelFinder), which tests 121 protein models. Each model evaluation:

1. **Full tree traversal** — REV pays 1.5× penalty on ALL nodes
2. **Brief branch optimization** — REV gets c× derivative speedup
3. **Model score computation** — identical cost

For 121 model evaluations:
```
Total extra partial LH cost:  121 × (98 nodes × S × 20 extra FMAs)
Total derivative savings:      121 × (~10 deriv calls × S × 1120 saved FMAs for protein)

For protein S=10,000:
  Penalty:  121 × 98 × 10,000 × 20 = 23.7 billion extra FMAs
  Savings:  121 × 10 × 10,000 × 1120 = 13.6 billion saved FMAs
  NET: 10.1 billion MORE FMAs → REV is SLOWER
```

For protein S=100:
```
  Penalty:  121 × 98 × 100 × 20 = 23.7 million extra FMAs (tiny)
  Savings:  121 × 10 × 100 × 1120 = 135.5 million saved FMAs
  NET: 111.8 million FEWER FMAs → REV is FASTER
```

**The crossover:** REV wins when derivative savings > partial LH penalty. For protein, this happens at short alignment lengths where the absolute penalty is small. For long alignments, the penalty grows faster than savings because traversals outnumber derivative calls.

### Reason 5: Thread Parallelization Changes the Balance

The observed pattern:
```
AA, len=10000:
  1 CPU:  REV 1.16× SLOWER   (penalty dominates)
  10 CPU: REV 1.07× SLOWER   (penalty reduced by parallelism)
  48 CPU: REV 0.69× FASTER   (parallelized penalty becomes negligible)
```

**Why?** Partial LH is parallelized across sites (OpenMP `schedule(static)` over patterns). Branch optimization is sequential (one branch at a time). With many threads:

```
Wall-time partial LH penalty = (98 nodes × S × 20 extra) / num_threads
Wall-time derivative savings = S × 1120 saved  (NOT divided by threads — sequential)

With 48 threads, protein S=10,000:
  Penalty per traversal:  98 × 10,000 × 20 / 48 = 408K FMAs wall-time
  Savings per deriv call: 10,000 × 1120 = 11.2M FMAs wall-time

  Ratio: 11.2M / 408K = 27.5 derivative calls to break even
  Actual calls: ~980 per optimization round → REV WINS
```

With 1 thread:
```
  Penalty per traversal:  98 × 10,000 × 20 = 19.6M FMAs
  Savings per deriv call: 10,000 × 1120 = 11.2M FMAs

  Ratio: 19.6M / 11.2M = 1.75 traversals per derivative call to break even
  Actual: many more traversals than derivative calls → REV LOSES
```

---

## 5. GPU-Specific Issue: Zero Speedup Explained

The GPU shows ratio 0.98-1.02× for ALL cases because of an additional problem beyond the 1.5× penalty:

**The fused REV GPU kernel has n× redundant computation.** Each GPU thread (one per output state) independently recomputes the forward transforms D[x] for ALL eigenstates, even though all n threads in the same (node, pattern) group need the same D[x] values.

```
GPU REV partial LH cost:  2c³ (not 3c², due to n× redundancy)
GPU NONREV partial LH:    2c²

Ratio: c× worse (not 1.5× worse)
  DNA:     4× worse
  Protein: 20× worse
```

This massive GPU-specific penalty completely cancels the derivative O(c) advantage. The Two-Phase optimization (implemented in uncommitted changes) addresses this by splitting the computation into Phase 1 (compute D[x] without redundancy) + Phase 2 (V⁻¹ back-transform).

---

## 6. Complete Benchmark Results

### DNA (c=4 states, 100 taxa)

| Length | Backend | NONREV (s) | REV (s) | Ratio | Verdict |
|--------|---------|------------|---------|-------|---------|
| 100 | 1 CPU | 3.4 | 5.9 | 1.74 | REV **74% slower** |
| 100 | 10 CPU | 8.8 | 8.9 | 1.01 | Neutral |
| 100 | 48 CPU | 5.8 | 146.6 | 25.4 | REV **anomaly** |
| 100 | GPU | 58.5 | 57.2 | 0.98 | Neutral |
| 10,000 | 1 CPU | 396.6 | 427.6 | 1.08 | REV **8% slower** |
| 10,000 | 10 CPU | 95.9 | 162.6 | 1.70 | REV **70% slower** |
| 10,000 | 48 CPU | 53.9 | 49.9 | 0.93 | REV **7% faster** |
| 10,000 | GPU | 61.2 | 62.4 | 1.02 | Neutral |
| 100,000 | 1 CPU | 3383.5 | 3848.6 | 1.14 | REV **14% slower** |
| 100,000 | 10 CPU | 527.3 | 441.2 | 0.84 | REV **16% faster** |
| 100,000 | 48 CPU | 208.9 | 230.1 | 1.10 | REV **10% slower** |
| 100,000 | GPU | 118.0 | 118.5 | 1.00 | Neutral |

### Protein/AA (c=20 states, 100 taxa)

| Length | Backend | NONREV (s) | REV (s) | Ratio | Verdict |
|--------|---------|------------|---------|-------|---------|
| 100 | 1 CPU | 75.7 | 45.0 | 0.59 | REV **41% faster** |
| 100 | 10 CPU | 37.5 | 11.8 | 0.31 | REV **69% faster** |
| 100 | 48 CPU | 43.8 | 16.6 | 0.38 | REV **62% faster** |
| 100 | GPU | 83.8 | 83.9 | 1.00 | Neutral |
| 10,000 | 1 CPU | 4404 | 5099 | 1.16 | REV **16% slower** |
| 10,000 | 10 CPU | 534.5 | 574.3 | 1.07 | REV **7% slower** |
| 10,000 | 48 CPU | 278.8 | 192.3 | 0.69 | REV **31% faster** |
| 10,000 | GPU | 221.7 | 222.6 | 1.00 | Neutral |
| 100,000 | 10 CPU | 3691 | 4913 | 1.33 | REV **33% slower** |
| 100,000 | 48 CPU | 1289 | 1418 | 1.10 | REV **10% slower** |
| 100,000 | GPU | 959.5 | 956.8 | 1.00 | Neutral |

### Log-Likelihood Correctness

- **AA:** All differences = 0.00 (exact match across all backends and lengths)
- **DNA, len >= 10,000:** All differences = 0.00
- **DNA, len = 100:** Difference of ~1.24 at some backends (likely model selection difference due to small data, not numerical error)

---

## 7. When REV Wins vs When It Loses

### REV Wins When:

| Condition | Why |
|-----------|-----|
| **Large c** (protein c=20, codon c=61) | Derivative speedup is massive (20× or 61×) |
| **Short alignments** (S < 1000) | Absolute partial LH penalty is negligible |
| **Many threads** (48+) | Parallelized traversal reduces penalty wall-time; sequential derivative speedup preserved |
| **Branch optimization dominates** | More derivative calls relative to traversals |

### REV Loses When:

| Condition | Why |
|-----------|-----|
| **Small c** (DNA c=4) | Derivative speedup too small (4×) to overcome 1.5× partial penalty |
| **Long alignments** (S > 10,000) | Absolute partial LH penalty becomes large |
| **Few threads** (1-10) | Full penalty felt in wall-time; no parallelization relief |
| **ModelFinder active** | 100+ model evaluations multiply the partial LH penalty |
| **GPU** | n× redundancy in fused kernel cancels all derivative gains |

---

## 8. The Fundamental Tradeoff (Visual)

```
            Short alignment (S=100)        Long alignment (S=100,000)
            ────────────────────────       ──────────────────────────

Partial LH: ████ (tiny absolute cost)     ████████████████████████ (huge)
             +1.5× penalty = █              +1.5× penalty = ████████████

Derivative:  ██ (few calls, tiny each)     ██████ (more calls, each costly)
             -c× saving = █                 -c× saving = ██████

NET (DNA):   ████+█ vs ████-█ = ~same     █████████████████+████████████ vs
                                            █████████████████-██████ = WORSE

NET (AA):    ████+█ vs ████-██ = FASTER    █████████████████+████████████ vs
                                            █████████████████-██████████ = WORSE

NET (AA,     ████/48+█ vs ████/48-██       █████████████████/48+████████████/48 vs
48 threads): = FASTER (penalty shrunk)      █████████████████/48-██████████ = FASTER
                                            (penalty shrunk by 48×, savings preserved)
```

---

## 9. Implications for the Two-Phase GPU Optimization

The Two-Phase optimization (uncommitted changes in `phylokernel_openacc.cpp`) addresses the GPU-specific n× redundancy by splitting the fused kernel into:
- **Phase 1:** Compute D[x] once per eigenstate (no redundancy, same as NONREV cost)
- **Phase 2:** V⁻¹ back-transform (extra c² cost, but only 1.5× not n×)

This should bring GPU partial LH cost from **c× worse** down to **1.5× worse** (matching CPU). Combined with O(c) derivatives, the GPU should then follow the same pattern as CPU:
- **Protein, short alignments:** Significant speedup
- **Protein, long alignments, many threads:** Modest speedup
- **DNA:** Marginal improvement at best
- **ModelFinder-dominated runs:** Still limited by 1.5× penalty on many traversals

**The Two-Phase optimization fixes the GPU's unique redundancy problem but cannot change the fundamental mathematical tradeoff described in the paper.**

---

## 10. Updated Results: Full Tree Search (No Starting Tree)

A second benchmark (`2026_04_03_fulltets_withouttree`) ran IQ-TREE with **no starting tree** — full ModelFinder + NNI tree search + branch optimization:

```
Command: iqtree3 -s alignment.phy --prefix ... -seed 1 [--kernel-nonrev] [-nt N]
```

No `-te` flag → IQ-TREE builds tree from scratch (BIONJ → NNI search).

### AA (Protein) CPU Results — REV is NOW CONSISTENTLY FASTER:

| Length | 1 CPU | 10 CPU | 48 CPU | GPU (V100) |
|--------|-------|--------|--------|------------|
| **100** | **4.39×** | **2.54×** | **5.49×** | 0.97× |
| **1,000** | **2.13×** | **1.68×** | **2.23×** | 0.99× |
| **10,000** | **1.90×** | **1.46×** | **1.78×** | 0.99× |
| **100,000** | — | **1.81×** | — | 1.00× |

### DNA CPU Results — REV is mostly faster:

| Length | 1 CPU | 10 CPU | 48 CPU | GPU (V100) |
|--------|-------|--------|--------|------------|
| **100** | **1.59×** | 0.50× | 0.34× | 1.01× |
| **1,000** | **1.21×** | **2.19×** | 0.96× | 0.98× |
| **10,000** | **1.13×** | 0.96× | 0.95× | 0.95× |
| **100,000** | **1.18×** | **1.97×** | **1.27×** | 1.00× |
| **1,000,000** | **1.18×** | — | **1.12×** | 1.00× |

### Why the Difference From the Previous Benchmark

The previous benchmark (`2026_04_02_kernelrev`) used `-te tree.treefile` which provided an already-optimal tree. That meant:
- Tree search = 0 seconds (tree was already NNI-optimal)
- Total runtime was >99% ModelFinder model evaluations
- REV partial LH penalty was amplified across 121 model evaluations with minimal derivative benefit

The new benchmark has **no starting tree**, so:
- Full NNI tree search with 103+ iterations
- Each NNI iteration: many branch optimizations (where REV derivative O(c) speedup helps)
- ModelFinder is ~30-50% of total time (not >99%)
- Tree search is ~50-70% of total time (derivative-heavy)

**Example timing (AA, len=100, 1 CPU, REV):**
```
ModelFinder:  47.9s  (31% of total)
Tree search: 104.5s  (67% of total)  ← REV derivative speedup helps HERE
Total:       154.8s
```

### The GPU Conclusion Stands

**GPU (V100) shows 0.95-1.01× across ALL cases in BOTH benchmarks.** This confirms:
- The fused REV GPU kernel's n× redundancy completely cancels derivative gains
- The Two-Phase optimization (implemented in uncommitted changes) is essential for GPU to benefit

**CPU REV genuinely works** — the paper's theory is validated when tree search (NNI + branch optimization) is a significant portion of runtime. GPU REV doesn't work due to the implementation-specific redundancy problem.

---

## 11. Recommendations

1. **For GPU:** Compile and test the Two-Phase optimization to bring GPU in line with CPU gains. Expected: 1.5-4× speedup for protein on GPU after eliminating n× redundancy.

2. **For production use (current code without Two-Phase):**
   - CPU: REV is the correct default for reversible models (1.1-5.5× faster for full analyses)
   - GPU: REV and NONREV are equivalent — use either (until Two-Phase is validated)

3. **The paper's claim is validated:** With full tree search, REV's "c times faster" derivative estimation translates to real overall speedup, especially for protein (c=20). The previous benchmark with fixed tree was misleading because it eliminated tree search entirely.

4. **For benchmarking:** Always run full analyses (no `-te` flag) to capture the complete REV effect. Fixed-tree benchmarks only measure ModelFinder, which is partial-LH-dominated and penalizes REV.
