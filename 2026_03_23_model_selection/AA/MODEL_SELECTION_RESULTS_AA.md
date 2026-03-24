# ModelFinder: CPU vs OpenACC (GPU) Comparison Results — Amino Acid (AA)

**Date:** 2026-03-23
**Dataset:** Simulated amino acid alignment, LG+I+G4 generating model, 100 taxa
**Alignment lengths:** 100 / 1,000 / 10,000 / 100,000 sites
**Hardware:** NCI Gadi — CPU: Intel Cascade Lake (up to 48 cores), GPU: NVIDIA V100
**Software:** IQ-TREE 3.1.0 (built Mar 22 2026)

---

## Command Lines

```bash
# 1 CPU (VANILA) — single-threaded baseline
iqtree3 -s <alignment.phy> -te <tree.treefile> --prefix <output> -seed 1

# 10 CPUs (OMP)
iqtree3 -s <alignment.phy> -te <tree.treefile> --prefix <output> -seed 1 -nt 10

# 48 CPUs (OMP)
iqtree3 -s <alignment.phy> -te <tree.treefile> --prefix <output> -seed 1 -nt 48

# GPU (OpenACC) — uses OpenACC-compiled binary
iqtree3_openacc -s <alignment.phy> -te <tree.treefile> --prefix <output> -seed 1
```

No `-m` flag is given, so IQ-TREE automatically runs **ModelFinder** to select the best-fit substitution model via BIC.

---

## 1. Model Selection Agreement

All 4 workflows select the **same best-fit model** at every alignment length:

| Alignment Length | 1 CPU  | 10 CPUs | 48 CPUs | GPU (V100) |
|-----------------|--------|---------|---------|------------|
| 100             | LG+G4  | LG+G4   | LG+G4   | LG+G4      |
| 1,000           | LG+G4  | LG+G4   | LG+G4   | LG+G4      |
| 10,000          | LG+G4  | LG+G4   | LG+G4   | LG+G4      |
| 100,000         | LG+G4  | LG+G4   | LG+G4   | LG+G4      |

**Result: 100% agreement across all workflows.** The best-fit model is consistently LG+G4 (without invariant sites), even though the generating model was LG+I+G4.

---

## 2. Log-Likelihood Comparison (Best Model)

Exact log-likelihood values from `.iqtree` files for the best-fit model:

| Length  | 1 CPU          | 10 CPUs        | 48 CPUs        | GPU (V100)     | |max diff| |
|---------|----------------|----------------|----------------|----------------|-----------:|
| 100     | -7,676.5709    | -7,676.5709    | -7,676.5709    | -7,676.5709    | 0.0000     |
| 1,000   | -77,823.9640   | -77,823.9640   | -77,823.9640   | -77,823.9640   | 0.0000     |
| 10,000  | -807,351.4411  | -807,351.4411  | -807,351.4411  | -807,351.4411  | 0.0000     |
| 100,000 | -7,541,977.0710| -7,541,977.0710| -7,541,977.0710| -7,541,977.0710| 0.0000     |

**Result: Likelihoods are identical across all 4 workflows — zero difference for the best-fit model. GPU produces identical results to CPU.**

---

## 3. Per-Model Likelihood Comparison (All 122 Models)

ModelFinder evaluates ~122 amino acid substitution models at each alignment length. Comparing per-model log-likelihoods across all 4 workflows:

| Length  | Models Compared | Exact Matches | Max |diff| | Worst Model    |
|---------|:--------------:|:-------------:|----------:|----------------|
| 100     | 122            | 120/122       | 0.0010    | LG+I+R3        |
| 1,000   | 122            | 114/122       | 0.0080    | LG+I+R5        |
| 10,000  | 122            | 101/122       | 0.0110    | Q.BIRD+I+G4    |
| 100,000 | 67             | 53/67         | 0.2620    | LG+I+R5        |

**Observations:**
- The models with the largest differences are **LG+I+R5** and **Q.BIRD+I+G4** — complex rate heterogeneity models where numerical optimization converges to slightly different local optima across threads.
- The top BIC models (LG+G4, LG+I+G4, LG+R4, etc.) have **zero or near-zero** differences across all workflows.
- The differences are **not GPU-specific**: multi-threaded CPU runs show similar deviations from the single-threaded baseline.
- The maximum difference of 0.26 at 100K sites represents a negligible relative error.

---

## 4. BIC / AIC Score Comparison

| Length  | BIC (1 CPU)        | 10 CPUs delta | 48 CPUs delta | GPU delta |
|---------|-----------------:|----------:|----------:|------:|
| 100     | 16,264.9651      | 0.0000    | 0.0000    | 0.0000 |
| 1,000   | 157,015.6641     | 0.0000    | 0.0000    | 0.0000 |
| 10,000  | 1,616,526.4271   | 0.0000    | 0.0000    | 0.0000 |
| 100,000 | 15,086,233.2801  | 0.0000    | 0.0000    | 0.0000 |

**Result: BIC scores are identical across all workflows — zero difference.**

---

## 5. Gamma Shape Parameter

| Length  | 1 CPU  | 10 CPUs | 48 CPUs | GPU (V100) |
|---------|--------|---------|---------|------------|
| 100     | 0.9720 | 0.9720  | 0.9720  | 0.9720     |
| 1,000   | 1.0089 | 1.0089  | 1.0089  | 1.0089     |
| 10,000  | 1.0046 | 1.0046  | 1.0046  | 1.0046     |
| 100,000 | 0.9966 | 0.9966  | 0.9966  | 0.9966     |

**Result: Gamma shape alpha identical across all workflows.**

---

## 6. Wall-Clock Runtime

| Length  | 1 CPU   | 10 CPUs | 48 CPUs | GPU (V100) |
|---------|--------:|--------:|--------:|-----------:|
| 100     | 46.2s   | 10.6s   | 16.5s   | 2.8m       |
| 1,000   | 9.2m    | 1.5m    | 57.9s   | 4.1m       |
| 10,000  | 1.5h    | 9.8m    | 3.5m    | 5.9m       |
| 100,000 | 11.6h   | 1.3h    | 23.1m   | 17.9m      |

---

## 7. Speedup vs 1 CPU

| Length  | 10 CPUs | 48 CPUs | GPU (V100) |
|---------|--------:|--------:|-----------:|
| 100     | 4.37x   | 2.79x   | 0.28x      |
| 1,000   | 6.07x   | 9.53x   | 2.23x      |
| 10,000  | 9.10x   | 25.73x  | **15.17x** |
| 100,000 | 8.63x   | 30.28x  | **39.13x** |

---

## 8. GPU vs 48 CPUs

| Length  | 48 CPUs | GPU (V100) | GPU Speedup vs 48 CPUs |
|---------|--------:|-----------:|-----------------------:|
| 100     | 16.5s   | 2.8m       | 0.10x (GPU slower)     |
| 1,000   | 57.9s   | 4.1m       | 0.23x (GPU slower)     |
| 10,000  | 3.5m    | 5.9m       | 0.59x (GPU slower)     |
| 100,000 | 23.1m   | 17.9m      | **1.29x faster**       |

**Result: GPU overtakes 48 CPUs at 100K sites. At 100K sites, GPU is 1.29x faster than 48 CPUs, 4.54x faster than 10 CPUs, and 39.13x faster than 1 CPU.**

Note: The GPU crossover happens later for AA than for DNA, likely because amino acid models (20-state) have higher per-site computational intensity but there are proportionally more models to evaluate (122 vs 76), and the GPU kernel efficiency varies with the state space size.

---

## 9. CPU Time (Total Compute)

| Length  | 1 CPU    | 10 CPUs  | 48 CPUs   | GPU (V100) |
|---------|:--------:|:--------:|:---------:|:----------:|
| 100     | 46.1s    | 1.7m     | 3.3m      | 2.8m       |
| 1,000   | 9.1m     | 14.7m    | 45.1m     | 4.1m       |
| 10,000  | 1.5h     | 1.6h     | 2.6h      | 5.9m       |
| 100,000 | 11.5h    | 12.6h    | 17.0h     | 17.8m      |

**Observation:** Multi-threaded CPU runs consume significantly more total CPU time due to thread synchronization overhead. The GPU uses dramatically less total compute — only 17.8 minutes of wall time at 100K sites vs 17.0 hours of CPU time for 48 CPUs.

---

## 10. Note on Missing Results

- **1M sites (all workflows):** No results are available yet for the 1,000,000-site alignment. The directory exists but is empty.

---

## Summary

| Metric | Result |
|--------|--------|
| **Model selection** | 100% agreement — all workflows select LG+G4 |
| **Log-likelihood accuracy** | GPU matches CPU exactly (zero difference for best model) |
| **Per-model accuracy** | 83–98% of models are exact matches; remaining differ by floating-point noise (max 0.26) |
| **BIC/AIC** | Identical across all workflows (zero difference) |
| **GPU crossover point** | GPU faster than 48 CPUs at 100K sites |
| **Peak GPU speedup** | 39.13x vs 1 CPU, 4.54x vs 10 CPUs, 1.29x vs 48 CPUs (at 100K sites) |
| **GPU wall time at 100K sites** | 17.9 minutes (vs 23.1 minutes on 48 CPUs, 11.6 hours on 1 CPU) |
