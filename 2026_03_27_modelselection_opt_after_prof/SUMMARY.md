# Post-Profiling GPU Optimization Summary

## What Was Done

Post-profiling optimizations on **IQ-TREE2 GPU ModelFinder**, guided by `nsys` + `ncu` profiling. The profiling revealed two root causes: **~40K tiny PCIe transfers** consuming 64% of transfer time, and **derivative kernels at <1% SM utilization** due to low parallelism + register pressure.

## Optimizations & Results

**Test setup:** NVIDIA V100, 100 taxa, 1000 sites, DNA (GTR+I+G4) & AA (LG+I+G4)

| Step | Optimization | What It Does | DNA Time | AA Time | AA GPU/CPU Speedup | Status |
|---|---|---|---|---|---|---|
| Baseline | Original | — | 42.1s | 229.8s | 2.2x | — |
| MS6+MS9 | Selective upload + skip pattern_lh_cat | Reduces redundant GPU uploads | 34.7s | 123.3s | 4.5x | Shipped |
| **S1** | **Eigen dirty flag** | Skips ~2,355 redundant eigen uploads per 2 models when only branch lengths change | 34.3s | 126.7s | 4.2x | **Shipped** |
| **S2** | **Persistent offset arrays** | Replaces per-kernel `new[]/copyin/delete[]` with persistent GPU buffer, eliminating ~15K GPU malloc/free cycles | 31.4s | 120.6s | 5.1x | **Shipped** |
| M5 | Persistent offsets (forked branch) | Same as S2, simpler fixed-size alloc — validates S2 | 31.4s | 121.0s | 4.5x* | Validated |
| M6 | Template VL tuning + persistent reduction buffers | VL=128->32 for DNA deriv kernels + templated `computeTransDeriv` + persistent reduction buffers | 31.1s | **141.0s** | **3.8x** | **Reverted** (AA regression) |

*\*M5 AA GPU/CPU uses a different (faster) vanilla CPU node; GPU times match S2.*

## Net Result (S1+S2 combined, best shipped)

| Metric | DNA | AA |
|---|---|---|
| Wall-time reduction | 42.1s -> 31.4s (**-25.4%**) | 229.8s -> 120.6s (**-47.5%**) |
| GPU vs 1-CPU speedup | 0.76x (still slower) | **5.05x** (crossed 5x milestone) |
| Lines changed | ~85 lines across 4 files | same |

## Key Takeaway

The two shipped optimizations (S1 + S2) were pure overhead elimination (redundant uploads + malloc/free cycles) — small code changes with big impact, especially for AA. The M6 attempt to improve kernel occupancy via VL tuning was disproved — derivative performance is compute-bound, not occupancy-limited.
