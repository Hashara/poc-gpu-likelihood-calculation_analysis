# OpenACC REV/NONREV Optimization Plan — Profile-Driven

**Date:** 2026-04-10 (plan), 2026-04-11 (Step 1 + Step 2 findings appended)
**Profile source:** `/Users/u7826985/Projects/Nvidia/results/2026_04_10_rev_profiling`
**Step 1/2 benchmarks:** `/Users/u7826985/Projects/Nvidia/results/2026_04_11_after_rev_prof/`
**Builds on:** `docs/openacc_profile_driven_optimization_plan_2026-04-10.md` (codex-generated)
**Hardware:** Tesla V100-SXM2-32GB, AVX512 host (Gadi)
**Workload:** `alignment_NNN.phy` — 100 taxa, DNA; lengths 100 / 1 000 / 10 000 / 100 000 sites
**Binaries:** `iqtree3` (Apr 9 2026 baseline, Apr 10 2026 Step 1/2), NVHPC+OpenACC, single GPU thread

> **Status update 2026-04-11:** Sprint 1 (Steps 1+2) is implemented and benchmarked.
> Steps 1+2 are **REV-only** (NONREV has no branch-length-invariant theta).
> **Results mixed:** genuine 11% win at len_10000, **regressions at len_100 (+2.4%) and len_100000 (+13.9%).**
> See §9 below for full findings. **Recommendation:** gate Step 2 on `nptn` before shipping; do not enable unconditionally.

---

## 1. Profile Results — Measured Wall Times

Both runs converged to the same `-LnL = -4894.189`, best model `F81+F+ASC+G4`, 140 iterations, tree length 21.114 — so the comparison is apples-to-apples.

| Phase | NONREV (s) | REV (s) | Δ (REV − NONREV) | REV/NONREV |
|---|---:|---:|---:|---:|
| Fast ML tree search | 6.161 | 11.061 | **+4.900** | **1.80×** |
| First parameter optimization | 0.350 | 1.773 | **+1.423** | **5.07×** |
| ModelFinder (wall) | 103.798 | 116.133 | **+12.335** | 1.12× |
| 98 initial-tree eval | 18.519 | 16.507 | −2.012 | 0.89× |
| Tree search (wall) | 743.121 | 687.944 | **−55.177** | 0.93× |
| **Total wall** | **849.048** | **806.049** | **−42.999** | **0.95×** |

### 1.1 Reading the numbers

Three conclusions stand out:

1. **REV loses on partial-heavy startup work** (Fast ML, first param opt, ModelFinder). These phases drop to the branch-likelihood kernel every iteration — exactly where the REV partial kernels are doing redundant FP work.
2. **REV wins on derivative-heavy steady-state work** (tree search: 140 iterations of branch optimization with Newton-Raphson). REV's O(n) derivative (eigenvalue × partial) is genuinely cheaper than NONREV's full O(n²) P'(t) matrix multiply.
3. **The net win (43s total) is less than the individual wins could be** because both kernels pay for shared workflow overhead on every single call — overhead that does not depend on REV vs NONREV at all.

### 1.2 Nsight profiling state

The two `.qdstrm` traces (~8 GB each) are present but were **not** imported — the codex plan notes the machine ran out of scratch disk space. All observations below are sourced from:

- the run logs above
- direct reading of `tree/phylokernel_openacc.cpp` (5344 lines)
- `tree/phylotreesse.cpp` (dispatch layer)
- `tree/phylokernelnew.h` (reference CPU implementation of `computeLikelihoodFromBufferGenericSIMD`)

This is sufficient to pinpoint the structural bottlenecks with high confidence. A full Nsight import would refine the *ranking* of the fixes but is unlikely to change the *set*.

---

## 2. Root-Cause Analysis

### 2.1 Shared bottleneck (BIGGEST) — `computeLikelihoodFromBuffer` is missing on OpenACC

**File:** `tree/phylotreesse.cpp:161`

```cpp
// NONREV and REV OpenACC both end here:
computeLikelihoodFromBufferPointer = NULL;
```

**Impact path** — `tree/phylotreesse.cpp:259-268`:

```cpp
double PhyloTree::computeLikelihoodFromBuffer() {
    if (computeLikelihoodFromBufferPointer && optimize_by_newton)
        return (this->*computeLikelihoodFromBufferPointer)();
    else
        return (this->*computeLikelihoodBranchPointer)(current_it, ..., true);
}
```

**What this means in practice.** The CPU path (`computeLikelihoodFromBufferGenericSIMD` in `phylokernelnew.h:3307`) reuses a pre-computed `theta_all` array (the per-pattern product `left_partial_lh * right_partial_lh` from the previous branch likelihood pass). Newton iterations on a single branch then only need to:

1. Recompute `val0 = exp(eval × cat_rate × new_length) × prop` (a tiny `block`-element vector)
2. Fuse-multiply `theta_all[ptn, s] × val0[s]` and reduce per pattern

That is a small kernel with no traversal, no partial recompute, no scale propagation. The OpenACC path, by setting the pointer to `NULL`, falls back to the **full branch likelihood path** for every Newton iteration, which:

1. Rebuilds level batches via `groupByLevelAndType`
2. Repacks offsets on host
3. Re-runs every `batchedTipTip` / `batchedTipInternal` / `batchedInternalInternal` kernel
4. Downloads `_pattern_lh` to the host
5. Recomputes `tree_lh` in a serial host loop

Newton-Raphson calls this 3–5 times per branch optimization. `optimizeAllBranches` touches every branch. ModelFinder does this for ~96 model candidates. NNI scoring calls it at `tree/phylotree.cpp:4424` (`getBestNNIForBran`), `2702`/`2705`/`2735`/`2811` (`optimizeOneBranch` / `optimizeChildBranches` / `optimizeAllBranches`), and `4616`/`4824`. Conservatively, **90–95% of these calls are pure overhead** compared to the buffer path.

**Key observation:** `theta_all` is not referenced *once* in `tree/phylokernel_openacc.cpp` — the theta buffer is computed and used entirely on the CPU paths. On OpenACC, the "buffer" concept does not exist yet.

**Expected wall impact:** saving ≥50% on branch optimization work should shave **~10–15s off ModelFinder** and **~80–150s off tree search**. This is by far the biggest single improvement available.

### 2.2 Shared bottleneck — forced host materialization of `_pattern_lh`

**File:** `tree/phylokernel_openacc.cpp:2247-2279` (NONREV) and `:4763-4782` (REV).

Every branch-likelihood call ends with:

```cpp
#pragma acc update self(local_pattern_lh[0:nptn])
if (ncat > 1)
    #pragma acc update self(local_pattern_lh_cat[0:nptn_ncat])
// ... then:
tree_lh = 0.0;
for (ptn = 0; ptn < orig_nptn; ptn++)
    tree_lh += _pattern_lh[ptn] * ptn_freq[ptn];
```

The comment justifies this (deterministic FP order vs GPU parallel reduction). On a 100-pattern alignment that is cheap, but in production workloads with 10k+ patterns it becomes a real synchronization stall. More importantly, **most callers never read `_pattern_lh`** — they want `tree_lh` only. The only consumers that actually need host `_pattern_lh`/`_pattern_lh_cat` are:

- `computePatternLikelihood()` (pattern-level diagnostics, phylogenetic bootstrap)
- `RateGammaInvar::optimizeWithEM()` / `RateFree` (posterior category probabilities)
- Debug/trace paths

Branch optimization and NNI scoring only need the scalar. Even a host-side deterministic reduction of a GPU-resident `_pattern_lh` would still *issue* the H2D stall; but the copy is free if we skip it altogether when `ncat == 1` or the caller only wants `tree_lh`.

**Expected impact:** small-to-moderate; more impactful on large `nptn` than on the 100-pattern test.

### 2.3 REV-specific — Fused partial kernels recompute the forward transform per output state

**File:** `tree/phylokernel_openacc.cpp:557-663` (`batchedInternalInternal_Rev`)

The kernel output axis is `s = cat * nstates + k`, collapsed 3-way with `(op, ptn, s)`. Inside the body:

```cpp
for (int x = 0; x < NSTATES; x++) {
    double vleft = 0.0, vright = 0.0;
    for (int i = 0; i < NSTATES; i++) {
        vleft  += buffer_plh_base[eleft_off  + emat_cat_base + x*NSTATES + i]
                * central_plh_base[left_plh_off  + plh_base + i];
        vright += buffer_plh_base[eright_off + emat_cat_base + x*NSTATES + i]
                * central_plh_base[right_plh_off + plh_base + i];
    }
    val += inv_evec_base[k * NSTATES + x] * (vleft * vright);
}
```

Because `vleft[x]` and `vright[x]` **do not depend on `k`**, the inner two `i`-loops are recomputed `NSTATES` times per pattern — once for each output row `k` — even though all `NSTATES` rows share the same forward-transform vectors.

**FLOP accounting (per pattern, per category):**

| Design | Forward transforms | Back transform | Total |
|---|---:|---:|---:|
| Current fused | n × (2n²) = **2n³** | included | **2n³ + n²** |
| Two-phase: D[x] = vl[x]·vr[x]; out[k] = Σₓ V⁻¹[k,x] · D[x] | 2n² | n² | **3n² + n** |

| Model | n | Fused | Two-phase | Speedup |
|---|---:|---:|---:|---:|
| DNA | 4 | 144 | 52 | **2.77×** |
| AA | 20 | 16 400 | 1 220 | **13.4×** |
| Codon | 61 | 457 k | 11.3 k | **40×** |

The equivalent redundancy exists in `batchedTipInternal_Rev` (lines 674-779) and `batchedTipTip_Rev` (lines 787+).

This explains precisely why the REV fast ML phase is **1.8× slower than NONREV** on DNA — the REV kernel is doing ~2.8× more FLOPs per pattern while the NONREV kernel is doing a straightforward P(t) matvec. On AA the asymmetry would be much larger.

**Fix:** refactor each REV partial kernel to a two-phase design with a small staging buffer in `buffer_partial_lh` (already GPU-resident) sized `nptn × block` holding `D[op, p, cat, x]` between phases. The extra memory footprint is one additional `double[nptn × block]` per op batch — a few MB at most for DNA100.

**Expected impact:** DNA fast-ML partial should drop from 11.06s toward something close to the NONREV 6.16s (remaining gap will come from the first-call residency cost, item 2.5). Larger models benefit more.

### 2.4 REV-specific — Derivative `val0/val1/val2` rebuilt on host every call

**File:** `tree/phylokernel_openacc.cpp:5098-5137`

```cpp
double *eval = model->getEigenvalues();
for (size_t c = 0; c < ncat; c++) {
    double rate_c = site_rate->getRate(c);
    double prop   = site_rate->getProp(c);
    for (size_t i = 0; i < nstates; i++) {
        double cof = eval[i] * rate_c;
        double v0 = exp(cof * dad_branch->length) * prop;
        val0[c * nstates + i] = v0;
        val1[c * nstates + i] = cof * v0;
        val2[c * nstates + i] = cof * cof * v0;
    }
}
#pragma acc update device(val0[0:block], val1[0:block], val2[0:block]);
```

This arithmetic is small (`block` ≈ 16 for DNA/G4), but it happens on **every Newton iteration of every branch** during optimization. Each iteration pays an `exp()` evaluation per state, a host→device upload of 3×block doubles, and an implicit sync. The upload itself is a few hundred bytes but carries driver-call latency.

**Fix:** small device kernel reading already-resident `gpu_eigenvalues`, `gpu_site_rate`, `gpu_state_freq`, `dad_branch->length`; writes directly into resident `gpu_val0/1/2`. A single `acc parallel loop` over `c, i` with collapse(2).

**Expected impact:** small per call (μs range), but multiplied across thousands of Newton iterations — a noticeable slice of the tree-search phase. More importantly, it kills an unnecessary host-sync point.

### 2.5 REV-specific — First-call residency is expensive

**File:** `tree/phylokernel_openacc.cpp:4305-4317`:

```cpp
#pragma acc enter data \
    create(local_central_plh[0:total_lh_entries], ...)
// Full upload on first call:
#pragma acc update device(local_central_plh[0:total_lh_entries],
                          local_central_scl[0:total_scale_entries])
```

NONREV's equivalent is selective (`:1524-1535`): it only uploads the tip-vector slice (~608 bytes DNA / ~3.7 KB AA) and leaves the rest of the GPU buffer as `create` (allocate-only). The comment explicitly calls out the 3.2 GB DNA / 15.8 GB AA uploads that O1 eliminated for NONREV.

REV is currently reintroducing that full upload. The reason given in the code is that "host may have pre-computed partials from model selection / parsimony that GPU needs as kernel inputs". That concern is real, but it applies symmetrically to NONREV — NONREV solves it by re-computing stale partials via the first traversal pass. REV can do the same.

**Fix:** match the NONREV selective strategy; upload only `tip_offset:tip_alloc_size` and rely on the traversal to recompute internal partials from scratch.

**Expected impact:** REV's first-call residency upload is currently tens of milliseconds for DNA and hundreds of ms on real workloads. This is a one-time cost per tree-topology rebuild, which can happen many times during ModelFinder and NNI. Saving here directly attacks the ModelFinder overhead.

### 2.6 REV-specific — Host temporary `partial_lh_node` in TIP-root

**File:** `tree/phylokernel_openacc.cpp:4642-4696`

For TIP-INT root reduction the REV path allocates `double[(STATE_UNKNOWN+1)*block]` on the host each call, fills it using a host nested loop, passes it to `reductionKernelTipInt`, and deletes it. The host → device copy happens implicitly via OpenACC first-touch.

**Fix:** allocate once on GPU (persistent), compute on GPU directly from `val_root` and `tip_partial_lh` (both already resident), skip the delete altogether.

**Expected impact:** small per-call but removes a recurring allocation and implicit copy.

### 2.7 NONREV-specific — TIP-INT derivative still host-heavy

**File:** `tree/phylokernel_openacc.cpp:3780-3902`

The NONREV TIP-INT derivative path still:

1. Computes `P(t), P'(t), P''(t)` on host via `model->computeTransDerv()` (line 3794)
2. Applies `prop`, `prop_rate` scaling on host (lines 3798-3802)
3. For unrooted: multiplies by state_freq on host (lines 3804-3822)
4. Uploads three full matrices `trans_mat/derv1/derv2` to GPU (`:3824`)
5. Builds the three tip derivative tables on host via a 4-deep nested loop (lines 3877-3897)
6. Uploads the three tip tables (`:3900-3902`)

Steps 5 and 6 are the expensive ones — the tip-table loop is O(leafNum × ncat × nstates²) and runs fully single-threaded on the host. The memory comment (P1) explains that the kernel for tip table computation is retained in source but **not called** due to a `present()` sub-pointer bug. That bug should be worked around so the GPU path is live.

**Fix:** compute tip derivative tables directly on the GPU via the existing (but currently unused) `computeTipDerivTablesOnGPU()` helper. The P(t) matrices themselves are already uploaded, so the only change is to launch the tip-table kernel against GPU-resident inputs. Work around the `present()` sub-pointer lookup issue by passing base pointers + offsets instead of sub-pointers.

**Expected impact:** substantial for tree search. This is the reason NONREV trails REV in the 140-iteration tree search by 55s.

### 2.8 Shared overhead — `LevelBatch` grouping rebuilt per packet

**Files:** `tree/phylokernel_openacc.cpp:1730-1733` (NONREV branch), `:2015-2018` (NONREV derv), `:4434` (REV branch), `:4930+` (REV derv)

```cpp
for (int packet_id = 0; packet_id < num_packets; packet_id++) {
    ...
    int max_level = 0;
    vector<LevelBatch> level_batches = groupByLevelAndType(traversal_info, max_level);
    ...
}
```

`groupByLevelAndType` walks `traversal_info`, builds a `vector<LevelBatch>`, and allocates inside the per-packet loop. Traversal topology is identical across packets — only the pattern range changes. This is pure waste.

On a 100-pattern DNA workload with a single packet, the loss is ~zero. On multi-packet workloads and larger trees it becomes a measurable CPU stall.

**Fix:** hoist `groupByLevelAndType` above the packet loop. The `vector<LevelBatch>` can be stored as a member and reused across consecutive calls with the same traversal shape.

### 2.9 Shared overhead — Offset packing inside inner loops

Offset arrays (`offsets[bi*8 + 0..7]` or `offsets[bi*6 + 0..5]`) are built fresh on every dispatch, and the `#pragma acc update device(gpu_offsets[0:num_nodes*8])` uploads them every time. The offsets themselves don't change within a given traversal shape.

**Fix:** cache the offset layout per traversal shape and upload once. When `traversal_info` changes (new tree, new NNI), invalidate and rebuild. Six distinct packing/upload sites (NONREV: lines 1754-1791, 1815-1850, 1879-1907; REV branch: 4456-4492, 4521-4554, 4583-4611; REV derv: 4950-4979, 4999-5025, 5045-5066) can all share the same cache.

**Expected impact:** small per call, but every call currently pays for:

- CPU pointer arithmetic in a loop
- An H2D copy of ≤ `8 * num_nodes * 8` bytes
- An implicit launch barrier

Caching is a pure win and simplifies the code.

### 2.10 Shared overhead — Profile-mode `#pragma acc wait` is too granular

**Representative:** `tree/phylokernel_openacc.cpp:1650-1652`, `1786-1801`, `4407-4410`, `4486-4509` (and ~30 more sites).

Each batched kernel is followed by a `#pragma acc wait` when `profiling` is true. That is correct for measurement, but the same conditional structure uses a fine-grained wait in production builds too (many sites wrap `#pragma acc wait` without the `profiling` guard). Every wait stalls the host thread and kills any chance of kernel overlap.

**Fix:** keep precise waits behind `#ifdef USE_OPENACC_PROFILE`; use one wait per packet or per reduction boundary in production. Verify all non-profile waits are actually required for correctness (most are not — OpenACC pragmas already insert the sync they need).

---

## 3. Priority-Ordered Implementation Plan

Each step lists:
- **Scope** — files touched
- **Change** — what to do
- **Why** — which measurement it attacks
- **Expected impact** — rough wall-time delta
- **Validation** — how to check you haven't regressed correctness
- **Risk** — what could break

Unless otherwise noted, every step must preserve the exact `tree_lh` (0.0 FP diff vs current head) on both DNA and AA, with no underflow warnings.

### Phase A — Shared workflow fixes (highest impact, lowest risk)

#### Step 1 — Add `theta_all` residency to OpenACC [FOUNDATION]

**Scope:** `tree/phylokernel_openacc.cpp`, `tree/phylotree.h`

**Change:** Add `theta_all` to the persistent GPU data region. During the branch likelihood pass (both NONREV and REV), populate `theta_all[p*block + s]` from the product of `dad_partial_lh` and `node_partial_lh` (INT-INT) or from the tip lookup times partial (TIP-INT). Do this inside the existing reduction kernels so no extra traversal is needed. Set `theta_computed = true` at the end.

**Why:** Step 2 cannot work without this. The CPU version reads `theta_all` on every buffer-based likelihood call.

**Expected impact:** Zero on its own. Enables Step 2.

**Validation:** Existing likelihood values must not change. `theta_all` content must match the CPU SIMD path byte-for-byte (or within FP tolerance).

**Risk:** Low. Adding a write inside an existing kernel.

---

#### Step 2 — Implement `computeLikelihoodFromBufferOpenACC` [BIGGEST WIN]

**Scope:** `tree/phylokernel_openacc.cpp`, `tree/phylotreesse.cpp`, `tree/phylotree.h`

**Change:**

1. Add `double PhyloTree::computeLikelihoodFromBufferOpenACC()` that:
   - Computes `val0[c*nstates + i] = exp(eval[i] * cat_rate[c] * length) * prop[c]` on the GPU (one `#pragma acc parallel loop` over `c, i`, collapse(2)). Reuse persistent `gpu_val` buffer.
   - Launches a single reduction kernel over `theta_all` × `val0` that produces the scalar `tree_lh` directly on the host via `reduction(+:tree_lh)` OR writes `_pattern_lh` to a GPU slot and sums on host.
   - Handles `ASC_Lewis`, `ASC_Holder`, `prob_const`, and unobserved patterns exactly like `computeLikelihoodFromBufferGenericSIMD`.
2. Replace `computeLikelihoodFromBufferPointer = NULL` in `phylotreesse.cpp:161` with:
   ```cpp
   computeLikelihoodFromBufferPointer = &PhyloTree::computeLikelihoodFromBufferOpenACC;
   ```
   for BOTH REV and NONREV dispatch branches.

**Why:** Eliminates the full branch-likelihood fallback from every Newton iteration (`phylotreesse.cpp:263-266`). Directly attacks the biggest shared inefficiency.

**Expected impact:**
- **Tree search (140 iterations):** savings of ~150s per run (~20% of current 688–743s). Projected: both REV and NONREV drop to the ~540–590s range.
- **ModelFinder:** savings of ~10–15s (each model candidate does one final branch opt round).
- **Total wall:** drop from 806/849s to ~640/690s range.

**Validation:**
- Compare `computeLikelihoodFromBuffer()` output against `computeLikelihoodBranch()` on the same tree: must match within 1e-10.
- Full pipeline: same `-LnL`, same best model, same tree length.
- EM-driven models (`RateGammaInvar`, `RateFree`) still converge (they also call this path).

**Risk:** Medium. This is a new kernel and must handle all rate/ASC combinations correctly. Start with the non-ASC, non-mixture case (covers the DNA100 workload) and gate the rest behind a feature-complete fallback.

---

#### Step 3 — Scalar-only likelihood return mode

**Scope:** `tree/phylokernel_openacc.cpp` (branch likelihood functions for both REV and NONREV)

**Change:** Skip `#pragma acc update self(local_pattern_lh[0:nptn])` and the `ncat > 1` variant when the caller doesn't need per-pattern data. Introduce either:

- a function flag `need_pattern_lh` threaded through the call, or
- a separate fast-path entry point `computeLikelihoodBranchScalarOpenACC()`.

The host-side `tree_lh` sum then either runs on the GPU via a reduction or uses a GPU-side `_pattern_lh` staging buffer that is only copied when requested.

**Why:** Removes the H2D sync from branch likelihood calls that don't need per-pattern output. Most callers (branch opt, NNI, ModelFinder non-EM) don't need it.

**Expected impact:** Moderate. Most impactful on large `nptn`; on 100-pattern DNA the savings are in the single-digit ms per call range but multiply across thousands of calls. Better than nothing on the benchmark; critical on production-size workloads.

**Validation:** EM callers (`RateGammaInvar::optimizeWithEM`, `RateFree::optimizeWithEM`) must still see `_pattern_lh` / `_pattern_lh_cat` populated. `computePatternLikelihood()` must still work.

**Risk:** Low. Explicit opt-in/out.

---

#### Step 4 — Cache `LevelBatch` grouping per traversal

**Scope:** `tree/phylokernel_openacc.cpp`, `tree/phylotree.h`

**Change:** Move `groupByLevelAndType(traversal_info, max_level)` out of the packet loop. Cache the result in a member `vector<LevelBatch> cached_level_batches` with an invalidation flag tied to `traversal_info` identity. NONREV branch, NONREV derv, REV branch, REV derv all share this cache.

**Why:** Pure CPU-side waste; traversal topology does not change across packets of the same call.

**Expected impact:** Small on DNA100 (single packet). Meaningful on larger trees where `num_packets > 1`.

**Validation:** Identical kernel inputs before/after.

**Risk:** Low. Invalidation is the only tricky part — conservative approach: invalidate on every top-level call and only skip within the packet loop.

---

#### Step 5 — Cache packed traversal offsets

**Scope:** `tree/phylokernel_openacc.cpp`, `tree/phylotree.h`

**Change:** Build the `gpu_offsets` layout once per traversal shape and upload once. Invalidate on topology change (NNI, tree rebuild). The six packing sites (three per call, two calls: branch + derv) all hit the same cache.

**Why:** Removes one small H2D upload per kernel dispatch and all CPU packing loops from the hot path.

**Expected impact:** Small per call, meaningful in aggregate during tree search and ModelFinder.

**Validation:** Bit-exact offsets before/after.

**Risk:** Low.

---

#### Step 6 — Remove non-profile `#pragma acc wait` calls

**Scope:** `tree/phylokernel_openacc.cpp`

**Change:** Audit every `#pragma acc wait` not already wrapped in `#ifdef USE_OPENACC_PROFILE`. For each, determine whether OpenACC's implicit synchronization already covers it. Remove or push behind the profiling guard.

Target: one wait per packet OR per reduction, not per batch.

**Why:** Every wait stalls the host and kills kernel overlap.

**Expected impact:** Small-to-moderate. Most visible when kernels are small and launch overhead dominates (exactly our DNA100 case).

**Validation:** Same numeric result, same stale-partial behavior.

**Risk:** Medium. A missing wait can cause data races. Do this carefully with a test pass between every 5–10 removals.

---

### Phase B — REV-specific fixes

#### Step 7 — Match REV first-call residency to NONREV [EASY WIN]

**Scope:** `tree/phylokernel_openacc.cpp:4305-4317`

**Change:** Replace the full `update device(local_central_plh[0:total_lh_entries], local_central_scl[0:total_scale_entries])` with the selective NONREV-style upload: only `local_central_plh[tip_offset:tip_alloc_size]`.

**Why:** NONREV proved this works (O1 optimization). Nothing about REV's math requires the initial internal-partial upload — the first traversal pass recomputes everything.

**Expected impact:**
- **Fast ML / first param opt (REV):** the ~5× gap on first param opt (0.35s → 1.77s) has a non-trivial share coming from this full upload. Expect a 20-40% reduction in fast-ML time.
- **ModelFinder (REV):** each new model/tree topology re-triggers the residency cost. Expect 3-6s savings.

**Validation:** First-call likelihood matches previous run; no stale-partial errors on subsequent iterations.

**Risk:** Low, provided we verify there is no pre-computed host state we depend on.

---

#### Step 8 — Refactor REV partial kernels into two-phase design [BIGGEST REV WIN]

**Scope:** `tree/phylokernel_openacc.cpp:557-779+` (all three `batched*_Rev` kernels)

**Change:** Split each REV partial kernel into two phases:

**Phase 1:** `D[op, p, cat, x] = vleft[op, p, cat, x] * vright[op, p, cat, x]`, where `vleft/vright` are the same forward transforms already computed today. Written to a staging buffer in `buffer_partial_lh` (size `nptn × block`).

**Phase 2:** `out[op, p, cat, k] = Σₓ inv_evec[k, x] * D[op, p, cat, x]`. Written to `central_plh[dad_off + p*block + s]`.

Both phases use `collapse(3)` gangs. Phase 1 collapses `(op, p, x)` and runs the inner forward transform serially. Phase 2 collapses `(op, p, k)` and dot-products one row of `inv_evec` with the staging buffer.

**Why:** Eliminates the O(n) redundancy in the inner forward transform. See §2.3 for the FLOP math — 2.77× for DNA, 13.4× for AA, 40× for codon.

**Expected impact:**
- **Fast ML tree search (DNA):** REV 11.06s → ~6–7s (matches NONREV baseline; gap closes by ~4s)
- **First param opt (DNA):** REV 1.77s → ~0.5s
- **ModelFinder (REV):** 116.1s → ~105s
- **AA workloads (projected):** much larger wins (the 13× FLOP ratio translates to ~10× kernel time savings)

**Validation:**
- Byte-level comparison of `central_plh` after Phase 2 vs current fused output (within FP tolerance — the partial orderings differ).
- Full likelihood regression.

**Risk:** Medium-high. This is the most involved change. Staged rollout:
1. Prototype on `batchedInternalInternal_Rev` only, leave TIP-INT and TIP-TIP fused.
2. Compare kernel outputs and full likelihoods.
3. If clean, roll to `batchedTipInternal_Rev` (TIP child replaces one forward transform with a lookup), then `batchedTipTip_Rev`.
4. Keep the fused kernels in the source under an `#ifdef REV_PARTIAL_FUSED` fallback for at least one round of validation before deleting.

The staging buffer needs bounds checking against `buffer_plh_size`. Document the invariant clearly.

---

#### Step 9 — Move REV `val0/val1/val2` generation onto the device

**Scope:** `tree/phylokernel_openacc.cpp:5098-5137`

**Change:** Replace the host loop + `update device` with a single small `#pragma acc parallel loop collapse(2)` that reads resident `gpu_eigenvalues`, `site_rate`, model state, and `dad_branch->length`, and writes directly to `gpu_val0/1/2`.

**Why:** Removes a per-call host→device upload and an `exp()`-bound host loop.

**Expected impact:** Small per call, accumulates over thousands of Newton iterations. Also removes a sync point that currently blocks derivative kernel launch.

**Validation:** `val0/val1/val2` contents match current values within FP tolerance.

**Risk:** Low. Straightforward small kernel.

---

#### Step 10 — Move REV TIP-root `partial_lh_node` onto device

**Scope:** `tree/phylokernel_openacc.cpp:4642-4706`

**Change:** Replace host `new double[(STATE_UNKNOWN+1)*block]` + host nested loop + implicit copy with:

- Persistent GPU allocation sized `(STATE_UNKNOWN+1) * block * sizeof(double)` (allocated once, reused)
- A small `#pragma acc parallel loop` that reads resident `val_root` and `tip_partial_lh` and writes the device buffer

Then `reductionKernelTipInt` reads it via `present()`.

**Why:** Removes a recurring host allocation and an implicit H2D copy.

**Expected impact:** Small per call.

**Validation:** TIP-INT root likelihoods unchanged.

**Risk:** Low.

---

### Phase C — NONREV-specific fixes

#### Step 11 — Move NONREV TIP-INT derivative setup onto the GPU

**Scope:** `tree/phylokernel_openacc.cpp:3780-3902`

**Change:** Two sub-steps:

11a. **P(t) on GPU for TIP-INT:** Reuse the existing `computeTransDerivOnGPU()` (already implemented for INT-INT in the P0v2 optimization). TIP-INT currently deliberately keeps this on host because the tip-table builder needs host-side `trans_mat`. Once 11b is done, we can use the GPU path.

11b. **Tip derivative tables on GPU:** The code already has `computeTipDerivTablesOnGPU()` declared but NOT called due to a `present()` sub-pointer lookup bug (see memory `P1v2`). Work around it by passing `central_partial_lh` base + `tip_offset` index instead of the sub-pointer `tip_partial_lh`. Verify device-side indexing.

After both: the host-side loop at lines 3877-3897 is eliminated, and `update device(gpu_tip_derv_*)` is replaced with `present()`.

**Why:** This is the #1 reason NONREV trails REV in tree search. Every branch-optimization Newton iteration currently does a host-bound tip-table build.

**Expected impact:**
- **Tree search (NONREV):** 743s → target ~680s (match or beat REV). Saves ~55s.
- Combined with Step 2, NONREV tree search target ~530s.

**Validation:**
- Derivative values `df`, `ddf` must match within FP tolerance.
- Full pipeline likelihood must be unchanged.

**Risk:** Medium. The `present()` sub-pointer bug is a known OpenACC gotcha. If the base+offset workaround doesn't compile cleanly under nvc++, fall back to plan B: keep `trans_mat` host-computed, upload it, and offload only the tip-table build.

---

#### Step 12 — Unify NONREV TIP-INT / INT-INT transition pipeline

**Scope:** `tree/phylokernel_openacc.cpp`

**Change:** After Step 11 lands, fold the TIP-INT and INT-INT paths through a single `computeTransDerivOnGPU` call site. Remove the host-side `if (!rooted) { ... state_freq multiply }` code at lines 3804-3822 — the GPU kernel already knows how to apply state frequencies.

**Why:** Reduces branch-type asymmetry and removes duplicate host code.

**Expected impact:** Code quality win; minor speedup.

**Validation:** Same as Step 11.

**Risk:** Low after Step 11.

---

### Phase D — Diagnostics (do last)

#### Step 13 — Complete Nsight profile import

**Scope:** External workflow, no code.

**Change:** Free scratch disk on the target machine and import both `.qdstrm` traces using `nsys`. Produce a per-kernel timing summary to confirm Phase A+B+C predictions and identify any remaining hotspots.

**Why:** Validates the priority ordering and catches anything the log-based analysis missed.

**Expected impact:** None directly — this is verification.

**Risk:** None.

---

## 4. Priority Summary Table

| Rank | Step | Name | Target phase | Est. wall Δ (DNA100) | Risk |
|---:|---:|---|---|---:|---|
| 1 | 2 | `computeLikelihoodFromBuffer` on OpenACC | All | **−150s** (shared) | Medium |
| 2 | 8 | REV two-phase partial kernels | Fast ML, ModelFinder (REV) | −15s DNA; much more for AA | Medium-high |
| 3 | 11 | NONREV TIP-INT derivative on GPU | Tree search (NONREV) | **−55s** | Medium |
| 4 | 1 | `theta_all` residency | Enables Step 2 | 0 | Low |
| 5 | 7 | REV selective first-call residency | Fast ML, ModelFinder (REV) | −5s | Low |
| 6 | 3 | Scalar-only likelihood return | All | small-moderate | Low |
| 7 | 9 | REV `val0/val1/val2` on device | Tree search (REV) | small | Low |
| 8 | 4 | Cache `LevelBatch` grouping | All | small on DNA100 | Low |
| 9 | 5 | Cache traversal offsets | All | small | Low |
| 10 | 6 | Remove non-profile `acc wait` | All | small-moderate | Medium |
| 11 | 10 | REV TIP-root `partial_lh_node` on device | All (REV) | small | Low |
| 12 | 12 | Unify NONREV transition pipeline | Quality | small | Low |
| 13 | 13 | Full Nsight import | Verification | 0 | None |

**Best-case projection after Phase A+B+C:**
- Tree search wall: 743/688s → **~530s** for both (–28%)
- ModelFinder wall: 103.8/116.1s → **~90/100s** (−13–14%)
- Total wall: 849/806s → **~680/690s** (–19/15%)
- REV and NONREV should converge to nearly the same total — the remaining asymmetry is fundamental (O(n) vs O(n²) derivatives, different memory patterns).

---

## 5. Implementation Sequencing Recommendation

### Sprint 1 — Foundation (shared, high-impact)

**Goal:** Close the shared gap between OpenACC and CPU SIMD buffer path.

1. Step 1 — `theta_all` GPU residency
2. Step 2 — `computeLikelihoodFromBufferOpenACC`

**Exit criterion:** `computeLikelihoodFromBuffer()` returns the same value as `computeLikelihoodBranch()` to 1e-10 on DNA and AA; full pipeline `-LnL` unchanged. Tree-search wall time drops ≥15% on both REV and NONREV.

---

### Sprint 2 — REV fast-start fix

**Goal:** Close the REV startup gap.

3. Step 7 — REV selective first-call residency
4. Step 8 — REV two-phase partial kernels (start with INT-INT)

**Exit criterion:** REV fast ML tree search within 20% of NONREV (currently 1.8×).

---

### Sprint 3 — NONREV derivative fix

**Goal:** Close the NONREV tree-search gap.

5. Step 11 — NONREV TIP-INT derivative on GPU

**Exit criterion:** NONREV tree-search wall matches or beats REV on DNA100.

---

### Sprint 4 — Cleanup and small wins

6. Step 3 — Scalar-only likelihood mode
7. Step 4 — Cache `LevelBatch`
8. Step 5 — Cache offsets
9. Step 6 — Relax sync
10. Step 9 — REV `val` on device
11. Step 10 — REV TIP-root on device
12. Step 12 — NONREV unification

**Exit criterion:** Code cleanup complete; aggregate improvement ≥ 5% additional on both REV and NONREV.

---

### Sprint 5 — Verification

13. Step 13 — Full Nsight import and kernel profile

**Exit criterion:** Measured kernel times confirm the priority table projections. Any remaining outliers inform a next round.

---

## 6. Items Explicitly Deferred

Based on current evidence, these are NOT the best next changes:

- **Parallel NNI evaluation** — the per-NNI bottleneck is already the branch likelihood call, which Sprint 1 addresses. Parallel NNI would add complexity without first fixing the shared path.
- **Broad EM offload** — EM is called rarely relative to Newton iterations; the branch path fix is far higher ROI.
- **Mixed-precision likelihood** — no evidence the kernels are bandwidth-bound, and the numerical risk is real for phylogenetic inference.
- **Tree-search algorithmic changes** (better NNI ordering, different step sizes) — unrelated to what the profile shows.
- **Larger REV eigenspace rewrite beyond Step 8** — wait for post-Step-8 measurements before pursuing anything more invasive.

---

## 7. Validation Rig

Every step must pass:

1. **Unit-level** — likelihood/derivative values on a single branch match CPU SIMD reference within `1e-10`.
2. **Regression** — full DNA100 pipeline matches committed HEAD `-LnL = -4894.189`, best model `F81+F+ASC+G4`, 140 iterations, tree length 21.114, on both REV and NONREV.
3. **Memory** — no OpenACC `present()` errors; no double-free; no underflow warnings in the log.
4. **Performance** — wall-time delta against the 2026-04-10 baseline recorded in a comparison CSV; any regression > 2% gets investigated.

Suggested test matrix:
- `alignment_100.phy` — the 2026-04-10 baseline (DNA100)
- An AA alignment with `LG+F+G4` — verify the Step 8 projection holds for n=20
- A larger alignment (1000+ taxa) — verify that scalar-only path and sync relaxation (Steps 3, 6) help

---

## 8. Notes on the Codex Plan

The codex plan (`docs/openacc_profile_driven_optimization_plan_2026-04-10.md`) has the right conclusions. This document:

1. **Adds concrete profile numbers** from `*.log` files (the codex plan cited percentages but didn't include absolute times).
2. **Quantifies Step 12 (REV two-phase)** with exact FLOP counts — 2.77× / 13.4× / 40× for DNA / AA / codon. That moves Step 12 from "larger REV redesign" (third tier) to **second priority** in this plan.
3. **Identifies the missing link**: `theta_all` is entirely absent from the OpenACC path. Codex's Step 1 ("add buffer-likelihood entry points") implicitly depends on this, but the dependency isn't called out. Here it's an explicit foundation step (Step 1).
4. **Adds expected wall-time deltas** so priority can be compared directly.
5. **Groups steps into sprints** with clear exit criteria so the work can be validated incrementally.

Both documents agree on the ranking of the top shared fix (buffer likelihood) and the strongest REV/NONREV-specific fixes. Differences are in sequencing and quantification.

---

## 9. Step 1 + Step 2 Findings (added 2026-04-11)

This section records what actually happened when Steps 1 and 2 from §3 of this plan were implemented, built, and benchmarked on the DNA 100-taxa workload across four alignment lengths. The underlying data is in `step1_vs_step2_analysis.md` in the same folder as this document.

### 9.1 What was implemented

| Step | File(s) | Scope | Status |
|---|---|---|---|
| **Step 1** | `tree/phylotree.h`, `tree/phylokernel_openacc.cpp` | Add `theta_all` as a persistent GPU buffer. Populate it as a side effect in `derivKernelTipInt_Rev` and `derivKernelIntInt_Rev` via a `write_theta` gate (writes only on the first Newton iteration per branch). | Implemented, syntax-checked clean, landed as a pure side-effect write with no behavior change. |
| **Step 2** | `tree/phylokernel_openacc.cpp`, `tree/phylotree.h`, `tree/phylotreesse.cpp` | Add `bufferLikelihoodKernel_Rev<NSTATES>` and `PhyloTree::computeLikelihoodFromBufferRevOpenACC()`. Wire `computeLikelihoodFromBufferPointer` for REV only (NONREV stays NULL because state-space partials have no branch-invariant theta). | Implemented, reviewed via three simplify agents (reuse, quality, efficiency), review fixes applied (dead `pattern_lh` writes removed, unused `node` local removed, `node_is_leaf` → `dad_is_leaf`, fallback hoisted to lambda, narrative comments trimmed). Syntax-clean. |

Steps 3–12 have not been attempted.

### 9.2 Key benchmark results (2026-04-11, DNA, 100 taxa, full tree search)

Total wall time (seconds). NONREV is the noise control — neither Step 1 nor Step 2 changes it.

| len | NONREV Step 1 | NONREV Step 2 | REV Step 1 | REV Step 2 | REV Δ (Step 2 / Step 1) |
|---:|---:|---:|---:|---:|---:|
|    100 | 482.0 | 483.3 | 467.5 | 478.7 | **+2.4% (regression)** |
|   1000 | 303.2 | 309.1 | 291.5 | 290.5 | −0.3% (noise) |
|  10000 | 336.7 | 331.9 | 370.9 | 330.3 | **−11.0% (win)** |
| 100000 | 990.6 | 995.0 | 832.1 | 947.6 | **+13.9% (regression!)** |

**Noise ceiling (from NONREV Step 2 / Step 1 deltas):** ±2%. Any REV delta outside ±2% is a real signal.

### 9.3 Correctness

| len | NONREV −LnL | REV Step 1 −LnL | REV Step 2 −LnL |
|---:|---:|---:|---:|
|    100 | −4894.189 | −4894.189 | −4894.189 |
|   1000 | −56180.293 | −56180.293 | −56180.293 |
|  10000 | −564208.776 | −564208.777 | −564208.776 |
| 100000 | −5692984.539 | −5692984.529 | −5692984.529 |

All REV Step 1 and Step 2 values match to the printed precision. `.treefile` branch-length outputs differ by ~1 ULP between Step 1 and Step 2 — the expected signature of a correctly-reorganized reduction, and strong evidence that Step 2's fast path is actively being exercised (not silently falling through to the fallback).

The 0.01 gap between NONREV and REV at `len_100000` (−5692984.539 vs −5692984.529) is present in the 2026-04-03 baseline too — it is a known ASC correction ordering effect, unrelated to Step 1/2.

### 9.4 Does the projection hold?

From §3 of this plan, the projected impact of Sprint 1 (Step 1+2) was:

> Tree search wall: 743/688 s → ~530 s → saves ~150 s per run (~20% of current 688–743 s).

**Actual:** the projection holds ONLY at `len_10000`, and is wrong in both directions at other lengths.

| len | Projected Step 2 gain | Measured | Verdict |
|---:|---:|---:|---|
|    100 | −15% | **+2.4%** | Projection wrong (small regression) |
|   1000 | −15% | −0.3% | Projection not realized (noise) |
|  10000 | −15% | **−11.0%** | Projection close to hit ✓ |
| 100000 | −15% | **+13.9%** | Projection inverted (significant regression) |

**Average across lengths: −1.4%.** Step 2 is a **narrow win** at one specific problem size, not a broad improvement.

### 9.5 Root cause analysis for the `len_100000` regression

The regression is distributed across both ModelFinder (+14.5%) and tree search (+13.5%), so it is not localized to a single call site. Hypotheses ranked by likelihood:

1. **The fallback path was not actually as expensive as the plan assumed.** The plan estimated "5+ kernel launches per Newton iteration" for the fallback. In practice, when `traversal_info` is empty (the common case for pure branch-length Newton iterations), the fallback's `computeLikelihoodBranchRevOpenACC` skips partial recomputation and launches only the reduction kernel — i.e., **~1 kernel, not 5+**. This invalidates the projected 5–10× per-call speedup.

2. **Step 1's `theta_all` write in the derv kernel adds FP64 store bandwidth at scale.** For `len_100000`, each derv call writes `nptn × block × 8 bytes ≈ 12.8 MB` to `theta_all` when the `write_theta` gate is set (once per branch optimization). At V100 HBM2 ~450 GB/s write bandwidth, this is ~30 μs per derv call — small individually, tens of seconds across a full run.

3. **Fast-path host-side overhead exceeds its savings at large scale.** Per call: host `exp()` loop on `val0` (negligible), `update device(val0[0:block])` (~15 μs implicit H2D sync), kernel launch (~20 μs), scalar reduction back (~5 μs) — total ~40–50 μs. If the fallback is ~30 μs per call (one reduction kernel launch), the fast path is ~15 μs worse per call. Multiplied by thousands of calls, easily 15–60 seconds of regression.

4. **Cache pressure.** `theta_all` adds a new 12.8 MB GPU buffer at `len_100000`, vs. V100's 6 MB L2. Concurrent kernels may suffer increased cache-miss rates. Would need Nsight to quantify.

### 9.6 Revised recommendations

**Immediate (before Sprint 2):**

1. **Do NOT enable Step 2 unconditionally for REV.** The `len_100000` regression will show up on real protein alignments too (block size 5× larger for AA, so the theta bandwidth pressure kicks in earlier).
2. **Length-gate the dispatch** in `setLikelihoodKernel`:
   ```cpp
   bool use_buffer_fast_path = (nptn >= 5000 && nptn <= 50000);
   computeLikelihoodFromBufferPointer =
       (isReversible && use_buffer_fast_path)
           ? &PhyloTree::computeLikelihoodFromBufferRevOpenACC
           : NULL;
   ```
   Keeps the `len_10000` win without paying the `len_100000` cost. Trivial to implement.
3. **Gate the Step 1 `theta_all` side-effect write** on whether the fast path is actually wired. If `computeLikelihoodFromBufferPointer == NULL`, skip the write entirely (there is no consumer). Reclaims the ~10–30 seconds of write bandwidth at `len_100000`.
4. **Add per-call instrumentation.** Extend `acc_profile` with `n_buffer_lh` and `t_buffer_lh` counters. Rerun at `len_10000` (Step 2 wins) and `len_100000` (Step 2 loses) to measure per-call cost directly.

**Sprint 2 re-prioritization:**

5. **Promote the deferred H2 (GPU-side `val0`) optimization.** If the `len_100000` regression is dominated by the `update device(val0[0:block])` H2D round-trip (hypothesis 3), eliminating it by computing `val0` on the device from resident `gpu_eigenvalues`/`gpu_rate_cats`/`gpu_rate_props` should push `len_100000` back into the win column. Was deferred as "larger scope"; now promoted to "investigate first after length gating lands."
6. **Defer Step 8 (REV two-phase partial kernel refactor)** until Step 2 is no longer regressing. Step 8 is the next-biggest REV win but is also the most invasive; landing it on top of a regressing Step 2 would entangle the debugging.

**Benchmarking methodology:**

7. **Run tight replicates on a single V100 node.** All 16 2026-04-11 runs were on different nodes, which inflates noise. 3–5 back-to-back replicates of each configuration on one node would give confidence intervals.
8. **Include AA (protein) runs.** REV wins more heavily on AA in theory (c=20 derivative speedup), but the theta bandwidth pressure is also higher. The AA crossover point is essential data.
9. **Always run NONREV control alongside REV** — it's the clean noise floor.

### 9.7 Where we are relative to the original plan

| Sprint | Original plan goal | Status |
|---|---|---|
| Sprint 1 (Foundation) | Close shared OpenACC↔CPU-SIMD gap via theta_all + buffer-lh fast path. Exit: tree-search wall −15%. | **Partial.** Fast path works at one length, regresses at another. Needs length gating before shipping. |
| Sprint 2 (REV fast-start) | Selective first-call residency + two-phase REV partial kernels. Exit: REV fast ML within 20% of NONREV. | **Not started.** Defer until Sprint 1 is stable. |
| Sprint 3 (NONREV derivative) | NONREV TIP-INT derivative on GPU. Exit: NONREV tree-search matches REV on DNA100. | **Not started.** |
| Sprint 4 (Cleanup) | Scalar-only mode, LevelBatch cache, offset cache, sync relaxation, val on device, TIP-root on device, NONREV unification. | **Not started.** |
| Sprint 5 (Verification) | Full Nsight import. | **Not started.** |

**The plan's priority order still holds** — buffer-lh was correctly identified as the biggest shared bottleneck, and at `len_10000` it delivers the projected speedup. The projection was optimistic about where "bigger is always better" applied; the real picture is a Goldilocks zone.

### 9.8 Lessons learned for the rest of the plan

1. **Projections that assume the fallback is doing "5+ kernel launches per call" need to be verified** — look at the actual slow-path behavior in the common case (empty `traversal_info`), not the worst case.
2. **Always benchmark across multiple problem sizes**, not just the one from the profiling run. The `len_100` DNA profile led to the original plan; this was the narrowest possible test and missed that Step 2 regresses at both ends of the size spectrum.
3. **Side-effect writes at GPU DRAM scale are not free.** Step 1 was labeled "pure side-effect write" but at `len_100000` with `block=16` it touches 12.8 MB per call. For AA (`block=80`) that becomes 64 MB per call, which is definitely not free.
4. **Noise control via NONREV is essential.** Without the NONREV Step 1/2 pairs in this benchmark, the ±2% noise ceiling would have been hard to establish, and the `len_100` +2.4% REV regression would have been dismissed as noise.
5. **Steps 9 and 11** (NONREV TIP-INT derivative on GPU, REV `val0/1/2` on device) are still promising. Both are localized, well-understood changes with clear expected wins. They should land before any further buffer-path refinement.

