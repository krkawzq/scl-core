# SCL Kernel Optimization Analysis

This document analyzes all kernel implementations in scl/kernel/ and identifies optimization opportunities based on the design patterns in docs/design.md.

**CRITICAL RULE**: All optimization suggestions MUST use SCL custom operators instead of std:: functions. See docs/design.md Section 9 for complete replacement guide.

## Available Core Tools

### scl/core/algo.hpp
- Binary search: lower_bound, upper_bound
- Memory ops: copy, fill, zero
- Reductions: sum, max, min (SIMD optimized)
- Heap ops: heap_sift_down, heap_sift_up, make_heap, partial_sort
- Sparse ops: sparse_dot, sparse_dot_gallop, sparse_dot_adaptive
- Utility: swap, min2, max2, clamp, argmax, argmin, minmax

### scl/core/vectorize.hpp
- SIMD sum, product, dot
- SIMD find, count, contains
- SIMD min_element, max_element, minmax
- SIMD scale, add_scalar, add, sub, mul, div
- SIMD gather, scatter, scatter_add
- SIMD clamp, clamp_min, clamp_max
- SIMD abs_inplace, sum_abs, sum_squared
- SIMD fma, axpy
- SIMD sqrt, rsqrt, square, negate
- SIMD count_nonzero, all, any

### scl/core/memory.hpp
- aligned_alloc, aligned_free, AlignedBuffer
- fill, zero (SIMD optimized)
- copy_fast, copy, stream_copy
- prefetch utilities
- swap, swap_ranges, reverse

### scl/core/macros.hpp
- SCL_LIKELY, SCL_UNLIKELY
- SCL_FORCE_INLINE, SCL_RESTRICT
- SCL_PREFETCH_READ, SCL_PREFETCH_WRITE
- SCL_HOT, SCL_COLD
- SCL_UNROLL, SCL_VECTORIZE
- SCL_ASSUME, SCL_UNREACHABLE

### scl/core/sort.hpp
- **VQSort (SIMD-optimized, parallel)**: `sort`, `sort_descending`
- **Key-value sorting**: `sort_pairs`, `sort_pairs_descending`
- **Performance**: 2-5x faster than `std::sort`

---

## MANDATORY Optimization Rules

**CRITICAL**: All kernel implementations MUST follow these rules for maximum performance:

1. **NO std:: functions in hot paths**:
   - Replace `std::sort` → `scl::sort::sort` or `scl::sort::sort_pairs` (VQSort, 2-5x faster)
   - Replace `std::partial_sort` → `scl::algo::partial_sort`
   - Replace `std::nth_element` → `scl::algo::nth_element`
   - Replace `std::memcpy` → `scl::memory::copy_fast` or `scl::algo::copy`
   - Replace `std::memset` → `scl::memory::zero` or `scl::algo::zero`
   - Replace `std::min`/`std::max` → `scl::algo::min2`/`max2` or `scl::algo::min`/`max` for arrays
   - Replace `std::exp`/`std::log` in loops → `scl::simd::Exp`/`scl::simd::Log` for batch processing

2. **SIMD for all vectorizable operations**:
   - Use `scl::vectorize::sum` instead of manual sum loops
   - Use `scl::vectorize::dot` for dot products
   - Use `scl::vectorize::scale` for scaling operations
   - Use `scl::simd::Exp`, `scl::simd::Log`, `scl::simd::Sqrt` for transcendental functions

3. **Parallelization**:
   - Use `scl::threading::parallel_for` for all parallelizable loops
   - Use `scl::threading::WorkspacePool` for thread-local buffers (avoid allocations in hot paths)

4. **Memory management**:
   - Use `scl::memory::aligned_alloc`/`aligned_free` instead of `new[]`/`delete[]`
   - Use `scl::memory::AlignedBuffer` for RAII memory management
   - NO `std::vector` in hot paths

See `docs/design.md` Section 9 for complete replacement table.

---

## Kernel Analysis


### Batch 1: log1p.hpp, qc.hpp, resample.hpp, normalize.hpp

#### 1. log1p.hpp
**Status**: Well-optimized
- Uses 4-way SIMD unrolling with prefetch
- Proper scalar cleanup loop
- Uses scl::simd::Log1p, Expm1

**No major optimization needed.**

---

#### 2. qc.hpp
**Status**: Partially optimized

**Current Issues**:
- `fused_total_subset_sum` uses scalar loop with mask[indices[k]] indirect access

**Optimization Opportunities**:
- [ ] `fused_total_subset_sum`: Consider prefetching mask data ahead
- [ ] Add branch hints: `SCL_LIKELY` for non-zero values path
- [ ] **MANDATORY**: Replace any `std::` functions with SCL custom operators (see design.md Section 9)

**Suggested Change**:
```cpp
// Add prefetch for indirect mask access
for (; k + 4 <= len; k += 4) {
    if (SCL_LIKELY(k + 8 < len)) {
        SCL_PREFETCH_READ(&mask[indices[k + 8]], 0);
    }
    // ... rest of loop
}
```

---

#### 3. resample.hpp
**Status**: Needs improvement

**Current Issues**:
- `sum_simd_4way` for non-Real types: Only uses scalar 4-way unroll, no SIMD
- `binomial_resample` and `poisson_resample` inner loops are sequential

**Optimization Opportunities**:
- [ ] **MANDATORY**: Use `scl::vectorize::sum` directly instead of custom implementation
- [ ] Add prefetch in sampling loops
- [ ] Consider batching RNG calls (generate multiple random numbers at once)
- [ ] **MANDATORY**: Replace any `std::` functions with SCL custom operators

**Suggested Change**:
```cpp
// Replace custom sum with vectorize::sum
template <typename T>
SCL_FORCE_INLINE Real sum_simd_4way(const T* vals, Size len) {
    // For any T, cast to Real array and use vectorize
    return scl::vectorize::sum(Array<const Real>(
        reinterpret_cast<const Real*>(vals), len));
}
```

---

#### 4. normalize.hpp
**Status**: Well-optimized

**Current Issues**:
- `sum_masked_simd`: Indirect mask access prevents SIMD optimization
- `detect_highly_expressed`: Uses `__atomic_store_n` directly (compiler-specific)

**Optimization Opportunities**:
- [ ] Replace `__atomic_store_n` with `std::atomic` (acceptable for atomic operations)
- [ ] Add prefetch for mask indirect access in `sum_masked_simd`
- [ ] **MANDATORY**: Replace any `std::` functions with SCL custom operators

**Minor Suggested Change**:
```cpp
// Add prefetch for better cache behavior
for (; k + 4 <= len; k += 4) {
    SCL_PREFETCH_READ(&mask[indices[k + 8]], 0);  // Prefetch ahead
    // ... rest unchanged
}
```

---


### Batch 2: algebra.hpp, feature.hpp, sparse.hpp, scale.hpp

#### 5. algebra.hpp
**Status**: Well-optimized

- Adaptive SpMV with 3-tier strategy (short/medium/long rows)
- 8-way unroll with prefetch for long rows
- Uses horizontal_sum_8 for accumulator reduction

**Optimization Opportunities**:
- [ ] Consider SIMD gather for dense x vector lookup (if indices are consecutive)
- [ ] `scale_output`: Missing prefetch in beta scaling loop

**Minor Issue**: `detail::scale_output` uses single-lane SIMD loop; could use 4-way unroll.

---

#### 6. feature.hpp
**Status**: Well-optimized

- `compute_sum_sq_simd`: Excellent fused sum+sumsq with 4-way SIMD + FMA
- `compute_clipped_sum_sq_simd`: Good clipped version with SIMD min
- `dispersion`: Full SIMD with masked division

**No major optimization needed.**

---

#### 7. sparse.hpp
**Status**: Well-optimized

- Uses `scl::vectorize::sum` for primary_sums/means
- Fused sum+sumsq helper matches design patterns
- Clean parallel_for structure

**Optimization Opportunities**:
- [ ] `primary_nnz`: Could batch multiple rows per thread for better cache usage

---

#### 8. scale.hpp
**Status**: Excellent optimization

- 3-tier adaptive strategy (short < 16, medium < 128, long >= 128)
- 8-way SIMD unroll for long rows with prefetch
- Branch on zero_center/do_clip lifted outside inner loop

**Best practice example for adaptive optimization.**

**No optimization needed.**

---


### Batch 3: softmax.hpp, mwu.hpp, neighbors.hpp, mmd.hpp

#### 9. softmax.hpp
**Status**: Excellent optimization

- 3-tier adaptive strategy (short/medium/long)
- 8-way SIMD unroll for long rows
- Fused exp+sum computation
- Proper numerical stability (max subtraction)

**Best practice example for transcendental functions.**

**No optimization needed.**

---

#### 10. mwu.hpp
**Status**: Well-optimized

- Uses DualWorkspacePool for per-thread buffers
- Uses scl::sort::sort (VQSort) for SIMD sorting
- 4-way unrolled partition loop

**Optimization Opportunities**:
- [ ] `compute_rank_sum_sparse`: No SIMD (sequential merge) - hard to vectorize due to tie handling
- [ ] Consider using `scl::vectorize::count` for group counting (already done)

**Minor Issue**: Sequential while loops for negative value scanning could use binary search.

---

#### 11. neighbors.hpp
**Status**: Excellent optimization

- Adaptive sparse dot: linear/binary/gallop based on ratio
- 8-way/4-way skip optimization for range checks
- Cauchy-Schwarz lower bound pruning for early exit
- WorkspacePool for heap storage
- Custom heap implementation for HeapElement struct

**Best practice example for adaptive algorithms.**

**No optimization needed.**

---

#### 12. mmd.hpp
**Status**: Excellent optimization

- 8-way SIMD unroll with prefetch
- Block tiling for cross-kernel sum (BLOCK_X=64, BLOCK_Y=512)
- Symmetric kernel optimization (compute upper triangle only)
- DualWorkspacePool for cache buffers

**Best practice example for O(n^2) algorithms.**

**No optimization needed.**

---


### Batch 4: spatial.hpp, bbknn.hpp, correlation.hpp, gram.hpp

#### 13. spatial.hpp
**Status**: Partially optimized

**Current Issues**:
- `morans_i` inner loop is sequential (O(n_cells) per feature)
- Weighted neighbor sum uses scalar 4-way unroll, not SIMD

**Optimization Opportunities**:
- [ ] `compute_weighted_neighbor_sum`: No SIMD because of indirect z[] access
- [ ] Consider parallelizing inner loop for very large cell counts
- [ ] `morans_i`: Inner loop over cells is sequential - potential for nested parallelism

**Note**: Indirect access pattern `z[indices[k]]` prevents efficient SIMD gather.

---

#### 14. bbknn.hpp
**Status**: Excellent optimization

- Batch-grouped processing for locality
- Custom KHeap with manual sift operations
- Sparse dot with 8/4-way skip optimization
- Cauchy-Schwarz lower bound pruning
- Efficient batch group structure (contiguous storage + offsets)

**Best practice example for batch-constrained KNN.**

**No optimization needed.**

---

#### 15. correlation.hpp
**Status**: Excellent optimization

- Fused sum+sumsq with 4-way SIMD + FMA
- Algebraic identity for sparse centered dot product
- 8/4-way skip optimization in merge
- Symmetric matrix optimization (upper triangle only)
- Chunk-based parallel processing

**Best practice example for correlation computation.**

**No optimization needed.**

---

#### 16. gram.hpp
**Status**: Excellent optimization

- Adaptive sparse dot: linear/binary/gallop
- Range narrowing via lower_bound/upper_bound
- 8/4-way skip optimization
- Symmetric output (only compute upper triangle)

**Best practice example for Gram matrix.**

**No optimization needed.**

---


### Batch 5: annotation.hpp, centrality.hpp, coexpression.hpp, communication.hpp

#### 17. annotation.hpp
**Status**: Needs significant optimization

**Current Issues**:
- `compute_row_mean`, `compute_row_norm`: Sequential loops, no SIMD
- `sparse_dot_product`: Basic merge without skip optimization
- `pearson_correlation`: Allocates dense arrays - slow for large n_features
- `top_markers_per_type`: Insertion sort O(n^2) instead of partial_sort

**Optimization Opportunities**:
- [ ] **MANDATORY**: Use `scl::vectorize::sum` for row sums (replace any manual sum loops)
- [ ] **MANDATORY**: Use `scl::algo::sparse_dot_adaptive` for sparse dot product
- [ ] **MANDATORY**: Use `scl::algo::partial_sort` or heap-based top-k selection (replace `std::sort` or insertion sort)
- [ ] Add `scl::threading::parallel_for` for `reference_mapping`, `correlation_assignment`
- [ ] **MANDATORY**: Replace `std::exp` with `scl::simd::Exp` in marker_gene_score
- [ ] **MANDATORY**: Replace `std::sort` with `scl::sort::sort` or `scl::sort::sort_pairs` for any sorting operations
- [ ] **MANDATORY**: Replace any `std::memcpy`, `std::memset` with `scl::memory::copy_fast`, `scl::memory::zero`

---

#### 18. centrality.hpp
**Status**: Excellent optimization

- SIMD helpers: sum, norm_squared, l1_diff, scale, axpby
- Parallel PageRank/HITS with atomic accumulation
- Parallel Brandes betweenness with per-thread workspace
- Fast BFS queue with prefetch
- Xoshiro256++ RNG for sampled betweenness

**Best practice example for graph algorithms.**

**No optimization needed.**

---

#### 19. coexpression.hpp
**Status**: Well-optimized

- SIMD dot, sum, norm_squared, axpy, scale
- Shell sort for ranking (Spearman)
- Quickselect for median (bicor)
- Parallel correlation matrix (upper triangular)
- Parallel TOM computation
- Power iteration for eigengene (parallel SpMV)
- WorkspacePool for thread-local buffers

**Minor Issues**:
- `quickselect_median`: `std::swap` instead of `scl::algo::swap` (minor, but should use SCL for consistency)
- Hierarchical clustering uses sequential O(n^2) loop in some cases
- **MANDATORY**: Replace any `std::` functions with SCL custom operators

**No major optimization needed.**

---

#### 20. communication.hpp
**Status**: Excellent optimization

- Xoshiro256++ PRNG with Lemire's bounded random
- SIMD sum and dot product
- Precomputed gene expressions (cache-friendly)
- TypeInfo structure for efficient mask-based computation
- Parallel permutation tests with thread-local RNG
- 4-way unrolled Fisher-Yates shuffle
- Branchless score computation

**Best practice example for permutation tests.**

**No optimization needed.**

---


### Batch 6: diffusion.hpp, doublet.hpp, enrichment.hpp, gnn.hpp

#### 21. diffusion.hpp
**Status**: Excellent optimization

- 4-way SIMD unrolling with FMA in dot_product_simd, axpy_simd, scale_simd
- Parallel SpMV with 4-way unrolled accumulation and prefetch
- Block-wise SpMM for cache efficiency
- Fused SpMV with linear combination (spmv_fused_linear)
- SIMD convergence check (check_convergence_simd)
- Modified Gram-Schmidt with reorthogonalization
- Parallel DPT computation
- Memory-efficient large graph handling (row-by-row for n > 1000)

**Best practice example for diffusion algorithms.**

**No optimization needed.**

---

#### 22. doublet.hpp
**Status**: Excellent optimization

- `squared_distance`: 4-way SIMD unrolling with FMA and prefetch
- `partial_sort_k_smallest`: Heap-based O(n log k) selection (max-heap)
- `compute_knn_doublet_scores`: Parallel with WorkspacePool for thread-local buffers
- `estimate_threshold`: Uses `scl::sort::sort` for O(n log n) sorting
- `doublet_score_stats`: SIMD sum/variance + `scl::sort::sort` for median
- `combined_doublet_score`: SIMD vectorized normalization and combination
- Full parallelization in main scoring functions with adaptive thresholds
- Xoshiro256++ RNG with Lemire's bounded random for simulation
- 4-way unrolled scatter operations in sparse-to-dense conversion

**No optimization needed.**

---

#### 23. enrichment.hpp
**Status**: Well-optimized

- Branch hints (SCL_LIKELY, SCL_UNLIKELY) throughout
- Fast path for weight_exponent == 1 in GSEA
- Parallel batch ORA with threshold checking
- Uses `scl::sort::sort_pairs` for efficient sorting (BH correction)
- SIMD Bonferroni correction with proper clamping
- Parallel pathway activity with 4-way unrolled loop
- Per-thread workspace for GSVA/ssGSEA
- Parallel enrichment map computation

**No major optimization needed.**

---

#### 24. gnn.hpp
**Status**: Excellent optimization

- Full SIMD feature operations: add, add_scaled, max, min, scale, fill, dot
- Vectorized ReLU activation
- Fused softmax + aggregation
- Precomputed GCN normalization factors
- WorkspacePool for attention scores/probs
- Parallel message passing with proper initialization
- Multi-head attention with per-head parallelization
- Parallel global pool with thread-local reduction
- SIMD skip connection
- Parallel layer/batch normalization

**Best practice example for GNN operations.**

**No optimization needed.**

---



### Batch 7: entropy.hpp, grn.hpp, hotspot.hpp, impute.hpp

#### 25. entropy.hpp
**Status**: Excellent optimization

**Current Optimizations**:
- SIMD sum with 2-way unrolling
- 4-way unrolled p*log(p) computation with scl::simd::Log for ILP
- 4-way unrolled JS divergence computation
- 4-way unrolled entropy_from_counts computation
- Thread-local histogram accumulation (parallel histogram)
- Parallel discretization with SIMD min/max reduction
- Shell sort for equal-frequency discretization
- Parallel MI computation with WorkspacePool
- Parallel mRMR feature selection
- Uses `scl::algo::partial_sort` for O(n log k) feature selection

**Best practice example for information theory.**

**No major optimization needed.**

---

#### 26. grn.hpp
**Status**: Excellent optimization

**Current Optimizations**:
- SIMD mean/variance computation with 2-way unrolling
- SIMD Pearson correlation with multi-accumulator (6 accumulators)
- Uses `scl::sort::sort_pairs` for efficient ranking (Spearman)
- SIMD min/max for mutual information histogram
- Binary search with `scl::algo::lower_bound`
- Parallel correlation/MI network computation
- DualWorkspacePool for GENIE3 importance
- Parallel gene extraction
- Parallel TF activity and regulon scoring
- Parallel clustering coefficient computation

**Minor Issues**:
- `correlation_network`: Gene pair loop is sequential (commented "sequential for simplicity")
- `mutual_information`: Histogram building is sequential

**Best practice example for GRN inference.**

**No major optimization needed.**

---

#### 27. hotspot.hpp
**Status**: Excellent optimization

**Current Optimizations**:
- Xoshiro256++ PRNG with Lemire's nearly divisionless method
- AVX2 SIMD for statistics (compute_sum, compute_variance, standardize)
- 4-way unrolled Fisher-Yates shuffle
- 4-way unrolled spatial lag computation
- Parallel local Moran's I, Gi*, Geary's C
- Thread-local workspace with WorkspacePool for permutation tests
- Branchless quadrant classification
- Parallel BH FDR correction with `scl::sort::sort_pairs` (VQSort)
- Parallel distance band/KNN weight construction
- SIMD fused mean subtraction and variance in standardize

**Minor Issues**:
- Uses direct AVX intrinsics which may affect portability (has scalar fallback though)

**Best practice example for spatial statistics.**

**No major optimization needed.**

---

#### 28. impute.hpp
**Status**: Excellent optimization

**Current Optimizations**:
- SIMD axpy, scale, dot with 2-way unrolling
- Branchless binary search for gene lookup (adaptive: linear for small arrays)
- Block processing for cache efficiency (GENE_BLOCK_SIZE=64, CELL_BLOCK_SIZE=32)
- Parallel SpMM for diffusion steps
- Double buffering for diffusion (avoids memory allocation per step)
- WorkspacePool for thread-local buffers
- Parallel power iteration for truncated SVD (ALRA)
- Parallel dropout detection with atomic accumulation
- Parallel imputation quality (correlation) with thread-local reduction
- Precomputed weights in KNN imputation
- Uses `scl::algo::copy` for memory operations (no std::memcpy)

**Best practice example for imputation algorithms.**

**No optimization needed.**

---


### Batch 8: kernel.hpp, leiden.hpp, markers.hpp, multiple_testing.hpp

#### 29. kernel.hpp
**Status**: Well-optimized

**Current Optimizations**:
- SIMD Gaussian kernel batch evaluation (gaussian_kernel_batch)
- SIMD-optimized mean and variance computation with 2-way unrolling
- 4-way loop unrolling in KDE and local bandwidth computation
- Parallel processing with threshold (PARALLEL_THRESHOLD = 500)
- Precomputed kernel parameters (KernelParams structure)
- SIMD acceleration for Nystrom approximation (Gram-Schmidt)
- SIMD for mean shift step high-dimensional accumulation

**Minor Issues**:
- FastRNG uses simple XorShift, could use Xoshiro256++ like other kernels
- No prefetch hints in dense computation sections

**No major optimization needed.**

---

#### 30. leiden.hpp
**Status**: Excellent optimization

**Current Optimizations**:
- Open-addressing hash table with Fibonacci hashing for O(1) community lookup
- Parallel local moving with atomic community updates
- Fixed-point scaling for atomic sigma_tot updates (SCALE = 1000000)
- SIMD-accelerated degree computation (compute_node_degrees_simd)
- Cache-aligned data structures (alignas(64))
- Per-thread WorkspacePool for hash table keys/values
- Lock-free atomic updates
- Fisher-Yates shuffle with FastRNG (Xoshiro128+)
- Parallel modularity computation with thread-local partial reduction
- Stochastic refinement phase with probabilistic acceptance

**Best practice example for graph clustering algorithms.**

**No optimization needed.**

---

#### 31. markers.hpp
**Status**: Well-optimized

**Current Optimizations**:
- SIMD-optimized log2 fold change computation (compute_log2_fc_batch with scl::simd::Log)
- SIMD for Gini coefficient calculation
- SIMD for Tau specificity score (Max, SumOfLanes, MaxOfLanes)
- Parallel processing over genes (CSC format benefits)
- 4-way loop unrolling in group mean accumulation
- Uses `scl::sort::sort` and `sort_pairs_descending` for efficient ranking
- WorkspacePool for thread-local sorting buffers
- Branch hints (SCL_LIKELY) for common paths

**Optimization Opportunities**:
- [ ] `marker_overlap_jaccard`: Uses O(n*m) nested loops - consider sorted merge or hash set
- [ ] Some sum operations could use `scl::vectorize::sum` directly

**No major optimization needed.**

---

#### 32. multiple_testing.hpp
**Status**: Needs improvement

**Current Issues**:
- **CRITICAL**: Uses `std::sort` with lambda instead of `scl::sort::sort_pairs` (VQSort is 2-5x faster)
- No SIMD optimization for p-value adjustments (should use `scl::vectorize::scale`)
- Sequential loops throughout (no parallelization with `scl::threading::parallel_for`)
- KDE estimation is O(n * n_grid) without SIMD or parallelization
- `empirical_fdr` has O(n * n_perms * n) complexity with nested loops
- No branch hints or prefetch
- **CRITICAL**: Uses `std::log` instead of `scl::simd::Log` for batch log transformations

**Optimization Opportunities**:
- [ ] **MANDATORY**: Replace `std::sort` with `scl::sort::sort_pairs` for index-value sorting
- [ ] Parallelize Benjamini-Hochberg, Storey, Holm adjustments using `scl::threading::parallel_for`
- [ ] **MANDATORY**: SIMD for p-value multiplication in Bonferroni: use `scl::vectorize::scale`
- [ ] Parallelize KDE estimation for local FDR using `scl::threading::parallel_for`
- [ ] Parallelize empirical FDR permutation counting using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Use SIMD for log transformations: replace `std::log` with `scl::simd::Log` in `neglog10_pvalues`
- [ ] Add `SCL_LIKELY`/`SCL_UNLIKELY` branch hints
- [ ] **MANDATORY**: Replace all `std::` functions with SCL custom operators (see design.md Section 9)

**Suggested Changes** (MANDATORY: All std:: functions replaced):
```cpp
// MANDATORY: Replace std::sort with scl::sort::sort_pairs
void benjamini_hochberg(...) {
    Real* sorted_pvalues = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Index* sorted_indices = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    
    // Copy p-values and initialize indices (use scl::algo::copy for fast copy)
    scl::algo::copy(p_values.ptr, sorted_pvalues, n);
    for (Size i = 0; i < n; ++i) {
        sorted_indices[i] = static_cast<Index>(i);
    }
    
    // MANDATORY: Use scl::sort::sort_pairs (VQSort - SIMD-optimized, parallel)
    scl::sort::sort_pairs(
        Array<Real>(sorted_pvalues, n),
        Array<Index>(sorted_indices, n)
    );
    
    // Parallelize adjustment computation
    scl::threading::parallel_for(Size(0), n, [&](size_t i) {
        Real rank = static_cast<Real>(i + 1);
        adjusted_sorted[i] = sorted_pvalues[i] * n_real / rank;
    });
    
    scl::memory::aligned_free(sorted_pvalues, SCL_ALIGNMENT);
    scl::memory::aligned_free(sorted_indices, SCL_ALIGNMENT);
    // ... rest
}

// MANDATORY: SIMD Bonferroni using scl::vectorize
void bonferroni(...) {
    scl::vectorize::scale(
        Array<Real>(adjusted_p_values.ptr, n),
        Array<const Real>(p_values.ptr, n),
        n_real  // multiplier
    );
    scl::vectorize::clamp_max(
        Array<Real>(adjusted_p_values.ptr, n),
        Real(1.0)
    );
}
```

---



### Batch 9: permutation.hpp, projection.hpp, propagation.hpp, pseudotime.hpp

#### 33. permutation.hpp
**Status**: Excellent optimization

**Current Optimizations**:
- Xoshiro256++ PRNG with jump() function for parallel independent streams
- Lemire's nearly divisionless bounded random (2-3x faster than modulo)
- AVX2/AVX SIMD for p-value counting (`count_geq_simd`, `count_abs_geq_simd`)
- 4-way unrolled Fisher-Yates shuffle
- Branchless group statistics with 4-way unrolling
- Fast correlation with precomputed x-statistics
- Adaptive early stopping in permutation tests (EARLY_CHECK_INTERVAL)
- Kahan summation for BY FDR correction (numerical stability)
- Parallel correlation test with WorkspacePool
- Uses `scl::argsort::argsort` for FDR corrections
- SIMD Bonferroni correction (AVX2)

**Best practice example for permutation testing.**

**No optimization needed.**

---

#### 34. projection.hpp
**Status**: Excellent optimization

**Current Optimizations**:
- Xoshiro256** PRNG for fast bulk random generation
- Fast hash with dual hash computation (bucket + sign)
- 4-way SIMD unrolled accumulation with MulAdd (`accumulate_simd`)
- Block-wise Gaussian projection for cache efficiency (BLOCK_SIZE=256)
- Multiple projection types: Gaussian, Achlioptas, Sparse, CountSketch, FeatureHash
- Cache-aligned ProjectionMatrix (alignas(64)) and SparseProjectionMatrix
- Prefetch hints in dense matrix transform
- Auto-selection of projection method based on dimensions (`project_auto`)
- Johnson-Lindenstrauss optimal dimension computation
- Memory-efficient on-the-fly projections (no precomputation)
- Adaptive sparsity mask for sparse projections

**Best practice example for random projection.**

**No optimization needed.**

---

#### 35. propagation.hpp
**Status**: Excellent optimization

**Current Optimizations**:
- SIMD convergence check with 4-way unrolling for integer labels (`check_convergence`)
- Multi-accumulator pattern for real convergence check with prefetch
- Parallel row sum computation with 2-way unrolled SIMD
- 4-way unrolled argmax for class voting (`find_argmax`)
- SIMD row normalization (`normalize_row`)
- WorkspacePool for per-thread class votes
- Parallel label propagation with chunk-based processing
- Parallel label spreading with SIMD accumulation
- Parallel inductive transfer with weighted voting
- Parallel confidence propagation
- Parallel harmonic function (Jacobi-style with atomic max)
- Vectorized soft label initialization with SIMD fill

**Best practice example for graph-based semi-supervised learning.**

**No optimization needed.**

---

#### 36. pseudotime.hpp
**Status**: Excellent optimization

**Current Optimizations**:
- SIMD dot, axpy, scale operations with 2-way unrolling
- 4-ary min heap (faster than binary for Dijkstra, HEAP_ARITY=4)
- Cache-aligned FastMinHeap structure (alignas(64))
- Parallel multi-source Dijkstra with per-thread heaps
- Parallel SpMV with 4-way unrolling for transition matrix
- Block SpMM for diffusion components (`spmm_block`)
- Modified Gram-Schmidt orthogonalization for diffusion components
- Parallel diffusion pseudotime computation
- Parallel branch point detection with thread-local counts
- Parallel pseudotime smoothing
- Parallel pseudotime-gene correlation with two-pass algorithm
- Parallel velocity-weighted pseudotime refinement

**Best practice example for trajectory analysis.**

**No optimization needed.**

---


### Batch 10: scoring.hpp, transition.hpp, velocity.hpp, hvg.hpp

#### 37. scoring.hpp
**Status**: Excellent optimization

**Current Optimizations**:
- Xoshiro128+ FastRNG with cache-aligned structure (alignas(16))
- SIMD sum, scale, axpy with 2-way unrolling
- Bitset GeneSetLookup for O(1) gene membership check
- Shell sort + insertion sort (adaptive) for ranking
- Parallel mean/weighted/AUC scoring with WorkspacePool
- Atomic accumulation with fixed-point scaling for CSC format
- Fused z-score computation with precomputed zero z-scores
- Parallel multi-signature scoring over gene sets
- Branchless bin assignment (find_bin)

**Best practice example for gene set scoring.**

**No optimization needed.**

---

#### 38. transition.hpp
**Status**: Excellent optimization

**Current Optimizations**:
- AVX2/AVX SIMD with scalar fallback: dot_product, vector_norm, vector_sum, scale_vector, axpy, max_abs_diff
- SOR (Successive Over-Relaxation) with omega=1.5 for faster convergence
- Aitken delta-squared acceleration for power iteration
- Parallel SpMV with 4-way unrolling
- Parallel absorption probability computation with SOR
- Parallel metastable states (k-means++ initialization, parallel assignment)
- Thread-local accumulation for coarse-grain transition
- Lock-free atomic max updates for convergence tracking
- Fused normalize L1/L2 with sum/norm return

**Minor Issue**: Duplicate namespace closing at lines 1269 and 1272.

**Best practice example for Markov chain analysis.**

**No major optimization needed.**

---

#### 39. velocity.hpp
**Status**: Excellent optimization

**Current Optimizations**:
- SIMD linear regression with multi-accumulator pattern (6 accumulators)
- SIMD cosine similarity with 2-way unrolling
- SIMD softmax with scl::simd::Exp and vectorized normalization
- SIMD vector diff/accumulate helpers
- DualWorkspacePool for thread-local gene value buffers
- Parallel gene kinetics fitting (CSR: binary search, CSC: direct column access)
- Parallel velocity computation with CSR/CSC awareness
- Parallel velocity graph with softmax transition probabilities
- Parallel velocity confidence computation
- WorkspacePool for delta buffers in embedding projection
- Branch hints (SCL_LIKELY, SCL_UNLIKELY) throughout

**Best practice example for RNA velocity analysis.**

**No optimization needed.**

---

#### 40. hvg.hpp
**Status**: Excellent optimization

**Current Optimizations**:
- 4-way SIMD dispersion computation with prefetch hints
- SIMD normalized dispersion with masked operations
- Uses `scl::vectorize::sum` and `scl::vectorize::sum_squared` for moments
- Uses `scl::algo::partial_sort` for efficient top-k selection
- Uses `scl::algo::iota` for index initialization
- Parallel compute_moments with `scl::threading::parallel_for`
- Parallel compute_clipped_moments for VST method
- Masked SIMD division with IfThenElse for safe dispersion

**Best practice example for highly variable gene selection.**

**No optimization needed.**

---


### Batch 11: ttest.hpp, merge.hpp, slice.hpp, group.hpp

#### 41. ttest.hpp
**Status**: Partially optimized

**Current Optimizations**:
- Fast erfc approximation using Horner polynomial (5-term expansion)
- 4-way scalar unrolling in `compute_group_stats` inner loop
- Parallel processing over features via `parallel_for`
- Two-pass algorithm: accumulate statistics, then finalize in parallel
- Welch's t-test support

**Optimization Opportunities**:
- [ ] `compute_group_stats`: 4-way unroll is scalar, not SIMD (indirect access prevents vectorization)
- [ ] `fast_erfc`: Uses `std::exp` instead of `scl::simd::Exp` for batch processing
- [ ] No batch t-test computation with SIMD - could vectorize across features
- [ ] Consider using `scl::vectorize::sum` for sum/sum_sq accumulation where applicable

**Note**: The indirect access pattern `group_ids[indices[k]]` makes SIMD difficult. Current scalar unrolling is appropriate.

---

#### 42. merge.hpp
**Status**: Well-optimized

**Current Optimizations**:
- SIMD `add_offset_simd` for index offsetting with 2-way unrolling
- Parallel memcpy with prefetch for large data blocks (`parallel_memcpy`)
- Parallel vstack/hstack over primary dimension segments
- Early exit for zero offset (direct memcpy)
- Chunk-based parallel processing for large arrays

**Best practice example for matrix concatenation.**

**No major optimization needed.**

---

#### 43. slice.hpp
**Status**: Well-optimized

**Current Optimizations**:
- 8-way scalar unrolling in `count_masked_fast` for mask counting
- Parallel reduce for nnz computation (`parallel_reduce_nnz`)
- Prefetch-optimized copy (`fast_copy_with_prefetch`)
- Adaptive parallelization with thresholds (PARALLEL_THRESHOLD_ROWS = 512, PARALLEL_THRESHOLD_NNZ = 10000)
- Index mapping construction for secondary dimension filtering
- Inspect + materialize pattern for two-phase slicing

**Optimization Opportunities**:
- [ ] `count_masked_fast`: 8-way unroll is scalar; SIMD gather could be faster but indirect access `mask[indices[k]]` limits options
- [ ] Consider SIMD comparison for contiguous mask regions

**Note**: Indirect memory access patterns limit SIMD opportunities. Current optimization is near-optimal for the access patterns.

---

#### 44. group.hpp
**Status**: Partially optimized

**Current Optimizations**:
- 4-way scalar unrolling with prefetch hints in inner loop
- Stack allocation for small group counts (nnz_counts_local[256])
- Parallel processing over primary dimension
- Branch hints with `SCL_LIKELY` for common path
- Configurable PREFETCH_DISTANCE = 64
- Welford-style variance computation with ddof support

**Optimization Opportunities**:
- [ ] Inner loop 4-way unroll is scalar, not SIMD (indirect access `group_ids[indices[k]]`)
- [ ] Consider using local accumulators per thread to reduce cache contention
- [ ] Add `#pragma unroll` directive (already present but compiler-dependent)

**Note**: Like ttest.hpp, the indirect access pattern prevents efficient SIMD. Current scalar unrolling is the best approach for this access pattern.

---


### Batch 12: reorder.hpp, sparse_opt.hpp, niche.hpp, louvain.hpp

#### 45. reorder.hpp
**Status**: Well-optimized

**Current Optimizations**:
- 3-tier adaptive strategy: short (<32), medium (<256), long (>=256)
- 4-way unrolled `count_valid_unrolled` with prefetch
- Multi-level prefetch hints (PREFETCH_DIST1=16, PREFETCH_DIST2=32 for long rows)
- Parallel processing over primary dimension via `parallel_for`
- Uses `scl::sort::sort_pairs` for sorted index remapping
- `SCL_LIKELY` branch hints for common valid index path
- Branchless valid index check using unsigned comparison

**Best practice example for sparse matrix reordering.**

**No major optimization needed.**

---

#### 46. sparse_opt.hpp
**Status**: Stub file (declarations only - TODO)

This file contains only function declarations and TODO comments without implementations. Planned features include:
- Coordinate descent for Lasso regression
- Elastic net via coordinate descent
- Proximal gradient methods (ISTA, FISTA)
- Iterative hard thresholding (IHT)
- Group Lasso
- Regularization path computation

**Cannot analyze optimization - implementation needed first.**

---

#### 47. niche.hpp
**Status**: Stub file (declarations only - TODO)

This file contains only function declarations and TODO comments without implementations. Planned features include:
- Neighborhood composition computation
- Niche clustering
- Cell-cell contact frequency
- Co-localization scoring with permutation tests

**Cannot analyze optimization - implementation needed first.**

---

#### 48. louvain.hpp
**Status**: Needs significant optimization

**Current Optimizations**:
- Multi-level Louvain algorithm with graph aggregation
- Modularity gain computation
- Community relabeling to contiguous indices

**Optimization Opportunities (Compare with leiden.hpp)**:
- [ ] `compute_total_weight`: Sequential loop, no SIMD or parallelization
- [ ] `compute_node_degrees`: Sequential loop, could use `scl::vectorize::sum` per row
- [ ] `local_moving_phase`: Sequential node processing, no parallelization
- [ ] `aggregate_graph`: O(n * edges_per_comm) linear search for community edges - should use open-addressing hash table
- [ ] `compute_k_i_in`: No loop unrolling or prefetch hints
- [ ] No use of WorkspacePool for thread-local buffers
- [ ] No atomic updates or fixed-point scaling for parallel community updates
- [ ] Missing `SCL_FORCE_INLINE` on some frequently called helpers
- [ ] Community neighbor search uses linear scan instead of hash table

**Suggested Changes (Reference leiden.hpp patterns)**:
```cpp
// Use open-addressing hash table for neighbor community lookup
// leiden.hpp uses Fibonacci hashing with open addressing

// Parallelize node degree computation
void compute_node_degrees_parallel(...) {
    scl::threading::parallel_for(Size(0), n, [&](size_t i) {
        auto values = adj.primary_values(i);
        degrees[i] = scl::vectorize::sum(
            Array<const Real>(values.ptr, adj.primary_length(i)));
    });
}

// Use hash table instead of linear search in local_moving_phase
// Reference: leiden.hpp detail::HashTable with Fibonacci hashing
```

---


### Batch 13: components.hpp, metrics.hpp, outlier.hpp, sampling.hpp

#### 49. components.hpp
**Status**: Excellent optimization

**Current Optimizations**:
- Lock-free parallel Union-Find with path splitting (ParallelUnionFind)
- Sequential Union-Find with full path compression for single-threaded use
- BitVector for cache-efficient visited tracking with atomic_test_and_set
- Cache-line aligned data structures (alignas(64) for ParallelUnionFind, FastQueue)
- SIMD-accelerated sorted set intersection with adaptive algorithm selection:
  - Galloping search for highly skewed size ratios (ratio >= 32)
  - Linear merge for small arrays (< 16)
  - 8-way/4-way skip optimization for medium sizes
- Branchless comparison in intersection count
- FastQueue with batch push/pop and prefetch support
- Parallel BFS with direction-optimizing approach (bit-vector frontiers)
- Parallel triangle counting with sorted intersection
- K-core decomposition with bucket sort (O(n + m) algorithm)
- Parallel degree statistics with cache-aligned per-thread accumulators
- WorkspacePool for per-thread BFS buffers

**Best practice example for graph algorithms.**

**No optimization needed.**

---

#### 50. metrics.hpp
**Status**: Needs significant optimization

**Current Issues**:
- All loops are sequential, no parallelization
- No SIMD optimization for accumulations
- `silhouette_score`: Sequential loop over cells with nested cluster distance computation
- `adjusted_rand_index`, `normalized_mutual_information`: Sequential contingency table building
- `batch_entropy`, `lisi`: Sequential per-cell computation
- Uses `std::max`/`std::min` instead of `scl::algo::max2`/`min2`

**Optimization Opportunities**:
- [ ] Parallelize `silhouette_score` and `silhouette_samples` over cells
- [ ] Parallelize contingency table building for ARI/NMI
- [ ] Parallelize `batch_entropy` and `lisi` computations
- [ ] Use `scl::vectorize::sum` for accumulations
- [ ] Add WorkspacePool for per-thread cluster_dist_sum buffers
- [ ] Use atomic accumulation for thread-safe contingency table updates
- [ ] Add branch hints (`SCL_LIKELY`) for common paths

**Suggested Changes**:
```cpp
// Parallelize silhouette computation
template <typename T, bool IsCSR>
Real silhouette_score_parallel(...) {
    scl::threading::WorkspacePool<Real> workspace;
    workspace.init(n_threads, n_clusters);

    std::atomic<Real> total_silhouette{0};
    std::atomic<Size> valid_count{0};

    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i, size_t thread_rank) {
        Real* cluster_dist_sum = workspace.get(thread_rank);
        // ... compute silhouette for cell i
        // atomic add to total
    });
}

// Parallelize batch_entropy
void batch_entropy_parallel(...) {
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        // ... compute entropy for cell i
    });
}
```

---

#### 51. outlier.hpp
**Status**: Needs significant optimization

**Current Issues**:
- All computations are sequential (no parallelization with `scl::threading::parallel_for`)
- **CRITICAL**: Uses `std::sort` and `std::nth_element` instead of `scl::sort::sort` and `scl::algo::nth_element`
- `local_outlier_factor`: Sequential LOF computation, no SIMD (should use `scl::vectorize::*`)
- `ambient_detection`: Sequential ambient profile computation
- **CRITICAL**: `empty_drops`: Uses `std::sort` twice for p-value ordering (should use `scl::sort::sort_pairs`)
- `outlier_genes`: Two-pass sequential loops
- `doublet_score`: Nested sequential loops with O(n^2 * features) complexity
- `qc_filter`: Sequential per-cell QC check
- No branch hints or prefetch
- **MANDATORY**: Replace all `std::` functions with SCL custom operators

**Optimization Opportunities**:
- [ ] Parallelize `isolation_score`, `local_outlier_factor`, `ambient_detection` using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Replace `std::sort` with `scl::sort::sort` or `scl::sort::sort_pairs` (VQSort is 2-5x faster)
- [ ] **MANDATORY**: Replace `std::nth_element` with `scl::algo::nth_element` or `scl::algo::partial_sort`
- [ ] **MANDATORY**: Use `scl::vectorize::sum` for total UMI computation (replace manual sum loops)
- [ ] Add `scl::threading::WorkspacePool` for thread-local buffers
- [ ] Parallelize `empty_drops` deviance computation using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Add SIMD for distance/variance computations: use `scl::vectorize::dot`, `scl::vectorize::sum_squared`
- [ ] Add `SCL_LIKELY`/`SCL_UNLIKELY` branch hints
- [ ] **MANDATORY**: Replace all `std::memcpy`, `std::memset` with `scl::memory::copy_fast`, `scl::memory::zero`

**Suggested Changes**:
```cpp
// Parallelize LOF computation
template <typename T, bool IsCSR>
void local_outlier_factor_parallel(...) {
    // Parallel k-distance computation
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        // compute k-distance for cell i
    });

    // Parallel LRD computation
    scl::threading::WorkspacePool<Real> reach_pool;
    reach_pool.init(n_threads, k);

    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i, size_t tid) {
        Real* reach_dists = reach_pool.get(tid);
        // compute LRD for cell i
    });

    // Parallel LOF computation
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        // compute LOF for cell i
    });
}
```

---

#### 52. sampling.hpp
**Status**: Partially optimized

**Current Issues**:
- Uses simple LCG PRNG instead of Xoshiro256++ (lower quality randomness)
- No parallelization in any sampling function (should use `scl::threading::parallel_for`)
- **CRITICAL**: `geometric_sketching`: Sequential point processing and sorting (uses `std::sort` instead of `scl::sort::sort`)
- `density_preserving`: Sequential systematic sampling
- `landmark_selection`: Sequential KMeans++ initialization
- **CRITICAL**: `representative_cells`: Uses `std::partial_sort` instead of `scl::algo::partial_sort`
- `balanced_sampling`: Sequential group processing
- No SIMD optimization for distance computations (should use `scl::vectorize::dot` for squared distances)
- No `scl::threading::WorkspacePool` usage
- **MANDATORY**: Replace all `std::` functions with SCL custom operators

**Optimization Opportunities**:
- [ ] Replace LCG with Xoshiro256++ PRNG for better randomness (like hotspot.hpp)
- [ ] Parallelize `geometric_sketching` grid assignment and sorting using `scl::threading::parallel_for`
- [ ] Parallelize `kmeans_pp_init` distance updates (most expensive part) using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Use `scl::sort::sort` instead of `std::sort` for sorted indices (VQSort is 2-5x faster)
- [ ] **MANDATORY**: Use `scl::algo::partial_sort` for top-k selection (replace `std::partial_sort`)
- [ ] **MANDATORY**: Add SIMD for squared distance computation: use `scl::vectorize::dot` or `scl::vectorize::sum_squared` in KMeans++
- [ ] Parallelize `representative_cells` distance computations using `scl::threading::parallel_for`
- [ ] Add Lemire's nearly divisionless bounded random (like permutation.hpp)
- [ ] **MANDATORY**: Replace all `std::` functions with SCL custom operators

**Suggested Changes**:
```cpp
// Replace LCG with Xoshiro256++
struct Xoshiro256PP {
    uint64_t state[4];

    SCL_FORCE_INLINE uint64_t next() noexcept {
        const uint64_t result = rotl(state[0] + state[3], 23) + state[0];
        const uint64_t t = state[1] << 17;
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        state[2] ^= t;
        state[3] = rotl(state[3], 45);
        return result;
    }

    // Lemire's bounded random
    SCL_FORCE_INLINE Size bounded(Size n) noexcept {
        uint64_t x = next();
        __uint128_t m = (__uint128_t)x * (__uint128_t)n;
        uint64_t l = (uint64_t)m;
        if (l < n) {
            uint64_t t = -n % n;
            while (l < t) {
                x = next();
                m = (__uint128_t)x * (__uint128_t)n;
                l = (uint64_t)m;
            }
        }
        return (Size)(m >> 64);
    }
};

// Parallelize KMeans++ distance updates
template <typename T, bool IsCSR>
void kmeans_pp_init_parallel(...) {
    scl::threading::parallel_for(Size(0), n, [&](size_t i) {
        // Compute distance to last center
        Real dist = compute_sparse_distance(data, i, last_center);
        min_dist[i] = scl::algo::min2(min_dist[i], dist);  // MANDATORY: Replace std::min
    });
}
```

---

### Batch 14: Files 53-57 (entropy.hpp, state.hpp, comparison.hpp, association.hpp, alignment.hpp)

#### 53. entropy.hpp
**Status**: Partially optimized

**Current Issues**:
- `row_entropy`: Sequential loop over rows (no parallelization)
- `js_divergence`: Manual memory allocation with aligned_alloc/aligned_free (no RAII)
- `discretize_equal_frequency`: Uses insertion sort O(n^2) for sorted indices
- `mrmr_selection`: O(n_features^2 * n_samples) sequential nested loops
- `select_features_mi`: Uses insertion sort for descending score sort
- `adjusted_mi`: Large contingency table with sequential processing
- Uses simple `FastRNG` instead of Xoshiro256++ (though not heavily used)
- No SIMD for entropy/divergence computations
- No WorkspacePool for thread-local buffers

**Optimization Opportunities**:
- [ ] Parallelize `row_entropy` over rows
- [ ] Parallelize `mrmr_selection` feature relevance computation
- [ ] Replace insertion sort with `scl::sort::sort` in discretize_equal_frequency
- [ ] Replace insertion sort with `scl::algo::partial_sort` for top-k features
- [ ] Use SIMD for sum/log computations in entropy calculations
- [ ] Add WorkspacePool for binned_features allocation in mrmr_selection
- [ ] Parallelize joint_entropy histogram computation with atomic_fetch_add
- [ ] Add branch hints for sparsity checks

**Suggested Changes**:
```cpp
// Parallelize row entropy computation
template <typename T, bool IsCSR>
void row_entropy_parallel(
    const Sparse<T, IsCSR>& X,
    Array<Real> entropies,
    bool normalize,
    bool use_log2
) {
    const Index n = X.rows();
    scl::threading::parallel_for(Index(0), n, [&](Index i) {
        auto values = X.row_values(i);
        const Index len = X.row_length(i);

        if (len == 0) {
            entropies[i] = Real(0);
            return;
        }

        // Compute entropy for row i (vectorized)
        Real sum = scl::vectorize::sum(values, len);
        if (sum < config::EPSILON) {
            entropies[i] = Real(0);
            return;
        }

        Real H = Real(0);
        for (Index k = 0; k < len; ++k) {
            Real p = values[k] / sum;
            if (p > config::EPSILON) {
                // MANDATORY: For batch processing, use scl::simd::Log; for scalar, std::log is acceptable
                H -= use_log2 ? p * std::log2(p) : p * std::log(p);
            }
        }
        // MANDATORY: For batch processing, use scl::simd::Log; for scalar, std::log is acceptable
        entropies[i] = normalize ? H / std::log(Real(len)) : H;
    });
}

// Use scl::sort instead of insertion sort
template <typename T>
void discretize_equal_frequency_optimized(
    const T* values, Size n, Index n_bins, Index* binned
) {
    Index* sorted_idx = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    for (Size i = 0; i < n; ++i) sorted_idx[i] = static_cast<Index>(i);

    // Replace insertion sort with scl::sort
    scl::sort::sort_by_key(sorted_idx, sorted_idx + n,
        [&](Index a, Index b) { return values[a] < values[b]; });

    // Assign bins based on rank
    Size items_per_bin = (n + n_bins - 1) / n_bins;
    for (Size i = 0; i < n; ++i) {
        Index bin = static_cast<Index>(i / items_per_bin);
        binned[sorted_idx[i]] = scl::algo::min2(bin, static_cast<Index>(n_bins - 1));  // MANDATORY: Replace std::min
    }
    scl::memory::aligned_free(sorted_idx, SCL_ALIGNMENT);
}
```

---

#### 54. state.hpp
**Status**: Needs significant optimization

**Current Issues**:
- `compute_geneset_score`: Linear search O(row_length * n_genes) for each cell
- `differentiation_potential`: O(n_cells * n_genes * row_length) sequential triple nested loops
- `cell_cycle_score`: Sequential gene set scoring
- **CRITICAL**: `rank_transform`: Uses `std::sort` instead of `scl::sort::sort` (VQSort is 2-5x faster)
- `multi_signature_score`: Sequential column normalization
- All scoring functions are completely sequential (no parallelization with `scl::threading::parallel_for`)
- Gene lookup in sparse rows is linear search (no binary search - should use `scl::algo::lower_bound`)
- No SIMD for mean/variance computations (should use `scl::vectorize::sum`, `scl::vectorize::sum_squared`)
- No `scl::threading::WorkspacePool` usage
- **MANDATORY**: Replace all `std::` functions with SCL custom operators

**Optimization Opportunities**:
- [ ] Parallelize all scoring functions using `scl::threading::parallel_for` (`stemness_score`, `proliferation_score`, etc.)
- [ ] **MANDATORY**: Use binary search for gene lookup: `scl::algo::lower_bound` in sorted sparse row indices
- [ ] **MANDATORY**: Replace `std::sort` with `scl::sort::sort` in rank_transform (VQSort is 2-5x faster)
- [ ] Pre-build gene index map for O(1) lookup in gene sets
- [ ] **MANDATORY**: Use SIMD for zscore_normalize: `scl::vectorize::sum` for mean, `scl::vectorize::sum_squared` for variance
- [ ] Parallelize differentiation_potential with `scl::threading::WorkspacePool`
- [ ] Parallelize multi_signature_score column normalization using `scl::threading::parallel_for`
- [ ] Add `SCL_LIKELY` branch hints for common cases
- [ ] **MANDATORY**: Replace all `std::` functions with SCL custom operators

**Suggested Changes**:
```cpp
// Binary search for gene lookup (indices are sorted in CSR)
SCL_FORCE_INLINE Real find_gene_expression(
    const Index* col_indices,
    const Real* values,
    Index start, Index end,
    Index gene
) {
    // Binary search since CSR indices are sorted
    while (start < end) {
        Index mid = start + (end - start) / 2;
        if (col_indices[mid] < gene) {
            start = mid + 1;
        } else if (col_indices[mid] > gene) {
            end = mid;
        } else {
            return values[mid];
        }
    }
    return Real(0.0);
}

// Parallelize stemness_score
template <typename T, bool IsCSR>
void stemness_score_parallel(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> stemness_genes,
    Array<Real> scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());

    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        Real sum = Real(0.0);
        Size count = 0;
        const Index row_start = expression.row_indices()[i];
        const Index row_end = expression.row_indices()[i + 1];

        for (Size g = 0; g < stemness_genes.len; ++g) {
            Index gene = stemness_genes[g];
            // Use binary search for O(log n) lookup
            Real val = find_gene_expression(
                expression.col_indices(), expression.values(),
                row_start, row_end, gene);
            if (val > Real(0.0)) {
                sum += val;
                ++count;
            }
        }
        scores.ptr[i] = (count > 0) ? sum / Real(count) : Real(0.0);
    });

    // Parallel zscore normalization
    // ... (use parallel reduction for mean/variance)
}

// SIMD zscore normalization
SCL_FORCE_INLINE void zscore_normalize_simd(Real* values, Size n) {
    if (n < 2) return;

    // SIMD sum for mean
    Real sum = scl::vectorize::sum(values, n);
    Real mean = sum / Real(n);

    // SIMD variance computation
    Real var_sum = Real(0.0);
    constexpr size_t kLanes = scl::simd::Lanes<Real>();
    size_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        auto v = scl::simd::Load(values + i);
        auto diff = scl::simd::Sub(v, scl::simd::Set(mean));
        var_sum += scl::simd::SumOfLanes(scl::simd::Mul(diff, diff));
    }
    for (; i < n; ++i) {
        Real diff = values[i] - mean;
        var_sum += diff * diff;
    }

    Real std_dev = std::sqrt(var_sum / Real(n - 1));
    if (std_dev < config::EPSILON) {
        scl::algo::fill(values, n, Real(0.0));
        return;
    }

    // SIMD normalization
    Real inv_std = Real(1.0) / std_dev;
    scl::vectorize::transform(values, n, [=](Real x) {
        return (x - mean) * inv_std;
    });
}
```

---

#### 55. comparison.hpp
**Status**: Needs optimization

**Current Issues**:
- **CRITICAL**: `wilcoxon_pvalue`: Uses `std::sort` instead of `scl::sort::sort` or `scl::sort::sort_pairs` (VQSort is 2-5x faster)
- `composition_analysis`: Sequential chi-squared computation
- `abundance_test`: Sequential Fisher's test approximation
- **CRITICAL**: `differential_abundance`: Uses `std::sort` via Wilcoxon test (should use `scl::sort::sort_pairs`)
- `condition_response`: O(n_genes * n_cells * row_length) sequential
- `effect_size`, `glass_delta`, `hedges_g`: Sequential variance computation (should use `scl::vectorize::sum_squared`)
- No parallelization in any function (should use `scl::threading::parallel_for`)
- No SIMD for sum/variance computations (should use `scl::vectorize::sum`, `scl::vectorize::sum_squared`)
- Memory allocation inside loops (wilcoxon_pvalue allocates per call - should use `scl::threading::WorkspacePool`)
- **MANDATORY**: Replace all `std::` functions with SCL custom operators

**Optimization Opportunities**:
- [ ] Replace `std::sort` with `scl::sort::sort` in Wilcoxon test
- [ ] Parallelize `condition_response` over genes
- [ ] Parallelize `composition_analysis` per cell-type chi-squared tests
- [ ] Use SIMD for mean/variance computation in effect_size functions
- [ ] Pre-allocate buffers outside loops in differential_abundance
- [ ] Parallelize abundance_test Fisher's tests
- [ ] Add WorkspacePool for Wilcoxon test scratch space

**Suggested Changes**:
```cpp
// Parallelize condition_response over genes
template <typename T, bool IsCSR>
void condition_response_parallel(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> conditions,
    Real* response_scores,
    Real* p_values,
    Size n_genes
) {
    const Size n_cells = static_cast<Size>(expression.rows());

    Size n_cond0 = 0, n_cond1 = 0;
    for (Size i = 0; i < n_cells; ++i) {
        if (conditions.ptr[i] == 0) ++n_cond0;
        else if (conditions.ptr[i] == 1) ++n_cond1;
    }

    // Pre-allocate workspace per thread
    scl::threading::DualWorkspacePool<Real> pool;
    pool.init(n_threads, n_cond0, n_cond1);

    scl::threading::parallel_for(Size(0), n_genes, [&](size_t g, size_t tid) {
        Real* group0 = pool.get_first(tid);
        Real* group1 = pool.get_second(tid);

        Size idx0 = 0, idx1 = 0;
        for (Size c = 0; c < n_cells; ++c) {
            Real val = find_gene_expression(expression, c, g);
            if (conditions.ptr[c] == 0) group0[idx0++] = val;
            else group1[idx1++] = val;
        }

        // Compute log2 fold change and Wilcoxon test
        // ...
    });
}

// Use scl::sort in Wilcoxon test
SCL_FORCE_INLINE Real wilcoxon_pvalue_optimized(
    Real* group1, Size n1,
    Real* group2, Size n2,
    Real* combined,  // Pre-allocated scratch
    Index* indices,
    Real* ranks
) {
    Size n_total = n1 + n2;

    // Combine groups
    // MANDATORY: Replace std::memcpy with scl::algo::copy or scl::memory::copy_fast
    scl::algo::copy(group1, combined, n1);
    scl::algo::copy(group2, combined + n1, n2);

    for (Size i = 0; i < n_total; ++i) indices[i] = i;

    // Use scl::sort instead of std::sort
    scl::sort::sort_by_key(indices, indices + n_total,
        [&](Index a, Index b) { return combined[a] < combined[b]; });

    // Compute ranks and test statistic...
}
```

---

#### 56. association.hpp
**Status**: Needs significant optimization

**Current Issues**:
- `pearson_correlation`: Allocates O(n_cells) per call, extremely expensive for all-pairs (should use `scl::threading::WorkspacePool`)
- **CRITICAL**: `spearman_correlation`: Allocates 5 arrays per call, uses `std::sort` twice (should use `scl::sort::sort_pairs`)
- `gene_peak_correlation`: O(n_genes * n_peaks * n_cells) sequential triple nested loop
- `cis_regulatory`: Sequential correlation computation
- `enhancer_gene_link`: Sequential O(n_genes * n_peaks) loop
- **CRITICAL**: `multimodal_neighbors`: O(n_cells^2 * features) sequential, uses `std::partial_sort` (should use `scl::algo::partial_sort`)
- `feature_coupling`: O(n_features1 * n_features2 * n_cells) sequential
- `correlation_in_subset`: Allocates per call (should use `scl::threading::WorkspacePool`)
- No parallelization anywhere (should use `scl::threading::parallel_for`)
- No SIMD for correlation computations (should use `scl::vectorize::dot`, `scl::vectorize::sum`, `scl::vectorize::sum_squared`)
- **MANDATORY**: Replace all `std::` functions with SCL custom operators

**Optimization Opportunities**:
- [ ] Pre-extract dense feature vectors for batch correlation computation
- [ ] Parallelize gene_peak_correlation over gene-peak pairs using `scl::threading::parallel_for`
- [ ] Parallelize multimodal_neighbors over cells using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Replace `std::sort` with `scl::sort::sort_pairs` in Spearman correlation (VQSort is 2-5x faster)
- [ ] **MANDATORY**: Replace `std::partial_sort` with `scl::algo::partial_sort`
- [ ] **MANDATORY**: Use SIMD for Pearson correlation: `scl::vectorize::dot` for numerator, `scl::vectorize::sum`/`sum_squared` for denominator
- [ ] Add `scl::threading::WorkspacePool` for correlation scratch buffers
- [ ] Cache-block correlation matrix computation
- [ ] Add early termination for low-correlation pairs
- [ ] **MANDATORY**: Replace all `std::` functions with SCL custom operators

**Suggested Changes**:
```cpp
// Pre-extract feature vectors for batch processing
template <typename T, bool IsCSR>
void extract_feature_vectors(
    const Sparse<T, IsCSR>& data,
    Real* dense_features,  // [n_features * n_cells]
    Size n_cells,
    Size n_features
) {
    // Initialize to zero
    scl::algo::zero(dense_features, n_features * n_cells);

    // Parallel extraction
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t c) {
        const Index row_start = data.row_indices()[c];
        const Index row_end = data.row_indices()[c + 1];
        for (Index j = row_start; j < row_end; ++j) {
            Index col = data.col_indices()[j];
            dense_features[col * n_cells + c] = static_cast<Real>(data.values()[j]);
        }
    });
}

// SIMD Pearson correlation on pre-extracted vectors
SCL_FORCE_INLINE Real pearson_correlation_simd(
    const Real* x, const Real* y, Size n
) {
    Real sum_x = Real(0), sum_y = Real(0);
    Real sum_xx = Real(0), sum_yy = Real(0), sum_xy = Real(0);

    constexpr size_t kLanes = scl::simd::Lanes<Real>();
    size_t i = 0;

    // SIMD accumulation
    auto acc_x = scl::simd::Zero<Real>();
    auto acc_y = scl::simd::Zero<Real>();
    auto acc_xx = scl::simd::Zero<Real>();
    auto acc_yy = scl::simd::Zero<Real>();
    auto acc_xy = scl::simd::Zero<Real>();

    for (; i + kLanes <= n; i += kLanes) {
        auto vx = scl::simd::Load(x + i);
        auto vy = scl::simd::Load(y + i);
        acc_x = scl::simd::Add(acc_x, vx);
        acc_y = scl::simd::Add(acc_y, vy);
        acc_xx = scl::simd::MulAdd(vx, vx, acc_xx);
        acc_yy = scl::simd::MulAdd(vy, vy, acc_yy);
        acc_xy = scl::simd::MulAdd(vx, vy, acc_xy);
    }

    sum_x = scl::simd::SumOfLanes(acc_x);
    sum_y = scl::simd::SumOfLanes(acc_y);
    sum_xx = scl::simd::SumOfLanes(acc_xx);
    sum_yy = scl::simd::SumOfLanes(acc_yy);
    sum_xy = scl::simd::SumOfLanes(acc_xy);

    // Scalar tail
    for (; i < n; ++i) {
        sum_x += x[i]; sum_y += y[i];
        sum_xx += x[i] * x[i];
        sum_yy += y[i] * y[i];
        sum_xy += x[i] * y[i];
    }

    Real mean_x = sum_x / n, mean_y = sum_y / n;
    Real cov_xy = sum_xy / n - mean_x * mean_y;
    Real var_x = sum_xx / n - mean_x * mean_x;
    Real var_y = sum_yy / n - mean_y * mean_y;

    if (var_x < EPSILON || var_y < EPSILON) return Real(0);
    return cov_xy / (std::sqrt(var_x) * std::sqrt(var_y));
}

// Parallelize gene_peak_correlation
template <typename T, bool IsCSR>
void gene_peak_correlation_parallel(
    const Real* rna_dense,   // Pre-extracted [n_genes * n_cells]
    const Real* atac_dense,  // Pre-extracted [n_peaks * n_cells]
    Size n_genes, Size n_peaks, Size n_cells,
    Index* gene_indices, Index* peak_indices,
    Real* correlations, Size& n_correlations,
    Real min_correlation
) {
    std::atomic<Size> count{0};

    scl::threading::parallel_for(Size(0), n_genes * n_peaks, [&](size_t idx) {
        Size g = idx / n_peaks;
        Size p = idx % n_peaks;

        Real corr = pearson_correlation_simd(
            rna_dense + g * n_cells,
            atac_dense + p * n_cells,
            n_cells);

        if (std::abs(corr) >= min_correlation) {
            Size i = count.fetch_add(1, std::memory_order_relaxed);
            gene_indices[i] = g;
            peak_indices[i] = p;
            correlations[i] = corr;
        }
    });

    n_correlations = count.load();
}
```

---

#### 57. alignment.hpp
**Status**: Needs significant optimization

**Current Issues**:
- `find_cross_knn`: O(n1 * n2 * features) sequential, **CRITICAL**: uses `std::partial_sort` (should use `scl::algo::partial_sort`)
- `mnn_pairs`: Sequential MNN detection, no parallelization (should use `scl::threading::parallel_for`)
- `find_anchors`: Sequential anchor finding with O(k^2) shared neighbor check
- `transfer_labels`: Sequential label aggregation over anchors
- `integration_score`: Sequential batch entropy computation
- `batch_mixing`: Sequential per-cell mixing score
- `compute_correction_vectors`: Sequential dense conversion and accumulation
- `smooth_correction_vectors`: O(n2^2) Gaussian kernel smoothing, sequential
- `cca_projection`: Sequential random projection (simplified CCA)
- `kbet_score`: Sequential chi-squared test per cell
- **CRITICAL**: Uses `std::partial_sort` instead of `scl::algo::partial_sort`
- No SIMD for distance computations (should use `scl::vectorize::dot` for squared distances)
- **MANDATORY**: Replace all `std::` functions with SCL custom operators

**Optimization Opportunities**:
- [ ] Parallelize `find_cross_knn` over query points using `scl::threading::parallel_for`
- [ ] Parallelize `mnn_pairs` MNN detection over cell pairs using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Replace `std::partial_sort` with `scl::algo::partial_sort`
- [ ] **MANDATORY**: Use SIMD for `sparse_distance_squared`: `scl::vectorize::dot` or `scl::vectorize::sum_squared`
- [ ] Parallelize `integration_score` entropy computation using `scl::threading::parallel_for`
- [ ] Parallelize `batch_mixing` per-cell scores using `scl::threading::parallel_for`
- [ ] Parallelize `smooth_correction_vectors` with Gaussian kernel using `scl::threading::parallel_for`
- [ ] Add kd-tree or ball tree for approximate NN search (reduce O(n^2) to O(n log n))
- [ ] Parallelize `kbet_score` chi-squared tests using `scl::threading::parallel_for`
- [ ] Add `scl::threading::WorkspacePool` for distance/indices scratch buffers
- [ ] **MANDATORY**: Replace all `std::` functions with SCL custom operators

**Suggested Changes**:
```cpp
// Parallelize find_cross_knn
template <typename T, bool IsCSR>
void find_cross_knn_parallel(
    const Sparse<T, IsCSR>& data1,
    const Sparse<T, IsCSR>& data2,
    Index k,
    Index* knn_indices,
    Real* knn_distances
) {
    const Size n1 = static_cast<Size>(data1.rows());
    const Size n2 = static_cast<Size>(data2.rows());

        k = scl::algo::min2(k, static_cast<Index>(n2));  // MANDATORY: Replace std::min

    // Thread-local workspace for distances and indices
    scl::threading::DualWorkspacePool<Real, Index> pool;
    pool.init(n_threads, n2, n2);

    scl::threading::parallel_for(Size(0), n1, [&](size_t i, size_t tid) {
        Real* all_dists = pool.get_first(tid);
        Index* all_indices = pool.get_second(tid);

        // Compute distances to all points in data2
        for (Size j = 0; j < n2; ++j) {
            all_dists[j] = sparse_distance_squared_simd(
                data1, static_cast<Index>(i),
                data2, static_cast<Index>(j));
            all_indices[j] = static_cast<Index>(j);
        }

        // Use scl::algo::partial_sort for top-k
        scl::algo::partial_sort(all_indices, k, n2,
            [&](Index a, Index b) { return all_dists[a] < all_dists[b]; });

        // Store results
        for (Index ki = 0; ki < k; ++ki) {
            knn_indices[i * k + ki] = all_indices[ki];
            knn_distances[i * k + ki] = all_dists[all_indices[ki]];
        }
    });
}

// SIMD sparse distance squared
template <typename T, bool IsCSR>
SCL_FORCE_INLINE Real sparse_distance_squared_simd(
    const Sparse<T, IsCSR>& data1, Index row1,
    const Sparse<T, IsCSR>& data2, Index row2
) {
    Real dist = Real(0.0);

    const Index start1 = data1.row_indices()[row1];
    const Index end1 = data1.row_indices()[row1 + 1];
    const Index start2 = data2.row_indices()[row2];
    const Index end2 = data2.row_indices()[row2 + 1];

    Index i1 = start1, i2 = start2;

    // Process matching indices
    while (i1 < end1 && i2 < end2) {
        Index col1 = data1.col_indices()[i1];
        Index col2 = data2.col_indices()[i2];

        if (col1 == col2) {
            Real diff = static_cast<Real>(data1.values()[i1]) -
                       static_cast<Real>(data2.values()[i2]);
            dist += diff * diff;
            ++i1; ++i2;
        } else if (col1 < col2) {
            Real val = static_cast<Real>(data1.values()[i1]);
            dist += val * val;
            ++i1;
        } else {
            Real val = static_cast<Real>(data2.values()[i2]);
            dist += val * val;
            ++i2;
        }
    }

    // Process remaining (could vectorize these tails)
    while (i1 < end1) {
        Real val = static_cast<Real>(data1.values()[i1++]);
        dist += val * val;
    }
    while (i2 < end2) {
        Real val = static_cast<Real>(data2.values()[i2++]);
        dist += val * val;
    }

    return dist;
}

// Parallelize batch_mixing
template <bool IsCSR>
void batch_mixing_parallel(
    Array<const Index> batch_labels,
    const Sparse<Index, IsCSR>& neighbors,
    Array<Real> mixing_scores
) {
    const Size n_cells = static_cast<Size>(neighbors.rows());

    Index n_batches = 0;
    for (Size i = 0; i < n_cells; ++i) {
        n_batches = scl::algo::max2(n_batches, batch_labels.ptr[i] + 1);  // MANDATORY: Replace std::max
    }

    scl::threading::WorkspacePool<Size> pool;
    pool.init(n_threads, n_batches);

    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i, size_t tid) {
        Size* batch_counts = pool.get(tid);
        scl::algo::zero(batch_counts, n_batches);

        const Index row_start = neighbors.row_indices()[i];
        const Index row_end = neighbors.row_indices()[i + 1];
        Size n_neighbors = row_end - row_start;

        Index my_batch = batch_labels.ptr[i];
        for (Index j = row_start; j < row_end; ++j) {
            ++batch_counts[batch_labels.ptr[neighbors.col_indices()[j]]];
        }

        Size same_batch = batch_counts[my_batch];
        mixing_scores.ptr[i] = (n_neighbors > 0) ?
            Real(n_neighbors - same_batch) / Real(n_neighbors) : Real(0);
    });
}
```

---

### Batch 15: Files 58-61 (log1p.hpp, feature.hpp, mmd.hpp, bbknn.hpp)

#### 58. log1p.hpp
**Status**: Excellent optimization

**Current Strengths**:
- 4-way unrolled SIMD for log1p, log2p1, expm1 transforms
- Uses `SCL_PREFETCH_READ` with configurable distance
- `SCL_RESTRICT` for pointer aliasing hints
- Parallelized over primary dimension via `parallel_for`
- Scalar tail handling for non-aligned elements

**No Further Optimization Needed**: This file follows all design patterns correctly.

---

#### 59. feature.hpp
**Status**: Excellent optimization

**Current Strengths**:
- 4-way unrolled SIMD with dual accumulators (v_sum0/v_sum1, v_sq0/v_sq1)
- Uses `MulAdd` for fused multiply-add
- Prefetch with `SCL_LIKELY` branch hints
- Parallel statistics computation via `parallel_for`
- SIMD dispersion with masked division (`IfThenElse`)
- Clipped moments variant with SIMD `Min` operation

**No Further Optimization Needed**: This file follows all design patterns correctly.

---

#### 60. mmd.hpp
**Status**: Excellent optimization

**Current Strengths**:
- 8-way unrolled SIMD accumulation in `unary_exp_sum_ultra`
- Cache-blocked cross-kernel computation (64x512 blocks)
- Symmetric optimization for self-kernel (upper triangle only)
- `DualWorkspacePool` for thread-local cache buffers
- Prefetch in blocked loops
- `SCL_UNLIKELY` for early exit on empty vectors

**No Further Optimization Needed**: This file is a model implementation.

---

#### 61. bbknn.hpp
**Status**: Excellent optimization

**Current Strengths**:
- Custom fixed-size K-heap with manual sift operations
- 8-way/4-way skip optimization for sparse dot product
- O(1) range disjointness check before merge
- Batch-grouped processing for memory locality
- Prefetch hints in merge loop
- Pre-computed norm squares for distance pruning
- `SCL_LIKELY`/`SCL_UNLIKELY` branch hints

**No Further Optimization Needed**: This file is exceptionally well optimized.

---

### Batch 16: Files 62-65 (correlation.hpp, hvg.hpp, ttest.hpp, louvain.hpp)

#### 62. correlation.hpp
**Status**: Excellent optimization

**Current Strengths**:
- 4-way unrolled SIMD for sum/sq_sum computation
- 8-way/4-way skip optimization for sparse centered dot
- Efficient algebraic identity: cov = sum_ab - ma*sum_b - mb*sum_a + n*ma*mb
- Symmetric matrix optimization (upper triangle only, mirror)
- Cache-blocked processing (CHUNK_SIZE = 64)
- Zero-variance early exit optimization
- Prefetch in merge loops

**No Further Optimization Needed**: This file follows all design patterns correctly.

---

#### 63. hvg.hpp
**Status**: Excellent optimization

**Current Strengths**:
- 4-way unrolled SIMD dispersion computation with masked division
- Uses `scl::algo::partial_sort` for top-k selection
- Uses `scl::vectorize::sum` and `sum_squared`
- Parallel moment computation via `parallel_for`
- SIMD normalization with `IfThenElse` masking

**No Further Optimization Needed**: This file follows all design patterns correctly.

---

#### 64. ttest.hpp
**Status**: Partially optimized

**Current Issues**:
- `compute_group_stats`: Manual 4-way unrolling but not using SIMD intrinsics
- Group accumulation has potential race conditions (but per-feature parallelism avoids this)
- Uses atomic-style separate mean/var computation instead of single-pass

**Optimization Opportunities**:
- [ ] Use SIMD for group statistic accumulation
- [ ] Consider using `scl::vectorize::sum` for per-group sums
- [ ] Could benefit from WorkspacePool for group buffers

**Note**: The current implementation is correct and reasonably efficient. Minor improvements possible but not critical.

---

#### 65. louvain.hpp
**Status**: Needs significant optimization

**Current Issues**:
- `compute_total_weight`: Sequential loop (no parallelization, no SIMD)
- `compute_node_degrees`: Sequential loop over all nodes
- `local_moving_phase`: Completely sequential inner loop with O(n * max_neighbors) complexity
- `aggregate_graph`: Sequential with O(n * communities) nested iteration
- No parallelization in any hot path
- Neighbor community lookup uses linear search O(k) for each neighbor

**Optimization Opportunities**:
- [ ] Parallelize `compute_total_weight` with parallel reduction
- [ ] Parallelize `compute_node_degrees` over nodes
- [ ] Use hash map or BitVector for neighbor community lookup in local_moving_phase
- [ ] Parallelize independent node moves (with careful ordering or parallel-friendly relaxation)
- [ ] Use `scl::vectorize::sum` for degree computation
- [ ] Consider randomized node ordering for better parallelism

**Suggested Changes**:
```cpp
// Parallelize degree computation
template <typename T, bool IsCSR>
void compute_node_degrees_parallel(
    const Sparse<T, IsCSR>& adj,
    Real* degrees
) {
    const Index n = adj.primary_dim();

    scl::threading::parallel_for(Index(0), n, [&](Index i) {
        auto values = adj.primary_values(i);
        degrees[i] = scl::vectorize::sum(Array<const T>(values.data(), values.size()));
    });
}

// Parallel total weight with reduction
template <typename T, bool IsCSR>
Real compute_total_weight_parallel(const Sparse<T, IsCSR>& adj) {
    const Index n = adj.primary_dim();

    std::atomic<Real> total{Real(0)};

    scl::threading::parallel_for(Index(0), n, [&](Index i) {
        auto values = adj.primary_values(i);
        Real local = scl::vectorize::sum(Array<const T>(values.data(), values.size()));
        // Atomic accumulation (or use thread-local reduction)
        Real expected = total.load(std::memory_order_relaxed);
        while (!total.compare_exchange_weak(expected, expected + local,
                                            std::memory_order_relaxed));
    });

    return total.load() / Real(2);
}

// Use hash table for neighbor community lookup
struct FastCommunityMap {
    static constexpr Size HASH_SIZE = 1024;
    Index keys[HASH_SIZE];
    Real values[HASH_SIZE];
    Size count;

    void clear() { 
        count = 0; 
        // MANDATORY: Replace std::memset with scl::algo::fill or scl::memory::fill
        scl::algo::fill(keys, sizeof(keys) / sizeof(keys[0]), static_cast<Index>(-1));
    }

    SCL_FORCE_INLINE Size hash(Index k) const {
        return static_cast<Size>(k) & (HASH_SIZE - 1);
    }

    SCL_FORCE_INLINE void insert_or_add(Index comm, Real weight) {
        Size h = hash(comm);
        while (keys[h] != -1 && keys[h] != comm) {
            h = (h + 1) & (HASH_SIZE - 1);
        }
        if (keys[h] == -1) {
            keys[h] = comm;
            values[h] = weight;
            count++;
        } else {
            values[h] += weight;
        }
    }
};
```

---


### Batch 17: Files 66-70 (subpopulation.hpp, clonotype.hpp, lineage.hpp, spatial_pattern.hpp, tissue.hpp)

#### 66. subpopulation.hpp
**Status**: Needs significant optimization

**Current Issues**:
- `subpopulation_clusters`: Sequential K-means, no parallelization (should use `scl::threading::parallel_for`)
- **CRITICAL**: Uses `std::sort` instead of `scl::sort::sort` for distance sorting (VQSort is 2-5x faster)
- Uses simple LCG RNG instead of Xoshiro256++ (lower quality randomness)
- `subpopulation_markers`: O(n_cells * n_genes) sequential, no SIMD (should use `scl::vectorize::sum`)
- `subpopulation_trajectory`: Sequential pseudotime computation, no parallelization
- `subpopulation_composition`: Sequential composition counting (should use parallel histogram)
- No `scl::threading::WorkspacePool` usage for thread-local buffers
- No SIMD for distance computations (should use `scl::vectorize::dot` or `scl::vectorize::sum_squared`)
- **MANDATORY**: Replace all `std::` functions with SCL custom operators

**Optimization Opportunities**:
- [ ] **MANDATORY**: Replace `std::sort` with `scl::sort::sort` or `scl::sort::sort_pairs` (VQSort, 2-5x faster)
- [ ] **MANDATORY**: Replace LCG with Xoshiro256++ PRNG (like hotspot.hpp, permutation.hpp)
- [ ] **MANDATORY**: Parallelize K-means assignment and update steps using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Add SIMD for squared distance computation: use `scl::vectorize::dot` or `scl::vectorize::sum_squared`
- [ ] **MANDATORY**: Parallelize marker score computation over genes using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Add `scl::threading::WorkspacePool` for thread-local buffers (avoid allocations in hot paths)
- [ ] Use parallel reduction for centroid updates (thread-local accumulation + atomic merge)
- [ ] **MANDATORY**: Replace any `std::memcpy`, `std::memset` with `scl::memory::copy_fast`, `scl::memory::zero`

---

#### 67. clonotype.hpp
**Status**: Needs significant optimization

**Current Issues**:
- **CRITICAL**: `gini_coefficient`: Uses `std::sort` instead of `scl::sort::sort` (VQSort is 2-5x faster)
- `shannon_entropy`: Sequential entropy computation, no SIMD log (should use `scl::simd::Log` for batch processing)
- `clonal_expansion`: Sequential expansion scoring (should use `scl::threading::parallel_for`)
- `clone_diversity`: Sequential diversity metrics (should use parallel computation)
- **CRITICAL**: `median_clone_size`: Uses `std::nth_element` instead of `scl::algo::nth_element`
- Uses simple LCG RNG instead of Xoshiro256++ (lower quality randomness)
- No parallelization in any function (should use `scl::threading::parallel_for`)
- No `scl::threading::WorkspacePool` usage
- **MANDATORY**: Replace all `std::` functions with SCL custom operators

**Optimization Opportunities**:
- [ ] **MANDATORY**: Replace `std::sort` with `scl::sort::sort` in gini_coefficient (VQSort, 2-5x faster)
- [ ] **MANDATORY**: Replace `std::nth_element` with `scl::algo::nth_element` in median_clone_size
- [ ] **MANDATORY**: Parallelize per-cell clone statistics using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Add SIMD for entropy computation: use `scl::simd::Log` for batch log processing
- [ ] **MANDATORY**: Replace LCG with Xoshiro256++ PRNG (like hotspot.hpp, permutation.hpp)
- [ ] **MANDATORY**: Add `scl::threading::WorkspacePool` for thread-local clone count buffers
- [ ] Use `scl::vectorize::sum` for clone size accumulation (replace manual sum loops)
- [ ] **MANDATORY**: Replace any `std::memcpy`, `std::memset` with `scl::memory::copy_fast`, `scl::memory::zero`

---

#### 68. lineage.hpp
**Status**: Needs significant optimization

**Current Issues**:
- `union_find`: Sequential Union-Find, no parallel path compression (should use components.hpp ParallelUnionFind)
- **CRITICAL**: `build_lineage_tree`: Uses `std::sort` for edge sorting (should use `scl::sort::sort_pairs`, VQSort is 2-5x faster)
- `find_lca`: O(n * depth) LCA without binary lifting optimization (should implement binary lifting for O(log n))
- `lineage_distance`: O(n^2 * depth) sequential pairwise distances (should use `scl::threading::parallel_for`)
- `lineage_entropy`: Sequential entropy computation (should use `scl::simd::Log` for batch processing)
- No parallelization in any function (should use `scl::threading::parallel_for`)
- No SIMD optimizations (should use `scl::vectorize::*` where applicable)
- Uses simple depth-first traversal for tree operations
- **MANDATORY**: Replace all `std::` functions with SCL custom operators

**Optimization Opportunities**:
- [ ] **MANDATORY**: Replace `std::sort` with `scl::sort::sort_pairs` for edge sorting (VQSort, 2-5x faster)
- [ ] Implement binary lifting for O(log n) LCA queries (replace O(n * depth) with O(log n))
- [ ] **MANDATORY**: Parallelize lineage_distance pairwise computations using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Parallelize tree depth/size computations using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Add `scl::threading::WorkspacePool` for tree traversal buffers
- [ ] **MANDATORY**: Use lock-free parallel Union-Find (like components.hpp ParallelUnionFind)
- [ ] Use `scl::vectorize::sum` for distance accumulation (replace manual sum loops)
- [ ] **MANDATORY**: Replace any `std::memcpy`, `std::memset` with `scl::memory::copy_fast`, `scl::memory::zero`

---

#### 69. spatial_pattern.hpp
**Status**: Needs significant optimization

**Current Issues**:
- `spatial_weights`: O(n^2) brute-force neighbor search, no spatial indexing (should use KD-tree for O(n log n))
- `morans_i`: Sequential Moran's I computation, no parallelization (should use `scl::threading::parallel_for`)
- `gearys_c`: Sequential Geary's C, same issues as Moran's I (should use `scl::threading::parallel_for`)
- **CRITICAL**: Uses `std::partial_sort` instead of `scl::algo::partial_sort`
- `spatial_lag`: Sequential weighted neighbor sum (should use `scl::vectorize::sum` and parallelization)
- `local_indicators`: Sequential LISA computation (should use `scl::threading::parallel_for`)
- `hotspot_analysis`: Sequential G* statistic (should use `scl::threading::parallel_for`)
- Uses simple LCG RNG instead of Xoshiro256++ (lower quality randomness)
- No SIMD for distance computations (should use `scl::vectorize::sum_squared`)
- No `scl::threading::WorkspacePool` usage
- **MANDATORY**: Replace all `std::` functions with SCL custom operators

**Optimization Opportunities**:
- [ ] **MANDATORY**: Replace `std::partial_sort` with `scl::algo::partial_sort`
- [ ] **MANDATORY**: Parallelize per-cell spatial lag computation using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Parallelize Moran's I, Geary's C over cells using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Add SIMD for distance computation: use `scl::vectorize::sum_squared` for Euclidean distance
- [ ] Consider KD-tree or Ball-tree for O(n log n) neighbor search (replace O(n^2) brute-force)
- [ ] **MANDATORY**: Replace LCG with Xoshiro256++ PRNG (like hotspot.hpp, permutation.hpp)
- [ ] **MANDATORY**: Add `scl::threading::WorkspacePool` for neighbor buffers
- [ ] **MANDATORY**: Parallelize permutation tests with thread-local RNG using `scl::threading::parallel_for`
- [ ] Use `scl::vectorize::sum` for weighted neighbor sum (replace manual sum loops)
- [ ] **MANDATORY**: Replace any `std::memcpy`, `std::memset` with `scl::memory::copy_fast`, `scl::memory::zero`

---

#### 70. tissue.hpp
**Status**: Needs significant optimization

**Current Issues**:
- `find_knn`: O(n^2) brute-force KNN, no spatial indexing (should use KD-tree for O(n log n))
- **CRITICAL**: Uses `std::partial_sort` instead of `scl::algo::partial_sort` (in find_knn and tissue_module)
- `tissue_architecture`: Sequential per-cell computation, O(n * k^2) clustering coefficient (should use `scl::threading::parallel_for`)
- `layer_assignment`: Sequential layer assignment (should use `scl::threading::parallel_for`)
- `zonation_score`: Sequential correlation computation (should use `scl::vectorize::dot` and parallelization)
- `morphological_features`: O(n^2) convex hull, sequential processing (should use parallel algorithm)
- `tissue_module`: Sequential K-means, **CRITICAL**: uses `std::partial_sort` (should use `scl::algo::partial_sort`)
- `neighborhood_composition`: Sequential composition counting (should use parallel histogram)
- `cell_type_interaction`: O(n^2) pairwise distance computation (should use `scl::threading::parallel_for`)
- Uses simple LCG RNG instead of Xoshiro256++ (lower quality randomness)
- No parallelization in any function (should use `scl::threading::parallel_for`)
- No SIMD for distance/statistics computations (should use `scl::vectorize::*`)
- No `scl::threading::WorkspacePool` usage
- **MANDATORY**: Replace all `std::` functions with SCL custom operators

**Optimization Opportunities**:
- [ ] **MANDATORY**: Replace `std::partial_sort` with `scl::algo::partial_sort` (in find_knn and tissue_module)
- [ ] Implement KD-tree or Ball-tree for O(n log n) neighbor search (replace O(n^2) brute-force)
- [ ] **MANDATORY**: Parallelize tissue_architecture over cells using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Parallelize K-means in tissue_module (assignment and update steps) using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Add SIMD for Euclidean distance: use `scl::vectorize::sum_squared`
- [ ] **MANDATORY**: Parallelize morphological_features over groups using `scl::threading::parallel_for`
- [ ] **MANDATORY**: Parallelize cell_type_interaction using parallel histogram with `scl::threading::parallel_for`
- [ ] **MANDATORY**: Replace LCG with Xoshiro256++ PRNG (like hotspot.hpp, permutation.hpp)
- [ ] **MANDATORY**: Add `scl::threading::WorkspacePool` for KNN/clustering buffers
- [ ] Use parallel convex hull algorithm for large point sets
- [ ] Use `scl::vectorize::dot` for correlation computation (replace manual dot product loops)
- [ ] **MANDATORY**: Replace any `std::memcpy`, `std::memset` with `scl::memory::copy_fast`, `scl::memory::zero`

---

## Summary

### Files Analyzed: 70 total

### Optimization Status Distribution:

| Status | Count | Files |
|--------|-------|-------|
| **Excellent (No optimization needed)** | ~15 | softmax.hpp, mmd.hpp, bbknn.hpp, correlation.hpp, hvg.hpp, neighbors.hpp, centrality.hpp, communication.hpp, diffusion.hpp, gnn.hpp, grn.hpp, hotspot.hpp, impute.hpp, leiden.hpp, permutation.hpp, projection.hpp, propagation.hpp, pseudotime.hpp, scoring.hpp, transition.hpp, velocity.hpp, components.hpp |
| **Well-optimized (Minor issues)** | ~20 | log1p.hpp, qc.hpp, normalize.hpp, algebra.hpp, feature.hpp, sparse.hpp, scale.hpp, mwu.hpp, kernel.hpp, markers.hpp, enrichment.hpp, coexpression.hpp, merge.hpp, slice.hpp, reorder.hpp, ttest.hpp, group.hpp, entropy.hpp |
| **Needs significant optimization** | ~35 | annotation.hpp, doublet.hpp, multiple_testing.hpp, louvain.hpp, metrics.hpp, outlier.hpp, sampling.hpp, state.hpp, comparison.hpp, association.hpp, alignment.hpp, subpopulation.hpp, clonotype.hpp, lineage.hpp, spatial_pattern.hpp, tissue.hpp |
| **Stub files (TODO)** | 2 | sparse_opt.hpp, niche.hpp |

### Common Optimization Patterns Missing (MANDATORY Fixes):

1. **MANDATORY - Sorting**: Many files use `std::sort` instead of `scl::sort::sort` or `scl::sort::sort_pairs` (VQSort is 2-5x faster, SIMD-optimized, parallel)
2. **MANDATORY - Parallelization**: Many files have sequential loops that MUST use `scl::threading::parallel_for`
3. **MANDATORY - SIMD**: Missing `scl::vectorize::sum`, `dot`, `scale`, `sum_squared` for vectorizable operations (replace manual loops)
4. **MANDATORY - RNG**: Simple LCG instead of Xoshiro256++ with Lemire bounded random (like hotspot.hpp, permutation.hpp)
5. **MANDATORY - Workspace**: Missing `scl::threading::WorkspacePool` for thread-local buffers (avoid allocations in hot paths)
6. **MANDATORY - Memory**: Using `std::memcpy`, `std::memset` instead of `scl::memory::copy_fast`, `scl::memory::zero` or `scl::algo::copy`, `scl::algo::zero`
7. **MANDATORY - Math**: Using `std::exp`, `std::log` in loops instead of `scl::simd::Exp`, `scl::simd::Log` for batch processing
8. **MANDATORY - Selection**: Using `std::partial_sort`, `std::nth_element` instead of `scl::algo::partial_sort`, `scl::algo::nth_element`
9. **Spatial indexing**: O(n^2) brute-force instead of KD-tree for neighbor search (performance optimization)

### Priority Files for Optimization:

1. **multiple_testing.hpp** - Critical path, uses std::sort extensively
2. **doublet.hpp** - O(n^2) complexity, no parallelization
3. **metrics.hpp** - Silhouette/ARI/NMI all sequential
4. **outlier.hpp** - Uses std::sort/std::nth_element
5. **sampling.hpp** - Sequential sampling, LCG RNG
6. **association.hpp** - O(n^2) correlation, std::sort in Spearman
7. **alignment.hpp** - O(n^2) KNN, std::partial_sort
8. **tissue.hpp** - O(n^2) KNN, sequential K-means
9. **spatial_pattern.hpp** - O(n^2) spatial weights
10. **louvain.hpp** - Sequential compared to leiden.hpp

