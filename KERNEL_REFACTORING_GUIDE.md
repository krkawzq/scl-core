# Kernel Refactoring Guide

## Task Overview

Systematically refactor all kernel files in `scl/kernel/` to use the new concept-based interface system, ensuring optimal performance through SIMD, concurrency, and proper abstraction layers.

---

## Background: New Concept System

### 1. Concept Hierarchy

```
ArrayLike (1D arrays)
  - value_type
  - data() -> const T*
  - size() -> Size
  - operator[]
  - begin()/end()
  
  Implementations:
    - MappedArray<T>
    - std::vector<T>
    - std::deque<T>
    - Span<T>

DenseLike (2D matrices)
  - ValueType, Tag = TagDense
  - rows(), cols()
  - ptr() -> const T*
  - operator()(i, j)
  
  Implementations:
    - DenseArray<T>
    - DenseDeque<T>

SparseLike<M, IsCSR> (Sparse matrices)
  - ValueType, Tag = TagSparse<IsCSR>
  - rows(), cols(), nnz()
  - row_values(i) / col_values(j)
  - row_indices(i) / col_indices(j)
  - row_length(i) / col_length(i)
  
  Implementations:
    - CustomSparse<T, IsCSR>
    - VirtualSparse<T, IsCSR>
    - MappedCustomSparse<T>  (memory-mapped)
    - MappedVirtualSparse<T> (memory-mapped)
```

### 2. Type Aliases

**OLD (deprecated)**:
- `CSRLike` / `CSCLike`
- `TagCSR` / `TagCSC`
- `MountMatrix` / `VirtualMountMatrix`

**NEW (required)**:
- `SparseLike<M, true>` (for CSR)
- `SparseLike<M, false>` (for CSC)
- `TagSparse<true>` / `TagSparse<false>`
- `MappedCustomSparse` / `MappedVirtualSparse`

### 3. Unified Accessors

**ALWAYS use these instead of direct member access**:
```cpp
scl::rows(matrix)           // instead of matrix.rows
scl::cols(matrix)           // instead of matrix.cols
scl::nnz(matrix)            // instead of matrix.nnz
scl::primary_values(mat, i) // row_values for CSR, col_values for CSC
scl::primary_indices(mat, i)
scl::primary_length(mat, i)
```

---

## Refactoring Requirements

### 1. Concept Usage (CRITICAL)

**Replace ALL occurrences**:
```cpp
// OLD:
template <CSCLike MatrixT>
void algorithm(const MatrixT& matrix) { ... }

// NEW:
template <typename MatrixT>
    requires SparseLike<MatrixT, false>
void algorithm(const MatrixT& matrix) { ... }
```

**Replace ALL occurrences**:
```cpp
// OLD:
template <CSRLike MatrixT>
void algorithm(const MatrixT& matrix) { ... }

// NEW:
template <typename MatrixT>
    requires SparseLike<MatrixT, true>
void algorithm(const MatrixT& matrix) { ... }
```

### 2. Implementation Layering (REQUIRED)

Each algorithm MUST provide implementations at these levels:

#### Level 1: ISparse Virtual Interface (Base Implementation)
```cpp
/// @brief Algorithm description (Virtual Interface, CSC).
///
/// Generic implementation using ISparse base class.
/// Works with any sparse matrix type but has virtual call overhead.
template <typename T>
void algorithm(const ICSC<T>& matrix, ...) {
    // Use matrix.primary_values(i), matrix.primary_indices(i)
    // Works for ALL SparseLike types via virtual dispatch
}

/// @brief Algorithm description (Virtual Interface, CSR).
template <typename T>
void algorithm(const ICSR<T>& matrix, ...) {
    // CSR version
}
```

#### Level 2: Concept-Based Optimized (High-Performance)
```cpp
/// @brief Algorithm description (Concept-based, Optimized, CSC).
///
/// High-performance implementation for SparseLike<false> matrices.
/// Uses unified accessors for zero-overhead abstraction.
template <typename MatrixT>
    requires SparseLike<MatrixT, false>
void algorithm(const MatrixT& matrix, ...) {
    // Use scl::rows(), scl::cols(), scl::primary_values()
    // Works for CustomSparse, VirtualSparse, MappedCustomSparse
    // SIMD + Parallel optimizations
}

/// @brief Algorithm description (Concept-based, Optimized, CSR).
template <typename MatrixT>
    requires SparseLike<MatrixT, true>
void algorithm(const MatrixT& matrix, ...) {
    // CSR version
}
```

#### Level 3: Custom/Virtual Specializations (ONLY IF NEEDED)
```cpp
// ONLY create separate Custom/Virtual versions if:
// 1. Unified implementation causes >10% performance loss
// 2. Memory access patterns fundamentally differ
// 3. Algorithm structure requires it

/// @brief Algorithm for Custom pattern (contiguous arrays).
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void algorithm_custom(const CustomSparse<T, IsCSR>& matrix, ...) {
    // Can use matrix.data, matrix.indices, matrix.indptr directly
    // Batch SIMD operations on entire arrays
}

/// @brief Algorithm for Virtual pattern (pointer arrays).
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void algorithm_virtual(const VirtualSparse<T, IsCSR>& matrix, ...) {
    // Use matrix.data_ptrs, matrix.indices_ptrs, matrix.lengths
    // Per-row processing pattern
}
```

### 3. Ignore Mapped Types (IMPORTANT)

**DO NOT create special implementations for MappedCustomSparse/MappedVirtualSparse**.

Rationale:
- Memory-mapped data has IO overhead >> virtual function overhead
- They use the fallback implementations (ISparse or SparseLike)
- Future cache-aware optimizations will be added separately

### 4. Performance Requirements (MANDATORY)

#### A. SIMD Optimization
```cpp
namespace s = scl::simd;
const s::Tag d;
const size_t lanes = s::lanes();

auto v_sum = s::Zero(d);
for (size_t k = 0; k + lanes <= n; k += lanes) {
    auto v = s::Load(d, data + k);
    v_sum = s::Add(v_sum, v);
}
Real sum = s::GetLane(s::SumOfLanes(d, v_sum));

// Scalar tail
for (; k < n; ++k) {
    sum += data[k];
}
```

#### B. Parallel Processing
```cpp
scl::threading::parallel_for(0, static_cast<size_t>(n), [&](size_t i) {
    // Parallel work
});
```

#### C. Use Core High-Performance Utilities
- `scl::sort::sort_pairs()` - SIMD-optimized VQSort
- `scl::sort::argsort_inplace()` - Vectorized argsort
- `scl::sort::partition()` - SIMD partitioning
- Memory operations in `scl/core/` modules

### 5. Code Style (From AGENT.md)

#### Documentation Format
- Use Doxygen triple-slash (`///`)
- **STRICTLY NO Markdown or LaTeX syntax**
- Write formulas in plain text: `sigma^2 = sum((x - mu)^2) / (n - 1)`
- English only

#### Naming Conventions
```cpp
// Functions: snake_case
void compute_statistics(...);

// Classes: PascalCase
class MappedCustomSparse;

// Templates: PascalCase
template <typename MatrixT>

// Variables: snake_case
const Index num_rows = ...;

// Constants: UPPER_SNAKE_CASE
constexpr Size CHUNK_SIZE = 4096;
```

#### Namespace Organization
```cpp
namespace scl::kernel::module_name {

namespace detail {
    // Internal helpers
}

// Public API

} // namespace scl::kernel::module_name
```

---

## Files to Refactor

### Already Completed (No Action Needed)
- [x] `spatial.hpp`
- [x] `ttest.hpp`
- [x] `sparse.hpp`
- [x] `softmax.hpp`
- [x] `scale.hpp`
- [x] `qc.hpp`
- [x] `resample.hpp`
- [x] `reorder.hpp`
- [x] `feature.hpp`
- [x] `neighbors.hpp`
- [x] `merge.hpp`
- [x] `correlation.hpp`
- [x] `hvg.hpp`

### Pending Refactoring

#### High Priority
1. **`group.hpp`** - Group aggregation kernels
   - CSRLike/CSCLike count: 12
   - Patterns: Group statistics (mean, sum, variance)
   
2. **`mmd.hpp`** - Maximum Mean Discrepancy
   - CSRLike/CSCLike count: 2
   - Patterns: Distance-based statistics

3. **`mwu.hpp`** - Mann-Whitney U Test
   - CSRLike/CSCLike count: 2
   - Patterns: Rank-based statistical tests

4. **`normalize.hpp`** - Normalization kernels
   - CSRLike/CSCLike count: 6
   - Patterns: Scaling, centering, standardization

#### Medium Priority
5. **`algebra.hpp`** - Matrix algebra operations
   - CSRLike/CSCLike count: 6
   - Patterns: Add, subtract, multiply

6. **`bbknn.hpp`** - Batch-balanced KNN
   - CSRLike/CSCLike count: 4
   - Patterns: Neighbor search with batch correction

#### Low Priority (No CSRLike/CSCLike usage)
7. **`log1p.hpp`** - Logarithm transformations
8. **`gram.hpp`** - Gram matrix computation

---

## Detailed Refactoring Steps

For EACH file, perform these steps:

### Step 1: Update Concept Constraints

Search for:
```cpp
template <CSCLike MatrixT>
template <CSRLike MatrixT>
```

Replace with:
```cpp
template <typename MatrixT>
    requires SparseLike<MatrixT, false>  // for CSC
    
template <typename MatrixT>
    requires SparseLike<MatrixT, true>   // for CSR
```

### Step 2: Replace Direct Member Access

Search and replace:
```cpp
matrix.rows  -> scl::rows(matrix)
matrix.cols  -> scl::cols(matrix)
matrix.nnz   -> scl::nnz(matrix)
matrix.data  -> (keep if CustomSparse, use accessors otherwise)
matrix.indices -> (keep if CustomSparse, use accessors otherwise)
matrix.indptr -> (keep if CustomSparse, use accessors otherwise)
```

### Step 3: Update Tag Checks

Search for:
```cpp
if constexpr (std::is_same_v<Tag, TagCSR>)
if constexpr (std::is_same_v<Tag, TagCSC>)
```

Replace with:
```cpp
if constexpr (tag_is_csr_v<typename MatrixT::Tag>)
if constexpr (!tag_is_csr_v<typename MatrixT::Tag>)
```

### Step 4: Verify Implementation Layers

Check that EACH algorithm has:
1. ISparse-based version (virtual interface)
2. SparseLike concept-based version (optimized)
3. Custom/Virtual specializations (ONLY if performance requires)

### Step 5: Optimize Performance

Ensure ALL hot loops use:
- SIMD vectorization (scl::simd)
- Parallel processing (scl::threading::parallel_for)
- Core high-performance utilities (scl::sort, scl::argsort)

### Step 6: Update Documentation

- Fix comment style (no Markdown/LaTeX)
- Update function descriptions
- Ensure all parameters documented
- Add performance notes

### Step 7: Verify Compilation

```bash
# Check for errors
grep -r "CSRLike\|CSCLike" scl/kernel/filename.hpp
# Should return: 0 matches

# Compile check (if build system available)
# Build test
```

---

## Common Patterns and Solutions

### Pattern 1: Simple Column/Row Iteration

**Before**:
```cpp
template <CSCLike MatrixT>
void algo(const MatrixT& matrix) {
    const Index n = matrix.cols;
    for (Index j = 0; j < n; ++j) {
        auto vals = matrix.col_values(j);
        // process
    }
}
```

**After**:
```cpp
template <typename MatrixT>
    requires SparseLike<MatrixT, false>
void algo(const MatrixT& matrix) {
    const Index n = scl::cols(matrix);
    for (Index j = 0; j < n; ++j) {
        auto vals = scl::primary_values(matrix, j);
        // process
    }
}
```

### Pattern 2: Generic Tag Dispatch

**Before**:
```cpp
template <typename MatrixT>
void algo_impl(const MatrixT& matrix) {
    using Tag = typename MatrixT::Tag;
    if constexpr (std::is_same_v<Tag, TagCSC>) {
        // CSC path
    } else if constexpr (std::is_same_v<Tag, TagCSR>) {
        // CSR path
    }
}
```

**After**:
```cpp
template <AnySparse MatrixT>
void algo_impl(const MatrixT& matrix) {
    if constexpr (!tag_is_csr_v<typename MatrixT::Tag>) {
        // CSC path
    } else {
        // CSR path
    }
}
```

### Pattern 3: Direct Data Access (CustomSparse only)

**When you need direct pointer access**:
```cpp
// Only for CustomSparseLike types (contiguous arrays)
template <typename T, bool IsCSR>
void custom_algo(CustomSparse<T, IsCSR>& matrix) {
    // Can access directly
    T* vals = matrix.data;
    Index* idxs = matrix.indices;
    Index* ptrs = matrix.indptr;
    
    // Batch SIMD on entire array
    const Index nnz_val = scl::nnz(matrix);
    for (Index k = 0; k < nnz_val; ++k) {
        vals[k] *= 2.0;  // Example
    }
}
```

### Pattern 4: Virtual Interface for Polymorphism

```cpp
/// @brief Algorithm using virtual base class.
///
/// Works with any sparse matrix but has virtual call overhead.
/// Use concept-based overloads for performance-critical paths.
template <typename T>
void algorithm(const ICSC<T>& matrix, ...) {
    const Index n = matrix.cols();
    scl::threading::parallel_for(0, static_cast<size_t>(n), [&](size_t j) {
        auto vals = matrix.primary_values(static_cast<Index>(j));
        // Process
    });
}
```

---

## File-Specific Notes

### `group.hpp` - Group Aggregation

**Current Issues**:
- Uses `CSCLike`/`CSRLike` (12 occurrences)
- Direct member access: `matrix.cols()`, `matrix.rows()`

**Required Changes**:
1. Replace all `CSCLike` -> `SparseLike<MatrixT, false>`
2. Replace all `CSRLike` -> `SparseLike<MatrixT, true>`
3. Update all `matrix.cols()` -> `scl::cols(matrix)`
4. Update all `matrix.rows()` -> `scl::rows(matrix)`
5. Ensure ISparse implementations exist for all public APIs

**Performance Critical**:
- `group_mean()` - Parallel over features, SIMD accumulation
- `group_sum()` - Parallel over features
- `group_variance()` - Two-pass algorithm, SIMD

### `mmd.hpp` - Maximum Mean Discrepancy

**Current Issues**:
- Uses `CSCLike`/`CSRLike` (2 occurrences)
- Implements kernel distance computations

**Required Changes**:
1. Replace concepts
2. Add ISparse implementations if missing
3. Optimize kernel matrix computation with SIMD

### `mwu.hpp` - Mann-Whitney U Test

**Current Issues**:
- Uses `CSCLike`/`CSRLike` (2 occurrences)
- Rank-based statistics require sorting

**Required Changes**:
1. Replace concepts
2. Use `scl::sort::argsort_inplace()` for ranking
3. Ensure proper use of unified accessors

### `normalize.hpp` - Normalization

**Current Issues**:
- Uses `CSRLike`/`CSCLike` (6 occurrences)
- Many in-place operations

**Required Changes**:
1. Replace all concepts
2. Distinguish read-only (ISparse) vs. writable (Custom) operations
3. SIMD optimization for scaling operations
4. Parallel processing for per-feature normalization

### `algebra.hpp` - Matrix Algebra

**Current Issues**:
- Uses `CSRLike`/`CSCLike` (6 occurrences)
- Element-wise operations

**Required Changes**:
1. Replace concepts
2. SIMD batch operations for aligned data (CustomSparse)
3. Row-wise operations for Virtual
4. Fused operations where possible

### `bbknn.hpp` - Batch-Balanced KNN

**Current Issues**:
- Uses `CSRLike` (4 occurrences)
- Complex neighbor search algorithm

**Required Changes**:
1. Replace concepts
2. Use unified accessors
3. Optimize distance computation with SIMD
4. Leverage existing `neighbors.hpp` infrastructure

---

## Verification Checklist

For EACH file after refactoring:

- [ ] No occurrences of `CSRLike` or `CSCLike` remain
- [ ] All direct member access replaced with unified accessors
- [ ] ISparse implementations exist for all public APIs
- [ ] SparseLike concept-based implementations exist
- [ ] SIMD optimizations applied to hot loops
- [ ] Parallel processing used where beneficial
- [ ] Documentation updated (no Markdown/LaTeX)
- [ ] File compiles without errors
- [ ] Static assertions pass
- [ ] Performance validated (if possible)

---

## Code Quality Standards

### 1. Error Checking
```cpp
SCL_CHECK_DIM(out.size == expected, "Description");
SCL_CHECK_ARG(value > 0, "Description");
SCL_ASSERT(condition, "Debug-only check");
```

### 2. SIMD Pattern
```cpp
namespace s = scl::simd;
const s::Tag d;
const size_t lanes = s::lanes();

auto v_acc = s::Zero(d);
size_t k = 0;

// Vectorized loop
for (; k + lanes <= n; k += lanes) {
    auto v = s::Load(d, ptr + k);
    v_acc = s::Add(v_acc, v);
}

// Scalar tail
for (; k < n; ++k) {
    // Scalar operations
}
```

### 3. Parallel Pattern
```cpp
// Parallel over primary dimension
scl::threading::parallel_for(0, static_cast<size_t>(scl::primary_size(matrix)), [&](size_t i) {
    auto vals = scl::primary_values(matrix, static_cast<Index>(i));
    // Per-row/column processing
});
```

### 4. Documentation Pattern
```cpp
/// @brief One-line summary.
///
/// Detailed description of algorithm.
///
/// Algorithm:
/// 1. Step 1 description
/// 2. Step 2 description
///
/// Performance:
/// - Time: O(...)
/// - Space: O(...)
///
/// @tparam MatrixT Matrix type satisfying SparseLike
/// @param matrix Input matrix
/// @param output Output buffer
template <typename MatrixT>
    requires SparseLike<MatrixT, IsCSR>
void function_name(...) {
    // Implementation
}
```

---

## Testing Strategy

### 1. Concept Verification (Compile-Time)
```cpp
// At end of file
static_assert(SparseLike<MappedCustomSparse<Real>, true>);
static_assert(ArrayLike<MappedArray<Real>>);
```

### 2. Manual Verification
```bash
# Search for old patterns
grep -r "CSRLike\|CSCLike" scl/kernel/

# Search for direct access
grep -r "matrix\.rows\|matrix\.cols\|matrix\.nnz" scl/kernel/

# Check for old tags
grep -r "TagCSR\|TagCSC" scl/kernel/
```

---

## Current Progress

### Completed
1. ✅ Core system redesign (matrix.hpp, sparse.hpp, dense.hpp)
2. ✅ ArrayLike concept added
3. ✅ MappedArray implements ArrayLike
4. ✅ Renamed MountMatrix -> MappedCustomSparse
5. ✅ Renamed VirtualMountMatrix -> MappedVirtualSparse
6. ✅ Updated 13 kernel files (spatial, ttest, sparse, softmax, scale, qc, resample, reorder, feature, neighbors, merge, correlation, hvg)

### Remaining
7. ⏳ `group.hpp` - 12 occurrences to fix
8. ⏳ `mmd.hpp` - 2 occurrences to fix
9. ⏳ `mwu.hpp` - 2 occurrences to fix
10. ⏳ `normalize.hpp` - 6 occurrences to fix
11. ⏳ `algebra.hpp` - 6 occurrences to fix
12. ⏳ `bbknn.hpp` - 4 occurrences to fix
13. ⏳ `log1p.hpp` - Check for any issues
14. ⏳ `gram.hpp` - Check for any issues

---

## Example: Complete Refactoring of a Function

### Before
```cpp
template <CSCLike MatrixT>
void compute_stats(const MatrixT& matrix, MutableSpan<Real> output) {
    const Index n_genes = matrix.cols;
    
    for (Index j = 0; j < n_genes; ++j) {
        auto vals = matrix.col_values(j);
        Real sum = 0.0;
        for (Size k = 0; k < vals.size; ++k) {
            sum += vals[k];
        }
        output[j] = sum;
    }
}
```

### After (Optimized)
```cpp
/// @brief Compute column sums (Virtual Interface, CSC).
///
/// Generic implementation using ISparse base class.
template <typename T>
void compute_stats(const ICSC<T>& matrix, MutableSpan<Real> output) {
    const Index n_genes = matrix.cols();
    SCL_CHECK_DIM(output.size == static_cast<Size>(n_genes), "Output size mismatch");
    
    scl::threading::parallel_for(0, static_cast<size_t>(n_genes), [&](size_t j) {
        auto vals = matrix.primary_values(static_cast<Index>(j));
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_sum = s::Zero(d);
        size_t k = 0;
        
        for (; k + lanes <= vals.size; k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            v_sum = s::Add(v_sum, v);
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        
        for (; k < vals.size; ++k) {
            sum += vals[k];
        }
        
        output[j] = sum;
    });
}

/// @brief Compute column sums (Concept-based, Optimized, CSC).
///
/// High-performance implementation for SparseLike<false> matrices.
/// Uses SIMD and parallel processing.
template <typename MatrixT>
    requires SparseLike<MatrixT, false>
void compute_stats(const MatrixT& matrix, MutableSpan<Real> output) {
    const Index n_genes = scl::cols(matrix);
    SCL_CHECK_DIM(output.size == static_cast<Size>(n_genes), "Output size mismatch");
    
    scl::threading::parallel_for(0, static_cast<size_t>(n_genes), [&](size_t j) {
        auto vals = scl::primary_values(matrix, static_cast<Index>(j));
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_sum = s::Zero(d);
        size_t k = 0;
        
        for (; k + lanes <= vals.size; k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            v_sum = s::Add(v_sum, v);
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        
        for (; k < vals.size; ++k) {
            sum += vals[k];
        }
        
        output[j] = sum;
    });
}
```

---

## Important Notes

### 1. DO NOT Break Existing APIs

- Keep function signatures compatible
- Add new overloads, don't remove old ones (unless explicitly deprecated)
- Maintain backward compatibility where possible

### 2. Performance is CRITICAL

- Every hot loop MUST use SIMD
- Parallelize at appropriate granularity
- Avoid unnecessary allocations
- Use core utilities (sort, argsort) instead of std::

### 3. Concept-First Design

- Prefer `SparseLike<M, IsCSR>` over `CSRLike`/`CSCLike`
- More explicit, more flexible
- Enables template parameter passing

### 4. Memory-Mapped Types

- MappedCustomSparse and MappedVirtualSparse automatically work
- They satisfy SparseLike concept
- No special handling needed
- IO overhead dominates, so generic implementations are fine

---

## Quick Reference Commands

### Find remaining work
```bash
grep -r "CSRLike\|CSCLike" scl/kernel/ | wc -l
```

### Check specific file
```bash
grep -n "CSRLike\|CSCLike" scl/kernel/group.hpp
```

### Verify unified accessors
```bash
grep -n "matrix\\.rows\|matrix\\.cols\|matrix\\.nnz" scl/kernel/group.hpp
```

### Check tag usage
```bash
grep -n "TagCSR\|TagCSC\|std::is_same_v<Tag" scl/kernel/group.hpp
```

---

## Contact and Questions

If unclear about:
- **Concept usage**: Refer to `scl/core/matrix.hpp` lines 170-450
- **Sparse implementations**: Refer to `scl/core/sparse.hpp`
- **Example refactoring**: See `scl/kernel/spatial.hpp`, `scl/kernel/correlation.hpp`
- **SIMD patterns**: See `scl/kernel/feature.hpp`
- **Code style**: See `AGENT.md`

---

## Success Criteria

The refactoring is complete when:

1. ✅ ZERO occurrences of `CSRLike` or `CSCLike` in scl/kernel/
2. ✅ ZERO occurrences of `TagCSR` or `TagCSC` (use `TagSparse<bool>`)
3. ✅ ALL hot loops use SIMD optimization
4. ✅ ALL algorithms have proper implementation layers
5. ✅ ALL files compile without errors
6. ✅ Performance validated (same or better than before)

---

## Estimated Effort

- `group.hpp`: ~30 minutes (12 replacements + optimization review)
- `mmd.hpp`: ~15 minutes (2 replacements + algorithm review)
- `mwu.hpp`: ~15 minutes (2 replacements + sorting optimization)
- `normalize.hpp`: ~25 minutes (6 replacements + SIMD review)
- `algebra.hpp`: ~25 minutes (6 replacements + element-wise optimizations)
- `bbknn.hpp`: ~20 minutes (4 replacements + distance computation review)
- `log1p.hpp` + `gram.hpp`: ~20 minutes (verification only)

**Total**: ~2.5 hours for systematic refactoring

---

## Priorities

1. **Correctness**: Must compile, must satisfy concepts
2. **Performance**: SIMD + Parallel + Core utilities
3. **Maintainability**: Clean abstractions, proper layering
4. **Documentation**: Clear, accurate, no Markdown/LaTeX

Work systematically through each file. Do not rush. Verify each change.

