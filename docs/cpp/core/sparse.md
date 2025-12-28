# Sparse Matrix Core

Block-allocated discontiguous sparse matrix with zero-copy slicing and registry-managed lifecycle.

**Location**: `scl/core/sparse.hpp`

---

## Overview

The `Sparse<T, IsCSR>` struct is SCL-Core's primary sparse matrix data structure with several key innovations:

**Key Features:**
- **Block-allocated storage**: Balances memory reuse and fragmentation
- **Zero-copy slicing**: Shares memory via registry reference counting
- **Registry-managed lifecycle**: Automatic memory management with alias refcounting
- **Traditional format compatible**: Can wrap or convert to/from standard CSR/CSC
- **SIMD-optimized operations**: Uses scl::algo custom operators for zero-overhead
- **Prefetch hints**: Strategic prefetching for large data operations

**Design Philosophy:**
- Indices within each row/column are **strictly sorted** (invariant enforced by all operations)
- Lifecycle managed by registry (no manual memory management)
- Zero-copy when possible, copy when necessary
- Compatible with external data (NumPy, memory-mapped files) via wrap_traditional

---

## Memory Architecture

**Discontiguous Storage (Pointer-based)**:
```
data_ptrs    = [block1+offset0, block1+offset1, block2+offset0, ...]
indices_ptrs = [block1+offset0, block1+offset1, block2+offset0, ...]
lengths      = [len0, len1, len2, ...]
```

**Lifecycle Management**:
1. **Buffers** (Layer 1): Real memory blocks allocated via registry
2. **Aliases** (Layer 2): Access pointers with reference counting
3. **Instances** (Layer 3): Sparse objects that hold aliases

**Benefits**:
- Each row/column can be in separate allocation (flexible)
- Zero-copy slicing via alias sharing (efficient)
- Partial memory release possible (blocks can be freed independently)
- Compatible with traditional contiguous formats

---

## Sparse<T, IsCSR>

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
struct Sparse {
    Pointer* data_ptrs;      // Pointer array to row/col values
    Pointer* indices_ptrs;   // Pointer array to row/col indices
    Index* lengths;          // Length of each row/col
    Index rows_;
    Index cols_;
    Index nnz_;
};
```

**TEMPLATE PARAMETERS:**
- T: Value type (typically Real, float, or double)
- IsCSR: true for CSR (row-major), false for CSC (column-major)

**TYPE ALIASES:**
```cpp
using CSRMatrix<T> = Sparse<T, true>;
using CSCMatrix<T> = Sparse<T, false>;
using CSR = CSRMatrix<Real>;
using CSC = CSCMatrix<Real>;
```

**INVARIANTS:**
- Indices are **strictly sorted** within each primary dimension
- All pointers are registry-managed or explicitly unmanaged (wrap_traditional)
- lengths[i] == number of non-zeros in primary dimension i

---

## Construction and Lifecycle

### Constructors

```cpp
constexpr Sparse() noexcept;  // Empty matrix

// Direct construction (advanced use)
constexpr Sparse(Pointer* dp, Pointer* ip, Index* len,
                 Index r, Index c, Index n) noexcept;

// Copy disabled - use clone() for deep copy
Sparse(const Sparse&) = delete;

// Move enabled - transfers ownership
Sparse(Sparse&& other) noexcept;
```

---

## Factory Methods

### zeros

**SUMMARY:**
Create empty matrix with zero non-zeros.

**SIGNATURE:**
```cpp
[[nodiscard]] static Sparse zeros(Index rows, Index cols);
```

**OPTIMIZATION:**
Uses scl::algo::zero instead of std::memset for metadata initialization.

---

### create

**SUMMARY:**
Create sparse matrix from nnz counts with block allocation strategy.

**SIGNATURE:**
```cpp
[[nodiscard]] static Sparse create(
    Index rows,
    Index cols,
    std::span<const Index> primary_nnzs,
    BlockStrategy strategy = BlockStrategy::adaptive()
);
```

**PARAMETERS:**
- rows         [in] Number of rows
- cols         [in] Number of columns
- primary_nnzs [in] NNZ count per primary dimension [primary_dim]
- strategy     [in] Block allocation strategy

**PRECONDITIONS:**
- rows >= 0, cols >= 0
- primary_nnzs.size() == primary_dim
- All primary_nnzs[i] >= 0

**POSTCONDITIONS:**
- Returns matrix with allocated data blocks
- All values initialized to zero
- Blocks registered with registry as buffers with aliases

**ALGORITHM:**
1. Compute block size from strategy
2. Allocate blocks to minimize count while respecting constraints
3. Register blocks as buffers, create aliases for each row/column
4. Set up pointer arrays

**BLOCK STRATEGIES:**
```cpp
BlockStrategy::contiguous();   // Single block (traditional CSR/CSC)
BlockStrategy::small_blocks();  // Many small blocks (max 16K elements)
BlockStrategy::large_blocks();  // Few large blocks (max 1M elements)
BlockStrategy::adaptive();      // Auto-tune based on nnz and hardware
```

**OPTIMIZATION:**
- Batched alias creation reduces registry overhead
- Aligned block allocation for SIMD operations

---

### from_traditional

**SUMMARY:**
Create from traditional CSR/CSC arrays (copies data).

**SIGNATURE:**
```cpp
[[nodiscard]] static Sparse from_traditional(
    Index rows,
    Index cols,
    std::span<const T> values,
    std::span<const Index> indices,
    std::span<const Index> offsets,
    BlockStrategy strategy = BlockStrategy::adaptive()
);
```

**PARAMETERS:**
- rows     [in] Number of rows
- cols     [in] Number of columns
- values   [in] Contiguous values array
- indices  [in] Contiguous secondary indices
- offsets  [in] Primary dimension offsets [primary_dim+1]
- strategy [in] Block allocation strategy

**PRECONDITIONS:**
- offsets.size() == primary_dim + 1
- offsets[0] == 0
- offsets is non-decreasing
- values.size() >= offsets[primary_dim]
- indices.size() >= offsets[primary_dim]
- Indices are sorted within each primary dimension

**POSTCONDITIONS:**
- Returns new matrix with copied data
- Data allocated via registry with specified strategy

**OPTIMIZATION:**
- Uses scl::algo::copy instead of std::memcpy
- Prefetch hints for large copies
- SCL_LIKELY branch hints for non-empty rows

---

### wrap_traditional

**SUMMARY:**
Wrap existing traditional arrays (zero-copy, external ownership).

**SIGNATURE:**
```cpp
[[nodiscard]] static Sparse wrap_traditional(
    Index rows,
    Index cols,
    T* values,
    Index* indices,
    std::span<const Index> offsets
);
```

**PARAMETERS:**
- rows    [in] Number of rows
- cols    [in] Number of columns
- values  [in] Pointer to values array (caller owns)
- indices [in] Pointer to indices array (caller owns)
- offsets [in] Offset array [primary_dim+1]

**PRECONDITIONS:**
- values and indices must not be null
- Arrays must outlive the Sparse object
- offsets[0] == 0, offsets is non-decreasing

**POSTCONDITIONS:**
- Returns matrix wrapping external arrays (no copy)
- Data pointers are NOT registered with registry
- Caller must manage array lifetime

**LIFECYCLE:**
When destroyed, only metadata is freed (data_ptrs, indices_ptrs, lengths).
Original data arrays are NOT freed.

**WARNING:**
For proper lifecycle management with registry, use from_contiguous_arrays with take_ownership=true.

---

### from_coo

**SUMMARY:**
Create from COO (Coordinate) format.

**SIGNATURE:**
```cpp
[[nodiscard]] static Sparse from_coo(
    Index rows,
    Index cols,
    std::span<const Index> row_indices,
    std::span<const Index> col_indices,
    std::span<const T> values,
    BlockStrategy strategy = BlockStrategy::adaptive()
);
```

**PARAMETERS:**
- rows        [in] Number of rows
- cols        [in] Number of columns
- row_indices [in] Row coordinates [nnz]
- col_indices [in] Column coordinates [nnz]
- values      [in] Values [nnz]
- strategy    [in] Block allocation strategy

**PRECONDITIONS:**
- All arrays have same size
- All row_indices[i] in [0, rows)
- All col_indices[i] in [0, cols)

**POSTCONDITIONS:**
- Returns sorted sparse matrix
- Indices are sorted via sort_indices()
- Duplicates NOT summed (last value kept)

**OPTIMIZATION:**
- Prefetch hints with configurable distance (16 elements ahead)
- Unrolled final sort loops

---

### from_dense

**SUMMARY:**
Create from dense matrix (row-major layout).

**SIGNATURE:**
```cpp
template <typename Pred = std::nullptr_t>
[[nodiscard]] static Sparse from_dense(
    Index rows,
    Index cols,
    std::span<const T> data,
    Pred&& is_nonzero = nullptr,
    BlockStrategy strategy = BlockStrategy::adaptive()
);
```

**PARAMETERS:**
- rows        [in] Number of rows
- cols        [in] Number of columns
- data        [in] Dense matrix [rows*cols], row-major
- is_nonzero  [in] Custom predicate (default: x != 0)
- strategy    [in] Block allocation strategy

**PRECONDITIONS:**
- data.size() >= rows * cols

**POSTCONDITIONS:**
- Returns sparse matrix containing non-zero elements
- Elements where is_nonzero(x) == true are included

---

## Element Access

### at

**SUMMARY:**
Get element value at (row, col) with bounds checking.

**SIGNATURE:**
```cpp
[[nodiscard]] SCL_FORCE_INLINE T at(Index row, Index col) const noexcept;
```

**COMPLEXITY:**
O(log n) - binary search on sorted indices

**OPTIMIZATION:**
- SCL_FORCE_INLINE for zero-overhead
- SCL_UNLIKELY for bounds check failures
- Uses scl::algo::lower_bound (faster than std::)
- SCL_LIKELY for found case

---

### at_unsafe

**SUMMARY:**
Get element without bounds checking (caller guarantees validity).

**SIGNATURE:**
```cpp
[[nodiscard]] SCL_FORCE_INLINE T at_unsafe(Index row, Index col) const noexcept;
```

**WARNING:**
Caller must guarantee matrix is valid and indices are in bounds.

**OPTIMIZATION:**
- No bounds checks
- Direct binary search
- SCL_UNLIKELY for empty check

---

## Slicing

### row_slice_view

**SUMMARY:**
Zero-copy row slice (CSR only) with shared memory.

**SIGNATURE:**
```cpp
[[nodiscard]] Sparse row_slice_view(std::span<const Index> row_indices) const
    requires (IsCSR);
```

**PARAMETERS:**
- row_indices [in] Rows to include in slice

**PRECONDITIONS:**
- All row_indices[i] in [0, rows())

**POSTCONDITIONS:**
- Returns new matrix sharing memory with original
- Modifications affect both matrices
- Registry ref_count incremented for shared aliases

**ALGORITHM:**
1. Allocate new metadata (data_ptrs, indices_ptrs, lengths)
2. Copy pointers for selected rows
3. Collect aliases and call registry.alias_incref_batch
4. Return new matrix

**LIFECYCLE:**
Shared aliases use reference counting. When both original and slice are destroyed, memory is freed.

**OPTIMIZATION:**
- Batched alias_incref for efficiency
- Safe for unregistered pointers (wrap_traditional)

---

### row_slice_copy

**SUMMARY:**
Deep copy row slice with independent memory.

**SIGNATURE:**
```cpp
[[nodiscard]] Sparse row_slice_copy(
    std::span<const Index> row_indices,
    BlockStrategy strategy = BlockStrategy::adaptive()
) const requires (IsCSR);
```

**POSTCONDITIONS:**
- Returns independent matrix
- Modifications do NOT affect original

**OPTIMIZATION:**
Uses scl::algo::copy with prefetch hints.

---

## Operations

### clone

**SUMMARY:**
Deep copy with optional new block strategy.

**SIGNATURE:**
```cpp
[[nodiscard]] Sparse clone(
    BlockStrategy strategy = BlockStrategy::adaptive()
) const;
```

**OPTIMIZATION:**
- Prefetch next row while copying current
- scl::algo::copy for zero-overhead

---

### transpose

**SUMMARY:**
Convert between CSR and CSC.

**SIGNATURE:**
```cpp
[[nodiscard]] TransposeType transpose() const;
```

**ALGORITHM:**
1. Count nnz per new primary dimension
2. Allocate result
3. Fill transposed data
4. Sort indices

**OPTIMIZATION:**
- Prefetch hints during transpose fill
- Parallel counting of new nnzs

---

### sort_indices

**SUMMARY:**
Sort indices within each primary dimension (in-place).

**SIGNATURE:**
```cpp
void sort_indices();
```

**ALGORITHM:**
1. Find max length for buffer preallocation
2. Preallocate temp arrays (reused across all rows)
3. For each row: create permutation, apply via temp buffers

**OPTIMIZATION:**
- SCL_UNLIKELY for early exits
- Prefetch next row while sorting current
- Reuses temp buffers to avoid repeated allocation
- scl::algo::max2 instead of std::max

---

### scale

**SUMMARY:**
Multiply all values by constant (in-place).

**SIGNATURE:**
```cpp
void scale(T factor);
```

**MUTABILITY:**
INPLACE - modifies values

---

## Format Conversion

### to_traditional

**SUMMARY:**
Export to traditional CSR/CSC format.

**SIGNATURE:**
```cpp
struct TraditionalFormat {
    std::vector<T> values;
    std::vector<Index> indices;
    std::vector<Index> offsets;
};

[[nodiscard]] TraditionalFormat to_traditional() const;
```

**POSTCONDITIONS:**
- Returns contiguous arrays
- offsets[0] = 0, offsets[primary_dim] = nnz

---

### to_dense

**SUMMARY:**
Convert to dense row-major matrix.

**SIGNATURE:**
```cpp
[[nodiscard]] std::vector<T> to_dense() const;
```

**RETURN VALUE:**
Vector of size rows*cols, row-major layout with zeros filled in.

---

## Layout Information

### is_contiguous

**SUMMARY:**
Check if data is in single contiguous block.

**SIGNATURE:**
```cpp
[[nodiscard]] bool is_contiguous() const noexcept;
```

**ALGORITHM:**
Check if all non-empty rows/columns point to sequential memory.

**OPTIMIZATION:**
- SCL_UNLIKELY for edge cases
- Early exit for empty matrix

---

### layout_info

**SUMMARY:**
Get detailed memory layout statistics.

**SIGNATURE:**
```cpp
struct SparseLayoutInfo {
    Index data_block_count;
    Index index_block_count;
    Size data_bytes;
    Size index_bytes;
    Size metadata_bytes;
    bool is_contiguous;
    bool is_traditional_format;
};

[[nodiscard]] SparseLayoutInfo layout_info() const noexcept;
```

**ALGORITHM:**
Uses unordered_set for O(1) unique block detection via registry BufferIDs.

---

## BlockStrategy

**SUMMARY:**
Configuration for block allocation.

**SIGNATURE:**
```cpp
struct BlockStrategy {
    Index min_block_elements = 4096;
    Index max_block_elements = 262144;  // 256K elements
    Index target_block_count = 0;       // 0 = auto
    bool force_contiguous = false;
    
    static constexpr BlockStrategy contiguous();
    static constexpr BlockStrategy small_blocks();
    static constexpr BlockStrategy large_blocks();
    static constexpr BlockStrategy adaptive();
    
    [[nodiscard]] Index compute_block_size(Index total_nnz, Index primary_dim) const;
};
```

**STRATEGIES:**
- **contiguous()**: Single block (compatible with traditional CSR/CSC)
- **small_blocks()**: 1K-16K elements per block (fine-grained release)
- **large_blocks()**: 64K-1M elements per block (fewer allocations)
- **adaptive()**: Auto-tune based on nnz and hardware concurrency

---

## Optimization Summary

**Custom Operators:**
- scl::algo::copy instead of std::memcpy
- scl::algo::zero instead of std::memset
- scl::algo::max2 instead of std::max
- scl::algo::lower_bound instead of std::lower_bound

**Branch Prediction:**
- SCL_LIKELY for common paths (non-empty rows, found elements)
- SCL_UNLIKELY for error cases (empty, out of bounds)

**Prefetching:**
- Strategic SCL_PREFETCH_READ for sequential operations
- Distance tuned for typical cache line sizes
- Prefetch next row while processing current

**Inlining:**
- SCL_FORCE_INLINE for element access (at, at_unsafe)
- Zero-overhead abstractions for hot paths

**Memory Layout:**
- Block allocation balances fragmentation vs. release granularity
- Cache-friendly sequential access patterns
- Registry-managed lifecycle eliminates manual tracking

---

## Thread Safety

**Read-only operations**: Safe for concurrent access
**Modifications**: Unsafe - caller must synchronize
**Registry operations**: Thread-safe (sharded with fine-grained locking)

---

## Type Aliases

```cpp
template <typename T> using CSRMatrix = Sparse<T, true>;
template <typename T> using CSCMatrix = Sparse<T, false>;

using CSR = Sparse<Real, true>;
using CSC = Sparse<Real, false>;
using CSRf = Sparse<float, true>;
using CSCf = Sparse<float, false>;
using CSRd = Sparse<double, true>;
using CSCd = Sparse<double, false>;
```

---

## Utility Functions

### vstack

**SUMMARY:**
Concatenate CSR matrices vertically.

**SIGNATURE:**
```cpp
template <typename T>
[[nodiscard]] Sparse<T, true> vstack(
    std::span<const Sparse<T, true>> matrices,
    BlockStrategy strategy = BlockStrategy::adaptive()
);
```

**OPTIMIZATION:**
Prefetch next matrix row during copy.

---

### hstack

**SUMMARY:**
Concatenate CSC matrices horizontally.

**SIGNATURE:**
```cpp
template <typename T>
[[nodiscard]] Sparse<T, false> hstack(
    std::span<const Sparse<T, false>> matrices,
    BlockStrategy strategy = BlockStrategy::adaptive()
);
```

**OPTIMIZATION:**
Prefetch next matrix column during copy.
