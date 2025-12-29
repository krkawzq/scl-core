---
title: scl_algebra_spmm
---

<script setup>
// Algebra module functions for navigation
const algebraFunctions = [
  { id: 'scl_algebra_spmv', name: 'spmv', href: './spmv', brief: 'Sparse matrix-vector multiply' },
  { id: 'scl_algebra_spmm', name: 'spmm', href: './spmm', brief: 'Sparse matrix-matrix multiply' },
  { id: 'scl_algebra_spmv_t', name: 'spmv_t', href: './spmv_t', brief: 'Transposed SpMV' },
  { id: 'scl_algebra_axpy', name: 'axpy', href: './axpy', brief: 'Vector addition with scaling' }
]

// Function metadata from YAML (codegen generated)
const funcData = {
  id: 'scl_algebra_spmm',
  return_type: 'scl_error_t',
  source: { file: 'scl/binding/c_api/algebra.h', line: 98 },
  params: [
    { name: 'A', type: 'scl_sparse_t', dir: 'in', nullable: false },
    { name: 'B', type: 'const scl_real_t*', dir: 'in', nullable: false },
    { name: 'B_rows', type: 'scl_size_t', dir: 'in', nullable: false },
    { name: 'B_cols', type: 'scl_size_t', dir: 'in', nullable: false },
    { name: 'B_stride', type: 'scl_size_t', dir: 'in', nullable: false },
    { name: 'C', type: 'scl_real_t*', dir: 'out', nullable: false },
    { name: 'C_rows', type: 'scl_size_t', dir: 'in', nullable: false },
    { name: 'C_cols', type: 'scl_size_t', dir: 'in', nullable: false },
    { name: 'C_stride', type: 'scl_size_t', dir: 'in', nullable: false }
  ],
  errors: [
    { code: 'SCL_ERROR_NULL_POINTER', condition: 'If A, B, or C is NULL' },
    { code: 'SCL_ERROR_DIMENSION_MISMATCH', condition: 'If A.cols ≠ B_rows or A.rows ≠ C_rows or B_cols ≠ C_cols' },
    { code: 'SCL_ERROR_INVALID_STRIDE', condition: 'If stride < cols for row-major layout' }
  ],
  complexity: { time: 'O(nnz × k)', space: 'O(1)' },
  version: '0.4.0',
  status: 'stable'
}
</script>

<!-- Module Navigation -->
<ModuleNav
  module="scl_algebra"
  current="scl_algebra_spmm"
  :functions="algebraFunctions"
/>

# scl_algebra_spmm

<ApiFunctionRenderer :data="funcData" />

---

## Brief

Sparse-dense matrix-matrix multiplication, computing C = A × B where A is sparse and B is dense.

## Formula

$$C \leftarrow A \cdot B$$

where:
- $A$ is an $m \times k$ sparse matrix (CSR or CSC format)
- $B$ is a $k \times n$ dense matrix (row-major layout)
- $C$ is an $m \times n$ dense output matrix (row-major layout)

---

## Description

Performs **sparse-dense matrix-matrix multiplication (SpMM)**, a key operation for batch processing and multi-column computations. This is particularly useful for:

- **Batch SpMV**: Multiplying the same sparse matrix with multiple vectors
- **Feature transformation**: Applying sparse weights to dense feature matrices
- **Graph neural networks**: Aggregating neighbor features with sparse adjacency
- **Dimensionality reduction**: Sparse projection matrices

### Memory Layout

Both B and C are stored in **row-major** format with configurable stride:

```
┌─────────────────────────────┐
│ B[0,0] B[0,1] ... B[0,n-1]  │ ← stride = B_stride
│ B[1,0] B[1,1] ... B[1,n-1]  │
│ ...                          │
│ B[k-1,0] ...    B[k-1,n-1]  │
└─────────────────────────────┘
```

The stride allows working with submatrices of larger allocations.

### Performance Characteristics

- **Better cache utilization** than repeated SpMV calls
- **Vectorizable** across columns of B
- **Parallelizable** across rows of A
- **Memory bound** for large matrices

---

## FFI Stability

<Callout type="tip" title="Stable ABI">

This function is part of the **stable C ABI** since version 0.4.0.

The row-major layout convention is fixed and will not change. Column-major support may be added via a separate `scl_algebra_spmm_cm` function in the future.

</Callout>

---

## Data Guarantees

### Input Guarantees

| Parameter | Guarantee |
|-----------|-----------|
| `A` | Read-only, sparse matrix unchanged |
| `B` | Read-only, dense matrix unchanged |
| `B_stride` | Minimum value is `B_cols` |

### Output Guarantees

| Condition | Behavior |
|-----------|----------|
| Success | `C` contains the full product A × B |
| Error | `C` contents are **undefined** |

### Memory Guarantees

- Output `C` is **fully overwritten** (not accumulated)
- No heap allocations during execution
- Pointer validity required only during call

---

## Mutability

<Callout type="warning" title="OVERWRITE Operation">

This function **fully overwrites** the output matrix `C`. Unlike SpMV, there is no beta parameter for accumulation.

**Caller responsibilities:**
1. Pre-allocate `C` with at least `C_rows × C_stride` elements
2. Ensure dimension compatibility: `A.cols == B_rows`, `A.rows == C_rows`, `B_cols == C_cols`
3. Previous values in `C` are discarded

**For accumulation**, manually add: `C_new = C_old + A × B`

</Callout>

---

## Thread Safety

<Callout type="info" title="Conditionally Thread-Safe">

**Safe concurrent access:**
- Same matrix `A` from multiple threads ✓
- Same matrix `B` from multiple threads ✓
- Different output matrices `C` from multiple threads ✓

**Unsafe concurrent access:**
- Same output matrix `C` from multiple threads ✗
- Modifying `A` or `B` while calling this function ✗

</Callout>

### Parallel Batch Processing

```c
// Process multiple batches in parallel
#pragma omp parallel for
for (int batch = 0; batch < num_batches; batch++) {
    scl_real_t* B_batch = B + batch * B_stride * k;
    scl_real_t* C_batch = C + batch * C_stride * m;
    scl_algebra_spmm(A, B_batch, k, n, B_stride,
                        C_batch, m, n, C_stride);
}
```

---

## Notes

### Dimension Requirements

```
A: [m × k]  sparse
B: [k × n]  dense, stride ≥ n
C: [m × n]  dense, stride ≥ n

Requirement: A.cols == B_rows == k
             A.rows == C_rows == m
             B_cols == C_cols == n
```

### Performance Tips

1. **Batch vectors** into matrices for better throughput
2. **Match stride to cache line** (64 bytes = 8 doubles) for aligned access
3. **Prefer narrow B** (small n) for memory efficiency
4. Consider **transposed SpMM** if access pattern is column-oriented

### When to Use SpMM vs Repeated SpMV

| Use Case | Recommendation |
|----------|----------------|
| n = 1 | Use `scl_algebra_spmv` |
| n = 2-8 | Either works, SpMM slightly faster |
| n > 8 | Use `scl_algebra_spmm` |
| n varies | Use SpMM with stride for flexibility |

---

<SeeAlso :links="[
  { href: './spmv', text: 'scl_algebra_spmv - Vector multiply' },
  { href: './spmv_t', text: 'scl_algebra_spmv_t - Transposed SpMV' },
  { href: '../sparse', text: 'Sparse Matrix Types' }
]" />
