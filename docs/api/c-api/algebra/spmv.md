---
title: scl_algebra_spmv
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
  id: 'scl_algebra_spmv',
  return_type: 'scl_error_t',
  source: { file: 'scl/binding/c_api/algebra.h', line: 42 },
  params: [
    { name: 'A', type: 'scl_sparse_t', dir: 'in', nullable: false },
    { name: 'x', type: 'const scl_real_t*', dir: 'in', nullable: false },
    { name: 'x_size', type: 'scl_size_t', dir: 'in', nullable: false },
    { name: 'y', type: 'scl_real_t*', dir: 'inout', nullable: false },
    { name: 'y_size', type: 'scl_size_t', dir: 'in', nullable: false },
    { name: 'alpha', type: 'scl_real_t', dir: 'in', nullable: false, default: '1.0' },
    { name: 'beta', type: 'scl_real_t', dir: 'in', nullable: false, default: '0.0' }
  ],
  errors: [
    { code: 'SCL_ERROR_NULL_POINTER', condition: 'If A, x, or y is NULL' },
    { code: 'SCL_ERROR_DIMENSION_MISMATCH', condition: 'If x_size ≠ A.cols or y_size ≠ A.rows' },
    { code: 'SCL_ERROR_INVALID_FORMAT', condition: 'If A is not in CSR or CSC format' }
  ],
  complexity: { time: 'O(nnz)', space: 'O(1)' },
  version: '0.4.0',
  status: 'stable'
}
</script>

<!-- Module Navigation -->
<ModuleNav
  module="scl_algebra"
  current="scl_algebra_spmv"
  :functions="algebraFunctions"
/>

# scl_algebra_spmv

<ApiFunctionRenderer :data="funcData" />

---

## Brief

General sparse matrix-vector multiplication with scaling factors, supporting both CSR and CSC formats.

## Formula

$$y \leftarrow \alpha \cdot A \cdot x + \beta \cdot y$$

where:
- $A$ is an $m \times n$ sparse matrix
- $x$ is an input vector of length $n$
- $y$ is an output vector of length $m$
- $\alpha, \beta$ are scalar coefficients

---

## Description

Performs **general sparse matrix-vector multiplication (SpMV)**, one of the most fundamental operations in sparse linear algebra. This operation is the core building block for:

- **Iterative solvers**: Conjugate Gradient, GMRES, BiCGSTAB
- **Eigenvalue computations**: Power iteration, Lanczos algorithm
- **Graph algorithms**: PageRank, spectral clustering
- **Machine learning**: Sparse neural networks, feature transformations

### Matrix Format Support

| Format | Primary Dim | Access Pattern | Best Use Case |
|--------|-------------|----------------|---------------|
| **CSR** | Rows | Row-major iteration | Row-wise operations, most common |
| **CSC** | Columns | Column-major iteration | Column-wise operations, transposed access |

The function automatically detects the matrix format and optimizes the iteration pattern accordingly.

### Performance Characteristics

- **Memory bandwidth bound**: Performance scales with memory bandwidth
- **Cache-friendly**: Sequential access pattern within rows/columns
- **SIMD optimized**: Uses vectorized operations when available
- **No internal allocations**: Zero heap allocations during execution

---

## FFI Stability

<Callout type="tip" title="Stable ABI">

This function is part of the **stable C ABI** since version 0.4.0.

**Guarantees:**
- Function signature will not change in minor versions (0.x.y)
- Symbol name is stable and exported
- Parameter order and types are fixed
- Return type semantics are fixed

**Migration path:** If breaking changes are needed, a new function `scl_algebra_spmv2` will be introduced, and this function will be deprecated with a 2-version notice period.

</Callout>

---

## Data Guarantees

### Input Guarantees

| Parameter | Guarantee |
|-----------|-----------|
| `A` | Read-only access, not modified |
| `x` | Read-only access, not modified |
| `x_size` | Used for bounds checking only |
| `y_size` | Used for bounds checking only |
| `alpha`, `beta` | Values copied, originals not accessed after call |

### Output Guarantees

| Condition | Behavior |
|-----------|----------|
| `beta == 0` | `y` is overwritten completely, initial values ignored |
| `beta != 0` | `y` is read and updated: `y = alpha*A*x + beta*y` |
| On error | `y` contents are **undefined** (partial writes may occur) |

### Memory Guarantees

- **No heap allocations** during execution
- **No internal state** retained after return
- **Thread-local only**: No shared mutable state
- Pointer validity required only during call

---

## Mutability

<Callout type="warning" title="INPLACE Operation">

This function performs **in-place modification** of the output vector `y`.

**Caller responsibilities:**
1. Pre-allocate `y` with at least `y_size` elements
2. Ensure `y_size == A.rows` (for CSR) or `y_size == A.cols` (for CSC)
3. Initialize `y` values if `beta != 0`

**Memory layout:**
```
y[0..y_size] = alpha * (A @ x) + beta * y[0..y_size]
```

</Callout>

---

## Thread Safety

<Callout type="info" title="Conditionally Thread-Safe">

**Safe concurrent access:**
- Same matrix `A` from multiple threads ✓
- Same input vector `x` from multiple threads ✓
- Different output vectors `y` from multiple threads ✓

**Unsafe concurrent access:**
- Same output vector `y` from multiple threads ✗ (data race)
- Modifying `A` while calling this function ✗ (undefined behavior)

</Callout>

### Recommended Parallel Pattern

```c
// Safe: each thread has its own output buffer
#pragma omp parallel
{
    scl_real_t* y_local = allocate_vector(m);
    scl_algebra_spmv(A, x, n, y_local, m, 1.0, 0.0);
    // ... use y_local ...
}
```

---

## Notes

### Edge Cases

| Condition | Behavior |
|-----------|----------|
| `alpha == 0 && beta == 0` | `y` is zeroed |
| `alpha == 0 && beta == 1` | `y` unchanged (no-op) |
| `alpha == 0 && beta != 0,1` | `y` scaled by `beta` |
| Empty matrix (nnz == 0) | `y = beta * y` |

### Performance Tips

1. **Prefer CSR format** for row-major access patterns
2. **Align vectors** to 64-byte boundaries for best SIMD performance
3. **Reuse output buffer** across iterations to avoid allocation overhead
4. **Use beta=0** when possible to skip read of y

### Numerical Considerations

- Floating-point operations follow IEEE 754 semantics
- No special handling for NaN/Inf (propagated through computation)
- Order of operations may vary between implementations (non-associative)

---

<SeeAlso :links="[
  { href: './spmm', text: 'scl_algebra_spmm - Matrix-matrix multiply' },
  { href: './spmv_t', text: 'scl_algebra_spmv_t - Transposed SpMV' },
  { href: '../sparse', text: 'Sparse Matrix Types' },
  { href: '../core', text: 'Core Types Reference' }
]" />
