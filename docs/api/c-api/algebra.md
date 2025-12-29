---
title: scl_algebra
description: High-performance sparse linear algebra kernels
---

# scl_algebra

C API for high-performance sparse linear algebra kernels including SpMV, SpMM, and row operations.

## Overview

<Callout type="info" title="Performance">
All operations are SIMD-accelerated (AVX2/AVX-512) with automatic parallelization for large matrices.
</Callout>

<SupportMatrix :features="[
  { name: 'scl_algebra_spmv', numpy: true, sparse: true, dask: false, gpu: false },
  { name: 'scl_algebra_spmv_transpose', numpy: true, sparse: true, dask: false, gpu: false },
  { name: 'scl_algebra_spmm', numpy: true, sparse: true, dask: false, gpu: false },
  { name: 'scl_algebra_row_norms', numpy: true, sparse: true, dask: false, gpu: false },
  { name: 'scl_algebra_row_sums', numpy: true, sparse: true, dask: false, gpu: false },
  { name: 'scl_algebra_scale_rows', numpy: true, sparse: true, dask: false, gpu: false }
]" />

---

## scl_algebra_spmv

<Badge type="version">0.4.0</Badge>
<Badge type="status" color="green">Stable</Badge>

Sparse matrix-vector multiplication: $y = \alpha \cdot A \cdot x + \beta \cdot y$

<ApiSignature return-type="scl_error_t" name="scl_algebra_spmv">
  <span class="type">scl_sparse_t</span> <span class="param-name">A</span>,
  <span class="keyword">const</span> <span class="type">scl_real_t</span>* <span class="param-name">x</span>,
  <span class="type">scl_size_t</span> <span class="param-name">x_size</span>,
  <span class="type">scl_real_t</span>* <span class="param-name">y</span>,
  <span class="type">scl_size_t</span> <span class="param-name">y_size</span>,
  <span class="type">scl_real_t</span> <span class="param-name">alpha</span>,
  <span class="type">scl_real_t</span> <span class="param-name">beta</span>
</ApiSignature>

### Parameters

<ParamTable :params="[
  { name: 'A', type: 'scl_sparse_t', dir: 'in', description: 'Sparse matrix handle (CSR or CSC format)' },
  { name: 'x', type: 'const scl_real_t*', dir: 'in', description: 'Input vector of size [secondary_dim]' },
  { name: 'x_size', type: 'scl_size_t', dir: 'in', description: 'Size of input vector' },
  { name: 'y', type: 'scl_real_t*', dir: 'inout', description: 'Output vector of size [primary_dim]' },
  { name: 'y_size', type: 'scl_size_t', dir: 'in', description: 'Size of output vector' },
  { name: 'alpha', type: 'scl_real_t', dir: 'in', description: 'Scaling factor for A·x', default: '1.0' },
  { name: 'beta', type: 'scl_real_t', dir: 'in', description: 'Scaling factor for y', default: '0.0' }
]" />

### Returns

`scl_error_t` - `SCL_OK` on success, error code otherwise.

### Errors

| Code | Condition |
|------|-----------|
| `SCL_ERROR_NULL_POINTER` | If `A`, `x`, or `y` is NULL |
| `SCL_ERROR_DIMENSION_MISMATCH` | If `x_size` ≠ secondary dimension or `y_size` ≠ primary dimension |

### Thread Safety

<Badge color="green">Thread Safe</Badge> Safe to call from multiple threads on different data.

### Complexity

<Badge type="complexity">Time: O(nnz)</Badge>
<Badge type="complexity">Space: O(1)</Badge>

<Callout type="note" title="CSR vs CSC">

- **CSR format**: primary = rows, secondary = cols
- **CSC format**: primary = cols, secondary = rows

</Callout>

<SourceLink file="scl/binding/c_api/algebra.h" :line="42" />

---

## scl_algebra_spmv_transpose

<Badge type="version">0.4.0</Badge>
<Badge type="status" color="green">Stable</Badge>

Transposed sparse matrix-vector multiplication: $y = \alpha \cdot A^T \cdot x + \beta \cdot y$

<ApiSignature return-type="scl_error_t" name="scl_algebra_spmv_transpose">
  <span class="type">scl_sparse_t</span> <span class="param-name">A</span>,
  <span class="keyword">const</span> <span class="type">scl_real_t</span>* <span class="param-name">x</span>,
  <span class="type">scl_size_t</span> <span class="param-name">x_size</span>,
  <span class="type">scl_real_t</span>* <span class="param-name">y</span>,
  <span class="type">scl_size_t</span> <span class="param-name">y_size</span>,
  <span class="type">scl_real_t</span> <span class="param-name">alpha</span>,
  <span class="type">scl_real_t</span> <span class="param-name">beta</span>
</ApiSignature>

### Parameters

<ParamTable :params="[
  { name: 'A', type: 'scl_sparse_t', dir: 'in', description: 'Sparse matrix handle (CSR or CSC format)' },
  { name: 'x', type: 'const scl_real_t*', dir: 'in', description: 'Input vector of size [primary_dim]' },
  { name: 'x_size', type: 'scl_size_t', dir: 'in', description: 'Size of input vector' },
  { name: 'y', type: 'scl_real_t*', dir: 'inout', description: 'Output vector of size [secondary_dim]' },
  { name: 'y_size', type: 'scl_size_t', dir: 'in', description: 'Size of output vector' },
  { name: 'alpha', type: 'scl_real_t', dir: 'in', description: 'Scaling factor for Aᵀ·x', default: '1.0' },
  { name: 'beta', type: 'scl_real_t', dir: 'in', description: 'Scaling factor for y', default: '0.0' }
]" />

### Returns

`scl_error_t` - `SCL_OK` on success, error code otherwise.

### Errors

| Code | Condition |
|------|-----------|
| `SCL_ERROR_NULL_POINTER` | If `A`, `x`, or `y` is NULL |
| `SCL_ERROR_DIMENSION_MISMATCH` | If dimensions don't match transposed operation |

### Thread Safety

<Badge color="green">Thread Safe</Badge> Safe to call from multiple threads on different data.

### Complexity

<Badge type="complexity">Time: O(nnz)</Badge>
<Badge type="complexity">Space: O(1)</Badge>

<SourceLink file="scl/binding/c_api/algebra.h" :line="68" />

---

## scl_algebra_spmm

<Badge type="version">0.4.0</Badge>
<Badge type="status" color="green">Stable</Badge>

Sparse matrix - dense matrix multiplication: $C = \alpha \cdot A \cdot B + \beta \cdot C$

<ApiSignature return-type="scl_error_t" name="scl_algebra_spmm">
  <span class="type">scl_sparse_t</span> <span class="param-name">A</span>,
  <span class="type">scl_dense_t</span> <span class="param-name">B</span>,
  <span class="type">scl_dense_t</span> <span class="param-name">C</span>,
  <span class="type">scl_real_t</span> <span class="param-name">alpha</span>,
  <span class="type">scl_real_t</span> <span class="param-name">beta</span>
</ApiSignature>

### Parameters

<ParamTable :params="[
  { name: 'A', type: 'scl_sparse_t', dir: 'in', description: 'Sparse matrix [M × K]' },
  { name: 'B', type: 'scl_dense_t', dir: 'in', description: 'Dense matrix [K × N]' },
  { name: 'C', type: 'scl_dense_t', dir: 'inout', description: 'Output dense matrix [M × N]' },
  { name: 'alpha', type: 'scl_real_t', dir: 'in', description: 'Scaling factor for A·B', default: '1.0' },
  { name: 'beta', type: 'scl_real_t', dir: 'in', description: 'Scaling factor for C', default: '0.0' }
]" />

### Returns

`scl_error_t` - `SCL_OK` on success, error code otherwise.

### Errors

| Code | Condition |
|------|-----------|
| `SCL_ERROR_NULL_POINTER` | If `A`, `B`, or `C` is NULL |
| `SCL_ERROR_DIMENSION_MISMATCH` | If matrix dimensions are incompatible |

### Thread Safety

<Badge color="yellow">Conditionally Safe</Badge> Safe if `C` is not shared across threads.

### Complexity

<Badge type="complexity">Time: O(nnz × N)</Badge>
<Badge type="complexity">Space: O(1)</Badge>

<SourceLink file="scl/binding/c_api/algebra.h" :line="95" />

---

## scl_algebra_row_norms

<Badge type="version">0.4.0</Badge>
<Badge type="status" color="green">Stable</Badge>

Compute L2 norm of each row: $\text{norms}[i] = \|A[i, :]\|_2$

<ApiSignature return-type="scl_error_t" name="scl_algebra_row_norms">
  <span class="type">scl_sparse_t</span> <span class="param-name">A</span>,
  <span class="type">scl_real_t</span>* <span class="param-name">norms</span>,
  <span class="type">scl_size_t</span> <span class="param-name">norms_size</span>
</ApiSignature>

### Parameters

<ParamTable :params="[
  { name: 'A', type: 'scl_sparse_t', dir: 'in', description: 'Sparse matrix (must be CSR format)' },
  { name: 'norms', type: 'scl_real_t*', dir: 'out', description: 'Output array for row norms' },
  { name: 'norms_size', type: 'scl_size_t', dir: 'in', description: 'Size of norms array (must equal number of rows)' }
]" />

### Returns

`scl_error_t` - `SCL_OK` on success, error code otherwise.

### Errors

| Code | Condition |
|------|-----------|
| `SCL_ERROR_NULL_POINTER` | If `A` or `norms` is NULL |
| `SCL_ERROR_DIMENSION_MISMATCH` | If `norms_size` ≠ number of rows |
| `SCL_ERROR_INVALID_ARGUMENT` | If matrix is not CSR format |

### Thread Safety

<Badge color="green">Thread Safe</Badge> Read-only operation on matrix.

### Complexity

<Badge type="complexity">Time: O(nnz)</Badge>
<Badge type="complexity">Space: O(1)</Badge>

<SourceLink file="scl/binding/c_api/algebra.h" :line="120" />

---

## scl_algebra_scale_rows

<Badge type="version">0.4.0</Badge>
<Badge type="status" color="green">Stable</Badge>

Scale each row by a factor: $A[i, :] \leftarrow A[i, :] \cdot \text{scales}[i]$

<ApiSignature return-type="scl_error_t" name="scl_algebra_scale_rows">
  <span class="type">scl_sparse_t</span> <span class="param-name">A</span>,
  <span class="keyword">const</span> <span class="type">scl_real_t</span>* <span class="param-name">scales</span>,
  <span class="type">scl_size_t</span> <span class="param-name">scales_size</span>
</ApiSignature>

### Parameters

<ParamTable :params="[
  { name: 'A', type: 'scl_sparse_t', dir: 'inout', description: 'Sparse matrix to scale (must be CSR format)' },
  { name: 'scales', type: 'const scl_real_t*', dir: 'in', description: 'Scaling factors for each row' },
  { name: 'scales_size', type: 'scl_size_t', dir: 'in', description: 'Size of scales array (must equal number of rows)' }
]" />

### Returns

`scl_error_t` - `SCL_OK` on success, error code otherwise.

### Errors

| Code | Condition |
|------|-----------|
| `SCL_ERROR_NULL_POINTER` | If `A` or `scales` is NULL |
| `SCL_ERROR_DIMENSION_MISMATCH` | If `scales_size` ≠ number of rows |
| `SCL_ERROR_INVALID_ARGUMENT` | If matrix is not CSR format |

### Thread Safety

<Badge color="red">Not Thread Safe</Badge> Modifies matrix data in-place.

### Complexity

<Badge type="complexity">Time: O(nnz)</Badge>
<Badge type="complexity">Space: O(1)</Badge>

<Callout type="warning">
This operation modifies the matrix in-place. Make a copy first if you need to preserve the original.
</Callout>

<SourceLink file="scl/binding/c_api/algebra.h" :line="145" />

---

## See Also

<SeeAlso :links="[
  { href: '/api/c-api/core', text: 'Core types and error handling' },
  { href: '/api/c-api/sparse', text: 'Sparse matrix operations' },
  { href: '/api/c-api/normalize', text: 'Normalization functions' }
]" />
