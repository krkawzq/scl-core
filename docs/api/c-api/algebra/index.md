---
title: scl_algebra - Sparse Linear Algebra
description: High-performance sparse matrix operations
---

<script setup>
const moduleInfo = {
  name: 'algebra',
  header: 'scl/binding/c_api/algebra.h',
  version: '0.4.0',
  status: 'stable',
  functionCount: 4
}

const functions = [
  {
    name: 'scl_algebra_spmv',
    brief: 'Sparse matrix-vector multiplication',
    formula: 'y = αAx + βy',
    complexity: 'O(nnz)',
    status: 'stable',
    href: './spmv'
  },
  {
    name: 'scl_algebra_spmm',
    brief: 'Sparse-dense matrix-matrix multiplication',
    formula: 'C = AB',
    complexity: 'O(nnz×k)',
    status: 'stable',
    href: './spmm'
  },
  {
    name: 'scl_algebra_spmv_t',
    brief: 'Transposed sparse matrix-vector multiplication',
    formula: 'y = αAᵀx + βy',
    complexity: 'O(nnz)',
    status: 'stable',
    href: './spmv_t'
  },
  {
    name: 'scl_algebra_axpy',
    brief: 'Vector addition with scaling',
    formula: 'y = αx + y',
    complexity: 'O(n)',
    status: 'stable',
    href: './axpy'
  }
]
</script>

# scl_algebra

<div class="module-header">
  <Badge type="version">v{{ moduleInfo.version }}</Badge>
  <Badge type="status" color="green">{{ moduleInfo.status }}</Badge>
  <span class="function-count">{{ moduleInfo.functionCount }} functions</span>
</div>

High-performance sparse linear algebra operations for scientific computing and machine learning.

---

## Overview

The `scl_algebra` module provides optimized implementations of fundamental sparse matrix operations. These operations form the building blocks for:

- **Iterative solvers** (CG, GMRES, BiCGSTAB)
- **Eigenvalue algorithms** (Power iteration, Lanczos)
- **Graph algorithms** (PageRank, spectral clustering)
- **Machine learning** (sparse neural networks, GNNs)

### Supported Matrix Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| **CSR** | Compressed Sparse Row | Row-oriented access, SpMV |
| **CSC** | Compressed Sparse Column | Column-oriented access, transposed ops |

### Performance Characteristics

<Callout type="info" title="Optimization Features">

- **SIMD vectorization** on x86-64 (AVX2/AVX-512)
- **Cache-optimized** memory access patterns
- **Zero heap allocations** in hot paths
- **Thread-safe** read operations

</Callout>

---

## Functions

<div class="function-grid">
  <div v-for="func in functions" :key="func.name" class="function-card">
    <div class="function-card__header">
      <a :href="func.href" class="function-card__name">{{ func.name }}</a>
      <Badge type="status" color="green" v-if="func.status === 'stable'">stable</Badge>
      <Badge type="status" color="yellow" v-else-if="func.status === 'beta'">beta</Badge>
    </div>
    <div class="function-card__brief">{{ func.brief }}</div>
    <div class="function-card__meta">
      <span class="formula">{{ func.formula }}</span>
      <Badge type="complexity">{{ func.complexity }}</Badge>
    </div>
  </div>
</div>

---

## Quick Reference

### Function Summary

| Function | Operation | Time | Space |
|----------|-----------|------|-------|
| [`spmv`](./spmv) | y = αAx + βy | O(nnz) | O(1) |
| [`spmm`](./spmm) | C = AB | O(nnz×k) | O(1) |
| [`spmv_t`](./spmv_t) | y = αAᵀx + βy | O(nnz) | O(1) |
| [`axpy`](./axpy) | y = αx + y | O(n) | O(1) |

### Common Error Codes

| Code | Meaning |
|------|---------|
| `SCL_OK` | Success |
| `SCL_ERROR_NULL_POINTER` | Required pointer is NULL |
| `SCL_ERROR_DIMENSION_MISMATCH` | Matrix/vector dimensions incompatible |
| `SCL_ERROR_INVALID_FORMAT` | Unsupported matrix format |

---

## Usage Example

```c
#include <scl/binding/c_api/algebra.h>
#include <scl/binding/c_api/sparse.h>

// Create a sparse matrix
scl_sparse_t A = scl_sparse_create_csr(
    m, n, nnz, row_ptr, col_idx, values
);

// Allocate vectors
scl_real_t* x = malloc(n * sizeof(scl_real_t));
scl_real_t* y = malloc(m * sizeof(scl_real_t));

// Perform SpMV: y = A * x
scl_error_t err = scl_algebra_spmv(
    A, x, n, y, m, 1.0, 0.0
);

if (err != SCL_OK) {
    // Handle error
}

// Cleanup
scl_sparse_destroy(A);
free(x);
free(y);
```

---

## Header

```c
#include <scl/binding/c_api/algebra.h>
```

---

<SeeAlso :links="[
  { href: '../sparse', text: 'Sparse Matrix Types' },
  { href: '../core', text: 'Core Types and Errors' },
  { href: '../normalize', text: 'Normalization Functions' }
]" />

<style>
.module-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
}

.function-count {
  margin-left: auto;
  font-size: 14px;
  color: var(--vp-c-text-3);
}

.function-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
  margin: 24px 0;
}

.function-card {
  padding: 16px;
  background: var(--scl-card-bg);
  border: 1px solid var(--scl-card-border);
  border-radius: var(--scl-radius-lg);
  transition: all 0.2s;
}

.function-card:hover {
  border-color: var(--scl-primary);
  box-shadow: var(--scl-card-hover-shadow);
}

.function-card__header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.function-card__name {
  font-family: var(--scl-font-mono);
  font-weight: 600;
  color: var(--vp-c-text-1);
  text-decoration: none;
}

.function-card__name:hover {
  color: var(--scl-primary);
}

.function-card__brief {
  font-size: 14px;
  color: var(--vp-c-text-2);
  margin-bottom: 12px;
}

.function-card__meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.formula {
  font-family: var(--scl-font-mono);
  font-size: 12px;
  color: var(--vp-c-text-3);
}
</style>
