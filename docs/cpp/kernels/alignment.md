---
title: Data Alignment
description: Multi-modal data alignment and batch integration
---

# Data Alignment

The `alignment` kernel provides efficient multi-modal data alignment and batch integration methods, including MNN-based correction and anchor-based integration.

## Overview

Data alignment is essential for:
- Batch correction across datasets
- Multi-modal data integration
- Label transfer between datasets
- Integration quality assessment

## Functions

### Cross-Dataset KNN

#### `find_cross_knn`

Find k nearest neighbors from dataset2 for each cell in dataset1.

```cpp
template <typename T, bool IsCSR1, bool IsCSR2>
void find_cross_knn(
    const Sparse<T, IsCSR1>& data1,
    const Sparse<T, IsCSR2>& data2,
    Index k,
    Index* knn_indices,   // [n1 * k]
    Real* knn_distances   // [n1 * k]
);
```

**Parameters**:
- `data1` [in]: First dataset (reference)
- `data2` [in]: Second dataset (query)
- `k` [in]: Number of neighbors
- `knn_indices` [out]: Indices of neighbors in data2
- `knn_distances` [out]: Distances to neighbors

**Example**:
```cpp
#include "scl/kernel/alignment.hpp"

// Find cross-dataset neighbors
constexpr Index K = 30;
auto knn_indices = memory::aligned_alloc<Index>(data1.rows() * K);
auto knn_distances = memory::aligned_alloc<Real>(data1.rows() * K);

kernel::alignment::detail::find_cross_knn(
    data1, data2, K,
    knn_indices.get(), knn_distances.get()
);
```

### MNN-Based Integration

#### `mnn_correct`

Mutual Nearest Neighbors (MNN) based batch correction.

```cpp
template <typename T, bool IsCSR>
void mnn_correct(
    const Sparse<T, IsCSR>& data1,
    const Sparse<T, IsCSR>& data2,
    Array<Index> anchors,
    Sparse<T, IsCSR>& corrected_data
);
```

**Parameters**:
- `data1` [in]: Reference dataset
- `data2` [in]: Dataset to correct
- `anchors` [in]: MNN anchor pairs
- `corrected_data` [out]: Corrected data2

**Example**:
```cpp
// Find MNN anchors
auto anchors = find_mnn_anchors(data1, data2, k=30);

// Correct batch effects
CSR corrected = CSR::create(data2.rows(), data2.cols(), data2.nnz());
kernel::alignment::mnn_correct(data1, data2, anchors, corrected);
```

## Configuration

```cpp
namespace scl::kernel::alignment::config {
    constexpr Real EPSILON = 1e-10;
    constexpr Size DEFAULT_K = 30;
    constexpr Real ANCHOR_SCORE_THRESHOLD = 0.5;
    constexpr Size MAX_ANCHORS_PER_CELL = 10;
    constexpr Size PARALLEL_THRESHOLD = 32;
}
```

## Related Documentation

- [Neighbors](./neighbors.md) - Neighbor search
- [Kernels Overview](./overview.md) - General kernel usage
