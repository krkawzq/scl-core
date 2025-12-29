---
title: Gene Set Scoring
description: High-performance gene set scoring and signature analysis
---

# Gene Set Scoring

The `scoring` kernel provides efficient gene set scoring methods for cell signature analysis, optimized with SIMD and parallel processing.

## Overview

Gene set scoring is used for:
- Cell type identification
- Pathway activity scoring
- Signature analysis
- Cell cycle phase detection

## Scoring Methods

```cpp
enum class ScoringMethod {
    Mean,           // Simple mean
    RankBased,      // Rank-based scoring
    Weighted,       // Weighted scoring
    SeuratModule,  // Seurat module scoring
    ZScore          // Z-score based
};
```

## Functions

### `score_cells`

Score cells using a gene set.

```cpp
template <typename T, bool IsCSR>
void score_cells(
    const Sparse<T, IsCSR>& matrix,
    Array<const Index> gene_set,
    Array<Real> scores,
    ScoringMethod method = ScoringMethod::SeuratModule
);
```

**Parameters**:
- `matrix` [in]: Expression matrix
- `gene_set` [in]: Indices of genes in the set
- `scores` [out]: Cell scores (length = matrix.rows())
- `method` [in]: Scoring method

**Example**:
```cpp
#include "scl/kernel/scoring.hpp"

// Define gene set
Array<Index> marker_genes = {gene1_idx, gene2_idx, gene3_idx};

// Score cells
auto scores = memory::aligned_alloc<Real>(matrix.rows());
Array<Real> scores_view = {scores.get(), static_cast<Size>(matrix.rows())};

kernel::scoring::score_cells(
    matrix, marker_genes, scores_view,
    ScoringMethod::SeuratModule
);
```

## Configuration

```cpp
namespace scl::kernel::scoring::config {
    constexpr Index DEFAULT_N_CONTROL = 100;
    constexpr Index DEFAULT_N_BINS = 25;
    constexpr Real DEFAULT_QUANTILE = 0.05;
    constexpr Real MIN_VAR = 1e-9;
    constexpr Size PARALLEL_THRESHOLD = 128;
}
```

## Related Documentation

- [Markers](./markers.md) - Marker gene detection
- [Kernels Overview](./overview.md) - General kernel usage
