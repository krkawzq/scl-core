---
title: Doublet Detection
description: Doublet detection for single-cell data
---

# Doublet Detection

The `doublet` kernel provides efficient doublet detection methods (Scrublet/DoubletFinder style).

## Overview

Doublet detection identifies:
- Cell doublets (two cells in one droplet)
- Multiplet artifacts
- Quality issues

## Detection Methods

```cpp
enum class DoubletMethod {
    Scrublet,      // Scrublet algorithm
    DoubletFinder, // DoubletFinder algorithm
    Hybrid         // Hybrid approach
};
```

## Functions

### `detect_doublets`

Detect doublets in single-cell data.

```cpp
template <typename T, bool IsCSR>
void detect_doublets(
    const Sparse<T, IsCSR>& matrix,
    Array<Real> doublet_scores,
    DoubletMethod method = DoubletMethod::Scrublet,
    Real doublet_rate = config::DEFAULT_DOUBLET_RATE
);
```

**Parameters**:
- `matrix` [in]: Expression matrix
- `doublet_scores` [out]: Doublet scores (higher = more likely doublet)
- `method` [in]: Detection method
- `doublet_rate` [in]: Expected doublet rate (default: 0.06)

**Example**:
```cpp
#include "scl/kernel/doublet.hpp"

auto scores = memory::aligned_alloc<Real>(matrix.rows());
Array<Real> scores_view = {scores.get(), static_cast<Size>(matrix.rows())};

kernel::doublet::detect_doublets(
    matrix, scores_view,
    DoubletMethod::Scrublet,
    doublet_rate=0.06
);
```

## Configuration

```cpp
namespace scl::kernel::doublet::config {
    constexpr Real DEFAULT_DOUBLET_RATE = 0.06;
    constexpr Real DEFAULT_THRESHOLD = 0.5;
    constexpr Index DEFAULT_N_NEIGHBORS = 30;
    constexpr Size PARALLEL_THRESHOLD = 500;
}
```

## Related Documentation

- [Outlier Detection](./outlier.md) - Outlier detection
- [QC](./qc.md) - Quality control
