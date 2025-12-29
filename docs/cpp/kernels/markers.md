---
title: Marker Gene Detection
description: Marker gene selection and specificity scoring
---

# Marker Gene Detection

The `markers` kernel provides efficient methods for identifying marker genes that distinguish cell groups.

## Overview

Marker gene detection is used for:
- Cell type identification
- Differential expression analysis
- Signature gene discovery
- Annotation transfer

## Functions

### `find_markers`

Find marker genes for each group.

```cpp
template <typename T, bool IsCSR>
void find_markers(
    const Sparse<T, IsCSR>& matrix,
    Array<const Index> group_labels,
    Array<Index> marker_genes,
    RankingMethod method = RankingMethod::Combined
);
```

**Parameters**:
- `matrix` [in]: Expression matrix
- `group_labels` [in]: Group assignment for each cell
- `marker_genes` [out]: Indices of marker genes
- `method` [in]: Ranking method

**Example**:
```cpp
#include "scl/kernel/markers.hpp"

Array<Index> group_labels = {labels_ptr, n_cells};
auto markers = memory::aligned_alloc<Index>(n_top);
Array<Index> markers_view = {markers.get(), n_top};

kernel::markers::find_markers(
    matrix, group_labels, markers_view,
    RankingMethod::Combined
);
```

## Ranking Methods

```cpp
enum class RankingMethod {
    FoldChange,    // Fold change ranking
    EffectSize,    // Cohen's d effect size
    PValue,        // P-value ranking
    Combined       // Combined score
};
```

## Configuration

```cpp
namespace scl::kernel::markers::config {
    constexpr Real DEFAULT_MIN_FC = 1.5;
    constexpr Real DEFAULT_MIN_PCT = 0.1;
    constexpr Real DEFAULT_MAX_PVAL = 0.05;
    constexpr Size PARALLEL_THRESHOLD = 500;
}
```

## Related Documentation

- [Statistics](./statistics.md) - Statistical tests
- [Scoring](./scoring.md) - Gene set scoring
