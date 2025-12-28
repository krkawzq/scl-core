# tissue.hpp

> scl/kernel/tissue.hpp · Tissue architecture and organization analysis

## Overview

This file provides functions for analyzing tissue architecture and spatial organization in spatial transcriptomics data. It includes layer assignment and zonation scoring along spatial axes.

**Header**: `#include "scl/kernel/tissue.hpp"`

Key features:
- Layer assignment based on spatial coordinates
- Zonation score computation along spatial axes
- Spatial pattern analysis

---

## Main APIs

### layer_assignment

::: source_code file="scl/kernel/tissue.hpp" symbol="layer_assignment" collapsed
:::

**Algorithm Description**

Assign cells to tissue layers based on spatial coordinates:

1. For each cell i:
   - Compute distance from cell to tissue boundary/center
   - Determine layer based on distance quantiles
   - Assign layer label: `layer_labels[i] = layer_id` in [0, n_layers-1]
2. Layers are assigned using equal-width or equal-frequency binning
3. Uses parallel processing over cells

**Edge Cases**

- **n_layers = 1**: All cells assigned to layer 0
- **n_layers = 0**: Undefined behavior
- **All cells at same position**: All assigned to same layer
- **Outlier coordinates**: May affect layer boundaries

**Data Guarantees (Preconditions)**

- `layer_labels` has capacity >= n_cells
- `coordinates` is row-major: `coordinates[i * n_dims + j]` is dimension j of cell i
- `n_layers > 0`
- `n_dims` matches coordinate array dimensionality

**Complexity Analysis**

- **Time**: O(n_cells * n_dims) - distance computation per cell
- **Space**: O(n_cells) auxiliary space

**Example**

```cpp
#include "scl/kernel/tissue.hpp"

const Real* coordinates = /* spatial coordinates [n_cells * n_dims] */;
scl::Array<Index> layer_labels(n_cells);

scl::kernel::tissue::layer_assignment(
    coordinates, n_cells, n_dims,
    layer_labels, 5  // n_layers
);

// layer_labels[i] contains layer ID (0-4) for cell i
```

---

### zonation_score

::: source_code file="scl/kernel/tissue.hpp" symbol="zonation_score" collapsed
:::

**Algorithm Description**

Compute zonation score along a spatial axis:

1. For each cell i:
   - Extract spatial coordinate along specified axis
   - Compute correlation between expression and spatial position
   - Score = normalized correlation (Pearson or Spearman)
2. Zonation score indicates gradient strength along axis
3. Higher scores indicate stronger spatial gradients

**Edge Cases**

- **axis >= n_dims**: Undefined behavior
- **Constant expression**: All scores are 0
- **Constant coordinates**: All scores are 0
- **Perfect gradient**: Scores approach 1.0

**Data Guarantees (Preconditions)**

- `scores` has capacity >= n_cells
- `axis < n_dims`
- `expression` and `coordinates` have matching n_cells
- Coordinates are row-major layout

**Complexity Analysis**

- **Time**: O(n_cells) - correlation computation
- **Space**: O(n_cells) auxiliary space

**Example**

```cpp
const Real* expression = /* gene expression [n_cells] */;
scl::Array<Real> scores(n_cells);

scl::kernel::tissue::zonation_score(
    coordinates, expression, n_cells,
    0,      // axis: 0 = x-axis, 1 = y-axis, 2 = z-axis
    scores
);

// scores[i] contains zonation score along specified axis
// Higher scores indicate stronger spatial gradient
```

---

## Configuration

Default parameters are defined in `scl::kernel::tissue::config`:

- `EPSILON = 1e-10`: Numerical tolerance
- `MIN_CELLS_PER_LAYER = 5`: Minimum cells required per layer
- `DEFAULT_N_NEIGHBORS = 15`: Default neighbors for spatial analysis
- `MAX_ITERATIONS = 100`: Maximum iterations for iterative methods
- `PI = 3.14159...`: Mathematical constant

---

## Notes

- Layer assignment assumes tissue has layered structure (e.g., cortex layers)
- Zonation scores are useful for identifying spatial expression gradients
- Spatial coordinates should be in consistent units (e.g., micrometers)
- Multiple axes can be analyzed separately for 3D tissue

## See Also

- [Spatial Module](./spatial) - For additional spatial analysis methods
- [Spatial Pattern Module](./spatial_pattern) - For pattern detection
