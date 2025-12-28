# Tissue Architecture Analysis

Tissue layer assignment and zonation analysis for spatial organization.

## Overview

Tissue architecture kernels provide:

- **Layer Assignment** - Assign cells to tissue layers based on spatial coordinates
- **Zonation Scoring** - Compute zonation scores along spatial axes
- **Tissue Organization** - Analyze spatial organization patterns
- **Architectural Analysis** - Characterize tissue structure

## Layer Assignment

### layer_assignment

Assign cells to tissue layers based on spatial coordinates:

```cpp
#include "scl/kernel/tissue.hpp"

const Real* coordinates = /* ... */;           // Spatial coordinates [n_cells * n_dims]
Size n_cells = /* ... */;
Size n_dims = 2;                                // Typically 2D or 3D
Index n_layers = 5;                             // Number of layers

Array<Index> layer_labels(n_cells);            // Pre-allocated output

scl::kernel::tissue::layer_assignment(
    coordinates, n_cells, n_dims,
    layer_labels,
    n_layers);

// layer_labels[i] contains layer ID (0 to n_layers-1) for cell i
```

**Parameters:**
- `coordinates`: Spatial coordinates, size = n_cells × n_dims
- `n_cells`: Number of cells
- `n_dims`: Number of spatial dimensions
- `layer_labels`: Output layer labels, must be pre-allocated, size = n_cells
- `n_layers`: Number of layers to assign

**Postconditions:**
- `layer_labels[i]` contains layer ID for cell i (0 to n_layers-1)
- Layers are assigned based on spatial position along primary axis
- Cells are evenly distributed across layers (approximately)

**Algorithm:**
- Project coordinates to primary axis
- Divide axis into n_layers segments
- Assign cells to layers based on position

**Complexity:**
- Time: O(n_cells * n_dims) - linear in cells
- Space: O(n_cells) auxiliary for labels

**Thread Safety:**
- Safe - parallelized over cells
- Each cell processed independently

**Use cases:**
- Tissue layer identification
- Stratified analysis (epidermis, dermis, etc.)
- Radial organization (cortex, medulla, etc.)
- Structural annotation

## Zonation Scoring

### zonation_score

Compute zonation score along a spatial axis:

```cpp
const Real* coordinates = /* ... */;
const Real* expression = /* ... */;            // Gene expression values [n_cells]
Size n_cells = /* ... */;
Size axis = 0;                                  // X-axis (0), Y-axis (1), or Z-axis (2)

Array<Real> scores(n_cells);                   // Pre-allocated output

scl::kernel::tissue::zonation_score(
    coordinates, expression,
    n_cells, axis,
    scores);

// scores[i] contains zonation score along specified axis for cell i
```

**Parameters:**
- `coordinates`: Spatial coordinates, size = n_cells × n_dims
- `expression`: Gene expression values, size = n_cells
- `n_cells`: Number of cells
- `axis`: Spatial axis index (0 = X, 1 = Y, 2 = Z)
- `scores`: Output zonation scores, must be pre-allocated, size = n_cells

**Postconditions:**
- `scores[i]` contains zonation score along specified axis for cell i
- Scores represent expression level relative to position along axis
- Useful for identifying gradient patterns

**Algorithm:**
- Project cells to specified axis
- Compute expression as function of axis position
- Generate zonation scores (may use smoothing or binning)

**Complexity:**
- Time: O(n_cells) - linear processing
- Space: O(n_cells) auxiliary for scores

**Thread Safety:**
- Safe - parallelized processing

**Use cases:**
- Portal-central zonation (liver)
- Cortical-medullary zonation (kidney)
- Apical-basal zonation (epithelia)
- Gradient pattern detection

## Configuration

### Default Parameters

```cpp
namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_CELLS_PER_LAYER = 5;
    constexpr Size DEFAULT_N_NEIGHBORS = 15;
    constexpr Size MAX_ITERATIONS = 100;
    constexpr Real PI = Real(3.14159265358979323846);
}
```

**Minimum Cells Per Layer:**
- Ensures each layer has sufficient cells for statistical analysis
- Adjust based on dataset size

**Spatial Axis:**
- Axis 0 = X (typically horizontal)
- Axis 1 = Y (typically vertical)
- Axis 2 = Z (depth, for 3D data)

## Examples

### Layer-Based Analysis

```cpp
#include "scl/kernel/tissue.hpp"

const Real* coords = /* ... */;  // [n_cells * 2]
Size n_cells = /* ... */;
Index n_layers = 4;  // e.g., epidermis, dermis, hypodermis, muscle

Array<Index> layers(n_cells);
scl::kernel::tissue::layer_assignment(
    coords, n_cells, 2,  // 2D coordinates
    layers,
    n_layers);

// Analyze expression per layer
Sparse<Real, true> expression = /* ... */;
for (Index layer = 0; layer < n_layers; ++layer) {
    // Extract cells in this layer
    std::vector<Index> layer_cells;
    for (Index i = 0; i < n_cells; ++i) {
        if (layers[i] == layer) {
            layer_cells.push_back(i);
        }
    }
    
    // Compute layer-specific statistics
    // ... analyze expression for layer_cells ...
}
```

### Zonation Analysis

```cpp
// Analyze zonation along X-axis (portal-central axis for liver)
const Real* expression = /* ... */;  // Expression of specific gene
Size axis = 0;  // X-axis

Array<Real> zonation(n_cells);
scl::kernel::tissue::zonation_score(
    coords, expression,
    n_cells, axis,
    zonation);

// Identify zones based on score thresholds
std::vector<Index> zone1, zone2, zone3;  // Portal, intermediate, central
Real threshold1 = 0.33;
Real threshold2 = 0.67;

for (Index i = 0; i < n_cells; ++i) {
    Real score = zonation[i];
    if (score < threshold1) {
        zone1.push_back(i);  // Portal zone
    } else if (score < threshold2) {
        zone2.push_back(i);  // Intermediate zone
    } else {
        zone3.push_back(i);  // Central zone
    }
}
```

### Multi-Axis Zonation

```cpp
// Analyze zonation along multiple axes
const Real* gene_expr = /* ... */;

// X-axis zonation
Array<Real> zonation_x(n_cells);
scl::kernel::tissue::zonation_score(coords, gene_expr, n_cells, 0, zonation_x);

// Y-axis zonation
Array<Real> zonation_y(n_cells);
scl::kernel::tissue::zonation_score(coords, gene_expr, n_cells, 1, zonation_y);

// Combined analysis (e.g., for radial patterns)
Array<Real> radial_pattern(n_cells);
for (Index i = 0; i < n_cells; ++i) {
    // Combine X and Y zonation scores
    Real x = zonation_x[i];
    Real y = zonation_y[i];
    radial_pattern[i] = std::sqrt(x * x + y * y);  // Distance from center
}
```

---

::: tip Layer vs. Zonation
Use layer assignment for discrete tissue regions, use zonation scoring for continuous gradients. Layers are categorical, zonation is quantitative.
:::

