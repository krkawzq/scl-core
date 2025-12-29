---
title: Pseudotime Inference
description: Trajectory analysis and pseudotime computation
---

# Pseudotime Inference

The `pseudotime` kernel provides efficient pseudotime inference methods for trajectory analysis.

## Overview

Pseudotime inference is used for:
- Trajectory reconstruction
- Developmental ordering
- Temporal dynamics analysis
- Cell fate prediction

## Methods

```cpp
enum class PseudotimeMethod {
    DiffusionPseudotime,  // Diffusion-based
    ShortestPath,          // Graph shortest path
    GraphDistance,         // Graph distance
    WatershedDescent       // Watershed algorithm
};
```

## Functions

### `compute_pseudotime`

Compute pseudotime for cells.

```cpp
template <typename T, bool IsCSR>
void compute_pseudotime(
    const Sparse<T, IsCSR>& graph,
    Array<const Index> root_cells,
    Array<Real> pseudotime,
    PseudotimeMethod method = PseudotimeMethod::DiffusionPseudotime
);
```

**Parameters**:
- `graph` [in]: Cell-cell graph
- `root_cells` [in]: Root cell indices
- `pseudotime` [out]: Pseudotime values
- `method` [in]: Computation method

**Example**:
```cpp
#include "scl/kernel/pseudotime.hpp"

Array<Index> roots = {root_indices, n_roots};
auto pseudotime = memory::aligned_alloc<Real>(n_cells);
Array<Real> pt_view = {pseudotime.get(), n_cells};

kernel::pseudotime::compute_pseudotime(
    graph, roots, pt_view,
    PseudotimeMethod::DiffusionPseudotime
);
```

## Configuration

```cpp
namespace scl::kernel::pseudotime::config {
    constexpr Index DEFAULT_N_DCS = 10;
    constexpr Real DEFAULT_DAMPING = 0.85;
    constexpr Real CONVERGENCE_TOL = 1e-6;
    constexpr Size PARALLEL_THRESHOLD = 256;
}
```

## Related Documentation

- [Diffusion](./diffusion.md) - Diffusion maps
- [Neighbors](./neighbors.md) - Neighbor search
