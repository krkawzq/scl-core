---
title: Diffusion Maps
description: Diffusion-based dimensionality reduction
---

# Diffusion Maps

The `diffusion` kernel provides efficient diffusion map computation for dimensionality reduction.

## Overview

Diffusion maps are used for:
- Non-linear dimensionality reduction
- Trajectory inference
- Manifold learning
- Data visualization

## Functions

### `diffusion_map`

Compute diffusion map embedding.

```cpp
template <typename T, bool IsCSR>
void diffusion_map(
    const Sparse<T, IsCSR>& graph,
    Array<Real> embedding,
    Index n_components,
    Index n_steps = config::DEFAULT_N_STEPS,
    Real alpha = config::DEFAULT_ALPHA
);
```

**Parameters**:
- `graph` [in]: Affinity graph
- `embedding` [out]: Diffusion map embedding
- `n_components` [in]: Number of components
- `n_steps` [in]: Diffusion steps (default: 3)
- `alpha` [in]: Diffusion parameter (default: 0.85)

**Example**:
```cpp
#include "scl/kernel/diffusion.hpp"

auto embedding = memory::aligned_alloc<Real>(n_cells * n_components);
Array<Real> emb_view = {embedding.get(), n_cells * n_components};

kernel::diffusion::diffusion_map(
    graph, emb_view, n_components=10,
    n_steps=3, alpha=0.85
);
```

## Configuration

```cpp
namespace scl::kernel::diffusion::config {
    constexpr Index DEFAULT_N_STEPS = 3;
    constexpr Real DEFAULT_ALPHA = 0.85;
    constexpr Real CONVERGENCE_TOL = 1e-6;
    constexpr Size PARALLEL_THRESHOLD = 256;
}
```

## Related Documentation

- [Projection](./projection.md) - Random projection
- [Pseudotime](./pseudotime.md) - Trajectory analysis
