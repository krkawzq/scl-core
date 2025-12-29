---
title: RNA Velocity
description: RNA velocity analysis for trajectory inference
---

# RNA Velocity

The `velocity` kernel provides efficient RNA velocity analysis for single-cell transcriptomics data.

## Overview

RNA velocity analysis:
- Predicts future cell states
- Infers developmental trajectories
- Models transcriptional dynamics
- Enables trajectory visualization

## Velocity Models

```cpp
enum class VelocityModel {
    SteadyState,   // Steady-state model
    Dynamical,     // Dynamical model
    Stochastic     // Stochastic model
};
```

## Functions

### `compute_velocity`

Compute RNA velocity vectors.

```cpp
template <typename T, bool IsCSR>
void compute_velocity(
    const Sparse<T, IsCSR>& spliced,
    const Sparse<T, IsCSR>& unspliced,
    Array<Real> velocity,
    VelocityModel model = VelocityModel::SteadyState
);
```

**Parameters**:
- `spliced` [in]: Spliced RNA counts
- `unspliced` [in]: Unspliced RNA counts
- `velocity` [out]: Velocity vectors
- `model` [in]: Velocity model

**Example**:
```cpp
#include "scl/kernel/velocity.hpp"

auto velocity = memory::aligned_alloc<Real>(n_cells * n_genes);
Array<Real> vel_view = {velocity.get(), n_cells * n_genes};

kernel::velocity::compute_velocity(
    spliced, unspliced, vel_view,
    VelocityModel::SteadyState
);
```

## Configuration

```cpp
namespace scl::kernel::velocity::config {
    constexpr Real DEFAULT_MIN_R2 = 0.01;
    constexpr Index DEFAULT_N_NEIGHBORS = 30;
    constexpr Real DEFAULT_ALPHA = 0.05;
    constexpr Size PARALLEL_THRESHOLD = 500;
}
```

## Related Documentation

- [Pseudotime](./pseudotime.md) - Trajectory analysis
- [Diffusion](./diffusion.md) - Diffusion maps
