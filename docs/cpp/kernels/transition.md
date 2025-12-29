---
title: Cell State Transitions
description: Transition analysis for trajectory inference
---

# Cell State Transitions

The `transition` kernel provides efficient cell state transition analysis (CellRank-style).

## Overview

Transition analysis is used for:
- Cell fate prediction
- Transition probability computation
- Lineage driver identification
- State transition visualization

## Transition Types

```cpp
enum class TransitionType {
    Forward,    // Forward transitions
    Backward,   // Backward transitions
    Symmetric   // Symmetric transitions
};
```

## Functions

### `compute_transition_matrix`

Compute cell-to-cell transition matrix.

```cpp
template <typename T, bool IsCSR>
CSR compute_transition_matrix(
    const Sparse<T, IsCSR>& velocity_graph,
    TransitionType type = TransitionType::Forward
);
```

**Parameters**:
- `velocity_graph` [in]: Velocity-based graph
- `type` [in]: Transition type

**Returns**: Transition probability matrix

**Example**:
```cpp
#include "scl/kernel/transition.hpp"

CSR transitions = kernel::transition::compute_transition_matrix(
    velocity_graph,
    TransitionType::Forward
);
```

## Configuration

```cpp
namespace scl::kernel::transition::config {
    constexpr Real DEFAULT_TOLERANCE = 1e-6;
    constexpr Index DEFAULT_MAX_ITER = 1000;
    constexpr Real SOR_OMEGA = 1.5;  // Over-relaxation factor
    constexpr Size PARALLEL_THRESHOLD = 256;
}
```

## Related Documentation

- [Velocity](./velocity.md) - RNA velocity
- [Pseudotime](./pseudotime.md) - Trajectory analysis
