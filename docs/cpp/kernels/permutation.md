---
title: Permutation Tests
description: Permutation testing framework for nonparametric inference
---

# Permutation Tests

The `permutation` kernel provides efficient permutation testing methods with SIMD optimization.

## Overview

Permutation tests are used for:
- Non-parametric hypothesis testing
- Correlation significance testing
- Group comparison
- P-value computation

## Functions

### `permutation_test`

Perform permutation test for two groups.

```cpp
template <typename T>
Real permutation_test(
    Array<const T> group1,
    Array<const T> group2,
    Size n_permutations = config::DEFAULT_N_PERMUTATIONS
);
```

**Parameters**:
- `group1` [in]: First group data
- `group2` [in]: Second group data
- `n_permutations` [in]: Number of permutations (default: 1000)

**Returns**: P-value

**Example**:
```cpp
#include "scl/kernel/permutation.hpp"

Array<Real> group1 = {data1, n1};
Array<Real> group2 = {data2, n2};
Real p_value = kernel::permutation::permutation_test(group1, group2, 1000);
```

## Configuration

```cpp
namespace scl::kernel::permutation::config {
    constexpr Size DEFAULT_N_PERMUTATIONS = 1000;
    constexpr Size MIN_PERMUTATIONS = 100;
    constexpr Size MAX_PERMUTATIONS = 100000;
    constexpr Size PARALLEL_THRESHOLD = 500;
}
```

## Related Documentation

- [Statistics](./statistics.md) - Statistical tests
- [Kernels Overview](./overview.md) - General kernel usage
