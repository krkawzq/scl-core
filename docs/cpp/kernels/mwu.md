---
title: Mann-Whitney U Test
description: Non-parametric statistical test for group comparison
---

# Mann-Whitney U Test

The `mwu` kernel provides efficient Mann-Whitney U test implementation with optimized algorithms.

## Overview

Mann-Whitney U test is a non-parametric alternative to t-test:
- No assumption of normal distribution
- Robust to outliers
- Suitable for small sample sizes
- Tests if two groups have different distributions

## Functions

### `mann_whitney_u`

Compute Mann-Whitney U statistic and p-value.

```cpp
template <typename T>
Real mann_whitney_u(
    Array<const T> group1,
    Array<const T> group2
);
```

**Parameters**:
- `group1` [in]: First group data
- `group2` [in]: Second group data

**Returns**: U statistic

**Example**:
```cpp
#include "scl/kernel/mwu.hpp"

Array<Real> group1 = {data1, n1};
Array<Real> group2 = {data2, n2};
Real u_stat = kernel::mwu::mann_whitney_u(group1, group2);
```

## Configuration

```cpp
namespace scl::kernel::mwu::config {
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Size BINARY_SEARCH_THRESHOLD = 32;
}
```

## Related Documentation

- [Statistics](./statistics.md) - Statistical tests
- [Kernels Overview](./overview.md) - General kernel usage
