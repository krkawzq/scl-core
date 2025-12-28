# group_partition.hpp

> scl/kernel/stat/group_partition.hpp · Group partitioning utilities for statistical tests

## Overview

This file provides efficient utilities for partitioning sparse matrix values into groups with moment accumulation:

- **Two-Group Partitioning**: Partition values into two groups with sum accumulation
- **Moment Accumulation**: Compute sums and sum-of-squares in single pass
- **K-Group Partitioning**: Partition into multiple groups for ANOVA
- **Statistics Finalization**: Compute mean and variance from accumulated moments

These utilities are building blocks for statistical tests (t-test, ANOVA, etc.) and are optimized for sparse data.

**Header**: `#include "scl/kernel/stat/group_partition.hpp"`

---

## Main APIs

### partition_two_groups

::: source_code file="scl/kernel/stat/group_partition.hpp" symbol="partition_two_groups" collapsed
:::

**Algorithm Description**

Partition sparse row/column values into two groups with sum accumulation:

1. **4-way unrolled iteration**: Process 4 elements at a time for efficiency
2. **Indirect access with prefetch**: 
   - Load group assignment: group = group_ids[indices[k]]
   - Prefetch next group assignment for cache optimization
3. **Partition and accumulate**: For each value
   - If group == 0: append to buf1, increment n1, add to sum1
   - If group == 1: append to buf2, increment n2, add to sum2
4. **Handle remaining elements**: Process any remaining elements after unrolled loop

The 4-way unroll and prefetching optimize cache performance for indirect memory access patterns.

**Edge Cases**

- **Invalid group IDs**: Negative or out-of-range IDs are ignored
- **Empty groups**: n1 or n2 may be 0
- **All zeros**: Sums are 0, but values still partitioned
- **Sparse indices**: Only processes non-zero elements

**Data Guarantees (Preconditions)**

- `buf1` and `buf2` have sufficient capacity (>= len)
- `group_ids[indices[k]]` is 0 or 1 for valid elements
- `indices` and `values` have same length

**Complexity Analysis**

- **Time**: O(len) with 4-way unroll optimization
- **Space**: O(1) auxiliary space

**Example**

```cpp
#include "scl/kernel/stat/group_partition.hpp"

// Sparse row values and indices
const Real* values = /* ... */;      // Sparse values
const Index* indices = /* ... */;    // Column indices
Size len = /* ... */;                 // Number of non-zeros

// Group assignment for each column
const int32_t* group_ids = /* ... */;  // [n_cols], 0 or 1

// Output buffers
Real buf1[len], buf2[len];
Size n1 = 0, n2 = 0;
double sum1 = 0.0, sum2 = 0.0;

// Partition values into two groups
scl::kernel::stat::partition::partition_two_groups(
    values, indices, len, group_ids,
    buf1, n1, buf2, n2,
    sum1, sum2
);

// buf1[0:n1] contains group 0 values, sum1 = sum of group 0
// buf2[0:n2] contains group 1 values, sum2 = sum of group 1
```

---

### partition_two_groups_moments

::: source_code file="scl/kernel/stat/group_partition.hpp" symbol="partition_two_groups_moments" collapsed
:::

**Algorithm Description**

Partition with sum and sum-of-squares accumulation for t-test:

1. **Same partitioning logic** as `partition_two_groups`
2. **Additional accumulation**: For each value
   - Accumulate sum: sum += value
   - Accumulate sum-of-squares: sum_sq += value * value
3. **Separate accumulators**: Maintain sum1, sum_sq1, sum2, sum_sq2

This enables online variance computation: var = (sum_sq - sum * mean) / (n - 1)

**Edge Cases**

- **Same as partition_two_groups**: All edge cases apply
- **Overflow**: Sum-of-squares may overflow for very large values (uses double precision)

**Data Guarantees (Preconditions)**

- Same as `partition_two_groups`
- All output parameters have sufficient capacity

**Complexity Analysis**

- **Time**: O(len) with 4-way unroll
- **Space**: O(1) auxiliary space

**Example**

```cpp
#include "scl/kernel/stat/group_partition.hpp"

Real buf1[len], buf2[len];
Size n1 = 0, n2 = 0;
double sum1 = 0.0, sum_sq1 = 0.0;
double sum2 = 0.0, sum_sq2 = 0.0;

// Partition with moment accumulation
scl::kernel::stat::partition::partition_two_groups_moments(
    values, indices, len, group_ids,
    buf1, n1, buf2, n2,
    sum1, sum_sq1, sum2, sum_sq2
);

// Compute means and variances
double mean1 = sum1 / n1;
double var1 = (sum_sq1 - sum1 * mean1) / (n1 - 1);
double mean2 = sum2 / n2;
double var2 = (sum_sq2 - sum2 * mean2) / (n2 - 1);
```

---

### partition_k_groups_moments

::: source_code file="scl/kernel/stat/group_partition.hpp" symbol="partition_k_groups_moments" collapsed
:::

**Algorithm Description**

Partition into k groups with moment accumulation for ANOVA:

1. **Initialize accumulators**: For each group g
   - counts[g] = 0
   - sums[g] = 0.0
   - sum_sqs[g] = 0.0

2. **Iterate over values**: For each sparse value
   - Get group assignment: g = group_ids[indices[k]]
   - If g is valid (0 <= g < n_groups):
     - Increment counts[g]
     - Add value to sums[g]
     - Add value^2 to sum_sqs[g]

3. **No buffer allocation**: Only accumulates statistics, doesn't store values

This is more memory-efficient than storing all values, suitable for ANOVA where only group statistics are needed.

**Edge Cases**

- **Invalid group IDs**: Negative or >= n_groups are ignored
- **Empty groups**: counts[g] = 0, sums[g] = 0, sum_sqs[g] = 0
- **All values in one group**: Other groups have zero statistics

**Data Guarantees (Preconditions)**

- `counts`, `sums`, `sum_sqs` have at least `n_groups` elements
- `group_ids` values are in [0, n_groups) or negative (ignored)

**Complexity Analysis**

- **Time**: O(len) single pass
- **Space**: O(1) auxiliary space

**Example**

```cpp
#include "scl/kernel/stat/group_partition.hpp"

Size n_groups = 3;  // Three groups
Size counts[n_groups];
double sums[n_groups];
double sum_sqs[n_groups];

// Initialize
for (Size g = 0; g < n_groups; ++g) {
    counts[g] = 0;
    sums[g] = 0.0;
    sum_sqs[g] = 0.0;
}

// Partition into k groups with moment accumulation
scl::kernel::stat::partition::partition_k_groups_moments(
    values, indices, len, group_ids, n_groups,
    counts, sums, sum_sqs
);

// Use for ANOVA: compute group means and variances
for (Size g = 0; g < n_groups; ++g) {
    double mean = sums[g] / counts[g];
    double var = (sum_sqs[g] - sums[g] * mean) / (counts[g] - 1);
}
```

---

### partition_k_groups_to_buffer

::: source_code file="scl/kernel/stat/group_partition.hpp" symbol="partition_k_groups_to_buffer" collapsed
:::

**Algorithm Description**

Partition all values into a single buffer with group tags:

1. **Iterate over values**: For each sparse value
   - Get group assignment: g = group_ids[indices[k]]
   - If g is valid (0 <= g < n_groups):
     - Append value to out_values
     - Append group ID to out_groups
     - Increment out_total

2. **Output format**: 
   - out_values[i] and out_groups[i] correspond for i in [0, out_total)
   - Values are ready for sorting and rank computation

This is useful for Kruskal-Wallis test or other rank-based tests that need all values together.

**Edge Cases**

- **Invalid group IDs**: Ignored, not added to output
- **Empty groups**: No values in output for that group
- **All values invalid**: out_total = 0

**Data Guarantees (Preconditions)**

- `out_values` and `out_groups` have sufficient capacity (>= len)
- `group_ids` values are in [0, n_groups) or negative (ignored)

**Complexity Analysis**

- **Time**: O(len) single pass
- **Space**: O(1) auxiliary space

**Example**

```cpp
#include "scl/kernel/stat/group_partition.hpp"

Real out_values[len];
Size out_groups[len];
Size out_total = 0;

// Partition all values into buffer with group tags
scl::kernel::stat::partition::partition_k_groups_to_buffer(
    values, indices, len, group_ids, n_groups,
    out_values, out_groups, out_total
);

// Sort values while maintaining group assignments
// Ready for rank computation in Kruskal-Wallis test
```

---

### finalize_group_stats

::: source_code file="scl/kernel/stat/group_partition.hpp" symbol="finalize_group_stats" collapsed
:::

**Algorithm Description**

Compute mean and variance from accumulated moments, accounting for implicit zeros in sparse data:

1. **Compute mean**: mean = sum / n_total
   - Includes zeros in denominator (n_total includes all elements, not just non-zeros)
   - For sparse data: mean accounts for implicit zeros

2. **Compute variance**: var = (sum_sq - sum * mean) / (n_total - ddof)
   - Uses Bessel's correction: ddof = 1 (default)
   - Adjusted for implicit zeros: uses n_total, not count
   - Clamped to >= 0 for numerical stability

3. **Handle edge cases**: Zero counts, zero sums, etc.

**Edge Cases**

- **Zero count**: mean = 0, var = 0
- **Zero sum**: mean = 0, var computed from sum_sq
- **n_total < ddof**: Variance undefined, returns 0
- **Negative variance**: Clamped to 0 (numerical stability)

**Data Guarantees (Preconditions)**

- `count` <= `n_total`
- `sum` and `sum_sq` are non-negative (for valid data)
- `ddof` >= 0 and < `n_total`

**Complexity Analysis**

- **Time**: O(1)
- **Space**: O(1)

**Example**

```cpp
#include "scl/kernel/stat/group_partition.hpp"

// Accumulated moments from partitioning
Size count = n1;           // Number of non-zeros
double sum = sum1;         // Sum of values
double sum_sq = sum_sq1;   // Sum of squares
Size n_total = n_cols;     // Total group size (including zeros)

double mean, var;

// Finalize statistics
scl::kernel::stat::partition::finalize_group_stats(
    count, sum, sum_sq, n_total,
    mean, var,
    1  // ddof = 1 for sample variance
);

// mean and var now account for implicit zeros in sparse data
```

---

## Utility Functions

### partition_two_groups_simple

Simple partition without sum accumulation.

::: source_code file="scl/kernel/stat/group_partition.hpp" symbol="partition_two_groups_simple" collapsed
:::

**Complexity**

- Time: O(len)
- Space: O(1) auxiliary

---

## Notes

- All partitioning functions use 4-way unroll and prefetching for optimal cache performance
- Moment accumulation enables online variance computation without storing all values
- Sparse data handling: implicit zeros are accounted for in mean/variance computation
- These utilities are building blocks for statistical tests (t-test, ANOVA, etc.)
- All operations are optimized for indirect memory access patterns
- Thread-safe when used with thread-local buffers

## See Also

- [T-Test](../ttest)
- [ANOVA](../oneway_anova)
- [Kruskal-Wallis Test](../kruskal_wallis)
- [Statistical Tests](../stat)

