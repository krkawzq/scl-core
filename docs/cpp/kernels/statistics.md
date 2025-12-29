---
title: Statistical Tests
description: Statistical hypothesis testing functions
---

# Statistical Tests

The `stat` kernel provides efficient statistical hypothesis testing functions for comparing groups, optimized with SIMD and parallelization.

## Overview

Statistical tests are essential for:
- Differential expression analysis
- Group comparisons
- Marker gene identification
- Quality control

All tests are optimized for sparse matrices and large datasets.

## Available Tests

### T-Test

**Location**: `scl/kernel/stat/ttest.hpp`

Independent samples t-test for comparing two groups.

```cpp
namespace scl::kernel::stat::ttest {
    Real t_test(
        Array<const Real> group1,
        Array<const Real> group2
    );
}
```

**Example**:
```cpp
#include "scl/kernel/stat/ttest.hpp"

Array<Real> group1 = {data1, n1};
Array<Real> group2 = {data2, n2};
Real t_stat = kernel::stat::ttest::t_test(group1, group2);
```

### Mann-Whitney U Test

**Location**: `scl/kernel/mwu.hpp`

Non-parametric test for comparing two groups.

```cpp
namespace scl::kernel::mwu {
    Real mann_whitney_u(
        Array<const Real> group1,
        Array<const Real> group2
    );
}
```

### Kruskal-Wallis Test

**Location**: `scl/kernel/stat/kruskal_wallis.hpp`

Non-parametric test for comparing multiple groups.

```cpp
namespace scl::kernel::stat::kruskal_wallis {
    Real kruskal_wallis(
        const std::vector<Array<const Real>>& groups
    );
}
```

### One-Way ANOVA

**Location**: `scl/kernel/stat/oneway_anova.hpp`

Parametric test for comparing multiple groups.

```cpp
namespace scl::kernel::stat::oneway_anova {
    Real oneway_anova(
        const std::vector<Array<const Real>>& groups
    );
}
```

### Kolmogorov-Smirnov Test

**Location**: `scl/kernel/stat/ks.hpp`

Test for comparing distributions.

```cpp
namespace scl::kernel::stat::ks {
    Real kolmogorov_smirnov(
        Array<const Real> group1,
        Array<const Real> group2
    );
}
```

### AUROC

**Location**: `scl/kernel/stat/auroc.hpp`

Area Under ROC Curve for binary classification.

```cpp
namespace scl::kernel::stat::auroc {
    Real auroc(
        Array<const Real> scores,
        Array<const bool> labels
    );
}
```

### Effect Size

**Location**: `scl/kernel/stat/effect_size.hpp`

Compute effect sizes (Cohen's d, etc.).

```cpp
namespace scl::kernel::stat::effect_size {
    Real cohens_d(
        Array<const Real> group1,
        Array<const Real> group2
    );
}
```

### Permutation Tests

**Location**: `scl/kernel/stat/permutation_stat.hpp`

Permutation-based statistical tests.

```cpp
namespace scl::kernel::stat::permutation {
    Real permutation_test(
        Array<const Real> group1,
        Array<const Real> group2,
        Index n_permutations = 1000
    );
}
```

## Common Patterns

### Differential Expression Analysis

```cpp
void differential_expression(
    const CSR& matrix,
    Array<const bool> group_labels,
    Array<Real>& p_values,
    Array<Real>& effect_sizes
) {
    Index n_genes = matrix.cols();
    
    for (Index g = 0; g < n_genes; ++g) {
        // Extract expression for gene g
        auto expr_group1 = extract_group(matrix, g, group_labels, true);
        auto expr_group2 = extract_group(matrix, g, group_labels, false);
        
        // Compute test statistic
        Real t_stat = kernel::stat::ttest::t_test(expr_group1, expr_group2);
        
        // Compute effect size
        Real cohens_d = kernel::stat::effect_size::cohens_d(expr_group1, expr_group2);
        
        p_values[g] = compute_p_value(t_stat, expr_group1.size() + expr_group2.size() - 2);
        effect_sizes[g] = cohens_d;
    }
}
```

### Multiple Testing Correction

```cpp
void correct_multiple_testing(
    Array<Real> p_values,
    Array<Real>& adjusted_p_values
) {
    Index n = p_values.size();
    
    // Sort by p-value
    std::vector<std::pair<Real, Index>> sorted;
    for (Index i = 0; i < n; ++i) {
        sorted.push_back({p_values[i], i});
    }
    std::sort(sorted.begin(), sorted.end());
    
    // Benjamini-Hochberg correction
    for (Index i = 0; i < n; ++i) {
        Real rank = static_cast<Real>(i + 1);
        Real adjusted = sorted[i].first * static_cast<Real>(n) / rank;
        adjusted_p_values[sorted[i].second] = adjusted;
    }
}
```

### Batch Statistical Testing

```cpp
void batch_statistical_tests(
    const CSR& matrix,
    Array<const bool> labels,
    Array<Real>& test_stats
) {
    Index n_genes = matrix.cols();
    
    // Parallel processing
    threading::parallel_for(0, n_genes, [&](size_t g) {
        auto group1 = extract_group(matrix, g, labels, true);
        auto group2 = extract_group(matrix, g, labels, false);
        
        // Use appropriate test
        if (group1.size() > 30 && group2.size() > 30) {
            // Parametric test for large samples
            test_stats[g] = kernel::stat::ttest::t_test(group1, group2);
        } else {
            // Non-parametric for small samples
            test_stats[g] = kernel::mwu::mann_whitney_u(group1, group2);
        }
    });
}
```

## Performance Considerations

### SIMD Optimization

Statistical computations use SIMD where applicable:

```cpp
// SIMD-accelerated mean computation
namespace s = scl::simd;
auto v_sum = s::Zero(d);
for (Size i = 0; i < n; i += lanes) {
    auto v = s::Load(d, data + i);
    v_sum = s::Add(v_sum, v);
}
Real mean = s::GetLane(s::SumOfLanes(d, v_sum)) / static_cast<Real>(n);
```

### Parallelization

Tests are parallelized across genes:

```cpp
// Parallel gene-wise testing
threading::parallel_for(0, n_genes, [&](size_t g) {
    compute_test_for_gene(g);
});
```

## Related Documentation

- [Kernels Overview](./overview.md) - General kernel usage
- [Markers](./markers.md) - Marker gene detection
- [Multiple Testing](./multiple_testing.md) - Multiple testing correction

