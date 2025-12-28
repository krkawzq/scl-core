# hotspot.hpp

> scl/kernel/hotspot.hpp · Spatial statistics and hotspot detection kernels

## Overview

This file provides comprehensive spatial autocorrelation analysis and hotspot detection for spatial transcriptomics and spatial data analysis. It implements Local and Global Moran's I, Getis-Ord Gi*, Geary's C, and LISA pattern classification with permutation-based inference.

This file provides:
- Local and global spatial autocorrelation statistics
- Hot spot and cold spot detection
- LISA (Local Indicators of Spatial Association) pattern classification
- Multiple testing correction for spatial tests
- Spatial weight matrix construction utilities

**Header**: `#include "scl/kernel/hotspot.hpp"`

---

## Main APIs

### local_morans_i

::: source_code file="scl/kernel/hotspot.hpp" symbol="local_morans_i" collapsed
:::

**Algorithm Description**

Compute Local Moran's I statistic for each observation, measuring local spatial autocorrelation:

1. **Standardization**: Standardize attribute values to z-scores:
   - mean = mean(values)
   - std = std(values)
   - z[i] = (values[i] - mean) / std

2. **Spatial lag computation**: For each observation i in parallel:
   - Compute spatial lag: lag[i] = sum_j(w_ij * z[j])
   - Where w_ij is the spatial weight between i and j
   - Uses sparse matrix-vector multiplication (SpMV)

3. **Local I computation**: For each observation i:
   - local_I[i] = z[i] * lag[i]
   - Positive I: similar values cluster (high-high or low-low)
   - Negative I: dissimilar values cluster (high-low or low-high)

4. **Permutation test** (if n_permutations > 0):
   - For each observation i:
     - Shuffle neighbor values n_permutations times
     - Recompute I for each permutation
     - p_value[i] = proportion of permuted I >= observed I[i]
   - Uses thread-local RNG via WorkspacePool for parallel safety

**Edge Cases**

- **Zero variance**: If std(values) = 0, all z-scores = 0, all local_I = 0
- **Isolated observations**: Observations with no neighbors (zero row sum) have lag = 0, local_I = 0
- **Empty weight matrix**: Returns all zeros if matrix has no non-zeros
- **Single observation**: Returns local_I = 0 (no spatial context)

**Data Guarantees (Preconditions)**

- `weights` must be square sparse matrix (n x n)
- `local_I` and `p_values` must be pre-allocated with n elements
- `n_permutations > 0` for permutation test (default: 999)
- Weight matrix should be row-normalized for interpretability (not required)
- Values should be continuous (not categorical)

**Complexity Analysis**

- **Time**: O(n * n_permutations * avg_neighbors)
  - O(n * avg_neighbors) for spatial lag computation
  - O(n * n_permutations * avg_neighbors) for permutation test
- **Space**: O(n_threads * n) for permutation buffers
  - Thread-local buffers for shuffled values and statistics

**Example**

```cpp
#include "scl/kernel/hotspot.hpp"

// Spatial weight matrix (n x n, CSR format)
Sparse<Real, true> weights = /* ... */;
Index n = weights.rows();

// Gene expression values per spot
Array<Real> expression(n);
// ... fill expression values ...

// Pre-allocate output
Array<Real> local_I(n);
Array<Real> p_values(n);

// Compute Local Moran's I with permutation test
scl::kernel::hotspot::local_morans_i(
    weights,
    expression.data(),
    n,
    local_I.data(),
    p_values.data(),
    999,    // n_permutations
    42      // seed
);

// Identify significant spatial clusters
for (Index i = 0; i < n; ++i) {
    if (p_values[i] < 0.05 && local_I[i] > 0) {
        // Significant positive autocorrelation (clustering)
        // High expression surrounded by high expression
    } else if (p_values[i] < 0.05 && local_I[i] < 0) {
        // Significant negative autocorrelation (dispersion)
        // High expression surrounded by low expression
    }
}
```

---

### getis_ord_g_star

::: source_code file="scl/kernel/hotspot.hpp" symbol="getis_ord_g_star" collapsed
:::

**Algorithm Description**

Compute Getis-Ord Gi* statistic for hotspot detection, measuring local concentration of high or low values:

1. **Local sum computation**: For each observation i in parallel:
   - Compute weighted local sum: local_sum[i] = sum_j(w_ij * values[j])
   - Includes observation i itself (self-inclusive)

2. **Expected value and variance**: Under null hypothesis of no spatial clustering:
   - Expected[i] = (sum_j(w_ij) / sum_all(w)) * sum_all(values)
   - Variance[i] computed from spatial weights structure
   - Accounts for spatial autocorrelation in null distribution

3. **Z-score computation**: For each observation i:
   - g_star[i] = (local_sum[i] - Expected[i]) / sqrt(Variance[i])
   - Positive z-score: high concentration (hot spot)
   - Negative z-score: low concentration (cold spot)

4. **P-value computation**: Two-tailed test under normal approximation:
   - p_value[i] = 2 * (1 - normal_cdf(|g_star[i]|))

**Edge Cases**

- **Zero weights**: Observations with no neighbors have undefined g_star (set to 0)
- **Constant values**: If all values are equal, all g_star = 0
- **Negative values**: Gi* assumes non-negative values for meaningful interpretation
- **Sparse neighborhoods**: Handles varying number of neighbors per observation

**Data Guarantees (Preconditions)**

- `weights` must be square sparse matrix (n x n)
- `g_star` and `p_values` must be pre-allocated with n elements
- `values` should be non-negative for meaningful hotspot interpretation
- Weight matrix should be symmetric or row-normalized

**Complexity Analysis**

- **Time**: O(nnz) where nnz is number of non-zeros in weight matrix
  - Single pass through sparse matrix structure
- **Space**: O(1) auxiliary space
  - Computes statistics on-the-fly without intermediate buffers

**Example**

```cpp
// Compute Gi* for hotspot detection
Array<Real> g_star(n);
Array<Real> p_values(n);

scl::kernel::hotspot::getis_ord_g_star(
    weights,
    expression.data(),
    n,
    g_star.data(),
    p_values.data()
);

// Identify significant hotspots and coldspots
Array<bool> is_hotspot(n, false);
Array<bool> is_coldspot(n, false);

scl::kernel::hotspot::identify_hotspots(
    g_star.data(),
    p_values.data(),
    n,
    is_hotspot.data(),
    is_coldspot.data(),
    0.05  // alpha
);
```

---

### classify_lisa_patterns

::: source_code file="scl/kernel/hotspot.hpp" symbol="classify_lisa_patterns" collapsed
:::

**Algorithm Description**

Classify observations into LISA (Local Indicators of Spatial Association) pattern categories based on standardized values and spatial lag:

1. **Quadrant classification**: For each observation i:
   - Compare z_values[i] and spatial_lag[i] to origin (0, 0)
   - Four quadrants:
     - Q1 (High-High): z > 0 and lag > 0
     - Q2 (Low-High): z < 0 and lag > 0
     - Q3 (Low-Low): z < 0 and lag < 0
     - Q4 (High-Low): z > 0 and lag < 0

2. **Significance filtering**: Only assign pattern if p_value[i] < alpha:
   - If significant: assign corresponding LISAPattern enum
   - If not significant: assign NotSignificant

3. **Pattern assignment**:
   - HighHigh: High value surrounded by high values (hot spot cluster)
   - LowLow: Low value surrounded by low values (cold spot cluster)
   - HighLow: High value surrounded by low values (spatial outlier)
   - LowHigh: Low value surrounded by high values (spatial outlier)

**Edge Cases**

- **Zero z-values**: Observations with z = 0 are classified as NotSignificant
- **Zero spatial lag**: Observations with lag = 0 are classified as NotSignificant
- **Tied values**: Uses strict inequality (>, <) for classification
- **Multiple testing**: Does not correct for multiple comparisons (use benjamini_hochberg_correction separately)

**Data Guarantees (Preconditions)**

- All input arrays must have length >= n
- `patterns` must be pre-allocated with n elements
- `z_values` should be standardized (mean=0, std=1)
- `spatial_lag` should be computed from same standardization
- `p_values` should be from local_morans_i or similar test

**Complexity Analysis**

- **Time**: O(n) - single pass through observations
- **Space**: O(1) - no intermediate storage

**Example**

```cpp
// After computing local_morans_i
Array<Real> z_values(n);  // Standardized values
Array<Real> spatial_lag(n);  // Spatial lag
Array<Real> p_values(n);  // From local_morans_i

// ... compute z_values and spatial_lag ...

Array<scl::kernel::hotspot::LISAPattern> patterns(n);

scl::kernel::hotspot::classify_lisa_patterns(
    z_values.data(),
    spatial_lag.data(),
    p_values.data(),
    n,
    patterns.data(),
    0.05  // alpha
);

// Count patterns
Index n_hotspots = 0, n_coldspots = 0;
for (Index i = 0; i < n; ++i) {
    if (patterns[i] == LISAPattern::HighHigh) n_hotspots++;
    if (patterns[i] == LISAPattern::LowLow) n_coldspots++;
}
```

---

### global_morans_i

::: source_code file="scl/kernel/hotspot.hpp" symbol="global_morans_i" collapsed
:::

**Algorithm Description**

Compute Global Moran's I statistic measuring overall spatial autocorrelation across all observations:

1. **Global I computation**:
   - I = (n / W) * (sum_i sum_j(w_ij * z[i] * z[j])) / (sum_i z[i]^2)
   - Where W = sum_i sum_j(w_ij) (sum of all weights)
   - z[i] are standardized values

2. **Expected value and variance**: Under null hypothesis of no spatial autocorrelation:
   - Expected[I] = -1 / (n - 1)
   - Variance[I] computed from weight matrix structure

3. **Z-score and inference**:
   - z_score = (I - Expected[I]) / sqrt(Variance[I])
   - p_value from permutation test (shuffle values, recompute I)

4. **Interpretation**:
   - I > 0: Positive spatial autocorrelation (clustering)
   - I < 0: Negative spatial autocorrelation (dispersion)
   - I ≈ 0: Random spatial pattern

**Edge Cases**

- **Small samples**: For n < 3, I is undefined (returns 0)
- **Zero weights**: If W = 0, I is undefined (returns 0)
- **Constant values**: If all values equal, I = 0

**Data Guarantees (Preconditions)**

- `weights` must be square sparse matrix (n x n)
- All output pointers must be valid
- `n_permutations > 0` for permutation test

**Complexity Analysis**

- **Time**: O(n * n_permutations * avg_neighbors)
  - O(nnz) for I computation
  - O(n * n_permutations * avg_neighbors) for permutation test
- **Space**: O(n_threads * n) for permutation buffers

**Example**

```cpp
Real global_I, expected_I, variance_I, z_score, p_value;

scl::kernel::hotspot::global_morans_i(
    weights,
    expression.data(),
    n,
    &global_I,
    &expected_I,
    &variance_I,
    &z_score,
    &p_value,
    999,  // n_permutations
    42    // seed
);

if (p_value < 0.05) {
    if (global_I > 0) {
        // Significant positive spatial autocorrelation
        // Values cluster spatially
    } else {
        // Significant negative spatial autocorrelation
        // Values disperse spatially
    }
}
```

---

### local_gearys_c

::: source_code file="scl/kernel/hotspot.hpp" symbol="local_gearys_c" collapsed
:::

**Algorithm Description**

Compute Local Geary's C statistic, an alternative measure of local spatial autocorrelation based on squared differences:

1. **Local C computation**: For each observation i:
   - local_C[i] = sum_j(w_ij * (values[i] - values[j])^2) / variance
   - Measures local dissimilarity (inverse of autocorrelation)

2. **Permutation test**: Similar to local_morans_i:
   - Shuffle neighbor values
   - Recompute C for each permutation
   - p_value = proportion of permuted C <= observed C

3. **Interpretation**:
   - Small C: Positive spatial association (similar values cluster)
   - Large C: Negative spatial association (dissimilar values cluster)

**Edge Cases**

- **Zero variance**: If variance = 0, all local_C undefined (set to 0)
- **Isolated observations**: No neighbors results in local_C = 0

**Data Guarantees (Preconditions)**

- `weights` must be square sparse matrix (n x n)
- `local_C` and `p_values` must be pre-allocated with n elements

**Complexity Analysis**

- **Time**: O(n * n_permutations * avg_neighbors)
- **Space**: O(n_threads * n) for permutation buffers

**Example**

```cpp
Array<Real> local_C(n);
Array<Real> p_values(n);

scl::kernel::hotspot::local_gearys_c(
    weights,
    expression.data(),
    n,
    local_C.data(),
    p_values.data(),
    999,  // n_permutations
    42    // seed
);
```

---

### global_gearys_c

::: source_code file="scl/kernel/hotspot.hpp" symbol="global_gearys_c" collapsed
:::

**Algorithm Description**

Compute Global Geary's C statistic, inverse measure of global spatial autocorrelation:

1. **Global C computation**:
   - C = ((n-1) / (2*W)) * (sum_i sum_j(w_ij * (values[i] - values[j])^2)) / (sum_i (values[i] - mean)^2)

2. **Expected value**: Under null: Expected[C] = 1

3. **Interpretation**:
   - C < 1: Positive spatial autocorrelation
   - C > 1: Negative spatial autocorrelation
   - C ≈ 1: Random spatial pattern

**Edge Cases**

- **Small samples**: For n < 2, C undefined (returns 1)
- **Zero variance**: If variance = 0, C undefined (returns 1)

**Data Guarantees (Preconditions)**

- `weights` must be square sparse matrix (n x n)
- All output pointers must be valid

**Complexity Analysis**

- **Time**: O(nnz)
- **Space**: O(1) auxiliary

**Example**

```cpp
Real global_C, expected_C, variance_C, z_score, p_value;

scl::kernel::hotspot::global_gearys_c(
    weights,
    expression.data(),
    n,
    &global_C,
    &expected_C,
    &variance_C,
    &z_score,
    &p_value
);
```

---

## Utility Functions

### identify_hotspots

Identify statistically significant hot spots and cold spots from Gi* z-scores.

::: source_code file="scl/kernel/hotspot.hpp" symbol="identify_hotspots" collapsed
:::

**Complexity**

- Time: O(n)
- Space: O(1)

---

### benjamini_hochberg_correction

Apply Benjamini-Hochberg FDR correction to p-values from multiple spatial tests.

::: source_code file="scl/kernel/hotspot.hpp" symbol="benjamini_hochberg_correction" collapsed
:::

**Complexity**

- Time: O(n log n) using VQSort
- Space: O(n) for sorted arrays

---

### distance_band_weights

Construct spatial weight matrix based on distance threshold.

::: source_code file="scl/kernel/hotspot.hpp" symbol="distance_band_weights" collapsed
:::

**Complexity**

- Time: O(n^2) brute-force distance computation
- Space: O(nnz) for output

---

### knn_weights

Construct K-nearest neighbors spatial weight matrix.

::: source_code file="scl/kernel/hotspot.hpp" symbol="knn_weights" collapsed
:::

**Complexity**

- Time: O(n^2 log k) using heap-based selection
- Space: O(n * k) for output

---

### bivariate_local_morans_i

Compute bivariate Local Moran's I between two variables.

::: source_code file="scl/kernel/hotspot.hpp" symbol="bivariate_local_morans_i" collapsed
:::

**Complexity**

- Time: O(n * n_permutations * avg_neighbors)
- Space: O(n_threads * n)

---

### detect_spatial_clusters

Detect and label spatial clusters from LISA patterns.

::: source_code file="scl/kernel/hotspot.hpp" symbol="detect_spatial_clusters" collapsed
:::

**Complexity**

- Time: O(n + nnz) using BFS for connected components
- Space: O(n) for BFS queue

---

### spatial_autocorrelation_summary

Compute comprehensive spatial autocorrelation summary statistics.

::: source_code file="scl/kernel/hotspot.hpp" symbol="spatial_autocorrelation_summary" collapsed
:::

**Complexity**

- Time: O(n * n_permutations * avg_neighbors)
- Space: O(n_threads * n)

---

## Notes

**Spatial Weight Matrices**:
- Weight matrices should typically be row-normalized for interpretability
- Common constructions: distance-band, KNN, inverse distance
- Symmetric weights are common but not required

**Permutation Tests**:
- Default 999 permutations provide good balance between accuracy and speed
- Use same seed for reproducibility
- Thread-safe via WorkspacePool

**Multiple Testing**:
- Spatial tests often involve many observations (multiple testing problem)
- Use benjamini_hochberg_correction to control FDR
- Consider spatial dependence when interpreting corrected p-values

**Performance**:
- All functions parallelized over observations
- Sparse matrix operations optimized for efficiency
- Memory-efficient with workspace pooling

## See Also

- [Spatial Analysis](/cpp/kernels/spatial) - Additional spatial analysis tools
- [Statistics](/cpp/kernels/statistics) - General statistical operations
