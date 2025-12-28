# entropy.hpp

> scl/kernel/entropy.hpp · Information theory measures for sparse data analysis

## Overview

This file provides information-theoretic measures for analyzing sparse single-cell data, including entropy, mutual information, and feature selection methods. All operations are optimized for sparse matrices and support parallel processing.

**Header**: `#include "scl/kernel/entropy.hpp"`

Key features:
- Shannon entropy computation
- Kullback-Leibler and Jensen-Shannon divergence
- Mutual information and normalized variants
- Feature selection via MI and mRMR
- Discretization methods for continuous data

---

## Main APIs

### count_entropy

::: source_code file="scl/kernel/entropy.hpp" symbol="count_entropy" collapsed
:::

**Algorithm Description**

Compute Shannon entropy from count array:

1. Compute total count: `total = sum(counts)`
2. For each non-zero count:
   - Compute probability: `p_i = counts[i] / total`
   - Accumulate: `entropy -= p_i * log(p_i)`
3. Returns entropy H = -sum(p_i * log(p_i))
4. Uses log base 2 if `use_log2 = true`, natural log otherwise

**Edge Cases**

- **Total count = 0**: Returns 0.0
- **Single non-zero count**: Returns 0.0 (no uncertainty)
- **Uniform distribution**: Returns maximum entropy = log(n)
- **All zeros**: Returns 0.0

**Data Guarantees (Preconditions)**

- All counts >= 0
- `n > 0`

**Complexity Analysis**

- **Time**: O(n)
- **Space**: O(1) auxiliary

**Example**

```cpp
#include "scl/kernel/entropy.hpp"

Real counts[] = {10, 20, 30, 40};
Size n = 4;

Real entropy = scl::kernel::entropy::count_entropy(counts, n, false);

// entropy = -sum((count/total) * log(count/total))
```

---

### row_entropy

::: source_code file="scl/kernel/entropy.hpp" symbol="row_entropy" collapsed
:::

**Algorithm Description**

Compute Shannon entropy for each row of a sparse matrix:

1. For each row i in parallel:
   - Extract non-zero values in row i
   - Compute row sum: `row_sum = sum(row_i)`
   - For each non-zero value:
     - Probability: `p_j = value / row_sum`
     - Accumulate: `entropy[i] -= p_j * log(p_j)`
2. If `normalize = true`: divide by maximum entropy (log(n_cols))
3. Returns entropy values in [0, 1] if normalized

**Edge Cases**

- **Empty row**: Entropy = 0.0
- **Single non-zero**: Entropy = 0.0
- **Uniform row**: Maximum entropy
- **All zeros**: Entropy = 0.0

**Data Guarantees (Preconditions)**

- `entropies.len >= X.rows()`
- X must be valid CSR or CSC format

**Complexity Analysis**

- **Time**: O(nnz) - proportional to non-zeros
- **Space**: O(1) auxiliary per row

**Example**

```cpp
scl::Sparse<Real, true> X = /* expression matrix */;
scl::Array<Real> entropies(X.rows());

scl::kernel::entropy::row_entropy(X, entropies, false, false);

// entropies[i] = entropy of row i (gene expression distribution)
```

---

### kl_divergence

::: source_code file="scl/kernel/entropy.hpp" symbol="kl_divergence" collapsed
:::

**Algorithm Description**

Compute Kullback-Leibler divergence between two probability distributions:

1. For each element i:
   - If `p[i] > 0` and `q[i] > 0`: accumulate `p[i] * log(p[i] / q[i])`
   - If `p[i] > 0` and `q[i] = 0`: return large value (infinity)
   - If `p[i] = 0`: skip (0 * log(0) = 0)
2. Returns KL(p || q) = sum(p_i * log(p_i / q_i))
3. Asymmetric: KL(p||q) != KL(q||p)

**Edge Cases**

- **q[i] = 0 and p[i] > 0**: Returns large value (divergence undefined)
- **p = q**: Returns 0.0
- **p all zeros**: Returns 0.0
- **q all zeros and p not all zeros**: Returns large value

**Data Guarantees (Preconditions)**

- `p.len == q.len`
- Both arrays represent probability distributions (sum to 1.0)
- All values >= 0

**Complexity Analysis**

- **Time**: O(n)
- **Space**: O(1) auxiliary

**Example**

```cpp
scl::Array<Real> p = {0.5, 0.3, 0.2};  // Distribution 1
scl::Array<Real> q = {0.4, 0.4, 0.2};  // Distribution 2

Real kl = scl::kernel::entropy::kl_divergence(p, q, false);

// kl = KL(p || q) = sum(p_i * log(p_i / q_i))
```

---

### js_divergence

::: source_code file="scl/kernel/entropy.hpp" symbol="js_divergence" collapsed
:::

**Algorithm Description**

Compute Jensen-Shannon divergence between two probability distributions:

1. Compute mixture: `m = (p + q) / 2`
2. Compute JS = 0.5 * KL(p || m) + 0.5 * KL(q || m)
3. Always finite and symmetric: JS(p||q) = JS(q||p)
4. Bounded: JS in [0, 1] if using log base 2

**Edge Cases**

- **p = q**: Returns 0.0
- **p and q disjoint**: Returns maximum JS
- **Always finite**: Unlike KL, never returns infinity

**Data Guarantees (Preconditions)**

- `p.len == q.len`
- Both arrays represent probability distributions
- All values >= 0

**Complexity Analysis**

- **Time**: O(n)
- **Space**: O(1) auxiliary

**Example**

```cpp
scl::Array<Real> p = {0.5, 0.3, 0.2};
scl::Array<Real> q = {0.4, 0.4, 0.2};

Real js = scl::kernel::entropy::js_divergence(p, q, false);

// js = 0.5 * KL(p || m) + 0.5 * KL(q || m) where m = (p+q)/2
// Always finite and symmetric
```

---

### mutual_information

::: source_code file="scl/kernel/entropy.hpp" symbol="mutual_information" collapsed
:::

**Algorithm Description**

Compute mutual information I(X; Y) from binned data:

1. Compute 2D histogram: `counts[i][j]` = number of samples in bin (i, j)
2. Compute joint entropy: H(X, Y) = -sum(p_ij * log(p_ij))
3. Compute marginal entropies: H(X) and H(Y)
4. Return MI = H(X) + H(Y) - H(X, Y)
5. Always >= 0, equals 0 if X and Y are independent

**Edge Cases**

- **X and Y independent**: MI = 0.0
- **X = Y**: MI = H(X) (maximum)
- **No samples**: Returns 0.0
- **All samples in one bin**: MI = 0.0

**Data Guarantees (Preconditions)**

- All bin indices are valid: `x_binned[i] in [0, n_bins_x)`, `y_binned[i] in [0, n_bins_y)`
- `n > 0`

**Complexity Analysis**

- **Time**: O(n + n_bins_x * n_bins_y)
- **Space**: O(n_bins_x * n_bins_y) auxiliary

**Example**

```cpp
// Discretize continuous values first
scl::Array<Index> x_binned = /* binned x values */;
scl::Array<Index> y_binned = /* binned y values */;

Real mi = scl::kernel::entropy::mutual_information(
    x_binned.data(), y_binned.data(), n,
    n_bins_x, n_bins_y, false
);

// mi = I(X; Y) = H(X) + H(Y) - H(X, Y)
// Higher MI indicates stronger dependence
```

---

### normalized_mi

::: source_code file="scl/kernel/entropy.hpp" symbol="normalized_mi" collapsed
:::

**Algorithm Description**

Compute normalized mutual information between two labelings:

1. Compute mutual information I(X; Y) from labelings
2. Compute marginal entropies H(X) and H(Y)
3. Return NMI = 2 * I(X; Y) / (H(X) + H(Y))
4. Values in [0, 1], where 1 indicates perfect agreement
5. Symmetric: NMI(X, Y) = NMI(Y, X)

**Edge Cases**

- **Perfect agreement**: NMI = 1.0
- **Independent labelings**: NMI = 0.0
- **One labeling has single cluster**: NMI = 0.0 (H = 0)

**Data Guarantees (Preconditions)**

- `labels1.len == labels2.len`
- All label indices are valid: `labels1[i] in [0, n_clusters1)`, `labels2[i] in [0, n_clusters2)`

**Complexity Analysis**

- **Time**: O(n + n_clusters1 * n_clusters2)
- **Space**: O(n_clusters1 * n_clusters2) auxiliary

**Example**

```cpp
scl::Array<Index> labels1 = /* first clustering */;
scl::Array<Index> labels2 = /* second clustering */;

Real nmi = scl::kernel::entropy::normalized_mi(
    labels1, labels2, n_clusters1, n_clusters2
);

// nmi in [0, 1], higher = better agreement
```

---

### select_features_mi

::: source_code file="scl/kernel/entropy.hpp" symbol="select_features_mi" collapsed
:::

**Algorithm Description**

Select top features using mutual information with target:

1. For each feature f:
   - Discretize feature values into n_bins
   - Compute MI between discretized feature and target labels
   - Store MI score
2. Sort features by MI score (descending)
3. Select top n_to_select features
4. Returns selected features and all MI scores

**Edge Cases**

- **n_to_select = 0**: Returns empty selection
- **n_to_select >= n_features**: Returns all features
- **Constant features**: MI = 0.0
- **Perfect correlation**: MI = H(target)

**Data Guarantees (Preconditions)**

- `selected_features` has capacity >= n_to_select
- `mi_scores` has capacity >= n_features
- `target` contains valid label indices
- X must be valid CSR or CSC format

**Complexity Analysis**

- **Time**: O(n_features * n_samples * log(nnz_per_sample))
- **Space**: O(n_samples) auxiliary

**Example**

```cpp
scl::Sparse<Real, true> X = /* feature matrix */;
scl::Array<Index> target = /* target labels */;
Index n_to_select = 100;

scl::Array<Index> selected_features(n_to_select);
scl::Array<Real> mi_scores(n_features);

scl::kernel::entropy::select_features_mi(
    X, target, n_features, n_to_select,
    selected_features, mi_scores, 10  // n_bins
);

// selected_features contains top 100 features by MI
// mi_scores contains MI score for all features
```

---

### mrmr_selection

::: source_code file="scl/kernel/entropy.hpp" symbol="mrmr_selection" collapsed
:::

**Algorithm Description**

Select features using minimum Redundancy Maximum Relevance (mRMR):

1. Initialize: select feature with highest MI with target
2. For each remaining selection:
   - For each unselected feature f:
     - Compute relevance: MI(f, target)
     - Compute redundancy: mean(MI(f, selected_features))
     - Score: relevance - redundancy
   - Select feature with highest score
3. Greedy selection balances relevance and redundancy
4. Returns selected features in selection order

**Edge Cases**

- **n_to_select = 0**: Returns empty selection
- **n_to_select = 1**: Returns single best feature
- **All features redundant**: May select fewer than requested

**Data Guarantees (Preconditions)**

- `selected_features` has capacity >= n_to_select
- `target` contains valid label indices
- X must be valid CSR or CSC format

**Complexity Analysis**

- **Time**: O(n_to_select * n_features * n_samples)
- **Space**: O(n_features * n_samples) auxiliary

**Example**

```cpp
scl::Array<Index> selected_features(n_to_select);

scl::kernel::entropy::mrmr_selection(
    X, target, n_features, n_to_select,
    selected_features, 10  // n_bins
);

// selected_features contains mRMR-selected features
// Features maximize relevance and minimize redundancy
```

---

## Utility Functions

### discretize_equal_width

Discretize continuous values into equal-width bins.

::: source_code file="scl/kernel/entropy.hpp" symbol="discretize_equal_width" collapsed
:::

**Complexity**

- Time: O(n)
- Space: O(1) auxiliary

---

### discretize_equal_frequency

Discretize continuous values into equal-frequency bins.

::: source_code file="scl/kernel/entropy.hpp" symbol="discretize_equal_frequency" collapsed
:::

**Complexity**

- Time: O(n log n) for sorting
- Space: O(n) auxiliary

---

### joint_entropy

Compute joint entropy H(X, Y) from binned data.

::: source_code file="scl/kernel/entropy.hpp" symbol="joint_entropy" collapsed
:::

**Complexity**

- Time: O(n + n_bins_x * n_bins_y)
- Space: O(n_bins_x * n_bins_y) auxiliary

---

### conditional_entropy

Compute conditional entropy H(Y | X) from binned data.

::: source_code file="scl/kernel/entropy.hpp" symbol="conditional_entropy" collapsed
:::

**Complexity**

- Time: O(n + n_bins_x * n_bins_y)
- Space: O(n_bins_x * n_bins_y) auxiliary

---

### adjusted_mi

Compute adjusted mutual information (corrected for chance).

::: source_code file="scl/kernel/entropy.hpp" symbol="adjusted_mi" collapsed
:::

**Complexity**

- Time: O(n + n_clusters1 * n_clusters2)
- Space: O(n_clusters1 * n_clusters2) auxiliary

---

## Notes

- Entropy requires probability distributions - ensure normalization
- Discretization is necessary for continuous data before computing MI
- mRMR is preferred over simple MI for feature selection (reduces redundancy)
- Normalized MI is useful for comparing clusterings of different sizes

## See Also

- [Feature Selection Module](./feature) - Additional feature selection methods
- [Statistics Module](../math/statistics) - Statistical measures
