# enrichment.hpp

> scl/kernel/enrichment.hpp · Gene set enrichment and pathway analysis

## Overview

This file provides statistical methods for gene set enrichment analysis (GSEA), over-representation analysis (ORA), and pathway activity scoring. All methods are optimized for sparse expression matrices and support parallel processing.

**Header**: `#include "scl/kernel/enrichment.hpp"`

Key features:
- Hypergeometric and Fisher's exact tests
- GSEA enrichment scoring
- Over-representation analysis (ORA)
- Pathway activity computation
- FDR correction (Benjamini-Hochberg)

---

## Main APIs

### hypergeometric_test

::: source_code file="scl/kernel/enrichment.hpp" symbol="hypergeometric_test" collapsed
:::

**Algorithm Description**

Compute hypergeometric test p-value for over-representation:

1. Given contingency table:
   - k successes in sample of size n
   - K successes in population of size N
2. Compute p-value: probability of observing k or more successes
3. Uses hypergeometric distribution: P(X >= k) = sum_{i=k}^{min(n,K)} C(K,i) * C(N-K, n-i) / C(N,n)
4. Optimized computation using log-space to avoid overflow

**Edge Cases**

- **k = 0**: Returns p-value = 1.0 (no enrichment)
- **k = n**: Returns p-value based on population success rate
- **k > K**: Invalid input (should not occur)
- **n = 0**: Returns 1.0

**Data Guarantees (Preconditions)**

- `k <= n <= N`
- `K <= N`
- `k <= K`
- All values are non-negative integers

**Complexity Analysis**

- **Time**: O(min(k, n-k)) - proportional to smaller tail
- **Space**: O(1) auxiliary

**Example**

```cpp
#include "scl/kernel/enrichment.hpp"

// Test: 10 DE genes in sample of 100, 50 pathway genes in 20000 total
Index k = 10;  // DE genes in pathway
Index n = 100; // Total DE genes
Index K = 50;  // Pathway size
Index N = 20000; // Total genes

Real p_value = scl::kernel::enrichment::hypergeometric_test(k, n, K, N);

// p_value is probability of observing 10+ pathway genes by chance
```

---

### fisher_exact_test

::: source_code file="scl/kernel/enrichment.hpp" symbol="fisher_exact_test" collapsed
:::

**Algorithm Description**

Compute Fisher's exact test p-value for 2x2 contingency table:

1. Given 2x2 table:
   ```
   [a  b]
   [c  d]
   ```
2. Compute two-tailed p-value using hypergeometric distribution
3. Sum probabilities of all tables with same marginals and equal or more extreme odds ratio
4. Uses log-space computation for numerical stability

**Edge Cases**

- **Zero counts**: Handled correctly (returns appropriate p-value)
- **All zeros**: Returns 1.0
- **Perfect association (b=0 or c=0)**: Returns very small p-value

**Data Guarantees (Preconditions)**

- All counts >= 0
- At least one count in each row and column

**Complexity Analysis**

- **Time**: O(min(a, b, c, d)) - proportional to smallest count
- **Space**: O(1) auxiliary

**Example**

```cpp
// Contingency table:
//           In Pathway  Not in Pathway
// DE genes      15           85
// Not DE         35         19865

Index a = 15;  // DE and in pathway
Index b = 85;  // DE and not in pathway
Index c = 35;  // Not DE and in pathway
Index d = 19865; // Not DE and not in pathway

Real p_value = scl::kernel::enrichment::fisher_exact_test(a, b, c, d);

// p_value is two-tailed Fisher's exact test p-value
```

---

### gsea

::: source_code file="scl/kernel/enrichment.hpp" symbol="gsea" collapsed
:::

**Algorithm Description**

Compute Gene Set Enrichment Analysis enrichment score and p-value:

1. Rank genes by statistic (e.g., fold change, t-statistic)
2. Compute running sum:
   - When gene in set: add |statistic| / sum(|statistics in set|)
   - When gene not in set: subtract 1 / (n_genes - n_in_set)
3. Enrichment score (ES) = maximum absolute deviation from zero
4. Normalize ES by expected value under null (NES)
5. Compute p-value via permutation test:
   - Shuffle gene set membership n_permutations times
   - Count fraction with |NES| >= observed |NES|

**Edge Cases**

- **Empty gene set**: ES = 0, p-value = 1.0
- **All genes in set**: ES = 1.0, p-value = 0.0
- **No enrichment**: ES near 0, p-value near 1.0
- **n_permutations = 0**: p-value undefined

**Data Guarantees (Preconditions)**

- `ranked_genes` contains valid gene indices
- `in_gene_set.len >= n_genes`
- Genes in `ranked_genes` are sorted by statistic (descending)
- `n_permutations > 0` for valid p-value

**Complexity Analysis**

- **Time**: O(n_genes * n_permutations)
- **Space**: O(n_genes) auxiliary

**Example**

```cpp
// Rank genes by differential expression statistic
scl::Array<Index> ranked_genes = /* sorted by statistic */;
scl::Array<bool> in_gene_set(n_genes);
// ... set in_gene_set[i] = true if gene i is in pathway ...

Real es, nes, p_value;
scl::kernel::enrichment::gsea(
    ranked_genes, in_gene_set, n_genes,
    es, nes, p_value,
    1000,  // n_permutations
    42     // seed
);

// es = enrichment score
// nes = normalized enrichment score
// p_value = permutation-based p-value
```

---

### ora

::: source_code file="scl/kernel/enrichment.hpp" symbol="ora" collapsed
:::

**Algorithm Description**

Perform Over-Representation Analysis for multiple pathways:

1. For each pathway p in parallel:
   - Count overlap: genes in both DE set and pathway
   - Compute hypergeometric p-value
   - Compute odds ratio: (overlap / (n_de - overlap)) / ((pathway_size - overlap) / (n_total - n_de - pathway_size + overlap))
   - Compute fold enrichment: (overlap / n_de) / (pathway_size / n_total)
2. Returns p-values, odds ratios, and fold enrichments for all pathways

**Edge Cases**

- **No overlap**: p-value = 1.0, odds_ratio = 0, fold_enrichment = 0
- **Perfect overlap**: Very small p-value, large odds_ratio
- **Empty pathway**: Undefined (should not occur)

**Data Guarantees (Preconditions)**

- All output arrays have length >= n_pathways
- All gene indices are valid
- `pathway_genes` is array of arrays (one per pathway)
- `pathway_sizes` contains size of each pathway

**Complexity Analysis**

- **Time**: O(n_pathways * avg_pathway_size)
- **Space**: O(n_total_genes) auxiliary

**Example**

```cpp
scl::Array<Index> de_genes = /* DE gene indices */;
const Index* pathway_genes[] = {pathway1, pathway2, ...};
const Index pathway_sizes[] = {50, 30, ...};
Index n_pathways = 100;

scl::Array<Real> p_values(n_pathways);
scl::Array<Real> odds_ratios(n_pathways);
scl::Array<Real> fold_enrichments(n_pathways);

scl::kernel::enrichment::ora(
    de_genes,
    pathway_genes, pathway_sizes, n_pathways,
    n_total_genes,
    p_values, odds_ratios, fold_enrichments
);

// p_values[p] = hypergeometric p-value for pathway p
// odds_ratios[p] = odds ratio for pathway p
// fold_enrichments[p] = fold enrichment for pathway p
```

---

### pathway_activity

::: source_code file="scl/kernel/enrichment.hpp" symbol="pathway_activity" collapsed
:::

**Algorithm Description**

Compute pathway activity score for each cell:

1. For each cell i in parallel:
   - Extract expression of pathway genes
   - Compute mean expression: `activity[i] = mean(expression[pathway_genes])`
2. Activity represents average expression of pathway genes per cell
3. Uses sparse matrix operations for efficiency

**Edge Cases**

- **Empty pathway**: All activities are 0
- **No expression**: All activities are 0
- **Missing genes**: Ignored (not in expression matrix)

**Data Guarantees (Preconditions)**

- `activity_scores.len >= n_cells`
- All gene indices are valid
- X must be CSR format (cells x genes)

**Complexity Analysis**

- **Time**: O(nnz * n_pathway_genes / n_genes) - proportional to pathway size
- **Space**: O(n_genes) auxiliary for gene lookup

**Example**

```cpp
scl::Array<Index> pathway_genes = /* pathway gene indices */;
scl::Array<Real> activity_scores(n_cells);

scl::kernel::enrichment::pathway_activity(
    X, pathway_genes, n_cells, n_genes, activity_scores
);

// activity_scores[i] = mean expression of pathway genes in cell i
```

---

### benjamini_hochberg

::: source_code file="scl/kernel/enrichment.hpp" symbol="benjamini_hochberg" collapsed
:::

**Algorithm Description**

Apply Benjamini-Hochberg FDR correction to enrichment p-values:

1. Sort p-values in ascending order (with original indices)
2. For each p-value at rank i:
   - Compute adjusted p-value: `q = p * n / rank`
   - Ensure monotonicity: `q[i] = min(q[i], q[i+1], ..., q[n-1])`
3. Return q-values (FDR-adjusted p-values)

**Edge Cases**

- **All p-values = 0**: All q-values = 0
- **All p-values = 1**: All q-values = 1
- **Empty array**: Returns empty array

**Data Guarantees (Preconditions)**

- `q_values.len >= p_values.len`
- All p-values in [0, 1]

**Complexity Analysis**

- **Time**: O(n log n) for sorting
- **Space**: O(n) auxiliary

**Example**

```cpp
scl::Array<Real> p_values = /* enrichment p-values */;
scl::Array<Real> q_values(p_values.len);

scl::kernel::enrichment::benjamini_hochberg(p_values, q_values);

// q_values[i] = FDR-adjusted p-value (q-value)
// q < 0.05 means FDR < 5%
```

---

## Utility Functions

### odds_ratio

Compute odds ratio from 2x2 contingency table.

::: source_code file="scl/kernel/enrichment.hpp" symbol="odds_ratio" collapsed
:::

**Complexity**

- Time: O(1)
- Space: O(1)

---

### gsea_running_sum

Compute GSEA running sum for visualization.

::: source_code file="scl/kernel/enrichment.hpp" symbol="gsea_running_sum" collapsed
:::

**Complexity**

- Time: O(n_genes)
- Space: O(1) auxiliary

---

### leading_edge_genes

Identify leading edge genes (genes contributing to enrichment peak).

::: source_code file="scl/kernel/enrichment.hpp" symbol="leading_edge_genes" collapsed
:::

**Complexity**

- Time: O(n_genes)
- Space: O(1) auxiliary

---

## Notes

- Hypergeometric test is most common for ORA
- GSEA requires ranked genes - ensure proper sorting
- FDR correction is essential for multiple testing
- Pathway activity can be used for cell type scoring

## See Also

- [Multiple Testing Module](./multiple_testing) - Additional correction methods
- [Statistics Module](../math/statistics) - Statistical tests
