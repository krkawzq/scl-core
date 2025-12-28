# state.hpp

> scl/kernel/state.hpp · Cell state scoring kernels for stemness, differentiation, proliferation, and other cellular states

## Overview

This file provides comprehensive cell state scoring methods for single-cell RNA-seq analysis. It computes scores for various cellular states including stemness, differentiation potential, proliferation, stress, cell cycle phases, metabolic states, and apoptosis based on gene expression signatures.

This file provides:
- Stemness score computation
- Differentiation potential (CytoTRACE-style) scoring
- Proliferation and cell cycle scoring
- Stress and apoptosis scoring
- Metabolic state scoring (glycolysis/OXPHOS)
- Custom gene signature scoring
- Expression diversity and complexity measures

**Header**: `#include "scl/kernel/state.hpp"`

---

## Main APIs

### stemness_score

::: source_code file="scl/kernel/state.hpp" symbol="stemness_score" collapsed
:::

**Algorithm Description**

Compute stemness score for each cell based on stemness gene expression:

1. **Gene expression aggregation**: For each cell i:
   - Compute mean expression of stemness genes: score[i] = mean(expression[i, stemness_genes])
   - Uses sparse matrix access for efficient computation

2. **Z-score normalization**: Normalize scores across all cells:
   - mean = mean(scores)
   - std = std(scores)
   - scores[i] = (scores[i] - mean) / std

3. **Output**: Z-score normalized scores with mean 0 and std 1

**Edge Cases**

- **No stemness genes**: If gene list is empty, returns zero scores
- **Constant expression**: If all cells have same expression, returns zero scores after normalization
- **Missing genes**: Invalid gene indices are ignored

**Data Guarantees (Preconditions)**

- `scores.len == expression.rows()`
- All gene indices in stemness_genes must be valid (in [0, n_genes))
- Expression matrix must be valid CSR format

**Complexity Analysis**

- **Time**: O(n_cells * n_stemness_genes * log(nnz_per_cell))
  - O(n_stemness_genes * log(nnz_per_cell)) per cell for sparse access
  - n_cells cells processed in parallel
- **Space**: O(n_cells) auxiliary space for scores and normalization

**Example**

```cpp
#include "scl/kernel/state.hpp"

// Expression matrix: cells x genes
Sparse<Real, true> expression = /* ... */;
Index n_cells = expression.rows();

// Stemness gene indices (e.g., Nanog, Oct4, Sox2)
Array<Index> stemness_genes = {0, 1, 2, 3};  // Example indices

// Pre-allocate output
Array<Real> stemness_scores(n_cells);

// Compute stemness scores
scl::kernel::state::stemness_score(
    expression,
    stemness_genes,
    stemness_scores
);

// Scores are z-score normalized: mean=0, std=1
// Higher scores indicate higher stemness
```

---

### differentiation_potential

::: source_code file="scl/kernel/state.hpp" symbol="differentiation_potential" collapsed
:::

**Algorithm Description**

Compute differentiation potential score (CytoTRACE-style) for each cell:

1. **Gene count correlation**: 
   - Count expressed genes per cell
   - Compute correlation of each gene with gene count
   - Select top correlated genes (genes that track with transcriptional diversity)

2. **Weighted expression sum**: For each cell i:
   - Compute weighted sum of top gene expressions
   - Weights based on correlation strength

3. **Normalization**: Normalize scores to [0, 1] range:
   - scores[i] = (score[i] - min) / (max - min)

**Edge Cases**

- **Empty matrix**: Returns zero scores
- **All cells identical**: Returns uniform scores
- **No expressed genes**: Cells with no expressed genes get zero score

**Data Guarantees (Preconditions)**

- `potency_scores.len == expression.rows()`
- Expression matrix must be valid CSR format

**Complexity Analysis**

- **Time**: O(n_cells * n_genes * log(nnz_per_cell) + n_genes * log(n_genes))
  - O(n_cells * n_genes) for gene count and correlation computation
  - O(n_genes * log(n_genes)) for sorting correlated genes
- **Space**: O(n_cells + n_genes) auxiliary space

**Example**

```cpp
Array<Real> potency_scores(n_cells);

scl::kernel::state::differentiation_potential(
    expression,
    potency_scores
);

// Scores in [0, 1], higher scores indicate greater differentiation potential
```

---

### cell_cycle_score

::: source_code file="scl/kernel/state.hpp" symbol="cell_cycle_score" collapsed
:::

**Algorithm Description**

Compute cell cycle phase scores (G1/S/G2M) and assign phase labels:

1. **Phase score computation**: For each cell i:
   - Compute S-phase score: s_score[i] = mean(expression[i, s_genes])
   - Compute G2/M-phase score: g2m_score[i] = mean(expression[i, g2m_genes])

2. **Z-score normalization**: Normalize both scores independently

3. **Phase assignment**: For each cell i:
   - If s_score[i] > 0 and s_score[i] > g2m_score[i]: phase = S (1)
   - Else if g2m_score[i] > 0 and g2m_score[i] > s_score[i]: phase = G2M (2)
   - Else: phase = G1 (0)

**Edge Cases**

- **Empty gene lists**: Returns zero scores and G1 phase for all cells
- **Tied scores**: G1 phase assigned when scores are equal
- **Negative scores**: Only positive scores contribute to phase assignment

**Data Guarantees (Preconditions)**

- All score arrays have length == expression.rows()
- All gene indices must be valid
- Expression matrix must be valid CSR format

**Complexity Analysis**

- **Time**: O(n_cells * (n_s_genes + n_g2m_genes) * log(nnz_per_cell))
- **Space**: O(n_cells) auxiliary space

**Example**

```cpp
// S-phase and G2/M-phase gene indices
Array<Index> s_genes = {10, 11, 12};      // Example S-phase genes
Array<Index> g2m_genes = {20, 21, 22};    // Example G2/M-phase genes

Array<Real> s_scores(n_cells);
Array<Real> g2m_scores(n_cells);
Array<Index> phase_labels(n_cells);

scl::kernel::state::cell_cycle_score(
    expression,
    s_genes,
    g2m_genes,
    s_scores,
    g2m_scores,
    phase_labels
);

// phase_labels[i] = 0 (G1), 1 (S), or 2 (G2M)
```

---

### signature_score

::: source_code file="scl/kernel/state.hpp" symbol="signature_score" collapsed
:::

**Algorithm Description**

Compute weighted gene signature score for each cell:

1. **Weighted sum**: For each cell i:
   - Compute weighted sum: score[i] = sum_j(weight[j] * expression[i, gene_indices[j]])
   - Normalize by sum of absolute weights

2. **Z-score normalization**: Normalize scores across all cells

**Edge Cases**

- **Empty signature**: Returns zero scores
- **Zero weights**: Genes with zero weight contribute nothing
- **Negative weights**: Handled correctly (can represent negative correlations)

**Data Guarantees (Preconditions)**

- `scores.len == expression.rows()`
- `gene_indices.len == gene_weights.len`
- All gene indices must be valid

**Complexity Analysis**

- **Time**: O(n_cells * n_signature_genes * log(nnz_per_cell))
- **Space**: O(n_cells) auxiliary space

**Example**

```cpp
// Custom gene signature with weights
Array<Index> gene_indices = {0, 1, 2, 3};
Array<Real> gene_weights = {1.0, 0.5, -0.5, 1.0};  // Positive and negative weights

Array<Real> signature_scores(n_cells);

scl::kernel::state::signature_score(
    expression,
    gene_indices,
    gene_weights,
    signature_scores
);
```

---

### state_entropy

::: source_code file="scl/kernel/state.hpp" symbol="state_entropy" collapsed
:::

**Algorithm Description**

Compute expression entropy (plasticity) for each cell, normalized by maximum possible entropy:

1. **Shannon entropy**: For each cell i:
   - Compute total expression: total = sum(expression[i, :])
   - Compute probabilities: p[j] = expression[i, j] / total
   - Compute entropy: entropy[i] = -sum_j(p[j] * log(p[j]))

2. **Normalization**: Normalize by maximum entropy (log(n_genes)):
   - entropy_scores[i] = entropy[i] / log(n_genes)

**Edge Cases**

- **Zero total expression**: Returns zero entropy
- **Single expressed gene**: Returns zero entropy (minimum diversity)
- **Uniform expression**: Returns 1.0 (maximum diversity)

**Data Guarantees (Preconditions)**

- `entropy_scores.len == expression.rows()`
- Expression matrix must be valid CSR format

**Complexity Analysis**

- **Time**: O(nnz) - single pass through non-zero values
- **Space**: O(1) auxiliary space per cell

**Example**

```cpp
Array<Real> entropy_scores(n_cells);

scl::kernel::state::state_entropy(
    expression,
    entropy_scores
);

// Scores in [0, 1], where 1 indicates maximum expression diversity
```

---

## Utility Functions

### proliferation_score

Compute proliferation score based on proliferation gene expression.

::: source_code file="scl/kernel/state.hpp" symbol="proliferation_score" collapsed
:::

**Complexity**

- Time: O(n_cells * n_proliferation_genes * log(nnz_per_cell))
- Space: O(n_cells)

---

### stress_score

Compute stress score based on stress gene expression.

::: source_code file="scl/kernel/state.hpp" symbol="stress_score" collapsed
:::

**Complexity**

- Time: O(n_cells * n_stress_genes * log(nnz_per_cell))
- Space: O(n_cells)

---

### quiescence_score

Compute quiescence score as difference between quiescence and proliferation scores.

::: source_code file="scl/kernel/state.hpp" symbol="quiescence_score" collapsed
:::

**Complexity**

- Time: O(n_cells * (n_quiescence_genes + n_proliferation_genes) * log(nnz_per_cell))
- Space: O(n_cells)

---

### metabolic_score

Compute glycolysis and OXPHOS scores.

::: source_code file="scl/kernel/state.hpp" symbol="metabolic_score" collapsed
:::

**Complexity**

- Time: O(n_cells * (n_glycolysis_genes + n_oxphos_genes) * log(nnz_per_cell))
- Space: O(n_cells)

---

### apoptosis_score

Compute apoptosis score based on apoptosis gene expression.

::: source_code file="scl/kernel/state.hpp" symbol="apoptosis_score" collapsed
:::

**Complexity**

- Time: O(n_cells * n_apoptosis_genes * log(nnz_per_cell))
- Space: O(n_cells)

---

### multi_signature_score

Compute scores for multiple gene signatures simultaneously.

::: source_code file="scl/kernel/state.hpp" symbol="multi_signature_score" collapsed
:::

**Complexity**

- Time: O(n_signatures * n_cells * avg_signature_size * log(nnz_per_cell))
- Space: O(n_cells * n_signatures)

---

### transcriptional_diversity

Compute Simpson's diversity index for expression distribution.

::: source_code file="scl/kernel/state.hpp" symbol="transcriptional_diversity" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(1) per cell

---

### expression_complexity

Compute expression complexity as fraction of genes expressed above threshold.

::: source_code file="scl/kernel/state.hpp" symbol="expression_complexity" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(1) per cell

---

### combined_state_score

Compute combined state score from multiple gene sets with weights.

::: source_code file="scl/kernel/state.hpp" symbol="combined_state_score" collapsed
:::

**Complexity**

- Time: O(n_gene_sets * n_cells * avg_gene_set_size * log(nnz_per_cell))
- Space: O(n_gene_sets * n_cells)

---

## Notes

**Score Normalization**:
- Most scores are z-score normalized (mean=0, std=1)
- Entropy and complexity scores normalized to [0, 1]
- Differentiation potential normalized to [0, 1]

**Gene Signature Lists**:
- Gene indices must be valid and within [0, n_genes)
- Empty gene lists result in zero scores
- Invalid indices are ignored

**Performance**:
- All functions parallelized over cells
- Sparse matrix access optimized for efficiency
- Memory-efficient computation

**Typical Usage**:
- Cell state characterization
- Differentiation trajectory analysis
- Cell cycle analysis
- Metabolic state assessment
- Custom signature scoring

## See Also

- [Expression Analysis](/cpp/kernels/feature) - Additional expression analysis tools
- [Statistics](/cpp/kernels/statistics) - Statistical operations
