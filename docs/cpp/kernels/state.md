# State

Cell state scoring kernels for stemness, differentiation, proliferation, and other cellular states.

## Overview

State provides:

- **Stemness Score** - Stem cell marker-based scoring
- **Differentiation Potential** - CytoTRACE-style potency scoring
- **Proliferation Score** - Cell cycle and growth scoring
- **Stress Score** - Stress response scoring
- **Cell Cycle** - G1/S/G2M phase assignment
- **Metabolic Scores** - Glycolysis and OXPHOS scoring
- **Apoptosis Score** - Cell death scoring
- **Signature Scoring** - Custom gene signature scoring
- **Diversity Measures** - Expression entropy and complexity

## Basic State Scores

### stemness_score

Compute stemness score for each cell based on stemness gene expression:

```cpp
#include "scl/kernel/state.hpp"

Sparse<Real, true> expression = /* ... */;  // cells x genes
Array<const Index> stemness_genes = /* ... */;  // Stemness gene indices
Array<Real> scores(n_cells);

scl::kernel::state::stemness_score(
    expression,
    stemness_genes,
    scores
);
```

**Parameters:**
- `expression`: Expression matrix (cells x genes, CSR)
- `stemness_genes`: Indices of stemness genes [n_stemness_genes]
- `scores`: Output Z-score normalized stemness scores [n_cells]

**Postconditions:**
- `scores[i]` contains z-score normalized stemness score for cell i
- Scores have mean 0 and standard deviation 1
- Matrix is unchanged

**Algorithm:**
1. For each cell, compute mean expression of stemness genes
2. Z-score normalize scores across all cells

**Complexity:**
- Time: O(n_cells * n_stemness_genes * log(nnz_per_cell))
- Space: O(n_cells) auxiliary

**Use cases:**
- Stem cell identification
- Pluripotency assessment
- Developmental stage analysis

### differentiation_potential

Compute differentiation potential score (CytoTRACE-style) for each cell:

```cpp
Array<Real> potency_scores(n_cells);

scl::kernel::state::differentiation_potential(
    expression,
    potency_scores
);
```

**Parameters:**
- `expression`: Expression matrix (cells x genes, CSR)
- `potency_scores`: Normalized potency scores [0, 1] [n_cells]

**Postconditions:**
- `potency_scores[i]` contains normalized potency score in [0, 1]
- Higher scores indicate greater differentiation potential
- Matrix is unchanged

**Algorithm:**
1. Count expressed genes per cell
2. Compute correlation of each gene with gene count
3. Select top correlated genes
4. Compute weighted sum of top gene expressions
5. Normalize to [0, 1] range

**Complexity:**
- Time: O(n_cells * n_genes * log(nnz_per_cell) + n_genes * log(n_genes))
- Space: O(n_cells + n_genes) auxiliary

**Use cases:**
- Differentiation potential ranking
- Developmental trajectory analysis
- Stemness vs differentiation

### proliferation_score

Compute proliferation score for each cell based on proliferation gene expression:

```cpp
Array<const Index> proliferation_genes = /* ... */;
Array<Real> scores(n_cells);

scl::kernel::state::proliferation_score(
    expression,
    proliferation_genes,
    scores
);
```

**Parameters:**
- `proliferation_genes`: Proliferation gene indices [n_proliferation_genes]
- `scores`: Output Z-score normalized proliferation scores [n_cells]

**Postconditions:**
- `scores[i]` contains z-score normalized proliferation score
- Scores have mean 0 and standard deviation 1

**Use cases:**
- Cell cycle activity
- Growth rate estimation
- Proliferating cell identification

### stress_score

Compute stress score for each cell based on stress gene expression:

```cpp
Array<const Index> stress_genes = /* ... */;
Array<Real> scores(n_cells);

scl::kernel::state::stress_score(
    expression,
    stress_genes,
    scores
);
```

**Parameters:**
- `stress_genes`: Stress gene indices [n_stress_genes]
- `scores`: Output Z-score normalized stress scores [n_cells]

**Use cases:**
- Stress response detection
- Cell health assessment
- Quality control

## Cell Cycle

### cell_cycle_score

Compute cell cycle phase scores (G1/S/G2M) and assign phase labels:

```cpp
Array<const Index> s_genes = /* ... */;  // S-phase genes
Array<const Index> g2m_genes = /* ... */;  // G2/M-phase genes
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
```

**Parameters:**
- `s_genes`: Indices of S-phase genes [n_s_genes]
- `g2m_genes`: Indices of G2/M-phase genes [n_g2m_genes]
- `s_scores`: Output Z-score normalized S-phase scores [n_cells]
- `g2m_scores`: Output Z-score normalized G2/M-phase scores [n_cells]
- `phase_labels`: Output phase labels: 0=G1, 1=S, 2=G2M [n_cells]

**Postconditions:**
- `s_scores[i]` and `g2m_scores[i]` are z-score normalized
- `phase_labels[i]` indicates assigned phase (0, 1, or 2)

**Algorithm:**
1. Compute S-phase and G2/M-phase scores for each cell
2. Z-score normalize both scores
3. Assign phase: S if s_score > 0 and s_score > g2m_score,
   G2M if g2m_score > 0 and g2m_score > s_score, else G1

**Complexity:**
- Time: O(n_cells * (n_s_genes + n_g2m_genes) * log(nnz_per_cell))
- Space: O(n_cells) auxiliary

**Use cases:**
- Cell cycle phase assignment
- Proliferation analysis
- Cell cycle regression

## Metabolic Scores

### metabolic_score

Compute glycolysis and OXPHOS (oxidative phosphorylation) scores:

```cpp
Array<const Index> glycolysis_genes = /* ... */;
Array<const Index> oxphos_genes = /* ... */;
Array<Real> glycolysis_scores(n_cells);
Array<Real> oxphos_scores(n_cells);

scl::kernel::state::metabolic_score(
    expression,
    glycolysis_genes,
    oxphos_genes,
    glycolysis_scores,
    oxphos_scores
);
```

**Parameters:**
- `glycolysis_genes`: Glycolysis gene indices [n_glycolysis_genes]
- `oxphos_genes`: OXPHOS gene indices [n_oxphos_genes]
- `glycolysis_scores`: Output Z-score normalized glycolysis scores [n_cells]
- `oxphos_scores`: Output Z-score normalized OXPHOS scores [n_cells]

**Postconditions:**
- Both scores are z-score normalized
- Matrix is unchanged

**Use cases:**
- Metabolic pathway activity
- Energy production analysis
- Metabolic state classification

## Apoptosis

### apoptosis_score

Compute apoptosis score for each cell based on apoptosis gene expression:

```cpp
Array<const Index> apoptosis_genes = /* ... */;
Array<Real> scores(n_cells);

scl::kernel::state::apoptosis_score(
    expression,
    apoptosis_genes,
    scores
);
```

**Parameters:**
- `apoptosis_genes`: Apoptosis gene indices [n_apoptosis_genes]
- `scores`: Output Z-score normalized apoptosis scores [n_cells]

**Use cases:**
- Cell death detection
- Apoptosis pathway activity
- Cell viability assessment

## Signature Scoring

### signature_score

Compute weighted gene signature score for each cell:

```cpp
Array<const Index> gene_indices = /* ... */;  // Signature gene indices
Array<const Real> gene_weights = /* ... */;  // Gene weights
Array<Real> scores(n_cells);

scl::kernel::state::signature_score(
    expression,
    gene_indices,
    gene_weights,
    scores
);
```

**Parameters:**
- `gene_indices`: Indices of signature genes [n_signature_genes]
- `gene_weights`: Weights for each signature gene [n_signature_genes]
- `scores`: Output Z-score normalized signature scores [n_cells]

**Postconditions:**
- `scores[i]` contains weighted signature score for cell i
- Scores are z-score normalized

**Algorithm:**
1. For each cell, compute weighted sum of signature gene expressions
2. Normalize by sum of absolute weights
3. Z-score normalize across all cells

**Complexity:**
- Time: O(n_cells * n_signature_genes * log(nnz_per_cell))
- Space: O(n_cells) auxiliary

**Use cases:**
- Custom gene signature scoring
- Pathway activity assessment
- Functional state analysis

### multi_signature_score

Compute scores for multiple gene signatures simultaneously:

```cpp
const Index* signature_gene_indices = /* ... */;  // Flat array
const Size* signature_offsets = /* ... */;  // [n_signatures + 1]
Real* score_matrix = /* allocate n_cells * n_signatures */;

scl::kernel::state::multi_signature_score(
    expression,
    signature_gene_indices,
    signature_offsets,
    n_signatures,
    score_matrix
);
```

**Parameters:**
- `signature_gene_indices`: Flat array of all gene indices [total_genes]
- `signature_offsets`: Start offset for each signature [n_signatures + 1]
- `n_signatures`: Number of signatures
- `score_matrix`: Output score matrix [n_cells * n_signatures]

**Postconditions:**
- `score_matrix[i * n_signatures + s]` contains z-score normalized score
  for cell i and signature s
- Each signature column is independently z-score normalized

**Complexity:**
- Time: O(n_signatures * n_cells * avg_signature_size * log(nnz_per_cell))
- Space: O(n_cells * n_signatures) auxiliary

**Use cases:**
- Multiple signature scoring
- Pathway activity matrix
- Functional state profiling

## Diversity Measures

### state_entropy

Compute expression entropy (plasticity) for each cell:

```cpp
Array<Real> entropy_scores(n_cells);

scl::kernel::state::state_entropy(
    expression,
    entropy_scores
);
```

**Parameters:**
- `entropy_scores`: Normalized entropy scores [0, 1] [n_cells]

**Postconditions:**
- `entropy_scores[i]` contains normalized Shannon entropy for cell i
- Scores are in [0, 1], where 1 indicates maximum diversity

**Algorithm:**
For each cell:
1. Compute total expression
2. Compute Shannon entropy: -sum(p_i * log(p_i))
3. Normalize by maximum possible entropy (log(n_genes))

**Complexity:**
- Time: O(nnz)
- Space: O(1) auxiliary per cell

**Use cases:**
- Expression diversity
- Cell plasticity
- State heterogeneity

### transcriptional_diversity

Compute Simpson's diversity index for expression distribution:

```cpp
Array<Real> diversity_scores(n_cells);

scl::kernel::state::transcriptional_diversity(
    expression,
    diversity_scores
);
```

**Parameters:**
- `diversity_scores`: Diversity scores [0, 1] [n_cells]

**Postconditions:**
- `diversity_scores[i]` contains Simpson's diversity index for cell i
- Scores are in [0, 1], where 1 indicates maximum diversity

**Algorithm:**
For each cell:
1. Compute total expression and sum of squared expressions
2. Simpson's index = 1 - sum(p_i^2) where p_i = value_i / total

**Complexity:**
- Time: O(nnz)
- Space: O(1) auxiliary per cell

**Use cases:**
- Expression diversity
- Alternative to entropy
- Diversity quantification

### expression_complexity

Compute expression complexity as fraction of genes expressed above threshold:

```cpp
Array<Real> complexity_scores(n_cells);

scl::kernel::state::expression_complexity(
    expression,
    expression_threshold,  // Minimum expression value
    complexity_scores
);
```

**Parameters:**
- `expression_threshold`: Minimum expression value to count as expressed
- `complexity_scores`: Complexity scores [0, 1] [n_cells]

**Postconditions:**
- `complexity_scores[i]` = n_expressed_genes / n_total_genes for cell i
- Scores are in [0, 1]

**Algorithm:**
For each cell:
1. Count genes with expression > threshold
2. Normalize by total number of genes

**Complexity:**
- Time: O(nnz)
- Space: O(1) auxiliary per cell

**Use cases:**
- Gene expression complexity
- Transcriptional activity
- Cell state complexity

## Combined Scoring

### quiescence_score

Compute quiescence score as difference between quiescence and proliferation:

```cpp
Array<const Index> quiescence_genes = /* ... */;
Array<const Index> proliferation_genes = /* ... */;
Array<Real> scores(n_cells);

scl::kernel::state::quiescence_score(
    expression,
    quiescence_genes,
    proliferation_genes,
    scores
);
```

**Parameters:**
- `quiescence_genes`: Quiescence gene indices [n_quiescence_genes]
- `proliferation_genes`: Proliferation gene indices [n_proliferation_genes]
- `scores`: Quiescence scores [n_cells]

**Postconditions:**
- `scores[i]` = quiescence_score[i] - proliferation_score[i]
- Scores are z-score normalized differences

**Use cases:**
- Quiescent cell identification
- Growth vs quiescence balance
- Cell state classification

### combined_state_score

Compute combined state score from multiple gene sets with weights:

```cpp
const Index* const* gene_sets = /* ... */;  // Array of gene set pointers
const Size* gene_set_sizes = /* ... */;
const Real* weights = /* ... */;
Array<Real> combined_scores(n_cells);

scl::kernel::state::combined_state_score(
    expression,
    gene_sets,
    gene_set_sizes,
    weights,
    n_gene_sets,
    combined_scores
);
```

**Parameters:**
- `gene_sets`: Array of gene set pointers [n_gene_sets]
- `gene_set_sizes`: Size of each gene set [n_gene_sets]
- `weights`: Weight for each gene set [n_gene_sets]
- `n_gene_sets`: Number of gene sets
- `combined_scores`: Combined scores [n_cells]

**Postconditions:**
- `combined_scores[i]` contains weighted combination of all gene set scores
- Matrix is unchanged

**Algorithm:**
1. Compute individual scores for each gene set
2. Z-score normalize each gene set score
3. Compute weighted combination

**Complexity:**
- Time: O(n_gene_sets * n_cells * avg_gene_set_size * log(nnz_per_cell))
- Space: O(n_gene_sets * n_cells) auxiliary

**Use cases:**
- Multi-gene set scoring
- Composite state scores
- Weighted signature combination

## Configuration

Default parameters in `scl::kernel::state::config`:

```cpp
namespace config {
    constexpr Real EPSILON = 1e-10;
    constexpr Size MIN_GENES_FOR_SCORE = 3;
    constexpr Real PSEUDOCOUNT = 1.0;
    constexpr Size PARALLEL_THRESHOLD = 64;
}
```

## Performance Considerations

### Parallelization

All state scoring functions are parallelized:
- `stemness_score`: Parallel over cells
- `differentiation_potential`: Parallel over cells and genes
- `cell_cycle_score`: Parallel over cells
- `multi_signature_score`: Parallel over signatures

### Memory Efficiency

- Pre-allocated output buffers
- Efficient sparse matrix access
- Minimal temporary allocations

## Best Practices

### 1. Use Appropriate Gene Sets

```cpp
// Standard gene sets
Array<const Index> stemness_genes = /* ... */;  // e.g., NANOG, SOX2, OCT4
Array<const Index> proliferation_genes = /* ... */;  // e.g., MKI67, PCNA

scl::kernel::state::stemness_score(expression, stemness_genes, scores);
```

### 2. Combine Multiple Scores

```cpp
// Compute multiple state scores
Array<Real> stemness(n_cells);
Array<Real> proliferation(n_cells);
Array<Real> stress(n_cells);

scl::kernel::state::stemness_score(expression, stemness_genes, stemness);
scl::kernel::state::proliferation_score(expression, prolif_genes, proliferation);
scl::kernel::state::stress_score(expression, stress_genes, stress);

// Use for downstream analysis
```

### 3. Use Multi-Signature for Efficiency

```cpp
// When scoring many signatures
const Index* all_genes = /* ... */;
const Size* offsets = /* ... */;
Real* score_matrix = /* allocate n_cells * n_signatures */;

scl::kernel::state::multi_signature_score(
    expression, all_genes, offsets, n_signatures, score_matrix
);
```

## Examples

### Complete State Analysis

```cpp
// 1. Stemness
Array<Real> stemness(n_cells);
scl::kernel::state::stemness_score(expression, stemness_genes, stemness);

// 2. Differentiation potential
Array<Real> potency(n_cells);
scl::kernel::state::differentiation_potential(expression, potency);

// 3. Cell cycle
Array<Real> s_scores(n_cells), g2m_scores(n_cells);
Array<Index> phases(n_cells);
scl::kernel::state::cell_cycle_score(
    expression, s_genes, g2m_genes, s_scores, g2m_scores, phases
);

// 4. Metabolic state
Array<Real> glycolysis(n_cells), oxphos(n_cells);
scl::kernel::state::metabolic_score(
    expression, glyco_genes, oxphos_genes, glycolysis, oxphos
);

// 5. Expression diversity
Array<Real> entropy(n_cells);
scl::kernel::state::state_entropy(expression, entropy);
```

---

::: tip Gene Sets
Use well-validated gene sets (e.g., from MSigDB, CellMarker) for reliable state scoring.
:::

::: warning Normalization
All scores are z-score normalized. Consider the distribution of scores when interpreting results.
:::

