# Statistics

Statistical analysis and differential expression.

::: tip Status
This section is under construction.
:::

## Overview

The `scl.stats` module provides statistical tests for differential expression and other analyses.

## Functions

### Differential Expression
- `mannwhitneyu()` - Mann-Whitney U test
- `wilcoxon()` - Wilcoxon rank-sum test
- `t_test()` - Student's t-test

### Multiple Testing
- `benjamini_hochberg()` - FDR correction

### Group Analysis
- `rank_genes_groups()` - DE analysis for all groups

## Coming Soon

Detailed API reference will be auto-generated from Python docstrings.

## Example

```python
import scl

# Differential expression between groups
scl.stats.rank_genes_groups(adata, groupby='cell_type', method='mannwhitneyu')

# Access results
results = adata.uns['rank_genes_groups']
```

