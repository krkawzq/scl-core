# Utilities

Helper functions and utilities.

::: tip Status
This section is under construction.
:::

## Overview

The `scl.utils` module provides utility functions for data manipulation and conversion.

## Functions

### Configuration
- `set_num_threads()` - Set parallelization level
- `get_num_threads()` - Get current thread count

### Data Validation
- `validate_sparse()` - Check sparse matrix format
- `validate_anndata()` - Check AnnData structure

### Conversion
- `to_csr()` - Convert to CSR format
- `to_dense()` - Convert sparse to dense

## Coming Soon

Detailed API reference will be auto-generated from Python docstrings.

## Example

```python
import scl

# Set thread count
scl.utils.set_num_threads(8)

# Validate data
scl.utils.validate_sparse(matrix)
```

