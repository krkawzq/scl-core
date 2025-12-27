# Kernels Overview

High-performance computational kernels exposed through the C API.

::: tip Status
This section is under construction.
:::

## Available Kernels

### Normalization
- [Normalize](/api/c-api/kernels/normalize) - Matrix normalization operations

### Neighbors
- Neighbors (coming soon) - K-nearest neighbors computation

### Algebra
- Algebra (coming soon) - Matrix algebra operations

### Statistics
- Statistics (coming soon) - Statistical tests and analysis

## Kernel Design

All kernels follow these principles:

1. **Thread-safe**: Can be called from multiple threads
2. **Zero-overhead**: Direct access to optimized implementations
3. **Explicit errors**: All errors returned via error codes
4. **Minimal allocation**: Preallocated buffers where possible

## Coming Soon

Detailed documentation for each kernel.

