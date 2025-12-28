# gnn.hpp

> scl/kernel/gnn.hpp · Graph neural network operations for learning node representations

## Overview

This file provides efficient graph neural network (GNN) operations for single-cell and spatial transcriptomics data:

- **Graph Convolution**: Standard GCN layer with degree normalization
- **Graph Attention**: GAT-style attention mechanism with learnable weights

All operations are optimized for sparse adjacency matrices and parallelized for efficient computation.

**Header**: `#include "scl/kernel/gnn.hpp"`

---

## Main APIs

### graph_convolution

::: source_code file="scl/kernel/gnn.hpp" symbol="graph_convolution" collapsed
:::

**Algorithm Description**

Apply graph convolution layer following the GCN formulation: H' = (D^-1 A) H W

1. **Normalize adjacency**: Compute inverse degree matrix D^-1 and apply to adjacency A
   - For each node i: `D[i,i] = sum(adjacency[i,:])` (degree)
   - Normalized adjacency: `D^-1 A` (row-normalized)
2. **Graph convolution**: Multiply normalized adjacency with node features
   - `H_intermediate = (D^-1 A) H`
   - Each node aggregates features from its neighbors weighted by inverse degree
3. **Linear transformation**: Apply weight matrix W
   - `H' = H_intermediate W`
   - Projects features from input dimension to output dimension

The implementation uses sparse matrix-vector multiplication for efficient neighbor aggregation.

**Edge Cases**

- **Isolated nodes (degree=0)**: Division by zero avoided, isolated nodes use identity features
- **Empty graph**: Returns zero output
- **Self-loops**: Handled correctly in degree computation
- **Disconnected components**: Each component processed independently

**Data Guarantees (Preconditions)**

- `output` has capacity >= `n_nodes * n_output_features`
- Adjacency matrix should be CSR format for optimal performance
- Node features stored in row-major layout: `features[i * n_features + j]` = feature j of node i
- Weight matrix stored in row-major layout: `weights[i * n_output_features + j]` = weight from input feature i to output feature j

**Complexity Analysis**

- **Time**: O(nnz * n_features + n_nodes * n_features * n_output_features)
  - O(nnz * n_features) for graph convolution (sparse mat-vec)
  - O(n_nodes * n_features * n_output_features) for linear transformation
- **Space**: O(n_nodes * n_output_features) auxiliary space for intermediate results

**Example**

```cpp
#include "scl/kernel/gnn.hpp"

// Build k-NN graph from expression data
scl::Sparse<Real, true> adjacency = /* ... */;  // [n_nodes x n_nodes]

// Node features (e.g., gene expression)
const Real* node_features = /* ... */;  // [n_nodes * n_features]
scl::Index n_nodes = 1000;
scl::Index n_features = 2000;

// Weight matrix (learnable parameters)
const Real* weights = /* ... */;  // [n_features * n_output_features]
scl::Index n_output_features = 128;

// Pre-allocate output
Real* output = new Real[n_nodes * n_output_features];

// Apply graph convolution
scl::kernel::gnn::graph_convolution(
    adjacency, node_features, n_nodes, n_features,
    weights, n_output_features, output
);

// Output contains convolved features: output[i * n_output_features + j]
```

---

### graph_attention

::: source_code file="scl/kernel/gnn.hpp" symbol="graph_attention" collapsed
:::

**Algorithm Description**

Apply graph attention layer following GAT (Graph Attention Network) formulation:

1. **Compute attention scores**: For each edge (i, j) in the graph
   - Transform node features: `h_i' = W h_i`, `h_j' = W h_j`
   - Compute attention: `a_ij = LeakyReLU(concat(h_i', h_j') @ attention_weights)`
   - LeakyReLU with slope `alpha` (default 0.5): `max(alpha * x, x)`
2. **Normalize attention**: Apply softmax over neighbors
   - `a_ij' = softmax_j(a_ij)` for each node i
   - Ensures attention weights sum to 1 for each node
3. **Aggregate features**: Weighted sum of neighbor features
   - `h'_i = sum_j(a_ij' * W h_j)`
   - Each node aggregates features from neighbors weighted by attention scores

The attention mechanism allows nodes to learn which neighbors are most important for the task.

**Edge Cases**

- **Isolated nodes**: Attention = 1.0 (self-attention), output = transformed self-features
- **Empty graph**: Returns zero output
- **Self-loops**: Included in attention computation
- **Disconnected components**: Each component processed independently

**Data Guarantees (Preconditions)**

- `output` has capacity >= `n_nodes * n_features`
- Adjacency matrix should be CSR format
- `attention_weights` is a square matrix [n_features * n_features] for computing attention scores
- Node features in row-major layout

**Complexity Analysis**

- **Time**: O(nnz * n_features + n_nodes * n_features^2)
  - O(nnz * n_features) for neighbor aggregation
  - O(n_nodes * n_features^2) for attention score computation
- **Space**: O(n_nodes * n_features) auxiliary space

**Example**

```cpp
#include "scl/kernel/gnn.hpp"

scl::Sparse<Real, true> adjacency = /* ... */;
const Real* node_features = /* ... */;  // [n_nodes * n_features]
scl::Index n_nodes = 1000;
scl::Index n_features = 2000;

// Attention weight matrix (learnable)
const Real* attention_weights = /* ... */;  // [n_features * n_features]

// Pre-allocate output
Real* output = new Real[n_nodes * n_features];

// Apply graph attention with LeakyReLU alpha = 0.2
scl::kernel::gnn::graph_attention(
    adjacency, node_features, n_nodes, n_features,
    attention_weights, output, 0.2
);

// Output contains attended features
```

---

## Configuration

The module provides configuration constants in `scl::kernel::gnn::config`:

- `DEFAULT_ALPHA`: Default LeakyReLU slope for attention (0.5)
- `EPSILON`: Small constant for numerical stability (1e-15)
- `PARALLEL_THRESHOLD`: Minimum size for parallelization (256)
- `SIMD_THRESHOLD`: Minimum size for SIMD operations (16)

---

## Notes

- Graph convolution uses normalized adjacency (D^-1 A) internally, so input adjacency can be unnormalized
- Attention mechanism is more expressive but computationally more expensive than convolution
- For large graphs, consider using approximate attention or neighbor sampling
- Feature dimensions should be power-of-2 for optimal SIMD performance
- Both operations are thread-safe and parallelized

## See Also

- [Sparse Matrix Operations](../core/sparse)
- [Neighbor Graph Construction](./neighbors)
- [Graph Algorithms](./components)
