# C++ API Overview

This section documents the C++ kernel APIs of SCL Core.

## Modules

| Module | Description |
|--------|-------------|
| `scl::core` | Core types, memory utilities, and error handling |
| `scl::kernel` | Computational kernels |
| `scl::math` | Mathematical functions |
| `scl::threading` | Parallelization layer |

## Quick Example

Below is an example demonstrating how documentation components work:

<ApiSignature return-type="void" name="compute_pca" template-params="typename T">
  <span class="type">Span&lt;T&gt;</span> <span class="param-name">data</span>,
  <span class="type">Span&lt;T&gt;</span> <span class="param-name">output</span>,
  <span class="type">Index</span> <span class="param-name">n_components</span>
</ApiSignature>

<ParamTable :params="[
  { name: 'data', type: 'Span<T>', direction: 'in', description: 'Input data matrix (n_samples × n_features)', required: true },
  { name: 'output', type: 'Span<T>', direction: 'out', description: 'Output principal components', required: true },
  { name: 'n_components', type: 'Index', direction: 'in', description: 'Number of components to compute', default: '50' }
]" />

### Complexity

<ComplexityBadge type="time" complexity="O(n·d·k)" />
<ComplexityBadge type="space" complexity="O(d·k)" />

<AlgoCard title="PCA Algorithm">

Principal Component Analysis finds orthogonal directions of maximum variance:

$$
\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$

The principal components are the columns of $\mathbf{V}$.

</AlgoCard>

<SourceLink file="scl/kernel/pca.hpp" :line="42" />
