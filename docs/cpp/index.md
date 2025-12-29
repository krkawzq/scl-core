# C++ API Overview

This section documents the C++ kernel APIs of SCL Core.

## Modules

| Module | Description |
|--------|-------------|
| `scl::core` | Core types, memory utilities, and error handling |
| `scl::kernel` | Computational kernels |
| `scl::math` | Mathematical functions |
| `scl::threading` | Parallelization layer |

---

## Component Showcase

### Badges

<Badge>Default</Badge>
<Badge type="version">0.4.0</Badge>
<Badge type="status" color="green">Stable</Badge>
<Badge type="complexity">O(n log n)</Badge>
<Badge color="red">Experimental</Badge>

### Callouts

<Callout type="tip" title="Performance Tip">
Use SIMD-optimized functions for best performance on large datasets.
</Callout>

<Callout type="warning">
This function modifies the input data in-place. Make a copy if you need to preserve the original.
</Callout>

<Callout type="info" title="Thread Safety" collapsible>
All functions in this module are thread-safe when operating on different data regions.
</Callout>

### API Signature

<ApiSignature
  return-type="void"
  name="compute_pca"
  :template="['typename T']"
>
  <span class="type">Span&lt;T&gt;</span> <span class="param-name">data</span>,
  <span class="type">Span&lt;T&gt;</span> <span class="param-name">output</span>,
  <span class="type">Index</span> <span class="param-name">n_components</span> <span class="default-value">= 50</span>
</ApiSignature>

### Parameter Table

<ParamTable :params="[
  { name: 'data', type: 'Span<T>', dir: 'in', description: 'Input data matrix (n_samples × n_features)', required: true },
  { name: 'output', type: 'Span<T>', dir: 'out', description: 'Output principal components', required: true },
  { name: 'n_components', type: 'Index', dir: 'in', description: 'Number of components to compute', default: '50' }
]" />

### Support Matrix

<SupportMatrix :features="[
  { name: 'compute_pca', numpy: true, sparse: true, dask: true, gpu: false },
  { name: 'neighbors', numpy: true, sparse: true, dask: 'partial', gpu: true },
  { name: 'leiden', numpy: true, sparse: true, dask: false, gpu: false }
]" />

### Algorithm Card

<AlgoCard title="Principal Component Analysis" icon="📊" :references="['Pearson 1901', 'Hotelling 1933']">
  <template #formula>

$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

  </template>

PCA finds orthogonal directions of maximum variance in the data. The principal components are the columns of $\mathbf{V}$.

  <template #complexity>
    <Badge type="complexity">Time: O(n·d·k)</Badge>
    <Badge type="complexity">Space: O(d·k)</Badge>
  </template>
</AlgoCard>

### Steps

<Steps>
  <Step title="Install dependencies">

```bash
pip install scl-core
```

  </Step>
  <Step title="Import module">

```python
import scl
```

  </Step>
  <Step title="Run analysis">

```python
result = scl.kernel.compute_pca(data, n_components=50)
```

  </Step>
</Steps>

### Code Tabs

<CodeTabs :tabs="['C++', 'Python', 'C API']">
  <template #c-->

```cpp
#include <scl/kernel/pca.hpp>

scl::kernel::compute_pca(data, output, 50);
```

  </template>
  <template #python>

```python
import scl

scl.kernel.compute_pca(data, n_components=50)
```

  </template>
  <template #c-api>

```c
#include <scl/c_api.h>

scl_compute_pca(data, output, 50);
```

  </template>
</CodeTabs>

### See Also

<SeeAlso :links="[
  { href: '/cpp/kernels/svd', text: 'SVD decomposition' },
  { href: '/cpp/kernels/tsne', text: 't-SNE embedding' },
  { href: '/cpp/kernels/umap', text: 'UMAP projection' }
]" />

### Source Link

<SourceLink file="scl/kernel/pca.hpp" :line="42" />

### Version Badge

<Since version="0.3.0" /> Added in version 0.3.0

### Deprecated Warning

<Deprecated since="0.4.0" use="compute_pca_v2">
This function has performance issues with sparse matrices.
</Deprecated>
