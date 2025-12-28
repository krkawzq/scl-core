---
layout: home

hero:
  name: "SCL-Core"
  text: "High-Performance Biological Operator Library"
  tagline: Zero-overhead C++ kernels with stable C-ABI for Python integration
  image:
    src: /logo.svg
    alt: SCL-Core
  actions:
    - theme: brand
      text: Get Started
      link: /cpp/getting-started
    - theme: alt
      text: C++ Developer Guide
      link: /cpp/
    - theme: alt
      text: View on GitHub
      link: https://github.com/krkawzq/scl-core

features:
  - icon: ⚡
    title: Zero-Overhead Performance
    details: Hand-optimized C++ kernels with SIMD support and cache-friendly algorithms for maximum throughput
  
  - icon: 🔬
    title: Biological Operators
    details: Comprehensive suite of operators for single-cell analysis, spatial transcriptomics, and genomics
  
  - icon: 🔗
    title: Stable C-ABI
    details: Clean C interface for seamless Python integration without sacrificing performance
  
  - icon: 🧬
    title: Scientific Computing
    details: Matrix operations, statistical tests, neighborhood graphs, and advanced algorithms
  
  - icon: 🚀
    title: Parallel by Default
    details: Built-in multi-threading support with optimal work distribution and minimal overhead
  
  - icon: 📊
    title: Production Ready
    details: Battle-tested kernels with extensive benchmarks and comprehensive test coverage

---

## Quick Example

```python
import scl_core as scl
import numpy as np

# High-performance sparse matrix normalization
data = scl.load_data("expression.h5ad")
scl.normalize_total(data, target_sum=1e4)
scl.log1p(data)

# Fast nearest neighbor search
scl.neighbors(data, n_neighbors=15, method="annoy")

# Spatial analysis with optimized kernels
scl.spatial.neighbors(data, coord_key="spatial")
scl.spatial.morans_i(data, genes=["CD3D", "CD8A"])
```

## Why SCL-Core?

SCL-Core is designed from the ground up for **maximum performance** in biological data analysis:

- **10-100x faster** than pure Python implementations
- **Memory efficient** sparse matrix operations
- **Cache-friendly** algorithms with optimal data layouts
- **SIMD-accelerated** compute kernels
- **Zero-copy** data sharing between C++ and Python

## Installation

```bash
pip install scl-core
```

Or build from source:

```bash
git clone https://github.com/krkawzq/scl-core.git
cd scl-core
pip install -e .
```

## Core Modules

- **Normalization**: Total count, log transform, scale, clip
- **Neighbors**: KNN, approximate NN, batch-aware BBKNN
- **Statistical Tests**: T-test, Mann-Whitney U, permutation tests
- **Spatial**: Moran's I, spatial autocorrelation, neighborhood graphs
- **Algebra**: Matrix operations, reductions, transformations
- **Feature Selection**: Highly variable genes, marker detection

## Benchmarks

<PerformanceChart 
  title="Normalization Performance"
  description="Comparison of SCL-Core vs. pure Python implementation on 10k cells × 2k genes"
/>

::: tip Performance Tip
SCL-Core automatically detects the number of available CPU cores and parallelizes operations. For optimal performance, ensure your data is in CSR format for row-based operations and CSC format for column-based operations.
:::

## Community

- [GitHub Discussions](https://github.com/krkawzq/scl-core/discussions) - Ask questions and share ideas
- [Issue Tracker](https://github.com/krkawzq/scl-core/issues) - Report bugs and request features
- [Contributing Guide](/CONTRIBUTING.md) - Learn how to contribute

## License

SCL-Core is released under the [MIT License](https://github.com/krkawzq/scl-core/blob/main/LICENSE).

