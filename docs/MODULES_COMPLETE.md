# SCL-Core Complete Module Inventory

## Strategic Position: Sparse + Nonlinear

This document provides a complete inventory of all SCL-Core modules, both implemented and planned.

---

## 📊 Module Statistics

```
Total Modules: 40
├─ Implemented: 23 (57.5%)
└─ Planned: 17 (42.5%)

By Tier:
├─ Tier 1 (Core Battlefield): 14 modules
│  ├─ Implemented: 7
│  └─ Planned: 7
├─ Tier 2 (Extension): 6 modules
│  ├─ Implemented: 3
│  └─ Planned: 3
├─ Tier 3 (Biology-Specific): 10 modules
│  ├─ Implemented: 5
│  └─ Planned: 5
└─ Tier 4 (Advanced): 10 modules
   ├─ Implemented: 8
   └─ Planned: 2
```

---

## ✅ Implemented Modules (23)

### Preprocessing & Normalization (4)
| Module | File | Core Functions | Status |
|--------|------|----------------|--------|
| Normalize | `normalize.hpp` | Row/column normalization, scaling | ✅ v0.3 |
| Log Transform | `log1p.hpp` | log(1+x) transformation | ✅ v0.3 |
| Scale | `scale.hpp` | Standardization, z-score | ✅ v0.3 |
| Softmax | `softmax.hpp` | Softmax normalization | ✅ v0.3 |

### Feature Selection (2)
| Module | File | Core Functions | Status |
|--------|------|----------------|--------|
| HVG | `hvg.hpp` | Highly variable gene selection | ✅ v0.3 |
| Feature | `feature.hpp` | Feature counting, filtering | ✅ v0.3 |

### Nearest Neighbors & Graphs (3)
| Module | File | Core Functions | Status |
|--------|------|----------------|--------|
| Neighbors | `neighbors.hpp` | KNN computation, cosine/euclidean | ✅ v0.3 |
| BBKNN | `bbknn.hpp` | Batch-balanced KNN | ✅ v0.3 |
| Gram | `gram.hpp` | Gram matrix, sparse dot product | ✅ v0.3 |

### Statistical Tests (3)
| Module | File | Core Functions | Status |
|--------|------|----------------|--------|
| T-test | `ttest.hpp` | Welch/Student t-tests | ✅ v0.3 |
| Mann-Whitney U | `mwu.hpp` | Nonparametric rank test | ✅ v0.3 |
| MMD | `mmd.hpp` | Maximum Mean Discrepancy | ✅ v0.3 |

### Spatial Analysis (2)
| Module | File | Core Functions | Status |
|--------|------|----------------|--------|
| Spatial | `spatial.hpp` | Moran's I, Geary's C, spatial lag | ✅ v0.3 |
| Correlation | `correlation.hpp` | Pearson/Spearman correlation | ✅ v0.3 |

### Linear Algebra (1)
| Module | File | Core Functions | Status |
|--------|------|----------------|--------|
| Algebra | `algebra.hpp` | SpMV, SpMM, transpose | ✅ v0.3 |

### Matrix Operations (4)
| Module | File | Core Functions | Status |
|--------|------|----------------|--------|
| Slice | `slice.hpp` | Matrix slicing, submatrix extraction | ✅ v0.3 |
| Merge | `merge.hpp` | Matrix concatenation | ✅ v0.3 |
| Reorder | `reorder.hpp` | Matrix reordering, permutation | ✅ v0.3 |
| Sparse Utils | `sparse.hpp` | CSR↔CSC conversion | ✅ v0.3 |

### Quality Control (1)
| Module | File | Core Functions | Status |
|--------|------|----------------|--------|
| QC | `qc.hpp` | QC metrics, outlier detection | ✅ v0.3 |

### Utilities (2)
| Module | File | Core Functions | Status |
|--------|------|----------------|--------|
| Group | `group.hpp` | Grouping, aggregation | ✅ v0.3 |
| Resample | `resample.hpp` | Bootstrap, permutation, sampling | ✅ v0.3 |

---

## 📝 Planned Modules (17)

### Tier 1: Core Battlefield (7 modules) - v0.4.0

| Module | File | Core Functions | Priority | Dependencies |
|--------|------|----------------|----------|--------------|
| **Louvain** | `louvain.hpp` | Graph clustering, modularity | ⭐⭐⭐⭐⭐ | neighbors |
| **Leiden** | `leiden.hpp` | Advanced graph clustering | ⭐⭐⭐⭐⭐ | louvain, neighbors |
| **Components** | `components.hpp` | Connected components, BFS/DFS | ⭐⭐⭐⭐⭐ | - |
| **Propagation** | `propagation.hpp` | Label propagation, semi-supervised | ⭐⭐⭐⭐⭐ | neighbors |
| **Imputation** | `impute.hpp` | KNN/diffusion imputation | ⭐⭐⭐⭐⭐ | neighbors, diffusion |
| **Projection** | `projection.hpp` | Sparse random projection (JL) | ⭐⭐⭐⭐ | - |
| **Permutation** | `permutation.hpp` | Permutation tests, FDR correction | ⭐⭐⭐⭐ | resample |

### Tier 2: Extension Battlefield (3 modules) - v0.5.0

| Module | File | Core Functions | Priority | Dependencies |
|--------|------|----------------|----------|--------------|
| **Diffusion** | `diffusion.hpp` | DPT, random walk, diffusion kernel | ⭐⭐⭐ | neighbors |
| **Centrality** | `centrality.hpp` | PageRank, HITS, betweenness | ⭐⭐⭐⭐ | components |
| **GNN** | `gnn.hpp` | Message passing, attention | ⭐⭐⭐ | neighbors, softmax |

### Tier 3: Biology-Specific (5 modules) - v0.5.0/v0.6.0

| Module | File | Core Functions | Priority | Dependencies |
|--------|------|----------------|----------|--------------|
| **Markers** | `markers.hpp` | Marker gene selection, specificity | ⭐⭐⭐⭐ | ttest, mwu |
| **Scoring** | `scoring.hpp` | Gene set scoring, AUCell, module | ⭐⭐⭐⭐ | - |
| **Hotspot** | `hotspot.hpp` | LISA, Gi*, local spatial stats | ⭐⭐⭐ | spatial |
| **Kernel Methods** | `kernel.hpp` | KDE, RBF kernels, smoothing | ⭐⭐⭐ | neighbors |
| **Doublet** | `doublet.hpp` | Doublet detection, simulation | ⭐⭐⭐ | neighbors |

### Tier 4: Advanced (2 modules) - v0.6.0+

| Module | File | Core Functions | Priority | Dependencies |
|--------|------|----------------|----------|--------------|
| **Entropy** | `entropy.hpp` | MI, KL/JS divergence, feature selection | ⭐⭐ | - |
| **Sparse Opt** | `sparse_opt.hpp` | Lasso, elastic net, proximal methods | ⭐⭐ | - |

---

## 🔗 Module Dependency Graph

```
Level 0 (No dependencies):
├─ normalize, log1p, scale, softmax
├─ hvg, feature, qc
├─ algebra, sparse
├─ group, resample
├─ components
├─ projection
├─ entropy, sparse_opt
└─ scoring (basic)

Level 1 (Depends on Level 0):
├─ neighbors ─────┐
├─ spatial        │
├─ correlation    │
├─ ttest, mwu, mmd│
├─ slice, merge,  │
└─ reorder        │
                  │
Level 2 (Depends on Level 1):
├─ bbknn ◄────────┤
├─ gram ◄─────────┤
├─ louvain ◄──────┤
├─ components ────┤
├─ propagation ◄──┤
├─ centrality ◄───┤
├─ diffusion ◄────┤
├─ kernel ◄───────┤
├─ doublet ◄──────┤
└─ markers ◄──────┘
                  
Level 3 (Depends on Level 2):
├─ leiden ◄─── louvain
├─ impute ◄─── neighbors + diffusion
├─ gnn ◄────── neighbors + softmax
├─ hotspot ◄── spatial
└─ permutation ◄ resample + (ttest/mwu)
```

---

## 📅 Implementation Timeline

### v0.4.0 "Graph & Imputation" (Q2 2025)
**Focus:** Core graph algorithms and data quality

- [x] Framework: louvain, leiden, components, propagation
- [x] Framework: impute, projection, permutation
- [ ] Implementation: louvain (first)
- [ ] Implementation: components
- [ ] Implementation: projection (very sparse variant)
- [ ] Implementation: propagation
- [ ] Implementation: leiden
- [ ] Implementation: permutation
- [ ] Implementation: impute (KNN variant)

### v0.5.0 "Diffusion & Biology" (Q3 2025)
**Focus:** Diffusion processes and biology-specific tools

- [x] Framework: diffusion, centrality, hotspot, markers, scoring
- [ ] Implementation: diffusion (core)
- [ ] Implementation: markers
- [ ] Implementation: scoring
- [ ] Implementation: centrality (PageRank)
- [ ] Implementation: hotspot (LISA, Gi*)
- [ ] Implementation: kernel methods
- [ ] Implementation: doublet detection

### v0.6.0 "Advanced Methods" (Q4 2025)
**Focus:** Advanced algorithms and GNN support

- [x] Framework: gnn, kernel, doublet, entropy, sparse_opt
- [ ] Implementation: GNN primitives
- [ ] Implementation: entropy measures
- [ ] Implementation: sparse optimization
- [ ] Integration examples
- [ ] Performance benchmarks

---

## 🎯 Implementation Priority Queue

### Immediate (This Month)
1. ✅ Create all module frameworks
2. 🔨 Implement `louvain` (simplest graph clustering)
3. 🔨 Implement `projection` (very sparse variant)

### Next Month
4. Implement `components` (graph infrastructure)
5. Implement `propagation` (label propagation)
6. Implement `permutation` (extends resample)

### Following Months
7. Implement `leiden` (extends louvain)
8. Implement `impute` (KNN variant first)
9. Implement `markers` (combines ttest/mwu)
10. Implement `scoring` (gene set scoring)

---

## 💡 Module Design Principles

### 1. Sparse-First Design
Every module preserves sparsity where possible:
```cpp
// ✅ Good: Sparse input → Sparse output
template <typename T, bool IsCSR>
void impute(
    const Sparse<T, IsCSR>& X,
    Sparse<T, IsCSR>& X_imputed  // Still sparse!
);

// ❌ Avoid: Force densification
// void impute(..., DenseMatrix& output);
```

### 2. Consistent Interfaces
All modules follow the same patterns:
```cpp
namespace scl::kernel::<module> {
    namespace config { /* constants */ }
    
    // Main API
    template <typename T, bool IsCSR>
    void main_function(...);
    
    namespace detail { /* helpers */ }
}
```

### 3. Clear Dependencies
Modules have explicit, minimal dependencies:
- Level 0: No dependencies
- Level 1: Only core modules
- Level 2+: Well-defined dependency chain

### 4. Documented Contracts
Every function has:
- Clear preconditions
- Guaranteed postconditions
- Complexity analysis
- Thread safety guarantees

---

## 📝 Next Steps

1. **Review frameworks** - Check interface designs
2. **Start with `louvain`** - Simplest new algorithm
3. **Write tests first** - TDD approach
4. **Implement incrementally** - One module at a time
5. **Benchmark continuously** - Track performance

---

## 🤔 Open Questions

1. Should `components` be in `core/` instead of `kernel/`?
2. Optimal balance between `kernel` methods and full implementations?
3. Priority order: biological utility vs algorithmic complexity?
4. When to add GPU acceleration?

---

## 📚 References

- **Louvain/Leiden**: Fast unfolding of communities (2008), Leiden algorithm (2019)
- **Label Propagation**: Zhou et al. (2004)
- **Random Projection**: Johnson-Lindenstrauss, Li et al. (sparse)
- **LISA**: Anselin (1995)
- **PageRank**: Brin & Page (1998)
- **Sparse Optimization**: Beck & Teboulle (FISTA, 2009)

