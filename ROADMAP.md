# SCL-Core Development Roadmap

## Strategic Position

**Core Philosophy:** Sparse + Nonlinear

SCL-Core focuses on algorithms that:
1. ✅ Operate efficiently on sparse matrices
2. ✅ Involve nonlinear operations
3. ✅ Avoid overlapping with mature dense linear algebra libraries (Eigen/BLAS)

```
                    Dense                    Sparse
        ┌─────────────────────┬─────────────────────────┐
        │                     │                         │
Linear  │  ❌ Eigen/BLAS      │  ⚠️ Has competition    │
        │  (PCA, SVD, etc.)   │  (SpMV, SpMM)          │
        │                     │                         │
        ├─────────────────────┼─────────────────────────┤
        │                     │                         │
Nonlin  │  ⚠️ Selective entry │  ✅ SCL Battlefield     │
        │  (if unique value)  │  (Graph, Stats,         │
        │                     │   Spatial, Sampling)    │
        │                     │                         │
        └─────────────────────┴─────────────────────────┘
```

---

## Implementation Status

## Current Status Summary

### ✅ Implemented Modules (23 modules)

#### Preprocessing & Normalization (4 modules)
- [x] `normalize.hpp` - Row/column normalization
- [x] `log1p.hpp` - Log transformation
- [x] `scale.hpp` - Standardization
- [x] `softmax.hpp` - Softmax normalization

#### Feature Selection (2 modules)
- [x] `hvg.hpp` - Highly variable genes
- [x] `feature.hpp` - Feature counting/filtering

#### Nearest Neighbors (3 modules)
- [x] `neighbors.hpp` - KNN computation
- [x] `bbknn.hpp` - Batch-balanced KNN
- [x] `gram.hpp` - Gram matrix

#### Statistical Tests (3 modules)
- [x] `ttest.hpp` - T-tests
- [x] `mwu.hpp` - Mann-Whitney U test
- [x] `mmd.hpp` - Maximum Mean Discrepancy

#### Spatial Analysis (2 modules)
- [x] `spatial.hpp` - Spatial autocorrelation
- [x] `correlation.hpp` - Correlation matrices

#### Matrix Operations (5 modules)
- [x] `algebra.hpp` - SpMV, SpMM
- [x] `slice.hpp` - Matrix slicing
- [x] `merge.hpp` - Matrix concatenation
- [x] `reorder.hpp` - Matrix reordering
- [x] `sparse.hpp` - Format conversion

#### Quality Control (1 module)
- [x] `qc.hpp` - QC metrics

#### Utilities (2 modules)
- [x] `group.hpp` - Grouping/aggregation
- [x] `resample.hpp` - Resampling

**Total: 23 modules**

### 📝 Framework Created (17 new modules)

#### Tier 1: Core Battlefield (7 modules)
- [ ] `leiden.hpp` - Leiden clustering ⭐⭐⭐⭐⭐
- [ ] `louvain.hpp` - Louvain clustering ⭐⭐⭐⭐⭐
- [ ] `impute.hpp` - Sparse-preserving imputation ⭐⭐⭐⭐⭐
- [ ] `projection.hpp` - Sparse random projection ⭐⭐⭐⭐
- [ ] `permutation.hpp` - Permutation testing ⭐⭐⭐⭐
- [ ] `components.hpp` - Connected components ⭐⭐⭐⭐⭐
- [ ] `propagation.hpp` - Label propagation ⭐⭐⭐⭐⭐

#### Tier 2: Extension Battlefield (3 modules)
- [ ] `diffusion.hpp` - Diffusion processes ⭐⭐⭐
- [ ] `gnn.hpp` - GNN primitives ⭐⭐⭐
- [ ] `centrality.hpp` - Graph centrality ⭐⭐⭐⭐

#### Tier 3: Biology-Specific (5 modules)
- [ ] `markers.hpp` - Marker gene selection ⭐⭐⭐⭐
- [ ] `scoring.hpp` - Gene set scoring ⭐⭐⭐⭐
- [ ] `hotspot.hpp` - Local spatial statistics ⭐⭐⭐
- [ ] `kernel.hpp` - Sparse kernel methods ⭐⭐⭐
- [ ] `doublet.hpp` - Doublet detection ⭐⭐⭐

#### Tier 4: Advanced (2 modules)
- [ ] `entropy.hpp` - Information theory ⭐⭐
- [ ] `sparse_opt.hpp` - Sparse optimization ⭐⭐

**New Total: 40 modules (23 implemented + 17 planned)**

---

## 🎯 Tier 1: Core Battlefield (Immediate Priority)

### 1. Graph Clustering

#### Status: 📝 Framework Created

**Files:**
- `scl/kernel/leiden.hpp` - Leiden clustering
- `scl/kernel/louvain.hpp` - Louvain clustering

**Priority:** ⭐⭐⭐⭐⭐

**Why:**
- Graph algorithms, naturally sparse
- Nonlinear optimization (modularity)
- Zero overlap with Eigen/BLAS
- Essential for cell type identification

**Implementation Order:**
1. Start with Louvain (simpler)
2. Extend to Leiden (refinement phase)

**Dependencies:**
- Existing: `neighbors.hpp` for graph construction
- None external

**Target:** v0.4.0

---

### 2. Sparse Imputation

#### Status: 📝 Framework Created

**Files:**
- `scl/kernel/impute.hpp`

**Priority:** ⭐⭐⭐⭐⭐

**Why:**
- Leverages existing KNN infrastructure
- Maintains sparsity (unique advantage)
- Nonlinear weighted aggregation
- Most libraries convert to dense

**Methods:**
1. KNN-based (distance-weighted)
2. KNN with custom weights
3. MAGIC-like diffusion (requires `diffusion.hpp`)

**Dependencies:**
- Requires: `neighbors.hpp`, `diffusion.hpp`
- Optional: Custom distance kernels

**Target:** v0.4.0

---

### 3. Sparse Random Projection

#### Status: 📝 Framework Created

**Files:**
- `scl/kernel/projection.hpp`

**Priority:** ⭐⭐⭐⭐

**Why:**
- Faster than PCA (no eigendecomposition)
- Sparse-to-sparse transformation
- Distance-preserving (Johnson-Lindenstrauss)
- Lightweight preprocessing

**Methods:**
1. Gaussian projection (baseline)
2. Sparse projection (Li et al.)
3. Very sparse projection (Achlioptas)
4. Count sketch

**Dependencies:**
- None

**Target:** v0.4.0

---

### 4. Permutation Testing

#### Status: 📝 Framework Created

**Files:**
- `scl/kernel/permutation.hpp`

**Priority:** ⭐⭐⭐⭐

**Why:**
- Nonparametric, distribution-free
- Sparse-friendly (operates on indices)
- Extends existing `resample` module
- Critical for DE analysis

**Features:**
1. Generic permutation test framework
2. FDR correction (BH, BY, Bonferroni)
3. Specialized tests (correlation, spatial)

**Dependencies:**
- Extends: `resample.hpp`
- Uses: Existing statistical test modules

**Target:** v0.4.0

---

## 🔶 Tier 2: Extension Battlefield

### 5. Diffusion Processes

#### Status: 📝 Framework Created

**Files:**
- `scl/kernel/diffusion.hpp`

**Priority:** ⭐⭐⭐

**Why:**
- Sparse graph iterations
- Nonlinear (diffusion kernel)
- Multiple applications

**Applications:**
1. Diffusion Pseudotime (DPT)
2. MAGIC-like imputation
3. Graph smoothing
4. Random walk with restart

**Dependencies:**
- Requires: `neighbors.hpp` for graph
- None external

**Target:** v0.5.0

---

### 6. Graph Neural Network Primitives

#### Status: 📝 Framework Created

**Files:**
- `scl/kernel/gnn.hpp`

**Priority:** ⭐⭐⭐

**Why:**
- Foundation for GNN support
- Sparse message passing
- No autograd needed for inference

**Scope:**
- Message passing primitives
- Attention mechanisms
- Graph pooling
- Feature propagation

**NOT in scope:**
- Training (requires autograd)
- Full GNN framework

**Dependencies:**
- Requires: `neighbors.hpp`
- Optional: `softmax.hpp` for attention

**Target:** v0.6.0

---

### 7. Extended Spatial Statistics

#### Status: 🔨 Extends Existing

**Files:**
- Extend `scl/kernel/spatial.hpp`

**Priority:** ⭐⭐⭐

**New Methods:**
1. LISA (Local Indicators of Spatial Association)
2. Getis-Ord Gi* (hot spot analysis)
3. Spatial clustering (DBSCAN on graph)
4. Ripley's K function

**Dependencies:**
- Extends: `spatial.hpp`
- Requires: `neighbors.hpp`

**Target:** v0.5.0

---

## ❌ Explicitly Out of Scope

### Dense Linear Algebra
- **PCA/SVD** - Eigen/BLAS territory
- **LU/QR/Cholesky** - Standard LAPACK
- **Dense matrix operations** - BLAS Level 2/3

**Rationale:** Mature libraries (Eigen) handle this better

### Deep Learning
- **Autograd** - Requires compute graph
- **DCA/scVI** - Needs neural network framework
- **Training loops** - Different architecture

**Rationale:** Separate project/module

### Full UMAP
- **SGD optimization** - Dense embedding optimization
- **Spectral initialization** - Requires eigendecomposition

**Rationale:** Leave to Python layer or specialized library

---

## Dependency Strategy

### Current: Zero External Dependencies ✅
- All algorithms self-contained
- Only C++20 standard library
- Threading: OpenMP/TBB/BS::thread_pool (configurable)

### Future: Optional Dependencies
```cpp
#ifdef SCL_HAS_EIGEN
  // Use Eigen for specific operations
#else
  // Fallback to self-implementation
#endif
```

**Candidates for optional dependency:**
- Eigen (for advanced linear algebra if user wants)
- Spectra (for sparse eigensolvers)

**Philosophy:** Core remains dependency-free, extensions are optional

---

## Version Milestones

### v0.4.0 - "Graph & Imputation"
**Target:** Q2 2025

**Features:**
- [ ] Louvain clustering
- [ ] Leiden clustering
- [ ] KNN imputation (sparse-preserving)
- [ ] Sparse random projection
- [ ] Permutation testing framework
- [ ] FDR correction

**Deliverables:**
- Complete implementations
- Unit tests
- Benchmarks vs Python libraries
- Documentation

---

### v0.5.0 - "Diffusion & Spatial"
**Target:** Q3 2025

**Features:**
- [ ] Diffusion pseudotime
- [ ] Diffusion-based imputation
- [ ] Random walk with restart
- [ ] Extended spatial statistics (LISA, Gi*)
- [ ] Spatial clustering

**Deliverables:**
- Complete implementations
- Integration with existing modules
- Performance benchmarks
- Use case examples

---

### v0.6.0 - "Graph Learning"
**Target:** Q4 2025

**Features:**
- [ ] GNN message passing primitives
- [ ] Graph attention mechanisms
- [ ] Graph pooling
- [ ] Label propagation
- [ ] Feature smoothing

**Deliverables:**
- Inference-only GNN support
- Integration examples
- Performance analysis

---

## Performance Targets

### Benchmarks Against
1. **Scanpy** (Python) - baseline
2. **Seurat** (R/C++) - strong competitor
3. **Squidpy** (spatial) - spatial analysis
4. **Rapids-singlecell** (GPU) - GPU comparison

### Target Speedups
- **10-50x** vs pure Python (Scanpy)
- **2-5x** vs optimized C++ (Seurat)
- **0.5-2x** vs GPU (Rapids) on CPU

### Memory Efficiency
- Maintain sparsity where possible
- In-place operations preferred
- Block processing for large datasets

---

## Development Principles

### 1. Sparse First
- Always consider sparse representation
- Dense only when necessary
- Preserve sparsity in operations

### 2. SIMD Optimized
- All hot paths use SIMD (via Highway)
- 4-way/8-way unrolling
- Prefetching for cache efficiency

### 3. Thread-Safe
- Parallel by default
- Lock-free where possible
- Work-stealing task scheduler

### 4. Zero-Copy
- Array views, not copies
- In-place operations
- Move semantics

### 5. Well-Documented
- Dual-file documentation (.hpp + .h)
- Algorithm complexity noted
- Preconditions/postconditions clear

---

## Questions for Discussion

### 1. Leiden vs Louvain Priority?
- Louvain is simpler (implement first)
- Leiden is current standard (more demand)
- **Proposal:** Implement Louvain in v0.4.0, Leiden in v0.4.1

### 2. Sparse Random Projection Details?
- Which variant to prioritize?
- **Proposal:** Start with Very Sparse (simplest), add others later

### 3. Diffusion Applications?
- DPT, MAGIC, or both?
- **Proposal:** Generic framework, then specialize

### 4. Optional Dependencies?
- Keep zero-dependency philosophy?
- **Proposal:** Core zero-dep, optional Eigen for advanced features

---

## Contributing

See `CONTRIBUTING.md` for:
- Code style guidelines
- Testing requirements
- Documentation standards
- Review process

## License

MIT License - see `LICENSE` file

