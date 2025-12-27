# SCL-Core Complete Module Inventory (62 Modules)

## Executive Summary

```
═══════════════════════════════════════════════════════════════
 SCL-CORE: Comprehensive Single-Cell Analysis Kernel Library
═══════════════════════════════════════════════════════════════

Total Modules: 62
├─ ✅ Implemented:  23 (37%)
└─ 📝 Planned:      39 (63%)

Strategic Position: SPARSE + NONLINEAR
Core Mission: High-performance biological operators
```

---

## 📊 Module Distribution

### By Implementation Status
```
Implemented (v0.3):        23 modules (37%)
Tier 1 (Core Battlefield):  7 modules (11%)
Tier 2 (Extensions):         3 modules ( 5%)
Tier 3 (Biology-Specific):  14 modules (23%)
Tier 4 (Advanced):           2 modules ( 3%)
Statistics & Tools:         13 modules (21%)
```

### By Category
```
┌─────────────────────────────────────────────────────────┐
│ Category                    │ Implemented │ Planned │ Total │
├─────────────────────────────┼─────────────┼─────────┼───────┤
│ Preprocessing & Norm        │      4      │    0    │   4   │
│ Feature Selection           │      2      │    0    │   2   │
│ Neighbors & Graphs          │      3      │    4    │   7   │
│ Statistical Tests           │      3      │    0    │   3   │
│ Spatial Analysis            │      2      │    4    │   6   │
│ Trajectory & Dynamics       │      0      │    3    │   3   │
│ Cell Communication          │      0      │    2    │   2   │
│ Gene Regulation             │      0      │    2    │   2   │
│ Cell Type & State           │      0      │    3    │   3   │
│ Multi-Omics Integration     │      0      │    2    │   2   │
│ Clonality & Lineage         │      0      │    2    │   2   │
│ Enrichment & Stats          │      0      │    3    │   3   │
│ Quality & Sampling          │      1      │    5    │   6   │
│ Linear Algebra              │      1      │    0    │   1   │
│ Matrix Operations           │      4      │    0    │   4   │
│ Utilities                   │      2      │    5    │   7   │
│ Advanced (Optimization)     │      0      │    2    │   2   │
│ Imputation & Projection     │      0      │    2    │   2   │
│ Clustering                  │      0      │    2    │   2   │
│ Label Propagation           │      0      │    1    │   1   │
│ Information Theory          │      0      │    1    │   1   │
│ Subpopulation Analysis      │      1      │    0    │   1   │
│ Tissue Architecture         │      0      │    1    │   1   │
└─────────────────────────────────────────────────────────┘
```

---

## ✅ Implemented Modules (23)

### 1. Preprocessing & Normalization (4)
- ✅ `normalize.hpp` - Row/column normalization
- ✅ `log1p.hpp` - Log transformation
- ✅ `scale.hpp` - Standardization, z-score
- ✅ `softmax.hpp` - Softmax normalization

### 2. Feature Selection (2)
- ✅ `hvg.hpp` - Highly variable genes
- ✅ `feature.hpp` - Feature filtering

### 3. Nearest Neighbors & Graphs (3)
- ✅ `neighbors.hpp` - KNN computation
- ✅ `bbknn.hpp` - Batch-balanced KNN
- ✅ `gram.hpp` - Gram matrix

### 4. Statistical Tests (3)
- ✅ `ttest.hpp` - T-tests
- ✅ `mwu.hpp` - Mann-Whitney U
- ✅ `mmd.hpp` - Maximum Mean Discrepancy

### 5. Spatial Analysis (2)
- ✅ `spatial.hpp` - Moran's I, Geary's C
- ✅ `correlation.hpp` - Pearson/Spearman

### 6. Linear Algebra (1)
- ✅ `algebra.hpp` - SpMV, SpMM, transpose

### 7. Matrix Operations (4)
- ✅ `slice.hpp` - Matrix slicing
- ✅ `merge.hpp` - Matrix concatenation
- ✅ `reorder.hpp` - Matrix reordering
- ✅ `sparse.hpp` - CSR↔CSC conversion

### 8. Quality Control (1)
- ✅ `qc.hpp` - QC metrics

### 9. Utilities (2)
- ✅ `group.hpp` - Grouping, aggregation
- ✅ `resample.hpp` - Bootstrap, permutation

---

## 📝 Planned Modules (39)

### **TIER 1: Core Battlefield - Graph Algorithms (7 modules)** ⭐⭐⭐⭐⭐

#### Graph Clustering
- 📝 `louvain.hpp` - Louvain clustering
- 📝 `leiden.hpp` - Leiden clustering

#### Graph Infrastructure
- 📝 `components.hpp` - Connected components, BFS/DFS
- 📝 `centrality.hpp` - PageRank, HITS, betweenness

#### Semi-Supervised Learning
- 📝 `propagation.hpp` - Label propagation

#### Dimensionality & Testing
- 📝 `projection.hpp` - Sparse random projection
- 📝 `permutation.hpp` - Permutation testing framework

---

### **TIER 2: Extensions - Advanced Graph Methods (3 modules)** ⭐⭐⭐⭐

#### Diffusion & Dynamics
- 📝 `diffusion.hpp` - Diffusion processes, DPT

#### Graph Neural Networks
- 📝 `gnn.hpp` - Message passing, graph attention

#### Kernel Methods
- 📝 `kernel.hpp` - Sparse kernel methods, KDE

---

### **TIER 3: Biology-Specific (14 modules)** ⭐⭐⭐⭐

#### A. Trajectory & Dynamics (3 modules)
- 📝 `pseudotime.hpp` - DPT, graph pseudotime, branching
- 📝 `velocity.hpp` - RNA velocity, latent time
- 📝 `transition.hpp` - State transitions (CellRank-style)

#### B. Cell Communication (2 modules)
- 📝 `communication.hpp` - Ligand-receptor analysis
- 📝 `niche.hpp` - Cellular neighborhood

#### C. Gene Regulation (2 modules)
- 📝 `grn.hpp` - Gene regulatory networks
- 📝 `coexpression.hpp` - Co-expression modules (WGCNA)

#### D. Cell Type & State (3 modules)
- 📝 `annotation.hpp` - Cell type annotation
- 📝 `state.hpp` - Stemness, differentiation potential
- 📝 `subpopulation.hpp` - Sub-clustering, rare cells

#### E. Multi-Omics (2 modules)
- 📝 `alignment.hpp` - Multi-modal alignment (MNN)
- 📝 `association.hpp` - Gene-peak correlation (RNA+ATAC)

#### F. Spatial Advanced (2 modules)
- 📝 `spatial_pattern.hpp` - SpatialDE-style analysis
- 📝 `tissue.hpp` - Tissue architecture

---

### **TIER 4: Advanced & Statistical (15 modules)** ⭐⭐⭐

#### A. Clonality & Lineage (2 modules)
- 📝 `clonotype.hpp` - TCR/BCR clonal analysis
- 📝 `lineage.hpp` - Lineage tracing

#### B. Enrichment & Statistics (3 modules)
- 📝 `enrichment.hpp` - GSEA, ORA, hypergeometric test
- 📝 `comparison.hpp` - Group comparison, DA
- 📝 `multiple_testing.hpp` - FDR, q-value, local FDR

#### C. Quality & Sampling (5 modules)
- 📝 `sampling.hpp` - Geometric sketching, density-preserving
- 📝 `metrics.hpp` - Silhouette, ARI, NMI, LISI
- 📝 `outlier.hpp` - LOF, ambient RNA, empty drops
- 📝 `doublet.hpp` - Doublet detection
- 📝 `hotspot.hpp` - Local spatial statistics (LISA, Gi*)

#### D. Advanced Methods (3 modules)
- 📝 `impute.hpp` - KNN/diffusion imputation
- 📝 `entropy.hpp` - Mutual information, KL divergence
- 📝 `sparse_opt.hpp` - Lasso, elastic net

#### E. Scoring (2 modules)
- 📝 `markers.hpp` - Marker gene selection
- 📝 `scoring.hpp` - Gene set scoring (AUCell-style)

---

## 🗺️ Complete Dependency Graph

```
LEVEL 0: Foundation (No Dependencies)
├─ Core: sparse, algebra, slice, merge, reorder
├─ Preprocessing: normalize, log1p, scale, softmax
├─ Feature: hvg, feature, qc
├─ Utilities: resample, group
├─ Standalone: projection, entropy, sparse_opt
└─ Testing: permutation, multiple_testing

LEVEL 1: Primary Kernels (Depends on Level 0)
├─ neighbors ────────────┐
├─ spatial              │
├─ correlation          │
├─ ttest, mwu, mmd      │
├─ scoring (basic)      │
├─ components           │
├─ sampling             │
└─ outlier              │
                        │
LEVEL 2: Advanced Kernels (Depends on neighbors/spatial/stats)
├─ bbknn ◄──────────────┤
├─ gram ◄───────────────┤
├─ louvain ◄────────────┤
├─ centrality ◄─────────┤
├─ propagation ◄────────┤
├─ diffusion ◄──────────┤
├─ kernel ◄─────────────┤
├─ doublet ◄────────────┤
├─ markers ◄────────────┤ (also needs ttest/mwu)
├─ annotation ◄─────────┤
├─ alignment ◄──────────┤
├─ metrics ◄────────────┤
├─ hotspot ◄────────────┘ (also needs spatial)
│
LEVEL 3: Specialized Modules
├─ leiden ◄─── louvain + components
├─ pseudotime ◄─── diffusion + components
├─ velocity ◄─── neighbors
├─ transition ◄─── velocity/diffusion
├─ impute ◄─── neighbors + diffusion
├─ gnn ◄─── neighbors + softmax
├─ communication ◄─── permutation + spatial
├─ niche ◄─── spatial + neighbors
├─ grn ◄─── correlation + entropy
├─ coexpression ◄─── correlation
├─ state ◄─── scoring
├─ subpopulation ◄─── leiden + resample
├─ association ◄─── correlation
├─ spatial_pattern ◄─── spatial + hotspot
├─ tissue ◄─── spatial_pattern
├─ clonotype ◄─── entropy
├─ lineage ◄─── clonotype
├─ enrichment ◄─── permutation + multiple_testing
└─ comparison ◄─── ttest/mwu + multiple_testing
```

---

## 📅 Implementation Roadmap

### **Phase 1: Foundation Complete** ✅ (v0.3.0 - Done)
**23 modules implemented**

### **Phase 2: Graph Infrastructure** 🔨 (v0.4.0 - Q2 2025)
**Target: 7 modules**

**Month 1-2:** Core Graph
- `louvain` (2 weeks)
- `components` (1 week)
- `projection` (3 days)
- `permutation` (4 days)

**Month 3:** Advanced Graph
- `propagation` (1 week)
- `centrality` (1.5 weeks)
- `leiden` (2 weeks)

### **Phase 3: Trajectory & Communication** 🎯 (v0.5.0 - Q3 2025)
**Target: 10 modules**

**Month 4:** Trajectory
- `diffusion` (2 weeks)
- `pseudotime` (1.5 weeks)

**Month 5:** Dynamics & Communication
- `velocity` (2 weeks)
- `transition` (1 week)
- `communication` (1 week)

**Month 6:** Biology Tools
- `markers` (1 week)
- `scoring` (1 week)
- `annotation` (1 week)
- `state` (1 week)
- `niche` (1 week)

### **Phase 4: Multi-Omics & Spatial** 🧬 (v0.6.0 - Q4 2025)
**Target: 10 modules**

**Month 7:** Spatial & Tissue
- `hotspot` (1.5 weeks)
- `spatial_pattern` (2 weeks)
- `tissue` (1 week)

**Month 8:** Multi-Omics
- `alignment` (2 weeks)
- `association` (1 week)
- `impute` (1 week)

**Month 9:** Regulation & Co-expression
- `grn` (2 weeks)
- `coexpression` (1.5 weeks)
- `gnn` (2 weeks)

### **Phase 5: Statistics & Quality** 📊 (v0.7.0 - Q1 2026)
**Target: 8 modules**

**Month 10:** Statistics
- `enrichment` (1.5 weeks)
- `comparison` (1 week)
- `multiple_testing` (1 week)
- `entropy` (1.5 weeks)

**Month 11:** Quality & Sampling
- `sampling` (1 week)
- `metrics` (1 week)
- `outlier` (1 week)
- `doublet` (1 week)

### **Phase 6: Advanced Methods** 🚀 (v0.8.0 - Q2 2026)
**Target: 4 modules**

**Month 12:** Advanced
- `sparse_opt` (3 weeks)
- `kernel` (1 week)
- `subpopulation` (1 week)
- `clonotype` (1 week)
- `lineage` (1 week)

---

## 🎯 Priority Matrix

### Critical Path (Must-Have First)
```
1. louvain → leiden (clustering foundation)
2. diffusion → pseudotime (trajectory analysis)
3. neighbors → propagation → annotation (cell typing)
4. ttest/mwu → markers (differential analysis)
5. permutation → multiple_testing (statistics)
```

### High-Impact Modules (Maximum User Value)
```
Priority A (Do Next):
- leiden, pseudotime, markers, annotation, multiple_testing

Priority B (Important):
- velocity, transition, communication, scoring, enrichment

Priority C (Nice to Have):
- spatial_pattern, grn, impute, alignment, metrics
```

### Complexity vs. Impact
```
High Impact + Low Complexity:
✓ projection, permutation, scoring, metrics, multiple_testing

High Impact + High Complexity:
⚠ leiden, diffusion, pseudotime, velocity, impute, sparse_opt

Low Impact + Low Complexity:
○ sampling, outlier

Low Impact + High Complexity:
✗ Avoid for now
```

---

## 💡 Implementation Guidelines

### Module Size Estimates
```
Small (< 500 lines):     projection, permutation, metrics, multiple_testing
Medium (500-800 lines):  scoring, markers, annotation, state, sampling
Large (800-1200 lines):  louvain, diffusion, pseudotime, centrality, grn
Very Large (> 1200):     leiden, velocity, impute, sparse_opt, gnn
```

### Testing Requirements
```
Unit Tests:         All modules
Integration Tests:  Modules with dependencies (Level 2+)
Performance Tests:  Graph algorithms, diffusion, impute
Benchmark Tests:    Compare with Scanpy, Seurat, Signac
```

### Documentation Requirements
Each module must have:
- [ ] .h API documentation file
- [ ] Inline minimal comments in .hpp
- [ ] Usage examples in docs/
- [ ] Performance benchmarks
- [ ] Complexity analysis

---

## 🤔 Strategic Questions

1. **Should we prioritize breadth or depth?**
   - Breadth: Implement all 39 planned modules (coverage)
   - Depth: Perfect 10-15 core modules (quality)
   
2. **When to add GPU acceleration?**
   - After v0.6.0 (Phase 4 complete)
   - Focus on: neighbors, leiden, diffusion, impute
   
3. **Python bindings priority?**
   - Parallel with C++ development
   - Start after Phase 2 (v0.4.0)
   
4. **Should tissue/clonotype modules be separate library?**
   - Pro: Focused scope
   - Con: Fragmented ecosystem

---

## 📚 References by Module

### Graph Algorithms
- Louvain: Blondel et al. (2008)
- Leiden: Traag et al. (2019)
- Label Propagation: Zhou et al. (2004)

### Trajectory
- DPT: Haghverdi et al. (2016)
- RNA Velocity: La Manno et al. (2018), Bergen et al. (2020)
- CellRank: Lange et al. (2022)

### Communication
- CellChat: Jin et al. (2021)
- CellPhoneDB: Efremova et al. (2020)

### Spatial
- SpatialDE: Svensson et al. (2018)
- LISA: Anselin (1995)
- Getis-Ord: Getis & Ord (1992)

### Multi-Omics
- MNN: Haghverdi et al. (2018)
- Seurat Integration: Stuart et al. (2019)

### Statistics
- GSEA: Subramanian et al. (2005)
- FDR: Benjamini & Hochberg (1995)
- q-value: Storey & Tibshirani (2003)

---

## 🎉 Conclusion

With **62 comprehensive modules**, SCL-Core will become:

✅ **Most complete** sparse+nonlinear operator library
✅ **High-performance** C++ kernels with zero-overhead
✅ **Biology-focused** with state-of-the-art algorithms
✅ **Production-ready** with extensive testing

**Timeline:** 18-24 months for full implementation
**Resources:** 2-3 core developers + community contributions
**Impact:** Foundation for next-generation single-cell tools

Let's build the future of biological data analysis! 🚀

