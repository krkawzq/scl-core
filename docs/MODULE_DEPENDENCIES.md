# SCL-Core Module Dependencies

## Visual Dependency Graph

```
                    ┌──────────────────────────────────────────────┐
                    │         LEVEL 0: Foundation                  │
                    │         (No Dependencies)                    │
                    └──────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────┐           ┌────────────────┐           ┌────────────────┐
│  Core Types   │           │  Preprocessing │           │   Utilities    │
├───────────────┤           ├────────────────┤           ├────────────────┤
│ • sparse      │           │ • normalize    │           │ • resample     │
│ • algebra     │           │ • log1p        │           │ • group        │
│ • slice       │           │ • scale        │           │ • components   │
│ • merge       │           │ • softmax      │           │ • projection   │
│ • reorder     │           │ • hvg          │           │ • entropy      │
│               │           │ • feature      │           │ • sparse_opt   │
│               │           │ • qc           │           │ • scoring      │
└───────────────┘           └────────────────┘           └────────────────┘
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │         LEVEL 1: Core Kernels               │
                    │      (Depends on Foundation)                │
                    └─────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────┐           ┌────────────────┐           ┌────────────────┐
│    Graphs     │           │   Statistics   │           │    Spatial     │
├───────────────┤           ├────────────────┤           ├────────────────┤
│ • neighbors ◄─┼───────────┤ • ttest        │           │ • spatial      │
│ • gram        │           │ • mwu          │           │ • correlation  │
│               │           │ • mmd          │           │                │
└───────┬───────┘           └────────────────┘           └────────────────┘
        │                                                         │
        │                                                         │
        │       ┌─────────────────────────────────────────────────┘
        │       │
        │       │   ┌──────────────────────────────────────────────────┐
        │       │   │         LEVEL 2: Advanced Kernels                │
        │       │   │    (Depends on Neighbors/Spatial/Stats)         │
        │       │   └──────────────────────────────────────────────────┘
        │       │                        │
        │       │   ┌────────────────────┼────────────────────┐
        │       │   │                    │                    │
        ▼       ▼   ▼                    ▼                    ▼
    ┌───────────────┐           ┌────────────────┐   ┌────────────────┐
    │ Graph Algos   │           │   Extensions   │   │   Biology      │
    ├───────────────┤           ├────────────────┤   ├────────────────┤
    │ • louvain     │           │ • bbknn        │   │ • markers      │
    │ • leiden      │           │ • kernel       │   │ • doublet      │
    │ • propagation │           │ • diffusion    │   │ • hotspot      │
    │ • centrality  │           │ • gnn          │   │                │
    └───────┬───────┘           └────────────────┘   └────────────────┘
            │                            │
            │                            ▼
            │                   ┌────────────────┐
            │                   │    Impute      │
            └──────────────────►│  (Level 3)     │
                                │ Dependencies:  │
                                │ • neighbors    │
                                │ • diffusion    │
                                └────────────────┘


                ┌─────────────────────────────────────────┐
                │  KEY STRATEGIC AREAS                    │
                ├─────────────────────────────────────────┤
                │  ✅ Sparse + Nonlinear (Core Focus)     │
                │  ⚠️  Sparse + Linear (Selective)        │
                │  ❌ Dense + Linear (Avoid)              │
                └─────────────────────────────────────────┘
```

---

## Dependency Matrix

|  Module        | L0 Core | L0 Prep | neighbors | spatial | stats | graph |
|----------------|---------|---------|-----------|---------|-------|-------|
| **neighbors**  | ✓       | ✓       | -         | -       | -     | -     |
| **louvain**    | ✓       | ✓       | ✓         | -       | -     | -     |
| **leiden**     | ✓       | ✓       | ✓         | -       | -     | ✓     |
| **propagation**| ✓       | -       | ✓         | -       | -     | -     |
| **diffusion**  | ✓       | -       | ✓         | -       | -     | -     |
| **centrality** | ✓       | -       | -         | -       | -     | ✓     |
| **gnn**        | ✓       | ✓       | ✓         | -       | -     | -     |
| **impute**     | ✓       | -       | ✓         | -       | -     | ✓     |
| **markers**    | ✓       | -       | -         | -       | ✓     | -     |
| **scoring**    | ✓       | -       | -         | -       | -     | -     |
| **hotspot**    | ✓       | -       | -         | ✓       | -     | -     |
| **kernel**     | ✓       | -       | ✓         | -       | -     | -     |
| **doublet**    | ✓       | -       | ✓         | -       | -     | -     |

Legend:
- ✓ = Direct dependency
- - = No dependency

---

## Implementation Order by Dependencies

### Wave 1 (No external dependencies - parallel implementation)
```
├─ projection      (very sparse random projection)
├─ permutation     (extends resample)
├─ components      (graph infrastructure)
└─ scoring         (basic gene set scoring)
```
**Rationale:** Can be implemented in parallel, no inter-dependencies

### Wave 2 (Depends on neighbors only)
```
├─ louvain         (graph clustering)
├─ propagation     (label propagation)
├─ kernel          (sparse kernel methods)
├─ doublet         (doublet detection)
└─ diffusion       (diffusion processes)
```
**Rationale:** All depend on neighbors, which is already implemented

### Wave 3 (Depends on Wave 2)
```
├─ leiden          (depends on louvain + components)
├─ centrality      (depends on components)
├─ impute          (depends on neighbors + diffusion)
└─ gnn             (depends on neighbors + softmax)
```
**Rationale:** Build on Wave 2 results

### Wave 4 (Extends existing modules)
```
├─ markers         (combines ttest/mwu)
├─ hotspot         (extends spatial)
├─ entropy         (standalone)
└─ sparse_opt      (standalone)
```
**Rationale:** Can be added after core graph algorithms

---

## Critical Path Analysis

### Shortest Path to Clustering (Priority A)
```
neighbors (✅) → louvain (📝) → leiden (📝)
                          ↓
                    components (📝)
```
**Timeline:** 2-3 months

### Shortest Path to Imputation (Priority B)
```
neighbors (✅) → diffusion (📝) → impute (📝)
```
**Timeline:** 2-3 months

### Shortest Path to Label Transfer (Priority C)
```
neighbors (✅) → propagation (📝)
```
**Timeline:** 1 month

---

## Module Complexity Estimates

| Module | Lines of Code (est) | Dev Time | Test Time | Complexity |
|--------|---------------------|----------|-----------|------------|
| **projection** | 300-500 | 3 days | 2 days | Low |
| **permutation** | 400-600 | 4 days | 3 days | Medium |
| **components** | 500-700 | 1 week | 1 week | Medium |
| **propagation** | 600-800 | 1 week | 1 week | Medium |
| **louvain** | 800-1200 | 2 weeks | 2 weeks | High |
| **diffusion** | 700-1000 | 2 weeks | 1 week | High |
| **leiden** | 1000-1500 | 3 weeks | 2 weeks | Very High |
| **centrality** | 600-900 | 1.5 weeks | 1 week | Medium |
| **impute** | 500-800 | 1 week | 1 week | Medium |
| **kernel** | 400-600 | 1 week | 4 days | Medium |
| **markers** | 600-800 | 1 week | 1 week | Medium |
| **scoring** | 500-700 | 1 week | 4 days | Low-Medium |
| **hotspot** | 700-900 | 1.5 weeks | 1 week | Medium-High |
| **doublet** | 500-700 | 1 week | 1 week | Medium |
| **gnn** | 800-1200 | 2 weeks | 2 weeks | High |
| **entropy** | 600-800 | 1.5 weeks | 1 week | Medium |
| **sparse_opt** | 1000-1500 | 3 weeks | 2 weeks | Very High |

**Total Estimated Time:** ~25-30 weeks (6-7 months) for full implementation

---

## Parallelization Opportunities

### Team of 3 Developers

**Developer 1: Graph Algorithms**
```
Week 1-2:   components
Week 3-6:   louvain
Week 7-9:   propagation
Week 10-14: leiden
Week 15-17: centrality
```

**Developer 2: Diffusion & Imputation**
```
Week 1-2:   projection
Week 3-6:   diffusion
Week 7-9:   kernel
Week 10-12: impute
Week 13-15: gnn
```

**Developer 3: Biology Tools**
```
Week 1-2:   permutation
Week 3-4:   scoring
Week 5-7:   markers
Week 8-10:  doublet
Week 11-13: hotspot
Week 14-17: entropy
```

**Parallel Timeline:** ~4 months for 17 modules

---

## Risk Assessment

### High Risk Modules (Complex + Critical)
- **leiden** - Complex algorithm, critical for clustering
- **sparse_opt** - Optimization theory, numerical stability
- **impute** - Must preserve sparsity, many edge cases

**Mitigation:** Start early, extensive testing, iterative refinement

### Medium Risk Modules
- **louvain** - Well-documented but complex
- **diffusion** - Numerical stability concerns
- **gnn** - Relatively new territory

**Mitigation:** Reference implementations, benchmarks

### Low Risk Modules
- **projection** - Simple linear algebra
- **permutation** - Extends existing resample
- **scoring** - Straightforward aggregations

**Mitigation:** Standard testing

---

## Integration Testing Strategy

### Test Suites by Dependency Level

**Level 0 Tests:** Unit tests only
```python
test_projection()
test_components_basic()
test_permutation()
```

**Level 1 Tests:** Integration with neighbors
```python
test_louvain_on_knn()
test_propagation_on_graph()
test_diffusion_on_neighbors()
```

**Level 2 Tests:** Full pipelines
```python
test_leiden_clustering_pipeline()
test_imputation_pipeline()
test_marker_selection_pipeline()
```

---

## Performance Targets

| Module | Target (10K cells, 2K genes) | Stretch Goal |
|--------|------------------------------|--------------|
| louvain | < 2 seconds | < 1 second |
| leiden | < 5 seconds | < 3 seconds |
| propagation | < 1 second | < 500ms |
| diffusion | < 3 seconds | < 2 seconds |
| impute | < 5 seconds | < 3 seconds |
| markers | < 2 seconds | < 1 second |
| scoring | < 500ms | < 200ms |
| doublet | < 3 seconds | < 2 seconds |

---

## Next Steps

1. ✅ **Frameworks created** (just completed!)
2. 🔨 **Start with `louvain`** - Foundational graph algorithm
3. 🔨 **Parallel: `projection`** - Quick win
4. ✅ **Write comprehensive tests** - TDD approach
5. 📊 **Benchmark against reference** - Scanpy, igraph
6. 📝 **Document as you go** - Update .h files
7. 🎯 **Iterate based on feedback** - Adjust priorities

Ready to start implementation! 🚀

