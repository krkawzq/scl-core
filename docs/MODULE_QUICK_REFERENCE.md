# SCL-Core Module Quick Reference

## All 62 Modules At a Glance

### ✅ Implemented (23)

| # | Module | Category | Key Functions |
|---|--------|----------|---------------|
| 1 | `normalize` | Preprocessing | row_norms, normalize_rows_inplace |
| 2 | `log1p` | Preprocessing | log1p_sparse, log1p_inplace |
| 3 | `scale` | Preprocessing | standardize, center_scale |
| 4 | `softmax` | Preprocessing | softmax_rows, softmax_columns |
| 5 | `hvg` | Feature | highly_variable_genes, seurat_v3 |
| 6 | `feature` | Feature | count_features, filter_features |
| 7 | `neighbors` | Graph | knn_bruteforce, knn_cosine |
| 8 | `bbknn` | Graph | batch_balanced_knn |
| 9 | `gram` | Graph | gram_matrix, sparse_dot_product |
| 10 | `ttest` | Statistics | welch_ttest, students_ttest |
| 11 | `mwu` | Statistics | mann_whitney_u, rank_sum |
| 12 | `mmd` | Statistics | maximum_mean_discrepancy |
| 13 | `spatial` | Spatial | morans_i, gearys_c, spatial_lag |
| 14 | `correlation` | Spatial | pearson_correlation, spearman_rank |
| 15 | `algebra` | Linear Algebra | spmv, spmm, transpose |
| 16 | `slice` | Matrix Ops | slice_rows, slice_columns |
| 17 | `merge` | Matrix Ops | concatenate_rows, concatenate_columns |
| 18 | `reorder` | Matrix Ops | reorder_rows, reorder_columns |
| 19 | `sparse` | Matrix Ops | csr_to_csc, csc_to_csr |
| 20 | `qc` | Quality | compute_qc_metrics, filter_cells |
| 21 | `group` | Utilities | group_by, aggregate |
| 22 | `resample` | Utilities | bootstrap, permutation, sample |
| 23 | Completed in v0.3.0 | | |

### 📝 Tier 1: Core Battlefield (7)

| # | Module | Priority | Key Functions |
|---|--------|----------|---------------|
| 24 | `louvain` | ⭐⭐⭐⭐⭐ | cluster, modularity_optimization |
| 25 | `leiden` | ⭐⭐⭐⭐⭐ | cluster, refine_partition |
| 26 | `components` | ⭐⭐⭐⭐⭐ | connected_components, bfs, dfs |
| 27 | `propagation` | ⭐⭐⭐⭐⭐ | label_propagation, label_spreading |
| 28 | `centrality` | ⭐⭐⭐⭐ | pagerank, hits, betweenness |
| 29 | `projection` | ⭐⭐⭐⭐ | sparse_random_projection |
| 30 | `permutation` | ⭐⭐⭐⭐ | permutation_test, permute_labels |

### 📝 Tier 2: Extensions (3)

| # | Module | Priority | Key Functions |
|---|--------|----------|---------------|
| 31 | `diffusion` | ⭐⭐⭐ | diffusion_map, diffusion_kernel |
| 32 | `gnn` | ⭐⭐⭐ | message_passing, graph_attention |
| 33 | `kernel` | ⭐⭐⭐ | rbf_sparse, kernel_density |

### 📝 Tier 3: Biology-Specific (14)

#### Trajectory & Dynamics (3)
| # | Module | Priority | Key Functions |
|---|--------|----------|---------------|
| 34 | `pseudotime` | ⭐⭐⭐⭐⭐ | diffusion_pseudotime, detect_branches |
| 35 | `velocity` | ⭐⭐⭐⭐ | compute_velocity, velocity_graph |
| 36 | `transition` | ⭐⭐⭐⭐ | transition_matrix, absorption_probability |

#### Communication (2)
| # | Module | Priority | Key Functions |
|---|--------|----------|---------------|
| 37 | `communication` | ⭐⭐⭐⭐ | lr_score, communication_probability |
| 38 | `niche` | ⭐⭐⭐⭐ | neighborhood_composition, niche_clustering |

#### Regulation (2)
| # | Module | Priority | Key Functions |
|---|--------|----------|---------------|
| 39 | `grn` | ⭐⭐⭐⭐ | correlation_network, tf_target_score |
| 40 | `coexpression` | ⭐⭐⭐ | wgcna_adjacency, detect_modules |

#### Cell Type & State (3)
| # | Module | Priority | Key Functions |
|---|--------|----------|---------------|
| 41 | `annotation` | ⭐⭐⭐⭐⭐ | reference_mapping, correlation_assignment |
| 42 | `state` | ⭐⭐⭐⭐ | stemness_score, differentiation_potential |
| 43 | `subpopulation` | ⭐⭐⭐ | subclustering, cluster_stability |

#### Multi-Omics (2)
| # | Module | Priority | Key Functions |
|---|--------|----------|---------------|
| 44 | `alignment` | ⭐⭐⭐⭐ | mnn_pairs, find_anchors |
| 45 | `association` | ⭐⭐⭐ | gene_peak_correlation, cis_regulatory |

#### Spatial Advanced (2)
| # | Module | Priority | Key Functions |
|---|--------|----------|---------------|
| 46 | `spatial_pattern` | ⭐⭐⭐⭐ | spatial_variability, spatial_gradient |
| 47 | `tissue` | ⭐⭐⭐ | tissue_architecture, layer_assignment |

### 📝 Tier 4: Statistics & Advanced (15)

#### Clonality & Lineage (2)
| # | Module | Priority | Key Functions |
|---|--------|----------|---------------|
| 48 | `clonotype` | ⭐⭐⭐ | clone_size_distribution, clonal_diversity |
| 49 | `lineage` | ⭐⭐⭐ | lineage_tree, fate_bias |

#### Enrichment & Statistics (3)
| # | Module | Priority | Key Functions |
|---|--------|----------|---------------|
| 50 | `enrichment` | ⭐⭐⭐⭐⭐ | hypergeometric_test, gsea_score |
| 51 | `comparison` | ⭐⭐⭐⭐ | composition_analysis, differential_abundance |
| 52 | `multiple_testing` | ⭐⭐⭐⭐⭐ | benjamini_hochberg, storey_qvalue |

#### Quality & Sampling (5)
| # | Module | Priority | Key Functions |
|---|--------|----------|---------------|
| 53 | `markers` | ⭐⭐⭐⭐ | rank_genes_groups, tau_specificity |
| 54 | `scoring` | ⭐⭐⭐⭐ | gene_set_score, auc_score, module_score |
| 55 | `hotspot` | ⭐⭐⭐ | local_morans_i, getis_ord_g_star |
| 56 | `doublet` | ⭐⭐⭐ | simulate_doublets, doublet_score |
| 57 | `outlier` | ⭐⭐⭐ | local_outlier_factor, empty_drops |

#### Advanced Methods (3)
| # | Module | Priority | Key Functions |
|---|--------|----------|---------------|
| 58 | `impute` | ⭐⭐⭐⭐⭐ | knn_impute, diffusion_impute |
| 59 | `entropy` | ⭐⭐ | shannon_entropy, mutual_information |
| 60 | `sparse_opt` | ⭐⭐ | lasso_coordinate_descent, fista |

#### Utilities (2)
| # | Module | Priority | Key Functions |
|---|--------|----------|---------------|
| 61 | `sampling` | ⭐⭐⭐ | geometric_sketching, density_preserving |
| 62 | `metrics` | ⭐⭐⭐⭐ | silhouette_score, adjusted_rand_index |

---

## Quick Lookup by Use Case

### I want to... → Use these modules

**Cluster cells**
→ `neighbors` → `louvain` or `leiden`

**Find marker genes**
→ `ttest` or `mwu` → `markers`

**Analyze trajectory**
→ `neighbors` → `diffusion` → `pseudotime` → `velocity`

**Cell type annotation**
→ `neighbors` → `propagation` or `annotation`

**Cell-cell communication**
→ `spatial` → `communication` or `niche`

**Spatial patterns**
→ `spatial` → `hotspot` → `spatial_pattern`

**Gene regulatory networks**
→ `correlation` → `grn` or `coexpression`

**Batch correction**
→ `neighbors` → `alignment`

**Multi-omics integration**
→ `alignment` + `association`

**Quality control**
→ `qc` → `doublet` → `outlier`

**Gene set enrichment**
→ `enrichment` + `multiple_testing`

**Downsample large dataset**
→ `sampling`

**Evaluate clustering**
→ `metrics`

---

## Module Dependencies (Simplified)

```
Level 0 (Independent):
- All preprocessing, feature, QC
- projection, entropy, sparse_opt
- permutation, multiple_testing

Level 1 (Needs Level 0):
- neighbors ← (all graph methods depend on this!)
- spatial, correlation
- ttest, mwu, mmd
- scoring, sampling, outlier

Level 2 (Needs neighbors):
- bbknn, gram, louvain, leiden
- propagation, centrality, diffusion
- kernel, doublet, markers, annotation
- alignment, metrics, hotspot

Level 3 (Needs Level 2):
- pseudotime (diffusion)
- velocity (neighbors)
- transition (velocity/diffusion)
- impute (neighbors + diffusion)
- gnn (neighbors)
- communication, niche (spatial + neighbors)
- grn, coexpression (correlation)
- state, subpopulation (scoring/leiden)
- spatial_pattern, tissue (spatial)
- clonotype, lineage (entropy)
- enrichment, comparison (stats + multiple_testing)
```

---

## Implementation Status Tracker

### Phase 1: Foundation ✅ (v0.3.0)
- [x] 23 modules implemented

### Phase 2: Graph Infrastructure 🔨 (v0.4.0 - Target Q2 2025)
- [ ] louvain
- [ ] leiden  
- [ ] components
- [ ] centrality
- [ ] propagation
- [ ] projection
- [ ] permutation

### Phase 3: Trajectory & Communication 🎯 (v0.5.0 - Target Q3 2025)
- [ ] diffusion
- [ ] pseudotime
- [ ] velocity
- [ ] transition
- [ ] communication
- [ ] niche
- [ ] markers
- [ ] scoring
- [ ] annotation
- [ ] state

### Phase 4: Multi-Omics & Spatial 🧬 (v0.6.0 - Target Q4 2025)
- [ ] hotspot
- [ ] spatial_pattern
- [ ] tissue
- [ ] alignment
- [ ] association
- [ ] impute
- [ ] grn
- [ ] coexpression
- [ ] gnn

### Phase 5: Statistics & Quality 📊 (v0.7.0 - Target Q1 2026)
- [ ] enrichment
- [ ] comparison
- [ ] multiple_testing
- [ ] entropy
- [ ] sampling
- [ ] metrics
- [ ] outlier
- [ ] doublet

### Phase 6: Advanced Methods 🚀 (v0.8.0 - Target Q2 2026)
- [ ] sparse_opt
- [ ] kernel
- [ ] subpopulation
- [ ] clonotype
- [ ] lineage

---

## File Locations

All modules: `/home/wzq/Code/Projects/scl-core/scl/kernel/`

```
scl/kernel/
├─ normalize.hpp        ✅
├─ log1p.hpp           ✅
├─ scale.hpp           ✅
├─ softmax.hpp         ✅
├─ hvg.hpp             ✅
├─ feature.hpp         ✅
├─ neighbors.hpp       ✅
├─ bbknn.hpp           ✅
├─ gram.hpp            ✅
├─ ttest.hpp           ✅
├─ mwu.hpp             ✅
├─ mmd.hpp             ✅
├─ spatial.hpp         ✅
├─ correlation.hpp     ✅
├─ algebra.hpp         ✅
├─ slice.hpp           ✅
├─ merge.hpp           ✅
├─ reorder.hpp         ✅
├─ sparse.hpp          ✅
├─ qc.hpp              ✅
├─ group.hpp           ✅
├─ resample.hpp        ✅
├─ louvain.hpp         📝 (framework created)
├─ leiden.hpp          📝
├─ components.hpp      📝
├─ propagation.hpp     📝
├─ centrality.hpp      📝
├─ projection.hpp      📝
├─ permutation.hpp     📝
├─ diffusion.hpp       📝
├─ gnn.hpp             📝
├─ kernel.hpp          📝
├─ pseudotime.hpp      📝
├─ velocity.hpp        📝
├─ transition.hpp      📝
├─ communication.hpp   📝
├─ niche.hpp           📝
├─ grn.hpp             📝
├─ coexpression.hpp    📝
├─ annotation.hpp      📝
├─ state.hpp           📝
├─ subpopulation.hpp   📝
├─ alignment.hpp       📝
├─ association.hpp     📝
├─ spatial_pattern.hpp 📝
├─ tissue.hpp          📝
├─ clonotype.hpp       📝
├─ lineage.hpp         📝
├─ enrichment.hpp      📝
├─ comparison.hpp      📝
├─ multiple_testing.hpp 📝
├─ markers.hpp         📝
├─ scoring.hpp         📝
├─ hotspot.hpp         📝
├─ doublet.hpp         📝
├─ outlier.hpp         📝
├─ impute.hpp          📝
├─ entropy.hpp         📝
├─ sparse_opt.hpp      📝
├─ sampling.hpp        📝
└─ metrics.hpp         📝
```

**Total: 62 modules (23 ✅ implemented, 39 📝 frameworks created)**

---

## Next Steps

1. ✅ **Frameworks Complete** - All 62 module structures created
2. 🔨 **Start Implementation** - Begin with `louvain` (simplest clustering)
3. 📝 **Write Tests** - TDD approach for each module
4. 📊 **Benchmark** - Compare with Scanpy, Seurat, Signac
5. 📚 **Document** - Update .h files with API docs
6. 🚀 **Release** - v0.4.0 with first 7 new modules

**Let's build it! 🎉**

