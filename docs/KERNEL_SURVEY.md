# SCL-Core Kernel Operators Survey

## Current Operators (23 Modules)

### 1. **Preprocessing & Normalization**

#### normalize.hpp
- `compute_row_sums` - Compute row/column sums
- `scale_primary` - Scale rows/columns by factors
- `primary_sums_masked` - Masked sum computation
- `detect_highly_expressed` - Detect highly expressed genes

#### log1p.hpp
- `apply_log1p` - Apply log(1+x) transformation
- `log1p_inplace` - In-place log1p transform

#### scale.hpp
- `standardize_rows` - Standardize to zero mean, unit variance
- `standardize_cols` - Column-wise standardization
- `clip_values` - Clip values to range

#### softmax.hpp
- `softmax_rows` - Row-wise softmax normalization
- `softmax_cols` - Column-wise softmax
- `log_softmax` - Numerically stable log-softmax

### 2. **Feature Selection**

#### hvg.hpp (Highly Variable Genes)
- `compute_dispersion` - Gene dispersion calculation
- `select_hvg_seurat` - Seurat method for HVG
- `select_hvg_cell_ranger` - Cell Ranger method

#### feature.hpp
- `feature_counts` - Count features per observation
- `filter_features` - Filter by expression thresholds

### 3. **Nearest Neighbors & Graphs**

#### neighbors.hpp
- `compute_knn` - K-nearest neighbors (exact)
- `cosine_knn` - Cosine similarity based KNN
- `euclidean_knn` - Euclidean distance KNN
- `dot_product_similarity` - Sparse dot product for similarity

#### bbknn.hpp (Batch-Balanced KNN)
- `bbknn_compute` - Batch-aware nearest neighbor search
- `batch_balanced_graph` - Construct batch-balanced graph

#### gram.hpp
- `gram_matrix` - Compute Gram matrix (X @ X.T)
- `sparse_dot_adaptive` - Adaptive sparse dot product
- `pairwise_distances` - Pairwise distance computation

### 4. **Statistical Tests**

#### ttest.hpp (T-test)
- `compute_group_stats` - Group-wise mean/variance
- `welch_ttest` - Welch's t-test (unequal variance)
- `students_ttest` - Student's t-test
- `t_to_pvalue` - Convert t-statistic to p-value

#### mwu.hpp (Mann-Whitney U)
- `mann_whitney_u` - Non-parametric test
- `rank_sum_sparse` - Rank-sum for sparse data
- `wilcoxon_test` - Wilcoxon signed-rank test

#### mmd.hpp (Maximum Mean Discrepancy)
- `compute_mmd` - MMD with RBF kernel
- `mmd_permutation_test` - Permutation-based significance
- `kernel_two_sample` - Two-sample kernel test

### 5. **Spatial Analysis**

#### spatial.hpp
- `morans_i` - Moran's I spatial autocorrelation
- `gearys_c` - Geary's C statistic
- `spatial_lag` - Spatial lag operator
- `weighted_neighbor_sum` - Weighted neighborhood aggregation

#### correlation.hpp
- `pearson_correlation` - Pearson correlation matrix
- `spearman_correlation` - Spearman rank correlation
- `sparse_correlation` - Correlation for sparse matrices

### 6. **Linear Algebra**

#### algebra.hpp
- `spmv` - Sparse matrix-vector multiplication (y = α·A·x + β·y)
- `spmm` - Sparse matrix-matrix multiplication
- `elementwise_multiply` - Element-wise product
- `transpose` - Matrix transpose

### 7. **Matrix Operations**

#### slice.hpp
- `slice_rows` - Extract row submatrix
- `slice_cols` - Extract column submatrix
- `slice_submatrix` - Extract arbitrary submatrix
- `gather_rows` - Gather rows by indices

#### merge.hpp
- `merge_horizontal` - Concatenate matrices horizontally
- `merge_vertical` - Concatenate matrices vertically
- `union_merge` - Union of sparse matrices

#### reorder.hpp
- `reorder_rows` - Reorder rows by permutation
- `reorder_cols` - Reorder columns
- `permute_inplace` - In-place permutation

#### sparse.hpp (Matrix format operations)
- `csr_to_csc` - CSR to CSC conversion
- `csc_to_csr` - CSC to CSR conversion
- `densify` - Convert sparse to dense
- `sparsify` - Convert dense to sparse

### 8. **Quality Control**

#### qc.hpp
- `compute_qc_metrics` - Cell QC metrics (n_genes, n_counts)
- `filter_cells` - Filter cells by QC thresholds
- `detect_outliers` - MAD-based outlier detection
- `mitochondrial_percentage` - Mito percentage calculation

### 9. **Grouping & Aggregation**

#### group.hpp
- `group_by` - Group observations by labels
- `aggregate_sum` - Sum within groups
- `aggregate_mean` - Mean within groups
- `group_variance` - Variance within groups

### 10. **Resampling**

#### resample.hpp
- `bootstrap_sample` - Bootstrap resampling
- `subsample` - Random subsampling
- `stratified_sample` - Stratified sampling
- `permutation_indices` - Generate random permutations

---

## Missing Operators (Recommendations)

### High Priority

#### 1. Dimensionality Reduction
**Missing:**
- `pca` - Principal Component Analysis
- `truncated_svd` - Truncated SVD for sparse matrices
- `randomized_pca` - Randomized PCA for large datasets
- `incremental_pca` - Mini-batch PCA

**Rationale:** Core preprocessing step in single-cell analysis

#### 2. Clustering
**Missing:**
- `kmeans` - K-means clustering
- `leiden` - Leiden community detection
- `louvain` - Louvain clustering
- `hierarchical_clustering` - Agglomerative clustering
- `dbscan` - Density-based clustering

**Rationale:** Essential for cell type identification

#### 3. Differential Expression
**Missing:**
- `logistic_regression` - Logistic regression for DE
- `negative_binomial_regression` - NB regression for counts
- `anova` - One-way ANOVA
- `permutation_fdr` - FDR correction

**Rationale:** Core analysis task

#### 4. Imputation
**Missing:**
- `knn_impute` - KNN-based imputation
- `magic_impute` - MAGIC algorithm
- `dca_denoise` - Deep count autoencoder
- `alra_impute` - ALRA method

**Rationale:** Handle dropout events in sparse data

#### 5. Batch Correction
**Missing:**
- `combat` - ComBat batch correction
- `harmony` - Harmony integration
- `scanorama` - Scanorama alignment
- `mnn_correct` - Mutual nearest neighbors

**Rationale:** Critical for multi-batch integration

### Medium Priority

#### 6. Manifold Learning
**Missing:**
- `umap` - UMAP projection
- `tsne` - t-SNE embedding
- `diffusion_map` - Diffusion maps
- `force_atlas2` - Force-directed layout

**Rationale:** Visualization and exploration

#### 7. Trajectory Analysis
**Missing:**
- `pseudotime` - Pseudotime calculation
- `velocity` - RNA velocity
- `lineage_tracing` - Lineage inference
- `diffusion_pseudotime` - DPT algorithm

**Rationale:** Developmental biology applications

#### 8. Gene Set Enrichment
**Missing:**
- `gsea` - Gene Set Enrichment Analysis
- `ora` - Over-Representation Analysis
- `gsva` - Gene Set Variation Analysis
- `auc_cell` - AUCell scoring

**Rationale:** Functional interpretation

#### 9. Cell-Cell Communication
**Missing:**
- `ligand_receptor` - L-R interaction scoring
- `cellchat` - CellChat analysis
- `cpdb` - CellPhoneDB method
- `nichenet` - NicheNet algorithm

**Rationale:** Emerging important analysis

#### 10. Network Analysis
**Missing:**
- `pagerank` - PageRank centrality
- `betweenness` - Betweenness centrality
- `modularity` - Modularity optimization
- `connected_components` - Component detection

**Rationale:** Graph-based analysis

### Low Priority

#### 11. Survival Analysis
**Missing:**
- `kaplan_meier` - Survival curves
- `log_rank_test` - Log-rank test
- `cox_regression` - Cox proportional hazards

#### 12. Image Analysis
**Missing:**
- `spot_detection` - Spatial transcriptomics spots
- `cell_segmentation` - Cell boundary detection
- `tissue_registration` - Image alignment

#### 13. Multi-Omics
**Missing:**
- `wnn` - Weighted nearest neighbor (multimodal)
- `mofa` - Multi-Omics Factor Analysis
- `seurat_integration` - Seurat v3/v4 integration

---

## Operator Categories by Complexity

### Low Complexity (Easy to Implement)
- Matrix norms and distances
- Simple statistical functions (median, quantile)
- Element-wise operations (exp, sqrt, etc.)
- Sparse matrix utilities (nnz_per_row, etc.)

### Medium Complexity
- PCA/SVD (via Eigen or similar)
- K-means clustering
- Basic regression models
- Permutation tests

### High Complexity
- UMAP (requires optimization)
- Leiden/Louvain (graph partitioning)
- Batch correction methods (iterative algorithms)
- Deep learning methods (DCA, scVI)

---

## Integration Priorities

### Immediate (Next Release)
1. **PCA/SVD** - Most requested, high impact
2. **K-means** - Standard clustering
3. **Logistic Regression** - DE analysis
4. **Basic imputation** - KNN-based

### Short-term (1-2 Releases)
1. **Leiden clustering** - Community detection
2. **Harmony** - Batch correction
3. **Basic UMAP** - Dimensionality reduction
4. **GSEA** - Enrichment analysis

### Long-term (Future)
1. **RNA velocity** - Complex algorithm
2. **CellChat** - Requires extensive DB
3. **Deep learning methods** - Requires DL backend

---

## Performance Considerations

### CPU-Bound Operators (SIMD Optimization Priority)
- All statistical tests
- Distance computations
- Matrix operations
- Correlation calculations

### Memory-Bound Operators (Cache Optimization Priority)
- Graph algorithms
- Clustering
- Batch correction
- Imputation

### Communication-Bound (Parallelization Strategy)
- Embarrassingly parallel: statistical tests, QC
- Requires synchronization: clustering, graph algorithms
- Sequential: some trajectory methods

---

## Recommendations Summary

**Top 5 Operators to Add:**
1. ✅ **PCA/SVD** - Fundamental, high demand
2. ✅ **Leiden Clustering** - Better than Louvain
3. ✅ **Logistic Regression** - DE analysis
4. ✅ **KNN Imputation** - Data quality
5. ✅ **Harmony** - Best batch correction

**Architecture Recommendations:**
- Separate sparse/dense implementations where beneficial
- Provide both exact and approximate variants
- Consider GPU acceleration for large-scale operations
- Maintain SIMD optimization for all kernels

**API Design:**
- Consistent naming: `compute_`, `select_`, `filter_`
- Always provide in-place and copy variants
- Template on matrix format (CSR/CSC)
- Clear precondition documentation

