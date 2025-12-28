# Kernel Functions Summary

This document provides a comprehensive list of all functions and operators available in each kernel file.

## algebra.h / algebra.hpp
Sparse linear algebra operations:
- spmv: Sparse matrix-vector multiplication
- spmv_simple: Simple sparse matrix-vector multiplication
- spmv_add: Sparse matrix-vector multiplication with addition
- spmv_scaled: Scaled sparse matrix-vector multiplication

## alignment.hpp
Cell alignment and integration:
- find_cross_knn: Find cross-knn pairs between batches
- mnn_pairs: Find mutual nearest neighbor pairs
- find_anchors: Find anchor cells for alignment
- transfer_labels: Transfer labels between datasets
- integration_score: Compute integration quality score
- batch_mixing: Measure batch mixing
- compute_correction_vectors: Compute batch correction vectors
- smooth_correction_vectors: Smooth correction vectors
- cca_projection: Canonical correlation analysis projection
- kbet_score: k-nearest neighbor batch effect test score

## annotation.hpp
Cell type annotation:
- cosine_similarity: Compute cosine similarity
- count_votes: Count votes for cell types
- majority_vote: Majority voting assignment
- weighted_majority_vote: Weighted majority voting
- reference_mapping: Map cells to reference
- correlation_assignment: Assign using correlation
- build_reference_profiles: Build reference expression profiles
- marker_gene_score: Score based on marker genes
- detect_novel_types: Detect novel cell types
- detect_novel_types_by_distance: Detect novel types by distance
- label_propagation: Propagate labels on graph
- cell_type_marker_expression: Get marker expression per type
- hierarchical_annotation: Hierarchical annotation
- differential_markers: Find differential markers
- annotation_entropy: Compute annotation entropy
- annotation_quality_metrics: Compute annotation quality metrics
- assign_from_marker_scores: Assign from marker scores
- cluster_novel_cells: Cluster novel cells
- confusion_matrix: Build confusion matrix
- consensus_annotation: Consensus annotation
- top_markers_per_type: Top markers per type

## association.hpp
Multi-modal associations:
- gene_peak_correlation: Correlate genes with peaks
- cis_regulatory: Find cis-regulatory associations
- enhancer_gene_link: Link enhancers to genes
- multimodal_neighbors: Find multimodal neighbors
- feature_coupling: Couple features across modalities
- correlation_in_subset: Correlation in subset
- peak_to_gene_activity: Compute peak-to-gene activity

## bbknn.h / bbknn.hpp
Batch-balanced k-nearest neighbors:
- build_batch_groups: Build batch groups
- free_batch_groups: Free batch groups
- compute_norms: Compute norms for distance calculation
- bbknn: Batch-balanced k-nearest neighbors

## centrality.hpp
Graph centrality measures:
- compute_out_degrees: Compute out-degrees
- degree_centrality: Degree centrality
- weighted_degree_centrality: Weighted degree centrality
- pagerank: PageRank algorithm
- personalized_pagerank: Personalized PageRank
- hits: HITS algorithm
- eigenvector_centrality: Eigenvector centrality
- katz_centrality: Katz centrality
- closeness_centrality: Closeness centrality
- betweenness_centrality: Betweenness centrality
- betweenness_centrality_sampled: Sampled betweenness centrality
- harmonic_centrality: Harmonic centrality
- current_flow_betweenness_approx: Approximate current flow betweenness

## clonotype.hpp
T-cell/B-cell clonotype analysis:
- clone_size_distribution: Compute clone size distribution
- clonal_diversity: Compute clonal diversity
- clone_dynamics: Analyze clone dynamics
- shared_clonotypes: Find shared clonotypes
- clone_phenotype: Analyze clone phenotypes
- clonality_score: Compute clonality score
- repertoire_overlap_morisita: Morisita overlap index
- diversity_per_cluster: Diversity per cluster
- clone_transition_matrix: Clone transition matrix
- rarefaction_diversity: Rarefaction diversity
- detect_expanded_clones: Detect expanded clones
- clone_size_statistics: Clone size statistics

## communication.hpp
Cell-cell communication:
- extract_gene_expression: Extract gene expression
- build_type_masks: Build cell type masks
- compute_mean_expression: Compute mean expression
- lr_score_matrix: Ligand-receptor score matrix
- lr_score_batch: Ligand-receptor score batch
- lr_permutation_test: Permutation test for LR scores
- communication_probability: Communication probability
- sender_score: Sender cell scores
- receiver_score: Receiver cell scores
- spatial_communication_score: Spatial communication score
- differential_communication: Differential communication
- expression_specificity: Expression specificity
- natmi_edge_weight: NATMI edge weights
- aggregate_to_network: Aggregate to network
- aggregate_to_pathways: Aggregate to pathways
- filter_significant: Filter significant interactions
- network_centrality: Network centrality

## comparison.h / comparison.hpp
Statistical comparisons:
- composition_analysis: Composition analysis
- abundance_test: Abundance test
- differential_abundance: Differential abundance
- condition_response: Condition response analysis
- effect_size: Compute effect size
- glass_delta: Glass delta effect size
- hedges_g: Hedges g effect size

## components.hpp
Graph component analysis:
- connected_components: Find connected components
- is_connected: Check if graph is connected
- largest_component: Get largest component
- component_sizes: Get component sizes
- bfs: Breadth-first search
- multi_source_bfs: Multi-source BFS
- parallel_bfs: Parallel BFS
- dfs: Depth-first search
- topological_sort: Topological sort
- graph_diameter: Graph diameter
- average_path_length: Average path length
- clustering_coefficient: Clustering coefficient
- global_clustering_coefficient: Global clustering coefficient
- count_triangles: Count triangles
- degree_sequence: Degree sequence
- degree_statistics: Degree statistics
- degree_distribution: Degree distribution
- graph_density: Graph density
- kcore_decomposition: K-core decomposition

## correlation.h / correlation.hpp
Correlation computation:
- compute_stats: Compute correlation statistics
- pearson: Pearson correlation

## diffusion.hpp
Diffusion-based analysis:
- orthogonalize_vectors: Orthogonalize vectors
- compute_transition_matrix: Compute transition matrix
- diffuse_vector: Diffuse vector
- diffuse_matrix: Diffuse matrix
- compute_dpt: Compute diffusion pseudotime
- compute_dpt_multi_root: Multi-root diffusion pseudotime
- random_walk_with_restart: Random walk with restart
- diffusion_map_embedding: Diffusion map embedding
- heat_kernel_signature: Heat kernel signature
- magic_impute: MAGIC imputation
- diffusion_distance: Diffusion distance
- personalized_pagerank: Personalized PageRank via diffusion
- lazy_random_walk: Lazy random walk

## doublet.h / doublet.hpp
Doublet detection:
- simulate_doublets: Simulate doublets
- compute_knn_doublet_scores: KNN-based doublet scores
- compute_knn_doublet_scores_pca: KNN doublet scores with PCA
- scrublet_scores: Scrublet scores
- doubletfinder_pann: DoubletFinder PANN scores
- estimate_threshold: Estimate detection threshold
- call_doublets: Call doublets
- detect_bimodal_threshold: Detect bimodal threshold
- detect_doublets: Detect doublets
- doublet_score_stats: Doublet score statistics
- combined_doublet_score: Combined doublet score
- density_doublet_score: Density-based doublet score
- variance_doublet_score: Variance-based doublet score
- classify_doublet_types_knn: Classify doublet types
- cluster_doublet_enrichment: Cluster doublet enrichment
- expected_doublets: Expected number of doublets
- estimate_doublet_rate: Estimate doublet rate
- multiplet_rate_10x: Multiplet rate for 10x data
- get_singlet_indices: Get singlet cell indices

## enrichment.hpp
Gene set enrichment:
- pathway_activity: Pathway activity scores
- gsva_score: Gene Set Variation Analysis
- ssgsea: Single-sample GSEA
- gsea: Gene Set Enrichment Analysis
- gsea_running_sum: GSEA running sum
- ora_single_set: Over-representation analysis single set
- ora_batch: Over-representation analysis batch
- enrichment_map: Build enrichment map
- gene_set_overlap: Gene set overlap
- jaccard_similarity: Jaccard similarity
- leading_edge_genes: Leading edge genes
- rank_genes_by_score: Rank genes by score
- sort_by_pvalue: Sort by p-value
- filter_significant: Filter significant results
- benjamini_hochberg: Benjamini-Hochberg correction
- bonferroni: Bonferroni correction

## entropy.hpp
Entropy and information theory:
- count_entropy: Count entropy
- row_entropy: Row-wise entropy
- discretize_equal_width: Equal-width discretization
- discretize_equal_frequency: Equal-frequency discretization
- select_features_mi: Feature selection by mutual information
- mrmr_selection: Minimum redundancy maximum relevance selection
- conditional_entropy: Conditional entropy
- cross_entropy: Cross entropy
- gini_impurity: Gini impurity
- histogram_2d: 2D histogram
- information_gain: Information gain
- joint_entropy: Joint entropy
- js_divergence: Jensen-Shannon divergence
- kl_divergence: Kullback-Leibler divergence
- marginal_entropy: Marginal entropy
- mutual_information: Mutual information
- normalized_mi: Normalized mutual information
- perplexity: Perplexity
- symmetric_kl: Symmetric KL divergence
- adjusted_mi: Adjusted mutual information

## feature.h / feature.hpp
Feature statistics:
- standard_moments: Standard statistical moments
- clipped_moments: Clipped moments
- detection_rate: Detection rate per feature
- dispersion: Dispersion measure

## gnn.hpp
Graph neural networks:
- precompute_gcn_norm: Precompute GCN normalization
- message_passing: Message passing
- graph_attention: Graph attention mechanism
- multi_head_attention: Multi-head attention
- graph_convolution: Graph convolution
- sage_aggregate: GraphSAGE aggregation
- feature_smoothing: Feature smoothing
- global_pool: Global pooling
- hierarchical_pool: Hierarchical pooling
- compute_edge_features: Compute edge features
- batch_norm: Batch normalization
- dropout: Dropout
- dropout_inference: Dropout for inference
- layer_norm: Layer normalization
- skip_connection: Skip connection

## gram.h / gram.hpp
Gram matrix computation:
- gram: Compute Gram matrix

## grn.hpp
Gene regulatory network inference:
- correlation_network: Build correlation network
- correlation_network_sparse: Sparse correlation network
- partial_correlation_network: Partial correlation network
- mutual_information_network: Mutual information network
- tf_target_score: Transcription factor target score
- genie3_importance: GENIE3 importance scores
- regulon_activity: Regulon activity
- regulon_auc_score: Regulon AUC score
- infer_grn: Infer gene regulatory network
- tf_activity_from_regulons: TF activity from regulons
- build_regulons: Build regulons
- identify_hub_genes: Identify hub genes
- network_statistics: Network statistics

## group.h / group.hpp
Group statistics:
- group_stats: Compute group statistics

## hotspot.h / hotspot.hpp
Spatial hotspot analysis:
- local_morans_i: Local Moran's I
- getis_ord_g_star: Getis-Ord G* statistic
- classify_lisa_patterns: Classify LISA patterns
- identify_hotspots: Identify hotspots
- local_gearys_c: Local Geary's C
- global_morans_i: Global Moran's I
- global_gearys_c: Global Geary's C
- benjamini_hochberg_correction: Benjamini-Hochberg correction
- distance_band_weights: Distance band weights
- knn_weights: K-nearest neighbor weights
- bivariate_local_morans_i: Bivariate local Moran's I
- detect_spatial_clusters: Detect spatial clusters
- spatial_autocorrelation_summary: Spatial autocorrelation summary

## hvg.h / hvg.hpp
Highly variable genes:
- select_by_dispersion: Select by dispersion
- select_by_vst: Select by variance stabilizing transformation
- dispersion_simd: SIMD-optimized dispersion
- normalize_dispersion_simd: Normalize dispersion with SIMD
- select_top_k_partial: Select top k partially
- compute_moments: Compute moments
- compute_clipped_moments: Compute clipped moments

## impute.h / impute.hpp
Expression imputation:
- knn_impute_dense: KNN imputation for dense data
- knn_impute_weighted_dense: Weighted KNN imputation
- diffusion_impute_sparse_transition: Diffusion imputation with sparse transition
- diffusion_impute_dense: Diffusion imputation for dense data
- magic_impute: MAGIC imputation
- alra_impute: ALRA imputation
- impute_selected_genes: Impute selected genes
- detect_dropouts: Detect dropouts
- imputation_quality: Assess imputation quality
- smooth_expression: Smooth expression

## kernel.hpp
Kernel methods:
- kde_from_distances: Kernel density estimation from distances
- local_bandwidth: Local bandwidth selection
- adaptive_kde: Adaptive kernel density estimation
- compute_kernel_matrix: Compute kernel matrix
- kernel_row_sums: Kernel row sums
- nadaraya_watson: Nadaraya-Watson estimator
- kernel_smooth_graph: Kernel smoothing on graph
- local_linear_regression: Local linear regression
- nystrom_approximation: Nystrom approximation
- mean_shift_step: Mean shift step
- kernel_entropy: Kernel entropy
- find_bandwidth_for_perplexity: Find bandwidth for target perplexity
- kernel_mmd_from_groups: Kernel MMD between groups
- rbf_sparse: Sparse RBF kernel
- adaptive_rbf: Adaptive RBF kernel
- evaluate_kernel: Evaluate kernel function
- kernel_normalization: Kernel normalization
- scott_bandwidth: Scott's rule bandwidth
- silverman_bandwidth: Silverman's rule bandwidth
- params: Kernel parameters
- self_params: Self kernel parameters

## leiden.hpp
Leiden clustering:
- cluster: Leiden clustering
- cluster_multilevel: Multilevel Leiden clustering
- compute_modularity: Compute modularity
- sort_communities_by_size: Sort communities by size

## lineage.hpp
Lineage tracing:
- lineage_coupling: Lineage coupling analysis
- fate_bias: Cell fate bias
- build_lineage_tree: Build lineage tree
- lineage_distance: Lineage distance
- barcode_clone_assignment: Barcode-to-clone assignment
- clonal_fate_probability: Clonal fate probability
- fate_bias_per_type: Fate bias per cell type
- lineage_sharing: Lineage sharing
- lineage_commitment: Lineage commitment
- progenitor_score: Progenitor score
- lineage_transition_probability: Lineage transition probability
- clone_generation: Clone generation analysis

## log1p.h / log1p.hpp
Logarithmic transformations:
- log1p_inplace: log(1+x) in-place
- log2p1_inplace: log2(1+x) in-place
- expm1_inplace: exp(x)-1 in-place

## louvain.h / louvain.hpp
Louvain clustering:
- cluster: Louvain clustering
- compute_modularity: Compute modularity
- community_sizes: Get community sizes
- get_community_members: Get community members

## merge.h / merge.hpp
Matrix merging operations:
- vstack: Vertically stack two matrices (concatenate along primary axis)
- hstack: Horizontally stack two matrices (concatenate along secondary axis)

## markers.hpp
Marker gene detection:
- group_mean_expression: Group mean expression
- percent_expressed: Percentage of cells expressing
- log_fold_change: Log fold change
- one_vs_rest_stats: One-vs-rest statistics
- rank_genes_groups: Rank genes by groups
- expression_entropy: Expression entropy
- filter_markers: Filter markers
- find_unique_markers: Find unique markers
- gini_specificity: Gini specificity
- marker_overlap_jaccard: Marker overlap Jaccard
- rank_genes_by_score: Rank genes by score
- tau_specificity: Tau specificity
- top_n_markers: Top N markers
- volcano_score: Volcano plot score

## merge.h / merge.hpp
Matrix merging:
- vstack: Vertical stacking
- hstack: Horizontal stacking

## metrics.h / metrics.hpp
Clustering metrics:
- silhouette_score: Silhouette score
- silhouette_samples: Per-sample silhouette scores
- adjusted_rand_index: Adjusted Rand index
- normalized_mutual_information: Normalized mutual information
- graph_connectivity: Graph connectivity score
- batch_entropy: Batch mixing entropy
- lisi: Local inverse Simpson index
- fowlkes_mallows_index: Fowlkes-Mallows index
- v_measure: V-measure
- homogeneity_score: Homogeneity score
- completeness_score: Completeness score
- purity_score: Purity score
- mean_lisi: Mean LISI
- mean_batch_entropy: Mean batch entropy

## mmd.h / mmd.hpp
Maximum mean discrepancy:
- mmd_rbf: MMD with RBF kernel

## multiple_testing.hpp
Multiple testing correction:
- benjamini_hochberg: Benjamini-Hochberg FDR correction
- bonferroni: Bonferroni correction
- storey_qvalue: Storey q-value
- local_fdr: Local false discovery rate
- empirical_fdr: Empirical FDR
- benjamini_yekutieli: Benjamini-Yekutieli correction
- holm_bonferroni: Holm-Bonferroni correction
- hochberg: Hochberg correction
- count_significant: Count significant results
- significant_indices: Get significant indices
- neglog10_pvalues: Negative log10 p-values
- fisher_combine: Fisher's method for combining p-values
- stouffer_combine: Stouffer's method for combining p-values

## mwu.h / mwu.hpp
Mann-Whitney U test:
- count_groups: Count groups
- mwu_test: Mann-Whitney U test

## neighbors.h / neighbors.hpp
Nearest neighbors:
- compute_norms: Compute norms for distance
- knn: K-nearest neighbors

## niche.hpp
Spatial niche analysis:
- neighborhood_composition: Neighborhood composition
- neighborhood_enrichment: Neighborhood enrichment
- niche_clustering: Niche clustering
- identify_niche_patterns: Identify niche patterns
- cell_cell_contact: Cell-cell contact analysis
- contact_significance: Contact significance
- spatial_communication: Spatial communication
- colocalization_score: Colocalization score
- colocalization_matrix: Colocalization matrix
- single_cell_composition: Single-cell composition

## normalize.h / normalize.hpp
Normalization:
- compute_row_sums: Compute row sums
- scale_primary: Scale primary axis
- primary_sums_masked: Primary sums with mask
- detect_highly_expressed: Detect highly expressed genes

## outlier.h / outlier.hpp
Outlier detection:
- isolation_score: Isolation score
- local_outlier_factor: Local outlier factor
- ambient_detection: Ambient RNA detection
- empty_drops: Empty droplet detection
- outlier_genes: Outlier gene detection
- doublet_score: Doublet score
- mitochondrial_outliers: Mitochondrial outliers
- qc_filter: Quality control filtering

## permutation.h / permutation.hpp
Permutation testing:
- permutation_test: Permutation test
- permutation_correlation_test: Permutation correlation test
- fdr_correction_bh: FDR correction Benjamini-Hochberg
- fdr_correction_by: FDR correction Benjamini-Yekutieli
- bonferroni_correction: Bonferroni correction
- holm_correction: Holm correction
- count_significant: Count significant results
- get_significant_indices: Get significant indices
- batch_permutation_test: Batch permutation test

## projection.h / projection.hpp
Dimensionality reduction:
- project_with_matrix: Project with matrix
- project_gaussian_otf: Gaussian random projection on-the-fly
- project_achlioptas_otf: Achlioptas random projection on-the-fly
- project_sparse_otf: Sparse random projection on-the-fly
- project_countsketch: CountSketch projection
- project: Generic projection
- compute_jl_dimension: Compute Johnson-Lindenstrauss dimension
- valid: Validate projection parameters

## propagation.h / propagation.hpp
Label propagation:
- label_propagation: Label propagation
- label_spreading: Label spreading
- inductive_transfer: Inductive transfer
- confidence_propagation: Confidence propagation
- harmonic_function: Harmonic function
- get_hard_labels: Get hard labels from soft
- init_soft_labels: Initialize soft labels

## pseudotime.h / pseudotime.hpp
Pseudotime analysis:
- dijkstra_shortest_path: Dijkstra shortest path
- dijkstra_multi_source: Multi-source Dijkstra
- graph_pseudotime: Graph-based pseudotime
- diffusion_pseudotime: Diffusion pseudotime
- select_root_cell: Select root cell
- select_root_peripheral: Select peripheral root
- detect_branch_points: Detect branch points
- segment_trajectory: Segment trajectory
- smooth_pseudotime: Smooth pseudotime
- pseudotime_correlation: Pseudotime correlation
- velocity_weighted_pseudotime: Velocity-weighted pseudotime
- find_terminal_states: Find terminal states
- compute_backbone: Compute trajectory backbone
- compute_pseudotime: Compute pseudotime

## qc.h / qc.hpp
Quality control:
- compute_basic_qc: Compute basic QC metrics
- compute_subset_pct: Compute subset percentage
- compute_fused_qc: Compute fused QC metrics

## resample.h / resample.hpp
Resampling:
- downsample: Downsample to target count
- downsample_variable: Variable target downsampling
- binomial_resample: Binomial resampling
- poisson_resample: Poisson resampling

## reorder.hpp
Matrix reordering:
- align_secondary: Align secondary axis
- compute_filtered_nnz: Compute filtered non-zero count
- build_inverse_permutation: Build inverse permutation

## sampling.h / sampling.hpp
Cell sampling:
- geometric_sketching: Geometric sketching
- density_preserving: Density-preserving sampling
- landmark_selection: Landmark selection
- representative_cells: Representative cell selection
- balanced_sampling: Balanced sampling
- stratified_sampling: Stratified sampling
- uniform_sampling: Uniform sampling
- importance_sampling: Importance sampling
- reservoir_sampling: Reservoir sampling

## scale.h / scale.hpp
Scaling operations:
- standardize: Standardize (z-score)
- scale_rows: Scale rows
- shift_rows: Shift rows

## scoring.h / scoring.hpp
Gene signature scoring:
- compute_gene_means: Compute gene means
- mean_score: Mean signature score
- weighted_score: Weighted signature score
- auc_score: AUC score
- module_score: Module score
- zscore_score: Z-score signature score
- gene_set_score: Gene set score
- differential_score: Differential score
- cell_cycle_score: Cell cycle score
- quantile_score: Quantile score
- multi_signature_score: Multi-signature score

## slice.h / slice.hpp
Matrix slicing:
- inspect_slice_primary: Inspect primary axis slice
- materialize_slice_primary: Materialize primary slice
- inspect_filter_secondary: Inspect secondary filter
- materialize_filter_secondary: Materialize secondary filter

## softmax.h / softmax.hpp
Softmax operations:
- softmax_inplace: Softmax in-place
- log_softmax_inplace: Log softmax in-place

## sparse.h / sparse.hpp
Sparse matrix operations:
- primary_sums: Primary axis sums
- primary_means: Primary axis means
- primary_variances: Primary axis variances
- primary_nnz: Primary axis non-zero counts

## sparse_opt.hpp
Sparse optimization:
- lasso_coordinate_descent: Lasso coordinate descent
- elastic_net_coordinate_descent: Elastic net coordinate descent
- coordinate_descent_screening: Coordinate descent with screening
- proximal_gradient: Proximal gradient method
- fista: Fast iterative shrinkage-thresholding
- prox_l1: L1 proximal operator
- prox_elastic_net: Elastic net proximal operator
- ista: Iterative shrinkage-thresholding
- iht: Iterative hard thresholding
- lasso_path: Lasso regularization path
- lasso_path_cv: Lasso path with cross-validation
- group_lasso: Group lasso
- sparse_logistic_regression: Sparse logistic regression
- compute_coordinate_gradient: Compute coordinate gradient
- update_residuals: Update residuals
- check_convergence_coef: Check coefficient convergence
- estimate_lipschitz_constant: Estimate Lipschitz constant

## spatial.h / spatial.hpp
Spatial statistics:
- weight_sum: Weighted sum
- morans_i: Moran's I statistic
- gearys_c: Geary's C statistic

## spatial_pattern.hpp
Spatial pattern analysis:
- compute_spatial_weights: Compute spatial weights
- spatial_variability: Spatial variability
- spatial_gradient: Spatial gradient
- periodic_pattern: Periodic pattern detection
- boundary_detection: Boundary detection
- spatial_domain: Spatial domain identification
- hotspot_analysis: Hotspot analysis
- spatial_autocorrelation: Spatial autocorrelation
- spatial_smoothing: Spatial smoothing
- spatial_coexpression: Spatial coexpression
- ripleys_k: Ripley's K function
- spatial_entropy: Spatial entropy

## state.h / state.hpp
Cell state scoring:
- stemness_score: Stemness score
- differentiation_potential: Differentiation potential
- proliferation_score: Proliferation score
- stress_score: Stress score
- state_entropy: State entropy
- cell_cycle_score: Cell cycle score
- quiescence_score: Quiescence score
- metabolic_score: Metabolic score
- apoptosis_score: Apoptosis score
- signature_score: Signature score
- multi_signature_score: Multi-signature score
- transcriptional_diversity: Transcriptional diversity
- expression_complexity: Expression complexity
- combined_state_score: Combined state score

## subpopulation.hpp
Subpopulation analysis:
- kmeans_cluster: K-means clustering
- subclustering: Subclustering
- cluster_stability: Cluster stability
- cluster_purity: Cluster purity
- rare_cell_detection: Rare cell detection
- population_balance: Population balance
- cluster_cohesion: Cluster cohesion
- cluster_separation: Cluster separation
- identify_heterogeneous_clusters: Identify heterogeneous clusters
- marker_based_subpopulation: Marker-based subpopulation
- cluster_quality_score: Cluster quality score
- find_optimal_subclusters: Find optimal subclusters
- cell_type_proportions: Cell type proportions

## tissue.hpp
Tissue architecture:
- find_knn: Find k-nearest neighbors
- tissue_architecture: Tissue architecture analysis
- layer_assignment: Layer assignment
- radial_layer_assignment: Radial layer assignment
- zonation_score: Zonation score
- morphological_features: Morphological features
- tissue_module: Tissue module identification
- neighborhood_composition: Neighborhood composition
- cell_type_interaction: Cell type interaction
- boundary_cells: Boundary cell detection
- region_statistics: Region statistics
- spatial_coherence: Spatial coherence

## transition.h / transition.hpp
Transition matrix operations:
- sparse_matvec: Sparse matrix-vector multiplication
- sparse_matvec_transpose: Sparse matrix-vector transpose
- is_stochastic: Check if matrix is stochastic
- transition_matrix_from_velocity: Build transition from velocity
- row_normalize_to_stochastic: Normalize rows to stochastic
- symmetrize_transition: Symmetrize transition matrix
- stationary_distribution: Compute stationary distribution
- identify_terminal_states: Identify terminal states
- absorption_probability: Absorption probability
- hitting_time: Hitting time
- time_to_absorption: Time to absorption
- compute_top_eigenvectors: Compute top eigenvectors
- metastable_states: Metastable states
- coarse_grain_transition: Coarse-grain transition matrix
- lineage_drivers: Lineage drivers
- forward_committor: Forward committor probability
- directional_score: Directional score

## ttest.h / ttest.hpp
T-test:
- compute_group_stats: Compute group statistics
- ttest: Student's t-test

## velocity.h / velocity.hpp
RNA velocity:
- fit_gene_kinetics: Fit gene kinetics model
- compute_velocity: Compute RNA velocity
- splice_ratio: Splicing ratio
- velocity_graph: Velocity graph
- velocity_graph_cosine: Velocity graph with cosine
- velocity_embedding: Velocity embedding
- velocity_grid: Velocity grid
- velocity_confidence: Velocity confidence
- latent_time: Latent time
- cell_fate_probability: Cell fate probability
- select_velocity_genes: Select velocity genes
- velocity_pseudotime: Velocity pseudotime
- velocity_divergence: Velocity divergence
- select_root_by_velocity: Select root by velocity
- detect_terminal_states: Detect terminal states

