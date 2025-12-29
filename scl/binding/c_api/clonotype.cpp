// =============================================================================
// FILE: scl/binding/c_api/clonotype.cpp
// BRIEF: C API implementation for TCR/BCR clonal analysis
// =============================================================================

#include "scl/binding/c_api/clonotype.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/clonotype.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Clone Size Distribution
// =============================================================================

SCL_EXPORT scl_error_t scl_clonotype_size_distribution(
    const scl_index_t* clone_ids,
    const scl_size_t n_cells,
    scl_size_t* clone_sizes,
    scl_size_t* n_clones,
    const scl_size_t max_clones) {
    
    SCL_C_API_CHECK_NULL(clone_ids, "Clone IDs array is null");
    SCL_C_API_CHECK_NULL(clone_sizes, "Output clone sizes array is null");
    SCL_C_API_CHECK_NULL(n_clones, "Output n_clones pointer is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");
    SCL_C_API_CHECK(max_clones > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Max clones must be positive");
    
    SCL_C_API_TRY
        const Array<const Index> ids_arr(
            reinterpret_cast<const Index*>(clone_ids),
            n_cells
        );
        
        scl::kernel::clonotype::clone_size_distribution(
            ids_arr, clone_sizes, *n_clones, max_clones
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Clonal Diversity
// =============================================================================

SCL_EXPORT scl_error_t scl_clonotype_diversity(
    const scl_index_t* clone_ids,
    const scl_size_t n_cells,
    scl_real_t* shannon_diversity,
    scl_real_t* simpson_diversity,
    scl_real_t* gini_index) {
    
    SCL_C_API_CHECK_NULL(clone_ids, "Clone IDs array is null");
    SCL_C_API_CHECK_NULL(shannon_diversity, "Output Shannon diversity pointer is null");
    SCL_C_API_CHECK_NULL(simpson_diversity, "Output Simpson diversity pointer is null");
    SCL_C_API_CHECK_NULL(gini_index, "Output Gini index pointer is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");
    
    SCL_C_API_TRY
        const Array<const Index> ids_arr(
            reinterpret_cast<const Index*>(clone_ids),
            n_cells
        );
        
        Real shannon = Real(0);
        Real simpson = Real(0);
        Real gini = Real(0);
        
        scl::kernel::clonotype::clonal_diversity(
            ids_arr, shannon, simpson, gini
        );
        
        *shannon_diversity = static_cast<scl_real_t>(shannon);
        *simpson_diversity = static_cast<scl_real_t>(simpson);
        *gini_index = static_cast<scl_real_t>(gini);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Clone Dynamics
// =============================================================================

SCL_EXPORT scl_error_t scl_clonotype_dynamics(
    const scl_index_t* clone_ids_t1,
    const scl_size_t n_cells_t1,
    const scl_index_t* clone_ids_t2,
    const scl_size_t n_cells_t2,
    scl_real_t* expansion_rates,
    scl_size_t* n_clones,
    const scl_size_t max_clones) {
    
    SCL_C_API_CHECK_NULL(clone_ids_t1, "Clone IDs t1 array is null");
    SCL_C_API_CHECK_NULL(clone_ids_t2, "Clone IDs t2 array is null");
    SCL_C_API_CHECK_NULL(expansion_rates, "Output expansion rates array is null");
    SCL_C_API_CHECK_NULL(n_clones, "Output n_clones pointer is null");
    SCL_C_API_CHECK(n_cells_t1 > 0 && n_cells_t2 > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");
    
    SCL_C_API_TRY
        const Array<const Index> ids1_arr(
            reinterpret_cast<const Index*>(clone_ids_t1),
            n_cells_t1
        );
        const Array<const Index> ids2_arr(
            reinterpret_cast<const Index*>(clone_ids_t2),
            n_cells_t2
        );
        
        scl::kernel::clonotype::clone_dynamics(
            ids1_arr, ids2_arr,
            reinterpret_cast<Real*>(expansion_rates),
            *n_clones, max_clones
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Shared Clonotypes
// =============================================================================

SCL_EXPORT scl_error_t scl_clonotype_shared(
    const scl_index_t* clone_ids_sample1,
    const scl_size_t n_cells_1,
    const scl_index_t* clone_ids_sample2,
    const scl_size_t n_cells_2,
    scl_index_t* shared_clones,
    scl_size_t* n_shared,
    scl_real_t* jaccard_index,
    const scl_size_t max_shared) {
    
    SCL_C_API_CHECK_NULL(clone_ids_sample1, "Clone IDs sample 1 array is null");
    SCL_C_API_CHECK_NULL(clone_ids_sample2, "Clone IDs sample 2 array is null");
    SCL_C_API_CHECK_NULL(shared_clones, "Output shared clones array is null");
    SCL_C_API_CHECK_NULL(n_shared, "Output n_shared pointer is null");
    SCL_C_API_CHECK_NULL(jaccard_index, "Output Jaccard index pointer is null");
    
    SCL_C_API_TRY
        const Array<const Index> ids1_arr(
            reinterpret_cast<const Index*>(clone_ids_sample1),
            n_cells_1
        );
        const Array<const Index> ids2_arr(
            reinterpret_cast<const Index*>(clone_ids_sample2),
            n_cells_2
        );
        
        Real jaccard = Real(0);
        
        scl::kernel::clonotype::shared_clonotypes(
            ids1_arr, ids2_arr,
            reinterpret_cast<Index*>(shared_clones),
            *n_shared, jaccard, max_shared
        );
        
        *jaccard_index = static_cast<scl_real_t>(jaccard);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Clone Phenotype
// =============================================================================

SCL_EXPORT scl_error_t scl_clonotype_phenotype(
    scl_sparse_t expression,
    const scl_index_t* clone_ids,
    const scl_size_t n_cells,
    const scl_size_t n_genes,
    scl_real_t* clone_profiles,
    scl_size_t* n_clones,
    const scl_size_t max_clones) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(clone_ids, "Clone IDs array is null");
    SCL_C_API_CHECK_NULL(clone_profiles, "Output clone profiles array is null");
    SCL_C_API_CHECK_NULL(n_clones, "Output n_clones pointer is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Array<const Index> ids_arr(
            reinterpret_cast<const Index*>(clone_ids),
            n_cells
        );
        
        expression->visit([&](auto& m) {
            scl::kernel::clonotype::clone_phenotype(
                m, ids_arr,
                reinterpret_cast<Real*>(clone_profiles),
                *n_clones, max_clones
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Clonality Score
// =============================================================================

SCL_EXPORT scl_error_t scl_clonotype_clonality_score(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    const scl_size_t n_cells,
    scl_real_t* clonality_per_cluster,
    const scl_size_t n_clusters) {
    
    SCL_C_API_CHECK_NULL(clone_ids, "Clone IDs array is null");
    SCL_C_API_CHECK_NULL(cluster_labels, "Cluster labels array is null");
    SCL_C_API_CHECK_NULL(clonality_per_cluster, "Output clonality array is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");
    SCL_C_API_CHECK(n_clusters > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of clusters must be positive");
    
    SCL_C_API_TRY
        const Array<const Index> ids_arr(
            reinterpret_cast<const Index*>(clone_ids),
            n_cells
        );
        const Array<const Index> clusters_arr(
            reinterpret_cast<const Index*>(cluster_labels),
            n_cells
        );
        Array<Real> clonality_arr(
            reinterpret_cast<Real*>(clonality_per_cluster),
            n_clusters
        );
        
        scl::kernel::clonotype::clonality_score(
            ids_arr, clusters_arr, clonality_arr
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Repertoire Overlap
// =============================================================================

SCL_EXPORT scl_error_t scl_clonotype_repertoire_overlap(
    const scl_index_t* clone_ids_1,
    const scl_size_t n_cells_1,
    const scl_index_t* clone_ids_2,
    const scl_size_t n_cells_2,
    scl_real_t* overlap_index) {
    
    SCL_C_API_CHECK_NULL(clone_ids_1, "Clone IDs 1 array is null");
    SCL_C_API_CHECK_NULL(clone_ids_2, "Clone IDs 2 array is null");
    SCL_C_API_CHECK_NULL(overlap_index, "Output overlap index pointer is null");
    
    SCL_C_API_TRY
        const Array<const Index> ids1_arr(
            reinterpret_cast<const Index*>(clone_ids_1),
            n_cells_1
        );
        const Array<const Index> ids2_arr(
            reinterpret_cast<const Index*>(clone_ids_2),
            n_cells_2
        );
        
        const Real overlap = scl::kernel::clonotype::repertoire_overlap_morisita(
            ids1_arr, ids2_arr
        );
        
        *overlap_index = static_cast<scl_real_t>(overlap);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Diversity Per Cluster
// =============================================================================

SCL_EXPORT scl_error_t scl_clonotype_diversity_per_cluster(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    const scl_size_t n_cells,
    scl_real_t* shannon_per_cluster,
    scl_real_t* simpson_per_cluster,
    const scl_size_t n_clusters) {
    
    SCL_C_API_CHECK_NULL(clone_ids, "Clone IDs array is null");
    SCL_C_API_CHECK_NULL(cluster_labels, "Cluster labels array is null");
    SCL_C_API_CHECK_NULL(shannon_per_cluster, "Output Shannon array is null");
    SCL_C_API_CHECK_NULL(simpson_per_cluster, "Output Simpson array is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");
    SCL_C_API_CHECK(n_clusters > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of clusters must be positive");
    
    SCL_C_API_TRY
        const Array<const Index> ids_arr(
            reinterpret_cast<const Index*>(clone_ids),
            n_cells
        );
        const Array<const Index> clusters_arr(
            reinterpret_cast<const Index*>(cluster_labels),
            n_cells
        );
        Array<Real> shannon_arr(
            reinterpret_cast<Real*>(shannon_per_cluster),
            n_clusters
        );
        Array<Real> simpson_arr(
            reinterpret_cast<Real*>(simpson_per_cluster),
            n_clusters
        );
        
        scl::kernel::clonotype::diversity_per_cluster(
            ids_arr, clusters_arr, shannon_arr, simpson_arr
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Clone Transition Matrix
// =============================================================================

SCL_EXPORT scl_error_t scl_clonotype_transition_matrix(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    const scl_size_t n_cells,
    scl_real_t* transition_matrix,
    const scl_size_t n_clusters) {
    
    SCL_C_API_CHECK_NULL(clone_ids, "Clone IDs array is null");
    SCL_C_API_CHECK_NULL(cluster_labels, "Cluster labels array is null");
    SCL_C_API_CHECK_NULL(transition_matrix, "Output transition matrix is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");
    SCL_C_API_CHECK(n_clusters > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of clusters must be positive");
    
    SCL_C_API_TRY
        const Array<const Index> ids_arr(
            reinterpret_cast<const Index*>(clone_ids),
            n_cells
        );
        const Array<const Index> clusters_arr(
            reinterpret_cast<const Index*>(cluster_labels),
            n_cells
        );
        
        scl::kernel::clonotype::clone_transition_matrix(
            ids_arr, clusters_arr,
            reinterpret_cast<Real*>(transition_matrix),
            n_clusters
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Rarefaction Analysis
// =============================================================================

SCL_EXPORT scl_error_t scl_clonotype_rarefaction(
    const scl_index_t* clone_ids,
    const scl_size_t n_cells,
    const scl_size_t subsample_size,
    const scl_size_t n_iterations,
    scl_real_t* mean_diversity,
    scl_real_t* std_diversity,
    const uint64_t seed) {
    
    SCL_C_API_CHECK_NULL(clone_ids, "Clone IDs array is null");
    SCL_C_API_CHECK_NULL(mean_diversity, "Output mean diversity pointer is null");
    SCL_C_API_CHECK_NULL(std_diversity, "Output std diversity pointer is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");
    SCL_C_API_CHECK(subsample_size > 0 && subsample_size <= n_cells,
                   SCL_ERROR_INVALID_ARGUMENT,
                   "Subsample size must be in (0, n_cells]");
    SCL_C_API_CHECK(n_iterations > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of iterations must be positive");
    
    SCL_C_API_TRY
        const Array<const Index> ids_arr(
            reinterpret_cast<const Index*>(clone_ids),
            n_cells
        );
        
        Real mean = Real(0);
        Real std = Real(0);
        
        scl::kernel::clonotype::rarefaction_diversity(
            ids_arr, subsample_size, n_iterations,
            mean, std, seed
        );
        
        *mean_diversity = static_cast<scl_real_t>(mean);
        *std_diversity = static_cast<scl_real_t>(std);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Expanded Clone Detection
// =============================================================================

SCL_EXPORT scl_error_t scl_clonotype_detect_expanded(
    const scl_index_t* clone_ids,
    const scl_size_t n_cells,
    const scl_size_t expansion_threshold,
    scl_index_t* expanded_clones,
    scl_size_t* n_expanded,
    const scl_size_t max_expanded) {
    
    SCL_C_API_CHECK_NULL(clone_ids, "Clone IDs array is null");
    SCL_C_API_CHECK_NULL(expanded_clones, "Output expanded clones array is null");
    SCL_C_API_CHECK_NULL(n_expanded, "Output n_expanded pointer is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");
    SCL_C_API_CHECK(expansion_threshold > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Expansion threshold must be positive");
    
    SCL_C_API_TRY
        const Array<const Index> ids_arr(
            reinterpret_cast<const Index*>(clone_ids),
            n_cells
        );
        
        scl::kernel::clonotype::detect_expanded_clones(
            ids_arr, expansion_threshold,
            reinterpret_cast<Index*>(expanded_clones),
            *n_expanded, max_expanded
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Clone Size Statistics
// =============================================================================

SCL_EXPORT scl_error_t scl_clonotype_size_statistics(
    const scl_index_t* clone_ids,
    const scl_size_t n_cells,
    scl_real_t* mean_size,
    scl_real_t* median_size,
    scl_real_t* max_size,
    scl_size_t* n_singletons,
    scl_size_t* n_clones) {
    
    SCL_C_API_CHECK_NULL(clone_ids, "Clone IDs array is null");
    SCL_C_API_CHECK_NULL(mean_size, "Output mean size pointer is null");
    SCL_C_API_CHECK_NULL(median_size, "Output median size pointer is null");
    SCL_C_API_CHECK_NULL(max_size, "Output max size pointer is null");
    SCL_C_API_CHECK_NULL(n_singletons, "Output n_singletons pointer is null");
    SCL_C_API_CHECK_NULL(n_clones, "Output n_clones pointer is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");
    
    SCL_C_API_TRY
        const Array<const Index> ids_arr(
            reinterpret_cast<const Index*>(clone_ids),
            n_cells
        );
        
        Real mean = Real(0);
        Real median = Real(0);
        Real max = Real(0);
        Size singletons = 0;
        Size clones = 0;
        
        scl::kernel::clonotype::clone_size_statistics(
            ids_arr, mean, median, max, singletons, clones
        );
        
        *mean_size = static_cast<scl_real_t>(mean);
        *median_size = static_cast<scl_real_t>(median);
        *max_size = static_cast<scl_real_t>(max);
        *n_singletons = singletons;
        *n_clones = clones;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
