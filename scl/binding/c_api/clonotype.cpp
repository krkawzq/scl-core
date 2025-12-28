// =============================================================================
// FILE: scl/binding/c_api/clonotype/clonotype.cpp
// BRIEF: C API implementation for clonal analysis
// =============================================================================

#include "scl/binding/c_api/clonotype.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/clonotype.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::clonotype;

extern "C" {

scl_error_t scl_clone_size_distribution(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_size_t* clone_sizes,
    scl_size_t* n_clones,
    scl_size_t max_clones)
{
    if (!clone_ids || !clone_sizes || !n_clones) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Index> ids_arr(reinterpret_cast<const Index*>(clone_ids), n_cells);
        clone_size_distribution(ids_arr, clone_sizes, *n_clones, max_clones);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_clonal_diversity(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_real_t* shannon_diversity,
    scl_real_t* simpson_diversity,
    scl_real_t* gini_index)
{
    if (!clone_ids || !shannon_diversity || !simpson_diversity || !gini_index) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Index> ids_arr(reinterpret_cast<const Index*>(clone_ids), n_cells);
        Real shannon, simpson, gini;
        clonal_diversity(ids_arr, shannon, simpson, gini);
        *shannon_diversity = shannon;
        *simpson_diversity = simpson;
        *gini_index = gini;
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_clone_dynamics(
    const scl_index_t* clone_ids_t1,
    scl_size_t n_cells_t1,
    const scl_index_t* clone_ids_t2,
    scl_size_t n_cells_t2,
    scl_real_t* expansion_rates,
    scl_size_t* n_clones,
    scl_size_t max_clones)
{
    if (!clone_ids_t1 || !clone_ids_t2 || !expansion_rates || !n_clones) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Index> ids1_arr(reinterpret_cast<const Index*>(clone_ids_t1), n_cells_t1);
        Array<const Index> ids2_arr(reinterpret_cast<const Index*>(clone_ids_t2), n_cells_t2);
        Array<Real> rates_arr(reinterpret_cast<Real*>(expansion_rates), max_clones);
        clone_dynamics(ids1_arr, ids2_arr, rates_arr.ptr, *n_clones, max_clones);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_shared_clonotypes(
    const scl_index_t* clone_ids_sample1,
    scl_size_t n_cells_1,
    const scl_index_t* clone_ids_sample2,
    scl_size_t n_cells_2,
    scl_index_t* shared_clones,
    scl_size_t* n_shared,
    scl_real_t* jaccard_index,
    scl_size_t max_shared)
{
    if (!clone_ids_sample1 || !clone_ids_sample2 || !shared_clones || !n_shared || !jaccard_index) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Index> ids1_arr(reinterpret_cast<const Index*>(clone_ids_sample1), n_cells_1);
        Array<const Index> ids2_arr(reinterpret_cast<const Index*>(clone_ids_sample2), n_cells_2);
        Array<Index> shared_arr(reinterpret_cast<Index*>(shared_clones), max_shared);
        Real jaccard;
        shared_clonotypes(ids1_arr, ids2_arr, shared_arr.ptr, *n_shared, jaccard, max_shared);
        *jaccard_index = jaccard;
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_clone_phenotype(
    scl_sparse_t expression,
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_size_t n_genes,
    scl_real_t* clone_profiles,
    scl_size_t* n_clones,
    scl_size_t max_clones)
{
    if (!expression || !clone_ids || !clone_profiles || !n_clones) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(expression);
        Array<const Index> ids_arr(reinterpret_cast<const Index*>(clone_ids), n_cells);
        Array<Real> profiles_arr(reinterpret_cast<Real*>(clone_profiles), max_clones * n_genes);
        
        wrapper->visit([&](auto& m) {
            clone_phenotype(m, ids_arr, profiles_arr.ptr, *n_clones, max_clones);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_clonality_score(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_real_t* clonality_per_cluster)
{
    if (!clone_ids || !cluster_labels || !clonality_per_cluster) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Index> ids_arr(reinterpret_cast<const Index*>(clone_ids), n_cells);
        Array<const Index> clusters_arr(reinterpret_cast<const Index*>(cluster_labels), n_cells);
        
        // Find number of clusters
        Index n_clusters = 0;
        for (Size i = 0; i < n_cells; ++i) {
            if (cluster_labels[i] >= n_clusters) {
                n_clusters = cluster_labels[i] + 1;
            }
        }
        
        Array<Real> clonality_arr(reinterpret_cast<Real*>(clonality_per_cluster), static_cast<Size>(n_clusters));
        clonality_score(ids_arr, clusters_arr, clonality_arr);
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_repertoire_overlap_morisita(
    const scl_index_t* clone_ids_1,
    scl_size_t n_cells_1,
    const scl_index_t* clone_ids_2,
    scl_size_t n_cells_2,
    scl_real_t* overlap_index)
{
    if (!clone_ids_1 || !clone_ids_2 || !overlap_index) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Index> ids1_arr(reinterpret_cast<const Index*>(clone_ids_1), n_cells_1);
        Array<const Index> ids2_arr(reinterpret_cast<const Index*>(clone_ids_2), n_cells_2);
        *overlap_index = repertoire_overlap_morisita(ids1_arr, ids2_arr);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_diversity_per_cluster(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_real_t* shannon_per_cluster,
    scl_real_t* simpson_per_cluster)
{
    if (!clone_ids || !cluster_labels || !shannon_per_cluster || !simpson_per_cluster) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Index> ids_arr(reinterpret_cast<const Index*>(clone_ids), n_cells);
        Array<const Index> clusters_arr(reinterpret_cast<const Index*>(cluster_labels), n_cells);
        
        // Find number of clusters
        Index n_clusters = 0;
        for (Size i = 0; i < n_cells; ++i) {
            if (cluster_labels[i] >= n_clusters) {
                n_clusters = cluster_labels[i] + 1;
            }
        }
        
        Array<Real> shannon_arr(reinterpret_cast<Real*>(shannon_per_cluster), static_cast<Size>(n_clusters));
        Array<Real> simpson_arr(reinterpret_cast<Real*>(simpson_per_cluster), static_cast<Size>(n_clusters));
        diversity_per_cluster(ids_arr, clusters_arr, shannon_arr, simpson_arr);
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_clone_transition_matrix(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_real_t* transition_matrix,
    scl_size_t n_clusters)
{
    if (!clone_ids || !cluster_labels || !transition_matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Index> ids_arr(reinterpret_cast<const Index*>(clone_ids), n_cells);
        Array<const Index> clusters_arr(reinterpret_cast<const Index*>(cluster_labels), n_cells);
        clone_transition_matrix(ids_arr, clusters_arr, transition_matrix, n_clusters);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_rarefaction_diversity(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_size_t subsample_size,
    scl_size_t n_iterations,
    scl_real_t* mean_diversity,
    scl_real_t* std_diversity,
    uint64_t seed)
{
    if (!clone_ids || !mean_diversity || !std_diversity) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Index> ids_arr(reinterpret_cast<const Index*>(clone_ids), n_cells);
        Real mean, std;
        rarefaction_diversity(ids_arr, subsample_size, n_iterations, mean, std, seed);
        *mean_diversity = mean;
        *std_diversity = std;
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_detect_expanded_clones(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_size_t expansion_threshold,
    scl_index_t* expanded_clones,
    scl_size_t* n_expanded,
    scl_size_t max_expanded)
{
    if (!clone_ids || !expanded_clones || !n_expanded) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Index> ids_arr(reinterpret_cast<const Index*>(clone_ids), n_cells);
        Array<Index> expanded_arr(reinterpret_cast<Index*>(expanded_clones), max_expanded);
        detect_expanded_clones(ids_arr, expansion_threshold, expanded_arr.ptr, *n_expanded, max_expanded);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_clone_size_statistics(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_real_t* mean_size,
    scl_real_t* median_size,
    scl_real_t* max_size,
    scl_size_t* n_singletons,
    scl_size_t* n_clones)
{
    if (!clone_ids || !mean_size || !median_size || !max_size || !n_singletons || !n_clones) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Index> ids_arr(reinterpret_cast<const Index*>(clone_ids), n_cells);
        Real mean, median, max;
        Size singletons, clones;
        clone_size_statistics(ids_arr, mean, median, max, singletons, clones);
        *mean_size = mean;
        *median_size = median;
        *max_size = max;
        *n_singletons = singletons;
        *n_clones = clones;
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

