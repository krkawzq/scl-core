// =============================================================================
// FILE: scl/binding/c_api/clonotype.cpp
// BRIEF: C API implementation for TCR/BCR clonal analysis
// =============================================================================

#include "scl/binding/c_api/clonotype.h"
#include "scl/kernel/clonotype.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

extern "C" {

scl_error_t scl_clone_size_distribution(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_size_t* clone_sizes,
    scl_size_t* n_clones,
    scl_size_t max_clones
) {
    try {
        if (!clone_ids || !clone_sizes || !n_clones) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> clone_arr(
            reinterpret_cast<const scl::Index*>(clone_ids),
            n_cells
        );

        scl::Size n_clones_result = 0;
        scl::kernel::clonotype::clone_size_distribution(
            clone_arr,
            reinterpret_cast<scl::Size*>(clone_sizes),
            n_clones_result,
            max_clones
        );

        *n_clones = n_clones_result;
        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_clonal_diversity(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_real_t* shannon_diversity,
    scl_real_t* simpson_diversity,
    scl_real_t* gini_index
) {
    try {
        if (!clone_ids || !shannon_diversity || !simpson_diversity || !gini_index) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> clone_arr(
            reinterpret_cast<const scl::Index*>(clone_ids),
            n_cells
        );

        scl::Real shannon, simpson, gini;
        scl::kernel::clonotype::clonal_diversity(clone_arr, shannon, simpson, gini);

        *shannon_diversity = shannon;
        *simpson_diversity = simpson;
        *gini_index = gini;

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_clone_dynamics(
    const scl_index_t* clone_ids_t1,
    scl_size_t n_cells_t1,
    const scl_index_t* clone_ids_t2,
    scl_size_t n_cells_t2,
    scl_real_t* expansion_rates,
    scl_size_t* n_clones,
    scl_size_t max_clones
) {
    try {
        if (!clone_ids_t1 || !clone_ids_t2 || !expansion_rates || !n_clones) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> t1_arr(
            reinterpret_cast<const scl::Index*>(clone_ids_t1),
            n_cells_t1
        );
        scl::Array<const scl::Index> t2_arr(
            reinterpret_cast<const scl::Index*>(clone_ids_t2),
            n_cells_t2
        );

        scl::Size n_clones_result = 0;
        scl::kernel::clonotype::clone_dynamics(
            t1_arr,
            t2_arr,
            reinterpret_cast<scl::Real*>(expansion_rates),
            n_clones_result,
            max_clones
        );

        *n_clones = n_clones_result;
        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_shared_clonotypes(
    const scl_index_t* clone_ids_sample1,
    scl_size_t n_cells_sample1,
    const scl_index_t* clone_ids_sample2,
    scl_size_t n_cells_sample2,
    scl_index_t* shared_clones,
    scl_size_t* n_shared,
    scl_real_t* jaccard_index,
    scl_size_t max_shared
) {
    try {
        if (!clone_ids_sample1 || !clone_ids_sample2 || !shared_clones || !n_shared || !jaccard_index) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> s1_arr(
            reinterpret_cast<const scl::Index*>(clone_ids_sample1),
            n_cells_sample1
        );
        scl::Array<const scl::Index> s2_arr(
            reinterpret_cast<const scl::Index*>(clone_ids_sample2),
            n_cells_sample2
        );

        scl::Size n_shared_result = 0;
        scl::Real jaccard;
        scl::kernel::clonotype::shared_clonotypes(
            s1_arr,
            s2_arr,
            reinterpret_cast<scl::Index*>(shared_clones),
            n_shared_result,
            jaccard,
            max_shared
        );

        *n_shared = n_shared_result;
        *jaccard_index = jaccard;

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_clone_phenotype(
    scl_sparse_matrix_t expression,
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_real_t* clone_profiles,
    scl_size_t* n_clones,
    scl_size_t max_clones
) {
    try {
        if (!expression || !clone_ids || !clone_profiles || !n_clones) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(expression);
        const scl::Index n_genes = sparse->cols();

        scl::Array<const scl::Index> clone_arr(
            reinterpret_cast<const scl::Index*>(clone_ids),
            n_cells
        );

        scl::Size n_clones_result = 0;
        scl::kernel::clonotype::clone_phenotype(
            *sparse,
            clone_arr,
            reinterpret_cast<scl::Real*>(clone_profiles),
            n_clones_result,
            max_clones
        );

        *n_clones = n_clones_result;
        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_clonality_score(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_real_t* clonality_per_cluster,
    scl_size_t n_clusters
) {
    try {
        if (!clone_ids || !cluster_labels || !clonality_per_cluster) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> clone_arr(
            reinterpret_cast<const scl::Index*>(clone_ids),
            n_cells
        );
        scl::Array<const scl::Index> cluster_arr(
            reinterpret_cast<const scl::Index*>(cluster_labels),
            n_cells
        );
        scl::Array<scl::Real> clonality_arr(
            reinterpret_cast<scl::Real*>(clonality_per_cluster),
            n_clusters
        );

        scl::kernel::clonotype::clonality_score(clone_arr, cluster_arr, clonality_arr);

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_repertoire_overlap_morisita(
    const scl_index_t* clone_ids_1,
    scl_size_t n_cells_1,
    const scl_index_t* clone_ids_2,
    scl_size_t n_cells_2,
    scl_real_t* overlap_index
) {
    try {
        if (!clone_ids_1 || !clone_ids_2 || !overlap_index) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> arr1(
            reinterpret_cast<const scl::Index*>(clone_ids_1),
            n_cells_1
        );
        scl::Array<const scl::Index> arr2(
            reinterpret_cast<const scl::Index*>(clone_ids_2),
            n_cells_2
        );

        scl::Real overlap = scl::kernel::clonotype::repertoire_overlap_morisita(arr1, arr2);
        *overlap_index = overlap;

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_diversity_per_cluster(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_real_t* shannon_per_cluster,
    scl_real_t* simpson_per_cluster,
    scl_size_t n_clusters
) {
    try {
        if (!clone_ids || !cluster_labels || !shannon_per_cluster || !simpson_per_cluster) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> clone_arr(
            reinterpret_cast<const scl::Index*>(clone_ids),
            n_cells
        );
        scl::Array<const scl::Index> cluster_arr(
            reinterpret_cast<const scl::Index*>(cluster_labels),
            n_cells
        );
        scl::Array<scl::Real> shannon_arr(
            reinterpret_cast<scl::Real*>(shannon_per_cluster),
            n_clusters
        );
        scl::Array<scl::Real> simpson_arr(
            reinterpret_cast<scl::Real*>(simpson_per_cluster),
            n_clusters
        );

        scl::kernel::clonotype::diversity_per_cluster(
            clone_arr,
            cluster_arr,
            shannon_arr,
            simpson_arr
        );

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_clone_transition_matrix(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_real_t* transition_matrix,
    scl_size_t n_clusters
) {
    try {
        if (!clone_ids || !cluster_labels || !transition_matrix) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> clone_arr(
            reinterpret_cast<const scl::Index*>(clone_ids),
            n_cells
        );
        scl::Array<const scl::Index> cluster_arr(
            reinterpret_cast<const scl::Index*>(cluster_labels),
            n_cells
        );

        scl::kernel::clonotype::clone_transition_matrix(
            clone_arr,
            cluster_arr,
            reinterpret_cast<scl::Real*>(transition_matrix),
            n_clusters
        );

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_rarefaction_diversity(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_size_t subsample_size,
    scl_size_t n_iterations,
    scl_real_t* mean_diversity,
    scl_real_t* std_diversity,
    uint64_t seed
) {
    try {
        if (!clone_ids || !mean_diversity || !std_diversity) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> clone_arr(
            reinterpret_cast<const scl::Index*>(clone_ids),
            n_cells
        );

        scl::Real mean, std;
        scl::kernel::clonotype::rarefaction_diversity(
            clone_arr,
            subsample_size,
            n_iterations,
            mean,
            std,
            seed
        );

        *mean_diversity = mean;
        *std_diversity = std;

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_detect_expanded_clones(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_size_t expansion_threshold,
    scl_index_t* expanded_clones,
    scl_size_t* n_expanded,
    scl_size_t max_expanded
) {
    try {
        if (!clone_ids || !expanded_clones || !n_expanded) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> clone_arr(
            reinterpret_cast<const scl::Index*>(clone_ids),
            n_cells
        );

        scl::Size n_expanded_result = 0;
        scl::kernel::clonotype::detect_expanded_clones(
            clone_arr,
            expansion_threshold,
            reinterpret_cast<scl::Index*>(expanded_clones),
            n_expanded_result,
            max_expanded
        );

        *n_expanded = n_expanded_result;
        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_clone_size_statistics(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_real_t* mean_size,
    scl_real_t* median_size,
    scl_real_t* max_size,
    scl_size_t* n_singletons,
    scl_size_t* n_clones
) {
    try {
        if (!clone_ids || !mean_size || !median_size || !max_size || !n_singletons || !n_clones) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> clone_arr(
            reinterpret_cast<const scl::Index*>(clone_ids),
            n_cells
        );

        scl::Real mean, median, max;
        scl::Size n_sing, n_cl;
        scl::kernel::clonotype::clone_size_statistics(
            clone_arr,
            mean,
            median,
            max,
            n_sing,
            n_cl
        );

        *mean_size = mean;
        *median_size = median;
        *max_size = max;
        *n_singletons = n_sing;
        *n_clones = n_cl;

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

} // extern "C"

