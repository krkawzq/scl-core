// =============================================================================
// FILE: scl/binding/c_api/coexpression.cpp
// BRIEF: C API implementation for co-expression module detection
// =============================================================================

#include "scl/binding/c_api/coexpression.h"
#include "scl/kernel/coexpression.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

extern "C" {

static scl::kernel::coexpression::CorrelationType convert_correlation_type(scl_correlation_type_t type) {
    switch (type) {
        case SCL_CORRELATION_PEARSON:
            return scl::kernel::coexpression::CorrelationType::Pearson;
        case SCL_CORRELATION_SPEARMAN:
            return scl::kernel::coexpression::CorrelationType::Spearman;
        case SCL_CORRELATION_BICOR:
            return scl::kernel::coexpression::CorrelationType::Bicor;
        default:
            return scl::kernel::coexpression::CorrelationType::Pearson;
    }
}

static scl::kernel::coexpression::AdjacencyType convert_adjacency_type(scl_adjacency_type_t type) {
    switch (type) {
        case SCL_ADJACENCY_UNSIGNED:
            return scl::kernel::coexpression::AdjacencyType::Unsigned;
        case SCL_ADJACENCY_SIGNED:
            return scl::kernel::coexpression::AdjacencyType::Signed;
        case SCL_ADJACENCY_SIGNED_HYBRID:
            return scl::kernel::coexpression::AdjacencyType::SignedHybrid;
        default:
            return scl::kernel::coexpression::AdjacencyType::Unsigned;
    }
}

scl_error_t scl_correlation_matrix(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* corr_matrix,
    scl_correlation_type_t corr_type
) {
    try {
        if (!expression || !corr_matrix) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(expression);
        auto cpp_corr_type = convert_correlation_type(corr_type);

        scl::kernel::coexpression::correlation_matrix(
            *sparse,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            reinterpret_cast<scl::Real*>(corr_matrix),
            cpp_corr_type
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

scl_error_t scl_wgcna_adjacency(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t power,
    scl_real_t* adjacency,
    scl_correlation_type_t corr_type,
    scl_adjacency_type_t adj_type
) {
    try {
        if (!expression || !adjacency) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(expression);
        auto cpp_corr_type = convert_correlation_type(corr_type);
        auto cpp_adj_type = convert_adjacency_type(adj_type);

        scl::kernel::coexpression::wgcna_adjacency(
            *sparse,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Real>(power),
            reinterpret_cast<scl::Real*>(adjacency),
            cpp_corr_type,
            cpp_adj_type
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

scl_error_t scl_topological_overlap_matrix(
    const scl_real_t* adjacency,
    scl_index_t n_genes,
    scl_real_t* tom
) {
    try {
        if (!adjacency || !tom) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::coexpression::topological_overlap_matrix(
            reinterpret_cast<const scl::Real*>(adjacency),
            static_cast<scl::Index>(n_genes),
            reinterpret_cast<scl::Real*>(tom)
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

scl_error_t scl_tom_dissimilarity(
    const scl_real_t* tom,
    scl_index_t n_genes,
    scl_real_t* dissim
) {
    try {
        if (!tom || !dissim) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::coexpression::tom_dissimilarity(
            reinterpret_cast<const scl::Real*>(tom),
            static_cast<scl::Index>(n_genes),
            reinterpret_cast<scl::Real*>(dissim)
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

scl_error_t scl_detect_modules(
    const scl_real_t* dissim,
    scl_index_t n_genes,
    scl_index_t* module_labels,
    scl_index_t* n_modules,
    scl_index_t min_module_size,
    scl_real_t merge_cut_height
) {
    try {
        if (!dissim || !module_labels || !n_modules) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Index n_mods = scl::kernel::coexpression::detect_modules(
            reinterpret_cast<const scl::Real*>(dissim),
            static_cast<scl::Index>(n_genes),
            reinterpret_cast<scl::Index*>(module_labels),
            static_cast<scl::Index>(min_module_size),
            static_cast<scl::Real>(merge_cut_height)
        );

        *n_modules = n_mods;
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

scl_error_t scl_module_eigengene(
    scl_sparse_matrix_t expression,
    const scl_index_t* module_labels,
    scl_index_t module_id,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* eigengene
) {
    try {
        if (!expression || !module_labels || !eigengene) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(expression);
        const scl::Size N = static_cast<scl::Size>(n_cells);

        scl::Array<const scl::Index> labels_arr(
            reinterpret_cast<const scl::Index*>(module_labels),
            static_cast<scl::Size>(n_genes)
        );
        scl::Array<scl::Real> eigengene_arr(
            reinterpret_cast<scl::Real*>(eigengene),
            N
        );

        scl::kernel::coexpression::module_eigengene(
            *sparse,
            labels_arr,
            static_cast<scl::Index>(module_id),
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            eigengene_arr
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

scl_error_t scl_all_module_eigengenes(
    scl_sparse_matrix_t expression,
    const scl_index_t* module_labels,
    scl_index_t n_modules,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* eigengenes
) {
    try {
        if (!expression || !module_labels || !eigengenes) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(expression);

        scl::Array<const scl::Index> labels_arr(
            reinterpret_cast<const scl::Index*>(module_labels),
            static_cast<scl::Size>(n_genes)
        );

        scl::kernel::coexpression::all_module_eigengenes(
            *sparse,
            labels_arr,
            static_cast<scl::Index>(n_modules),
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            reinterpret_cast<scl::Real*>(eigengenes)
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

scl_error_t scl_module_trait_correlation(
    const scl_real_t* eigengenes,
    const scl_real_t* traits,
    scl_index_t n_samples,
    scl_index_t n_modules,
    scl_index_t n_traits,
    scl_real_t* correlations,
    scl_real_t* p_values
) {
    try {
        if (!eigengenes || !traits || !correlations) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::coexpression::module_trait_correlation(
            reinterpret_cast<const scl::Real*>(eigengenes),
            reinterpret_cast<const scl::Real*>(traits),
            static_cast<scl::Index>(n_samples),
            static_cast<scl::Index>(n_modules),
            static_cast<scl::Index>(n_traits),
            reinterpret_cast<scl::Real*>(correlations),
            p_values ? reinterpret_cast<scl::Real*>(p_values) : nullptr
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

scl_error_t scl_pick_soft_threshold(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    const scl_real_t* powers_to_test,
    scl_index_t n_powers,
    scl_real_t* scale_free_fits,
    scl_real_t* mean_connectivity,
    scl_real_t* best_power,
    scl_correlation_type_t corr_type
) {
    try {
        if (!expression || !powers_to_test || !scale_free_fits || !mean_connectivity || !best_power) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(expression);
        auto cpp_corr_type = convert_correlation_type(corr_type);

        scl::Real best = scl::kernel::coexpression::pick_soft_threshold(
            *sparse,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            reinterpret_cast<scl::Real*>(const_cast<scl_real_t*>(powers_to_test)),
            static_cast<scl::Index>(n_powers),
            reinterpret_cast<scl::Real*>(scale_free_fits),
            reinterpret_cast<scl::Real*>(mean_connectivity),
            cpp_corr_type
        );

        *best_power = best;
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

scl_error_t scl_blockwise_modules(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t block_size,
    scl_real_t power,
    scl_index_t min_module_size,
    scl_index_t* module_labels,
    scl_index_t* n_modules,
    scl_correlation_type_t corr_type
) {
    try {
        if (!expression || !module_labels || !n_modules) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(expression);
        auto cpp_corr_type = convert_correlation_type(corr_type);

        scl::Index n_mods = 0;
        scl::kernel::coexpression::blockwise_modules(
            *sparse,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Index>(block_size),
            static_cast<scl::Real>(power),
            static_cast<scl::Index>(min_module_size),
            reinterpret_cast<scl::Index*>(module_labels),
            n_mods,
            cpp_corr_type
        );

        *n_modules = n_mods;
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

