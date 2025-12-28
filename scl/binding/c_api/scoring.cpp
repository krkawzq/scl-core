// =============================================================================
// FILE: scl/binding/c_api/scoring.cpp
// BRIEF: C API implementation for gene set scoring
// =============================================================================

#include "scl/binding/c_api/scoring.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/scoring.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

static scl_error_t handle_exception() {
    try {
        throw;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::TypeError&) {
        return SCL_ERROR_TYPE;
    } catch (const scl::IOError&) {
        return SCL_ERROR_IO;
    } catch (const scl::NotImplementedError&) {
        return SCL_ERROR_NOT_IMPLEMENTED;
    } catch (const scl::InternalError&) {
        return SCL_ERROR_INTERNAL;
    } catch (const scl::Exception& e) {
        return static_cast<scl_error_t>(static_cast<int>(e.code()));
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_scoring_compute_gene_means(
    scl_sparse_matrix_t matrix,
    scl_real_t* out_means,
    scl_size_t n_genes
) {
    if (!matrix || !out_means) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<scl::Real> means_arr(
            reinterpret_cast<scl::Real*>(out_means),
            n_genes
        );
        scl::kernel::scoring::compute_gene_means(*sparse, means_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_scoring_mean_score(
    scl_sparse_matrix_t matrix,
    const scl_index_t* gene_set,
    scl_size_t n_genes_in_set,
    scl_real_t* scores,
    scl_size_t n_cells
) {
    if (!matrix || !gene_set || !scores) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Index> gene_set_arr(
            reinterpret_cast<const scl::Index*>(gene_set),
            n_genes_in_set
        );
        scl::Array<scl::Real> scores_arr(
            reinterpret_cast<scl::Real*>(scores),
            n_cells
        );
        scl::kernel::scoring::mean_score(*sparse, gene_set_arr, scores_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_scoring_weighted_score(
    scl_sparse_matrix_t matrix,
    const scl_index_t* gene_set,
    const scl_real_t* weights,
    scl_size_t n_genes_in_set,
    scl_real_t* scores,
    scl_size_t n_cells
) {
    if (!matrix || !gene_set || !weights || !scores) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Index> gene_set_arr(
            reinterpret_cast<const scl::Index*>(gene_set),
            n_genes_in_set
        );
        scl::Array<const scl::Real> weights_arr(
            reinterpret_cast<const scl::Real*>(weights),
            n_genes_in_set
        );
        scl::Array<scl::Real> scores_arr(
            reinterpret_cast<scl::Real*>(scores),
            n_cells
        );
        scl::kernel::scoring::weighted_score(*sparse, gene_set_arr, weights_arr, scores_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_scoring_auc_score(
    scl_sparse_matrix_t matrix,
    const scl_index_t* gene_set,
    scl_size_t n_genes_in_set,
    scl_real_t* scores,
    scl_real_t quantile,
    scl_size_t n_cells,
    scl_size_t n_genes
) {
    if (!matrix || !gene_set || !scores) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Index> gene_set_arr(
            reinterpret_cast<const scl::Index*>(gene_set),
            n_genes_in_set
        );
        scl::Array<scl::Real> scores_arr(
            reinterpret_cast<scl::Real*>(scores),
            n_cells
        );
        scl::kernel::scoring::auc_score(
            *sparse,
            gene_set_arr,
            scores_arr,
            static_cast<scl::Real>(quantile),
            static_cast<scl::Index>(n_genes)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_scoring_module_score(
    scl_sparse_matrix_t matrix,
    const scl_index_t* gene_set,
    scl_size_t n_genes_in_set,
    scl_real_t* scores,
    scl_index_t n_control,
    scl_real_t quantile,
    scl_size_t n_cells,
    scl_size_t n_genes,
    uint64_t seed
) {
    if (!matrix || !gene_set || !scores) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Index> gene_set_arr(
            reinterpret_cast<const scl::Index*>(gene_set),
            n_genes_in_set
        );
        scl::Array<scl::Real> scores_arr(
            reinterpret_cast<scl::Real*>(scores),
            n_cells
        );
        scl::kernel::scoring::module_score(
            *sparse,
            gene_set_arr,
            scores_arr,
            static_cast<scl::Index>(n_control),
            static_cast<scl::Real>(quantile),
            static_cast<scl::Index>(n_genes),
            seed
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::scoring::compute_gene_means<scl::Real, true>(
    const scl::CSR&,
    scl::Array<scl::Real>
);

template void scl::kernel::scoring::mean_score<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    scl::Array<scl::Real>
);

template void scl::kernel::scoring::weighted_score<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    scl::Array<const scl::Real>,
    scl::Array<scl::Real>
);

template void scl::kernel::scoring::auc_score<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    scl::Array<scl::Real>,
    scl::Real,
    scl::Index
);

template void scl::kernel::scoring::module_score<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    scl::Array<scl::Real>,
    scl::Index,
    scl::Real,
    scl::Index,
    uint64_t
);

} // extern "C"

