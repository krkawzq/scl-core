// =============================================================================
// FILE: scl/binding/c_api/velocity.cpp
// BRIEF: C API implementation for RNA velocity analysis
// =============================================================================

#include "scl/binding/c_api/velocity.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/velocity.hpp"
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

static scl::kernel::velocity::VelocityModel convert_velocity_model(scl_velocity_model_t model) {
    switch (model) {
        case SCL_VELOCITY_STEADY_STATE: return scl::kernel::velocity::VelocityModel::SteadyState;
        case SCL_VELOCITY_DYNAMICAL: return scl::kernel::velocity::VelocityModel::Dynamical;
        case SCL_VELOCITY_STOCHASTIC: return scl::kernel::velocity::VelocityModel::Stochastic;
        default: return scl::kernel::velocity::VelocityModel::SteadyState;
    }
}

scl_error_t scl_velocity_fit_kinetics(
    scl_sparse_matrix_t spliced,
    scl_sparse_matrix_t unspliced,
    scl_real_t* gamma,
    scl_real_t* r2,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_velocity_model_t model
) {
    if (!spliced || !unspliced || !gamma || !r2) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* s_sparse = static_cast<const scl::CSR*>(spliced);
        const auto* u_sparse = static_cast<const scl::CSR*>(unspliced);
        scl::Array<scl::Real> gamma_arr(
            reinterpret_cast<scl::Real*>(gamma),
            static_cast<scl::Size>(n_genes)
        );
        scl::Array<scl::Real> r2_arr(
            reinterpret_cast<scl::Real*>(r2),
            static_cast<scl::Size>(n_genes)
        );
        scl::kernel::velocity::fit_kinetics(
            *s_sparse,
            *u_sparse,
            gamma_arr,
            r2_arr,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            convert_velocity_model(model)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_velocity_compute_velocity(
    scl_sparse_matrix_t spliced,
    scl_sparse_matrix_t unspliced,
    const scl_real_t* gamma,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* velocity_out
) {
    if (!spliced || !unspliced || !gamma || !velocity_out) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* s_sparse = static_cast<const scl::CSR*>(spliced);
        const auto* u_sparse = static_cast<const scl::CSR*>(unspliced);
        scl::Array<const scl::Real> gamma_arr(
            reinterpret_cast<const scl::Real*>(gamma),
            static_cast<scl::Size>(n_genes)
        );
        scl::kernel::velocity::compute_velocity(
            *s_sparse,
            *u_sparse,
            gamma_arr,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            reinterpret_cast<scl::Real*>(velocity_out)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_velocity_splice_ratio(
    scl_sparse_matrix_t spliced,
    scl_sparse_matrix_t unspliced,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* ratio_out
) {
    if (!spliced || !unspliced || !ratio_out) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* s_sparse = static_cast<const scl::CSR*>(spliced);
        const auto* u_sparse = static_cast<const scl::CSR*>(unspliced);
        scl::kernel::velocity::splice_ratio(
            *s_sparse,
            *u_sparse,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            reinterpret_cast<scl::Real*>(ratio_out)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_velocity_velocity_graph(
    const scl_real_t* velocity,
    const scl_real_t* expression,
    scl_sparse_matrix_t knn,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* transition_probs,
    scl_index_t k_neighbors
) {
    if (!velocity || !expression || !knn || !transition_probs) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* knn_sparse = static_cast<const scl::CSR*>(knn);
        scl::kernel::velocity::velocity_graph(
            reinterpret_cast<const scl::Real*>(velocity),
            reinterpret_cast<const scl::Real*>(expression),
            *knn_sparse,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            reinterpret_cast<scl::Real*>(transition_probs),
            static_cast<scl::Index>(k_neighbors)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::velocity::fit_kinetics<scl::Real, true>(
    const scl::CSR&, const scl::CSR&, scl::Array<scl::Real>, scl::Array<scl::Real>, scl::Index, scl::Index, scl::kernel::velocity::VelocityModel);
template void scl::kernel::velocity::compute_velocity<scl::Real, true>(
    const scl::CSR&, const scl::CSR&, scl::Array<const scl::Real>, scl::Index, scl::Index, scl::Real*);
template void scl::kernel::velocity::splice_ratio<scl::Real, true>(
    const scl::CSR&, const scl::CSR&, scl::Index, scl::Index, scl::Real*);
template void scl::kernel::velocity::velocity_graph<scl::Real, true>(
    const scl::Real*, const scl::Real*, const scl::CSR&, scl::Index, scl::Index, scl::Real*, scl::Index);

} // extern "C"
