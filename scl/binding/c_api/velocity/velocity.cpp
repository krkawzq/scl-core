// =============================================================================
// FILE: scl/binding/c_api/velocity/velocity.cpp
// BRIEF: C API implementation for RNA velocity analysis
// =============================================================================

#include "scl/binding/c_api/velocity/velocity.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/velocity.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

scl_error_t scl_velocity_fit_kinetics(
    scl_sparse_t spliced,
    scl_sparse_t unspliced,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* gamma,
    scl_real_t* r2,
    scl_velocity_model_t model)
{
    if (!spliced || !unspliced || !gamma || !r2) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (n_cells <= 0 || n_genes <= 0) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid dimensions");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        if (!spliced->valid() || !unspliced->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<Real> gamma_arr(
            reinterpret_cast<Real*>(gamma),
            static_cast<Size>(n_genes)
        );
        Array<Real> r2_arr(
            reinterpret_cast<Real*>(r2),
            static_cast<Size>(n_genes)
        );
        
        scl::kernel::velocity::VelocityModel model_enum;
        switch (model) {
            case SCL_VELOCITY_STEADY_STATE:
                model_enum = scl::kernel::velocity::VelocityModel::SteadyState;
                break;
            case SCL_VELOCITY_DYNAMICAL:
                model_enum = scl::kernel::velocity::VelocityModel::Dynamical;
                break;
            case SCL_VELOCITY_STOCHASTIC:
                model_enum = scl::kernel::velocity::VelocityModel::Stochastic;
                break;
            default:
                set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid velocity model");
                return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        spliced->visit([&](auto& s) {
            unspliced->visit([&](auto& u) {
                scl::kernel::velocity::fit_gene_kinetics(
                    s, u,
                    n_cells,
                    n_genes,
                    gamma_arr,
                    r2_arr,
                    model_enum
                );
            });
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_velocity_compute(
    scl_sparse_t spliced,
    scl_sparse_t unspliced,
    const scl_real_t* gamma,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* velocity_out)
{
    if (!spliced || !unspliced || !gamma || !velocity_out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (n_cells <= 0 || n_genes <= 0) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid dimensions");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        if (!spliced->valid() || !unspliced->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<const Real> gamma_arr(
            reinterpret_cast<const Real*>(gamma),
            static_cast<Size>(n_genes)
        );
        
        spliced->visit([&](auto& s) {
            unspliced->visit([&](auto& u) {
                scl::kernel::velocity::compute_velocity(
                    s, u,
                    gamma_arr,
                    n_cells,
                    n_genes,
                    reinterpret_cast<Real*>(velocity_out)
                );
            });
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_velocity_splice_ratio(
    scl_sparse_t spliced,
    scl_sparse_t unspliced,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* ratio_out)
{
    if (!spliced || !unspliced || !ratio_out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (n_cells <= 0 || n_genes <= 0) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid dimensions");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        if (!spliced->valid() || !unspliced->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        spliced->visit([&](auto& s) {
            unspliced->visit([&](auto& u) {
                scl::kernel::velocity::splice_ratio(
                    s, u,
                    n_cells,
                    n_genes,
                    reinterpret_cast<Real*>(ratio_out)
                );
            });
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_velocity_graph(
    const scl_real_t* velocity,
    const scl_real_t* expression,
    scl_sparse_t knn,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* transition_probs,
    scl_index_t k_neighbors)
{
    if (!velocity || !expression || !knn || !transition_probs) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (n_cells <= 0 || n_genes <= 0 || k_neighbors <= 0) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid dimensions");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        if (!knn->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        knn->visit([&](auto& k) {
            scl::kernel::velocity::velocity_graph(
                reinterpret_cast<const Real*>(velocity),
                reinterpret_cast<const Real*>(expression),
                k,
                n_cells,
                n_genes,
                reinterpret_cast<Real*>(transition_probs),
                k_neighbors
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

