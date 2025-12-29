// =============================================================================
// FILE: scl/binding/c_api/projection/projection.cpp
// BRIEF: C API implementation for random projection
// =============================================================================

#include "scl/binding/c_api/projection.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/projection.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

scl_error_t scl_projection_project(
    scl_sparse_t matrix,
    scl_size_t output_dim,
    scl_real_t* output,
    scl_projection_type_t type,
    uint64_t seed)
{
    if (!matrix || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (output_dim == 0) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Output dimension must be > 0");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index rows = matrix->rows();
        Size total_size = static_cast<Size>(rows) * static_cast<Size>(output_dim);
        
        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            total_size
        );
        
        scl::kernel::projection::ProjectionType proj_type{};
        switch (type) {
            case SCL_PROJECTION_GAUSSIAN:
                proj_type = scl::kernel::projection::ProjectionType::Gaussian;
                break;
            case SCL_PROJECTION_ACHLIOPTAS:
                proj_type = scl::kernel::projection::ProjectionType::Achlioptas;
                break;
            case SCL_PROJECTION_SPARSE:
                proj_type = scl::kernel::projection::ProjectionType::Sparse;
                break;
            case SCL_PROJECTION_COUNTSKETCH:
                proj_type = scl::kernel::projection::ProjectionType::CountSketch;
                break;
            case SCL_PROJECTION_FEATURE_HASH:
                proj_type = scl::kernel::projection::ProjectionType::FeatureHash;
                break;
            default:
                set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid projection type");
                return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        // All projection methods currently require CSR format
        if (!matrix->is_csr_format()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Projection requires CSR format");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        matrix->visit([&](auto& m) {
            if constexpr (std::remove_reference_t<decltype(m)>::is_csr) {
                scl::kernel::projection::project(
                    m,
                    static_cast<Size>(output_dim),
                    output_arr,
                    proj_type,
                    seed
                );
            }
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_projection_project_auto(
    scl_sparse_t matrix,
    scl_size_t output_dim,
    scl_real_t* output,
    uint64_t seed)
{
    if (!matrix || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (output_dim == 0) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Output dimension must be > 0");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index rows = matrix->rows();
        Size total_size = static_cast<Size>(rows) * static_cast<Size>(output_dim);
        
        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            total_size
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::projection::project_auto(
                m,
                static_cast<Size>(output_dim),
                output_arr,
                seed
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_projection_gaussian(
    scl_sparse_t matrix,
    scl_size_t output_dim,
    scl_real_t* output,
    uint64_t seed)
{
    if (!matrix || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index rows = matrix->rows();
        Size total_size = static_cast<Size>(rows) * static_cast<Size>(output_dim);
        
        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            total_size
        );
        
        if (!matrix->is_csr_format()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "project_gaussian requires CSR format");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        matrix->visit([&](auto& m) {
            if constexpr (std::remove_reference_t<decltype(m)>::is_csr) {
                scl::kernel::projection::project_gaussian_otf(
                    m,
                    static_cast<Size>(output_dim),
                    output_arr,
                    seed
                );
            }
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_projection_achlioptas(
    scl_sparse_t matrix,
    scl_size_t output_dim,
    scl_real_t* output,
    uint64_t seed)
{
    if (!matrix || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index rows = matrix->rows();
        Size total_size = static_cast<Size>(rows) * static_cast<Size>(output_dim);
        
        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            total_size
        );
        
        if (!matrix->is_csr_format()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "project_achlioptas requires CSR format");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        matrix->visit([&](auto& m) {
            if constexpr (std::remove_reference_t<decltype(m)>::is_csr) {
                scl::kernel::projection::project_achlioptas_otf(
                    m,
                    static_cast<Size>(output_dim),
                    output_arr,
                    seed
                );
            }
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_projection_sparse(
    scl_sparse_t matrix,
    scl_size_t output_dim,
    scl_real_t* output,
    scl_real_t density,
    uint64_t seed)
{
    if (!matrix || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (density <= 0.0 || density > 1.0) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Density must be in (0, 1]");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index rows = matrix->rows();
        Size total_size = static_cast<Size>(rows) * static_cast<Size>(output_dim);
        
        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            total_size
        );
        
        if (!matrix->is_csr_format()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "project_sparse requires CSR format");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        matrix->visit([&](auto& m) {
            if constexpr (std::remove_reference_t<decltype(m)>::is_csr) {
                scl::kernel::projection::project_sparse_otf(
                    m,
                    static_cast<Size>(output_dim),
                    output_arr,
                    static_cast<Real>(density),
                    seed
                );
            }
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_projection_countsketch(
    scl_sparse_t matrix,
    scl_size_t output_dim,
    scl_real_t* output,
    uint64_t seed)
{
    if (!matrix || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index rows = matrix->rows();
        Size total_size = static_cast<Size>(rows) * static_cast<Size>(output_dim);
        
        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            total_size
        );
        
        if (!matrix->is_csr_format()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "project_countsketch requires CSR format");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        matrix->visit([&](auto& m) {
            if constexpr (std::remove_reference_t<decltype(m)>::is_csr) {
                scl::kernel::projection::project_countsketch(
                    m,
                    static_cast<Size>(output_dim),
                    output_arr,
                    seed
                );
            }
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_projection_feature_hash(
    scl_sparse_t matrix,
    scl_size_t output_dim,
    scl_real_t* output,
    scl_size_t n_hashes,
    uint64_t seed)
{
    if (!matrix || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (n_hashes == 0) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Number of hashes must be > 0");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index rows = matrix->rows();
        Size total_size = static_cast<Size>(rows) * static_cast<Size>(output_dim);
        
        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            total_size
        );
        
        if (!matrix->is_csr_format()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "project_feature_hash requires CSR format");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        matrix->visit([&](auto& m) {
            if constexpr (std::remove_reference_t<decltype(m)>::is_csr) {
                scl::kernel::projection::project_feature_hash(
                    m,
                    static_cast<Size>(output_dim),
                    output_arr,
                    static_cast<Size>(n_hashes),
                    seed
                );
            }
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_size_t scl_projection_jl_dimension(
    scl_size_t n_samples,
    scl_real_t epsilon)
{
    if (n_samples == 0 || epsilon <= 0.0) {
        return 0;
    }
    
    return static_cast<scl_size_t>(
        scl::kernel::projection::compute_jl_dimension(
            static_cast<Size>(n_samples),
            static_cast<Real>(epsilon)
        )
    );
}

} // extern "C"

