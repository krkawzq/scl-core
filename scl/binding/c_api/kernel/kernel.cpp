// =============================================================================
// FILE: scl/binding/c_api/kernel/kernel.cpp
// BRIEF: C API implementation for kernel methods
// =============================================================================

#include "scl/binding/c_api/kernel/kernel.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/kernel.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::kernel;

extern "C" {

// =============================================================================
// Helper: Convert kernel type
// =============================================================================

static KernelType convert_kernel_type(scl_kernel_type_t type) {
    switch (type) {
        case SCL_KERNEL_GAUSSIAN: return KernelType::Gaussian;
        case SCL_KERNEL_EPANECHNIKOV: return KernelType::Epanechnikov;
        case SCL_KERNEL_COSINE: return KernelType::Cosine;
        case SCL_KERNEL_LINEAR: return KernelType::Linear;
        case SCL_KERNEL_POLYNOMIAL: return KernelType::Polynomial;
        case SCL_KERNEL_LAPLACIAN: return KernelType::Laplacian;
        case SCL_KERNEL_CAUCHY: return KernelType::Cauchy;
        case SCL_KERNEL_SIGMOID: return KernelType::Sigmoid;
        case SCL_KERNEL_UNIFORM: return KernelType::Uniform;
        case SCL_KERNEL_TRIANGULAR: return KernelType::Triangular;
        default: return KernelType::Gaussian;
    }
}

// =============================================================================
// KDE from Distances
// =============================================================================

scl_error_t scl_kernel_kde_from_distances(
    scl_sparse_t distances,
    scl_real_t* density,
    scl_size_t n_points,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type)
{
    if (!distances || !density) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(distances);
        
        Index n = wrapper->rows();
        if (static_cast<scl_size_t>(n) != n_points) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Point count mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<Real> density_arr(
            reinterpret_cast<Real*>(density),
            n_points
        );

        wrapper->visit([&](auto& dist) {
            kde_from_distances(
                dist, density_arr,
                static_cast<Real>(bandwidth),
                convert_kernel_type(kernel_type)
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Local Bandwidth
// =============================================================================

scl_error_t scl_kernel_local_bandwidth(
    scl_sparse_t distances,
    scl_real_t* bandwidths,
    scl_size_t n_points,
    scl_index_t k)
{
    if (!distances || !bandwidths) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(distances);
        
        Index n = wrapper->rows();
        if (static_cast<scl_size_t>(n) != n_points) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Point count mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<Real> bandwidths_arr(
            reinterpret_cast<Real*>(bandwidths),
            n_points
        );

        wrapper->visit([&](auto& dist) {
            local_bandwidth(dist, bandwidths_arr, k);
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Adaptive KDE
// =============================================================================

scl_error_t scl_kernel_adaptive_kde(
    scl_sparse_t distances,
    scl_real_t* density,
    scl_size_t n_points,
    const scl_real_t* bandwidths,
    scl_kernel_type_t kernel_type)
{
    if (!distances || !density || !bandwidths) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(distances);
        
        Index n = wrapper->rows();
        if (static_cast<scl_size_t>(n) != n_points) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Point count mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<Real> density_arr(
            reinterpret_cast<Real*>(density),
            n_points
        );
        Array<const Real> bandwidths_arr(
            reinterpret_cast<const Real*>(bandwidths),
            n_points
        );

        wrapper->visit([&](auto& dist) {
            adaptive_kde(
                dist, density_arr, bandwidths_arr,
                convert_kernel_type(kernel_type)
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Compute Kernel Matrix
// =============================================================================

scl_error_t scl_kernel_compute_matrix(
    scl_sparse_t distances,
    scl_real_t* kernel_values,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type)
{
    if (!distances || !kernel_values) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(distances);
        
        wrapper->visit([&](auto& dist) {
            compute_kernel_matrix(
                dist,
                reinterpret_cast<Real*>(kernel_values),
                static_cast<Real>(bandwidth),
                convert_kernel_type(kernel_type)
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Kernel Row Sums
// =============================================================================

scl_error_t scl_kernel_row_sums(
    scl_sparse_t distances,
    scl_real_t* sums,
    scl_size_t n_points,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type)
{
    if (!distances || !sums) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(distances);
        
        Index n = wrapper->rows();
        if (static_cast<scl_size_t>(n) != n_points) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Point count mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<Real> sums_arr(
            reinterpret_cast<Real*>(sums),
            n_points
        );

        wrapper->visit([&](auto& dist) {
            kernel_row_sums(
                dist, sums_arr,
                static_cast<Real>(bandwidth),
                convert_kernel_type(kernel_type)
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Nadaraya-Watson Estimator
// =============================================================================

scl_error_t scl_kernel_nadaraya_watson(
    scl_sparse_t distances,
    const scl_real_t* y_values,
    scl_real_t* predictions,
    scl_size_t n_points,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type)
{
    if (!distances || !y_values || !predictions) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(distances);
        
        Index n = wrapper->rows();
        if (static_cast<scl_size_t>(n) != n_points) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Point count mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<const Real> y_arr(
            reinterpret_cast<const Real*>(y_values),
            n_points
        );
        Array<Real> pred_arr(
            reinterpret_cast<Real*>(predictions),
            n_points
        );

        wrapper->visit([&](auto& dist) {
            nadaraya_watson(
                dist, y_arr, pred_arr,
                static_cast<Real>(bandwidth),
                convert_kernel_type(kernel_type)
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

