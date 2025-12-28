// =============================================================================
// FILE: scl/binding/c_api/kernel.cpp
// BRIEF: C API implementation for kernel methods
// =============================================================================

#include "scl/binding/c_api/kernel.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/kernel.hpp"
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

static scl::kernel::kernel::KernelType convert_kernel_type(scl_kernel_type_t type) {
    switch (type) {
        case SCL_KERNEL_GAUSSIAN: return scl::kernel::kernel::KernelType::Gaussian;
        case SCL_KERNEL_EPANECHNIKOV: return scl::kernel::kernel::KernelType::Epanechnikov;
        case SCL_KERNEL_COSINE: return scl::kernel::kernel::KernelType::Cosine;
        case SCL_KERNEL_LINEAR: return scl::kernel::kernel::KernelType::Linear;
        case SCL_KERNEL_POLYNOMIAL: return scl::kernel::kernel::KernelType::Polynomial;
        case SCL_KERNEL_LAPLACIAN: return scl::kernel::kernel::KernelType::Laplacian;
        case SCL_KERNEL_CAUCHY: return scl::kernel::kernel::KernelType::Cauchy;
        case SCL_KERNEL_SIGMOID: return scl::kernel::kernel::KernelType::Sigmoid;
        case SCL_KERNEL_UNIFORM: return scl::kernel::kernel::KernelType::Uniform;
        case SCL_KERNEL_TRIANGULAR: return scl::kernel::kernel::KernelType::Triangular;
        default: return scl::kernel::kernel::KernelType::Gaussian;
    }
}

scl_error_t scl_kde_from_distances(
    const scl_sparse_matrix_t* distances,
    scl_real_t* density,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
) {
    if (!distances || !density) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = reinterpret_cast<const scl::CSR*>(distances);
        scl::Size n = static_cast<scl::Size>(sparse->rows());
        scl::Array<scl::Real> density_arr(reinterpret_cast<scl::Real*>(density), n);
        scl::kernel::kernel::kde_from_distances(
            *sparse, density_arr,
            static_cast<scl::Real>(bandwidth),
            convert_kernel_type(kernel_type)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_silverman_bandwidth(
    const scl_real_t* data,
    scl_size_t n,
    scl_index_t n_features,
    scl_real_t* bandwidth
) {
    if (!data || !bandwidth) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Real> data_arr(reinterpret_cast<const scl::Real*>(data), n);
        scl::Real h = scl::kernel::kernel::silverman_bandwidth(
            data_arr, static_cast<scl::Index>(n_features)
        );
        *bandwidth = static_cast<scl_real_t>(h);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_scott_bandwidth(
    const scl_real_t* data,
    scl_size_t n,
    scl_index_t n_features,
    scl_real_t* bandwidth
) {
    if (!data || !bandwidth) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Real> data_arr(reinterpret_cast<const scl::Real*>(data), n);
        scl::Real h = scl::kernel::kernel::scott_bandwidth(
            data_arr, static_cast<scl::Index>(n_features)
        );
        *bandwidth = static_cast<scl_real_t>(h);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_local_bandwidth(
    const scl_sparse_matrix_t* distances,
    scl_real_t* bandwidths,
    scl_index_t k
) {
    if (!distances || !bandwidths) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = reinterpret_cast<const scl::CSR*>(distances);
        scl::Size n = static_cast<scl::Size>(sparse->rows());
        scl::Array<scl::Real> bandwidths_arr(reinterpret_cast<scl::Real*>(bandwidths), n);
        scl::kernel::kernel::local_bandwidth(
            *sparse, bandwidths_arr, static_cast<scl::Index>(k)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_adaptive_kde(
    const scl_sparse_matrix_t* distances,
    scl_real_t* density,
    const scl_real_t* bandwidths,
    scl_kernel_type_t kernel_type
) {
    if (!distances || !density || !bandwidths) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = reinterpret_cast<const scl::CSR*>(distances);
        scl::Size n = static_cast<scl::Size>(sparse->rows());
        scl::Array<scl::Real> density_arr(reinterpret_cast<scl::Real*>(density), n);
        scl::Array<const scl::Real> bandwidths_arr(reinterpret_cast<const scl::Real*>(bandwidths), n);
        scl::kernel::kernel::adaptive_kde(
            *sparse, density_arr, bandwidths_arr,
            convert_kernel_type(kernel_type)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_compute_kernel_matrix(
    const scl_sparse_matrix_t* distances,
    scl_real_t* kernel_values,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
) {
    if (!distances || !kernel_values) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = reinterpret_cast<const scl::CSR*>(distances);
        scl::kernel::kernel::compute_kernel_matrix(
            *sparse,
            reinterpret_cast<scl::Real*>(kernel_values),
            static_cast<scl::Real>(bandwidth),
            convert_kernel_type(kernel_type)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_kernel_row_sums(
    const scl_sparse_matrix_t* distances,
    scl_real_t* sums,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
) {
    if (!distances || !sums) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = reinterpret_cast<const scl::CSR*>(distances);
        scl::Size n = static_cast<scl::Size>(sparse->rows());
        scl::Array<scl::Real> sums_arr(reinterpret_cast<scl::Real*>(sums), n);
        scl::kernel::kernel::kernel_row_sums(
            *sparse, sums_arr,
            static_cast<scl::Real>(bandwidth),
            convert_kernel_type(kernel_type)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_nadaraya_watson(
    const scl_sparse_matrix_t* distances,
    const scl_real_t* y_values,
    scl_real_t* predictions,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
) {
    if (!distances || !y_values || !predictions) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = reinterpret_cast<const scl::CSR*>(distances);
        scl::Size n = static_cast<scl::Size>(sparse->rows());
        scl::Array<const scl::Real> y_arr(reinterpret_cast<const scl::Real*>(y_values), n);
        scl::Array<scl::Real> pred_arr(reinterpret_cast<scl::Real*>(predictions), n);
        scl::kernel::kernel::nadaraya_watson(
            *sparse, y_arr, pred_arr,
            static_cast<scl::Real>(bandwidth),
            convert_kernel_type(kernel_type)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_kernel_smooth_graph(
    const scl_sparse_matrix_t* kernel_weights,
    const scl_real_t* values,
    scl_real_t* smoothed_values
) {
    if (!kernel_weights || !values || !smoothed_values) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = reinterpret_cast<const scl::CSR*>(kernel_weights);
        scl::Size n = static_cast<scl::Size>(sparse->rows());
        scl::Array<const scl::Real> values_arr(reinterpret_cast<const scl::Real*>(values), n);
        scl::Array<scl::Real> smoothed_arr(reinterpret_cast<scl::Real*>(smoothed_values), n);
        scl::kernel::kernel::kernel_smooth_graph(*sparse, values_arr, smoothed_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_local_linear_regression(
    const scl_sparse_matrix_t* distances,
    const scl_real_t* X,
    const scl_real_t* Y,
    scl_real_t* predictions,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
) {
    if (!distances || !X || !Y || !predictions) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = reinterpret_cast<const scl::CSR*>(distances);
        scl::Size n = static_cast<scl::Size>(sparse->rows());
        scl::Array<const scl::Real> X_arr(reinterpret_cast<const scl::Real*>(X), n);
        scl::Array<const scl::Real> Y_arr(reinterpret_cast<const scl::Real*>(Y), n);
        scl::Array<scl::Real> pred_arr(reinterpret_cast<scl::Real*>(predictions), n);
        scl::kernel::kernel::local_linear_regression(
            *sparse, X_arr, Y_arr, pred_arr,
            static_cast<scl::Real>(bandwidth),
            convert_kernel_type(kernel_type)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_kernel_entropy(
    const scl_sparse_matrix_t* distances,
    scl_real_t* entropy,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
) {
    if (!distances || !entropy) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = reinterpret_cast<const scl::CSR*>(distances);
        scl::Size n = static_cast<scl::Size>(sparse->rows());
        scl::Array<scl::Real> entropy_arr(reinterpret_cast<scl::Real*>(entropy), n);
        scl::kernel::kernel::kernel_entropy(
            *sparse, entropy_arr,
            static_cast<scl::Real>(bandwidth),
            convert_kernel_type(kernel_type)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_find_bandwidth_for_perplexity(
    const scl_sparse_matrix_t* distances,
    scl_real_t* bandwidths,
    scl_real_t target_perplexity,
    scl_index_t max_iter,
    scl_real_t tol
) {
    if (!distances || !bandwidths) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = reinterpret_cast<const scl::CSR*>(distances);
        scl::Size n = static_cast<scl::Size>(sparse->rows());
        scl::Array<scl::Real> bandwidths_arr(reinterpret_cast<scl::Real*>(bandwidths), n);
        scl::kernel::kernel::find_bandwidth_for_perplexity(
            *sparse, bandwidths_arr,
            static_cast<scl::Real>(target_perplexity),
            static_cast<scl::Index>(max_iter),
            static_cast<scl::Real>(tol)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_evaluate_kernel(
    scl_kernel_type_t type,
    scl_real_t distance,
    scl_real_t bandwidth,
    scl_real_t* result
) {
    if (!result) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Real val = scl::kernel::kernel::evaluate_kernel(
            convert_kernel_type(type),
            static_cast<scl::Real>(distance),
            static_cast<scl::Real>(bandwidth)
        );
        *result = static_cast<scl_real_t>(val);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::kernel::kde_from_distances(scl::CSR const&, scl::Array<scl::Real>, scl::Real, scl::kernel::kernel::KernelType);
template void scl::kernel::kernel::local_bandwidth(scl::CSR const&, scl::Array<scl::Real>, scl::Index);
template void scl::kernel::kernel::adaptive_kde(scl::CSR const&, scl::Array<scl::Real>, scl::Array<const scl::Real>, scl::kernel::kernel::KernelType);
template void scl::kernel::kernel::compute_kernel_matrix(scl::CSR const&, scl::Real*, scl::Real, scl::kernel::kernel::KernelType);
template void scl::kernel::kernel::kernel_row_sums(scl::CSR const&, scl::Array<scl::Real>, scl::Real, scl::kernel::kernel::KernelType);
template void scl::kernel::kernel::nadaraya_watson(scl::CSR const&, scl::Array<const scl::Real>, scl::Array<scl::Real>, scl::Real, scl::kernel::kernel::KernelType);
template void scl::kernel::kernel::kernel_smooth_graph(scl::CSR const&, scl::Array<const scl::Real>, scl::Array<scl::Real>);
template void scl::kernel::kernel::local_linear_regression(scl::CSR const&, scl::Array<const scl::Real>, scl::Array<const scl::Real>, scl::Array<scl::Real>, scl::Real, scl::kernel::kernel::KernelType);
template void scl::kernel::kernel::kernel_entropy(scl::CSR const&, scl::Array<scl::Real>, scl::Real, scl::kernel::kernel::KernelType);
template void scl::kernel::kernel::find_bandwidth_for_perplexity(scl::CSR const&, scl::Array<scl::Real>, scl::Real, scl::Index, scl::Real);

} // extern "C"

