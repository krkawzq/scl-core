// =============================================================================
// FILE: scl/binding/c_api/kernel.cpp
// BRIEF: C API implementation for kernel methods
// =============================================================================

#include "scl/binding/c_api/kernel.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/kernel.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Helper: Convert kernel type
// =============================================================================

namespace {
    [[nodiscard]] constexpr auto convert_kernel_type(
        scl_kernel_type_t type) noexcept -> scl::kernel::kernel::KernelType {
        using KT = scl::kernel::kernel::KernelType;
        switch (type) {
            case SCL_KERNEL_GAUSSIAN: return KT::Gaussian;
            case SCL_KERNEL_EPANECHNIKOV: return KT::Epanechnikov;
            case SCL_KERNEL_COSINE: return KT::Cosine;
            case SCL_KERNEL_LINEAR: return KT::Linear;
            case SCL_KERNEL_POLYNOMIAL: return KT::Polynomial;
            case SCL_KERNEL_LAPLACIAN: return KT::Laplacian;
            case SCL_KERNEL_CAUCHY: return KT::Cauchy;
            case SCL_KERNEL_SIGMOID: return KT::Sigmoid;
            case SCL_KERNEL_UNIFORM: return KT::Uniform;
            case SCL_KERNEL_TRIANGULAR: return KT::Triangular;
            default: return KT::Gaussian;
        }
    }
} // anonymous namespace

// =============================================================================
// KDE from Distances
// =============================================================================

SCL_EXPORT scl_error_t scl_kernel_kde_from_distances(
    scl_sparse_t distances,
    scl_real_t* density,
    const scl_size_t n_points,
    const scl_real_t bandwidth,
    const scl_kernel_type_t kernel_type) {
    
    SCL_C_API_CHECK_NULL(distances, "Distances matrix handle is null");
    SCL_C_API_CHECK_NULL(density, "Density output pointer is null");
    SCL_C_API_CHECK(n_points > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of points must be positive");

    SCL_C_API_TRY
        const Index n = distances->rows();
        const Size n_points_sz = static_cast<Size>(n_points);
        
        SCL_C_API_CHECK(static_cast<Size>(n) == n_points_sz, SCL_ERROR_DIMENSION_MISMATCH,
                       "Point count mismatch");

        Array<Real> density_arr(reinterpret_cast<Real*>(density), n_points_sz);

        distances->visit([&](auto& dist) {
            scl::kernel::kernel::kde_from_distances(dist, density_arr, static_cast<Real>(bandwidth),
                             convert_kernel_type(kernel_type));
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Local Bandwidth
// =============================================================================

SCL_EXPORT scl_error_t scl_kernel_local_bandwidth(
    scl_sparse_t distances,
    scl_real_t* bandwidths,
    const scl_size_t n_points,
    const scl_index_t k) {
    
    SCL_C_API_CHECK_NULL(distances, "Distances matrix handle is null");
    SCL_C_API_CHECK_NULL(bandwidths, "Bandwidths output pointer is null");
    SCL_C_API_CHECK(n_points > 0 && k > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");

    SCL_C_API_TRY
        const Index n = distances->rows();
        const Size n_points_sz = static_cast<Size>(n_points);
        
        SCL_C_API_CHECK(static_cast<Size>(n) == n_points_sz, SCL_ERROR_DIMENSION_MISMATCH,
                       "Point count mismatch");

        Array<Real> bandwidths_arr(reinterpret_cast<Real*>(bandwidths), n_points_sz);

        distances->visit([&](auto& dist) {
            scl::kernel::kernel::local_bandwidth(dist, bandwidths_arr, k);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Adaptive KDE
// =============================================================================

SCL_EXPORT scl_error_t scl_kernel_adaptive_kde(
    scl_sparse_t distances,
    scl_real_t* density,
    const scl_size_t n_points,
    const scl_real_t* bandwidths,
    const scl_kernel_type_t kernel_type) {
    
    SCL_C_API_CHECK_NULL(distances, "Distances matrix handle is null");
    SCL_C_API_CHECK_NULL(density, "Density output pointer is null");
    SCL_C_API_CHECK_NULL(bandwidths, "Bandwidths input pointer is null");
    SCL_C_API_CHECK(n_points > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of points must be positive");

    SCL_C_API_TRY
        const Index n = distances->rows();
        const Size n_points_sz = static_cast<Size>(n_points);
        
        SCL_C_API_CHECK(static_cast<Size>(n) == n_points_sz, SCL_ERROR_DIMENSION_MISMATCH,
                       "Point count mismatch");

        Array<Real> density_arr(reinterpret_cast<Real*>(density), n_points_sz);
        Array<const Real> bandwidths_arr(reinterpret_cast<const Real*>(bandwidths), n_points_sz);

        distances->visit([&](auto& dist) {
            scl::kernel::kernel::adaptive_kde(dist, density_arr, bandwidths_arr, convert_kernel_type(kernel_type));
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Compute Kernel Matrix
// =============================================================================

SCL_EXPORT scl_error_t scl_kernel_compute_matrix(
    scl_sparse_t distances,
    scl_real_t* kernel_values,
    const scl_real_t bandwidth,
    const scl_kernel_type_t kernel_type) {
    
    SCL_C_API_CHECK_NULL(distances, "Distances matrix handle is null");
    SCL_C_API_CHECK_NULL(kernel_values, "Kernel values output pointer is null");

    SCL_C_API_TRY
        distances->visit([&](auto& dist) {
            scl::kernel::kernel::compute_kernel_matrix(dist, reinterpret_cast<Real*>(kernel_values),
                                static_cast<Real>(bandwidth),
                                convert_kernel_type(kernel_type));
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Kernel Row Sums
// =============================================================================

SCL_EXPORT scl_error_t scl_kernel_row_sums(
    scl_sparse_t distances,
    scl_real_t* sums,
    const scl_size_t n_points,
    const scl_real_t bandwidth,
    const scl_kernel_type_t kernel_type) {
    
    SCL_C_API_CHECK_NULL(distances, "Distances matrix handle is null");
    SCL_C_API_CHECK_NULL(sums, "Sums output pointer is null");
    SCL_C_API_CHECK(n_points > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of points must be positive");

    SCL_C_API_TRY
        const Index n = distances->rows();
        const Size n_points_sz = static_cast<Size>(n_points);
        
        SCL_C_API_CHECK(static_cast<Size>(n) == n_points_sz, SCL_ERROR_DIMENSION_MISMATCH,
                       "Point count mismatch");

        Array<Real> sums_arr(reinterpret_cast<Real*>(sums), n_points_sz);

        distances->visit([&](auto& dist) {
            scl::kernel::kernel::kernel_row_sums(dist, sums_arr, static_cast<Real>(bandwidth),
                          convert_kernel_type(kernel_type));
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Nadaraya-Watson Estimator
// =============================================================================

SCL_EXPORT scl_error_t scl_kernel_nadaraya_watson(
    scl_sparse_t distances,
    const scl_real_t* y_values,
    scl_real_t* predictions,
    const scl_size_t n_points,
    const scl_real_t bandwidth,
    const scl_kernel_type_t kernel_type) {
    
    SCL_C_API_CHECK_NULL(distances, "Distances matrix handle is null");
    SCL_C_API_CHECK_NULL(y_values, "Y values input pointer is null");
    SCL_C_API_CHECK_NULL(predictions, "Predictions output pointer is null");
    SCL_C_API_CHECK(n_points > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of points must be positive");

    SCL_C_API_TRY
        const Index n = distances->rows();
        const Size n_points_sz = static_cast<Size>(n_points);
        
        SCL_C_API_CHECK(static_cast<Size>(n) == n_points_sz, SCL_ERROR_DIMENSION_MISMATCH,
                       "Point count mismatch");

        Array<const Real> y_arr(reinterpret_cast<const Real*>(y_values), n_points_sz);
        Array<Real> pred_arr(reinterpret_cast<Real*>(predictions), n_points_sz);

        distances->visit([&](auto& dist) {
            scl::kernel::kernel::nadaraya_watson(dist, y_arr, pred_arr, static_cast<Real>(bandwidth),
                          convert_kernel_type(kernel_type));
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
