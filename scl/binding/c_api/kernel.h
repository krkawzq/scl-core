#pragma once

// =============================================================================
// FILE: scl/binding/c_api/kernel/kernel.h
// BRIEF: C API for kernel methods and KDE
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Kernel Types
// =============================================================================

typedef enum {
    SCL_KERNEL_GAUSSIAN = 0,
    SCL_KERNEL_EPANECHNIKOV = 1,
    SCL_KERNEL_COSINE = 2,
    SCL_KERNEL_LINEAR = 3,
    SCL_KERNEL_POLYNOMIAL = 4,
    SCL_KERNEL_LAPLACIAN = 5,
    SCL_KERNEL_CAUCHY = 6,
    SCL_KERNEL_SIGMOID = 7,
    SCL_KERNEL_UNIFORM = 8,
    SCL_KERNEL_TRIANGULAR = 9
} scl_kernel_type_t;

// =============================================================================
// Kernel Density Estimation
// =============================================================================

scl_error_t scl_kernel_kde_from_distances(
    scl_sparse_t distances,
    scl_real_t* density,
    scl_size_t n_points,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
);

// =============================================================================
// Adaptive Bandwidth Estimation
// =============================================================================

scl_error_t scl_kernel_local_bandwidth(
    scl_sparse_t distances,
    scl_real_t* bandwidths,
    scl_size_t n_points,
    scl_index_t k
);

scl_error_t scl_kernel_adaptive_kde(
    scl_sparse_t distances,
    scl_real_t* density,
    scl_size_t n_points,
    const scl_real_t* bandwidths,
    scl_kernel_type_t kernel_type
);

// =============================================================================
// Kernel Matrix Computation
// =============================================================================

scl_error_t scl_kernel_compute_matrix(
    scl_sparse_t distances,
    scl_real_t* kernel_values,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
);

// =============================================================================
// Kernel Row Sums
// =============================================================================

scl_error_t scl_kernel_row_sums(
    scl_sparse_t distances,
    scl_real_t* sums,
    scl_size_t n_points,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
);

// =============================================================================
// Nadaraya-Watson Estimator
// =============================================================================

scl_error_t scl_kernel_nadaraya_watson(
    scl_sparse_t distances,
    const scl_real_t* y_values,
    scl_real_t* predictions,
    scl_size_t n_points,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
);

#ifdef __cplusplus
}
#endif
