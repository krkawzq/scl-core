#pragma once

// =============================================================================
// FILE: scl/binding/c_api/kernel.h
// BRIEF: C API for kernel methods
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "scl/binding/c_api/core_types.h"

// Kernel types
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

// Kernel density estimation from distance matrix
scl_error_t scl_kde_from_distances(
    const scl_sparse_matrix_t* distances,
    scl_real_t* density,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
);

// Silverman bandwidth estimation
scl_error_t scl_silverman_bandwidth(
    const scl_real_t* data,
    scl_size_t n,
    scl_index_t n_features,
    scl_real_t* bandwidth
);

// Scott bandwidth estimation
scl_error_t scl_scott_bandwidth(
    const scl_real_t* data,
    scl_size_t n,
    scl_index_t n_features,
    scl_real_t* bandwidth
);

// Local bandwidth estimation (k-NN based)
scl_error_t scl_local_bandwidth(
    const scl_sparse_matrix_t* distances,
    scl_real_t* bandwidths,
    scl_index_t k
);

// Adaptive KDE with local bandwidths
scl_error_t scl_adaptive_kde(
    const scl_sparse_matrix_t* distances,
    scl_real_t* density,
    const scl_real_t* bandwidths,
    scl_kernel_type_t kernel_type
);

// Compute kernel matrix
scl_error_t scl_compute_kernel_matrix(
    const scl_sparse_matrix_t* distances,
    scl_real_t* kernel_values,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
);

// Kernel row sums
scl_error_t scl_kernel_row_sums(
    const scl_sparse_matrix_t* distances,
    scl_real_t* sums,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
);

// Nadaraya-Watson estimator
scl_error_t scl_nadaraya_watson(
    const scl_sparse_matrix_t* distances,
    const scl_real_t* y_values,
    scl_real_t* predictions,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
);

// Kernel smoothing on graph
scl_error_t scl_kernel_smooth_graph(
    const scl_sparse_matrix_t* kernel_weights,
    const scl_real_t* values,
    scl_real_t* smoothed_values
);

// Local linear regression
scl_error_t scl_local_linear_regression(
    const scl_sparse_matrix_t* distances,
    const scl_real_t* X,
    const scl_real_t* Y,
    scl_real_t* predictions,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
);

// Kernel entropy estimation
scl_error_t scl_kernel_entropy(
    const scl_sparse_matrix_t* distances,
    scl_real_t* entropy,
    scl_real_t bandwidth,
    scl_kernel_type_t kernel_type
);

// Find bandwidth for target perplexity
scl_error_t scl_find_bandwidth_for_perplexity(
    const scl_sparse_matrix_t* distances,
    scl_real_t* bandwidths,
    scl_real_t target_perplexity,
    scl_index_t max_iter,
    scl_real_t tol
);

// Evaluate specific kernel function
scl_error_t scl_evaluate_kernel(
    scl_kernel_type_t type,
    scl_real_t distance,
    scl_real_t bandwidth,
    scl_real_t* result
);

#ifdef __cplusplus
}
#endif
