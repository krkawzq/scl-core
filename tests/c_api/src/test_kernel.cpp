// =============================================================================
// SCL Core - Kernel Methods Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/kernel.h
//
// Functions tested:
//   ✓ scl_kernel_kde_from_distances - Kernel density estimation
//   ✓ scl_kernel_local_bandwidth - Local bandwidth estimation
//   ✓ scl_kernel_adaptive_kde - Adaptive KDE
//   ✓ scl_kernel_compute_matrix - Kernel matrix computation
//   ✓ scl_kernel_row_sums - Kernel row sums
//   ✓ scl_kernel_nadaraya_watson - Nadaraya-Watson estimator
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"

using namespace scl::test;
using precision::Tolerance;

SCL_TEST_BEGIN

// =============================================================================
// Helper: Create symmetric distance matrix (sparse)
// =============================================================================

static Sparse create_distance_matrix(scl_size_t n_points, 
                                    const std::vector<scl_real_t>& distances) {
    // Create symmetric sparse matrix for distances
    // Only store upper triangle + diagonal
    std::vector<scl_index_t> indptr(n_points + 1);
    std::vector<scl_index_t> indices;
    std::vector<scl_real_t> data;
    
    scl_index_t nnz = 0;
    scl_size_t idx = 0;
    
    for (scl_index_t i = 0; i < static_cast<scl_index_t>(n_points); ++i) {
        indptr[i] = nnz;
        for (scl_index_t j = i; j < static_cast<scl_index_t>(n_points); ++j) {
            if (idx < distances.size()) {
                indices.push_back(j);
                data.push_back(distances[idx]);
                ++nnz;
                ++idx;
            }
        }
    }
    indptr[n_points] = nnz;
    
    return make_sparse_csr(n_points, n_points, nnz, indptr.data(), indices.data(), data.data());
}

// =============================================================================
// Helper: Create simple distance matrix
// =============================================================================

static Sparse create_simple_distance_matrix(scl_size_t n) {
    std::vector<scl_real_t> distances;
    for (scl_size_t i = 0; i < n; ++i) {
        for (scl_size_t j = i; j < n; ++j) {
            if (i == j) {
                distances.push_back(0.0);
            } else {
                distances.push_back(static_cast<scl_real_t>(j - i));
            }
        }
    }
    return create_distance_matrix(n, distances);
}

// =============================================================================
// Kernel Density Estimation Tests
// =============================================================================

SCL_TEST_SUITE(kde_from_distances)

SCL_TEST_CASE(kde_basic_gaussian) {
    scl_size_t n_points = 5;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    std::vector<scl_real_t> density(n_points);
    scl_real_t bandwidth = 1.0;
    
    scl_error_t err = scl_kernel_kde_from_distances(
        distances, density.data(), n_points, bandwidth, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Density should be positive
    for (size_t i = 0; i < n_points; ++i) {
        SCL_ASSERT_GT(density[i], 0.0);
    }
}

SCL_TEST_CASE(kde_all_kernel_types) {
    scl_size_t n_points = 4;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    scl_kernel_type_t kernel_types[] = {
        SCL_KERNEL_GAUSSIAN,
        SCL_KERNEL_EPANECHNIKOV,
        SCL_KERNEL_COSINE,
        SCL_KERNEL_LINEAR,
        SCL_KERNEL_POLYNOMIAL,
        SCL_KERNEL_LAPLACIAN,
        SCL_KERNEL_CAUCHY,
        SCL_KERNEL_SIGMOID,
        SCL_KERNEL_UNIFORM,
        SCL_KERNEL_TRIANGULAR
    };
    
    for (auto kernel_type : kernel_types) {
        std::vector<scl_real_t> density(n_points);
        scl_error_t err = scl_kernel_kde_from_distances(
            distances, density.data(), n_points, 1.0, kernel_type
        );
        
        SCL_ASSERT_EQ(err, SCL_OK);
        
        // All densities should be non-negative
        for (size_t i = 0; i < n_points; ++i) {
            SCL_ASSERT_GE(density[i], 0.0);
        }
    }
}

SCL_TEST_CASE(kde_zero_bandwidth) {
    scl_size_t n_points = 3;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    std::vector<scl_real_t> density(n_points);
    scl_error_t err = scl_kernel_kde_from_distances(
        distances, density.data(), n_points, 0.0, SCL_KERNEL_GAUSSIAN
    );
    
    // Zero bandwidth might be invalid or handled specially
    // Just check it doesn't crash
    if (err == SCL_OK) {
        for (size_t i = 0; i < n_points; ++i) {
            SCL_ASSERT_GE(density[i], 0.0);
        }
    }
}

SCL_TEST_CASE(kde_null_handle) {
    std::vector<scl_real_t> density(5);
    scl_error_t err = scl_kernel_kde_from_distances(
        nullptr, density.data(), 5, 1.0, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(kde_null_output) {
    scl_size_t n_points = 5;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    scl_error_t err = scl_kernel_kde_from_distances(
        distances, nullptr, n_points, 1.0, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(kde_single_point) {
    scl_size_t n_points = 1;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    std::vector<scl_real_t> density(n_points);
    scl_error_t err = scl_kernel_kde_from_distances(
        distances, density.data(), n_points, 1.0, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GT(density[0], 0.0);
}

SCL_TEST_SUITE_END

// =============================================================================
// Local Bandwidth Estimation Tests
// =============================================================================

SCL_TEST_SUITE(local_bandwidth)

SCL_TEST_CASE(local_bandwidth_basic) {
    scl_size_t n_points = 10;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    std::vector<scl_real_t> bandwidths(n_points);
    scl_index_t k = 3;
    
    scl_error_t err = scl_kernel_local_bandwidth(
        distances, bandwidths.data(), n_points, k
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Bandwidths should be positive
    for (size_t i = 0; i < n_points; ++i) {
        SCL_ASSERT_GT(bandwidths[i], 0.0);
    }
}

SCL_TEST_CASE(local_bandwidth_k_one) {
    scl_size_t n_points = 5;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    std::vector<scl_real_t> bandwidths(n_points);
    scl_error_t err = scl_kernel_local_bandwidth(
        distances, bandwidths.data(), n_points, 1
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(local_bandwidth_k_large) {
    scl_size_t n_points = 10;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    std::vector<scl_real_t> bandwidths(n_points);
    scl_error_t err = scl_kernel_local_bandwidth(
        distances, bandwidths.data(), n_points, n_points - 1
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(local_bandwidth_null_handle) {
    std::vector<scl_real_t> bandwidths(5);
    scl_error_t err = scl_kernel_local_bandwidth(
        nullptr, bandwidths.data(), 5, 3
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(local_bandwidth_null_output) {
    scl_size_t n_points = 5;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    scl_error_t err = scl_kernel_local_bandwidth(
        distances, nullptr, n_points, 3
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Adaptive KDE Tests
// =============================================================================

SCL_TEST_SUITE(adaptive_kde)

SCL_TEST_CASE(adaptive_kde_basic) {
    scl_size_t n_points = 5;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    // Compute local bandwidths first
    std::vector<scl_real_t> bandwidths(n_points);
    scl_kernel_local_bandwidth(distances, bandwidths.data(), n_points, 2);
    
    std::vector<scl_real_t> density(n_points);
    scl_error_t err = scl_kernel_adaptive_kde(
        distances, density.data(), n_points, bandwidths.data(), SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Density should be positive
    for (size_t i = 0; i < n_points; ++i) {
        SCL_ASSERT_GT(density[i], 0.0);
    }
}

SCL_TEST_CASE(adaptive_kde_all_kernels) {
    scl_size_t n_points = 4;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    std::vector<scl_real_t> bandwidths(n_points, 1.0);
    
    scl_kernel_type_t kernel_types[] = {
        SCL_KERNEL_GAUSSIAN,
        SCL_KERNEL_EPANECHNIKOV,
        SCL_KERNEL_LAPLACIAN
    };
    
    for (auto kernel_type : kernel_types) {
        std::vector<scl_real_t> density(n_points);
        scl_error_t err = scl_kernel_adaptive_kde(
            distances, density.data(), n_points, bandwidths.data(), kernel_type
        );
        
        SCL_ASSERT_EQ(err, SCL_OK);
    }
}

SCL_TEST_CASE(adaptive_kde_null_handle) {
    std::vector<scl_real_t> bandwidths(5, 1.0);
    std::vector<scl_real_t> density(5);
    
    scl_error_t err = scl_kernel_adaptive_kde(
        nullptr, density.data(), 5, bandwidths.data(), SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(adaptive_kde_null_bandwidths) {
    scl_size_t n_points = 5;
    Sparse distances = create_simple_distance_matrix(n_points);
    std::vector<scl_real_t> density(n_points);
    
    scl_error_t err = scl_kernel_adaptive_kde(
        distances, density.data(), n_points, nullptr, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(adaptive_kde_null_output) {
    scl_size_t n_points = 5;
    Sparse distances = create_simple_distance_matrix(n_points);
    std::vector<scl_real_t> bandwidths(n_points, 1.0);
    
    scl_error_t err = scl_kernel_adaptive_kde(
        distances, nullptr, n_points, bandwidths.data(), SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Kernel Matrix Computation Tests
// =============================================================================

SCL_TEST_SUITE(compute_matrix)

SCL_TEST_CASE(compute_matrix_basic) {
    scl_size_t n_points = 5;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    scl_index_t nnz;
    scl_sparse_nnz(distances, &nnz);
    
    std::vector<scl_real_t> kernel_values(nnz);
    scl_real_t bandwidth = 1.0;
    
    scl_error_t err = scl_kernel_compute_matrix(
        distances, kernel_values.data(), bandwidth, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Kernel values should be in [0, 1] for most kernels
    for (size_t i = 0; i < nnz; ++i) {
        SCL_ASSERT_GE(kernel_values[i], 0.0);
    }
}

SCL_TEST_CASE(compute_matrix_diagonal) {
    scl_size_t n_points = 3;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    scl_index_t nnz;
    scl_sparse_nnz(distances, &nnz);
    
    std::vector<scl_real_t> kernel_values(nnz);
    scl_error_t err = scl_kernel_compute_matrix(
        distances, kernel_values.data(), 1.0, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Diagonal (distance=0) should give maximum kernel value
    // For Gaussian: K(0) = 1.0
    SCL_ASSERT_NEAR(kernel_values[0], 1.0, 1e-6);
}

SCL_TEST_CASE(compute_matrix_null_handle) {
    std::vector<scl_real_t> kernel_values(10);
    scl_error_t err = scl_kernel_compute_matrix(
        nullptr, kernel_values.data(), 1.0, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(compute_matrix_null_output) {
    scl_size_t n_points = 5;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    scl_error_t err = scl_kernel_compute_matrix(
        distances, nullptr, 1.0, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Kernel Row Sums Tests
// =============================================================================

SCL_TEST_SUITE(row_sums)

SCL_TEST_CASE(row_sums_basic) {
    scl_size_t n_points = 5;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    std::vector<scl_real_t> sums(n_points);
    scl_real_t bandwidth = 1.0;
    
    scl_error_t err = scl_kernel_row_sums(
        distances, sums.data(), n_points, bandwidth, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Row sums should be positive
    for (size_t i = 0; i < n_points; ++i) {
        SCL_ASSERT_GT(sums[i], 0.0);
    }
}

SCL_TEST_CASE(row_sums_all_kernels) {
    scl_size_t n_points = 4;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    scl_kernel_type_t kernel_types[] = {
        SCL_KERNEL_GAUSSIAN,
        SCL_KERNEL_EPANECHNIKOV,
        SCL_KERNEL_COSINE,
        SCL_KERNEL_LINEAR
    };
    
    for (auto kernel_type : kernel_types) {
        std::vector<scl_real_t> sums(n_points);
        scl_error_t err = scl_kernel_row_sums(
            distances, sums.data(), n_points, 1.0, kernel_type
        );
        
        SCL_ASSERT_EQ(err, SCL_OK);
    }
}

SCL_TEST_CASE(row_sums_null_handle) {
    std::vector<scl_real_t> sums(5);
    scl_error_t err = scl_kernel_row_sums(
        nullptr, sums.data(), 5, 1.0, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(row_sums_null_output) {
    scl_size_t n_points = 5;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    scl_error_t err = scl_kernel_row_sums(
        distances, nullptr, n_points, 1.0, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Nadaraya-Watson Estimator Tests
// =============================================================================

SCL_TEST_SUITE(nadaraya_watson)

SCL_TEST_CASE(nadaraya_watson_basic) {
    scl_size_t n_points = 5;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    std::vector<scl_real_t> y_values = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<scl_real_t> predictions(n_points);
    scl_real_t bandwidth = 1.0;
    
    scl_error_t err = scl_kernel_nadaraya_watson(
        distances, y_values.data(), predictions.data(), n_points, 
        bandwidth, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Predictions should be weighted averages of y_values
    // All should be finite
    for (size_t i = 0; i < n_points; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(predictions[i]));
    }
}

SCL_TEST_CASE(nadaraya_watson_constant_y) {
    scl_size_t n_points = 4;
    Sparse distances = create_simple_distance_matrix(n_points);
    
    std::vector<scl_real_t> y_values(n_points, 5.0);
    std::vector<scl_real_t> predictions(n_points);
    
    scl_error_t err = scl_kernel_nadaraya_watson(
        distances, y_values.data(), predictions.data(), n_points,
        1.0, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // If all y are the same, predictions should be close to that value
    for (size_t i = 0; i < n_points; ++i) {
        SCL_ASSERT_NEAR(predictions[i], 5.0, 1e-6);
    }
}

SCL_TEST_CASE(nadaraya_watson_null_handle) {
    std::vector<scl_real_t> y_values(5, 1.0);
    std::vector<scl_real_t> predictions(5);
    
    scl_error_t err = scl_kernel_nadaraya_watson(
        nullptr, y_values.data(), predictions.data(), 5,
        1.0, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(nadaraya_watson_null_y_values) {
    scl_size_t n_points = 5;
    Sparse distances = create_simple_distance_matrix(n_points);
    std::vector<scl_real_t> predictions(n_points);
    
    scl_error_t err = scl_kernel_nadaraya_watson(
        distances, nullptr, predictions.data(), n_points,
        1.0, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(nadaraya_watson_null_predictions) {
    scl_size_t n_points = 5;
    Sparse distances = create_simple_distance_matrix(n_points);
    std::vector<scl_real_t> y_values(n_points, 1.0);
    
    scl_error_t err = scl_kernel_nadaraya_watson(
        distances, y_values.data(), nullptr, n_points,
        1.0, SCL_KERNEL_GAUSSIAN
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

