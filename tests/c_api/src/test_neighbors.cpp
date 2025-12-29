// =============================================================================
// SCL Kernel - K-Nearest Neighbors Tests
// =============================================================================
//
// 测试 scl/binding/c_api/neighbors.h 中的 KNN 函数
//
// 函数:
//   ✓ scl_neighbors_compute_norms (compute squared norms)
//   ✓ scl_knn (KNN with precomputed norms)
//   ✓ scl_knn_simple (KNN without precomputed norms)
//
// 参考实现: 标准欧氏距离计算
// 精度要求: Tolerance::normal() (rtol=1e-9, atol=1e-12)
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"
#include "scl/binding/c_api/neighbors.h"

using namespace scl::test;
using precision::Tolerance;

// Reference implementation
namespace reference {

inline double squared_norm(const Eigen::SparseVector<scl_real_t>& vec) {
    double sum = 0.0;
    for (Eigen::SparseVector<scl_real_t>::InnerIterator it(vec); it; ++it) {
        sum += it.value() * it.value();
    }
    return sum;
}

inline double squared_distance(const Eigen::SparseVector<scl_real_t>& x,
                               const Eigen::SparseVector<scl_real_t>& y) {
    // ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
    double x_norm_sq = squared_norm(x);
    double y_norm_sq = squared_norm(y);
    
    // Compute dot product
    double dot = 0.0;
    Eigen::SparseVector<scl_real_t>::InnerIterator x_it(x);
    Eigen::SparseVector<scl_real_t>::InnerIterator y_it(y);
    
    while (x_it && y_it) {
        if (x_it.index() < y_it.index()) {
            ++x_it;
        } else if (y_it.index() < x_it.index()) {
            ++y_it;
        } else {
            dot += x_it.value() * y_it.value();
            ++x_it;
            ++y_it;
        }
    }
    
    return x_norm_sq + y_norm_sq - 2.0 * dot;
}

} // namespace reference

SCL_TEST_BEGIN

// =============================================================================
// Compute Norms Tests
// =============================================================================

SCL_TEST_SUITE(compute_norms)

SCL_TEST_CASE(compute_norms_basic) {
    // Simple 3x2 matrix
    std::vector<scl_index_t> indptr = {0, 2, 4, 6};
    std::vector<scl_index_t> indices = {0, 1, 0, 1, 0, 1};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    Sparse mat = make_sparse_csr(3, 2, 6, indptr.data(), indices.data(), data.data());
    
    std::vector<scl_real_t> norms_sq(3);
    scl_error_t err = scl_neighbors_compute_norms(mat, norms_sq.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Row 0: [1, 2] -> norm^2 = 1 + 4 = 5
    SCL_ASSERT_NEAR(norms_sq[0], 5.0, 1e-10);
    // Row 1: [3, 4] -> norm^2 = 9 + 16 = 25
    SCL_ASSERT_NEAR(norms_sq[1], 25.0, 1e-10);
    // Row 2: [5, 6] -> norm^2 = 25 + 36 = 61
    SCL_ASSERT_NEAR(norms_sq[2], 61.0, 1e-10);
}

SCL_TEST_RETRY(compute_norms_random, 3)
{
    Random rng(42);
    
    auto [rows, cols] = random_shape(10, 50, rng);
    double density = random_density(0.05, 0.2, rng);
    
    auto mat_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> norms_sq(rows);
    scl_error_t err = scl_neighbors_compute_norms(mat, norms_sq.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Compare with reference
    for (scl_index_t i = 0; i < rows; ++i) {
        auto row = mat_eigen.row(i);
        double ref_norm_sq = reference::squared_norm(row);
        SCL_ASSERT_TRUE(precision::approx_equal(norms_sq[i], ref_norm_sq, Tolerance::normal()));
    }
}

SCL_TEST_CASE(compute_norms_null_matrix) {
    std::vector<scl_real_t> norms_sq(10);
    scl_error_t err = scl_neighbors_compute_norms(nullptr, norms_sq.data());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(compute_norms_null_output) {
    Random rng(42);
    auto mat_eigen = random_sparse_csr(10, 20, 0.1, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_neighbors_compute_norms(mat, nullptr);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(compute_norms_empty_matrix) {
    auto Z = zero_csr(0, 10);
    auto csr = from_eigen_csr(Z);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> norms_sq(0);
    scl_error_t err = scl_neighbors_compute_norms(mat, norms_sq.data());
    
    // Should handle empty matrix gracefully
    (void)err;
}

SCL_TEST_SUITE_END

// =============================================================================
// KNN Simple Tests (without precomputed norms)
// =============================================================================

SCL_TEST_SUITE(knn_simple)

SCL_TEST_CASE(knn_simple_basic) {
    // 4 points in 2D
    std::vector<scl_index_t> indptr = {0, 2, 4, 6, 8};
    std::vector<scl_index_t> indices = {0, 1, 0, 1, 0, 1, 0, 1};
    std::vector<scl_real_t> data = {
        0.0, 0.0,  // Point 0: (0, 0)
        1.0, 0.0,  // Point 1: (1, 0)
        0.0, 1.0,  // Point 2: (0, 1)
        1.0, 1.0   // Point 3: (1, 1)
    };
    
    Sparse mat = make_sparse_csr(4, 2, 8, indptr.data(), indices.data(), data.data());
    
    scl_size_t k = 2;
    std::vector<scl_index_t> indices_out(4 * k);
    std::vector<scl_real_t> distances_out(4 * k);
    
    scl_error_t err = scl_knn_simple(mat, 4, k, indices_out.data(), distances_out.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Point 0's nearest neighbors should be point 1 and 2 (distance 1.0)
    // Point 1's nearest neighbors should be point 0 and 3 (distance 1.0)
    // etc.
    
    // Verify distances are non-negative and sorted
    for (scl_index_t i = 0; i < 4; ++i) {
        for (scl_size_t j = 0; j < k; ++j) {
            size_t idx = i * k + j;
            SCL_ASSERT_GE(distances_out[idx], 0.0);
            SCL_ASSERT_LT(indices_out[idx], 4);
            
            // Distances should be sorted (ascending)
            if (j > 0) {
                SCL_ASSERT_LE(distances_out[idx - 1], distances_out[idx]);
            }
        }
    }
}

SCL_TEST_RETRY(knn_simple_random, 3)
{
    Random rng(42);
    
    scl_index_t n_samples = rng.uniform_int(10, 30);
    scl_index_t n_features = rng.uniform_int(5, 20);
    double density = random_density(0.1, 0.3, rng);
    scl_size_t k = rng.uniform_int(2, std::min(static_cast<scl_size_t>(n_samples - 1), static_cast<scl_size_t>(5)));
    
    auto mat_eigen = random_sparse_csr(n_samples, n_features, density, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> indices_out(n_samples * k);
    std::vector<scl_real_t> distances_out(n_samples * k);
    
    scl_error_t err = scl_knn_simple(mat, n_samples, k, indices_out.data(), distances_out.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify outputs
    for (scl_index_t i = 0; i < n_samples; ++i) {
        for (scl_size_t j = 0; j < k; ++j) {
            size_t idx = i * k + j;
            SCL_ASSERT_GE(distances_out[idx], 0.0);
            SCL_ASSERT_LT(indices_out[idx], n_samples);
            
            // Self-distance should be 0 (if j == 0 and point is its own neighbor)
            // Otherwise distances should be positive
            if (j > 0) {
                SCL_ASSERT_LE(distances_out[idx - 1], distances_out[idx]);
            }
        }
    }
}

SCL_TEST_CASE(knn_simple_k_equals_one) {
    Random rng(123);
    auto mat_eigen = random_sparse_csr(10, 5, 0.2, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> indices_out(10);
    std::vector<scl_real_t> distances_out(10);
    
    scl_error_t err = scl_knn_simple(mat, 10, 1, indices_out.data(), distances_out.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Each point should have exactly one neighbor
    for (scl_index_t i = 0; i < 10; ++i) {
        SCL_ASSERT_GE(distances_out[i], 0.0);
        SCL_ASSERT_LT(indices_out[i], 10);
    }
}

SCL_TEST_CASE(knn_simple_null_matrix) {
    std::vector<scl_index_t> indices_out(10);
    std::vector<scl_real_t> distances_out(10);
    
    scl_error_t err = scl_knn_simple(nullptr, 10, 2, indices_out.data(), distances_out.data());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(knn_simple_null_outputs) {
    Random rng(42);
    auto mat_eigen = random_sparse_csr(10, 5, 0.1, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> dummy_indices(10);
    std::vector<scl_real_t> dummy_distances(10);
    
    // NULL indices
    scl_error_t err1 = scl_knn_simple(mat, 10, 2, nullptr, dummy_distances.data());
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    // NULL distances
    scl_error_t err2 = scl_knn_simple(mat, 10, 2, dummy_indices.data(), nullptr);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(knn_simple_zero_k) {
    Random rng(42);
    auto mat_eigen = random_sparse_csr(10, 5, 0.1, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> indices_out(10);
    std::vector<scl_real_t> distances_out(10);
    
    scl_error_t err = scl_knn_simple(mat, 10, 0, indices_out.data(), distances_out.data());
    
    // Should either handle gracefully or return error
    (void)err;
}

SCL_TEST_CASE(knn_simple_k_too_large) {
    Random rng(42);
    auto mat_eigen = random_sparse_csr(10, 5, 0.1, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> indices_out(100);
    std::vector<scl_real_t> distances_out(100);
    
    // k > n_samples
    scl_error_t err = scl_knn_simple(mat, 10, 20, indices_out.data(), distances_out.data());
    
    // Should either handle gracefully or return error
    (void)err;
}

SCL_TEST_SUITE_END

// =============================================================================
// KNN Tests (with precomputed norms)
// =============================================================================

SCL_TEST_SUITE(knn_with_norms)

SCL_TEST_CASE(knn_with_norms_basic) {
    // 4 points in 2D
    std::vector<scl_index_t> indptr = {0, 2, 4, 6, 8};
    std::vector<scl_index_t> indices = {0, 1, 0, 1, 0, 1, 0, 1};
    std::vector<scl_real_t> data = {
        0.0, 0.0,  // Point 0: (0, 0)
        1.0, 0.0,  // Point 1: (1, 0)
        0.0, 1.0,  // Point 2: (0, 1)
        1.0, 1.0   // Point 3: (1, 1)
    };
    
    Sparse mat = make_sparse_csr(4, 2, 8, indptr.data(), indices.data(), data.data());
    
    // Precompute norms
    std::vector<scl_real_t> norms_sq(4);
    scl_neighbors_compute_norms(mat, norms_sq.data());
    
    scl_size_t k = 2;
    std::vector<scl_index_t> indices_out(4 * k);
    std::vector<scl_real_t> distances_out(4 * k);
    
    scl_error_t err = scl_knn(mat, norms_sq.data(), 4, k, indices_out.data(), distances_out.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify outputs are valid
    for (scl_index_t i = 0; i < 4; ++i) {
        for (scl_size_t j = 0; j < k; ++j) {
            size_t idx = i * k + j;
            SCL_ASSERT_GE(distances_out[idx], 0.0);
            SCL_ASSERT_LT(indices_out[idx], 4);
        }
    }
}

SCL_TEST_RETRY(knn_with_norms_random, 3)
{
    Random rng(456);
    
    scl_index_t n_samples = rng.uniform_int(10, 30);
    scl_index_t n_features = rng.uniform_int(5, 20);
    double density = random_density(0.1, 0.3, rng);
    scl_size_t k = rng.uniform_int(2, std::min(static_cast<scl_size_t>(n_samples - 1), static_cast<scl_size_t>(5)));
    
    auto mat_eigen = random_sparse_csr(n_samples, n_features, density, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Precompute norms
    std::vector<scl_real_t> norms_sq(n_samples);
    scl_neighbors_compute_norms(mat, norms_sq.data());
    
    std::vector<scl_index_t> indices_out(n_samples * k);
    std::vector<scl_real_t> distances_out(n_samples * k);
    
    scl_error_t err = scl_knn(mat, norms_sq.data(), n_samples, k, indices_out.data(), distances_out.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Compare with simple version (should give same results)
    std::vector<scl_index_t> indices_simple(n_samples * k);
    std::vector<scl_real_t> distances_simple(n_samples * k);
    
    scl_knn_simple(mat, n_samples, k, indices_simple.data(), distances_simple.data());
    
    // Results should be very similar (allowing for small numerical differences)
    for (size_t i = 0; i < n_samples * k; ++i) {
        // Indices might differ slightly due to ties, but distances should be close
        SCL_ASSERT_TRUE(
            std::abs(distances_out[i] - distances_simple[i]) < 1e-6 ||
            precision::approx_equal(distances_out[i], distances_simple[i], Tolerance::normal())
        );
    }
}

SCL_TEST_CASE(knn_with_norms_null_matrix) {
    std::vector<scl_real_t> norms_sq(10);
    std::vector<scl_index_t> indices_out(20);
    std::vector<scl_real_t> distances_out(20);
    
    scl_error_t err = scl_knn(nullptr, norms_sq.data(), 10, 2, indices_out.data(), distances_out.data());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(knn_with_norms_null_norms) {
    Random rng(42);
    auto mat_eigen = random_sparse_csr(10, 5, 0.1, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> indices_out(20);
    std::vector<scl_real_t> distances_out(20);
    
    scl_error_t err = scl_knn(mat, nullptr, 10, 2, indices_out.data(), distances_out.data());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(knn_with_norms_null_outputs) {
    Random rng(42);
    auto mat_eigen = random_sparse_csr(10, 5, 0.1, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> norms_sq(10);
    scl_neighbors_compute_norms(mat, norms_sq.data());
    
    std::vector<scl_index_t> dummy_indices(20);
    std::vector<scl_real_t> dummy_distances(20);
    
    // NULL indices
    scl_error_t err1 = scl_knn(mat, norms_sq.data(), 10, 2, nullptr, dummy_distances.data());
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    // NULL distances
    scl_error_t err2 = scl_knn(mat, norms_sq.data(), 10, 2, dummy_indices.data(), nullptr);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

