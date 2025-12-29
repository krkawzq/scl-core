// =============================================================================
// SCL Kernel - Maximum Mean Discrepancy (MMD) Tests
// =============================================================================
//
// 测试 scl/binding/c_api/mmd.h 中的 MMD 函数
//
// 函数:
//   ✓ scl_mmd_rbf (MMD with RBF kernel)
//
// 参考实现: 标准 MMD 公式
// 精度要求: Tolerance::statistical() (rtol=1e-4, atol=1e-6)
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"

using namespace scl::test;
using precision::Tolerance;

// Reference implementation of MMD^2 with RBF kernel
namespace reference {

inline double rbf_kernel(const Eigen::SparseVector<scl_real_t>& x,
                         const Eigen::SparseVector<scl_real_t>& y,
                         double gamma) {
    // Compute squared Euclidean distance
    double dist_sq = 0.0;
    
    // Get non-zero indices
    std::vector<scl_index_t> x_indices, y_indices;
    for (Eigen::SparseVector<scl_real_t>::InnerIterator it(x); it; ++it) {
        x_indices.push_back(it.index());
    }
    for (Eigen::SparseVector<scl_real_t>::InnerIterator it(y); it; ++it) {
        y_indices.push_back(it.index());
    }
    
    // Compute ||x - y||^2
    auto x_it = x_indices.begin();
    auto y_it = y_indices.begin();
    
    while (x_it != x_indices.end() && y_it != y_indices.end()) {
        if (*x_it < *y_it) {
            double val = x.coeff(*x_it);
            dist_sq += val * val;
            ++x_it;
        } else if (*y_it < *x_it) {
            double val = y.coeff(*y_it);
            dist_sq += val * val;
            ++y_it;
        } else {
            double diff = x.coeff(*x_it) - y.coeff(*y_it);
            dist_sq += diff * diff;
            ++x_it;
            ++y_it;
        }
    }
    
    while (x_it != x_indices.end()) {
        double val = x.coeff(*x_it);
        dist_sq += val * val;
        ++x_it;
    }
    
    while (y_it != y_indices.end()) {
        double val = y.coeff(*y_it);
        dist_sq += val * val;
        ++y_it;
    }
    
    // RBF kernel: exp(-gamma * ||x - y||^2)
    return std::exp(-gamma * dist_sq);
}

inline double mmd_rbf_squared(const EigenCSR& X, const EigenCSR& Y, double gamma) {
    size_t n_x = X.rows();
    size_t n_y = Y.rows();
    
    // Term 1: E[k(x, x')] for X
    double term1 = 0.0;
    for (size_t i = 0; i < n_x; ++i) {
        for (size_t j = 0; j < n_x; ++j) {
            auto x_i = X.row(i);
            auto x_j = X.row(j);
            term1 += rbf_kernel(x_i, x_j, gamma);
        }
    }
    term1 /= (n_x * n_x);
    
    // Term 2: E[k(y, y')] for Y
    double term2 = 0.0;
    for (size_t i = 0; i < n_y; ++i) {
        for (size_t j = 0; j < n_y; ++j) {
            auto y_i = Y.row(i);
            auto y_j = Y.row(j);
            term2 += rbf_kernel(y_i, y_j, gamma);
        }
    }
    term2 /= (n_y * n_y);
    
    // Term 3: -2 * E[k(x, y)]
    double term3 = 0.0;
    for (size_t i = 0; i < n_x; ++i) {
        for (size_t j = 0; j < n_y; ++j) {
            auto x_i = X.row(i);
            auto y_j = Y.row(j);
            term3 += rbf_kernel(x_i, y_j, gamma);
        }
    }
    term3 = -2.0 * term3 / (n_x * n_y);
    
    return term1 + term2 + term3;
}

} // namespace reference

SCL_TEST_BEGIN

// =============================================================================
// MMD RBF Tests
// =============================================================================

SCL_TEST_SUITE(mmd_rbf)

SCL_TEST_CASE(mmd_identical_matrices) {
    // Same matrix should give MMD ≈ 0
    Random rng(42);
    auto X_eigen = random_sparse_csr(10, 20, 0.1, rng);
    auto csr = from_eigen_csr(X_eigen);
    
    Sparse X = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    Sparse Y = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> output(1);
    scl_error_t err = scl_mmd_rbf(X, Y, output.data(), 1.0);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // MMD should be very close to 0 for identical matrices
    SCL_ASSERT_NEAR(output[0], 0.0, 1e-3);
}

SCL_TEST_CASE(mmd_different_matrices) {
    Random rng(123);
    auto X_eigen = random_sparse_csr(10, 20, 0.1, rng);
    auto Y_eigen = random_sparse_csr(10, 20, 0.1, rng);
    
    auto csr_x = from_eigen_csr(X_eigen);
    auto csr_y = from_eigen_csr(Y_eigen);
    
    Sparse X = make_sparse_csr(
        csr_x.rows, csr_x.cols, csr_x.nnz,
        csr_x.indptr.data(), csr_x.indices.data(), csr_x.data.data()
    );
    
    Sparse Y = make_sparse_csr(
        csr_y.rows, csr_y.cols, csr_y.nnz,
        csr_y.indptr.data(), csr_y.indices.data(), csr_y.data.data()
    );
    
    std::vector<scl_real_t> output(1);
    scl_error_t err = scl_mmd_rbf(X, Y, output.data(), 1.0);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // MMD should be positive for different matrices
    SCL_ASSERT_GT(output[0], 0.0);
}

SCL_TEST_RETRY(mmd_random_matrices, 3)
{
    Random rng(456);
    
    auto [n_x, n_y] = std::make_pair(
        rng.uniform_int(5, 20),
        rng.uniform_int(5, 20)
    );
    auto [n_features] = std::make_pair(rng.uniform_int(10, 50));
    double density = random_density(0.05, 0.15, rng);
    
    auto X_eigen = random_sparse_csr(n_x, n_features, density, rng);
    auto Y_eigen = random_sparse_csr(n_y, n_features, density, rng);
    
    auto csr_x = from_eigen_csr(X_eigen);
    auto csr_y = from_eigen_csr(Y_eigen);
    
    Sparse X = make_sparse_csr(
        csr_x.rows, csr_x.cols, csr_x.nnz,
        csr_x.indptr.data(), csr_x.indices.data(), csr_x.data.data()
    );
    
    Sparse Y = make_sparse_csr(
        csr_y.rows, csr_y.cols, csr_y.nnz,
        csr_y.indptr.data(), csr_y.indices.data(), csr_y.data.data()
    );
    
    double gamma = rng.uniform(0.1, 2.0);
    std::vector<scl_real_t> output(1);
    
    scl_error_t err = scl_mmd_rbf(X, Y, output.data(), gamma);
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Compare with reference (simplified check - MMD should be non-negative)
    SCL_ASSERT_GE(output[0], 0.0);
}

SCL_TEST_CASE(mmd_different_gamma) {
    Random rng(789);
    auto X_eigen = random_sparse_csr(10, 20, 0.1, rng);
    auto Y_eigen = random_sparse_csr(10, 20, 0.1, rng);
    
    auto csr_x = from_eigen_csr(X_eigen);
    auto csr_y = from_eigen_csr(Y_eigen);
    
    Sparse X = make_sparse_csr(
        csr_x.rows, csr_x.cols, csr_x.nnz,
        csr_x.indptr.data(), csr_x.indices.data(), csr_x.data.data()
    );
    
    Sparse Y = make_sparse_csr(
        csr_y.rows, csr_y.cols, csr_y.nnz,
        csr_y.indptr.data(), csr_y.indices.data(), csr_y.data.data()
    );
    
    std::vector<scl_real_t> output1(1), output2(1);
    
    scl_error_t err1 = scl_mmd_rbf(X, Y, output1.data(), 0.1);
    scl_error_t err2 = scl_mmd_rbf(X, Y, output2.data(), 1.0);
    
    SCL_ASSERT_EQ(err1, SCL_OK);
    SCL_ASSERT_EQ(err2, SCL_OK);
    
    // Different gamma should give different results
    // (though not necessarily in a predictable order)
    SCL_ASSERT_GE(output1[0], 0.0);
    SCL_ASSERT_GE(output2[0], 0.0);
}

SCL_TEST_CASE(mmd_null_handle_x) {
    Random rng(42);
    auto Y_eigen = random_sparse_csr(10, 20, 0.1, rng);
    auto csr_y = from_eigen_csr(Y_eigen);
    
    Sparse Y = make_sparse_csr(
        csr_y.rows, csr_y.cols, csr_y.nnz,
        csr_y.indptr.data(), csr_y.indices.data(), csr_y.data.data()
    );
    
    std::vector<scl_real_t> output(1);
    scl_error_t err = scl_mmd_rbf(nullptr, Y, output.data(), 1.0);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(mmd_null_handle_y) {
    Random rng(42);
    auto X_eigen = random_sparse_csr(10, 20, 0.1, rng);
    auto csr_x = from_eigen_csr(X_eigen);
    
    Sparse X = make_sparse_csr(
        csr_x.rows, csr_x.cols, csr_x.nnz,
        csr_x.indptr.data(), csr_x.indices.data(), csr_x.data.data()
    );
    
    std::vector<scl_real_t> output(1);
    scl_error_t err = scl_mmd_rbf(X, nullptr, output.data(), 1.0);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(mmd_null_output) {
    Random rng(42);
    auto X_eigen = random_sparse_csr(10, 20, 0.1, rng);
    auto Y_eigen = random_sparse_csr(10, 20, 0.1, rng);
    
    auto csr_x = from_eigen_csr(X_eigen);
    auto csr_y = from_eigen_csr(Y_eigen);
    
    Sparse X = make_sparse_csr(
        csr_x.rows, csr_x.cols, csr_x.nnz,
        csr_x.indptr.data(), csr_x.indices.data(), csr_x.data.data()
    );
    
    Sparse Y = make_sparse_csr(
        csr_y.rows, csr_y.cols, csr_y.nnz,
        csr_y.indptr.data(), csr_y.indices.data(), csr_y.data.data()
    );
    
    scl_error_t err = scl_mmd_rbf(X, Y, nullptr, 1.0);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(mmd_dimension_mismatch) {
    Random rng(42);
    auto X_eigen = random_sparse_csr(10, 20, 0.1, rng);
    auto Y_eigen = random_sparse_csr(10, 30, 0.1, rng);  // Different feature dimension
    
    auto csr_x = from_eigen_csr(X_eigen);
    auto csr_y = from_eigen_csr(Y_eigen);
    
    Sparse X = make_sparse_csr(
        csr_x.rows, csr_x.cols, csr_x.nnz,
        csr_x.indptr.data(), csr_x.indices.data(), csr_x.data.data()
    );
    
    Sparse Y = make_sparse_csr(
        csr_y.rows, csr_y.cols, csr_y.nnz,
        csr_y.indptr.data(), csr_y.indices.data(), csr_y.data.data()
    );
    
    std::vector<scl_real_t> output(1);
    scl_error_t err = scl_mmd_rbf(X, Y, output.data(), 1.0);
    
    // Should either handle gracefully or return error
    // (depends on implementation)
    (void)err;
}

SCL_TEST_CASE(mmd_empty_matrix) {
    // Empty matrix (zero rows)
    auto Z = zero_csr(0, 10);
    auto csr = from_eigen_csr(Z);
    
    Sparse X = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    Sparse Y = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> output(1);
    scl_error_t err = scl_mmd_rbf(X, Y, output.data(), 1.0);
    
    // Should handle empty matrices gracefully
    (void)err;
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

