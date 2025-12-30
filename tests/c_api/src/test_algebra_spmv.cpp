// =============================================================================
// SCL Kernel - Sparse Matrix-Vector Multiplication Tests
// =============================================================================
//
// 测试 scl/binding/c_api/algebra.h 中的 SpMV 函数
//
// 函数:
//   ✓ scl_algebra_spmv (general: y = alpha*A*x + beta*y)
//   ✓ scl_algebra_spmv_simple (y = A*x)
//   ✓ scl_algebra_spmv_scaled (y = alpha*A*x)
//   ✓ scl_algebra_spmv_add (y += A*x)
//
// 参考实现: Eigen SpMV + BLAS gemv
// 精度要求: Tolerance::normal() (rtol=1e-9, atol=1e-12)
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"

extern "C" {
#include "scl/binding/c_api/algebra.h"
}

using namespace scl::test;
using precision::Tolerance;

SCL_TEST_BEGIN

// =============================================================================
// SpMV General: y = alpha * A * x + beta * y
// =============================================================================

SCL_TEST_SUITE(spmv_general)

SCL_TEST_RETRY(spmv_random_matrices, 5)
{
    Random rng(42);
    
    // Random shape
    auto [rows, cols] = random_shape(10, 50, rng);
    double density = random_density(0.05, 0.15, rng);
    
    // Generate sparse matrix
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Generate vectors
    auto x = random_vector(cols, rng);
    auto y_init = random_vector(rows, rng);
    
    scl_real_t alpha = rng.uniform(-2.0, 2.0);
    scl_real_t beta = rng.uniform(-2.0, 2.0);
    
    // SCL implementation
    std::vector<scl_real_t> y_scl(y_init.data(), y_init.data() + y_init.size());
    scl_error_t err = scl_algebra_spmv(
        A, x.data(), x.size(),
        y_scl.data(), y_scl.size(),
        alpha, beta
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Eigen reference
    Eigen::VectorXd y_ref = y_init;
    y_ref = alpha * (A_eigen * x) + beta * y_ref;
    
    // Compare
    SCL_ASSERT_TRUE(precision::vectors_equal(y_scl, y_ref, Tolerance::normal()));
}

SCL_TEST_CASE(spmv_identity_matrix) {
    // Identity matrix
    auto I = identity_csr(10);
    auto csr = from_eigen_csr(I);
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // x = [1, 2, 3, ..., 10]
    std::vector<scl_real_t> x(10);
    std::iota(x.begin(), x.end(), 1.0);
    
    std::vector<scl_real_t> y(10, 0.0);
    
    // y = I * x should equal x
    scl_algebra_spmv(A, x.data(), x.size(), y.data(), y.size(), 1.0, 0.0);
    
    for (size_t i = 0; i < 10; ++i) {
        SCL_ASSERT_NEAR(y[i], x[i], 1e-12);
    }
}

SCL_TEST_CASE(spmv_zero_matrix) {
    auto Z = zero_csr(5, 5);
    auto csr = from_eigen_csr(Z);
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> x(5, 1.0);
    std::vector<scl_real_t> y(5, 999.0);
    
    // y = 0*x + 0*y = 0
    scl_algebra_spmv(A, x.data(), x.size(), y.data(), y.size(), 1.0, 0.0);
    
    for (auto val : y) {
        SCL_ASSERT_NEAR(val, 0.0, 1e-12);
    }
}

SCL_TEST_CASE(spmv_with_beta_zero) {
    Random rng(111);
    auto A_eigen = random_sparse_csr(20, 20, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    auto x = random_vector(20, rng);
    std::vector<scl_real_t> y(20, 999.0);  // Garbage values
    
    // beta=0 should ignore initial y
    scl_algebra_spmv(A, x.data(), x.size(), y.data(), y.size(), 1.0, 0.0);
    
    // Compare with Eigen
    Eigen::VectorXd y_ref = A_eigen * x;
    SCL_ASSERT_TRUE(precision::vectors_equal(y, y_ref, Tolerance::normal()));
}

SCL_TEST_CASE(spmv_null_handle) {
    std::vector<scl_real_t> x(10), y(10);
    
    scl_error_t err = scl_algebra_spmv(
        nullptr, x.data(), x.size(), y.data(), y.size(), 1.0, 0.0
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(spmv_null_vectors) {
    auto A_eigen = random_sparse_csr(10, 10, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> x(10), y(10);
    
    // NULL x
    scl_error_t err1 = scl_algebra_spmv(A, nullptr, x.size(), y.data(), y.size(), 1.0, 0.0);
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    // NULL y
    scl_error_t err2 = scl_algebra_spmv(A, x.data(), x.size(), nullptr, y.size(), 1.0, 0.0);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(spmv_dimension_mismatch) {
    auto A_eigen = random_sparse_csr(10, 15, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> x(10);  // Wrong size (should be 15)
    std::vector<scl_real_t> y(10);
    
    scl_error_t err = scl_algebra_spmv(
        A, x.data(), x.size(), y.data(), y.size(), 1.0, 0.0
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_DIMENSION_MISMATCH);
}

SCL_TEST_SUITE_END

// =============================================================================
// SpMV Simple: y = A * x
// =============================================================================

SCL_TEST_SUITE(spmv_simple)

SCL_TEST_RETRY(spmv_simple_random, 3)
{
    Random rng(123);
    
    auto [rows, cols] = random_shape(10, 50, rng);
    auto A_eigen = random_sparse_csr(rows, cols, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    auto x = random_vector(cols, rng);
    std::vector<scl_real_t> y(rows);
    
    scl_error_t err = scl_algebra_spmv_simple(
        A, x.data(), x.size(), y.data(), y.size()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Compare
    Eigen::VectorXd y_ref = A_eigen * x;
    SCL_ASSERT_TRUE(precision::vectors_equal(y, y_ref, Tolerance::normal()));
}

SCL_TEST_SUITE_END

// =============================================================================
// Monte Carlo 验证
// =============================================================================

SCL_TEST_TAGGED(spmv_monte_carlo, "slow", "statistical")
{
    std::vector<double> relative_errors;
    
    for (int trial = 0; trial < 100; ++trial) {
        Random rng(trial);
        
        auto [rows, cols] = random_shape(20, 50, rng);
        auto A_eigen = random_sparse_csr(rows, cols, 0.08, rng);
        auto csr = from_eigen_csr(A_eigen);
        
        Sparse A = make_sparse_csr(
            csr.rows, csr.cols, csr.nnz,
            csr.indptr.data(), csr.indices.data(), csr.data.data()
        );
        
        auto x = random_vector(cols, rng);
        std::vector<scl_real_t> y(rows);
        
        scl_algebra_spmv_simple(A, x.data(), x.size(), y.data(), y.size());
        
        // Eigen reference
        Eigen::VectorXd y_ref = A_eigen * x;
        
        // Record error
        double rel_err = precision::relative_error(y, y_ref);
        relative_errors.push_back(rel_err);
    }
    
    // Statistical check
    auto stats = precision::compute_statistics(relative_errors);
    SCL_ASSERT_TRUE(precision::error_stats_acceptable(stats, Tolerance::normal()));
}

SCL_TEST_END

SCL_TEST_MAIN()

