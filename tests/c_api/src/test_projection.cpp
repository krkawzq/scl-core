// =============================================================================
// SCL Core - Projection Module Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/projection.h
//
// Functions tested:
//   ✓ scl_projection_project
//   ✓ scl_projection_project_auto
//   ✓ scl_projection_gaussian
//   ✓ scl_projection_achlioptas
//   ✓ scl_projection_sparse
//   ✓ scl_projection_countsketch
//   ✓ scl_projection_feature_hash
//   ✓ scl_projection_jl_dimension
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"
#include "scl/binding/c_api/projection.h"

using namespace scl::test;
using precision::Tolerance;

SCL_TEST_BEGIN

// =============================================================================
// JL Dimension Computation
// =============================================================================

SCL_TEST_SUITE(jl_dimension)

SCL_TEST_CASE(jl_dimension_basic) {
    scl_size_t dim = scl_projection_jl_dimension(1000, 0.1);
    
    // Should return a positive dimension
    SCL_ASSERT_GT(dim, 0);
    SCL_ASSERT_LT(dim, 1000);  // Should be less than input
}

SCL_TEST_CASE(jl_dimension_small_epsilon) {
    scl_size_t dim1 = scl_projection_jl_dimension(1000, 0.01);
    scl_size_t dim2 = scl_projection_jl_dimension(1000, 0.1);
    
    // Smaller epsilon should require larger dimension
    SCL_ASSERT_GE(dim1, dim2);
}

SCL_TEST_CASE(jl_dimension_large_n) {
    scl_size_t dim = scl_projection_jl_dimension(1000000, 0.1);
    
    SCL_ASSERT_GT(dim, 0);
    SCL_ASSERT_LT(dim, 1000000);
}

SCL_TEST_CASE(jl_dimension_edge_cases) {
    // Very small n
    scl_size_t dim1 = scl_projection_jl_dimension(10, 0.1);
    SCL_ASSERT_GT(dim1, 0);
    
    // Very small epsilon
    scl_size_t dim2 = scl_projection_jl_dimension(1000, 0.001);
    SCL_ASSERT_GT(dim2, 0);
}

SCL_TEST_SUITE_END

// =============================================================================
// Gaussian Projection
// =============================================================================

SCL_TEST_SUITE(gaussian_projection)

SCL_TEST_RETRY(gaussian_projection_random, 3)
{
    Random rng(42);
    
    auto [rows, cols] = random_shape(20, 100, rng);
    double density = random_density(0.05, 0.15, rng);
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_size_t output_dim = rng.uniform_int(5, std::min(rows, cols) - 1);
    std::vector<scl_real_t> output(rows * output_dim);
    
    scl_error_t err = scl_projection_gaussian(
        A, output_dim, output.data(), rng.uniform_int(0, 10000)
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Check output is not all zeros
    bool has_nonzero = false;
    for (auto val : output) {
        if (std::abs(val) > 1e-10) {
            has_nonzero = true;
            break;
        }
    }
    SCL_ASSERT_TRUE(has_nonzero);
}

SCL_TEST_CASE(gaussian_projection_deterministic) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_size_t output_dim = 5;
    std::vector<scl_real_t> output1(10 * output_dim);
    std::vector<scl_real_t> output2(10 * output_dim);
    
    // Same seed should produce same result
    uint64_t seed = 12345;
    scl_projection_gaussian(A, output_dim, output1.data(), seed);
    scl_projection_gaussian(A, output_dim, output2.data(), seed);
    
    SCL_ASSERT_TRUE(precision::vectors_equal(output1, output2, Tolerance::strict()));
}

SCL_TEST_CASE(gaussian_projection_null_handle) {
    std::vector<scl_real_t> output(100);
    
    scl_error_t err = scl_projection_gaussian(
        nullptr, 10, output.data(), 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(gaussian_projection_null_output) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_projection_gaussian(A, 10, nullptr, 42);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(gaussian_projection_zero_dim) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> output(10);
    
    scl_error_t err = scl_projection_gaussian(A, 0, output.data(), 42);
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(gaussian_projection_large_dim) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> output(10 * 1000);
    
    // Output dim larger than input cols might be invalid
    scl_error_t err = scl_projection_gaussian(A, 1000, output.data(), 42);
    
    // Should either succeed or return appropriate error
    SCL_ASSERT_TRUE(err == SCL_OK || err == SCL_ERROR_INVALID_ARGUMENT);
}

SCL_TEST_SUITE_END

// =============================================================================
// Achlioptas Projection
// =============================================================================

SCL_TEST_SUITE(achlioptas_projection)

SCL_TEST_RETRY(achlioptas_projection_random, 3)
{
    Random rng(43);
    
    auto [rows, cols] = random_shape(20, 100, rng);
    double density = random_density(0.05, 0.15, rng);
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_size_t output_dim = rng.uniform_int(5, std::min(rows, cols) - 1);
    std::vector<scl_real_t> output(rows * output_dim);
    
    scl_error_t err = scl_projection_achlioptas(
        A, output_dim, output.data(), rng.uniform_int(0, 10000)
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(achlioptas_projection_null_handle) {
    std::vector<scl_real_t> output(100);
    
    scl_error_t err = scl_projection_achlioptas(
        nullptr, 10, output.data(), 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(achlioptas_projection_null_output) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_projection_achlioptas(A, 10, nullptr, 42);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Sparse Projection
// =============================================================================

SCL_TEST_SUITE(sparse_projection)

SCL_TEST_RETRY(sparse_projection_random, 3)
{
    Random rng(44);
    
    auto [rows, cols] = random_shape(20, 100, rng);
    double density = random_density(0.05, 0.15, rng);
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_size_t output_dim = rng.uniform_int(5, std::min(rows, cols) - 1);
    scl_real_t proj_density = rng.uniform(0.01, 0.1);
    std::vector<scl_real_t> output(rows * output_dim);
    
    scl_error_t err = scl_projection_sparse(
        A, output_dim, output.data(), proj_density, rng.uniform_int(0, 10000)
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(sparse_projection_invalid_density) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> output(10 * 5);
    
    // Negative density
    scl_error_t err1 = scl_projection_sparse(A, 5, output.data(), -0.1, 42);
    SCL_ASSERT_NE(err1, SCL_OK);
    
    // Density > 1
    scl_error_t err2 = scl_projection_sparse(A, 5, output.data(), 1.5, 42);
    SCL_ASSERT_NE(err2, SCL_OK);
}

SCL_TEST_CASE(sparse_projection_null_handle) {
    std::vector<scl_real_t> output(100);
    
    scl_error_t err = scl_projection_sparse(
        nullptr, 10, output.data(), 0.1, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(sparse_projection_null_output) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_projection_sparse(A, 10, nullptr, 0.1, 42);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// CountSketch Projection
// =============================================================================

SCL_TEST_SUITE(countsketch_projection)

SCL_TEST_RETRY(countsketch_projection_random, 3)
{
    Random rng(45);
    
    auto [rows, cols] = random_shape(20, 100, rng);
    double density = random_density(0.05, 0.15, rng);
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_size_t output_dim = rng.uniform_int(5, std::min(rows, cols) - 1);
    std::vector<scl_real_t> output(rows * output_dim);
    
    scl_error_t err = scl_projection_countsketch(
        A, output_dim, output.data(), rng.uniform_int(0, 10000)
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(countsketch_projection_null_handle) {
    std::vector<scl_real_t> output(100);
    
    scl_error_t err = scl_projection_countsketch(
        nullptr, 10, output.data(), 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(countsketch_projection_null_output) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_projection_countsketch(A, 10, nullptr, 42);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Feature Hash Projection
// =============================================================================

SCL_TEST_SUITE(feature_hash_projection)

SCL_TEST_RETRY(feature_hash_projection_random, 3)
{
    Random rng(46);
    
    auto [rows, cols] = random_shape(20, 100, rng);
    double density = random_density(0.05, 0.15, rng);
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_size_t output_dim = rng.uniform_int(5, std::min(rows, cols) - 1);
    scl_size_t n_hashes = rng.uniform_int(1, 5);
    std::vector<scl_real_t> output(rows * output_dim);
    
    scl_error_t err = scl_projection_feature_hash(
        A, output_dim, output.data(), n_hashes, rng.uniform_int(0, 10000)
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(feature_hash_projection_zero_hashes) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> output(10 * 5);
    
    scl_error_t err = scl_projection_feature_hash(A, 5, output.data(), 0, 42);
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(feature_hash_projection_null_handle) {
    std::vector<scl_real_t> output(100);
    
    scl_error_t err = scl_projection_feature_hash(
        nullptr, 10, output.data(), 2, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(feature_hash_projection_null_output) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_projection_feature_hash(A, 10, nullptr, 2, 42);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Generic Project Function
// =============================================================================

SCL_TEST_SUITE(generic_project)

SCL_TEST_RETRY(project_all_types, 2)
{
    Random rng(47);
    
    auto [rows, cols] = random_shape(20, 50, rng);
    double density = random_density(0.05, 0.15, rng);
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_size_t output_dim = rng.uniform_int(5, std::min(rows, cols) - 1);
    std::vector<scl_real_t> output(rows * output_dim);
    
    // Test all projection types
    scl_projection_type_t types[] = {
        SCL_PROJECTION_GAUSSIAN,
        SCL_PROJECTION_ACHLIOPTAS,
        SCL_PROJECTION_SPARSE,
        SCL_PROJECTION_COUNTSKETCH,
        SCL_PROJECTION_FEATURE_HASH
    };
    
    for (auto type : types) {
        std::fill(output.begin(), output.end(), 0.0);
        
        scl_error_t err = scl_projection_project(
            A, output_dim, output.data(), type, rng.uniform_int(0, 10000)
        );
        
        SCL_ASSERT_EQ(err, SCL_OK);
    }
}

SCL_TEST_CASE(project_invalid_type) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> output(10 * 5);
    
    // Invalid type (assuming max is 4)
    scl_projection_type_t invalid_type = static_cast<scl_projection_type_t>(99);
    scl_error_t err = scl_projection_project(
        A, 5, output.data(), invalid_type, 42
    );
    
    // Should either handle gracefully or return error
    SCL_ASSERT_TRUE(err == SCL_OK || err == SCL_ERROR_INVALID_ARGUMENT);
}

SCL_TEST_CASE(project_null_handle) {
    std::vector<scl_real_t> output(100);
    
    scl_error_t err = scl_projection_project(
        nullptr, 10, output.data(), SCL_PROJECTION_GAUSSIAN, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(project_null_output) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_projection_project(
        A, 10, nullptr, SCL_PROJECTION_GAUSSIAN, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Auto Project Function
// =============================================================================

SCL_TEST_SUITE(auto_project)

SCL_TEST_RETRY(project_auto_random, 3)
{
    Random rng(48);
    
    auto [rows, cols] = random_shape(20, 100, rng);
    double density = random_density(0.05, 0.15, rng);
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_size_t output_dim = rng.uniform_int(5, std::min(rows, cols) - 1);
    std::vector<scl_real_t> output(rows * output_dim);
    
    scl_error_t err = scl_projection_project_auto(
        A, output_dim, output.data(), rng.uniform_int(0, 10000)
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(project_auto_null_handle) {
    std::vector<scl_real_t> output(100);
    
    scl_error_t err = scl_projection_project_auto(
        nullptr, 10, output.data(), 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(project_auto_null_output) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse A = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_projection_project_auto(A, 10, nullptr, 42);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

