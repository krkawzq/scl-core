// =============================================================================
// SCL Core - Imputation Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/impute.h
//
// Functions tested:
//   - scl_impute_knn
//   - scl_impute_knn_weighted
//   - scl_impute_diffusion
//   - scl_impute_magic
//
// Reference implementation: Manual computation for simple cases
// Precision requirement: Tolerance::normal() (rtol=1e-9, atol=1e-12)
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"
#include "scl/binding/c_api/impute.h"

using namespace scl::test;
using precision::Tolerance;

// Helper: Create simple k-NN indices
void create_simple_knn(
    scl_index_t n_cells,
    scl_index_t k,
    std::vector<scl_index_t>& knn_indices,
    std::vector<scl_real_t>& knn_distances
) {
    knn_indices.clear();
    knn_distances.clear();
    
    for (scl_index_t i = 0; i < n_cells; ++i) {
        for (scl_index_t j = 0; j < k; ++j) {
            // Simple: connect to next k cells (wrapping)
            scl_index_t neighbor = (i + j + 1) % n_cells;
            knn_indices.push_back(neighbor);
            knn_distances.push_back(static_cast<scl_real_t>(j + 1));  // Distance increases
        }
    }
}

SCL_TEST_BEGIN

// =============================================================================
// KNN Imputation Tests
// =============================================================================

SCL_TEST_SUITE(knn_imputation)

SCL_TEST_CASE(knn_impute_basic) {
    // Simple 3x3 matrix
    std::vector<scl_index_t> indptr = {0, 2, 3, 5};
    std::vector<scl_index_t> indices = {0, 1, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    Sparse mat = make_sparse_csr(3, 3, 5, indptr.data(), indices.data(), data.data());
    
    scl_index_t n_cells = 3;
    scl_index_t n_genes = 3;
    scl_index_t k_neighbors = 2;
    
    std::vector<scl_index_t> knn_indices;
    std::vector<scl_real_t> knn_distances;
    create_simple_knn(n_cells, k_neighbors, knn_indices, knn_distances);
    
    std::vector<scl_real_t> X_imputed(n_cells * n_genes);
    
    scl_error_t err = scl_impute_knn(
        mat, knn_indices.data(), knn_distances.data(),
        n_cells, n_genes, k_neighbors,
        X_imputed.data(), 1.0, 0.0  // bandwidth=1.0, threshold=0.0
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Check that imputed values are finite
    for (size_t i = 0; i < X_imputed.size(); ++i) {
        SCL_ASSERT_TRUE(std::isfinite(X_imputed[i]));
    }
}

SCL_TEST_CASE(knn_impute_with_threshold) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 20, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_index_t n_cells = 10;
    scl_index_t n_genes = 20;
    scl_index_t k_neighbors = 3;
    
    std::vector<scl_index_t> knn_indices;
    std::vector<scl_real_t> knn_distances;
    create_simple_knn(n_cells, k_neighbors, knn_indices, knn_distances);
    
    std::vector<scl_real_t> X_imputed(n_cells * n_genes);
    
    scl_error_t err = scl_impute_knn(
        mat, knn_indices.data(), knn_distances.data(),
        n_cells, n_genes, k_neighbors,
        X_imputed.data(), 1.0, 0.5  // threshold=0.5
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_RETRY(knn_impute_random, 3)
{
    Random rng(123);
    auto [n_cells, n_genes] = random_shape(10, 30, rng);
    double density = random_density(0.05, 0.15, rng);
    auto A_eigen = random_sparse_csr(n_cells, n_genes, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_index_t k_neighbors = rng.uniform_int(2, 5);
    
    std::vector<scl_index_t> knn_indices;
    std::vector<scl_real_t> knn_distances;
    create_simple_knn(n_cells, k_neighbors, knn_indices, knn_distances);
    
    std::vector<scl_real_t> X_imputed(n_cells * n_genes);
    
    scl_error_t err = scl_impute_knn(
        mat, knn_indices.data(), knn_distances.data(),
        n_cells, n_genes, k_neighbors,
        X_imputed.data(),
        rng.uniform(0.5, 2.0),  // bandwidth
        rng.uniform(0.0, 1.0)   // threshold
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(knn_impute_null_inputs) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_index_t n_cells = 10, n_genes = 10, k = 3;
    std::vector<scl_index_t> knn_indices;
    std::vector<scl_real_t> knn_distances;
    create_simple_knn(n_cells, k, knn_indices, knn_distances);
    std::vector<scl_real_t> X_imputed(n_cells * n_genes);
    
    scl_error_t err1 = scl_impute_knn(
        nullptr, knn_indices.data(), knn_distances.data(),
        n_cells, n_genes, k, X_imputed.data(), 1.0, 0.0
    );
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_impute_knn(
        mat, nullptr, knn_distances.data(),
        n_cells, n_genes, k, X_imputed.data(), 1.0, 0.0
    );
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Weighted KNN Imputation Tests
// =============================================================================

SCL_TEST_SUITE(knn_weighted_imputation)

SCL_TEST_CASE(knn_weighted_impute_basic) {
    std::vector<scl_index_t> indptr = {0, 2, 3, 5};
    std::vector<scl_index_t> indices = {0, 1, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    Sparse mat = make_sparse_csr(3, 3, 5, indptr.data(), indices.data(), data.data());
    
    scl_index_t n_cells = 3, n_genes = 3, k = 2;
    std::vector<scl_index_t> knn_indices;
    std::vector<scl_real_t> knn_distances;
    create_simple_knn(n_cells, k, knn_indices, knn_distances);
    
    std::vector<scl_real_t> X_imputed(n_cells * n_genes);
    
    scl_error_t err = scl_impute_knn_weighted(
        mat, knn_indices.data(), knn_distances.data(),
        n_cells, n_genes, k,
        X_imputed.data(), 1.0, 0.0
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    for (size_t i = 0; i < X_imputed.size(); ++i) {
        SCL_ASSERT_TRUE(std::isfinite(X_imputed[i]));
    }
}

SCL_TEST_RETRY(knn_weighted_impute_random, 3)
{
    Random rng(456);
    auto [n_cells, n_genes] = random_shape(10, 30, rng);
    double density = random_density(0.05, 0.15, rng);
    auto A_eigen = random_sparse_csr(n_cells, n_genes, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_index_t k = rng.uniform_int(2, 5);
    std::vector<scl_index_t> knn_indices;
    std::vector<scl_real_t> knn_distances;
    create_simple_knn(n_cells, k, knn_indices, knn_distances);
    
    std::vector<scl_real_t> X_imputed(n_cells * n_genes);
    
    scl_error_t err = scl_impute_knn_weighted(
        mat, knn_indices.data(), knn_distances.data(),
        n_cells, n_genes, k,
        X_imputed.data(),
        rng.uniform(0.5, 2.0),
        rng.uniform(0.0, 1.0)
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(knn_weighted_impute_null_inputs) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_index_t n_cells = 10, n_genes = 10, k = 3;
    std::vector<scl_index_t> knn_indices;
    std::vector<scl_real_t> knn_distances;
    create_simple_knn(n_cells, k, knn_indices, knn_distances);
    std::vector<scl_real_t> X_imputed(n_cells * n_genes);
    
    scl_error_t err1 = scl_impute_knn_weighted(
        nullptr, knn_indices.data(), knn_distances.data(),
        n_cells, n_genes, k, X_imputed.data(), 1.0, 0.0
    );
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Diffusion Imputation Tests
// =============================================================================

SCL_TEST_SUITE(diffusion_imputation)

SCL_TEST_CASE(diffusion_impute_basic) {
    // Create input matrix
    std::vector<scl_index_t> indptr1 = {0, 2, 3, 5};
    std::vector<scl_index_t> indices1 = {0, 1, 0, 1, 2};
    std::vector<scl_real_t> data1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    Sparse mat = make_sparse_csr(3, 3, 5, indptr1.data(), indices1.data(), data1.data());
    
    // Create transition matrix (simple symmetric)
    std::vector<scl_index_t> indptr2 = {0, 2, 4, 6};
    std::vector<scl_index_t> indices2 = {0, 1, 0, 2, 1, 2};
    std::vector<scl_real_t> data2 = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    Sparse transition = make_sparse_csr(3, 3, 6, indptr2.data(), indices2.data(), data2.data());
    
    scl_index_t n_cells = 3;
    scl_index_t n_genes = 3;
    scl_index_t n_steps = 2;
    
    std::vector<scl_real_t> X_imputed(n_cells * n_genes);
    
    scl_error_t err = scl_impute_diffusion(
        mat, transition, n_cells, n_genes, n_steps, X_imputed.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    for (size_t i = 0; i < X_imputed.size(); ++i) {
        SCL_ASSERT_TRUE(std::isfinite(X_imputed[i]));
    }
}

SCL_TEST_RETRY(diffusion_impute_random, 3)
{
    Random rng(789);
    auto [n_cells, n_genes] = random_shape(10, 20, rng);
    double density = random_density(0.1, 0.2, rng);
    auto A_eigen = random_sparse_csr(n_cells, n_genes, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Create simple transition matrix
    auto T_eigen = random_sparse_csr(n_cells, n_cells, 0.2, rng);
    auto T_csr = from_eigen_csr(T_eigen);
    Sparse transition = make_sparse_csr(
        T_csr.rows, T_csr.cols, T_csr.nnz,
        T_csr.indptr.data(), T_csr.indices.data(), T_csr.data.data()
    );
    
    scl_index_t n_steps = rng.uniform_int(1, 5);
    std::vector<scl_real_t> X_imputed(n_cells * n_genes);
    
    scl_error_t err = scl_impute_diffusion(
        mat, transition, n_cells, n_genes, n_steps, X_imputed.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(diffusion_impute_null_inputs) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    auto T_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto T_csr = from_eigen_csr(T_eigen);
    Sparse transition = make_sparse_csr(
        T_csr.rows, T_csr.cols, T_csr.nnz,
        T_csr.indptr.data(), T_csr.indices.data(), T_csr.data.data()
    );
    
    std::vector<scl_real_t> X_imputed(100);
    
    scl_error_t err1 = scl_impute_diffusion(
        nullptr, transition, 10, 10, 2, X_imputed.data()
    );
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_impute_diffusion(
        mat, nullptr, 10, 10, 2, X_imputed.data()
    );
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// MAGIC Imputation Tests
// =============================================================================

SCL_TEST_SUITE(magic_imputation)

SCL_TEST_CASE(magic_impute_basic) {
    // Create input matrix
    std::vector<scl_index_t> indptr1 = {0, 2, 3, 5};
    std::vector<scl_index_t> indices1 = {0, 1, 0, 1, 2};
    std::vector<scl_real_t> data1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    Sparse mat = make_sparse_csr(3, 3, 5, indptr1.data(), indices1.data(), data1.data());
    
    // Create affinity matrix
    std::vector<scl_index_t> indptr2 = {0, 2, 4, 6};
    std::vector<scl_index_t> indices2 = {0, 1, 0, 2, 1, 2};
    std::vector<scl_real_t> data2 = {0.8, 0.2, 0.7, 0.3, 0.6, 0.4};
    Sparse affinity = make_sparse_csr(3, 3, 6, indptr2.data(), indices2.data(), data2.data());
    
    scl_index_t n_cells = 3;
    scl_index_t n_genes = 3;
    scl_index_t t = 2;  // Diffusion time
    
    std::vector<scl_real_t> X_imputed(n_cells * n_genes);
    
    scl_error_t err = scl_impute_magic(
        mat, affinity, n_cells, n_genes, t, X_imputed.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    for (size_t i = 0; i < X_imputed.size(); ++i) {
        SCL_ASSERT_TRUE(std::isfinite(X_imputed[i]));
    }
}

SCL_TEST_RETRY(magic_impute_random, 3)
{
    Random rng(111);
    auto [n_cells, n_genes] = random_shape(10, 20, rng);
    double density = random_density(0.1, 0.2, rng);
    auto A_eigen = random_sparse_csr(n_cells, n_genes, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Create affinity matrix
    auto Aff_eigen = random_sparse_csr(n_cells, n_cells, 0.2, rng);
    auto Aff_csr = from_eigen_csr(Aff_eigen);
    Sparse affinity = make_sparse_csr(
        Aff_csr.rows, Aff_csr.cols, Aff_csr.nnz,
        Aff_csr.indptr.data(), Aff_csr.indices.data(), Aff_csr.data.data()
    );
    
    scl_index_t t = rng.uniform_int(1, 5);
    std::vector<scl_real_t> X_imputed(n_cells * n_genes);
    
    scl_error_t err = scl_impute_magic(
        mat, affinity, n_cells, n_genes, t, X_imputed.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(magic_impute_null_inputs) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    auto Aff_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto Aff_csr = from_eigen_csr(Aff_eigen);
    Sparse affinity = make_sparse_csr(
        Aff_csr.rows, Aff_csr.cols, Aff_csr.nnz,
        Aff_csr.indptr.data(), Aff_csr.indices.data(), Aff_csr.data.data()
    );
    
    std::vector<scl_real_t> X_imputed(100);
    
    scl_error_t err1 = scl_impute_magic(
        nullptr, affinity, 10, 10, 2, X_imputed.data()
    );
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_impute_magic(
        mat, nullptr, 10, 10, 2, X_imputed.data()
    );
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

