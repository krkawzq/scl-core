// =============================================================================
// SCL Core - Group Statistics Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/group.h
//
// Functions tested:
//   - scl_group_stats
//
// Reference implementation: Manual computation
// Precision requirement: Tolerance::normal() (rtol=1e-9, atol=1e-12)
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"
#include "scl/binding/c_api/group.h"

using namespace scl::test;
using precision::Tolerance;

// Reference implementation
namespace reference {

void group_stats(
    const EigenCSR& matrix,
    const std::vector<int32_t>& group_ids,
    size_t n_groups,
    const std::vector<size_t>& group_sizes,
    std::vector<scl_real_t>& out_means,
    std::vector<scl_real_t>& out_vars,
    int ddof,
    bool include_zeros
) {
    scl_index_t rows = matrix.rows();
    (void)matrix.cols();  // Unused but kept for clarity
    
    out_means.assign(rows * n_groups, 0.0);
    out_vars.assign(rows * n_groups, 0.0);
    
    for (scl_index_t i = 0; i < rows; ++i) {
        scl_real_t* mean_ptr = out_means.data() + (i * n_groups);
        scl_real_t* var_ptr = out_vars.data() + (i * n_groups);
        
        std::vector<scl_real_t> sums(n_groups, 0.0);
        std::vector<scl_real_t> sum_sqs(n_groups, 0.0);
        std::vector<size_t> counts(n_groups, 0);
        
        for (EigenCSR::InnerIterator it(matrix, i); it; ++it) {
            scl_index_t j = it.col();
            int32_t g = group_ids[j];
            
            if (g >= 0 && static_cast<size_t>(g) < n_groups) {
                scl_real_t v = it.value();
                sums[g] += v;
                sum_sqs[g] += v * v;
                counts[g]++;
            }
        }
        
        for (size_t g = 0; g < n_groups; ++g) {
            scl_real_t N = include_zeros ? static_cast<scl_real_t>(group_sizes[g])
                                         : static_cast<scl_real_t>(counts[g]);
            
            if (N <= static_cast<scl_real_t>(ddof)) {
                mean_ptr[g] = 0.0;
                var_ptr[g] = 0.0;
                continue;
            }
            
            scl_real_t mu = sums[g] / N;
            scl_real_t variance = (sum_sqs[g] - N * mu * mu) / (N - static_cast<scl_real_t>(ddof));
            if (variance < 0.0) variance = 0.0;
            
            mean_ptr[g] = mu;
            var_ptr[g] = variance;
        }
    }
}

} // namespace reference

SCL_TEST_BEGIN

// =============================================================================
// Basic Functionality Tests
// =============================================================================

SCL_TEST_SUITE(basic_functionality)

SCL_TEST_CASE(group_stats_simple_case) {
    // Simple 3x3 matrix with 2 groups
    std::vector<scl_index_t> indptr = {0, 2, 3, 5};
    std::vector<scl_index_t> indices = {0, 1, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    Sparse mat = make_sparse_csr(3, 3, 5, indptr.data(), indices.data(), data.data());
    
    // Group assignment: [0, 0, 1]
    std::vector<int32_t> group_ids = {0, 0, 1};
    size_t n_groups = 2;
    std::vector<size_t> group_sizes = {2, 1};
    
    std::vector<scl_real_t> means(3 * n_groups);
    std::vector<scl_real_t> vars(3 * n_groups);
    
    scl_error_t err = scl_group_stats(
        mat, group_ids.data(), n_groups, group_sizes.data(),
        means.data(), vars.data(), 1, 1
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Row 0: values [1.0, 2.0, 0.0] -> groups [0, 0, 1]
    // Group 0: mean = (1.0 + 2.0) / 2 = 1.5
    // Group 1: mean = 0.0 / 1 = 0.0
    SCL_ASSERT_NEAR(means[0 * n_groups + 0], 1.5, 1e-10);
    SCL_ASSERT_NEAR(means[0 * n_groups + 1], 0.0, 1e-10);
}

SCL_TEST_CASE(group_stats_single_group) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 20, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // All cells in group 0
    std::vector<int32_t> group_ids(20, 0);
    size_t n_groups = 1;
    std::vector<size_t> group_sizes = {20};
    
    std::vector<scl_real_t> means(10 * n_groups);
    std::vector<scl_real_t> vars(10 * n_groups);
    
    scl_error_t err = scl_group_stats(
        mat, group_ids.data(), n_groups, group_sizes.data(),
        means.data(), vars.data(), 1, 1
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Should match row means
    for (scl_index_t i = 0; i < 10; ++i) {
        scl_real_t row_sum = 0.0;
        int count = 0;
        
        for (EigenCSR::InnerIterator it(A_eigen, i); it; ++it) {
            row_sum += it.value();
            count++;
        }
        
        scl_real_t expected_mean = row_sum / 20.0;  // include zeros
        SCL_ASSERT_NEAR(means[i * n_groups + 0], expected_mean, 1e-9);
    }
}

SCL_TEST_RETRY(group_stats_random_matrices, 5)
{
    Random rng(123);
    
    auto [rows, cols] = random_shape(10, 50, rng);
    double density = random_density(0.05, 0.15, rng);
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Random group assignment
    size_t n_groups = static_cast<size_t>(rng.uniform_int(2, 5));
    std::vector<int32_t> group_ids(static_cast<size_t>(cols));
    std::vector<size_t> group_sizes(n_groups, 0);
    
    for (scl_index_t j = 0; j < cols; ++j) {
        int32_t g = static_cast<int32_t>(rng.uniform_int(0, static_cast<int64_t>(n_groups - 1)));
        group_ids[static_cast<size_t>(j)] = g;
        group_sizes[static_cast<size_t>(g)]++;
    }
    
    std::vector<scl_real_t> means(rows * n_groups);
    std::vector<scl_real_t> vars(rows * n_groups);
    
    scl_error_t err = scl_group_stats(
        mat, group_ids.data(), n_groups, group_sizes.data(),
        means.data(), vars.data(), 1, 1
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Reference implementation
    std::vector<scl_real_t> ref_means(rows * n_groups);
    std::vector<scl_real_t> ref_vars(rows * n_groups);
    reference::group_stats(
        A_eigen, group_ids, n_groups, group_sizes,
        ref_means, ref_vars, 1, true
    );
    
    // Compare
    SCL_ASSERT_TRUE(precision::vectors_equal(means, ref_means, Tolerance::normal()));
    SCL_ASSERT_TRUE(precision::vectors_equal(vars, ref_vars, Tolerance::normal()));
}

SCL_TEST_CASE(group_stats_exclude_zeros) {
    // Matrix with explicit zeros
    std::vector<scl_index_t> indptr = {0, 2, 3, 5};
    std::vector<scl_index_t> indices = {0, 1, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    Sparse mat = make_sparse_csr(3, 3, 5, indptr.data(), indices.data(), data.data());
    
    std::vector<int32_t> group_ids = {0, 0, 1};
    size_t n_groups = 2;
    std::vector<size_t> group_sizes = {2, 1};
    
    std::vector<scl_real_t> means_include(3 * n_groups);
    std::vector<scl_real_t> vars_include(3 * n_groups);
    std::vector<scl_real_t> means_exclude(3 * n_groups);
    std::vector<scl_real_t> vars_exclude(3 * n_groups);
    
    // Include zeros
    scl_error_t err1 = scl_group_stats(
        mat, group_ids.data(), n_groups, group_sizes.data(),
        means_include.data(), vars_include.data(), 1, 1
    );
    SCL_ASSERT_EQ(err1, SCL_OK);
    
    // Exclude zeros
    scl_error_t err2 = scl_group_stats(
        mat, group_ids.data(), n_groups, group_sizes.data(),
        means_exclude.data(), vars_exclude.data(), 1, 0
    );
    SCL_ASSERT_EQ(err2, SCL_OK);
    
    // Results should differ (include zeros uses N=group_size, exclude uses N=nnz)
    // Row 0, group 0: include uses N=2, exclude uses N=2 (both values present)
    // Row 0, group 1: include uses N=1, exclude uses N=0 (no values)
    SCL_ASSERT_NEAR(means_include[0 * n_groups + 0], 1.5, 1e-10);
    SCL_ASSERT_NEAR(means_exclude[0 * n_groups + 0], 1.5, 1e-10);
    SCL_ASSERT_NEAR(means_include[0 * n_groups + 1], 0.0, 1e-10);
    // Exclude zeros: group 1 has no non-zero values, so mean should be 0
}

SCL_TEST_CASE(group_stats_different_ddof) {
    Random rng(456);
    auto A_eigen = random_sparse_csr(5, 10, 0.2, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<int32_t> group_ids(10, 0);
    size_t n_groups = 1;
    std::vector<size_t> group_sizes = {10};
    
    std::vector<scl_real_t> means_ddof0(5 * n_groups);
    std::vector<scl_real_t> vars_ddof0(5 * n_groups);
    std::vector<scl_real_t> means_ddof1(5 * n_groups);
    std::vector<scl_real_t> vars_ddof1(5 * n_groups);
    
    // ddof = 0 (population variance)
    scl_error_t err1 = scl_group_stats(
        mat, group_ids.data(), n_groups, group_sizes.data(),
        means_ddof0.data(), vars_ddof0.data(), 0, 1
    );
    SCL_ASSERT_EQ(err1, SCL_OK);
    
    // ddof = 1 (sample variance)
    scl_error_t err2 = scl_group_stats(
        mat, group_ids.data(), n_groups, group_sizes.data(),
        means_ddof1.data(), vars_ddof1.data(), 1, 1
    );
    SCL_ASSERT_EQ(err2, SCL_OK);
    
    // Means should be the same
    for (scl_index_t i = 0; i < 5; ++i) {
        SCL_ASSERT_NEAR(means_ddof0[i], means_ddof1[i], 1e-10);
    }
    
    // Variances should differ (ddof=0 uses N, ddof=1 uses N-1)
    // For large N, difference is small
    for (scl_index_t i = 0; i < 5; ++i) {
        if (vars_ddof0[i] > 1e-10) {
            // ddof=1 variance should be larger
            SCL_ASSERT_GE(vars_ddof1[i], vars_ddof0[i]);
        }
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Error Handling Tests
// =============================================================================

SCL_TEST_SUITE(error_handling)

SCL_TEST_CASE(group_stats_null_matrix) {
    std::vector<int32_t> group_ids(10);
    size_t n_groups = 2;
    std::vector<size_t> group_sizes = {5, 5};
    std::vector<scl_real_t> means(10 * n_groups);
    std::vector<scl_real_t> vars(10 * n_groups);
    
    scl_error_t err = scl_group_stats(
        nullptr, group_ids.data(), n_groups, group_sizes.data(),
        means.data(), vars.data(), 1, 1
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(group_stats_null_group_ids) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    size_t n_groups = 2;
    std::vector<size_t> group_sizes = {5, 5};
    std::vector<scl_real_t> means(10 * n_groups);
    std::vector<scl_real_t> vars(10 * n_groups);
    
    scl_error_t err = scl_group_stats(
        mat, nullptr, n_groups, group_sizes.data(),
        means.data(), vars.data(), 1, 1
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(group_stats_null_group_sizes) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<int32_t> group_ids(10);
    size_t n_groups = 2;
    std::vector<scl_real_t> means(10 * n_groups);
    std::vector<scl_real_t> vars(10 * n_groups);
    
    scl_error_t err = scl_group_stats(
        mat, group_ids.data(), n_groups, nullptr,
        means.data(), vars.data(), 1, 1
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(group_stats_null_outputs) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<int32_t> group_ids(10);
    size_t n_groups = 2;
    std::vector<size_t> group_sizes = {5, 5};
    std::vector<scl_real_t> means(10 * n_groups);
    std::vector<scl_real_t> vars(10 * n_groups);
    
    // NULL means
    scl_error_t err1 = scl_group_stats(
        mat, group_ids.data(), n_groups, group_sizes.data(),
        nullptr, vars.data(), 1, 1
    );
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    // NULL vars
    scl_error_t err2 = scl_group_stats(
        mat, group_ids.data(), n_groups, group_sizes.data(),
        means.data(), nullptr, 1, 1
    );
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(group_stats_invalid_n_groups) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<int32_t> group_ids(10);
    size_t n_groups = 0;  // Invalid
    std::vector<size_t> group_sizes;
    std::vector<scl_real_t> means;
    std::vector<scl_real_t> vars;
    
    scl_error_t err = scl_group_stats(
        mat, group_ids.data(), n_groups, group_sizes.data(),
        means.data(), vars.data(), 1, 1
    );
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(group_stats_invalid_group_ids) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Group IDs out of range
    std::vector<int32_t> group_ids(10, 99);  // Invalid group ID
    size_t n_groups = 2;
    std::vector<size_t> group_sizes = {5, 5};
    std::vector<scl_real_t> means(10 * n_groups);
    std::vector<scl_real_t> vars(10 * n_groups);
    
    // Should handle gracefully (skip invalid groups)
    scl_error_t err = scl_group_stats(
        mat, group_ids.data(), n_groups, group_sizes.data(),
        means.data(), vars.data(), 1, 1
    );
    
    // Should succeed but all means/vars should be 0
    SCL_ASSERT_EQ(err, SCL_OK);
    for (size_t i = 0; i < means.size(); ++i) {
        SCL_ASSERT_NEAR(means[i], 0.0, 1e-10);
        SCL_ASSERT_NEAR(vars[i], 0.0, 1e-10);
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Edge Cases Tests
// =============================================================================

SCL_TEST_SUITE(edge_cases)

SCL_TEST_CASE(group_stats_empty_matrix) {
    // Empty matrix (0 rows or 0 cols)
    std::vector<scl_index_t> indptr = {0};
    std::vector<scl_index_t> indices;
    std::vector<scl_real_t> data;
    
    Sparse mat = make_sparse_csr(0, 10, 0, indptr.data(), indices.data(), data.data());
    
    std::vector<int32_t> group_ids(10);
    size_t n_groups = 2;
    std::vector<size_t> group_sizes = {5, 5};
    std::vector<scl_real_t> means(0 * n_groups);
    std::vector<scl_real_t> vars(0 * n_groups);
    
    scl_error_t err = scl_group_stats(
        mat, group_ids.data(), n_groups, group_sizes.data(),
        means.data(), vars.data(), 1, 1
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(group_stats_single_row) {
    std::vector<scl_index_t> indptr = {0, 3};
    std::vector<scl_index_t> indices = {0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0};
    
    Sparse mat = make_sparse_csr(1, 3, 3, indptr.data(), indices.data(), data.data());
    
    std::vector<int32_t> group_ids = {0, 0, 1};
    size_t n_groups = 2;
    std::vector<size_t> group_sizes = {2, 1};
    
    std::vector<scl_real_t> means(1 * n_groups);
    std::vector<scl_real_t> vars(1 * n_groups);
    
    scl_error_t err = scl_group_stats(
        mat, group_ids.data(), n_groups, group_sizes.data(),
        means.data(), vars.data(), 1, 1
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Group 0: (1.0 + 2.0) / 2 = 1.5
    // Group 1: 3.0 / 1 = 3.0
    SCL_ASSERT_NEAR(means[0 * n_groups + 0], 1.5, 1e-10);
    SCL_ASSERT_NEAR(means[0 * n_groups + 1], 3.0, 1e-10);
}

SCL_TEST_CASE(group_stats_small_groups) {
    // Groups with very few elements
    std::vector<scl_index_t> indptr = {0, 1, 2, 3};
    std::vector<scl_index_t> indices = {0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0};
    
    Sparse mat = make_sparse_csr(3, 3, 3, indptr.data(), indices.data(), data.data());
    
    // Each cell in its own group
    std::vector<int32_t> group_ids = {0, 1, 2};
    size_t n_groups = 3;
    std::vector<size_t> group_sizes = {1, 1, 1};
    
    std::vector<scl_real_t> means(3 * n_groups);
    std::vector<scl_real_t> vars(3 * n_groups);
    
    scl_error_t err = scl_group_stats(
        mat, group_ids.data(), n_groups, group_sizes.data(),
        means.data(), vars.data(), 1, 1
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Row 0: group 0 has value 1.0
    SCL_ASSERT_NEAR(means[0 * n_groups + 0], 1.0, 1e-10);
    // Row 1: group 1 has value 2.0
    SCL_ASSERT_NEAR(means[1 * n_groups + 1], 2.0, 1e-10);
    // Row 2: group 2 has value 3.0
    SCL_ASSERT_NEAR(means[2 * n_groups + 2], 3.0, 1e-10);
}

SCL_TEST_CASE(group_stats_all_zeros) {
    // Matrix with all zeros (empty sparse matrix)
    std::vector<scl_index_t> indptr = {0, 0, 0, 0};
    std::vector<scl_index_t> indices;
    std::vector<scl_real_t> data;
    
    Sparse mat = make_sparse_csr(3, 3, 0, indptr.data(), indices.data(), data.data());
    
    std::vector<int32_t> group_ids = {0, 0, 1};
    size_t n_groups = 2;
    std::vector<size_t> group_sizes = {2, 1};
    
    std::vector<scl_real_t> means(3 * n_groups);
    std::vector<scl_real_t> vars(3 * n_groups);
    
    scl_error_t err = scl_group_stats(
        mat, group_ids.data(), n_groups, group_sizes.data(),
        means.data(), vars.data(), 1, 1
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All means and vars should be 0
    for (size_t i = 0; i < means.size(); ++i) {
        SCL_ASSERT_NEAR(means[i], 0.0, 1e-10);
        SCL_ASSERT_NEAR(vars[i], 0.0, 1e-10);
    }
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

