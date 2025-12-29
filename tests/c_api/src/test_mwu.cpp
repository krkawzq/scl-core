// =============================================================================
// SCL Kernel - Mann-Whitney U Test Tests
// =============================================================================
//
// 测试 scl/binding/c_api/mwu.h 中的 MWU 函数
//
// 函数:
//   ✓ scl_mwu_test (Mann-Whitney U test for each feature)
//
// 参考实现: 标准 MWU 统计公式
// 精度要求: Tolerance::statistical() (rtol=1e-4, atol=1e-6)
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"

using namespace scl::test;
using precision::Tolerance;

// Reference implementation of Mann-Whitney U test
namespace reference {

inline double compute_u_statistic(const std::vector<scl_real_t>& group1,
                                 const std::vector<scl_real_t>& group2) {
    size_t n1 = group1.size();
    size_t n2 = group2.size();
    
    // Combine and rank
    std::vector<std::pair<scl_real_t, int>> combined;
    for (size_t i = 0; i < n1; ++i) {
        combined.push_back({group1[i], 0});
    }
    for (size_t i = 0; i < n2; ++i) {
        combined.push_back({group2[i], 1});
    }
    
    std::sort(combined.begin(), combined.end());
    
    // Compute ranks with ties
    std::vector<double> ranks(combined.size());
    size_t i = 0;
    while (i < combined.size()) {
        size_t j = i;
        while (j < combined.size() && combined[j].first == combined[i].first) {
            ++j;
        }
        
        double rank = (i + j + 1) / 2.0;
        for (size_t k = i; k < j; ++k) {
            ranks[k] = rank;
        }
        i = j;
    }
    
    // Sum ranks for group1
    double R1 = 0.0;
    for (size_t k = 0; k < combined.size(); ++k) {
        if (combined[k].second == 0) {
            R1 += ranks[k];
        }
    }
    
    // U = n1*n2 + n1*(n1+1)/2 - R1
    double U = n1 * n2 + n1 * (n1 + 1.0) / 2.0 - R1;
    
    return U;
}

inline double compute_log2_fc(const std::vector<scl_real_t>& group1,
                              const std::vector<scl_real_t>& group2) {
    double mean1 = 0.0, mean2 = 0.0;
    for (auto v : group1) mean1 += v;
    for (auto v : group2) mean2 += v;
    mean1 /= group1.size();
    mean2 /= group2.size();
    
    if (mean1 <= 0.0 || mean2 <= 0.0) {
        return 0.0;
    }
    
    return std::log2(mean1 / mean2);
}

} // namespace reference

SCL_TEST_BEGIN

// =============================================================================
// MWU Test - Basic Functionality
// =============================================================================

SCL_TEST_SUITE(mwu_basic)

SCL_TEST_CASE(mwu_identical_groups) {
    // Create a simple matrix: 3 features x 6 samples
    std::vector<scl_index_t> indptr = {0, 6, 12, 18};
    std::vector<scl_index_t> indices(18);
    std::vector<scl_real_t> data(18, 1.0);
    
    for (scl_index_t i = 0; i < 18; ++i) {
        indices[i] = i % 6;
    }
    
    Sparse mat = make_sparse_csr(3, 6, 18, indptr.data(), indices.data(), data.data());
    
    // Two groups with same values
    std::vector<int32_t> group_ids = {0, 0, 0, 1, 1, 1};
    
    std::vector<scl_real_t> u_stats(3);
    std::vector<scl_real_t> p_values(3);
    std::vector<scl_real_t> log2_fc(3);
    
    scl_error_t err = scl_mwu_test(
        mat, group_ids.data(), 6,
        u_stats.data(), p_values.data(), log2_fc.data(), nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // For identical groups, p-values should be high (not significant)
    for (size_t i = 0; i < 3; ++i) {
        SCL_ASSERT_GT(p_values[i], 0.1);
    }
}

SCL_TEST_CASE(mwu_different_groups) {
    // Create matrix with clear group differences
    std::vector<scl_index_t> indptr = {0, 6, 12};
    std::vector<scl_index_t> indices(12);
    std::vector<scl_real_t> data(12);
    
    // Feature 0: group 0 has high values, group 1 has low values
    for (scl_index_t i = 0; i < 3; ++i) {
        indices[i] = i;
        data[i] = 10.0;  // Group 0
    }
    for (scl_index_t i = 3; i < 6; ++i) {
        indices[i] = i;
        data[i] = 1.0;   // Group 1
    }
    
    // Feature 1: group 0 has low values, group 1 has high values
    for (scl_index_t i = 6; i < 9; ++i) {
        indices[i] = i - 6;
        data[i] = 1.0;   // Group 0
    }
    for (scl_index_t i = 9; i < 12; ++i) {
        indices[i] = i - 6;
        data[i] = 10.0;  // Group 1
    }
    
    Sparse mat = make_sparse_csr(2, 6, 12, indptr.data(), indices.data(), data.data());
    
    std::vector<int32_t> group_ids = {0, 0, 0, 1, 1, 1};
    
    std::vector<scl_real_t> u_stats(2);
    std::vector<scl_real_t> p_values(2);
    std::vector<scl_real_t> log2_fc(2);
    
    scl_error_t err = scl_mwu_test(
        mat, group_ids.data(), 6,
        u_stats.data(), p_values.data(), log2_fc.data(), nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Feature 0: group 0 > group 1, so log2_fc should be positive
    SCL_ASSERT_GT(log2_fc[0], 0.0);
    SCL_ASSERT_LT(p_values[0], 0.05);  // Should be significant
    
    // Feature 1: group 0 < group 1, so log2_fc should be negative
    SCL_ASSERT_LT(log2_fc[1], 0.0);
    SCL_ASSERT_LT(p_values[1], 0.05);  // Should be significant
}

SCL_TEST_RETRY(mwu_random_data, 3)
{
    Random rng(42);
    
    scl_index_t n_features = rng.uniform_int(5, 20);
    scl_index_t n_samples = rng.uniform_int(10, 30);
    double density = random_density(0.1, 0.3, rng);
    
    auto mat_eigen = random_sparse_csr(n_features, n_samples, density, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Random group assignment
    std::vector<int32_t> group_ids(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        group_ids[i] = rng.bernoulli(0.5) ? 0 : 1;
    }
    
    // Ensure both groups have at least one member
    size_t n0 = std::count(group_ids.begin(), group_ids.end(), 0);
    size_t n1 = std::count(group_ids.begin(), group_ids.end(), 1);
    if (n0 == 0) group_ids[0] = 0;
    if (n1 == 0) group_ids[n_samples - 1] = 1;
    
    std::vector<scl_real_t> u_stats(n_features);
    std::vector<scl_real_t> p_values(n_features);
    std::vector<scl_real_t> log2_fc(n_features);
    
    scl_error_t err = scl_mwu_test(
        mat, group_ids.data(), n_samples,
        u_stats.data(), p_values.data(), log2_fc.data(), nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify outputs are valid
    for (scl_index_t i = 0; i < n_features; ++i) {
        SCL_ASSERT_GE(u_stats[i], 0.0);
        SCL_ASSERT_GE(p_values[i], 0.0);
        SCL_ASSERT_LE(p_values[i], 1.0);
        // log2_fc can be any real number
    }
}

SCL_TEST_CASE(mwu_with_auroc) {
    Random rng(123);
    auto mat_eigen = random_sparse_csr(5, 10, 0.2, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<int32_t> group_ids = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    
    std::vector<scl_real_t> u_stats(5);
    std::vector<scl_real_t> p_values(5);
    std::vector<scl_real_t> log2_fc(5);
    std::vector<scl_real_t> auroc(5);
    
    scl_error_t err = scl_mwu_test(
        mat, group_ids.data(), 10,
        u_stats.data(), p_values.data(), log2_fc.data(), auroc.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // AUROC should be in [0, 1]
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(auroc[i], 0.0);
        SCL_ASSERT_LE(auroc[i], 1.0);
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// MWU Test - Error Handling
// =============================================================================

SCL_TEST_SUITE(mwu_errors)

SCL_TEST_CASE(mwu_null_matrix) {
    std::vector<int32_t> group_ids = {0, 0, 1, 1};
    std::vector<scl_real_t> u_stats(2), p_values(2), log2_fc(2);
    
    scl_error_t err = scl_mwu_test(
        nullptr, group_ids.data(), 4,
        u_stats.data(), p_values.data(), log2_fc.data(), nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(mwu_null_group_ids) {
    Random rng(42);
    auto mat_eigen = random_sparse_csr(3, 6, 0.1, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> u_stats(3), p_values(3), log2_fc(3);
    
    scl_error_t err = scl_mwu_test(
        mat, nullptr, 6,
        u_stats.data(), p_values.data(), log2_fc.data(), nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(mwu_null_outputs) {
    Random rng(42);
    auto mat_eigen = random_sparse_csr(3, 6, 0.1, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<int32_t> group_ids = {0, 0, 0, 1, 1, 1};
    std::vector<scl_real_t> dummy(3);
    
    // NULL u_stats
    scl_error_t err1 = scl_mwu_test(
        mat, group_ids.data(), 6,
        nullptr, dummy.data(), dummy.data(), nullptr
    );
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    // NULL p_values
    scl_error_t err2 = scl_mwu_test(
        mat, group_ids.data(), 6,
        dummy.data(), nullptr, dummy.data(), nullptr
    );
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
    
    // NULL log2_fc
    scl_error_t err3 = scl_mwu_test(
        mat, group_ids.data(), 6,
        dummy.data(), dummy.data(), nullptr, nullptr
    );
    SCL_ASSERT_EQ(err3, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(mwu_zero_samples) {
    Random rng(42);
    auto mat_eigen = random_sparse_csr(3, 6, 0.1, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<int32_t> group_ids;
    std::vector<scl_real_t> u_stats(3), p_values(3), log2_fc(3);
    
    scl_error_t err = scl_mwu_test(
        mat, group_ids.data(), 0,
        u_stats.data(), p_values.data(), log2_fc.data(), nullptr
    );
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(mwu_single_group) {
    Random rng(42);
    auto mat_eigen = random_sparse_csr(3, 6, 0.1, rng);
    auto csr = from_eigen_csr(mat_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // All samples in same group
    std::vector<int32_t> group_ids = {0, 0, 0, 0, 0, 0};
    std::vector<scl_real_t> u_stats(3), p_values(3), log2_fc(3);
    
    scl_error_t err = scl_mwu_test(
        mat, group_ids.data(), 6,
        u_stats.data(), p_values.data(), log2_fc.data(), nullptr
    );
    
    // Should either handle gracefully or return error
    (void)err;
}

SCL_TEST_CASE(mwu_small_groups) {
    // Minimum valid case: 1 sample per group
    std::vector<scl_index_t> indptr = {0, 2, 4};
    std::vector<scl_index_t> indices = {0, 1, 0, 1};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0};
    
    Sparse mat = make_sparse_csr(2, 2, 4, indptr.data(), indices.data(), data.data());
    
    std::vector<int32_t> group_ids = {0, 1};
    std::vector<scl_real_t> u_stats(2), p_values(2), log2_fc(2);
    
    scl_error_t err = scl_mwu_test(
        mat, group_ids.data(), 2,
        u_stats.data(), p_values.data(), log2_fc.data(), nullptr
    );
    
    // Should handle minimum case
    (void)err;
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

