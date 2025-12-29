// =============================================================================
// SCL Kernel - Effect Size Tests
// =============================================================================
//
// 测试 scl/binding/c_api/comparison.h 中的效应量函数
//
// 函数:
//   ✓ scl_comp_effect_size (Cohen's d)
//   ✓ scl_comp_glass_delta (Glass's delta)
//
// 参考实现: 标准统计公式
// 精度要求: Tolerance::statistical() (rtol=1e-4, atol=1e-6)
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"

using namespace scl::test;
using precision::Tolerance;

// Reference implementation
namespace reference {

/// Cohen's d: (mean1 - mean2) / pooled_stddev
inline double cohens_d(const std::vector<scl_real_t>& group1,
                       const std::vector<scl_real_t>& group2) {
    size_t n1 = group1.size();
    size_t n2 = group2.size();
    
    // Means
    double sum1 = 0, sum2 = 0;
    for (auto v : group1) sum1 += v;
    for (auto v : group2) sum2 += v;
    double mean1 = sum1 / n1;
    double mean2 = sum2 / n2;
    
    // Variances
    double var1 = 0, var2 = 0;
    for (auto v : group1) {
        double diff = v - mean1;
        var1 += diff * diff;
    }
    for (auto v : group2) {
        double diff = v - mean2;
        var2 += diff * diff;
    }
    var1 /= (n1 - 1);
    var2 /= (n2 - 1);
    
    // Pooled stddev
    double pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
    double pooled_std = std::sqrt(pooled_var);
    
    if (pooled_std < 1e-15) return 0.0;
    
    return (mean1 - mean2) / pooled_std;
}

/// Glass's delta: (mean_treatment - mean_control) / sd_control
inline double glass_delta(const std::vector<scl_real_t>& control,
                         const std::vector<scl_real_t>& treatment) {
    size_t n_ctrl = control.size();
    size_t n_treat = treatment.size();
    
    // Means
    double sum_ctrl = 0, sum_treat = 0;
    for (auto v : control) sum_ctrl += v;
    for (auto v : treatment) sum_treat += v;
    double mean_ctrl = sum_ctrl / n_ctrl;
    double mean_treat = sum_treat / n_treat;
    
    // Control stddev
    double var_ctrl = 0;
    for (auto v : control) {
        double diff = v - mean_ctrl;
        var_ctrl += diff * diff;
    }
    var_ctrl /= (n_ctrl - 1);
    double sd_ctrl = std::sqrt(var_ctrl);
    
    if (sd_ctrl < 1e-15) return 0.0;
    
    return (mean_treat - mean_ctrl) / sd_ctrl;
}

} // namespace reference

SCL_TEST_BEGIN

// =============================================================================
// Cohen's d Tests
// =============================================================================

SCL_TEST_SUITE(cohens_d)

SCL_TEST_CASE(cohens_d_identical_groups) {
    std::vector<scl_real_t> group1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<scl_real_t> group2 = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    scl_real_t effect_size;
    scl_error_t err = scl_comp_effect_size(
        group1.data(), group1.size(),
        group2.data(), group2.size(),
        &effect_size
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NEAR(effect_size, 0.0, 1e-6);
}

SCL_TEST_CASE(cohens_d_different_groups) {
    std::vector<scl_real_t> group1 = {1.0, 2.0, 3.0};
    std::vector<scl_real_t> group2 = {4.0, 5.0, 6.0};
    
    scl_real_t effect_size;
    scl_error_t err = scl_comp_effect_size(
        group1.data(), group1.size(),
        group2.data(), group2.size(),
        &effect_size
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Compute reference
    double ref = reference::cohens_d(group1, group2);
    SCL_ASSERT_TRUE(precision::approx_equal(effect_size, ref, Tolerance::statistical()));
}

SCL_TEST_RETRY(cohens_d_random_groups, 5)
{
    Random rng(42);
    
    // Random sizes
    size_t n1 = rng.uniform_int(10, 50);
    size_t n2 = rng.uniform_int(10, 50);
    
    // Generate groups
    auto group1 = random_vector(n1, rng);
    auto group2 = random_vector(n2, rng);
    
    // SCL implementation
    scl_real_t effect_size;
    scl_error_t err = scl_comp_effect_size(
        group1.data(), group1.size(),
        group2.data(), group2.size(),
        &effect_size
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Reference
    double ref = reference::cohens_d(group1, group2);
    
    SCL_ASSERT_TRUE(precision::approx_equal(effect_size, ref, Tolerance::statistical()));
}

SCL_TEST_CASE(cohens_d_null_inputs) {
    std::vector<scl_real_t> group(5);
    scl_real_t result;
    
    // NULL group1
    scl_error_t err1 = scl_comp_effect_size(nullptr, 5, group.data(), 5, &result);
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    // NULL group2
    scl_error_t err2 = scl_comp_effect_size(group.data(), 5, nullptr, 5, &result);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
    
    // NULL output
    scl_error_t err3 = scl_comp_effect_size(group.data(), 5, group.data(), 5, nullptr);
    SCL_ASSERT_EQ(err3, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(cohens_d_invalid_sizes) {
    std::vector<scl_real_t> group1(5), group2(5);
    scl_real_t result;
    
    // Zero size
    scl_error_t err = scl_comp_effect_size(
        group1.data(), 0, group2.data(), 5, &result
    );
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(cohens_d_single_element_groups) {
    std::vector<scl_real_t> group1 = {1.0};
    std::vector<scl_real_t> group2 = {2.0};
    
    scl_real_t result;
    scl_error_t err = scl_comp_effect_size(
        group1.data(), 1, group2.data(), 1, &result
    );
    
    // Should handle gracefully (variance undefined for n=1)
    // Either OK with special value or error
    (void)err;
}

SCL_TEST_SUITE_END

// =============================================================================
// Glass's Delta Tests
// =============================================================================

SCL_TEST_SUITE(glass_delta)

SCL_TEST_RETRY(glass_delta_random, 5)
{
    Random rng(999);
    
    size_t n_ctrl = rng.uniform_int(20, 50);
    size_t n_treat = rng.uniform_int(20, 50);
    
    auto control = random_vector(n_ctrl, rng);
    auto treatment = random_vector(n_treat, rng);
    
    scl_real_t delta;
    scl_error_t err = scl_comp_glass_delta(
        control.data(), control.size(),
        treatment.data(), treatment.size(),
        &delta
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Reference
    double ref = reference::glass_delta(control, treatment);
    
    SCL_ASSERT_TRUE(precision::approx_equal(delta, ref, Tolerance::statistical()));
}

SCL_TEST_CASE(glass_delta_null_safety) {
    std::vector<scl_real_t> vec(10);
    scl_real_t result;
    
    SCL_ASSERT_EQ(
        scl_comp_glass_delta(nullptr, 10, vec.data(), 10, &result),
        SCL_ERROR_NULL_POINTER
    );
    
    SCL_ASSERT_EQ(
        scl_comp_glass_delta(vec.data(), 10, nullptr, 10, &result),
        SCL_ERROR_NULL_POINTER
    );
    
    SCL_ASSERT_EQ(
        scl_comp_glass_delta(vec.data(), 10, vec.data(), 10, nullptr),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Statistical Validation (Monte Carlo)
// =============================================================================

SCL_TEST_TAGGED(effect_size_stability, "slow", "monte_carlo")
{
    std::vector<double> cohens_d_errors;
    std::vector<double> glass_delta_errors;
    
    for (int trial = 0; trial < 50; ++trial) {
        Random rng(trial * 1000);
        
        size_t n1 = rng.uniform_int(30, 100);
        size_t n2 = rng.uniform_int(30, 100);
        
        auto group1 = random_vector(n1, rng);
        auto group2 = random_vector(n2, rng);
        
        // Cohen's d
        scl_real_t d_scl;
        scl_comp_effect_size(group1.data(), n1, group2.data(), n2, &d_scl);
        double d_ref = reference::cohens_d(group1, group2);
        cohens_d_errors.push_back(std::abs(d_scl - d_ref));
        
        // Glass's delta
        scl_real_t delta_scl;
        scl_comp_glass_delta(group1.data(), n1, group2.data(), n2, &delta_scl);
        double delta_ref = reference::glass_delta(group1, group2);
        glass_delta_errors.push_back(std::abs(delta_scl - delta_ref));
    }
    
    // Check both are stable
    auto stats_d = precision::compute_statistics(cohens_d_errors);
    auto stats_delta = precision::compute_statistics(glass_delta_errors);
    
    SCL_ASSERT_TRUE(precision::error_stats_acceptable(stats_d, Tolerance::statistical()));
    SCL_ASSERT_TRUE(precision::error_stats_acceptable(stats_delta, Tolerance::statistical()));
}

SCL_TEST_END

SCL_TEST_MAIN()

