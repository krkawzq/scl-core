// =============================================================================
/// @file stats_tests.cpp
/// @brief Statistical Tests (mwu.hpp, ttest.hpp)
///
/// Provides statistical test implementations including Mann-Whitney U and t-tests.
// =============================================================================

#include "error.hpp"
#include "scl/kernel/core.hpp"

extern "C" {

// =============================================================================
// Statistical Tests
// =============================================================================

int scl_mwu_test_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    const int32_t* group_ids,
    scl::Real* out_u_stats,
    scl::Real* out_p_values,
    scl::Real* out_log2_fc
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols
        );
        scl::kernel::mwu::mwu_test(
            matrix,
            scl::Array<const int32_t>(group_ids, static_cast<scl::Size>(rows)),
            scl::Array<scl::Real>(out_u_stats, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_p_values, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_log2_fc, static_cast<scl::Size>(cols))
        );
    )
}

int scl_ttest_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    const int32_t* group_ids,
    scl::Size n_groups,
    scl::Real* out_t_stats,
    scl::Real* out_p_values,
    scl::Real* out_log2_fc,
    bool use_welch
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols
        );

        scl::Size output_size = static_cast<scl::Size>(cols) * (n_groups - 1);
        scl::kernel::diff_expr::ttest(
            matrix,
            scl::Array<const int32_t>(group_ids, static_cast<scl::Size>(rows)),
            n_groups,
            scl::Array<scl::Real>(out_t_stats, output_size),
            scl::Array<scl::Real>(out_p_values, output_size),
            scl::Array<scl::Real>(out_log2_fc, output_size),
            use_welch
        );
    )
}

} // extern "C"
