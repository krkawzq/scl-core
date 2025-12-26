// =============================================================================
/// @file stats_tests.cpp
/// @brief Statistical Tests (mwu.hpp, ttest.hpp)
///
/// Provides statistical test implementations including Mann-Whitney U and t-tests.
/// Includes both standard (CustomSparse) and mapped (MappedCustomSparse) versions.
// =============================================================================

#include "error.hpp"
#include "scl/kernel/core.hpp"
#include "scl/kernel/mwu_mapped_impl.hpp"
#include "scl/kernel/ttest_mapped_impl.hpp"
#include "scl/io/mmatrix.hpp"

#include <vector>

extern "C" {

// =============================================================================
// Statistical Tests - Standard
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

// =============================================================================
// Statistical Tests - Mapped
// =============================================================================

int scl_mwu_test_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const int32_t* group_ids,
    scl::Real* out_u_stats,
    scl::Real* out_p_values,
    scl::Real* out_log2_fc
) {
    SCL_C_API_WRAPPER(
        auto mapped_data = scl::io::MappedArray<scl::Real>::from_ptr(
            const_cast<scl::Real*>(data), static_cast<scl::Size>(nnz));
        auto mapped_indices = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(indices), static_cast<scl::Size>(nnz));
        auto mapped_indptr = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(indptr), static_cast<scl::Size>(cols + 1));
        
        scl::io::MappedCustomSparse<scl::Real, false> matrix(
            std::move(mapped_data),
            std::move(mapped_indices),
            std::move(mapped_indptr),
            rows, cols
        );
        
        scl::kernel::mwu::mapped::mwu_test_mapped(
            matrix,
            scl::Array<const int32_t>(group_ids, static_cast<scl::Size>(rows)),
            scl::Array<scl::Real>(out_u_stats, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_p_values, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_log2_fc, static_cast<scl::Size>(cols))
        );
    )
}

int scl_ttest_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const int32_t* group_ids,
    scl::Size n_groups,
    scl::Real* out_t_stats,
    scl::Real* out_p_values,
    scl::Real* out_log2_fc,
    bool use_welch
) {
    SCL_C_API_WRAPPER(
        auto mapped_data = scl::io::MappedArray<scl::Real>::from_ptr(
            const_cast<scl::Real*>(data), static_cast<scl::Size>(nnz));
        auto mapped_indices = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(indices), static_cast<scl::Size>(nnz));
        auto mapped_indptr = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(indptr), static_cast<scl::Size>(cols + 1));
        
        scl::io::MappedCustomSparse<scl::Real, false> matrix(
            std::move(mapped_data),
            std::move(mapped_indices),
            std::move(mapped_indptr),
            rows, cols
        );
        
        // Count group sizes
        std::vector<scl::Size> group_sizes(n_groups);
        scl::kernel::diff_expr::count_group_sizes(
            scl::Array<const int32_t>(group_ids, static_cast<scl::Size>(rows)),
            n_groups,
            scl::Array<scl::Size>(group_sizes.data(), n_groups)
        );
        
        scl::Size output_size = static_cast<scl::Size>(cols) * (n_groups - 1);
        scl::kernel::ttest::mapped::ttest_mapped(
            matrix,
            scl::Array<const int32_t>(group_ids, static_cast<scl::Size>(rows)),
            n_groups,
            scl::Array<const scl::Size>(group_sizes.data(), n_groups),
            scl::Array<scl::Real>(out_t_stats, output_size),
            scl::Array<scl::Real>(out_p_values, output_size),
            scl::Array<scl::Real>(out_log2_fc, output_size),
            use_welch
        );
    )
}

} // extern "C"
