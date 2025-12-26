#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <cmath>
#include <vector>

// =============================================================================
/// @file ttest_mapped_impl.hpp
/// @brief T-Test for Memory-Mapped Sparse Matrices
///
/// ## Key Optimizations
///
/// 1. Fused Group Statistics (Single Pass)
///    - sum, sum_sq, count per group in one traversal
///    - 4-way unrolled accumulation
///
/// 2. Prefetch Hints
///    - Sequential access pattern for mapped data
///
/// 3. Fast erfc Approximation
///    - Polynomial approximation for p-values
///
/// 4. SIMD Zero Initialization
// =============================================================================

namespace scl::kernel::ttest::mapped {

// =============================================================================
// SECTION 1: Utilities
// =============================================================================

namespace detail {

/// @brief Fast erfc approximation
SCL_FORCE_INLINE Real fast_erfc(Real x) {
    Real sign = Real(1);
    if (x < Real(0)) {
        sign = Real(-1);
        x = -x;
    }
    
    Real t = Real(1) / (Real(1) + Real(0.3275911) * x);
    Real t2 = t * t;
    Real t3 = t2 * t;
    Real t4 = t3 * t;
    Real t5 = t4 * t;
    
    Real poly = Real(0.254829592) * t 
              - Real(0.284496736) * t2 
              + Real(1.421413741) * t3 
              - Real(1.453152027) * t4 
              + Real(1.061405429) * t5;
    
    Real result = poly * std::exp(-x * x);
    return (sign > Real(0)) ? result : (Real(2) - result);
}

/// @brief Compute p-value from t-statistic
SCL_FORCE_INLINE Real t_to_pvalue(Real t_stat) {
    return Real(2) * fast_erfc(std::abs(t_stat) / Real(M_SQRT2));
}

/// @brief SIMD zero fill
SCL_FORCE_INLINE void zero_fill(Real* ptr, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    auto v_zero = s::Zero(d);
    
    Size k = 0;
    for (; k + lanes <= len; k += lanes) {
        s::Store(v_zero, d, ptr + k);
    }
    for (; k < len; ++k) {
        ptr[k] = Real(0);
    }
}

} // namespace detail

// =============================================================================
// SECTION 2: Group Statistics for MappedCustomSparse
// =============================================================================

/// @brief Fused group statistics (sum, sum_sq, count) - MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_group_stats_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Real> out_means,
    Array<Real> out_vars,
    Array<Size> out_counts
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size total_size = static_cast<Size>(primary_dim) * n_groups;

    SCL_CHECK_DIM(out_means.len >= total_size, "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= total_size, "Vars size mismatch");
    SCL_CHECK_DIM(out_counts.len >= total_size, "Counts size mismatch");

    detail::zero_fill(out_means.ptr, total_size);
    detail::zero_fill(out_vars.ptr, total_size);
    std::fill(out_counts.ptr, out_counts.ptr + total_size, Size(0));

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Size len = static_cast<Size>(end - start);

        if (len == 0) return;

        Real* mean_ptr = out_means.ptr + (p * n_groups);
        Real* var_ptr = out_vars.ptr + (p * n_groups);
        Size* count_ptr = out_counts.ptr + (p * n_groups);

        const T* SCL_RESTRICT vals = matrix.data.data() + start;
        const Index* SCL_RESTRICT inds = matrix.indices.data() + start;

        // 4-way unrolled accumulation
        Size k = 0;
        for (; k + 4 <= len; k += 4) {
            for (int u = 0; u < 4; ++u) {
                Index idx = inds[k + u];
                int32_t g = group_ids[idx];

                if (g >= 0 && static_cast<Size>(g) < n_groups) {
                    Real v = static_cast<Real>(vals[k + u]);
                    mean_ptr[g] += v;
                    var_ptr[g] += v * v;
                    count_ptr[g]++;
                }
            }
        }

        for (; k < len; ++k) {
            Index idx = inds[k];
            int32_t g = group_ids[idx];

            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                Real v = static_cast<Real>(vals[k]);
                mean_ptr[g] += v;
                var_ptr[g] += v * v;
                count_ptr[g]++;
            }
        }
    });

    // Finalize: convert sums to means and compute variance
    scl::threading::parallel_for(Size(0), total_size, [&](size_t i) {
        Size n = out_counts[i];
        if (n > 0) {
            Real mean = out_means[i] / static_cast<Real>(n);
            Real sum_sq = out_vars[i];
            Real var = (n > 1) ? (sum_sq / static_cast<Real>(n) - mean * mean) : Real(0);
            if (n > 1) {
                var = var * static_cast<Real>(n) / static_cast<Real>(n - 1);
            }
            out_means[i] = mean;
            out_vars[i] = var;
        }
    });
}

// =============================================================================
// SECTION 3: Group Statistics for MappedVirtualSparse
// =============================================================================

/// @brief Fused group statistics - MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_group_stats_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Real> out_means,
    Array<Real> out_vars,
    Array<Size> out_counts
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size total_size = static_cast<Size>(primary_dim) * n_groups;

    SCL_CHECK_DIM(out_means.len >= total_size, "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= total_size, "Vars size mismatch");
    SCL_CHECK_DIM(out_counts.len >= total_size, "Counts size mismatch");

    detail::zero_fill(out_means.ptr, total_size);
    detail::zero_fill(out_vars.ptr, total_size);
    std::fill(out_counts.ptr, out_counts.ptr + total_size, Size(0));

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        auto values = scl::primary_values(matrix, static_cast<Index>(p));
        auto indices = scl::primary_indices(matrix, static_cast<Index>(p));
        Size len = values.len;

        if (len == 0) return;

        Real* mean_ptr = out_means.ptr + (p * n_groups);
        Real* var_ptr = out_vars.ptr + (p * n_groups);
        Size* count_ptr = out_counts.ptr + (p * n_groups);

        const T* SCL_RESTRICT vals = values.ptr;
        const Index* SCL_RESTRICT inds = indices.ptr;

        Size k = 0;
        for (; k + 4 <= len; k += 4) {
            for (int u = 0; u < 4; ++u) {
                Index idx = inds[k + u];
                int32_t g = group_ids[idx];

                if (g >= 0 && static_cast<Size>(g) < n_groups) {
                    Real v = static_cast<Real>(vals[k + u]);
                    mean_ptr[g] += v;
                    var_ptr[g] += v * v;
                    count_ptr[g]++;
                }
            }
        }

        for (; k < len; ++k) {
            Index idx = inds[k];
            int32_t g = group_ids[idx];

            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                Real v = static_cast<Real>(vals[k]);
                mean_ptr[g] += v;
                var_ptr[g] += v * v;
                count_ptr[g]++;
            }
        }
    });

    // Finalize
    scl::threading::parallel_for(Size(0), total_size, [&](size_t i) {
        Size n = out_counts[i];
        if (n > 0) {
            Real mean = out_means[i] / static_cast<Real>(n);
            Real sum_sq = out_vars[i];
            Real var = (n > 1) ? (sum_sq / static_cast<Real>(n) - mean * mean) : Real(0);
            if (n > 1) {
                var = var * static_cast<Real>(n) / static_cast<Real>(n - 1);
            }
            out_means[i] = mean;
            out_vars[i] = var;
        }
    });
}

// =============================================================================
// SECTION 4: T-Test for Mapped Matrices
// =============================================================================

/// @brief T-test for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void ttest_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_t_stats,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc,
    bool use_welch
) {
    const Index n_features = scl::primary_size(matrix);
    const Size n_targets = n_groups - 1;
    const Size output_size = static_cast<Size>(n_features) * n_targets;

    SCL_CHECK_DIM(out_t_stats.len >= output_size, "T-stats size mismatch");
    SCL_CHECK_DIM(out_p_values.len >= output_size, "P-values size mismatch");
    SCL_CHECK_DIM(out_log2_fc.len >= output_size, "Log2FC size mismatch");

    std::vector<Real> means(static_cast<Size>(n_features) * n_groups);
    std::vector<Real> vars(static_cast<Size>(n_features) * n_groups);
    std::vector<Size> counts(static_cast<Size>(n_features) * n_groups);

    compute_group_stats_mapped(
        matrix, group_ids, n_groups,
        Array<Real>(means.data(), means.size()),
        Array<Real>(vars.data(), vars.size()),
        Array<Size>(counts.data(), counts.size())
    );

    const Size N_ref = group_sizes[0];
    const Real inv_N_ref = (N_ref > 0) ? (Real(1) / static_cast<Real>(N_ref)) : Real(0);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_features), [&](size_t i) {
        Real mean_ref = means[i * n_groups + 0];
        Real var_ref = vars[i * n_groups + 0];

        for (Size t = 0; t < n_targets; ++t) {
            Size target_group = t + 1;
            Size N_target = group_sizes[target_group];

            Real mean_target = means[i * n_groups + target_group];
            Real var_target = vars[i * n_groups + target_group];

            Real mean_diff = mean_target - mean_ref;

            constexpr Real PSEUDOCOUNT = Real(1e-9);
            Real log2_fc = std::log2((mean_target + PSEUDOCOUNT) / (mean_ref + PSEUDOCOUNT));

            Real t_stat = Real(0);
            Real p_value = Real(1);

            if (N_ref > 0 && N_target > 0) {
                Real inv_N_target = Real(1) / static_cast<Real>(N_target);

                if (use_welch) {
                    Real se_sq = var_ref * inv_N_ref + var_target * inv_N_target;
                    if (se_sq > Real(0)) {
                        t_stat = mean_diff / std::sqrt(se_sq);
                        p_value = detail::t_to_pvalue(t_stat);
                    }
                } else {
                    Real df_ref = static_cast<Real>(N_ref - 1);
                    Real df_target = static_cast<Real>(N_target - 1);
                    Real pooled_var = (df_ref * var_ref + df_target * var_target) / (df_ref + df_target);
                    Real se = std::sqrt(pooled_var * (inv_N_ref + inv_N_target));
                    if (se > Real(0)) {
                        t_stat = mean_diff / se;
                        p_value = detail::t_to_pvalue(t_stat);
                    }
                }
            }

            Size out_idx = i * n_targets + t;
            out_t_stats[out_idx] = t_stat;
            out_p_values[out_idx] = p_value;
            out_log2_fc[out_idx] = log2_fc;
        }
    });
}

/// @brief T-test for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void ttest_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_t_stats,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc,
    bool use_welch
) {
    const Index n_features = scl::primary_size(matrix);
    const Size n_targets = n_groups - 1;
    const Size output_size = static_cast<Size>(n_features) * n_targets;

    SCL_CHECK_DIM(out_t_stats.len >= output_size, "T-stats size mismatch");
    SCL_CHECK_DIM(out_p_values.len >= output_size, "P-values size mismatch");
    SCL_CHECK_DIM(out_log2_fc.len >= output_size, "Log2FC size mismatch");

    std::vector<Real> means(static_cast<Size>(n_features) * n_groups);
    std::vector<Real> vars(static_cast<Size>(n_features) * n_groups);
    std::vector<Size> counts(static_cast<Size>(n_features) * n_groups);

    compute_group_stats_mapped(
        matrix, group_ids, n_groups,
        Array<Real>(means.data(), means.size()),
        Array<Real>(vars.data(), vars.size()),
        Array<Size>(counts.data(), counts.size())
    );

    const Size N_ref = group_sizes[0];
    const Real inv_N_ref = (N_ref > 0) ? (Real(1) / static_cast<Real>(N_ref)) : Real(0);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_features), [&](size_t i) {
        Real mean_ref = means[i * n_groups + 0];
        Real var_ref = vars[i * n_groups + 0];

        for (Size t = 0; t < n_targets; ++t) {
            Size target_group = t + 1;
            Size N_target = group_sizes[target_group];

            Real mean_target = means[i * n_groups + target_group];
            Real var_target = vars[i * n_groups + target_group];

            Real mean_diff = mean_target - mean_ref;

            constexpr Real PSEUDOCOUNT = Real(1e-9);
            Real log2_fc = std::log2((mean_target + PSEUDOCOUNT) / (mean_ref + PSEUDOCOUNT));

            Real t_stat = Real(0);
            Real p_value = Real(1);

            if (N_ref > 0 && N_target > 0) {
                Real inv_N_target = Real(1) / static_cast<Real>(N_target);

                if (use_welch) {
                    Real se_sq = var_ref * inv_N_ref + var_target * inv_N_target;
                    if (se_sq > Real(0)) {
                        t_stat = mean_diff / std::sqrt(se_sq);
                        p_value = detail::t_to_pvalue(t_stat);
                    }
                } else {
                    Real df_ref = static_cast<Real>(N_ref - 1);
                    Real df_target = static_cast<Real>(N_target - 1);
                    Real pooled_var = (df_ref * var_ref + df_target * var_target) / (df_ref + df_target);
                    Real se = std::sqrt(pooled_var * (inv_N_ref + inv_N_target));
                    if (se > Real(0)) {
                        t_stat = mean_diff / se;
                        p_value = detail::t_to_pvalue(t_stat);
                    }
                }
            }

            Size out_idx = i * n_targets + t;
            out_t_stats[out_idx] = t_stat;
            out_p_values[out_idx] = p_value;
            out_log2_fc[out_idx] = log2_fc;
        }
    });
}

// =============================================================================
// SECTION 5: Unified Dispatcher
// =============================================================================

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void ttest_mapped_dispatch(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_t_stats,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc,
    bool use_welch
) {
    ttest_mapped(matrix, group_ids, n_groups, group_sizes,
                 out_t_stats, out_p_values, out_log2_fc, use_welch);
}

} // namespace scl::kernel::ttest::mapped
