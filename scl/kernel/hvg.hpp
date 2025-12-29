#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/sort.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <limits>

// =============================================================================
// FILE: scl/kernel/hvg.hpp
// BRIEF: Highly variable gene selection with SIMD optimization
// =============================================================================

namespace scl::kernel::hvg {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = 1e-12;
    constexpr Size PREFETCH_DISTANCE = 16;
}

// =============================================================================
// SIMD Utilities
// =============================================================================

namespace detail {

inline void dispersion_simd(
    Array<const Real> means,
    Array<const Real> vars,
    Array<Real> out_dispersion
) {
    const Size n = means.len;

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    const auto v_eps = s::Set(d, config::EPSILON);
    const auto v_zero = s::Zero(d);

    Size k = 0;

    for (; k + 4 * lanes <= n; k += 4 * lanes) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE * lanes < n)) {
            SCL_PREFETCH_READ(means.ptr + k + config::PREFETCH_DISTANCE * lanes, 0);
            SCL_PREFETCH_READ(vars.ptr + k + config::PREFETCH_DISTANCE * lanes, 0);
        }

        auto v_mean0 = s::Load(d, means.ptr + k + 0 * lanes);
        auto v_mean1 = s::Load(d, means.ptr + k + 1 * lanes);
        auto v_mean2 = s::Load(d, means.ptr + k + 2 * lanes);
        auto v_mean3 = s::Load(d, means.ptr + k + 3 * lanes);

        auto v_var0 = s::Load(d, vars.ptr + k + 0 * lanes);
        auto v_var1 = s::Load(d, vars.ptr + k + 1 * lanes);
        auto v_var2 = s::Load(d, vars.ptr + k + 2 * lanes);
        auto v_var3 = s::Load(d, vars.ptr + k + 3 * lanes);

        auto mask0 = s::Gt(v_mean0, v_eps);
        auto mask1 = s::Gt(v_mean1, v_eps);
        auto mask2 = s::Gt(v_mean2, v_eps);
        auto mask3 = s::Gt(v_mean3, v_eps);

        s::Store(s::IfThenElse(mask0, s::Div(v_var0, v_mean0), v_zero), d, out_dispersion.ptr + k + 0 * lanes);
        s::Store(s::IfThenElse(mask1, s::Div(v_var1, v_mean1), v_zero), d, out_dispersion.ptr + k + 1 * lanes);
        s::Store(s::IfThenElse(mask2, s::Div(v_var2, v_mean2), v_zero), d, out_dispersion.ptr + k + 2 * lanes);
        s::Store(s::IfThenElse(mask3, s::Div(v_var3, v_mean3), v_zero), d, out_dispersion.ptr + k + 3 * lanes);
    }

    for (; k + lanes <= n; k += lanes) {
        auto v_mean = s::Load(d, means.ptr + k);
        auto v_var = s::Load(d, vars.ptr + k);

        auto mask = s::Gt(v_mean, v_eps);
        auto v_div = s::Div(v_var, v_mean);
        auto v_res = s::IfThenElse(mask, v_div, v_zero);

        s::Store(v_res, d, out_dispersion.ptr + k);
    }

    for (; k < n; ++k) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        // PERFORMANCE: Safe narrowing - k is bounded by array size and fits in Index
        Real m = means[static_cast<Index>(k)];
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        Real v = vars[static_cast<Index>(k)];
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        out_dispersion[static_cast<Index>(k)] = (m > config::EPSILON) ? (v / m) : Real(0);
    }
}

inline void normalize_dispersion_simd(
    Array<Real> dispersions,
    Real min_mean,
    Real max_mean,
    Array<const Real> means
) {
    const Size n = dispersions.len;

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    Size valid_count = 0;
    Real disp_sum = Real(0);
    Real disp_sq = Real(0);

    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        // PERFORMANCE: Safe narrowing - i is bounded by array size and fits in Index
        Real m = means[static_cast<Index>(i)];
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        Real disp = dispersions[static_cast<Index>(i)];

        if (m >= min_mean && m <= max_mean && disp > Real(0)) {
            disp_sum += disp;
            disp_sq += disp * disp;
            valid_count++;
        } else {
            // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
            dispersions[static_cast<Index>(i)] = -std::numeric_limits<Real>::infinity();
        }
    }

    if (valid_count == 0) return;

    Real disp_mean = disp_sum / static_cast<Real>(valid_count);
    Real disp_var = (disp_sq / static_cast<Real>(valid_count)) - (disp_mean * disp_mean);
    Real disp_std = (disp_var > Real(0)) ? std::sqrt(disp_var) : Real(1);
    Real inv_std = Real(1) / disp_std;

    const auto v_mean = s::Set(d, disp_mean);
    const auto v_inv_std = s::Set(d, inv_std);
    const auto v_neg_inf = s::Set(d, -std::numeric_limits<Real>::infinity());

    Size k = 0;

    for (; k + lanes <= n; k += lanes) {
        auto v_disp = s::Load(d, dispersions.ptr + k);

        auto mask = s::Gt(v_disp, v_neg_inf);
        auto v_norm = s::Mul(s::Sub(v_disp, v_mean), v_inv_std);
        auto v_res = s::IfThenElse(mask, v_norm, v_neg_inf);

        s::Store(v_res, d, dispersions.ptr + k);
    }

    for (; k < n; ++k) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        // PERFORMANCE: Safe narrowing - k is bounded by array size and fits in Index
        Real disp = dispersions[static_cast<Index>(k)];
        if (disp > -std::numeric_limits<Real>::infinity()) {
            // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
            dispersions[static_cast<Index>(k)] = (disp - disp_mean) * inv_std;
        }
    }
}

inline void select_top_k_partial(
    Array<const Real> scores,
    Size k,
    Array<Index> out_indices,
    Array<uint8_t> out_mask
) {
    const Size n = scores.len;

    auto indices_ptr = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Index* indices = indices_ptr.release();
    scl::algo::iota(indices, n, Index(0));

    // Partial sort: first k elements will be the k largest (in descending order)
    scl::algo::partial_sort(indices, n, k, [&scores](Index a, Index b) {
        return scores[a] > scores[b];
    });

    scl::algo::zero(out_mask.ptr, n);

    for (Size i = 0; i < k; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        // PERFORMANCE: Safe narrowing - i is bounded by k and indices array, fits in Index
        Index idx = indices[static_cast<Index>(i)];
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        out_indices[static_cast<Index>(i)] = idx;
        out_mask[idx] = 1;
    }

    scl::memory::aligned_free(indices, SCL_ALIGNMENT);
}

template <typename T, bool IsCSR>
void compute_moments(
    const Sparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    const Index primary_dim = matrix.primary_dim();
    const Index secondary_dim = matrix.secondary_dim();
    const Real N = static_cast<Real>(secondary_dim);
    const Real denom = N - static_cast<Real>(ddof);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        const auto idx = static_cast<Index>(p);
        const auto values = matrix.primary_values_unsafe(idx);
        const auto len_sz = static_cast<Size>(values.len);

        Real sum = Real(0);
        Real sq_sum = Real(0);

        if (len_sz > 0) {
            Array<const T> vals_arr(values.ptr, len_sz);
            sum = scl::vectorize::sum(vals_arr);
            sq_sum = scl::vectorize::sum_squared(vals_arr);
        }

        Real mu = sum / N;
        Real var = (denom > Real(0)) ? ((sq_sum - sum * mu) / denom) : Real(0);
        if (var < Real(0)) var = Real(0);

        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        // PERFORMANCE: Safe narrowing - p is bounded by primary_dim and fits in Index
        out_means[static_cast<Index>(p)] = mu;
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        out_vars[static_cast<Index>(p)] = var;
    });
}

template <typename T, bool IsCSR>
void compute_clipped_moments(
    const Sparse<T, IsCSR>& matrix,
    Array<const Real> clip_vals,
    Array<Real> out_means,
    Array<Real> out_vars
) {
    const Index primary_dim = matrix.primary_dim();
    const Index secondary_dim = matrix.secondary_dim();
    const Real N = static_cast<Real>(secondary_dim);
    const Real N_minus_1 = N - Real(1);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        const auto idx = static_cast<Index>(p);
        const auto values = matrix.primary_values_unsafe(idx);
        const Index len = matrix.primary_length_unsafe(idx);
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        // PERFORMANCE: Safe narrowing - p is bounded by primary_dim and fits in Index
        const Real clip = clip_vals[static_cast<Index>(p)];

        Real sum = Real(0);
        Real sq_sum = Real(0);

        for (Index k = 0; k < len; ++k) {
            Real v = scl::algo::min2(static_cast<Real>(values[k]), clip);
            sum += v;
            sq_sum += v * v;
        }

        Real mu = sum / N;
        Real var = Real(0);
        if (N > Real(1)) {
            var = (sq_sum - N * mu * mu) / N_minus_1;
        }
        if (var < Real(0)) var = Real(0);

        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        // PERFORMANCE: Safe narrowing - p is bounded by primary_dim and fits in Index
        out_means[static_cast<Index>(p)] = mu;
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        out_vars[static_cast<Index>(p)] = var;
    });
}

} // namespace detail

// =============================================================================
// HVG Selection Functions
// =============================================================================

template <typename T, bool IsCSR>
void select_by_dispersion(
    const Sparse<T, IsCSR>& matrix,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask,
    Array<Real> out_dispersions
) {
    const Index primary_dim = matrix.primary_dim();
    const Size n = static_cast<Size>(primary_dim);

    auto means_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    auto vars_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* means = means_ptr.release();
    Real* vars = vars_ptr.release();

    detail::compute_moments(
        matrix,
        Array<Real>(means, n),
        Array<Real>(vars, n),
        1
    );

    detail::dispersion_simd(
        Array<const Real>(means, n),
        Array<const Real>(vars, n),
        out_dispersions
    );

    detail::select_top_k_partial(
        Array<const Real>(out_dispersions.ptr, n),
        n_top,
        out_indices,
        out_mask
    );

    scl::memory::aligned_free(means, SCL_ALIGNMENT);
    scl::memory::aligned_free(vars, SCL_ALIGNMENT);
}

template <typename T, bool IsCSR>
void select_by_vst(
    const Sparse<T, IsCSR>& matrix,
    Array<const Real> clip_vals,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask,
    Array<Real> out_variances
) {
    const Index primary_dim = matrix.primary_dim();
    const Size n = static_cast<Size>(primary_dim);

    auto means_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* means = means_ptr.release();

    detail::compute_clipped_moments(
        matrix,
        clip_vals,
        Array<Real>(means, n),
        out_variances
    );

    detail::select_top_k_partial(
        Array<const Real>(out_variances.ptr, n),
        n_top,
        out_indices,
        out_mask
    );

    scl::memory::aligned_free(means, SCL_ALIGNMENT);
}

} // namespace scl::kernel::hvg
