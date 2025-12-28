#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include "scl/core/algo.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/feature.hpp
// BRIEF: Feature statistics with SIMD optimization
// =============================================================================

namespace scl::kernel::feature {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size CHUNK_SIZE = 256;
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Real EPSILON = 1e-12;
}

// =============================================================================
// SIMD Utilities
// =============================================================================

namespace detail {

template <typename T>
SCL_FORCE_INLINE void compute_sum_sq_simd(
    const T* SCL_RESTRICT vals,
    Size len,
    Real& out_sum,
    Real& out_sq_sum
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sq0 = s::Zero(d);
    auto v_sq1 = s::Zero(d);

    Size k = 0;

    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE * lanes < len)) {
            SCL_PREFETCH_READ(vals + k + config::PREFETCH_DISTANCE * lanes, 0);
        }

        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        v_sum0 = s::Add(v_sum0, v0);
        v_sum1 = s::Add(v_sum1, v1);
        v_sum0 = s::Add(v_sum0, v2);
        v_sum1 = s::Add(v_sum1, v3);

        v_sq0 = s::MulAdd(v0, v0, v_sq0);
        v_sq1 = s::MulAdd(v1, v1, v_sq1);
        v_sq0 = s::MulAdd(v2, v2, v_sq0);
        v_sq1 = s::MulAdd(v3, v3, v_sq1);
    }

    auto v_sum = s::Add(v_sum0, v_sum1);
    auto v_sq = s::Add(v_sq0, v_sq1);

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_sum = s::Add(v_sum, v);
        v_sq = s::MulAdd(v, v, v_sq);
    }

    Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
    Real sq_sum = s::GetLane(s::SumOfLanes(d, v_sq));

    for (; k < len; ++k) {
        Real v = static_cast<Real>(vals[k]);
        sum += v;
        sq_sum += v * v;
    }

    out_sum = sum;
    out_sq_sum = sq_sum;
}

template <typename T>
SCL_FORCE_INLINE void compute_clipped_sum_sq_simd(
    const T* SCL_RESTRICT vals,
    Size len,
    Real clip,
    Real& out_sum,
    Real& out_sq_sum
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_clip = s::Set(d, clip);
    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sq0 = s::Zero(d);
    auto v_sq1 = s::Zero(d);

    Size k = 0;

    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE * lanes < len)) {
            SCL_PREFETCH_READ(vals + k + config::PREFETCH_DISTANCE * lanes, 0);
        }

        auto v0 = s::Min(s::Load(d, vals + k + 0 * lanes), v_clip);
        auto v1 = s::Min(s::Load(d, vals + k + 1 * lanes), v_clip);
        auto v2 = s::Min(s::Load(d, vals + k + 2 * lanes), v_clip);
        auto v3 = s::Min(s::Load(d, vals + k + 3 * lanes), v_clip);

        v_sum0 = s::Add(v_sum0, v0);
        v_sum1 = s::Add(v_sum1, v1);
        v_sum0 = s::Add(v_sum0, v2);
        v_sum1 = s::Add(v_sum1, v3);

        v_sq0 = s::MulAdd(v0, v0, v_sq0);
        v_sq1 = s::MulAdd(v1, v1, v_sq1);
        v_sq0 = s::MulAdd(v2, v2, v_sq0);
        v_sq1 = s::MulAdd(v3, v3, v_sq1);
    }

    auto v_sum = s::Add(v_sum0, v_sum1);
    auto v_sq = s::Add(v_sq0, v_sq1);

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Min(s::Load(d, vals + k), v_clip);
        v_sum = s::Add(v_sum, v);
        v_sq = s::MulAdd(v, v, v_sq);
    }

    Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
    Real sq_sum = s::GetLane(s::SumOfLanes(d, v_sq));

    for (; k < len; ++k) {
        Real v = scl::algo::min2(static_cast<Real>(vals[k]), clip);
        sum += v;
        sq_sum += v * v;
    }

    out_sum = sum;
    out_sq_sum = sq_sum;
}

} // namespace detail

// =============================================================================
// Transform Functions
// =============================================================================

template <typename T, bool IsCSR>
void standard_moments(
    const Sparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    const Index primary_dim = matrix.primary_dim();
    const Real N = static_cast<Real>(matrix.secondary_dim());
    const Real denom = N - static_cast<Real>(ddof);

    SCL_CHECK_DIM(out_means.len >= static_cast<Size>(primary_dim), "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= static_cast<Size>(primary_dim), "Vars size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);

        Real sum, sq_sum;
        if (len_sz > 0) {
            auto values = matrix.primary_values(idx);
            detail::compute_sum_sq_simd(values.ptr, len_sz, sum, sq_sum);
        } else {
            sum = Real(0);
            sq_sum = Real(0);
        }

        Real mu = sum / N;
        Real var = (denom > Real(0)) ? ((sq_sum - sum * mu) / denom) : Real(0);
        if (var < Real(0)) var = Real(0);

        out_means[p] = mu;
        out_vars[p] = var;
    });
}

template <typename T, bool IsCSR>
void clipped_moments(
    const Sparse<T, IsCSR>& matrix,
    Array<const Real> clip_vals,
    Array<Real> out_means,
    Array<Real> out_vars
) {
    const Index primary_dim = matrix.primary_dim();
    const Real N = static_cast<Real>(matrix.secondary_dim());
    const Real N_minus_1 = N - Real(1);

    SCL_CHECK_DIM(clip_vals.len >= static_cast<Size>(primary_dim), "Clip vals mismatch");
    SCL_CHECK_DIM(out_means.len >= static_cast<Size>(primary_dim), "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= static_cast<Size>(primary_dim), "Vars size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);
        Real clip = clip_vals[p];

        Real sum, sq_sum;
        if (len_sz > 0) {
            auto values = matrix.primary_values(idx);
            detail::compute_clipped_sum_sq_simd(values.ptr, len_sz, clip, sum, sq_sum);
        } else {
            sum = Real(0);
            sq_sum = Real(0);
        }

        Real mu = sum / N;
        Real var = Real(0);
        if (N > Real(1)) {
            var = (sq_sum - N * mu * mu) / N_minus_1;
        }
        if (var < Real(0)) var = Real(0);

        out_means[p] = mu;
        out_vars[p] = var;
    });
}

template <typename T, bool IsCSR>
void detection_rate(
    const Sparse<T, IsCSR>& matrix,
    Array<Real> out_rates
) {
    const Index primary_dim = matrix.primary_dim();
    const Real inv_N = Real(1) / static_cast<Real>(matrix.secondary_dim());

    SCL_CHECK_DIM(out_rates.len >= static_cast<Size>(primary_dim), "Rates size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        out_rates[p] = static_cast<Real>(len) * inv_N;
    });
}

inline void dispersion(
    Array<const Real> means,
    Array<const Real> vars,
    Array<Real> out_dispersion
) {
    const Size n = means.len;

    SCL_CHECK_DIM(vars.len >= n, "Vars size mismatch");
    SCL_CHECK_DIM(out_dispersion.len >= n, "Output size mismatch");

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

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
        Real m = means[k];
        Real v = vars[k];
        out_dispersion[k] = (m > config::EPSILON) ? (v / m) : Real(0);
    }
}

} // namespace scl::kernel::feature

