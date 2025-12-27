#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
// FILE: scl/kernel/sparse.hpp
// BRIEF: Sparse matrix statistics with SIMD optimization
// =============================================================================

namespace scl::kernel::sparse {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 16;
}

// =============================================================================
// SIMD Helpers
// =============================================================================

namespace detail {

template <typename T>
SCL_FORCE_INLINE void simd_sum_sumsq_fused(
    const T* SCL_RESTRICT vals,
    Size len,
    T& out_sum,
    T& out_sumsq
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_ssq0 = s::Zero(d);
    auto v_ssq1 = s::Zero(d);

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

        v_ssq0 = s::MulAdd(v0, v0, v_ssq0);
        v_ssq1 = s::MulAdd(v1, v1, v_ssq1);
        v_ssq0 = s::MulAdd(v2, v2, v_ssq0);
        v_ssq1 = s::MulAdd(v3, v3, v_ssq1);
    }

    auto v_sum = s::Add(v_sum0, v_sum1);
    auto v_ssq = s::Add(v_ssq0, v_ssq1);

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_sum = s::Add(v_sum, v);
        v_ssq = s::MulAdd(v, v, v_ssq);
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));
    T sumsq = s::GetLane(s::SumOfLanes(d, v_ssq));

    for (; k < len; ++k) {
        T v = vals[k];
        sum += v;
        sumsq += v * v;
    }

    out_sum = sum;
    out_sumsq = sumsq;
}

template <typename T>
SCL_FORCE_INLINE T compute_variance(T sum, T sum_sq, T N, T denom) {
    if (denom <= T(0)) return T(0);

    T mu = sum / N;
    T var = (sum_sq - sum * mu) / denom;

    return (var < T(0)) ? T(0) : var;
}

} // namespace detail

// =============================================================================
// Transform Functions
// =============================================================================

template <typename T, bool IsCSR>
void primary_sums(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index primary_dim = matrix.primary_dim();

    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);

        if (len_sz == 0) {
            output[p] = T(0);
            return;
        }

        auto values = matrix.primary_values(idx);
        output[p] = scl::vectorize::sum(Array<const T>(values.ptr, len_sz));
    });
}

template <typename T, bool IsCSR>
void primary_means(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index primary_dim = matrix.primary_dim();
    const T inv_n = T(1) / static_cast<T>(matrix.secondary_dim());

    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);

        if (len_sz == 0) {
            output[p] = T(0);
            return;
        }

        auto values = matrix.primary_values(idx);
        output[p] = scl::vectorize::sum(Array<const T>(values.ptr, len_sz)) * inv_n;
    });
}

template <typename T, bool IsCSR>
void primary_variances(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output,
    int ddof = 1
) {
    const Index primary_dim = matrix.primary_dim();
    const T N = static_cast<T>(matrix.secondary_dim());
    const T denom = N - static_cast<T>(ddof);

    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);

        T sum = T(0), sumsq = T(0);

        if (len_sz > 0) {
            auto values = matrix.primary_values(idx);
            detail::simd_sum_sumsq_fused(values.ptr, len_sz, sum, sumsq);
        }

        output[p] = detail::compute_variance(sum, sumsq, N, denom);
    });
}

template <typename T, bool IsCSR>
void primary_nnz(
    const Sparse<T, IsCSR>& matrix,
    Array<Index> output
) {
    const Index primary_dim = matrix.primary_dim();

    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        output[p] = matrix.primary_length(idx);
    });
}

} // namespace scl::kernel::sparse

