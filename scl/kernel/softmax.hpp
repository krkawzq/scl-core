#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <limits>

// =============================================================================
// FILE: scl/kernel/softmax.hpp
// BRIEF: Softmax operations with SIMD optimization
// =============================================================================

namespace scl::kernel::softmax {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size SHORT_THRESHOLD = 16;
    constexpr Size MEDIUM_THRESHOLD = 128;
    constexpr Size PREFETCH_DISTANCE = 16;
}

// =============================================================================
// SIMD Helpers
// =============================================================================

namespace detail {

template <typename T>
SCL_FORCE_INLINE T simd_max(const T* SCL_RESTRICT vals, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_max0 = s::Set(d, -std::numeric_limits<T>::infinity());
    auto v_max1 = v_max0;
    auto v_max2 = v_max0;
    auto v_max3 = v_max0;

    Size k = 0;

    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE * lanes < len)) {
            SCL_PREFETCH_READ(vals + k + config::PREFETCH_DISTANCE * lanes, 0);
        }

        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        v_max0 = s::Max(v_max0, v0);
        v_max1 = s::Max(v_max1, v1);
        v_max2 = s::Max(v_max2, v2);
        v_max3 = s::Max(v_max3, v3);
    }

    auto v_max = s::Max(s::Max(v_max0, v_max1), s::Max(v_max2, v_max3));

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_max = s::Max(v_max, v);
    }

    T max_val = s::GetLane(s::MaxOfLanes(d, v_max));

    for (; k < len; ++k) {
        if (vals[k] > max_val) max_val = vals[k];
    }

    return max_val;
}

template <typename T>
SCL_FORCE_INLINE T scalar_max(const T* SCL_RESTRICT vals, Size len) {
    T max_val = vals[0];
    for (Size k = 1; k < len; ++k) {
        if (vals[k] > max_val) max_val = vals[k];
    }
    return max_val;
}

template <typename T>
SCL_FORCE_INLINE T exp_sum_short(T* SCL_RESTRICT vals, Size len, T max_val) {
    T sum = T(0);
    for (Size k = 0; k < len; ++k) {
        T v = std::exp(vals[k] - max_val);
        vals[k] = v;
        sum += v;
    }
    return sum;
}

template <typename T>
SCL_FORCE_INLINE T exp_sum_medium(T* SCL_RESTRICT vals, Size len, T max_val) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_max = s::Set(d, max_val);
    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    Size k = 0;

    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE * lanes < len)) {
            SCL_PREFETCH_READ(vals + k + config::PREFETCH_DISTANCE * lanes, 0);
        }

        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        v0 = s::Exp(d, s::Sub(v0, v_max));
        v1 = s::Exp(d, s::Sub(v1, v_max));
        v2 = s::Exp(d, s::Sub(v2, v_max));
        v3 = s::Exp(d, s::Sub(v3, v_max));

        s::Store(v0, d, vals + k + 0 * lanes);
        s::Store(v1, d, vals + k + 1 * lanes);
        s::Store(v2, d, vals + k + 2 * lanes);
        s::Store(v3, d, vals + k + 3 * lanes);

        v_sum0 = s::Add(v_sum0, v0);
        v_sum1 = s::Add(v_sum1, v1);
        v_sum2 = s::Add(v_sum2, v2);
        v_sum3 = s::Add(v_sum3, v3);
    }

    auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v = s::Exp(d, s::Sub(v, v_max));
        s::Store(v, d, vals + k);
        v_sum = s::Add(v_sum, v);
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < len; ++k) {
        T v = std::exp(vals[k] - max_val);
        vals[k] = v;
        sum += v;
    }

    return sum;
}

template <typename T>
SCL_FORCE_INLINE T exp_sum_long(T* SCL_RESTRICT vals, Size len, T max_val) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_max = s::Set(d, max_val);
    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);
    auto v_sum4 = s::Zero(d);
    auto v_sum5 = s::Zero(d);
    auto v_sum6 = s::Zero(d);
    auto v_sum7 = s::Zero(d);

    Size k = 0;

    for (; k + 8 * lanes <= len; k += 8 * lanes) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE * lanes < len)) {
            SCL_PREFETCH_READ(vals + k + config::PREFETCH_DISTANCE * lanes, 0);
        }

        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);
        auto v4 = s::Load(d, vals + k + 4 * lanes);
        auto v5 = s::Load(d, vals + k + 5 * lanes);
        auto v6 = s::Load(d, vals + k + 6 * lanes);
        auto v7 = s::Load(d, vals + k + 7 * lanes);

        v0 = s::Exp(d, s::Sub(v0, v_max));
        v1 = s::Exp(d, s::Sub(v1, v_max));
        v2 = s::Exp(d, s::Sub(v2, v_max));
        v3 = s::Exp(d, s::Sub(v3, v_max));
        v4 = s::Exp(d, s::Sub(v4, v_max));
        v5 = s::Exp(d, s::Sub(v5, v_max));
        v6 = s::Exp(d, s::Sub(v6, v_max));
        v7 = s::Exp(d, s::Sub(v7, v_max));

        s::Store(v0, d, vals + k + 0 * lanes);
        s::Store(v1, d, vals + k + 1 * lanes);
        s::Store(v2, d, vals + k + 2 * lanes);
        s::Store(v3, d, vals + k + 3 * lanes);
        s::Store(v4, d, vals + k + 4 * lanes);
        s::Store(v5, d, vals + k + 5 * lanes);
        s::Store(v6, d, vals + k + 6 * lanes);
        s::Store(v7, d, vals + k + 7 * lanes);

        v_sum0 = s::Add(v_sum0, v0);
        v_sum1 = s::Add(v_sum1, v1);
        v_sum2 = s::Add(v_sum2, v2);
        v_sum3 = s::Add(v_sum3, v3);
        v_sum4 = s::Add(v_sum4, v4);
        v_sum5 = s::Add(v_sum5, v5);
        v_sum6 = s::Add(v_sum6, v6);
        v_sum7 = s::Add(v_sum7, v7);
    }

    auto v_sum = s::Add(
        s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3)),
        s::Add(s::Add(v_sum4, v_sum5), s::Add(v_sum6, v_sum7))
    );

    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        v0 = s::Exp(d, s::Sub(v0, v_max));
        v1 = s::Exp(d, s::Sub(v1, v_max));
        v2 = s::Exp(d, s::Sub(v2, v_max));
        v3 = s::Exp(d, s::Sub(v3, v_max));

        s::Store(v0, d, vals + k + 0 * lanes);
        s::Store(v1, d, vals + k + 1 * lanes);
        s::Store(v2, d, vals + k + 2 * lanes);
        s::Store(v3, d, vals + k + 3 * lanes);

        v_sum = s::Add(v_sum, s::Add(s::Add(v0, v1), s::Add(v2, v3)));
    }

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v = s::Exp(d, s::Sub(v, v_max));
        s::Store(v, d, vals + k);
        v_sum = s::Add(v_sum, v);
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < len; ++k) {
        T v = std::exp(vals[k] - max_val);
        vals[k] = v;
        sum += v;
    }

    return sum;
}

template <typename T>
SCL_FORCE_INLINE void normalize_simd(T* SCL_RESTRICT vals, Size len, T inv_sum) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_inv_sum = s::Set(d, inv_sum);

    Size k = 0;

    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        s::Store(s::Mul(v0, v_inv_sum), d, vals + k + 0 * lanes);
        s::Store(s::Mul(v1, v_inv_sum), d, vals + k + 1 * lanes);
        s::Store(s::Mul(v2, v_inv_sum), d, vals + k + 2 * lanes);
        s::Store(s::Mul(v3, v_inv_sum), d, vals + k + 3 * lanes);
    }

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        s::Store(s::Mul(v, v_inv_sum), d, vals + k);
    }

    for (; k < len; ++k) {
        vals[k] *= inv_sum;
    }
}

template <typename T>
SCL_FORCE_INLINE void normalize_scalar(T* SCL_RESTRICT vals, Size len, T inv_sum) {
    for (Size k = 0; k < len; ++k) {
        vals[k] *= inv_sum;
    }
}

template <typename T>
SCL_FORCE_INLINE void softmax_adaptive(T* SCL_RESTRICT vals, Size len) {
    if (len == 0) return;

    T max_val;
    if (len < config::SHORT_THRESHOLD) {
        max_val = scalar_max(vals, len);
    } else {
        max_val = simd_max(vals, len);
    }

    T sum;
    if (len < config::SHORT_THRESHOLD) {
        sum = exp_sum_short(vals, len, max_val);
    } else if (len < config::MEDIUM_THRESHOLD) {
        sum = exp_sum_medium(vals, len, max_val);
    } else {
        sum = exp_sum_long(vals, len, max_val);
    }

    if (sum > T(0)) {
        T inv_sum = T(1) / sum;
        if (len < config::SHORT_THRESHOLD) {
            normalize_scalar(vals, len, inv_sum);
        } else {
            normalize_simd(vals, len, inv_sum);
        }
    }
}

template <typename T>
SCL_FORCE_INLINE void log_softmax_adaptive(T* SCL_RESTRICT vals, Size len) {
    if (len == 0) return;

    T max_val = (len < config::SHORT_THRESHOLD) 
        ? scalar_max(vals, len) 
        : simd_max(vals, len);

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_max = s::Set(d, max_val);
    auto v_sum = s::Zero(d);

    Size k = 0;
    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_sum = s::Add(v_sum, s::Exp(d, s::Sub(v, v_max)));
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));
    for (; k < len; ++k) {
        sum += std::exp(vals[k] - max_val);
    }

    T log_sum = std::log(sum);
    T offset = max_val + log_sum;

    const auto v_offset = s::Set(d, offset);

    k = 0;
    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        s::Store(s::Sub(v0, v_offset), d, vals + k + 0 * lanes);
        s::Store(s::Sub(v1, v_offset), d, vals + k + 1 * lanes);
        s::Store(s::Sub(v2, v_offset), d, vals + k + 2 * lanes);
        s::Store(s::Sub(v3, v_offset), d, vals + k + 3 * lanes);
    }

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        s::Store(s::Sub(v, v_offset), d, vals + k);
    }

    for (; k < len; ++k) {
        vals[k] -= offset;
    }
}

} // namespace detail

// =============================================================================
// Transform Functions
// =============================================================================

template <typename T, bool IsCSR>
void softmax_inplace(Sparse<T, IsCSR>& matrix) {
    const Index primary_dim = matrix.primary_dim();

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        if (len == 0) return;

        auto values = matrix.primary_values(idx);
        detail::softmax_adaptive(values.ptr, static_cast<Size>(len));
    });
}

template <typename T, bool IsCSR>
void log_softmax_inplace(Sparse<T, IsCSR>& matrix) {
    const Index primary_dim = matrix.primary_dim();

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        if (len == 0) return;

        auto values = matrix.primary_values(idx);
        detail::log_softmax_adaptive(values.ptr, static_cast<Size>(len));
    });
}

} // namespace scl::kernel::softmax

