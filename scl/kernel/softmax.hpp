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
//
// OPTIMIZATIONS:
//   - 3-tier adaptive strategy: short (<16), medium (<128), long (>=128)
//   - 8-way SIMD unrolling with multiple accumulators for ILP
//   - Prefetch for cache optimization
//   - Fused exp+sum computation to minimize memory traffic
//   - Numerical stability via max subtraction
//   - Optional temperature scaling for softmax with temperature
//   - Log-softmax support with same 3-tier strategy
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

// -----------------------------------------------------------------------------
// Max computation with adaptive strategy
// -----------------------------------------------------------------------------

template <typename T>
SCL_FORCE_INLINE SCL_HOT T simd_max(const T* SCL_RESTRICT vals, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_max0 = s::Set(d, -std::numeric_limits<T>::infinity());
    auto v_max1 = v_max0;
    auto v_max2 = v_max0;
    auto v_max3 = v_max0;

    Size k = 0;

    // 4-way unrolled SIMD max with prefetch
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

    // Combine accumulators
    auto v_max = s::Max(s::Max(v_max0, v_max1), s::Max(v_max2, v_max3));

    // Handle remaining full SIMD lanes
    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_max = s::Max(v_max, v);
    }

    T max_val = s::GetLane(s::MaxOfLanes(d, v_max));

    // Scalar cleanup
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
SCL_FORCE_INLINE T adaptive_max(const T* SCL_RESTRICT vals, Size len) {
    return (len < config::SHORT_THRESHOLD) ? scalar_max(vals, len) : simd_max(vals, len);
}

// -----------------------------------------------------------------------------
// Exp + Sum computation: 3-tier strategy
// -----------------------------------------------------------------------------

template <typename T>
SCL_FORCE_INLINE SCL_HOT T exp_sum_short(T* SCL_RESTRICT vals, Size len, T max_val) {
    T sum = T(0);
    for (Size k = 0; k < len; ++k) {
        T v = std::exp(vals[k] - max_val);
        vals[k] = v;
        sum += v;
    }
    return sum;
}

template <typename T>
SCL_FORCE_INLINE SCL_HOT T exp_sum_medium(T* SCL_RESTRICT vals, Size len, T max_val) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_max = s::Set(d, max_val);
    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    Size k = 0;

    // 4-way unrolled loop with prefetch
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

    // Handle remaining full SIMD lanes
    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v = s::Exp(d, s::Sub(v, v_max));
        s::Store(v, d, vals + k);
        v_sum = s::Add(v_sum, v);
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));

    // Scalar cleanup
    for (; k < len; ++k) {
        T v = std::exp(vals[k] - max_val);
        vals[k] = v;
        sum += v;
    }

    return sum;
}

template <typename T>
SCL_FORCE_INLINE SCL_HOT T exp_sum_long(T* SCL_RESTRICT vals, Size len, T max_val) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_max = s::Set(d, max_val);
    // 8 accumulators for maximum ILP
    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);
    auto v_sum4 = s::Zero(d);
    auto v_sum5 = s::Zero(d);
    auto v_sum6 = s::Zero(d);
    auto v_sum7 = s::Zero(d);

    Size k = 0;

    // 8-way unrolled loop for long arrays
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

    // Combine 8 accumulators into one
    auto v_sum = s::Add(
        s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3)),
        s::Add(s::Add(v_sum4, v_sum5), s::Add(v_sum6, v_sum7))
    );

    // 4-way cleanup for remaining elements
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

    // Single lane cleanup
    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v = s::Exp(d, s::Sub(v, v_max));
        s::Store(v, d, vals + k);
        v_sum = s::Add(v_sum, v);
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));

    // Scalar cleanup
    for (; k < len; ++k) {
        T v = std::exp(vals[k] - max_val);
        vals[k] = v;
        sum += v;
    }

    return sum;
}

// Adaptive exp+sum dispatcher
template <typename T>
SCL_FORCE_INLINE T exp_sum_adaptive(T* SCL_RESTRICT vals, Size len, T max_val) {
    if (len < config::SHORT_THRESHOLD) {
        return exp_sum_short(vals, len, max_val);
    } else if (len < config::MEDIUM_THRESHOLD) {
        return exp_sum_medium(vals, len, max_val);
    } else {
        return exp_sum_long(vals, len, max_val);
    }
}

// -----------------------------------------------------------------------------
// Log-space Exp + Sum computation: 3-tier strategy
// -----------------------------------------------------------------------------

template <typename T>
SCL_FORCE_INLINE SCL_HOT T log_exp_sum_short(const T* SCL_RESTRICT vals, Size len, T max_val) {
    T sum = T(0);
    for (Size k = 0; k < len; ++k) {
        sum += std::exp(vals[k] - max_val);
    }
    return sum;
}

template <typename T>
SCL_FORCE_INLINE SCL_HOT T log_exp_sum_medium(const T* SCL_RESTRICT vals, Size len, T max_val) {
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

        v_sum0 = s::Add(v_sum0, s::Exp(d, s::Sub(v0, v_max)));
        v_sum1 = s::Add(v_sum1, s::Exp(d, s::Sub(v1, v_max)));
        v_sum2 = s::Add(v_sum2, s::Exp(d, s::Sub(v2, v_max)));
        v_sum3 = s::Add(v_sum3, s::Exp(d, s::Sub(v3, v_max)));
    }

    auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_sum = s::Add(v_sum, s::Exp(d, s::Sub(v, v_max)));
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < len; ++k) {
        sum += std::exp(vals[k] - max_val);
    }

    return sum;
}

template <typename T>
SCL_FORCE_INLINE SCL_HOT T log_exp_sum_long(const T* SCL_RESTRICT vals, Size len, T max_val) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_max = s::Set(d, max_val);
    // 8 accumulators for maximum ILP
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

        v_sum0 = s::Add(v_sum0, s::Exp(d, s::Sub(v0, v_max)));
        v_sum1 = s::Add(v_sum1, s::Exp(d, s::Sub(v1, v_max)));
        v_sum2 = s::Add(v_sum2, s::Exp(d, s::Sub(v2, v_max)));
        v_sum3 = s::Add(v_sum3, s::Exp(d, s::Sub(v3, v_max)));
        v_sum4 = s::Add(v_sum4, s::Exp(d, s::Sub(v4, v_max)));
        v_sum5 = s::Add(v_sum5, s::Exp(d, s::Sub(v5, v_max)));
        v_sum6 = s::Add(v_sum6, s::Exp(d, s::Sub(v6, v_max)));
        v_sum7 = s::Add(v_sum7, s::Exp(d, s::Sub(v7, v_max)));
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

        v_sum = s::Add(v_sum, s::Add(
            s::Add(s::Exp(d, s::Sub(v0, v_max)), s::Exp(d, s::Sub(v1, v_max))),
            s::Add(s::Exp(d, s::Sub(v2, v_max)), s::Exp(d, s::Sub(v3, v_max)))
        ));
    }

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_sum = s::Add(v_sum, s::Exp(d, s::Sub(v, v_max)));
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < len; ++k) {
        sum += std::exp(vals[k] - max_val);
    }

    return sum;
}

template <typename T>
SCL_FORCE_INLINE T log_exp_sum_adaptive(const T* SCL_RESTRICT vals, Size len, T max_val) {
    if (len < config::SHORT_THRESHOLD) {
        return log_exp_sum_short(vals, len, max_val);
    } else if (len < config::MEDIUM_THRESHOLD) {
        return log_exp_sum_medium(vals, len, max_val);
    } else {
        return log_exp_sum_long(vals, len, max_val);
    }
}

// -----------------------------------------------------------------------------
// Normalization helpers
// -----------------------------------------------------------------------------

template <typename T>
SCL_FORCE_INLINE void normalize_simd(T* SCL_RESTRICT vals, Size len, T inv_sum) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_inv_sum = s::Set(d, inv_sum);

    Size k = 0;

    // 4-way unrolled normalization
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
SCL_FORCE_INLINE void subtract_offset_simd(T* SCL_RESTRICT vals, Size len, T offset) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_offset = s::Set(d, offset);

    Size k = 0;

    // 4-way unrolled subtraction
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

template <typename T>
SCL_FORCE_INLINE void subtract_offset_scalar(T* SCL_RESTRICT vals, Size len, T offset) {
    for (Size k = 0; k < len; ++k) {
        vals[k] -= offset;
    }
}

// -----------------------------------------------------------------------------
// Main adaptive implementations
// -----------------------------------------------------------------------------

template <typename T>
SCL_FORCE_INLINE void softmax_adaptive(T* SCL_RESTRICT vals, Size len) {
    if (SCL_UNLIKELY(len == 0)) return;

    // Step 1: Find max for numerical stability
    T max_val = adaptive_max(vals, len);

    // Step 2: Compute exp(x - max) and sum
    T sum = exp_sum_adaptive(vals, len, max_val);

    // Step 3: Normalize by sum
    if (SCL_LIKELY(sum > T(0))) {
        T inv_sum = T(1) / sum;
        if (len < config::SHORT_THRESHOLD) {
            normalize_scalar(vals, len, inv_sum);
        } else {
            normalize_simd(vals, len, inv_sum);
        }
    }
}

template <typename T>
SCL_FORCE_INLINE void softmax_with_temperature(T* SCL_RESTRICT vals, Size len, T temperature) {
    if (SCL_UNLIKELY(len == 0)) return;

    if (SCL_UNLIKELY(temperature <= T(0))) {
        // For temperature <= 0, return one-hot at max
        T max_val = adaptive_max(vals, len);
        for (Size k = 0; k < len; ++k) {
            vals[k] = (vals[k] == max_val) ? T(1) : T(0);
        }
        return;
    }

    T inv_temp = T(1) / temperature;

    // Scale by inverse temperature
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    if (len >= config::SHORT_THRESHOLD) {
        const auto v_inv_temp = s::Set(d, inv_temp);
        Size k = 0;

        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Mul(v, v_inv_temp), d, vals + k);
        }

        for (; k < len; ++k) {
            vals[k] *= inv_temp;
        }
    } else {
        for (Size k = 0; k < len; ++k) {
            vals[k] *= inv_temp;
        }
    }

    // Apply standard softmax
    softmax_adaptive(vals, len);
}

template <typename T>
SCL_FORCE_INLINE void log_softmax_adaptive(T* SCL_RESTRICT vals, Size len) {
    if (SCL_UNLIKELY(len == 0)) return;

    // Step 1: Find max for numerical stability
    T max_val = adaptive_max(vals, len);

    // Step 2: Compute sum(exp(x - max)) using 3-tier strategy
    T sum = log_exp_sum_adaptive(vals, len, max_val);

    // Step 3: Compute log_softmax = x - max - log(sum)
    T log_sum = std::log(sum);
    T offset = max_val + log_sum;

    if (len < config::SHORT_THRESHOLD) {
        subtract_offset_scalar(vals, len, offset);
    } else {
        subtract_offset_simd(vals, len, offset);
    }
}

template <typename T>
SCL_FORCE_INLINE void log_softmax_with_temperature(T* SCL_RESTRICT vals, Size len, T temperature) {
    if (SCL_UNLIKELY(len == 0)) return;

    if (SCL_UNLIKELY(temperature <= T(0))) {
        T max_val = adaptive_max(vals, len);
        for (Size k = 0; k < len; ++k) {
            vals[k] = (vals[k] == max_val) ? T(0) : -std::numeric_limits<T>::infinity();
        }
        return;
    }

    T inv_temp = T(1) / temperature;

    // Scale by inverse temperature
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    if (len >= config::SHORT_THRESHOLD) {
        const auto v_inv_temp = s::Set(d, inv_temp);
        Size k = 0;

        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Mul(v, v_inv_temp), d, vals + k);
        }

        for (; k < len; ++k) {
            vals[k] *= inv_temp;
        }
    } else {
        for (Size k = 0; k < len; ++k) {
            vals[k] *= inv_temp;
        }
    }

    log_softmax_adaptive(vals, len);
}

} // namespace detail

// =============================================================================
// Public API: Dense Array Operations
// =============================================================================

/// @brief Apply softmax in-place to a dense array
/// @param vals Pointer to values array
/// @param len Length of array
template <typename T>
void softmax_inplace(T* vals, Size len) {
    detail::softmax_adaptive(vals, len);
}

/// @brief Apply softmax with temperature in-place to a dense array
/// @param vals Pointer to values array
/// @param len Length of array
/// @param temperature Temperature parameter (higher = more uniform)
template <typename T>
void softmax_inplace(T* vals, Size len, T temperature) {
    detail::softmax_with_temperature(vals, len, temperature);
}

/// @brief Apply log-softmax in-place to a dense array
/// @param vals Pointer to values array
/// @param len Length of array
template <typename T>
void log_softmax_inplace(T* vals, Size len) {
    detail::log_softmax_adaptive(vals, len);
}

/// @brief Apply log-softmax with temperature in-place to a dense array
/// @param vals Pointer to values array
/// @param len Length of array
/// @param temperature Temperature parameter
template <typename T>
void log_softmax_inplace(T* vals, Size len, T temperature) {
    detail::log_softmax_with_temperature(vals, len, temperature);
}

// =============================================================================
// Public API: Sparse Matrix Operations
// =============================================================================

/// @brief Apply softmax row-wise in-place to a sparse matrix
/// @param matrix Sparse matrix (CSR or CSC)
template <typename T, bool IsCSR>
void softmax_inplace(Sparse<T, IsCSR>& matrix) {
    const Index primary_dim = matrix.primary_dim();

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);

        if (SCL_UNLIKELY(len == 0)) return;

        auto values = matrix.primary_values(idx);
        detail::softmax_adaptive(values.ptr, static_cast<Size>(len));
    });
}

/// @brief Apply softmax with temperature row-wise in-place to a sparse matrix
/// @param matrix Sparse matrix (CSR or CSC)
/// @param temperature Temperature parameter
template <typename T, bool IsCSR>
void softmax_inplace(Sparse<T, IsCSR>& matrix, T temperature) {
    const Index primary_dim = matrix.primary_dim();

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);

        if (SCL_UNLIKELY(len == 0)) return;

        auto values = matrix.primary_values(idx);
        detail::softmax_with_temperature(values.ptr, static_cast<Size>(len), temperature);
    });
}

/// @brief Apply log-softmax row-wise in-place to a sparse matrix
/// @param matrix Sparse matrix (CSR or CSC)
template <typename T, bool IsCSR>
void log_softmax_inplace(Sparse<T, IsCSR>& matrix) {
    const Index primary_dim = matrix.primary_dim();

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);

        if (SCL_UNLIKELY(len == 0)) return;

        auto values = matrix.primary_values(idx);
        detail::log_softmax_adaptive(values.ptr, static_cast<Size>(len));
    });
}

/// @brief Apply log-softmax with temperature row-wise in-place to a sparse matrix
/// @param matrix Sparse matrix (CSR or CSC)
/// @param temperature Temperature parameter
template <typename T, bool IsCSR>
void log_softmax_inplace(Sparse<T, IsCSR>& matrix, T temperature) {
    const Index primary_dim = matrix.primary_dim();

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);

        if (SCL_UNLIKELY(len == 0)) return;

        auto values = matrix.primary_values(idx);
        detail::log_softmax_with_temperature(values.ptr, static_cast<Size>(len), temperature);
    });
}

} // namespace scl::kernel::softmax
