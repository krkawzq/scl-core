#pragma once

/**
 * @file scl/core/vectorize.hpp
 * @brief SIMD-optimized vectorized array operations.
 *
 * This header provides high-performance vectorized operations:
 *   - Reduction operations (sum, product, dot)
 *   - Search operations (find, count, contains)
 *   - Min/Max operations (min_value, max_value)
 *   - Transform operations (scale, add, mul)
 *   - Fill operations (fill, zero)
 *
 * @note Works with both std::span and scl::Span
 * @note All operations are SIMD-optimized via Google Highway
 * @note Compatible with C++20
 */

#include "scl/core/type.hpp"
#include "scl/core/macro.hpp"
#include "scl/core/error.hpp"
#include "scl/core/simd.hpp"

#include <algorithm>
#include <concepts>
#include <span>
#include <utility>

namespace scl::vectorize {

// =============================================================================
// SECTION 1: Concepts
// =============================================================================

/// @brief Concept for arithmetic types
template<typename T>
concept Arithmetic = std::integral<T> || std::floating_point<T>;

/// @brief Concept for totally ordered types
template<typename T>
concept TotallyOrdered = std::totally_ordered<T>;

// =============================================================================
// SECTION 2: Fill Operations
// =============================================================================

/// @brief Fill span with a value using SIMD
/// @tparam T Arithmetic element type
/// @param[out] data Span to fill
/// @param[in] value Fill value
template<Arithmetic T>
SCL_FORCE_INLINE
auto fill(std::span<T> data, T value) -> void {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = static_cast<Size>(data.size());
    const Size lanes = s::Lanes(d);

    if (N == 0) [[unlikely]] {
        return;
    }

    const auto v_val = s::Set(d, value);
    Size i = 0;

    // 4-way unrolled SIMD loop
    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        s::Store(v_val, d, data.data() + i);
        s::Store(v_val, d, data.data() + i + lanes);
        s::Store(v_val, d, data.data() + i + 2 * lanes);
        s::Store(v_val, d, data.data() + i + 3 * lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        s::Store(v_val, d, data.data() + i);
    }

    // Scalar tail
    for (; i < N; ++i) {
        data[i] = value;
    }
}

/// @brief Zero-initialize span using SIMD
/// @tparam T Arithmetic element type
/// @param[out] data Span to zero
template<Arithmetic T>
SCL_FORCE_INLINE
auto zero(std::span<T> data) -> void {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = static_cast<Size>(data.size());
    const Size lanes = s::Lanes(d);

    if (N == 0) [[unlikely]] {
        return;
    }

    const auto v_zero = s::Zero(d);
    Size i = 0;

    // 4-way unrolled SIMD loop
    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        s::Store(v_zero, d, data.data() + i);
        s::Store(v_zero, d, data.data() + i + lanes);
        s::Store(v_zero, d, data.data() + i + 2 * lanes);
        s::Store(v_zero, d, data.data() + i + 3 * lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        s::Store(v_zero, d, data.data() + i);
    }

    // Scalar tail
    for (; i < N; ++i) {
        data[i] = T{0};
    }
}

// =============================================================================
// SECTION 3: Reduction Operations
// =============================================================================

/// @brief Compute sum of elements using SIMD
/// @tparam T Arithmetic element type
/// @param[in] data Input span
/// @return Sum of all elements
template<Arithmetic T>
[[nodiscard]]
SCL_FORCE_INLINE
auto sum(std::span<const T> data) -> T {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = static_cast<Size>(data.size());
    const Size lanes = s::Lanes(d);

    if (N == 0) {
        return T{0};
    }

    auto sum0 = s::Zero(d);
    auto sum1 = s::Zero(d);
    auto sum2 = s::Zero(d);
    auto sum3 = s::Zero(d);

    Size i = 0;

    // 4-way unrolled SIMD loop
    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        sum0 = s::Add(sum0, s::Load(d, data.data() + i));
        sum1 = s::Add(sum1, s::Load(d, data.data() + i + lanes));
        sum2 = s::Add(sum2, s::Load(d, data.data() + i + 2 * lanes));
        sum3 = s::Add(sum3, s::Load(d, data.data() + i + 3 * lanes));
    }

    sum0 = s::Add(sum0, sum1);
    sum2 = s::Add(sum2, sum3);
    sum0 = s::Add(sum0, sum2);

    for (; i + lanes <= N; i += lanes) {
        sum0 = s::Add(sum0, s::Load(d, data.data() + i));
    }

    T result = s::GetLane(s::SumOfLanes(d, sum0));

    // Scalar tail
    for (; i < N; ++i) {
        result += data[i];
    }

    return result;
}

/// @brief Compute dot product of two vectors using SIMD
/// @tparam T Arithmetic element type
/// @param[in] a First input span
/// @param[in] b Second input span
/// @return Dot product a · b
/// @pre a.size() == b.size()
template<Arithmetic T>
[[nodiscard]]
SCL_FORCE_INLINE
auto dot(std::span<const T> a, std::span<const T> b) -> T {
    error::check_size_match(static_cast<Size>(a.size()), 
                           static_cast<Size>(b.size()));

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = static_cast<Size>(a.size());
    const Size lanes = s::Lanes(d);

    if (N == 0) {
        return T{0};
    }

    auto acc0 = s::Zero(d);
    auto acc1 = s::Zero(d);
    auto acc2 = s::Zero(d);
    auto acc3 = s::Zero(d);

    Size i = 0;

    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        acc0 = s::MulAdd(s::Load(d, a.data() + i), s::Load(d, b.data() + i), acc0);
        acc1 = s::MulAdd(s::Load(d, a.data() + i + lanes), s::Load(d, b.data() + i + lanes), acc1);
        acc2 = s::MulAdd(s::Load(d, a.data() + i + 2 * lanes), s::Load(d, b.data() + i + 2 * lanes), acc2);
        acc3 = s::MulAdd(s::Load(d, a.data() + i + 3 * lanes), s::Load(d, b.data() + i + 3 * lanes), acc3);
    }

    acc0 = s::Add(acc0, acc1);
    acc2 = s::Add(acc2, acc3);
    acc0 = s::Add(acc0, acc2);

    for (; i + lanes <= N; i += lanes) {
        acc0 = s::MulAdd(s::Load(d, a.data() + i), s::Load(d, b.data() + i), acc0);
    }

    T result = s::GetLane(s::SumOfLanes(d, acc0));

    for (; i < N; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

// =============================================================================
// SECTION 4: Search Operations
// =============================================================================

/// @brief Find first occurrence of value using SIMD
/// @tparam T Element type
/// @param[in] data Input span
/// @param[in] value Value to find
/// @return Index of first occurrence, or data.size() if not found
template<TotallyOrdered T>
[[nodiscard]]
SCL_FORCE_INLINE
auto find(std::span<const T> data, const T& value) -> Size {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = static_cast<Size>(data.size());
    const Size lanes = s::Lanes(d);

    const auto v_val = s::Set(d, value);

    Size i = 0;

    for (; i + lanes <= N; i += lanes) {
        auto v_data = s::Load(d, data.data() + i);
        auto mask = s::Eq(v_data, v_val);

        if (!s::AllFalse(d, mask)) {
            for (Size j = 0; j < lanes && i + j < N; ++j) {
                if (data[i + j] == value) {
                    return i + j;
                }
            }
        }
    }

    for (; i < N; ++i) {
        if (data[i] == value) {
            return i;
        }
    }

    return N;
}

// =============================================================================
// SECTION 5: Transform Operations
// =============================================================================

/// @brief Scale elements by a factor using SIMD
/// @tparam T Arithmetic element type
/// @param[in] src Source span
/// @param[out] dst Destination span
/// @param[in] factor Scale factor
/// @pre src.size() == dst.size()
template<Arithmetic T>
SCL_FORCE_INLINE
auto scale(std::span<const T> src, std::span<T> dst, T factor) -> void {
    error::check_size_match(static_cast<Size>(src.size()), 
                           static_cast<Size>(dst.size()));

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = static_cast<Size>(src.size());
    const Size lanes = s::Lanes(d);

    const auto v_scale = s::Set(d, factor);

    Size i = 0;

    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        s::Store(s::Mul(s::Load(d, src.data() + i), v_scale), d, dst.data() + i);
        s::Store(s::Mul(s::Load(d, src.data() + i + lanes), v_scale), d, dst.data() + i + lanes);
        s::Store(s::Mul(s::Load(d, src.data() + i + 2 * lanes), v_scale), d, dst.data() + i + 2 * lanes);
        s::Store(s::Mul(s::Load(d, src.data() + i + 3 * lanes), v_scale), d, dst.data() + i + 3 * lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Mul(s::Load(d, src.data() + i), v_scale), d, dst.data() + i);
    }

    for (; i < N; ++i) {
        dst[i] = src[i] * factor;
    }
}

/// @brief Element-wise addition using SIMD
/// @tparam T Arithmetic element type
/// @param[in] a First source span
/// @param[in] b Second source span
/// @param[out] dst Destination span
/// @pre a.size() == b.size() == dst.size()
template<Arithmetic T>
SCL_FORCE_INLINE
auto add(std::span<const T> a, std::span<const T> b, std::span<T> dst) -> void {
    const Size size_a = static_cast<Size>(a.size());
    const Size size_b = static_cast<Size>(b.size());
    const Size size_dst = static_cast<Size>(dst.size());
    
    error::check_size_match(size_a, size_b);
    error::check_size_match(size_a, size_dst);

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = size_a;
    const Size lanes = s::Lanes(d);

    Size i = 0;

    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        s::Store(s::Add(s::Load(d, a.data() + i), s::Load(d, b.data() + i)), d, dst.data() + i);
        s::Store(s::Add(s::Load(d, a.data() + i + lanes), s::Load(d, b.data() + i + lanes)), d, dst.data() + i + lanes);
        s::Store(s::Add(s::Load(d, a.data() + i + 2 * lanes), s::Load(d, b.data() + i + 2 * lanes)), d, dst.data() + i + 2 * lanes);
        s::Store(s::Add(s::Load(d, a.data() + i + 3 * lanes), s::Load(d, b.data() + i + 3 * lanes)), d, dst.data() + i + 3 * lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Add(s::Load(d, a.data() + i), s::Load(d, b.data() + i)), d, dst.data() + i);
    }

    for (; i < N; ++i) {
        dst[i] = a[i] + b[i];
    }
}

/// @brief Element-wise multiplication using SIMD
/// @tparam T Arithmetic element type
/// @param[in] a First source span
/// @param[in] b Second source span
/// @param[out] dst Destination span
/// @pre a.size() == b.size() == dst.size()
template<Arithmetic T>
SCL_FORCE_INLINE
auto mul(std::span<const T> a, std::span<const T> b, std::span<T> dst) -> void {
    const Size size_a = static_cast<Size>(a.size());
    const Size size_b = static_cast<Size>(b.size());
    const Size size_dst = static_cast<Size>(dst.size());
    
    error::check_size_match(size_a, size_b);
    error::check_size_match(size_a, size_dst);

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = size_a;
    const Size lanes = s::Lanes(d);

    Size i = 0;

    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        s::Store(s::Mul(s::Load(d, a.data() + i), s::Load(d, b.data() + i)), d, dst.data() + i);
        s::Store(s::Mul(s::Load(d, a.data() + i + lanes), s::Load(d, b.data() + i + lanes)), d, dst.data() + i + lanes);
        s::Store(s::Mul(s::Load(d, a.data() + i + 2 * lanes), s::Load(d, b.data() + i + 2 * lanes)), d, dst.data() + i + 2 * lanes);
        s::Store(s::Mul(s::Load(d, a.data() + i + 3 * lanes), s::Load(d, b.data() + i + 3 * lanes)), d, dst.data() + i + 3 * lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Mul(s::Load(d, a.data() + i), s::Load(d, b.data() + i)), d, dst.data() + i);
    }

    for (; i < N; ++i) {
        dst[i] = a[i] * b[i];
    }
}

// =============================================================================
// SECTION 7: Convenience Wrappers for scl::Span
// =============================================================================

/// @brief Check if value exists in span (SIMD-optimized)
/// @tparam T Element type
/// @param[in] data Input span
/// @param[in] value Value to find
/// @return true if found
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
auto contains(std::span<const T> data, const T& value) -> bool {
    return find(data, value) < static_cast<Size>(data.size());
}

// =============================================================================
// SECTION 6: Min/Max Operations
// =============================================================================

/// @brief Find minimum element using SIMD
/// @tparam T Totally ordered element type
/// @param[in] data Input span (must not be empty)
/// @return Minimum value
template<TotallyOrdered T>
[[nodiscard]]
SCL_FORCE_INLINE
auto min_value(std::span<const T> data) -> T {
    error::check_arg(!data.empty(), "min_value: empty span");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = static_cast<Size>(data.size());
    const Size lanes = s::Lanes(d);

    T min_val = data[0];

    if (N >= lanes) {
        auto v_min = s::Load(d, data.data());

        Size i = lanes;
        for (; i + lanes <= N; i += lanes) {
            auto v_data = s::Load(d, data.data() + i);
            v_min = s::Min(v_min, v_data);
        }

        min_val = s::GetLane(s::MinOfLanes(d, v_min));

        for (; i < N; ++i) {
            if (data[i] < min_val) {
                min_val = data[i];
            }
        }
    } else {
        for (Size i = 1; i < N; ++i) {
            if (data[i] < min_val) {
                min_val = data[i];
            }
        }
    }

    return min_val;
}

/// @brief Find maximum element using SIMD
/// @tparam T Totally ordered element type
/// @param[in] data Input span (must not be empty)
/// @return Maximum value
template<TotallyOrdered T>
[[nodiscard]]
SCL_FORCE_INLINE
auto max_value(std::span<const T> data) -> T {
    error::check_arg(!data.empty(), "max_value: empty span");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = static_cast<Size>(data.size());
    const Size lanes = s::Lanes(d);

    T max_val = data[0];

    if (N >= lanes) {
        auto v_max = s::Load(d, data.data());

        Size i = lanes;
        for (; i + lanes <= N; i += lanes) {
            auto v_data = s::Load(d, data.data() + i);
            v_max = s::Max(v_max, v_data);
        }

        max_val = s::GetLane(s::MaxOfLanes(d, v_max));

        for (; i < N; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
            }
        }
    } else {
        for (Size i = 1; i < N; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
            }
        }
    }

    return max_val;
}

}  // namespace scl::vectorize

