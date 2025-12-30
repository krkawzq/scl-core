#pragma once

/// @file scl/core/vectorize.hpp
/// @brief SIMD-optimized vectorized array operations
///
/// This header provides high-performance vectorized operations:
///   - Reduction operations (sum, product, dot)
///   - Search operations (find, count, contains)
///   - Min/Max operations (min_element, max_element, minmax)
///   - Transform operations (scale, add, sub, mul, div)
///   - Scatter/Gather operations
///   - Clamp operations
///   - Absolute value and mathematical functions
///   - Fused multiply-add operations
///   - Comparison operations
///
/// @note Uses std::span for non-owning views (C++20)
/// @note All operations are SIMD-optimized via Google Highway

#include "scl/core/type.hpp"
#include "scl/core/macro.hpp"
#include "scl/core/error.hpp"
#include "scl/core/simd.hpp"

#include <algorithm>
#include <cmath>
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

/// @brief Concept for copyable types
template<typename T>
concept Copyable = std::copyable<T>;

/// @brief Concept for unary operations
template<typename Op, typename T>
concept UnaryOperation = std::invocable<Op, T>;

/// @brief Concept for binary operations
template<typename Op, typename T, typename U>
concept BinaryOperation = std::invocable<Op, T, U>;

/// @brief Concept for index types
template<typename IdxT>
concept IndexType = std::integral<IdxT>;

// =============================================================================
// SECTION 2: Reduction Operations
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
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    if (N == 0) return T{0};

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

/// @brief Compute product of elements using SIMD
/// @tparam T Arithmetic element type
/// @param[in] data Input span
/// @return Product of all elements
template<Arithmetic T>
[[nodiscard]]
SCL_FORCE_INLINE
auto product(std::span<const T> data) -> T {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    if (N == 0) return T{1};

    auto prod0 = s::Set(d, T{1});
    auto prod1 = s::Set(d, T{1});

    Size i = 0;

    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        prod0 = s::Mul(prod0, s::Load(d, data.data() + i));
        prod1 = s::Mul(prod1, s::Load(d, data.data() + i + lanes));
    }

    prod0 = s::Mul(prod0, prod1);

    for (; i + lanes <= N; i += lanes) {
        prod0 = s::Mul(prod0, s::Load(d, data.data() + i));
    }

    // Reduce SIMD register to scalar
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
    alignas(64) T tmp[32];
    s::Store(prod0, d, tmp);

    T result = T{1};
    for (Size j = 0; j < lanes && j < 32; ++j) {
        result *= tmp[j];
    }

    for (; i < N; ++i) {
        result *= data[i];
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
    SCL_CHECK_ARG(a.size() == b.size(), "dot: size mismatch");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = a.size();
    const Size lanes = s::Lanes(d);

    if (N == 0) return T{0};

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
// SECTION 3: Search Operations
// =============================================================================

/// @brief Find first occurrence of value
/// @tparam T Element type
/// @param[in] data Input span
/// @param[in] value Value to find
/// @return Index of first occurrence, or data.size() if not found
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
auto find(std::span<const T> data, const T& value) -> Size {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    const auto v_val = s::Set(d, value);

    Size i = 0;

    for (; i + lanes <= N; i += lanes) {
        auto v_data = s::Load(d, data.data() + i);
        auto mask = s::Eq(v_data, v_val);

        if (!s::AllFalse(d, mask)) {
            for (Size j = 0; j < lanes && i + j < N; ++j) {
                if (data[i + j] == value) return i + j;
            }
        }
    }

    for (; i < N; ++i) {
        if (data[i] == value) return i;
    }

    return N;
}

/// @brief Count occurrences of value
/// @tparam T Element type
/// @param[in] data Input span
/// @param[in] value Value to count
/// @return Number of occurrences
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
auto count(std::span<const T> data, const T& value) -> Size {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    const auto v_val = s::Set(d, value);
    Size cnt = 0;

    Size i = 0;

    for (; i + lanes <= N; i += lanes) {
        auto v_data = s::Load(d, data.data() + i);
        auto mask = s::Eq(v_data, v_val);
        cnt += s::CountTrue(d, mask);
    }

    for (; i < N; ++i) {
        if (data[i] == value) ++cnt;
    }

    return cnt;
}

/// @brief Check if value exists in span
/// @tparam T Element type
/// @param[in] data Input span
/// @param[in] value Value to find
/// @return true if found
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
auto contains(std::span<const T> data, const T& value) -> bool {
    return find(data, value) < data.size();
}

// =============================================================================
// SECTION 4: Min/Max Operations
// =============================================================================

/// @brief Find index of minimum element
/// @tparam T Totally ordered element type
/// @param[in] data Input span (must not be empty)
/// @return Index of minimum element
template<TotallyOrdered T>
[[nodiscard]]
SCL_FORCE_INLINE
auto min_element(std::span<const T> data) -> Size {
    SCL_CHECK_ARG(!data.empty(), "min_element: empty span");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    T min_val = data[0];
    Size min_idx = 0;

    if (N >= lanes) {
        auto v_min = s::Load(d, data.data());

        Size i = lanes;
        for (; i + lanes <= N; i += lanes) {
            auto v_data = s::Load(d, data.data() + i);
            v_min = s::Min(v_min, v_data);
        }

        min_val = s::GetLane(s::MinOfLanes(d, v_min));

        for (; i < N; ++i) {
            if (data[i] < min_val) min_val = data[i];
        }

        // Find first occurrence of min value
        for (Size j = 0; j < N; ++j) {
            if (data[j] == min_val) {
                min_idx = j;
                break;
            }
        }
    } else {
        for (Size i = 1; i < N; ++i) {
            if (data[i] < min_val) {
                min_val = data[i];
                min_idx = i;
            }
        }
    }

    return min_idx;
}

/// @brief Find index of maximum element
/// @tparam T Totally ordered element type
/// @param[in] data Input span (must not be empty)
/// @return Index of maximum element
template<TotallyOrdered T>
[[nodiscard]]
SCL_FORCE_INLINE
auto max_element(std::span<const T> data) -> Size {
    SCL_CHECK_ARG(!data.empty(), "max_element: empty span");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    T max_val = data[0];
    Size max_idx = 0;

    if (N >= lanes) {
        auto v_max = s::Load(d, data.data());

        Size i = lanes;
        for (; i + lanes <= N; i += lanes) {
            auto v_data = s::Load(d, data.data() + i);
            v_max = s::Max(v_max, v_data);
        }

        max_val = s::GetLane(s::MaxOfLanes(d, v_max));

        for (; i < N; ++i) {
            if (data[i] > max_val) max_val = data[i];
        }

        for (Size j = 0; j < N; ++j) {
            if (data[j] == max_val) {
                max_idx = j;
                break;
            }
        }
    } else {
        for (Size i = 1; i < N; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
    }

    return max_idx;
}

/// @brief Find min and max values
/// @tparam T Totally ordered element type
/// @param[in] data Input span (must not be empty)
/// @return Pair of (min_value, max_value)
template<TotallyOrdered T>
[[nodiscard]]
SCL_FORCE_INLINE
auto minmax(std::span<const T> data) -> std::pair<T, T> {
    SCL_CHECK_ARG(!data.empty(), "minmax: empty span");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    T min_val = data[0];
    T max_val = data[0];

    if (N >= lanes) {
        auto v_min = s::Load(d, data.data());
        auto v_max = v_min;

        Size i = lanes;
        for (; i + lanes <= N; i += lanes) {
            auto v_data = s::Load(d, data.data() + i);
            v_min = s::Min(v_min, v_data);
            v_max = s::Max(v_max, v_data);
        }

        min_val = s::GetLane(s::MinOfLanes(d, v_min));
        max_val = s::GetLane(s::MaxOfLanes(d, v_max));

        for (; i < N; ++i) {
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }
    } else {
        for (Size i = 1; i < N; ++i) {
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }
    }

    return {min_val, max_val};
}

// =============================================================================
// SECTION 5: Transform Operations
// =============================================================================

/// @brief Transform elements in-place with unary operation
/// @tparam T Element type
/// @tparam UnaryOp Unary operation type
/// @param[in,out] data Span to transform
/// @param[in] op Unary operation
template<Copyable T, UnaryOperation<T> UnaryOp>
SCL_FORCE_INLINE
auto transform_inplace(std::span<T> data, UnaryOp op) -> void {
    for (auto& elem : data) {
        elem = op(elem);
    }
}

/// @brief Transform elements with unary operation
/// @tparam T Source element type
/// @tparam U Destination element type
/// @tparam UnaryOp Unary operation type
/// @param[in] src Source span
/// @param[out] dst Destination span
/// @param[in] op Unary operation
/// @pre src.size() == dst.size()
template<typename T, Copyable U, UnaryOperation<T> UnaryOp>
SCL_FORCE_INLINE
auto transform(std::span<const T> src, std::span<U> dst, UnaryOp op) -> void {
    SCL_CHECK_ARG(src.size() == dst.size(), "transform: size mismatch");

    for (Size i = 0; i < src.size(); ++i) {
        dst[i] = op(src[i]);
    }
}

/// @brief Transform elements with binary operation
/// @tparam T First source element type
/// @tparam U Second source element type
/// @tparam V Destination element type
/// @tparam BinaryOp Binary operation type
/// @param[in] a First source span
/// @param[in] b Second source span
/// @param[out] dst Destination span
/// @param[in] op Binary operation
/// @pre a.size() == b.size() == dst.size()
template<typename T, typename U, Copyable V, BinaryOperation<T, U> BinaryOp>
SCL_FORCE_INLINE
auto transform(std::span<const T> a, std::span<const U> b, std::span<V> dst, BinaryOp op) -> void {
    SCL_CHECK_ARG(a.size() == b.size() && b.size() == dst.size(), "transform: size mismatch");

    for (Size i = 0; i < a.size(); ++i) {
        dst[i] = op(a[i], b[i]);
    }
}

/// @brief Scale elements by a factor
/// @tparam T Arithmetic element type
/// @param[in] src Source span
/// @param[out] dst Destination span
/// @param[in] factor Scale factor
/// @pre src.size() == dst.size()
template<Arithmetic T>
SCL_FORCE_INLINE
auto scale(std::span<const T> src, std::span<T> dst, T factor) -> void {
    SCL_CHECK_ARG(src.size() == dst.size(), "scale: size mismatch");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = src.size();
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

/// @brief Scale elements in-place
/// @tparam T Arithmetic element type
/// @param[in,out] data Span to scale
/// @param[in] factor Scale factor
template<Arithmetic T>
SCL_FORCE_INLINE
auto scale_inplace(std::span<T> data, T factor) -> void {
    scale(std::span<const T>(data), data, factor);
}

/// @brief Add scalar to elements
/// @tparam T Arithmetic element type
/// @param[in] src Source span
/// @param[out] dst Destination span
/// @param[in] value Value to add
/// @pre src.size() == dst.size()
template<Arithmetic T>
SCL_FORCE_INLINE
auto add_scalar(std::span<const T> src, std::span<T> dst, T value) -> void {
    SCL_CHECK_ARG(src.size() == dst.size(), "add_scalar: size mismatch");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = src.size();
    const Size lanes = s::Lanes(d);

    const auto v_val = s::Set(d, value);

    Size i = 0;

    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        s::Store(s::Add(s::Load(d, src.data() + i), v_val), d, dst.data() + i);
        s::Store(s::Add(s::Load(d, src.data() + i + lanes), v_val), d, dst.data() + i + lanes);
        s::Store(s::Add(s::Load(d, src.data() + i + 2 * lanes), v_val), d, dst.data() + i + 2 * lanes);
        s::Store(s::Add(s::Load(d, src.data() + i + 3 * lanes), v_val), d, dst.data() + i + 3 * lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Add(s::Load(d, src.data() + i), v_val), d, dst.data() + i);
    }

    for (; i < N; ++i) {
        dst[i] = src[i] + value;
    }
}

/// @brief Element-wise addition
/// @tparam T Arithmetic element type
/// @param[in] a First source span
/// @param[in] b Second source span
/// @param[out] dst Destination span
/// @pre a.size() == b.size() == dst.size()
template<Arithmetic T>
SCL_FORCE_INLINE
auto add(std::span<const T> a, std::span<const T> b, std::span<T> dst) -> void {
    SCL_CHECK_ARG(a.size() == b.size() && b.size() == dst.size(), "add: size mismatch");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = a.size();
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

/// @brief Element-wise subtraction
/// @tparam T Arithmetic element type
/// @param[in] a First source span
/// @param[in] b Second source span
/// @param[out] dst Destination span
/// @pre a.size() == b.size() == dst.size()
template<Arithmetic T>
SCL_FORCE_INLINE
auto sub(std::span<const T> a, std::span<const T> b, std::span<T> dst) -> void {
    SCL_CHECK_ARG(a.size() == b.size() && b.size() == dst.size(), "sub: size mismatch");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = a.size();
    const Size lanes = s::Lanes(d);

    Size i = 0;

    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        s::Store(s::Sub(s::Load(d, a.data() + i), s::Load(d, b.data() + i)), d, dst.data() + i);
        s::Store(s::Sub(s::Load(d, a.data() + i + lanes), s::Load(d, b.data() + i + lanes)), d, dst.data() + i + lanes);
        s::Store(s::Sub(s::Load(d, a.data() + i + 2 * lanes), s::Load(d, b.data() + i + 2 * lanes)), d, dst.data() + i + 2 * lanes);
        s::Store(s::Sub(s::Load(d, a.data() + i + 3 * lanes), s::Load(d, b.data() + i + 3 * lanes)), d, dst.data() + i + 3 * lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Sub(s::Load(d, a.data() + i), s::Load(d, b.data() + i)), d, dst.data() + i);
    }

    for (; i < N; ++i) {
        dst[i] = a[i] - b[i];
    }
}

/// @brief Element-wise multiplication
/// @tparam T Arithmetic element type
/// @param[in] a First source span
/// @param[in] b Second source span
/// @param[out] dst Destination span
/// @pre a.size() == b.size() == dst.size()
template<Arithmetic T>
SCL_FORCE_INLINE
auto mul(std::span<const T> a, std::span<const T> b, std::span<T> dst) -> void {
    SCL_CHECK_ARG(a.size() == b.size() && b.size() == dst.size(), "mul: size mismatch");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = a.size();
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

/// @brief Element-wise division
/// @tparam T Arithmetic element type
/// @param[in] a First source span (dividend)
/// @param[in] b Second source span (divisor)
/// @param[out] dst Destination span
/// @pre a.size() == b.size() == dst.size()
template<Arithmetic T>
SCL_FORCE_INLINE
auto div(std::span<const T> a, std::span<const T> b, std::span<T> dst) -> void {
    SCL_CHECK_ARG(a.size() == b.size() && b.size() == dst.size(), "div: size mismatch");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = a.size();
    const Size lanes = s::Lanes(d);

    Size i = 0;

    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        s::Store(s::Div(s::Load(d, a.data() + i), s::Load(d, b.data() + i)), d, dst.data() + i);
        s::Store(s::Div(s::Load(d, a.data() + i + lanes), s::Load(d, b.data() + i + lanes)), d, dst.data() + i + lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Div(s::Load(d, a.data() + i), s::Load(d, b.data() + i)), d, dst.data() + i);
    }

    for (; i < N; ++i) {
        dst[i] = a[i] / b[i];
    }
}

// =============================================================================
// SECTION 6: Scatter/Gather Operations
// =============================================================================

/// @brief Gather elements from source using indices
/// @tparam T Element type
/// @tparam IdxT Index type
/// @param[in] src Source array
/// @param[in] indices Index span
/// @param[out] dst Destination span
/// @pre indices.size() == dst.size()
template<typename T, typename IdxT>
SCL_FORCE_INLINE
auto gather(
    const T* SCL_RESTRICT src,
    std::span<const IdxT> indices,
    std::span<T> dst
) -> void {
    SCL_CHECK_ARG(indices.size() == dst.size(), "gather: size mismatch");

    const Size N = indices.size();

    for (Size i = 0; i < N; ++i) {
        if (i + 8 < N) {
            SCL_PREFETCH_READ(&indices[i + 8], 0);
            SCL_PREFETCH_READ(src + indices[i + 4], 0);
        }
        dst[i] = src[indices[i]];
    }
}

/// @brief Scatter elements to destination using indices
/// @tparam T Element type
/// @tparam IdxT Index type
/// @param[in] src Source span
/// @param[in] indices Index span
/// @param[out] dst Destination array
/// @pre src.size() == indices.size()
template<typename T, typename IdxT>
SCL_FORCE_INLINE
auto scatter(
    std::span<const T> src,
    std::span<const IdxT> indices,
    T* SCL_RESTRICT dst
) -> void {
    SCL_CHECK_ARG(src.size() == indices.size(), "scatter: size mismatch");

    for (Size i = 0; i < src.size(); ++i) {
        dst[indices[i]] = src[i];
    }
}

/// @brief Scatter-add elements to destination using indices
/// @tparam T Arithmetic element type
/// @tparam IdxT Index type
/// @param[in] src Source span
/// @param[in] indices Index span
/// @param[out] dst Destination array
/// @pre src.size() == indices.size()
template<Arithmetic T, IndexType IdxT>
SCL_FORCE_INLINE
auto scatter_add(
    std::span<const T> src,
    std::span<const IdxT> indices,
    T* SCL_RESTRICT dst
) -> void {
    SCL_CHECK_ARG(src.size() == indices.size(), "scatter_add: size mismatch");

    for (Size i = 0; i < src.size(); ++i) {
        dst[indices[i]] += src[i];
    }
}

// =============================================================================
// SECTION 7: Clamp Operations
// =============================================================================

/// @brief Clamp elements to range [min_val, max_val]
/// @tparam T Totally ordered element type
/// @param[in,out] data Span to clamp
/// @param[in] min_val Minimum value
/// @param[in] max_val Maximum value
template<TotallyOrdered T>
SCL_FORCE_INLINE
auto clamp(std::span<T> data, T min_val, T max_val) -> void {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    const auto v_min = s::Set(d, min_val);
    const auto v_max = s::Set(d, max_val);

    Size i = 0;

    for (; i + lanes <= N; i += lanes) {
        auto v = s::Load(d, data.data() + i);
        v = s::Max(v, v_min);
        v = s::Min(v, v_max);
        s::Store(v, d, data.data() + i);
    }

    for (; i < N; ++i) {
        if (data[i] < min_val) data[i] = min_val;
        else if (data[i] > max_val) data[i] = max_val;
    }
}

/// @brief Clamp elements to minimum value
/// @tparam T Totally ordered element type
/// @param[in,out] data Span to clamp
/// @param[in] min_val Minimum value
template<TotallyOrdered T>
SCL_FORCE_INLINE
auto clamp_min(std::span<T> data, T min_val) -> void {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    const auto v_min = s::Set(d, min_val);

    Size i = 0;

    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        s::Store(s::Max(s::Load(d, data.data() + i), v_min), d, data.data() + i);
        s::Store(s::Max(s::Load(d, data.data() + i + lanes), v_min), d, data.data() + i + lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Max(s::Load(d, data.data() + i), v_min), d, data.data() + i);
    }

    for (; i < N; ++i) {
        if (data[i] < min_val) data[i] = min_val;
    }
}

/// @brief Clamp elements to maximum value
/// @tparam T Totally ordered element type
/// @param[in,out] data Span to clamp
/// @param[in] max_val Maximum value
template<TotallyOrdered T>
SCL_FORCE_INLINE
auto clamp_max(std::span<T> data, T max_val) -> void {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    const auto v_max = s::Set(d, max_val);

    Size i = 0;

    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        s::Store(s::Min(s::Load(d, data.data() + i), v_max), d, data.data() + i);
        s::Store(s::Min(s::Load(d, data.data() + i + lanes), v_max), d, data.data() + i + lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Min(s::Load(d, data.data() + i), v_max), d, data.data() + i);
    }

    for (; i < N; ++i) {
        if (data[i] > max_val) data[i] = max_val;
    }
}

// =============================================================================
// SECTION 8: Absolute Value Operations
// =============================================================================

/// @brief Compute absolute value in-place
/// @tparam T Arithmetic element type
/// @param[in,out] data Span to transform
template<Arithmetic T>
SCL_FORCE_INLINE
auto abs_inplace(std::span<T> data) -> void {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    Size i = 0;

    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        s::Store(s::Abs(s::Load(d, data.data() + i)), d, data.data() + i);
        s::Store(s::Abs(s::Load(d, data.data() + i + lanes)), d, data.data() + i + lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Abs(s::Load(d, data.data() + i)), d, data.data() + i);
    }

    for (; i < N; ++i) {
        data[i] = (data[i] < T{0}) ? -data[i] : data[i];
    }
}

/// @brief Compute sum of absolute values (L1 norm)
/// @tparam T Arithmetic element type
/// @param[in] data Input span
/// @return Sum of absolute values
template<Arithmetic T>
[[nodiscard]]
SCL_FORCE_INLINE
auto sum_abs(std::span<const T> data) -> T {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    if (N == 0) return T{0};

    auto acc0 = s::Zero(d);
    auto acc1 = s::Zero(d);

    Size i = 0;

    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        acc0 = s::Add(acc0, s::Abs(s::Load(d, data.data() + i)));
        acc1 = s::Add(acc1, s::Abs(s::Load(d, data.data() + i + lanes)));
    }

    acc0 = s::Add(acc0, acc1);

    for (; i + lanes <= N; i += lanes) {
        acc0 = s::Add(acc0, s::Abs(s::Load(d, data.data() + i)));
    }

    T result = s::GetLane(s::SumOfLanes(d, acc0));

    for (; i < N; ++i) {
        result += (data[i] < T{0}) ? -data[i] : data[i];
    }

    return result;
}

/// @brief Compute sum of squared values (squared L2 norm)
/// @tparam T Arithmetic element type
/// @param[in] data Input span
/// @return Sum of squared values
template<Arithmetic T>
[[nodiscard]]
SCL_FORCE_INLINE
auto sum_squared(std::span<const T> data) -> T {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    if (N == 0) return T{0};

    auto acc0 = s::Zero(d);
    auto acc1 = s::Zero(d);
    auto acc2 = s::Zero(d);
    auto acc3 = s::Zero(d);

    Size i = 0;

    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        auto v0 = s::Load(d, data.data() + i);
        auto v1 = s::Load(d, data.data() + i + lanes);
        auto v2 = s::Load(d, data.data() + i + 2 * lanes);
        auto v3 = s::Load(d, data.data() + i + 3 * lanes);

        acc0 = s::MulAdd(v0, v0, acc0);
        acc1 = s::MulAdd(v1, v1, acc1);
        acc2 = s::MulAdd(v2, v2, acc2);
        acc3 = s::MulAdd(v3, v3, acc3);
    }

    acc0 = s::Add(acc0, acc1);
    acc2 = s::Add(acc2, acc3);
    acc0 = s::Add(acc0, acc2);

    for (; i + lanes <= N; i += lanes) {
        auto v = s::Load(d, data.data() + i);
        acc0 = s::MulAdd(v, v, acc0);
    }

    T result = s::GetLane(s::SumOfLanes(d, acc0));

    for (; i < N; ++i) {
        result += data[i] * data[i];
    }

    return result;
}

// =============================================================================
// SECTION 9: Fused Multiply-Add Operations
// =============================================================================

/// @brief Fused multiply-add: dst = a * b + c
/// @tparam T Arithmetic element type
/// @param[in] a First multiplicand span
/// @param[in] b Second multiplicand span
/// @param[in] c Addend span
/// @param[out] dst Destination span
/// @pre All spans must have same size
template<Arithmetic T>
SCL_FORCE_INLINE
auto fma(
    std::span<const T> a,
    std::span<const T> b,
    std::span<const T> c,
    std::span<T> dst
) -> void {
    SCL_CHECK_ARG(a.size() == b.size() && b.size() == c.size() && c.size() == dst.size(),
                  "fma: size mismatch");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = a.size();
    const Size lanes = s::Lanes(d);

    Size i = 0;

    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        auto va0 = s::Load(d, a.data() + i);
        auto vb0 = s::Load(d, b.data() + i);
        auto vc0 = s::Load(d, c.data() + i);
        auto va1 = s::Load(d, a.data() + i + lanes);
        auto vb1 = s::Load(d, b.data() + i + lanes);
        auto vc1 = s::Load(d, c.data() + i + lanes);

        s::Store(s::MulAdd(va0, vb0, vc0), d, dst.data() + i);
        s::Store(s::MulAdd(va1, vb1, vc1), d, dst.data() + i + lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        auto va = s::Load(d, a.data() + i);
        auto vb = s::Load(d, b.data() + i);
        auto vc = s::Load(d, c.data() + i);
        s::Store(s::MulAdd(va, vb, vc), d, dst.data() + i);
    }

    for (; i < N; ++i) {
        dst[i] = a[i] * b[i] + c[i];
    }
}

/// @brief BLAS axpy: y = alpha * x + y
/// @tparam T Arithmetic element type
/// @param[in] alpha Scalar multiplier
/// @param[in] x Input span
/// @param[in,out] y Input/output span
/// @pre x.size() == y.size()
template<Arithmetic T>
SCL_FORCE_INLINE
auto axpy(T alpha, std::span<const T> x, std::span<T> y) -> void {
    SCL_CHECK_ARG(x.size() == y.size(), "axpy: size mismatch");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = x.size();
    const Size lanes = s::Lanes(d);

    const auto v_alpha = s::Set(d, alpha);

    Size i = 0;

    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        auto vy0 = s::Load(d, y.data() + i);
        auto vy1 = s::Load(d, y.data() + i + lanes);
        auto vy2 = s::Load(d, y.data() + i + 2 * lanes);
        auto vy3 = s::Load(d, y.data() + i + 3 * lanes);

        s::Store(s::MulAdd(v_alpha, s::Load(d, x.data() + i), vy0), d, y.data() + i);
        s::Store(s::MulAdd(v_alpha, s::Load(d, x.data() + i + lanes), vy1), d, y.data() + i + lanes);
        s::Store(s::MulAdd(v_alpha, s::Load(d, x.data() + i + 2 * lanes), vy2), d, y.data() + i + 2 * lanes);
        s::Store(s::MulAdd(v_alpha, s::Load(d, x.data() + i + 3 * lanes), vy3), d, y.data() + i + 3 * lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        auto vy = s::Load(d, y.data() + i);
        s::Store(s::MulAdd(v_alpha, s::Load(d, x.data() + i), vy), d, y.data() + i);
    }

    for (; i < N; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

// =============================================================================
// SECTION 10: Mathematical Functions
// =============================================================================

/// @brief Element-wise square root
/// @tparam T Floating-point element type
/// @param[in] src Source span
/// @param[out] dst Destination span
/// @pre src.size() == dst.size()
template<std::floating_point T>
SCL_FORCE_INLINE
auto sqrt(std::span<const T> src, std::span<T> dst) -> void {
    SCL_CHECK_ARG(src.size() == dst.size(), "sqrt: size mismatch");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = src.size();
    const Size lanes = s::Lanes(d);

    Size i = 0;

    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        s::Store(s::Sqrt(s::Load(d, src.data() + i)), d, dst.data() + i);
        s::Store(s::Sqrt(s::Load(d, src.data() + i + lanes)), d, dst.data() + i + lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Sqrt(s::Load(d, src.data() + i)), d, dst.data() + i);
    }

    for (; i < N; ++i) {
        dst[i] = std::sqrt(src[i]);
    }
}

/// @brief Element-wise square root in-place
/// @tparam T Floating-point element type
/// @param[in,out] data Span to transform
template<std::floating_point T>
SCL_FORCE_INLINE
auto sqrt_inplace(std::span<T> data) -> void {
    sqrt(std::span<const T>(data), data);
}

/// @brief Element-wise reciprocal square root (1/sqrt)
/// @tparam T Floating-point element type
/// @param[in] src Source span
/// @param[out] dst Destination span
/// @pre src.size() == dst.size()
template<std::floating_point T>
SCL_FORCE_INLINE
auto rsqrt(std::span<const T> src, std::span<T> dst) -> void {
    SCL_CHECK_ARG(src.size() == dst.size(), "rsqrt: size mismatch");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = src.size();
    const Size lanes = s::Lanes(d);

    const auto v_one = s::Set(d, T{1});

    Size i = 0;

    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        s::Store(s::Div(v_one, s::Sqrt(s::Load(d, src.data() + i))), d, dst.data() + i);
        s::Store(s::Div(v_one, s::Sqrt(s::Load(d, src.data() + i + lanes))), d, dst.data() + i + lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Div(v_one, s::Sqrt(s::Load(d, src.data() + i))), d, dst.data() + i);
    }

    for (; i < N; ++i) {
        dst[i] = T{1} / std::sqrt(src[i]);
    }
}

/// @brief Element-wise square
/// @tparam T Arithmetic element type
/// @param[in] src Source span
/// @param[out] dst Destination span
/// @pre src.size() == dst.size()
template<Arithmetic T>
SCL_FORCE_INLINE
auto square(std::span<const T> src, std::span<T> dst) -> void {
    SCL_CHECK_ARG(src.size() == dst.size(), "square: size mismatch");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = src.size();
    const Size lanes = s::Lanes(d);

    Size i = 0;

    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        auto v0 = s::Load(d, src.data() + i);
        auto v1 = s::Load(d, src.data() + i + lanes);
        auto v2 = s::Load(d, src.data() + i + 2 * lanes);
        auto v3 = s::Load(d, src.data() + i + 3 * lanes);

        s::Store(s::Mul(v0, v0), d, dst.data() + i);
        s::Store(s::Mul(v1, v1), d, dst.data() + i + lanes);
        s::Store(s::Mul(v2, v2), d, dst.data() + i + 2 * lanes);
        s::Store(s::Mul(v3, v3), d, dst.data() + i + 3 * lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        auto v = s::Load(d, src.data() + i);
        s::Store(s::Mul(v, v), d, dst.data() + i);
    }

    for (; i < N; ++i) {
        dst[i] = src[i] * src[i];
    }
}

/// @brief Element-wise square in-place
/// @tparam T Arithmetic element type
/// @param[in,out] data Span to transform
template<Arithmetic T>
SCL_FORCE_INLINE
auto square_inplace(std::span<T> data) -> void {
    square(std::span<const T>(data), data);
}

/// @brief Element-wise negation
/// @tparam T Arithmetic element type
/// @param[in] src Source span
/// @param[out] dst Destination span
/// @pre src.size() == dst.size()
template<Arithmetic T>
SCL_FORCE_INLINE
auto negate(std::span<const T> src, std::span<T> dst) -> void {
    SCL_CHECK_ARG(src.size() == dst.size(), "negate: size mismatch");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = src.size();
    const Size lanes = s::Lanes(d);

    const auto v_zero = s::Zero(d);

    Size i = 0;

    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        s::Store(s::Sub(v_zero, s::Load(d, src.data() + i)), d, dst.data() + i);
        s::Store(s::Sub(v_zero, s::Load(d, src.data() + i + lanes)), d, dst.data() + i + lanes);
        s::Store(s::Sub(v_zero, s::Load(d, src.data() + i + 2 * lanes)), d, dst.data() + i + 2 * lanes);
        s::Store(s::Sub(v_zero, s::Load(d, src.data() + i + 3 * lanes)), d, dst.data() + i + 3 * lanes);
    }

    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Sub(v_zero, s::Load(d, src.data() + i)), d, dst.data() + i);
    }

    for (; i < N; ++i) {
        dst[i] = -src[i];
    }
}

/// @brief Element-wise negation in-place
/// @tparam T Arithmetic element type
/// @param[in,out] data Span to transform
template<Arithmetic T>
SCL_FORCE_INLINE
auto negate_inplace(std::span<T> data) -> void {
    negate(std::span<const T>(data), data);
}

// =============================================================================
// SECTION 11: Comparison Operations
// =============================================================================

/// @brief Count non-zero elements
/// @tparam T Arithmetic element type
/// @param[in] data Input span
/// @return Number of non-zero elements
template<Arithmetic T>
[[nodiscard]]
SCL_FORCE_INLINE
auto count_nonzero(std::span<const T> data) -> Size {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    const auto v_zero = s::Zero(d);
    Size cnt = 0;

    Size i = 0;

    for (; i + lanes <= N; i += lanes) {
        auto v_data = s::Load(d, data.data() + i);
        auto mask = s::Ne(v_data, v_zero);
        cnt += s::CountTrue(d, mask);
    }

    for (; i < N; ++i) {
        if (data[i] != T{0}) ++cnt;
    }

    return cnt;
}

/// @brief Check if all elements equal value
/// @tparam T Element type
/// @param[in] data Input span
/// @param[in] value Value to compare
/// @return true if all elements equal value
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
auto all(std::span<const T> data, const T& value) -> bool {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    const auto v_val = s::Set(d, value);

    Size i = 0;

    for (; i + lanes <= N; i += lanes) {
        auto v_data = s::Load(d, data.data() + i);
        auto mask = s::Eq(v_data, v_val);
        if (!s::AllTrue(d, mask)) return false;
    }

    for (; i < N; ++i) {
        if (data[i] != value) return false;
    }

    return true;
}

/// @brief Check if any element equals value
/// @tparam T Element type
/// @param[in] data Input span
/// @param[in] value Value to compare
/// @return true if any element equals value
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
auto any(std::span<const T> data, const T& value) -> bool {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size N = data.size();
    const Size lanes = s::Lanes(d);

    const auto v_val = s::Set(d, value);

    Size i = 0;

    for (; i + lanes <= N; i += lanes) {
        auto v_data = s::Load(d, data.data() + i);
        auto mask = s::Eq(v_data, v_val);
        if (!s::AllFalse(d, mask)) return true;
    }

    for (; i < N; ++i) {
        if (data[i] == value) return true;
    }

    return false;
}

}  // namespace scl::vectorize

