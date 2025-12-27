#pragma once

#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/memory.hpp"

#include <utility>
#include <algorithm>
#include <cmath>

// =============================================================================
// FILE: scl/core/vectorize.hpp
// BRIEF: SIMD-optimized vectorized array operations
// =============================================================================

namespace scl::vectorize {

// =============================================================================
// 1. Reduction Operations
// =============================================================================

template <typename T>
SCL_FORCE_INLINE T sum(Array<const T> span) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    if (N == 0) return T(0);
    
    auto sum0 = s::Zero(d);
    auto sum1 = s::Zero(d);
    auto sum2 = s::Zero(d);
    auto sum3 = s::Zero(d);
    
    size_t i = 0;
    
    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        sum0 = s::Add(sum0, s::Load(d, span.ptr + i));
        sum1 = s::Add(sum1, s::Load(d, span.ptr + i + lanes));
        sum2 = s::Add(sum2, s::Load(d, span.ptr + i + 2 * lanes));
        sum3 = s::Add(sum3, s::Load(d, span.ptr + i + 3 * lanes));
    }
    
    sum0 = s::Add(sum0, sum1);
    sum2 = s::Add(sum2, sum3);
    sum0 = s::Add(sum0, sum2);
    
    for (; i + lanes <= N; i += lanes) {
        sum0 = s::Add(sum0, s::Load(d, span.ptr + i));
    }
    
    T result = s::GetLane(s::SumOfLanes(d, sum0));
    
    for (; i < N; ++i) {
        result += span[i];
    }
    
    return result;
}

template <typename T>
SCL_FORCE_INLINE T product(Array<const T> span) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    if (N == 0) return T(1);
    
    auto prod0 = s::Set(d, T(1));
    auto prod1 = s::Set(d, T(1));
    
    size_t i = 0;
    
    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        prod0 = s::Mul(prod0, s::Load(d, span.ptr + i));
        prod1 = s::Mul(prod1, s::Load(d, span.ptr + i + lanes));
    }
    
    prod0 = s::Mul(prod0, prod1);
    
    for (; i + lanes <= N; i += lanes) {
        prod0 = s::Mul(prod0, s::Load(d, span.ptr + i));
    }
    
    alignas(64) T tmp[32];
    s::Store(prod0, d, tmp);
    
    T result = T(1);
    for (size_t j = 0; j < lanes && j < 32; ++j) {
        result *= tmp[j];
    }
    
    for (; i < N; ++i) {
        result *= span[i];
    }
    
    return result;
}

template <typename T>
SCL_FORCE_INLINE T dot(Array<const T> a, Array<const T> b) {
    SCL_ASSERT(a.len == b.len, "dot: Size mismatch");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = a.len;
    const size_t lanes = s::Lanes(d);
    
    if (N == 0) return T(0);
    
    auto acc0 = s::Zero(d);
    auto acc1 = s::Zero(d);
    auto acc2 = s::Zero(d);
    auto acc3 = s::Zero(d);
    
    size_t i = 0;
    
    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        acc0 = s::MulAdd(s::Load(d, a.ptr + i), s::Load(d, b.ptr + i), acc0);
        acc1 = s::MulAdd(s::Load(d, a.ptr + i + lanes), s::Load(d, b.ptr + i + lanes), acc1);
        acc2 = s::MulAdd(s::Load(d, a.ptr + i + 2*lanes), s::Load(d, b.ptr + i + 2*lanes), acc2);
        acc3 = s::MulAdd(s::Load(d, a.ptr + i + 3*lanes), s::Load(d, b.ptr + i + 3*lanes), acc3);
    }
    
    acc0 = s::Add(acc0, acc1);
    acc2 = s::Add(acc2, acc3);
    acc0 = s::Add(acc0, acc2);
    
    for (; i + lanes <= N; i += lanes) {
        acc0 = s::MulAdd(s::Load(d, a.ptr + i), s::Load(d, b.ptr + i), acc0);
    }
    
    T result = s::GetLane(s::SumOfLanes(d, acc0));
    
    for (; i < N; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

// =============================================================================
// 2. Search Operations
// =============================================================================

template <typename T>
SCL_FORCE_INLINE size_t find(Array<const T> span, T value) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    const auto v_val = s::Set(d, value);
    
    size_t i = 0;
    
    for (; i + lanes <= N; i += lanes) {
        auto v_data = s::Load(d, span.ptr + i);
        auto mask = s::Eq(v_data, v_val);
        
        if (!s::AllFalse(d, mask)) {
            for (size_t j = 0; j < lanes && i + j < N; ++j) {
                if (span[i + j] == value) return i + j;
            }
        }
    }
    
    for (; i < N; ++i) {
        if (span[i] == value) return i;
    }
    
    return N;
}

template <typename T>
SCL_FORCE_INLINE size_t count(Array<const T> span, T value) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    const auto v_val = s::Set(d, value);
    size_t cnt = 0;
    
    size_t i = 0;
    
    for (; i + lanes <= N; i += lanes) {
        auto v_data = s::Load(d, span.ptr + i);
        auto mask = s::Eq(v_data, v_val);
        cnt += s::CountTrue(d, mask);
    }
    
    for (; i < N; ++i) {
        if (span[i] == value) ++cnt;
    }
    
    return cnt;
}

template <typename T>
SCL_FORCE_INLINE bool contains(Array<const T> span, T value) {
    return find(span, value) < span.len;
}

// =============================================================================
// 3. Min/Max Operations
// =============================================================================

template <typename T>
SCL_FORCE_INLINE size_t min_element(Array<const T> span) {
    SCL_ASSERT(span.len > 0, "min_element: Empty span");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    T min_val = span[0];
    size_t min_idx = 0;
    
    if (N >= lanes) {
        auto v_min = s::Load(d, span.ptr);
        
        size_t i = lanes;
        for (; i + lanes <= N; i += lanes) {
            auto v_data = s::Load(d, span.ptr + i);
            v_min = s::Min(v_min, v_data);
        }
        
        min_val = s::GetLane(s::MinOfLanes(d, v_min));
        
        for (; i < N; ++i) {
            if (span[i] < min_val) min_val = span[i];
        }
        
        for (size_t j = 0; j < N; ++j) {
            if (span[j] == min_val) {
                min_idx = j;
                break;
            }
        }
    } else {
        for (size_t i = 1; i < N; ++i) {
            if (span[i] < min_val) {
                min_val = span[i];
                min_idx = i;
            }
        }
    }
    
    return min_idx;
}

template <typename T>
SCL_FORCE_INLINE size_t max_element(Array<const T> span) {
    SCL_ASSERT(span.len > 0, "max_element: Empty span");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    T max_val = span[0];
    size_t max_idx = 0;
    
    if (N >= lanes) {
        auto v_max = s::Load(d, span.ptr);
        
        size_t i = lanes;
        for (; i + lanes <= N; i += lanes) {
            auto v_data = s::Load(d, span.ptr + i);
            v_max = s::Max(v_max, v_data);
        }
        
        max_val = s::GetLane(s::MaxOfLanes(d, v_max));
        
        for (; i < N; ++i) {
            if (span[i] > max_val) max_val = span[i];
        }
        
        for (size_t j = 0; j < N; ++j) {
            if (span[j] == max_val) {
                max_idx = j;
                break;
            }
        }
    } else {
        for (size_t i = 1; i < N; ++i) {
            if (span[i] > max_val) {
                max_val = span[i];
                max_idx = i;
            }
        }
    }
    
    return max_idx;
}

template <typename T>
SCL_FORCE_INLINE std::pair<T, T> minmax(Array<const T> span) {
    SCL_ASSERT(span.len > 0, "minmax: Empty span");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    T min_val = span[0];
    T max_val = span[0];
    
    if (N >= lanes) {
        auto v_min = s::Load(d, span.ptr);
        auto v_max = v_min;
        
        size_t i = lanes;
        for (; i + lanes <= N; i += lanes) {
            auto v_data = s::Load(d, span.ptr + i);
            v_min = s::Min(v_min, v_data);
            v_max = s::Max(v_max, v_data);
        }
        
        min_val = s::GetLane(s::MinOfLanes(d, v_min));
        max_val = s::GetLane(s::MaxOfLanes(d, v_max));
        
        for (; i < N; ++i) {
            if (span[i] < min_val) min_val = span[i];
            if (span[i] > max_val) max_val = span[i];
        }
    } else {
        for (size_t i = 1; i < N; ++i) {
            if (span[i] < min_val) min_val = span[i];
            if (span[i] > max_val) max_val = span[i];
        }
    }
    
    return {min_val, max_val};
}

// =============================================================================
// 4. Transform Operations
// =============================================================================

template <typename T, typename UnaryOp>
SCL_FORCE_INLINE void transform_inplace(Array<T> span, UnaryOp op) {
    for (size_t i = 0; i < span.len; ++i) {
        span[i] = op(span[i]);
    }
}

template <typename T, typename U, typename UnaryOp>
SCL_FORCE_INLINE void transform(Array<const T> src, Array<U> dst, UnaryOp op) {
    SCL_ASSERT(src.len == dst.len, "transform: Size mismatch");
    
    for (size_t i = 0; i < src.len; ++i) {
        dst[i] = op(src[i]);
    }
}

template <typename T, typename U, typename V, typename BinaryOp>
SCL_FORCE_INLINE void transform(Array<const T> a, Array<const U> b, Array<V> dst, BinaryOp op) {
    SCL_ASSERT(a.len == b.len && b.len == dst.len, "transform: Size mismatch");
    
    for (size_t i = 0; i < a.len; ++i) {
        dst[i] = op(a[i], b[i]);
    }
}

template <typename T>
SCL_FORCE_INLINE void scale(Array<const T> src, Array<T> dst, T scale_factor) {
    SCL_ASSERT(src.len == dst.len, "scale: Size mismatch");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = src.len;
    const size_t lanes = s::Lanes(d);
    
    const auto v_scale = s::Set(d, scale_factor);
    
    size_t i = 0;
    
    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        s::Store(s::Mul(s::Load(d, src.ptr + i), v_scale), d, dst.ptr + i);
        s::Store(s::Mul(s::Load(d, src.ptr + i + lanes), v_scale), d, dst.ptr + i + lanes);
        s::Store(s::Mul(s::Load(d, src.ptr + i + 2*lanes), v_scale), d, dst.ptr + i + 2*lanes);
        s::Store(s::Mul(s::Load(d, src.ptr + i + 3*lanes), v_scale), d, dst.ptr + i + 3*lanes);
    }
    
    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Mul(s::Load(d, src.ptr + i), v_scale), d, dst.ptr + i);
    }
    
    for (; i < N; ++i) {
        dst[i] = src[i] * scale_factor;
    }
}

template <typename T>
SCL_FORCE_INLINE void scale_inplace(Array<T> span, T scale_factor) {
    scale(Array<const T>(span.ptr, span.len), span, scale_factor);
}

template <typename T>
SCL_FORCE_INLINE void add_scalar(Array<const T> src, Array<T> dst, T value) {
    SCL_ASSERT(src.len == dst.len, "add_scalar: Size mismatch");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = src.len;
    const size_t lanes = s::Lanes(d);
    
    const auto v_val = s::Set(d, value);
    
    size_t i = 0;
    
    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        s::Store(s::Add(s::Load(d, src.ptr + i), v_val), d, dst.ptr + i);
        s::Store(s::Add(s::Load(d, src.ptr + i + lanes), v_val), d, dst.ptr + i + lanes);
        s::Store(s::Add(s::Load(d, src.ptr + i + 2*lanes), v_val), d, dst.ptr + i + 2*lanes);
        s::Store(s::Add(s::Load(d, src.ptr + i + 3*lanes), v_val), d, dst.ptr + i + 3*lanes);
    }
    
    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Add(s::Load(d, src.ptr + i), v_val), d, dst.ptr + i);
    }
    
    for (; i < N; ++i) {
        dst[i] = src[i] + value;
    }
}

template <typename T>
SCL_FORCE_INLINE void add(Array<const T> a, Array<const T> b, Array<T> dst) {
    SCL_ASSERT(a.len == b.len && b.len == dst.len, "add: Size mismatch");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = a.len;
    const size_t lanes = s::Lanes(d);
    
    size_t i = 0;
    
    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        s::Store(s::Add(s::Load(d, a.ptr + i), s::Load(d, b.ptr + i)), d, dst.ptr + i);
        s::Store(s::Add(s::Load(d, a.ptr + i + lanes), s::Load(d, b.ptr + i + lanes)), d, dst.ptr + i + lanes);
        s::Store(s::Add(s::Load(d, a.ptr + i + 2*lanes), s::Load(d, b.ptr + i + 2*lanes)), d, dst.ptr + i + 2*lanes);
        s::Store(s::Add(s::Load(d, a.ptr + i + 3*lanes), s::Load(d, b.ptr + i + 3*lanes)), d, dst.ptr + i + 3*lanes);
    }
    
    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Add(s::Load(d, a.ptr + i), s::Load(d, b.ptr + i)), d, dst.ptr + i);
    }
    
    for (; i < N; ++i) {
        dst[i] = a[i] + b[i];
    }
}

template <typename T>
SCL_FORCE_INLINE void sub(Array<const T> a, Array<const T> b, Array<T> dst) {
    SCL_ASSERT(a.len == b.len && b.len == dst.len, "sub: Size mismatch");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = a.len;
    const size_t lanes = s::Lanes(d);
    
    size_t i = 0;
    
    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        s::Store(s::Sub(s::Load(d, a.ptr + i), s::Load(d, b.ptr + i)), d, dst.ptr + i);
        s::Store(s::Sub(s::Load(d, a.ptr + i + lanes), s::Load(d, b.ptr + i + lanes)), d, dst.ptr + i + lanes);
        s::Store(s::Sub(s::Load(d, a.ptr + i + 2*lanes), s::Load(d, b.ptr + i + 2*lanes)), d, dst.ptr + i + 2*lanes);
        s::Store(s::Sub(s::Load(d, a.ptr + i + 3*lanes), s::Load(d, b.ptr + i + 3*lanes)), d, dst.ptr + i + 3*lanes);
    }
    
    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Sub(s::Load(d, a.ptr + i), s::Load(d, b.ptr + i)), d, dst.ptr + i);
    }
    
    for (; i < N; ++i) {
        dst[i] = a[i] - b[i];
    }
}

template <typename T>
SCL_FORCE_INLINE void mul(Array<const T> a, Array<const T> b, Array<T> dst) {
    SCL_ASSERT(a.len == b.len && b.len == dst.len, "mul: Size mismatch");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = a.len;
    const size_t lanes = s::Lanes(d);
    
    size_t i = 0;
    
    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        s::Store(s::Mul(s::Load(d, a.ptr + i), s::Load(d, b.ptr + i)), d, dst.ptr + i);
        s::Store(s::Mul(s::Load(d, a.ptr + i + lanes), s::Load(d, b.ptr + i + lanes)), d, dst.ptr + i + lanes);
        s::Store(s::Mul(s::Load(d, a.ptr + i + 2*lanes), s::Load(d, b.ptr + i + 2*lanes)), d, dst.ptr + i + 2*lanes);
        s::Store(s::Mul(s::Load(d, a.ptr + i + 3*lanes), s::Load(d, b.ptr + i + 3*lanes)), d, dst.ptr + i + 3*lanes);
    }
    
    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Mul(s::Load(d, a.ptr + i), s::Load(d, b.ptr + i)), d, dst.ptr + i);
    }
    
    for (; i < N; ++i) {
        dst[i] = a[i] * b[i];
    }
}

template <typename T>
SCL_FORCE_INLINE void div(Array<const T> a, Array<const T> b, Array<T> dst) {
    SCL_ASSERT(a.len == b.len && b.len == dst.len, "div: Size mismatch");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = a.len;
    const size_t lanes = s::Lanes(d);
    
    size_t i = 0;
    
    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        s::Store(s::Div(s::Load(d, a.ptr + i), s::Load(d, b.ptr + i)), d, dst.ptr + i);
        s::Store(s::Div(s::Load(d, a.ptr + i + lanes), s::Load(d, b.ptr + i + lanes)), d, dst.ptr + i + lanes);
    }
    
    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Div(s::Load(d, a.ptr + i), s::Load(d, b.ptr + i)), d, dst.ptr + i);
    }
    
    for (; i < N; ++i) {
        dst[i] = a[i] / b[i];
    }
}

// =============================================================================
// 5. Scatter/Gather Operations
// =============================================================================

template <typename T, typename IdxT>
SCL_FORCE_INLINE void gather(
    const T* SCL_RESTRICT src,
    Array<const IdxT> indices,
    Array<T> dst
) {
    SCL_ASSERT(indices.len == dst.len, "gather: Size mismatch");
    
    const size_t N = indices.len;
    
    for (size_t i = 0; i < N; ++i) {
        if (i + 8 < N) {
            SCL_PREFETCH_READ(&indices[i + 8], 0);
            SCL_PREFETCH_READ(src + indices[i + 4], 0);
        }
        dst[i] = src[indices[i]];
    }
}

template <typename T, typename IdxT>
SCL_FORCE_INLINE void scatter(
    Array<const T> src,
    Array<const IdxT> indices,
    T* SCL_RESTRICT dst
) {
    SCL_ASSERT(src.len == indices.len, "scatter: Size mismatch");
    
    const size_t N = src.len;
    
    for (size_t i = 0; i < N; ++i) {
        dst[indices[i]] = src[i];
    }
}

template <typename T, typename IdxT>
SCL_FORCE_INLINE void scatter_add(
    Array<const T> src,
    Array<const IdxT> indices,
    T* SCL_RESTRICT dst
) {
    SCL_ASSERT(src.len == indices.len, "scatter_add: Size mismatch");
    
    const size_t N = src.len;
    
    for (size_t i = 0; i < N; ++i) {
        dst[indices[i]] += src[i];
    }
}

// =============================================================================
// 6. Clamp Operations
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void clamp(Array<T> span, T min_val, T max_val) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    const auto v_min = s::Set(d, min_val);
    const auto v_max = s::Set(d, max_val);
    
    size_t i = 0;
    
    for (; i + lanes <= N; i += lanes) {
        auto v = s::Load(d, span.ptr + i);
        v = s::Max(v, v_min);
        v = s::Min(v, v_max);
        s::Store(v, d, span.ptr + i);
    }
    
    for (; i < N; ++i) {
        if (span[i] < min_val) span[i] = min_val;
        else if (span[i] > max_val) span[i] = max_val;
    }
}

template <typename T>
SCL_FORCE_INLINE void clamp_min(Array<T> span, T min_val) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    const auto v_min = s::Set(d, min_val);
    
    size_t i = 0;
    
    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        s::Store(s::Max(s::Load(d, span.ptr + i), v_min), d, span.ptr + i);
        s::Store(s::Max(s::Load(d, span.ptr + i + lanes), v_min), d, span.ptr + i + lanes);
    }
    
    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Max(s::Load(d, span.ptr + i), v_min), d, span.ptr + i);
    }
    
    for (; i < N; ++i) {
        if (span[i] < min_val) span[i] = min_val;
    }
}

template <typename T>
SCL_FORCE_INLINE void clamp_max(Array<T> span, T max_val) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    const auto v_max = s::Set(d, max_val);
    
    size_t i = 0;
    
    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        s::Store(s::Min(s::Load(d, span.ptr + i), v_max), d, span.ptr + i);
        s::Store(s::Min(s::Load(d, span.ptr + i + lanes), v_max), d, span.ptr + i + lanes);
    }
    
    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Min(s::Load(d, span.ptr + i), v_max), d, span.ptr + i);
    }
    
    for (; i < N; ++i) {
        if (span[i] > max_val) span[i] = max_val;
    }
}

// =============================================================================
// 7. Absolute Value Operations
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void abs_inplace(Array<T> span) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    size_t i = 0;
    
    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        s::Store(s::Abs(s::Load(d, span.ptr + i)), d, span.ptr + i);
        s::Store(s::Abs(s::Load(d, span.ptr + i + lanes)), d, span.ptr + i + lanes);
    }
    
    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Abs(s::Load(d, span.ptr + i)), d, span.ptr + i);
    }
    
    for (; i < N; ++i) {
        span[i] = (span[i] < T(0)) ? -span[i] : span[i];
    }
}

template <typename T>
SCL_FORCE_INLINE T sum_abs(Array<const T> span) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    if (N == 0) return T(0);
    
    auto acc0 = s::Zero(d);
    auto acc1 = s::Zero(d);
    
    size_t i = 0;
    
    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        acc0 = s::Add(acc0, s::Abs(s::Load(d, span.ptr + i)));
        acc1 = s::Add(acc1, s::Abs(s::Load(d, span.ptr + i + lanes)));
    }
    
    acc0 = s::Add(acc0, acc1);
    
    for (; i + lanes <= N; i += lanes) {
        acc0 = s::Add(acc0, s::Abs(s::Load(d, span.ptr + i)));
    }
    
    T result = s::GetLane(s::SumOfLanes(d, acc0));
    
    for (; i < N; ++i) {
        result += (span[i] < T(0)) ? -span[i] : span[i];
    }
    
    return result;
}

template <typename T>
SCL_FORCE_INLINE T sum_squared(Array<const T> span) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    if (N == 0) return T(0);
    
    auto acc0 = s::Zero(d);
    auto acc1 = s::Zero(d);
    auto acc2 = s::Zero(d);
    auto acc3 = s::Zero(d);
    
    size_t i = 0;
    
    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        auto v0 = s::Load(d, span.ptr + i);
        auto v1 = s::Load(d, span.ptr + i + lanes);
        auto v2 = s::Load(d, span.ptr + i + 2 * lanes);
        auto v3 = s::Load(d, span.ptr + i + 3 * lanes);
        
        acc0 = s::MulAdd(v0, v0, acc0);
        acc1 = s::MulAdd(v1, v1, acc1);
        acc2 = s::MulAdd(v2, v2, acc2);
        acc3 = s::MulAdd(v3, v3, acc3);
    }
    
    acc0 = s::Add(acc0, acc1);
    acc2 = s::Add(acc2, acc3);
    acc0 = s::Add(acc0, acc2);
    
    for (; i + lanes <= N; i += lanes) {
        auto v = s::Load(d, span.ptr + i);
        acc0 = s::MulAdd(v, v, acc0);
    }
    
    T result = s::GetLane(s::SumOfLanes(d, acc0));
    
    for (; i < N; ++i) {
        result += span[i] * span[i];
    }
    
    return result;
}

// =============================================================================
// 8. Fused Multiply-Add Operations
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void fma(
    Array<const T> a,
    Array<const T> b,
    Array<const T> c,
    Array<T> dst
) {
    SCL_ASSERT(a.len == b.len && b.len == c.len && c.len == dst.len, 
               "fma: Size mismatch");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = a.len;
    const size_t lanes = s::Lanes(d);
    
    size_t i = 0;
    
    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        auto va0 = s::Load(d, a.ptr + i);
        auto vb0 = s::Load(d, b.ptr + i);
        auto vc0 = s::Load(d, c.ptr + i);
        auto va1 = s::Load(d, a.ptr + i + lanes);
        auto vb1 = s::Load(d, b.ptr + i + lanes);
        auto vc1 = s::Load(d, c.ptr + i + lanes);
        
        s::Store(s::MulAdd(va0, vb0, vc0), d, dst.ptr + i);
        s::Store(s::MulAdd(va1, vb1, vc1), d, dst.ptr + i + lanes);
    }
    
    for (; i + lanes <= N; i += lanes) {
        auto va = s::Load(d, a.ptr + i);
        auto vb = s::Load(d, b.ptr + i);
        auto vc = s::Load(d, c.ptr + i);
        s::Store(s::MulAdd(va, vb, vc), d, dst.ptr + i);
    }
    
    for (; i < N; ++i) {
        dst[i] = a[i] * b[i] + c[i];
    }
}

template <typename T>
SCL_FORCE_INLINE void axpy(T alpha, Array<const T> x, Array<T> y) {
    SCL_ASSERT(x.len == y.len, "axpy: Size mismatch");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = x.len;
    const size_t lanes = s::Lanes(d);
    
    const auto v_alpha = s::Set(d, alpha);
    
    size_t i = 0;
    
    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        auto vy0 = s::Load(d, y.ptr + i);
        auto vy1 = s::Load(d, y.ptr + i + lanes);
        auto vy2 = s::Load(d, y.ptr + i + 2 * lanes);
        auto vy3 = s::Load(d, y.ptr + i + 3 * lanes);
        
        s::Store(s::MulAdd(v_alpha, s::Load(d, x.ptr + i), vy0), d, y.ptr + i);
        s::Store(s::MulAdd(v_alpha, s::Load(d, x.ptr + i + lanes), vy1), d, y.ptr + i + lanes);
        s::Store(s::MulAdd(v_alpha, s::Load(d, x.ptr + i + 2*lanes), vy2), d, y.ptr + i + 2*lanes);
        s::Store(s::MulAdd(v_alpha, s::Load(d, x.ptr + i + 3*lanes), vy3), d, y.ptr + i + 3*lanes);
    }
    
    for (; i + lanes <= N; i += lanes) {
        auto vy = s::Load(d, y.ptr + i);
        s::Store(s::MulAdd(v_alpha, s::Load(d, x.ptr + i), vy), d, y.ptr + i);
    }
    
    for (; i < N; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

// =============================================================================
// 9. Mathematical Functions (Extended)
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void sqrt(Array<const T> src, Array<T> dst) {
    SCL_ASSERT(src.len == dst.len, "sqrt: Size mismatch");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = src.len;
    const size_t lanes = s::Lanes(d);
    
    size_t i = 0;
    
    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        s::Store(s::Sqrt(s::Load(d, src.ptr + i)), d, dst.ptr + i);
        s::Store(s::Sqrt(s::Load(d, src.ptr + i + lanes)), d, dst.ptr + i + lanes);
    }
    
    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Sqrt(s::Load(d, src.ptr + i)), d, dst.ptr + i);
    }
    
    for (; i < N; ++i) {
        dst[i] = std::sqrt(src[i]);
    }
}

template <typename T>
SCL_FORCE_INLINE void sqrt_inplace(Array<T> span) {
    sqrt(Array<const T>(span.ptr, span.len), span);
}

template <typename T>
SCL_FORCE_INLINE void rsqrt(Array<const T> src, Array<T> dst) {
    SCL_ASSERT(src.len == dst.len, "rsqrt: Size mismatch");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = src.len;
    const size_t lanes = s::Lanes(d);
    
    const auto v_one = s::Set(d, T(1));
    
    size_t i = 0;
    
    for (; i + 2 * lanes <= N; i += 2 * lanes) {
        s::Store(s::Div(v_one, s::Sqrt(s::Load(d, src.ptr + i))), d, dst.ptr + i);
        s::Store(s::Div(v_one, s::Sqrt(s::Load(d, src.ptr + i + lanes))), d, dst.ptr + i + lanes);
    }
    
    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Div(v_one, s::Sqrt(s::Load(d, src.ptr + i))), d, dst.ptr + i);
    }
    
    for (; i < N; ++i) {
        dst[i] = T(1) / std::sqrt(src[i]);
    }
}

template <typename T>
SCL_FORCE_INLINE void square(Array<const T> src, Array<T> dst) {
    SCL_ASSERT(src.len == dst.len, "square: Size mismatch");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = src.len;
    const size_t lanes = s::Lanes(d);
    
    size_t i = 0;
    
    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        auto v0 = s::Load(d, src.ptr + i);
        auto v1 = s::Load(d, src.ptr + i + lanes);
        auto v2 = s::Load(d, src.ptr + i + 2*lanes);
        auto v3 = s::Load(d, src.ptr + i + 3*lanes);
        
        s::Store(s::Mul(v0, v0), d, dst.ptr + i);
        s::Store(s::Mul(v1, v1), d, dst.ptr + i + lanes);
        s::Store(s::Mul(v2, v2), d, dst.ptr + i + 2*lanes);
        s::Store(s::Mul(v3, v3), d, dst.ptr + i + 3*lanes);
    }
    
    for (; i + lanes <= N; i += lanes) {
        auto v = s::Load(d, src.ptr + i);
        s::Store(s::Mul(v, v), d, dst.ptr + i);
    }
    
    for (; i < N; ++i) {
        dst[i] = src[i] * src[i];
    }
}

template <typename T>
SCL_FORCE_INLINE void square_inplace(Array<T> span) {
    square(Array<const T>(span.ptr, span.len), span);
}

template <typename T>
SCL_FORCE_INLINE void negate(Array<const T> src, Array<T> dst) {
    SCL_ASSERT(src.len == dst.len, "negate: Size mismatch");
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = src.len;
    const size_t lanes = s::Lanes(d);
    
    const auto v_zero = s::Zero(d);
    
    size_t i = 0;
    
    for (; i + 4 * lanes <= N; i += 4 * lanes) {
        s::Store(s::Sub(v_zero, s::Load(d, src.ptr + i)), d, dst.ptr + i);
        s::Store(s::Sub(v_zero, s::Load(d, src.ptr + i + lanes)), d, dst.ptr + i + lanes);
        s::Store(s::Sub(v_zero, s::Load(d, src.ptr + i + 2*lanes)), d, dst.ptr + i + 2*lanes);
        s::Store(s::Sub(v_zero, s::Load(d, src.ptr + i + 3*lanes)), d, dst.ptr + i + 3*lanes);
    }
    
    for (; i + lanes <= N; i += lanes) {
        s::Store(s::Sub(v_zero, s::Load(d, src.ptr + i)), d, dst.ptr + i);
    }
    
    for (; i < N; ++i) {
        dst[i] = -src[i];
    }
}

template <typename T>
SCL_FORCE_INLINE void negate_inplace(Array<T> span) {
    negate(Array<const T>(span.ptr, span.len), span);
}

// =============================================================================
// 10. Comparison Operations
// =============================================================================

template <typename T>
SCL_FORCE_INLINE size_t count_nonzero(Array<const T> span) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    const auto v_zero = s::Zero(d);
    size_t cnt = 0;
    
    size_t i = 0;
    
    for (; i + lanes <= N; i += lanes) {
        auto v_data = s::Load(d, span.ptr + i);
        auto mask = s::Ne(v_data, v_zero);
        cnt += s::CountTrue(d, mask);
    }
    
    for (; i < N; ++i) {
        if (span[i] != T(0)) ++cnt;
    }
    
    return cnt;
}

template <typename T>
SCL_FORCE_INLINE bool all(Array<const T> span, T value) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    const auto v_val = s::Set(d, value);
    
    size_t i = 0;
    
    for (; i + lanes <= N; i += lanes) {
        auto v_data = s::Load(d, span.ptr + i);
        auto mask = s::Eq(v_data, v_val);
        if (!s::AllTrue(d, mask)) return false;
    }
    
    for (; i < N; ++i) {
        if (span[i] != value) return false;
    }
    
    return true;
}

template <typename T>
SCL_FORCE_INLINE bool any(Array<const T> span, T value) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t N = span.len;
    const size_t lanes = s::Lanes(d);
    
    const auto v_val = s::Set(d, value);
    
    size_t i = 0;
    
    for (; i + lanes <= N; i += lanes) {
        auto v_data = s::Load(d, span.ptr + i);
        auto mask = s::Eq(v_data, v_val);
        if (!s::AllFalse(d, mask)) return true;
    }
    
    for (; i < N; ++i) {
        if (span[i] == value) return true;
    }
    
    return false;
}

} // namespace scl::vectorize

