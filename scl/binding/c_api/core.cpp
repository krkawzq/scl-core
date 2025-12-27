// =============================================================================
/// @file core.cpp
/// @brief C ABI Wrapper for SCL Core Utilities
///
/// Provides C-compatible exports for:
/// - Memory operations: fill, zero, copy
/// - Array utilities: iota, reverse, unique
/// - Math utilities: sum, mean, variance, min, max
///
/// All functions follow the standard error protocol:
/// - Return 0 on success
/// - Return -1 on failure (error message in thread-local buffer)
// =============================================================================

#include "scl/binding/c_api/scl_c_api.h"
#include "scl/binding/c_api/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/argsort.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>

extern "C" {

// =============================================================================
// SECTION 1: Memory Operations
// =============================================================================

/// @brief Fill array with value (SIMD optimized)
///
/// @param data Array to fill [modified in-place]
/// @param size Number of elements
/// @param value Fill value
/// @return 0 on success, -1 on failure
int scl_fill_real(scl_real_t* data, scl_size_t size, scl_real_t value) {
    SCL_C_API_WRAPPER(
        if (!data) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<scl::Real> arr(data, static_cast<scl::Size>(size));
        scl::memory::fill(arr, value);
    );
}

/// @brief Fill index array with value
///
/// @param data Array to fill [modified in-place]
/// @param size Number of elements
/// @param value Fill value
/// @return 0 on success, -1 on failure
int scl_fill_index(scl_index_t* data, scl_size_t size, scl_index_t value) {
    SCL_C_API_WRAPPER(
        if (!data) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<scl::Index> arr(data, static_cast<scl::Size>(size));
        scl::memory::fill(arr, value);
    );
}

/// @brief Zero out array (memset optimized)
///
/// @param data Array to zero [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_zero_real(scl_real_t* data, scl_size_t size) {
    SCL_C_API_WRAPPER(
        if (!data) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<scl::Real> arr(data, static_cast<scl::Size>(size));
        scl::memory::zero(arr);
    );
}

/// @brief Zero out index array
///
/// @param data Array to zero [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_zero_index(scl_index_t* data, scl_size_t size) {
    SCL_C_API_WRAPPER(
        if (!data) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<scl::Index> arr(data, static_cast<scl::Size>(size));
        scl::memory::zero(arr);
    );
}

/// @brief Fast copy (assumes no overlap)
///
/// @param src Source array
/// @param dst Destination array [modified]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_copy_fast_real(
    const scl_real_t* src,
    scl_real_t* dst,
    scl_size_t size
) {
    SCL_C_API_WRAPPER(
        if (!src || !dst) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<const scl::Real> src_arr(src, static_cast<scl::Size>(size));
        scl::Array<scl::Real> dst_arr(dst, static_cast<scl::Size>(size));
        scl::memory::copy_fast(src_arr, dst_arr);
    );
}

/// @brief Safe copy (handles overlap)
///
/// @param src Source array
/// @param dst Destination array [modified]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_copy_safe_real(
    const scl_real_t* src,
    scl_real_t* dst,
    scl_size_t size
) {
    SCL_C_API_WRAPPER(
        if (!src || !dst) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<const scl::Real> src_arr(src, static_cast<scl::Size>(size));
        scl::Array<scl::Real> dst_arr(dst, static_cast<scl::Size>(size));
        scl::memory::copy(src_arr, dst_arr);
    );
}

/// @brief Stream copy (cache-bypassing, for large arrays)
///
/// @param src Source array
/// @param dst Destination array [modified]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_stream_copy_real(
    const scl_real_t* src,
    scl_real_t* dst,
    scl_size_t size
) {
    SCL_C_API_WRAPPER(
        if (!src || !dst) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<const scl::Real> src_arr(src, static_cast<scl::Size>(size));
        scl::Array<scl::Real> dst_arr(dst, static_cast<scl::Size>(size));
        scl::memory::stream_copy(src_arr, dst_arr);
    );
}

// =============================================================================
// SECTION 2: Array Utilities
// =============================================================================

/// @brief Fill array with sequence [0, 1, 2, ..., N-1]
///
/// @param data Output array [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_iota_index(scl_index_t* data, scl_size_t size) {
    SCL_C_API_WRAPPER(
        if (!data) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<scl::Index> arr(data, static_cast<scl::Size>(size));
        scl::sort::detail::iota_serial(arr);
    );
}

/// @brief Reverse array in-place
///
/// @param data Array to reverse [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_reverse_real(scl_real_t* data, scl_size_t size) {
    SCL_C_API_WRAPPER(
        if (!data) {
            throw scl::ValueError("Null pointer argument");
        }

        std::reverse(data, data + size);
    );
}

/// @brief Reverse index array in-place
///
/// @param data Array to reverse [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_reverse_index(scl_index_t* data, scl_size_t size) {
    SCL_C_API_WRAPPER(
        if (!data) {
            throw scl::ValueError("Null pointer argument");
        }

        std::reverse(data, data + size);
    );
}

/// @brief Remove consecutive duplicates (requires sorted input)
///
/// @param data Array [modified in-place]
/// @param size Number of elements
/// @param out_new_size Output new size after deduplication
/// @return 0 on success, -1 on failure
int scl_unique_real(
    scl_real_t* data,
    scl_size_t size,
    scl_size_t* out_new_size
) {
    SCL_C_API_WRAPPER(
        if (!data || !out_new_size) {
            throw scl::ValueError("Null pointer argument");
        }

        auto* end = std::unique(data, data + size);
        *out_new_size = static_cast<scl_size_t>(end - data);
    );
}

// =============================================================================
// SECTION 3: Math Utilities (Reductions)
// =============================================================================

/// @brief Sum of array elements (SIMD optimized)
///
/// @param data Input array
/// @param size Number of elements
/// @param out_sum Output sum
/// @return 0 on success, -1 on failure
int scl_sum_real(
    const scl_real_t* data,
    scl_size_t size,
    scl_real_t* out_sum
) {
    SCL_C_API_WRAPPER(
        if (!data || !out_sum) {
            throw scl::ValueError("Null pointer argument");
        }

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

        auto v_sum = s::Zero(d);
        size_t i = 0;

        for (; i + lanes <= size; i += lanes) {
            auto v = s::Load(d, data + i);
            v_sum = s::Add(v_sum, v);
        }

        scl_real_t sum = s::GetLane(s::SumOfLanes(d, v_sum));

        for (; i < size; ++i) {
            sum += data[i];
        }

        *out_sum = sum;
    );
}

/// @brief Mean of array elements
///
/// @param data Input array
/// @param size Number of elements
/// @param out_mean Output mean
/// @return 0 on success, -1 on failure
int scl_mean_real(
    const scl_real_t* data,
    scl_size_t size,
    scl_real_t* out_mean
) {
    SCL_C_API_WRAPPER(
        if (!data || !out_mean) {
            throw scl::ValueError("Null pointer argument");
        }

        if (size == 0) {
            *out_mean = 0.0;
            return 0;
        }

        scl_real_t sum;
        int status = scl_sum_real(data, size, &sum);
        if (status != 0) return status;

        *out_mean = sum / static_cast<scl_real_t>(size);
    );
}

/// @brief Variance of array elements
///
/// @param data Input array
/// @param size Number of elements
/// @param mean Mean value (if known, otherwise pass NaN to compute)
/// @param ddof Delta degrees of freedom (default 0)
/// @param out_var Output variance
/// @return 0 on success, -1 on failure
int scl_variance_real(
    const scl_real_t* data,
    scl_size_t size,
    scl_real_t mean,
    int ddof,
    scl_real_t* out_var
) {
    SCL_C_API_WRAPPER(
        if (!data || !out_var) {
            throw scl::ValueError("Null pointer argument");
        }

        if (size <= static_cast<scl_size_t>(ddof)) {
            *out_var = 0.0;
            return 0;
        }

        // Compute mean if not provided
        if (std::isnan(mean)) {
            int status = scl_mean_real(data, size, &mean);
            if (status != 0) return status;
        }

        // Compute sum of squared deviations (SIMD)
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

        const auto v_mean = s::Set(d, mean);
        auto v_sum_sq = s::Zero(d);

        size_t i = 0;
        for (; i + lanes <= size; i += lanes) {
            auto v = s::Load(d, data + i);
            auto diff = s::Sub(v, v_mean);
            v_sum_sq = s::MulAdd(diff, diff, v_sum_sq);
        }

        scl_real_t sum_sq = s::GetLane(s::SumOfLanes(d, v_sum_sq));

        for (; i < size; ++i) {
            scl_real_t diff = data[i] - mean;
            sum_sq += diff * diff;
        }

        *out_var = sum_sq / static_cast<scl_real_t>(size - ddof);
    );
}

/// @brief Minimum value in array
///
/// @param data Input array
/// @param size Number of elements
/// @param out_min Output minimum
/// @return 0 on success, -1 on failure
int scl_min_real(
    const scl_real_t* data,
    scl_size_t size,
    scl_real_t* out_min
) {
    SCL_C_API_WRAPPER(
        if (!data || !out_min) {
            throw scl::ValueError("Null pointer argument");
        }

        if (size == 0) {
            throw scl::ValueError("Cannot find min of empty array");
        }

        *out_min = *std::min_element(data, data + size);
    );
}

/// @brief Maximum value in array
///
/// @param data Input array
/// @param size Number of elements
/// @param out_max Output maximum
/// @return 0 on success, -1 on failure
int scl_max_real(
    const scl_real_t* data,
    scl_size_t size,
    scl_real_t* out_max
) {
    SCL_C_API_WRAPPER(
        if (!data || !out_max) {
            throw scl::ValueError("Null pointer argument");
        }

        if (size == 0) {
            throw scl::ValueError("Cannot find max of empty array");
        }

        *out_max = *std::max_element(data, data + size);
    );
}

/// @brief Min and max in single pass
///
/// @param data Input array
/// @param size Number of elements
/// @param out_min Output minimum
/// @param out_max Output maximum
/// @return 0 on success, -1 on failure
int scl_minmax_real(
    const scl_real_t* data,
    scl_size_t size,
    scl_real_t* out_min,
    scl_real_t* out_max
) {
    SCL_C_API_WRAPPER(
        if (!data || !out_min || !out_max) {
            throw scl::ValueError("Null pointer argument");
        }

        if (size == 0) {
            throw scl::ValueError("Cannot find minmax of empty array");
        }

        auto [min_it, max_it] = std::minmax_element(data, data + size);
        *out_min = *min_it;
        *out_max = *max_it;
    );
}

/// @brief Dot product of two arrays (SIMD optimized)
///
/// @param x First array
/// @param y Second array
/// @param size Number of elements
/// @param out_dot Output dot product
/// @return 0 on success, -1 on failure
int scl_dot_real(
    const scl_real_t* x,
    const scl_real_t* y,
    scl_size_t size,
    scl_real_t* out_dot
) {
    SCL_C_API_WRAPPER(
        if (!x || !y || !out_dot) {
            throw scl::ValueError("Null pointer argument");
        }

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

        auto v_sum = s::Zero(d);
        size_t i = 0;

        // 4-way unrolled
        for (; i + 4 * lanes <= size; i += 4 * lanes) {
            auto vx0 = s::Load(d, x + i + 0 * lanes);
            auto vy0 = s::Load(d, y + i + 0 * lanes);
            auto vx1 = s::Load(d, x + i + 1 * lanes);
            auto vy1 = s::Load(d, y + i + 1 * lanes);
            auto vx2 = s::Load(d, x + i + 2 * lanes);
            auto vy2 = s::Load(d, y + i + 2 * lanes);
            auto vx3 = s::Load(d, x + i + 3 * lanes);
            auto vy3 = s::Load(d, y + i + 3 * lanes);

            v_sum = s::MulAdd(vx0, vy0, v_sum);
            v_sum = s::MulAdd(vx1, vy1, v_sum);
            v_sum = s::MulAdd(vx2, vy2, v_sum);
            v_sum = s::MulAdd(vx3, vy3, v_sum);
        }

        for (; i + lanes <= size; i += lanes) {
            auto vx = s::Load(d, x + i);
            auto vy = s::Load(d, y + i);
            v_sum = s::MulAdd(vx, vy, v_sum);
        }

        scl_real_t dot = s::GetLane(s::SumOfLanes(d, v_sum));

        for (; i < size; ++i) {
            dot += x[i] * y[i];
        }

        *out_dot = dot;
    );
}

/// @brief Euclidean norm (L2 norm)
///
/// @param data Input array
/// @param size Number of elements
/// @param out_norm Output norm
/// @return 0 on success, -1 on failure
int scl_norm_real(
    const scl_real_t* data,
    scl_size_t size,
    scl_real_t* out_norm
) {
    SCL_C_API_WRAPPER(
        if (!data || !out_norm) {
            throw scl::ValueError("Null pointer argument");
        }

        scl_real_t dot;
        int status = scl_dot_real(data, data, size, &dot);
        if (status != 0) return status;

        *out_norm = std::sqrt(dot);
    );
}

// =============================================================================
// SECTION 4: Element-wise Operations
// =============================================================================

/// @brief Add scalar to array (y = x + scalar)
///
/// @param x Input array
/// @param size Number of elements
/// @param scalar Scalar to add
/// @param y Output array [pre-allocated]
/// @return 0 on success, -1 on failure
int scl_add_scalar_real(
    const scl_real_t* x,
    scl_size_t size,
    scl_real_t scalar,
    scl_real_t* y
) {
    SCL_C_API_WRAPPER(
        if (!x || !y) {
            throw scl::ValueError("Null pointer argument");
        }

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        const auto v_scalar = s::Set(d, scalar);

        size_t i = 0;
        for (; i + lanes <= size; i += lanes) {
            auto v = s::Load(d, x + i);
            s::Store(s::Add(v, v_scalar), d, y + i);
        }

        for (; i < size; ++i) {
            y[i] = x[i] + scalar;
        }
    );
}

/// @brief Multiply array by scalar (y = x * scalar)
///
/// @param x Input array
/// @param size Number of elements
/// @param scalar Scalar to multiply
/// @param y Output array [pre-allocated]
/// @return 0 on success, -1 on failure
int scl_mul_scalar_real(
    const scl_real_t* x,
    scl_size_t size,
    scl_real_t scalar,
    scl_real_t* y
) {
    SCL_C_API_WRAPPER(
        if (!x || !y) {
            throw scl::ValueError("Null pointer argument");
        }

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        const auto v_scalar = s::Set(d, scalar);

        size_t i = 0;
        for (; i + lanes <= size; i += lanes) {
            auto v = s::Load(d, x + i);
            s::Store(s::Mul(v, v_scalar), d, y + i);
        }

        for (; i < size; ++i) {
            y[i] = x[i] * scalar;
        }
    );
}

/// @brief Element-wise addition (z = x + y)
///
/// @param x First array
/// @param y Second array
/// @param size Number of elements
/// @param z Output array [pre-allocated]
/// @return 0 on success, -1 on failure
int scl_add_arrays_real(
    const scl_real_t* x,
    const scl_real_t* y,
    scl_size_t size,
    scl_real_t* z
) {
    SCL_C_API_WRAPPER(
        if (!x || !y || !z) {
            throw scl::ValueError("Null pointer argument");
        }

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

        size_t i = 0;
        for (; i + lanes <= size; i += lanes) {
            auto vx = s::Load(d, x + i);
            auto vy = s::Load(d, y + i);
            s::Store(s::Add(vx, vy), d, z + i);
        }

        for (; i < size; ++i) {
            z[i] = x[i] + y[i];
        }
    );
}

/// @brief Element-wise multiplication (z = x * y)
///
/// @param x First array
/// @param y Second array
/// @param size Number of elements
/// @param z Output array [pre-allocated]
/// @return 0 on success, -1 on failure
int scl_mul_arrays_real(
    const scl_real_t* x,
    const scl_real_t* y,
    scl_size_t size,
    scl_real_t* z
) {
    SCL_C_API_WRAPPER(
        if (!x || !y || !z) {
            throw scl::ValueError("Null pointer argument");
        }

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

        size_t i = 0;
        for (; i + lanes <= size; i += lanes) {
            auto vx = s::Load(d, x + i);
            auto vy = s::Load(d, y + i);
            s::Store(s::Mul(vx, vy), d, z + i);
        }

        for (; i < size; ++i) {
            z[i] = x[i] * y[i];
        }
    );
}

/// @brief Clip array to range [min_val, max_val]
///
/// @param data Array [modified in-place]
/// @param size Number of elements
/// @param min_val Minimum value
/// @param max_val Maximum value
/// @return 0 on success, -1 on failure
int scl_clip_real(
    scl_real_t* data,
    scl_size_t size,
    scl_real_t min_val,
    scl_real_t max_val
) {
    SCL_C_API_WRAPPER(
        if (!data) {
            throw scl::ValueError("Null pointer argument");
        }

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

        const auto v_min = s::Set(d, min_val);
        const auto v_max = s::Set(d, max_val);

        size_t i = 0;
        for (; i + lanes <= size; i += lanes) {
            auto v = s::Load(d, data + i);
            v = s::Min(s::Max(v, v_min), v_max);
            s::Store(v, d, data + i);
        }

        for (; i < size; ++i) {
            if (data[i] < min_val) data[i] = min_val;
            if (data[i] > max_val) data[i] = max_val;
        }
    );
}

} // extern "C"

