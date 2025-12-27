// =============================================================================
// FILE: scl/core/vectorize.h
// BRIEF: API reference for SIMD-optimized vectorized array operations
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>
#include <utility>

namespace scl::vectorize {

/* =============================================================================
 * MODULE: Vectorized Array Operations
 * =============================================================================
 * SUMMARY:
 *     High-performance SIMD-optimized operations on array views using Google Highway.
 *
 * DESIGN PURPOSE:
 *     Provides zero-overhead abstractions for common array operations with
 *     automatic SIMD vectorization. All operations use aggressive unrolling
 *     (2-4 way) and handle scalar tails automatically.
 *
 * PERFORMANCE:
 *     - 4-way unrolled SIMD loops for maximum ILP
 *     - Automatic scalar tail handling
 *     - Architecture-agnostic (AVX2, AVX-512, NEON, etc.)
 *     - Zero runtime overhead (all abstractions compile away)
 *
 * THREAD SAFETY:
 *     All operations are thread-safe (pure functions, no shared state).
 * -------------------------------------------------------------------------- */

// =============================================================================
// 1. Reduction Operations
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: sum
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute sum of all elements using SIMD-optimized reduction.
 *
 * PARAMETERS:
 *     span    [in] - Input array view
 *
 * RETURNS:
 *     Sum of all elements. Returns T(0) for empty arrays.
 *
 * ALGORITHM:
 *     1. 4-way unrolled SIMD accumulation
 *     2. Horizontal reduction using SumOfLanes
 *     3. Scalar tail handling
 *
 * COMPLEXITY:
 *     Time: O(N)
 *     Space: O(1)
 *
 * PERFORMANCE:
 *     Approximately 4-8x faster than naive loop on modern CPUs.
 * -------------------------------------------------------------------------- */
template <typename T>
T sum(Array<const T> span);

/* -----------------------------------------------------------------------------
 * FUNCTION: product
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute product of all elements.
 *
 * RETURNS:
 *     Product of all elements. Returns T(1) for empty arrays.
 *
 * PERFORMANCE:
 *     Uses 2-way unrolling for product accumulation.
 * -------------------------------------------------------------------------- */
template <typename T>
T product(Array<const T> span);

/* -----------------------------------------------------------------------------
 * FUNCTION: dot
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute dot product of two vectors.
 *
 * PARAMETERS:
 *     a    [in] - First vector
 *     b    [in] - Second vector
 *
 * PRECONDITIONS:
 *     a.len == b.len
 *
 * RETURNS:
 *     Dot product: sum(a[i] * b[i])
 *
 * PERFORMANCE:
 *     Uses MulAdd (FMA) for optimal performance.
 * -------------------------------------------------------------------------- */
template <typename T>
T dot(Array<const T> a, Array<const T> b);

// =============================================================================
// 2. Search Operations
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: find
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Find first occurrence of value.
 *
 * PARAMETERS:
 *     span   [in] - Input array
 *     value  [in] - Value to search for
 *
 * RETURNS:
 *     Index of first match, or span.len if not found.
 *
 * PERFORMANCE:
 *     Uses SIMD comparison with early exit on match.
 * -------------------------------------------------------------------------- */
template <typename T>
size_t find(Array<const T> span, T value);

/* -----------------------------------------------------------------------------
 * FUNCTION: count
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Count occurrences of value.
 *
 * RETURNS:
 *     Number of elements equal to value.
 *
 * PERFORMANCE:
 *     Uses CountTrue on comparison mask for efficient counting.
 * -------------------------------------------------------------------------- */
template <typename T>
size_t count(Array<const T> span, T value);

/* -----------------------------------------------------------------------------
 * FUNCTION: contains
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Check if array contains value.
 *
 * RETURNS:
 *     True if value found, false otherwise.
 * -------------------------------------------------------------------------- */
template <typename T>
bool contains(Array<const T> span, T value);

// =============================================================================
// 3. Min/Max Operations
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: min_element
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Find index of minimum element.
 *
 * PRECONDITIONS:
 *     span.len > 0
 *
 * RETURNS:
 *     Index of minimum element.
 *
 * PERFORMANCE:
 *     Uses MinOfLanes for horizontal reduction.
 * -------------------------------------------------------------------------- */
template <typename T>
size_t min_element(Array<const T> span);

/* -----------------------------------------------------------------------------
 * FUNCTION: max_element
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Find index of maximum element.
 *
 * PRECONDITIONS:
 *     span.len > 0
 *
 * RETURNS:
 *     Index of maximum element.
 * -------------------------------------------------------------------------- */
template <typename T>
size_t max_element(Array<const T> span);

/* -----------------------------------------------------------------------------
 * FUNCTION: minmax
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Find both minimum and maximum in single pass.
 *
 * PRECONDITIONS:
 *     span.len > 0
 *
 * RETURNS:
 *     Pair of (min_value, max_value).
 *
 * PERFORMANCE:
 *     More efficient than calling min_element and max_element separately.
 * -------------------------------------------------------------------------- */
template <typename T>
std::pair<T, T> minmax(Array<const T> span);

// =============================================================================
// 4. Transform Operations
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: transform_inplace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply unary operation in-place.
 *
 * PARAMETERS:
 *     span  [inout] - Array to transform
 *     op    [in]    - Unary operation: T -> T
 * -------------------------------------------------------------------------- */
template <typename T, typename UnaryOp>
void transform_inplace(Array<T> span, UnaryOp op);

/* -----------------------------------------------------------------------------
 * FUNCTION: transform (unary)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply unary operation: dst[i] = op(src[i]).
 *
 * PARAMETERS:
 *     src  [in]  - Source array
 *     dst  [out] - Destination array
 *     op   [in]  - Unary operation
 *
 * PRECONDITIONS:
 *     src.len == dst.len
 * -------------------------------------------------------------------------- */
template <typename T, typename U, typename UnaryOp>
void transform(Array<const T> src, Array<U> dst, UnaryOp op);

/* -----------------------------------------------------------------------------
 * FUNCTION: transform (binary)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply binary operation: dst[i] = op(a[i], b[i]).
 *
 * PRECONDITIONS:
 *     a.len == b.len == dst.len
 * -------------------------------------------------------------------------- */
template <typename T, typename U, typename V, typename BinaryOp>
void transform(Array<const T> a, Array<const U> b, Array<V> dst, BinaryOp op);

/* -----------------------------------------------------------------------------
 * FUNCTION: scale
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Scale all elements: dst[i] = src[i] * scale_factor.
 *
 * PERFORMANCE:
 *     Uses 4-way unrolled SIMD multiplication.
 * -------------------------------------------------------------------------- */
template <typename T>
void scale(Array<const T> src, Array<T> dst, T scale_factor);

/* -----------------------------------------------------------------------------
 * FUNCTION: scale_inplace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Scale in-place: span[i] *= scale_factor.
 * -------------------------------------------------------------------------- */
template <typename T>
void scale_inplace(Array<T> span, T scale_factor);

/* -----------------------------------------------------------------------------
 * FUNCTION: add_scalar
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Add scalar to all elements: dst[i] = src[i] + value.
 * -------------------------------------------------------------------------- */
template <typename T>
void add_scalar(Array<const T> src, Array<T> dst, T value);

/* -----------------------------------------------------------------------------
 * FUNCTION: add
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Element-wise addition: dst[i] = a[i] + b[i].
 * -------------------------------------------------------------------------- */
template <typename T>
void add(Array<const T> a, Array<const T> b, Array<T> dst);

/* -----------------------------------------------------------------------------
 * FUNCTION: sub
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Element-wise subtraction: dst[i] = a[i] - b[i].
 * -------------------------------------------------------------------------- */
template <typename T>
void sub(Array<const T> a, Array<const T> b, Array<T> dst);

/* -----------------------------------------------------------------------------
 * FUNCTION: mul
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Element-wise multiplication: dst[i] = a[i] * b[i].
 * -------------------------------------------------------------------------- */
template <typename T>
void mul(Array<const T> a, Array<const T> b, Array<T> dst);

/* -----------------------------------------------------------------------------
 * FUNCTION: div
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Element-wise division: dst[i] = a[i] / b[i].
 * -------------------------------------------------------------------------- */
template <typename T>
void div(Array<const T> a, Array<const T> b, Array<T> dst);

// =============================================================================
// 5. Scatter/Gather Operations
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: gather
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Gather elements: dst[i] = src[indices[i]].
 *
 * PARAMETERS:
 *     src      [in]  - Source array
 *     indices  [in]  - Index array
 *     dst      [out] - Destination array
 *
 * PRECONDITIONS:
 *     indices.len == dst.len
 *
 * PERFORMANCE:
 *     Uses prefetching for better cache performance.
 * -------------------------------------------------------------------------- */
template <typename T, typename IdxT>
void gather(const T* src, Array<const IdxT> indices, Array<T> dst);

/* -----------------------------------------------------------------------------
 * FUNCTION: scatter
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Scatter elements: dst[indices[i]] = src[i].
 *
 * WARNING:
 *     No checking for duplicate indices (last write wins).
 * -------------------------------------------------------------------------- */
template <typename T, typename IdxT>
void scatter(Array<const T> src, Array<const IdxT> indices, T* dst);

/* -----------------------------------------------------------------------------
 * FUNCTION: scatter_add
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Scatter-add: dst[indices[i]] += src[i].
 *
 * USE CASE:
 *     Histogram building, sparse accumulation.
 * -------------------------------------------------------------------------- */
template <typename T, typename IdxT>
void scatter_add(Array<const T> src, Array<const IdxT> indices, T* dst);

// =============================================================================
// 6. Clamp Operations
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: clamp
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Clamp all elements to [min_val, max_val] range.
 * -------------------------------------------------------------------------- */
template <typename T>
void clamp(Array<T> span, T min_val, T max_val);

/* -----------------------------------------------------------------------------
 * FUNCTION: clamp_min
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Clamp elements to minimum value (floor).
 * -------------------------------------------------------------------------- */
template <typename T>
void clamp_min(Array<T> span, T min_val);

/* -----------------------------------------------------------------------------
 * FUNCTION: clamp_max
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Clamp elements to maximum value (ceiling).
 * -------------------------------------------------------------------------- */
template <typename T>
void clamp_max(Array<T> span, T max_val);

// =============================================================================
// 7. Absolute Value Operations
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: abs_inplace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute absolute value in-place.
 * -------------------------------------------------------------------------- */
template <typename T>
void abs_inplace(Array<T> span);

/* -----------------------------------------------------------------------------
 * FUNCTION: sum_abs
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sum of absolute values (L1 norm).
 *
 * RETURNS:
 *     sum(|span[i]|)
 * -------------------------------------------------------------------------- */
template <typename T>
T sum_abs(Array<const T> span);

/* -----------------------------------------------------------------------------
 * FUNCTION: sum_squared
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sum of squared values (squared L2 norm).
 *
 * RETURNS:
 *     sum(span[i]^2)
 * -------------------------------------------------------------------------- */
template <typename T>
T sum_squared(Array<const T> span);

// =============================================================================
// 8. Fused Multiply-Add Operations
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: fma
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Fused multiply-add: dst[i] = a[i] * b[i] + c[i].
 *
 * PERFORMANCE:
 *     Uses hardware FMA when available for better accuracy and performance.
 * -------------------------------------------------------------------------- */
template <typename T>
void fma(Array<const T> a, Array<const T> b, Array<const T> c, Array<T> dst);

/* -----------------------------------------------------------------------------
 * FUNCTION: axpy
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Scaled add (AXPY): y[i] = alpha * x[i] + y[i].
 *
 * USE CASE:
 *     Classic BLAS-style operation, in-place modification of y.
 * -------------------------------------------------------------------------- */
template <typename T>
void axpy(T alpha, Array<const T> x, Array<T> y);

// =============================================================================
// 9. Mathematical Functions (Extended)
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: sqrt
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Element-wise square root: dst[i] = sqrt(src[i]).
 * -------------------------------------------------------------------------- */
template <typename T>
void sqrt(Array<const T> src, Array<T> dst);

/* -----------------------------------------------------------------------------
 * FUNCTION: sqrt_inplace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Square root in-place.
 * -------------------------------------------------------------------------- */
template <typename T>
void sqrt_inplace(Array<T> span);

/* -----------------------------------------------------------------------------
 * FUNCTION: rsqrt
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Reciprocal square root: dst[i] = 1 / sqrt(src[i]).
 *
 * PERFORMANCE:
 *     Faster than 1.0 / sqrt() on some architectures.
 * -------------------------------------------------------------------------- */
template <typename T>
void rsqrt(Array<const T> src, Array<T> dst);

/* -----------------------------------------------------------------------------
 * FUNCTION: square
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Element-wise square: dst[i] = src[i]^2.
 * -------------------------------------------------------------------------- */
template <typename T>
void square(Array<const T> src, Array<T> dst);

/* -----------------------------------------------------------------------------
 * FUNCTION: square_inplace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Square in-place.
 * -------------------------------------------------------------------------- */
template <typename T>
void square_inplace(Array<T> span);

/* -----------------------------------------------------------------------------
 * FUNCTION: negate
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Element-wise negation: dst[i] = -src[i].
 * -------------------------------------------------------------------------- */
template <typename T>
void negate(Array<const T> src, Array<T> dst);

/* -----------------------------------------------------------------------------
 * FUNCTION: negate_inplace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Negation in-place.
 * -------------------------------------------------------------------------- */
template <typename T>
void negate_inplace(Array<T> span);

// =============================================================================
// 10. Comparison Operations
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: count_nonzero
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Count non-zero elements.
 *
 * RETURNS:
 *     Number of elements != T(0).
 * -------------------------------------------------------------------------- */
template <typename T>
size_t count_nonzero(Array<const T> span);

/* -----------------------------------------------------------------------------
 * FUNCTION: all
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Check if all elements equal value.
 *
 * RETURNS:
 *     True if span[i] == value for all i.
 * -------------------------------------------------------------------------- */
template <typename T>
bool all(Array<const T> span, T value);

/* -----------------------------------------------------------------------------
 * FUNCTION: any
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Check if any element equals value.
 *
 * RETURNS:
 *     True if span[i] == value for any i.
 * -------------------------------------------------------------------------- */
template <typename T>
bool any(Array<const T> span, T value);

} // namespace scl::vectorize

