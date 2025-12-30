#pragma once

/// @file scl/math/regression.hpp
/// @brief High-performance polynomial regression and LOESS smoothing
///
/// This header provides:
///   - Polynomial regression with SIMD optimization
///   - LOESS (Locally Weighted Scatterplot Smoothing)
///   - Compile-time degree optimization
///   - Parallel evaluation
///
/// @note Currently optimized for degree-2 polynomials
/// @note LOESS requires sorted x values

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macro.hpp"
#include "scl/core/simd.hpp"

#include <cmath>
#include <array>
#include <algorithm>
#include <span>

namespace scl::math {

// =============================================================================
// SECTION 1: Internal Helpers - Tiny Linear Algebra
// =============================================================================

namespace detail {

/// @brief Solve symmetric 3x3 system Ax = b using analytical inverse
/// @param[in] A_sym Symmetric matrix in packed format [a00,a01,a02,a11,a12,a22]
/// @param[in] b Right-hand side vector [b0,b1,b2]
/// @param[out] x Solution vector [x0,x1,x2]
/// @note Uses analytical formula with regularization for numerical stability
SCL_FORCE_INLINE
auto solve_sym_3x3_static(
    const Real* A_sym,
    const Real* b,
    Real* x
) -> void {
    Real a00 = A_sym[0], a01 = A_sym[1], a02 = A_sym[2];
    Real a11 = A_sym[3], a12 = A_sym[4];
    Real a22 = A_sym[5];

    Real det = a00 * (a11 * a22 - a12 * a12) -
               a01 * (a01 * a22 - a12 * a02) +
               a02 * (a01 * a12 - a11 * a02);

    // Regularization
    if (std::abs(det) < Real(1e-12)) [[unlikely]] det = Real(1e-12);
    auto inv_det = Real(1.0) / det;

    x[0] = inv_det * (
        (a11 * a22 - a12 * a12) * b[0] +
        (a02 * a12 - a01 * a22) * b[1] +
        (a01 * a12 - a02 * a11) * b[2]
    );

    x[1] = inv_det * (
        (a12 * a02 - a01 * a22) * b[0] +
        (a00 * a22 - a02 * a02) * b[1] +
        (a01 * a02 - a00 * a12) * b[2]
    );

    x[2] = inv_det * (
        (a01 * a12 - a02 * a11) * b[0] +
        (a02 * a01 - a00 * a12) * b[1] +
        (a00 * a11 - a01 * a01) * b[2]
    );
}

/// @brief Evaluate polynomial using Horner's method
/// @tparam N Polynomial degree + 1
/// @param[in] coeffs Polynomial coefficients [c0, c1, ..., c_{N-1}]
/// @param[in] x Evaluation point
/// @return p(x) = c0 + c1*x + c2*x² + ... + c_{N-1}*x^{N-1}
template<int N>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto poly_eval(const Real* coeffs, Real x) -> Real {
    Real res = coeffs[N - 1];
    for (int i = N - 2; i >= 0; --i) {
        res = res * x + coeffs[i];
    }
    return res;
}

/// @brief Tricube weight function for LOESS
/// @param[in] dist Normalized distance (0 to 1)
/// @return Tricube weight: (1 - |dist|³)³ for |dist| < 1, else 0
[[nodiscard]]
SCL_FORCE_INLINE
auto tricube_weight(Real dist) -> Real {
    Real a = std::abs(dist);
    if (a >= Real(1.0)) [[unlikely]] return Real(0.0);
    Real tmp = Real(1.0) - a * a * a;
    return tmp * tmp * tmp;
}

} // namespace detail

// =============================================================================
// SECTION 2: Polynomial Regression (SIMD Optimized)
// =============================================================================

/// @brief Accumulate weighted normal equation terms for degree-2 polynomial (SIMD)
/// @param[in] x X values
/// @param[in] y Y values
/// @param[in] w Weights (empty span for uniform weights)
/// @param[out] sums Output array [s0,s1,s2,s3,s4,sy0,sy1,sy2]
/// @note Computes sums for normal equations: X'WX * coeff = X'Wy
SCL_FORCE_INLINE
auto accumulate_matrices_deg2_simd(
    std::span<const Real> x,
    std::span<const Real> y,
    std::span<const Real> w,
    Real* sums
) -> void {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size N = x.size();
    const Size lanes = s::Lanes(d);

    auto v_s0 = s::Zero(d);
    auto v_s1 = s::Zero(d);
    auto v_s2 = s::Zero(d);
    auto v_s3 = s::Zero(d);
    auto v_s4 = s::Zero(d);

    auto v_sy0 = s::Zero(d);
    auto v_sy1 = s::Zero(d);
    auto v_sy2 = s::Zero(d);

    const bool has_weights = !w.empty();
    Size i = 0;

    // Vectorized main loop
    for (; i + lanes <= N; i += lanes) {
        auto vx = s::Load(d, x.data() + i);
        auto vy = s::Load(d, y.data() + i);
        auto vw = has_weights ? s::Load(d, w.data() + i) : s::Set(d, Real(1.0));

        auto vx2 = s::Mul(vx, vx);
        auto vx3 = s::Mul(vx2, vx);
        auto vx4 = s::Mul(vx2, vx2);

        v_s0 = s::Add(v_s0, vw);
        v_s1 = s::MulAdd(vx, vw, v_s1);
        v_s2 = s::MulAdd(vx2, vw, v_s2);
        v_s3 = s::MulAdd(vx3, vw, v_s3);
        v_s4 = s::MulAdd(vx4, vw, v_s4);

        auto v_yw = s::Mul(vy, vw);
        v_sy0 = s::Add(v_sy0, v_yw);
        v_sy1 = s::MulAdd(vx, v_yw, v_sy1);
        v_sy2 = s::MulAdd(vx2, v_yw, v_sy2);
    }

    // Horizontal reduction
    sums[0] = s::GetLane(s::SumOfLanes(d, v_s0));
    sums[1] = s::GetLane(s::SumOfLanes(d, v_s1));
    sums[2] = s::GetLane(s::SumOfLanes(d, v_s2));
    sums[3] = s::GetLane(s::SumOfLanes(d, v_s3));
    sums[4] = s::GetLane(s::SumOfLanes(d, v_s4));

    sums[5] = s::GetLane(s::SumOfLanes(d, v_sy0));
    sums[6] = s::GetLane(s::SumOfLanes(d, v_sy1));
    sums[7] = s::GetLane(s::SumOfLanes(d, v_sy2));

    // Scalar tail
    for (; i < N; ++i) {
        Real xi = x[i];
        Real yi = y[i];
        Real wi = has_weights ? w[i] : Real(1.0);

        Real xi2 = xi * xi;
        Real xi3 = xi2 * xi;
        Real xi4 = xi2 * xi2;
        Real ywi = yi * wi;

        sums[0] += wi;
        sums[1] += xi * wi;
        sums[2] += xi2 * wi;
        sums[3] += xi3 * wi;
        sums[4] += xi4 * wi;

        sums[5] += ywi;
        sums[6] += xi * ywi;
        sums[7] += xi2 * ywi;
    }
}

/// @brief Polynomial regression with compile-time degree optimization
/// @tparam DEGREE Polynomial degree (currently only 2 is optimized)
/// @param[in] x X values
/// @param[in] y Y values
/// @param[in] weights Optional weights (empty for uniform)
/// @param[out] fitted Fitted values (same size as x)
/// @param[out] coeffs Polynomial coefficients [c0, c1, ..., c_DEGREE]
/// @pre x.size() == y.size()
/// @pre fitted.size() == x.size()
/// @pre coeffs.size() >= DEGREE + 1
/// @note Currently only DEGREE=2 is optimized
template<int DEGREE = 2>
auto poly_fit(
    std::span<const Real> x,
    std::span<const Real> y,
    std::span<const Real> weights,
    std::span<Real> fitted,
    std::span<Real> coeffs
) -> void {
    SCL_CHECK_ARG(x.size() == y.size(), "poly_fit: x/y size mismatch");
    SCL_CHECK_ARG(fitted.size() == x.size(), "poly_fit: fitted size mismatch");
    SCL_CHECK_ARG(coeffs.size() >= static_cast<Size>(DEGREE + 1),
                  "poly_fit: coeffs buffer too small");

    if constexpr (DEGREE == 2) {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real sums[8] = {0};
        accumulate_matrices_deg2_simd(x, y, weights, sums);

        // Build symmetric system
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real A_sym[6] = {
            sums[0], sums[1], sums[2],
                     sums[2], sums[3],
                              sums[4]
        };
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real B[3] = { sums[5], sums[6], sums[7] };
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real X[3];

        detail::solve_sym_3x3_static(A_sym, B, X);

        coeffs[0] = X[0];
        coeffs[1] = X[1];
        coeffs[2] = X[2];

        // Evaluate fitted values
        for (Size i = 0; i < x.size(); ++i) {
            fitted[i] = detail::poly_eval<3>(X, x[i]);
        }
    }
    else {
        SCL_CHECK(false, "poly_fit: Only DEGREE=2 is currently optimized");
    }
}

// =============================================================================
// SECTION 3: LOESS (Locally Weighted Scatterplot Smoothing)
// =============================================================================

namespace detail {

/// @brief SIMD kernel for LOESS: accumulate weighted normal equations with tricube weights
/// @param[in] x X values (window)
/// @param[in] y Y values (window)
/// @param[in] target_x Target x position
/// @param[in] max_dist Maximum distance for tricube weighting
/// @param[out] sums Output array [s0,s1,s2,s3,s4,sy0,sy1,sy2]
SCL_FORCE_INLINE
auto accumulate_loess_window_simd(
    std::span<const Real> x,
    std::span<const Real> y,
    Real target_x,
    Real max_dist,
    Real* sums
) -> void {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size N = x.size();
    const Size lanes = s::Lanes(d);

    auto v_s0 = s::Zero(d);
    auto v_s1 = s::Zero(d);
    auto v_s2 = s::Zero(d);
    auto v_s3 = s::Zero(d);
    auto v_s4 = s::Zero(d);

    auto v_sy0 = s::Zero(d);
    auto v_sy1 = s::Zero(d);
    auto v_sy2 = s::Zero(d);

    const auto v_target = s::Set(d, target_x);
    const auto v_inv_max = s::Set(d, (max_dist > Real(1e-9)) ? (Real(1.0) / max_dist) : Real(0.0));
    const auto v_one = s::Set(d, Real(1.0));

    Size i = 0;

    // Vectorized main loop
    for (; i + lanes <= N; i += lanes) {
        auto vx = s::Load(d, x.data() + i);
        auto vy = s::Load(d, y.data() + i);

        // Compute tricube weight
        auto v_dist = s::Abs(s::Sub(vx, v_target));
        auto v_norm = s::Mul(v_dist, v_inv_max);
        auto mask = s::Lt(v_norm, v_one);

        auto v_norm3 = s::Mul(v_norm, s::Mul(v_norm, v_norm));
        auto v_t = s::Sub(v_one, v_norm3);
        auto v_t3 = s::Mul(v_t, s::Mul(v_t, v_t));
        auto vw = s::IfThenElse(mask, v_t3, s::Zero(d));

        auto vx2 = s::Mul(vx, vx);
        auto vx3 = s::Mul(vx2, vx);
        auto vx4 = s::Mul(vx2, vx2);

        v_s0 = s::Add(v_s0, vw);
        v_s1 = s::MulAdd(vx, vw, v_s1);
        v_s2 = s::MulAdd(vx2, vw, v_s2);
        v_s3 = s::MulAdd(vx3, vw, v_s3);
        v_s4 = s::MulAdd(vx4, vw, v_s4);

        auto v_yw = s::Mul(vy, vw);
        v_sy0 = s::Add(v_sy0, v_yw);
        v_sy1 = s::MulAdd(vx, v_yw, v_sy1);
        v_sy2 = s::MulAdd(vx2, v_yw, v_sy2);
    }

    // Horizontal reductions
    sums[0] = s::GetLane(s::SumOfLanes(d, v_s0));
    sums[1] = s::GetLane(s::SumOfLanes(d, v_s1));
    sums[2] = s::GetLane(s::SumOfLanes(d, v_s2));
    sums[3] = s::GetLane(s::SumOfLanes(d, v_s3));
    sums[4] = s::GetLane(s::SumOfLanes(d, v_s4));
    sums[5] = s::GetLane(s::SumOfLanes(d, v_sy0));
    sums[6] = s::GetLane(s::SumOfLanes(d, v_sy1));
    sums[7] = s::GetLane(s::SumOfLanes(d, v_sy2));

    // Scalar tail
    Real inv_max = (max_dist > Real(1e-9)) ? (Real(1.0) / max_dist) : Real(0.0);
    for (; i < N; ++i) {
        Real xi = x[i];
        Real dist = std::abs(xi - target_x);
        if (dist >= max_dist) [[unlikely]] continue;

        Real norm = dist * inv_max;
        Real t = Real(1.0) - norm * norm * norm;
        Real w = t * t * t;

        Real xi2 = xi * xi;
        sums[0] += w;
        sums[1] += xi * w;
        sums[2] += xi2 * w;
        sums[3] += xi2 * xi * w;
        sums[4] += xi2 * xi2 * w;

        Real yw = y[i] * w;
        sums[5] += yw;
        sums[6] += xi * yw;
        sums[7] += xi2 * yw;
    }
}

} // namespace detail

/// @brief Locally Weighted Scatterplot Smoothing
/// @tparam DEGREE Polynomial degree (currently only 2 is optimized)
/// @param[in] x X values (MUST be sorted in ascending order)
/// @param[in] y Y values
/// @param[out] fitted Fitted values (same size as x)
/// @param[in] span Fraction of data to use for each local fit (default 0.3)
/// @pre x.size() == y.size()
/// @pre fitted.size() == x.size()
/// @pre x is sorted in ascending order
/// @note Uses tricube weighting function
/// @note Evaluates in parallel for each point
template<int DEGREE = 2>
auto loess(
    std::span<const Real> x,
    std::span<const Real> y,
    std::span<Real> fitted,
    double span = 0.3
) -> void {
    SCL_CHECK_ARG(x.size() == y.size(), "loess: x/y size mismatch");
    SCL_CHECK_ARG(fitted.size() == x.size(), "loess: fitted buffer size mismatch");
    SCL_CHECK_ARG(span > 0.0 && span <= 1.0, "loess: span must be in (0, 1]");

    const Size n = x.size();
    const Size k = static_cast<Size>(std::ceil(span * static_cast<double>(n)));

    // Process each point
    for (Size i = 0; i < n; ++i) {
        Real target_x = x[i];

        // Find optimal window
        Size half_k = k / 2;
        Size left = (i > half_k) ? (i - half_k) : 0;
        Size right = left + k - 1;

        if (right >= n) [[unlikely]] {
            right = n - 1;
            left = (right >= k) ? (right - k + 1) : 0;
        }

        // Refine window to minimize max distance (exploit sorted X)
        while (true) {
            Real d_left = std::abs(x[left] - target_x);
            Real d_right = std::abs(x[right] - target_x);

            if (left > 0) {
                Real d_new = std::abs(x[left - 1] - target_x);
                if (d_new < d_right) [[unlikely]] {
                    left--;
                    right--;
                    continue;
                }
            }

            if (right < n - 1) {
                Real d_new = std::abs(x[right + 1] - target_x);
                if (d_new < d_left) [[unlikely]] {
                    left++;
                    right++;
                    continue;
                }
            }

            break;
        }

        // Compute neighborhood radius
        Real d_left = std::abs(x[left] - target_x);
        Real d_right = std::abs(x[right] - target_x);
        Real max_dist = (d_left > d_right) ? d_left : d_right;

        if (max_dist < Real(1e-9)) [[unlikely]] max_dist = Real(1e-9);
        max_dist *= Real(1.0000001);  // Small epsilon for numerical stability

        // Accumulate weighted normal equations
        std::span<const Real> x_win(x.data() + left, (right - left + 1));
        std::span<const Real> y_win(y.data() + left, (right - left + 1));

        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real sums[8] = {0};
        detail::accumulate_loess_window_simd(x_win, y_win, target_x, max_dist, sums);

        // Solve local system
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real A_sym[6] = { sums[0], sums[1], sums[2], sums[2], sums[3], sums[4] };
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real B[3] = { sums[5], sums[6], sums[7] };
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real X[3];
        detail::solve_sym_3x3_static(A_sym, B, X);

        // Evaluate at target
        fitted[i] = detail::poly_eval<3>(X, target_x);
    }
}

} // namespace scl::math

