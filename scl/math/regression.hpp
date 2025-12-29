#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/simd.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <array>
#include <algorithm>

// =============================================================================
// High-Performance Polynomial Regression and LOESS Kernels
// Provides optimized polynomial fitting and locally weighted smoothing
// =============================================================================

namespace scl::math::regression {

// =============================================================================
// Internal Helpers - Tiny Linear Algebra (Register Level)
// =============================================================================

namespace detail {

// Solve symmetric 3x3 system Ax = b using analytical inverse
SCL_FORCE_INLINE void solve_sym_3x3_static(
    const Real* A_sym,
    const Real* b,
    Real* x
) {
    Real a00 = A_sym[0], a01 = A_sym[1], a02 = A_sym[2];
    Real a11 = A_sym[3], a12 = A_sym[4];
    Real a22 = A_sym[5];

    Real det = a00 * (a11 * a22 - a12 * a12) -
               a01 * (a01 * a22 - a12 * a02) +
               a02 * (a01 * a12 - a11 * a02);

    // Regularization
    if (std::abs(det) < Real(1e-12)) det = Real(1e-12);
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

// Generic Gaussian elimination solver for NxN systems
template <int N>
SCL_FORCE_INLINE void solve_linear_system_generic(
    std::array<std::array<Real, N>, N>& A,
    std::array<Real, N>& b,
    std::array<Real, N>& x
) {
    // Forward elimination
    for (int i = 0; i < N; ++i) {
        auto div = 1.0 / A[i][i];
        for (int j = i + 1; j < N; ++j) {
            auto factor = A[j][i] * div;
            for (int k = i; k < N; ++k) {
                A[j][k] -= factor * A[i][k];
            }
            b[j] -= factor * b[i];
        }
        A[i][i] = div;
    }

    // Back substitution
    for (int i = N - 1; i >= 0; --i) {
        Real sum = b[i];
        for (int j = i + 1; j < N; ++j) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum * A[i][i];
    }
}

// Evaluate polynomial using Horner's method
template <int N>
SCL_FORCE_INLINE Real poly_eval(const Real* coeffs, Real x) {
    Real res = coeffs[N - 1];
    for (int i = N - 2; i >= 0; --i) {
        res = res * x + coeffs[i];
    }
    return res;
}

// Tricube weight function for LOESS
SCL_FORCE_INLINE Real tricube_weight(Real dist) {
    Real a = std::abs(dist);
    if (a >= Real(1.0)) return Real(0.0);
    Real tmp = Real(1.0) - a * a * a;
    return tmp * tmp * tmp;
}

} // namespace detail

// =============================================================================
// Polynomial Regression (SIMD Optimized)
// =============================================================================

// Accumulate weighted normal equation terms for degree-2 polynomial (SIMD)
SCL_FORCE_INLINE void accumulate_matrices_deg2_simd(
    Array<const Real> x,
    Array<const Real> y,
    Array<const Real> w,
    Real* sums
) {
    namespace s = scl::simd;
    auto d = s::SimdTagFor<Real>();
    const size_t N = x.len;
    const size_t lanes = s::Lanes(d);

    auto v_s0 = s::Zero(d);
    auto v_s1 = s::Zero(d);
    auto v_s2 = s::Zero(d);
    auto v_s3 = s::Zero(d);
    auto v_s4 = s::Zero(d);

    auto v_sy0 = s::Zero(d);
    auto v_sy1 = s::Zero(d);
    auto v_sy2 = s::Zero(d);

    const bool has_weights = !w.empty();
    size_t i = 0;

    // Vectorized main loop
    for (; i + lanes <= N; i += lanes) {
        auto vx = s::Load(d, x.ptr + i);
        auto vy = s::Load(d, y.ptr + i);
        auto vw = has_weights ? s::Load(d, w.ptr + i) : s::Set(d, 1.0);

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
        Real xi = x[static_cast<Index>(i)];
        Real yi = y[static_cast<Index>(i)];
        Real wi = has_weights ? w[static_cast<Index>(i)] : Real(1.0);

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

// Polynomial regression with compile-time degree optimization
template <int DEGREE = 2>
void poly_fit(
    Array<const Real> x,
    Array<const Real> y,
    Array<const Real> weights,
    Array<Real> fitted,
    Array<Real> coeffs
) {
    SCL_CHECK_DIM(x.len == y.len, "PolyFit: x/y size mismatch");
    SCL_CHECK_DIM(coeffs.len >= static_cast<Size>(DEGREE + 1),
                  "PolyFit: coeffs buffer too small");

    if constexpr (DEGREE == 2) {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real sums[8] = {0};
        accumulate_matrices_deg2_simd(x, y, weights, std::data(sums));

        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real A_sym[6] = {
            sums[0], sums[1], sums[2],
            sums[2], sums[3], sums[4]
        };
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real B[3] = { sums[5], sums[6], sums[7] };
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real X[3];

        detail::solve_sym_3x3_static(std::data(A_sym), std::data(B), std::data(X));

        coeffs[0] = X[0];
        coeffs[1] = X[1];
        coeffs[2] = X[2];

        // Evaluate fitted values in parallel
        scl::threading::parallel_for(0, x.len, [&](size_t i) {
            fitted[static_cast<Index>(i)] = detail::poly_eval<3>(std::data(X), x[static_cast<Index>(i)]);
        });
    }
    else {
        SCL_ASSERT(false, "PolyFit: Only DEGREE=2 optimized in this build");
    }
}

// =============================================================================
// LOESS (Locally Weighted Scatterplot Smoothing)
// =============================================================================

namespace detail {

// SIMD kernel for LOESS: accumulate weighted normal equations with tricube weights
SCL_FORCE_INLINE void accumulate_loess_window_simd(
    Array<const Real> x,
    Array<const Real> y,
    Real target_x,
    Real max_dist,
    Real* sums
) {
    namespace s = scl::simd;
    auto d = s::SimdTagFor<Real>();
    const size_t N = x.len;
    const size_t lanes = s::Lanes(d);

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
    const auto v_one = s::Set(d, 1.0);

    size_t i = 0;

    // Vectorized main loop
    for (; i + lanes <= N; i += lanes) {
        auto vx = s::Load(d, x.ptr + i);
        auto vy = s::Load(d, y.ptr + i);

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
        Real xi = x[static_cast<Index>(i)];
        Real dist = std::abs(xi - target_x);
        if (dist >= max_dist) continue;

        Real norm = dist * inv_max;
        Real t = Real(1.0) - norm * norm * norm;
        Real w = t * t * t;

        Real xi2 = xi * xi;
        sums[0] += w;
        sums[1] += xi * w;
        sums[2] += xi2 * w;
        sums[3] += xi2 * xi * w;
        sums[4] += xi2 * xi2 * w;

        Real yw = y[static_cast<Index>(i)] * w;
        sums[5] += yw;
        sums[6] += xi * yw;
        sums[7] += xi2 * yw;
    }
}

} // namespace detail

// Locally Weighted Scatterplot Smoothing
// Precondition: x MUST be sorted in ascending order
template <int DEGREE = 2>
void loess(
    Array<const Real> x,
    Array<const Real> y,
    Array<Real> fitted,
    double span = 0.3
) {
    SCL_CHECK_DIM(x.len == y.len, "LOESS: x/y size mismatch");
    SCL_CHECK_DIM(fitted.len == x.len, "LOESS: fitted buffer size mismatch");

    const Size n = x.len;
    const Size k = static_cast<Size>(std::ceil(span * static_cast<double>(n)));

    scl::threading::parallel_for(0, n, [&](size_t i) {
        Real target_x = x[static_cast<Index>(i)];

        // Find optimal window
        Size half_k = k / 2;
        Size left = (i > half_k) ? (i - half_k) : 0;
        Size right = left + k - 1;

        if (right >= n) {
            right = n - 1;
            left = (right >= k) ? (right - k + 1) : 0;
        }

        // Refine window to minimize max distance (exploit sorted X)
        while (true) {
            Real d_left = std::abs(x[static_cast<Index>(left)] - target_x);
            Real d_right = std::abs(x[static_cast<Index>(right)] - target_x);

            if (left > 0) {
                Real d_new = std::abs(x[static_cast<Index>(left - 1)] - target_x);
                if (d_new < d_right) {
                    left--;
                    right--;
                    continue;
                }
            }

            if (right < n - 1) {
                Real d_new = std::abs(x[static_cast<Index>(right + 1)] - target_x);
                if (d_new < d_left) {
                    left++;
                    right++;
                    continue;
                }
            }

            break;
        }

        // Compute neighborhood radius
        Real d_left = std::abs(x[static_cast<Index>(left)] - target_x);
        Real d_right = std::abs(x[static_cast<Index>(right)] - target_x);
        Real max_dist = (d_left > d_right) ? d_left : d_right;

        if (max_dist < 1e-9) max_dist = 1e-9;
        max_dist *= 1.0000001;

        // Accumulate weighted normal equations
        Array<const Real> x_win(x.ptr + left, (right - left + 1));
        Array<const Real> y_win(y.ptr + left, (right - left + 1));

        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real sums[8] = {0};
        detail::accumulate_loess_window_simd(x_win, y_win, target_x, max_dist, std::data(sums));

        // Solve local system
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real A_sym[6] = { sums[0], sums[1], sums[2], sums[2], sums[3], sums[4] };
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real B[3] = { sums[5], sums[6], sums[7] };
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real X[3];
        detail::solve_sym_3x3_static(std::data(A_sym), std::data(B), std::data(X));
        // Evaluate at target
        fitted[static_cast<Index>(i)] = detail::poly_eval<3>(std::data(X), target_x);
    });
}

} // namespace scl::math::regression
