#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/simd.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>
#include <array>
#include <algorithm>

// =============================================================================
/// @file regression.hpp
/// @brief High-Performance Polynomial Regression and LOESS Kernels
///
/// Provides optimized implementations of polynomial regression and locally
/// weighted scatterplot smoothing (LOESS) for real-time analysis.
///
/// Algorithms Implemented:
///
/// 1. Polynomial Regression:
///    Fits f(x) = c0 + c1*x + c2*x^2 + ... + cd*x^d using weighted least
///    squares. Solved via normal equations: (X'WX)c = X'Wy
///
/// 2. LOESS (Locally Estimated Scatterplot Smoothing):
///    For each point x0, fits a local polynomial using tricube weights:
///    w(d) = (1 - |d|^3)^3 if |d| < 1, else 0
///    where d = |x - x0| / max_dist is normalized distance.
///
/// Performance Features:
///
/// - SIMD Accumulation: Vectorized normal equation construction
/// - Register-Level Solver: 3x3 linear systems solved in CPU registers
/// - Zero Heap Allocation: All computations use stack or external buffers
/// - Template Unrolling: Compile-time optimization for small matrices
///
/// Complexity:
///
/// - Polynomial regression: O(N) construction + O(1) solve + O(N) evaluation
/// - LOESS: O(N * K) where K = ceil(span * N)
///
/// Reference: W. S. Cleveland, "Robust Locally Weighted Regression", JASA 1979
// =============================================================================

namespace scl::math::regression {

// =============================================================================
// SECTION 1: Internal Helpers - Tiny Linear Algebra (Register Level)
// =============================================================================

namespace detail {

/// @brief Solve symmetric positive definite 3x3 system Ax = b in registers.
///
/// Uses analytical inverse formula (cofactor expansion) optimized for
/// regression matrices. Faster than Cholesky for small N and avoids sqrt.
///
/// Matrix Storage: Symmetric matrix stored as 6-element flat array.
/// A = | a00  a01  a02 |     Storage: [a00, a01, a02, a11, a12, a22]
///     | a01  a11  a12 |              [  0,   1,   2,   3,   4,   5]
///     | a02  a12  a22 |
///
/// Complexity: O(1) - exactly 27 multiplies, 17 additions
///
/// @param A_sym Symmetric matrix elements [a00, a01, a02, a11, a12, a22]
/// @param b Right-hand side vector [b0, b1, b2]
/// @param x Output solution vector [x0, x1, x2]
SCL_FORCE_INLINE void solve_sym_3x3_static(
    const Real* A_sym,
    const Real* b,
    Real* x
) {
    // Load matrix elements to registers
    Real a00 = A_sym[0], a01 = A_sym[1], a02 = A_sym[2];
    Real a11 = A_sym[3], a12 = A_sym[4];
    Real a22 = A_sym[5];

    // Compute determinant via cofactor expansion along row 0
    // det(A) = a00*(a11*a22 - a12^2) - a01*(a01*a22 - a12*a02)
    //        + a02*(a01*a12 - a11*a02)
    Real det = a00 * (a11 * a22 - a12 * a12) -
               a01 * (a01 * a22 - a12 * a02) +
               a02 * (a01 * a12 - a11 * a02);

    // Regularization: Prevent division by zero for near-singular matrices
    if (std::abs(det) < 1e-12) det = 1e-12;
    Real inv_det = 1.0 / det;

    // Compute x = A^(-1) * b using Cramer's rule (cofactor method)
    // Each x[i] = det(Ai) / det(A), where Ai replaces column i with b

    // x[0]: Minor with b replacing column 0
    x[0] = inv_det * (
        (a11 * a22 - a12 * a12) * b[0] +
        (a02 * a12 - a01 * a22) * b[1] +
        (a01 * a12 - a02 * a11) * b[2]
    );

    // x[1]: Minor with b replacing column 1
    x[1] = inv_det * (
        (a12 * a02 - a01 * a22) * b[0] +
        (a00 * a22 - a02 * a02) * b[1] +
        (a01 * a02 - a00 * a12) * b[2]
    );

    // x[2]: Minor with b replacing column 2
    x[2] = inv_det * (
        (a01 * a12 - a02 * a11) * b[0] +
        (a02 * a01 - a00 * a12) * b[1] +
        (a00 * a11 - a01 * a01) * b[2]
    );
}

/// @brief Generic Gaussian elimination solver for NxN systems (fallback).
///
/// Performs in-place LU decomposition without pivoting. Suitable for
/// well-conditioned matrices from regression problems with N > 3.
///
/// Warning: Pivot-free implementation. May fail for ill-conditioned systems.
///
/// @tparam N System dimension (must be known at compile time)
/// @param A Coefficient matrix (modified in-place)
/// @param b Right-hand side (modified in-place)
/// @param x Output solution vector
template <int N>
SCL_FORCE_INLINE void solve_linear_system_generic(
    std::array<std::array<Real, N>, N>& A,
    std::array<Real, N>& b,
    std::array<Real, N>& x
) {
    // Forward elimination
    for (int i = 0; i < N; ++i) {
        Real div = 1.0 / A[i][i];
        for (int j = i + 1; j < N; ++j) {
            Real factor = A[j][i] * div;
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

/// @brief Evaluate polynomial using Horner's method.
///
/// Computes p(x) = c0 + c1*x + c2*x^2 + ... + c(N-1)*x^(N-1)
/// with optimal O(N) operations and minimal rounding error.
///
/// Algorithm: p(x) = c0 + x*(c1 + x*(c2 + ...))
///
/// @tparam N Number of coefficients (polynomial degree + 1)
/// @param coeffs Polynomial coefficients [c0, c1, ..., c(N-1)]
/// @param x Evaluation point
/// @return p(x)
template <int N>
SCL_FORCE_INLINE Real poly_eval(const Real* coeffs, Real x) {
    Real res = coeffs[N - 1];
    for (int i = N - 2; i >= 0; --i) {
        res = res * x + coeffs[i];
    }
    return res;
}

/// @brief Compute tricube weight function for LOESS.
///
/// Tricube kernel: w(d) = (1 - |d|^3)^3 for |d| < 1, else 0.
///
/// This weight function has compact support and smooth derivatives,
/// making it ideal for local regression.
///
/// @param dist Normalized distance in [0, infinity)
/// @return Weight in [0, 1]
SCL_FORCE_INLINE Real tricube_weight(Real dist) {
    Real a = std::abs(dist);
    if (a >= 1.0) return 0.0;
    Real tmp = 1.0 - a * a * a;
    return tmp * tmp * tmp;
}

} // namespace detail

// =============================================================================
// SECTION 2: Polynomial Regression (SIMD Optimized)
// =============================================================================

/// @brief Accumulate weighted normal equation terms for degree-2 polynomial.
///
/// Computes all unique elements of X'WX and X'Wy in a single SIMD pass.
///
/// Output Layout (8 elements):
/// - sums[0..4]: sum(w), sum(w*x), sum(w*x^2), sum(w*x^3), sum(w*x^4)
/// - sums[5..7]: sum(w*y), sum(w*x*y), sum(w*x^2*y)
///
/// Matrix Construction:
/// LHS (Symmetric):          RHS:
/// | s0  s1  s2 |           | s5 |
/// | s1  s2  s3 |           | s6 |
/// | s2  s3  s4 |           | s7 |
///
/// @param x Input X coordinates
/// @param y Input Y coordinates
/// @param w Optional weights (empty span = uniform weights)
/// @param sums Output buffer [8 elements, caller-allocated]
SCL_FORCE_INLINE void accumulate_matrices_deg2_simd(
    Array<const Real> x,
    Array<const Real> y,
    Array<const Real> w,
    Real* sums
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t N = x.len;
    const size_t lanes = s::lanes();

    // Initialize SIMD accumulators (kept in vector registers)
    auto v_s0 = s::Zero(d);  // sum(w)
    auto v_s1 = s::Zero(d);  // sum(w*x)
    auto v_s2 = s::Zero(d);  // sum(w*x^2)
    auto v_s3 = s::Zero(d);  // sum(w*x^3)
    auto v_s4 = s::Zero(d);  // sum(w*x^4)

    auto v_sy0 = s::Zero(d); // sum(w*y)
    auto v_sy1 = s::Zero(d); // sum(w*x*y)
    auto v_sy2 = s::Zero(d); // sum(w*x^2*y)

    const bool has_weights = !w.empty();
    size_t i = 0;

    // Vectorized main loop
    for (; i + lanes <= N; i += lanes) {
        auto vx = s::Load(d, x.ptr + i);
        auto vy = s::Load(d, y.ptr + i);
        auto vw = has_weights ? s::Load(d, w.ptr + i) : s::Set(d, 1.0);

        // Compute powers: x^2, x^3, x^4
        auto vx2 = s::Mul(vx, vx);
        auto vx3 = s::Mul(vx2, vx);
        auto vx4 = s::Mul(vx2, vx2);

        // Accumulate LHS terms (X'WX components)
        v_s0 = s::Add(v_s0, vw);
        v_s1 = s::MulAdd(vx, vw, v_s1);
        v_s2 = s::MulAdd(vx2, vw, v_s2);
        v_s3 = s::MulAdd(vx3, vw, v_s3);
        v_s4 = s::MulAdd(vx4, vw, v_s4);

        // Accumulate RHS terms (X'Wy components)
        auto v_yw = s::Mul(vy, vw);
        v_sy0 = s::Add(v_sy0, v_yw);
        v_sy1 = s::MulAdd(vx, v_yw, v_sy1);
        v_sy2 = s::MulAdd(vx2, v_yw, v_sy2);
    }

    // Horizontal reduction: sum all lanes
    sums[0] = s::GetLane(s::SumOfLanes(d, v_s0));
    sums[1] = s::GetLane(s::SumOfLanes(d, v_s1));
    sums[2] = s::GetLane(s::SumOfLanes(d, v_s2));
    sums[3] = s::GetLane(s::SumOfLanes(d, v_s3));
    sums[4] = s::GetLane(s::SumOfLanes(d, v_s4));

    sums[5] = s::GetLane(s::SumOfLanes(d, v_sy0));
    sums[6] = s::GetLane(s::SumOfLanes(d, v_sy1));
    sums[7] = s::GetLane(s::SumOfLanes(d, v_sy2));

    // Scalar tail: process remaining elements
    for (; i < N; ++i) {
        Real xi = x[i];
        Real yi = y[i];
        Real wi = has_weights ? w[i] : 1.0;

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

/// @brief Polynomial regression with compile-time degree optimization.
///
/// Fits a polynomial f(x) = c0 + c1*x + c2*x^2 + ... + cd*x^d to data using
/// weighted least squares. Specialized fast path for DEGREE=2.
///
/// Algorithm:
/// 1. Construct normal equations via SIMD accumulation
/// 2. Solve using register-level 3x3 solver
/// 3. Evaluate fitted values in parallel
///
/// Complexity: O(N) construction + O(1) solve + O(N/cores) evaluation
///
/// @tparam DEGREE Polynomial degree (optimized for 2)
/// @param x Input X coordinates
/// @param y Input Y coordinates
/// @param weights Optional weights (empty = uniform)
/// @param fitted Output fitted values [must be pre-allocated, size = x.len]
/// @param coeffs Output coefficients [must be pre-allocated, size >= DEGREE+1]
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

    // Fast Path: Quadratic Regression (Degree 2)
    if constexpr (DEGREE == 2) {
        // Step 1: Accumulate moment matrices (SIMD)
        Real sums[8] = {0};
        accumulate_matrices_deg2_simd(x, y, weights, sums);

        // Step 2: Map sums to symmetric matrix storage
        Real A_sym[6] = {
            sums[0], sums[1], sums[2],
            sums[2], sums[3], sums[4]
        };
        Real B[3] = { sums[5], sums[6], sums[7] };
        Real X[3];

        // Step 3: Solve 3x3 system in registers
        detail::solve_sym_3x3_static(A_sym, B, X);

        // Step 4: Store coefficients
        coeffs[0] = X[0];
        coeffs[1] = X[1];
        coeffs[2] = X[2];

        // Step 5: Evaluate fitted values in parallel
        // f(x) = c0 + c1*x + c2*x^2
        scl::threading::parallel_for(0, x.len, [&](size_t i) {
            fitted[i] = detail::poly_eval<3>(X, x[i]);
        });
    }
    else {
        // Fallback for Higher Degrees (not optimized)
        SCL_ASSERT(false, "PolyFit: Only DEGREE=2 optimized in this build");
    }
}

// =============================================================================
// SECTION 3: LOESS (Locally Weighted Scatterplot Smoothing)
// =============================================================================

namespace detail {

/// @brief SIMD kernel for LOESS: Accumulate weighted normal equations.
///
/// Unlike global regression, LOESS weights depend on distance to the
/// target point, computed on-the-fly using tricube kernel.
///
/// Distance Weighting:
/// w_i = (1 - |(x_i - x0)/r|^3)^3 if |x_i - x0| < r, else 0
/// where r = max_dist is the neighborhood radius.
///
/// @param x Window X coordinates
/// @param y Window Y coordinates
/// @param target_x Center point x0
/// @param max_dist Neighborhood radius r
/// @param sums Output accumulator [8 elements]
SCL_FORCE_INLINE void accumulate_loess_window_simd(
    Array<const Real> x,
    Array<const Real> y,
    Real target_x,
    Real max_dist,
    Real* sums
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t N = x.len;
    const size_t lanes = s::lanes();

    // SIMD accumulators
    auto v_s0 = s::Zero(d);
    auto v_s1 = s::Zero(d);
    auto v_s2 = s::Zero(d);
    auto v_s3 = s::Zero(d);
    auto v_s4 = s::Zero(d);

    auto v_sy0 = s::Zero(d);
    auto v_sy1 = s::Zero(d);
    auto v_sy2 = s::Zero(d);

    // Broadcast constants
    const auto v_target = s::Set(d, target_x);
    const auto v_inv_max = s::Set(d, (max_dist > 1e-9) ? (1.0 / max_dist) : 0.0);
    const auto v_one = s::Set(d, 1.0);

    size_t i = 0;

    // Vectorized main loop
    for (; i + lanes <= N; i += lanes) {
        auto vx = s::Load(d, x.ptr + i);
        auto vy = s::Load(d, y.ptr + i);

        // Compute normalized distance: d = |x - target| / max_dist
        auto v_dist = s::Abs(s::Sub(vx, v_target));
        auto v_norm = s::Mul(v_dist, v_inv_max);
        auto mask = s::Lt(v_norm, v_one);

        // Tricube weight: w = (1 - d^3)^3
        auto v_norm3 = s::Mul(v_norm, s::Mul(v_norm, v_norm));
        auto v_t = s::Sub(v_one, v_norm3);
        auto v_t3 = s::Mul(v_t, s::Mul(v_t, v_t));
        auto vw = s::IfThenElse(mask, v_t3, s::Zero(d));

        // Compute powers of x
        auto vx2 = s::Mul(vx, vx);
        auto vx3 = s::Mul(vx2, vx);
        auto vx4 = s::Mul(vx2, vx2);

        // Accumulate LHS
        v_s0 = s::Add(v_s0, vw);
        v_s1 = s::MulAdd(vx, vw, v_s1);
        v_s2 = s::MulAdd(vx2, vw, v_s2);
        v_s3 = s::MulAdd(vx3, vw, v_s3);
        v_s4 = s::MulAdd(vx4, vw, v_s4);

        // Accumulate RHS
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
    Real inv_max = (max_dist > 1e-9) ? (1.0 / max_dist) : 0.0;
    for (; i < N; ++i) {
        Real xi = x[i];
        Real dist = std::abs(xi - target_x);
        if (dist >= max_dist) continue;

        Real norm = dist * inv_max;
        Real t = 1.0 - norm * norm * norm;
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

/// @brief Locally Weighted Scatterplot Smoothing (High Performance).
///
/// Fits a local polynomial to data around each point using distance-weighted
/// regression with tricube kernel. Produces smooth, non-parametric curves.
///
/// Algorithm: For each point x_i:
/// 1. Find k nearest neighbors (sliding window search on sorted X)
/// 2. Compute local weights via tricube kernel
/// 3. Solve weighted regression in local window
/// 4. Evaluate fitted value at x_i
///
/// Precondition: Input x MUST be sorted in ascending order.
/// Violating this precondition results in undefined behavior.
///
/// Complexity:
/// - Time: O(N * K) where K = ceil(span * N)
/// - Space: O(1) per thread (zero heap allocation)
///
/// @tparam DEGREE Local polynomial degree (optimized for 2)
/// @param x Input X coordinates (MUST BE SORTED)
/// @param y Input Y coordinates
/// @param fitted Output smoothed values [must be pre-allocated]
/// @param span Neighborhood fraction in (0, 1], default 0.3
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
    const Size k = static_cast<Size>(std::ceil(span * n));

    // Parallelize over target points (read-only access to x, y)
    scl::threading::parallel_for(0, n, [&](size_t i) {
        Real target_x = x[i];

        // Step 1: Find Optimal Window [left, right]
        Size half_k = k / 2;
        Size left = (i > half_k) ? (i - half_k) : 0;
        Size right = left + k - 1;

        // Clamp to array bounds
        if (right >= n) {
            right = n - 1;
            left = (right >= k) ? (right - k + 1) : 0;
        }

        // Refine window to minimize max distance (exploit sorted X)
        while (true) {
            Real d_left = std::abs(x[left] - target_x);
            Real d_right = std::abs(x[right] - target_x);

            // Try shifting left
            if (left > 0) {
                Real d_new = std::abs(x[left - 1] - target_x);
                if (d_new < d_right) {
                    left--;
                    right--;
                    continue;
                }
            }

            // Try shifting right
            if (right < n - 1) {
                Real d_new = std::abs(x[right + 1] - target_x);
                if (d_new < d_left) {
                    left++;
                    right++;
                    continue;
                }
            }

            break;
        }

        // Step 2: Compute Neighborhood Radius
        Real d_left = std::abs(x[left] - target_x);
        Real d_right = std::abs(x[right] - target_x);
        Real max_dist = (d_left > d_right) ? d_left : d_right;

        // Avoid zero radius (degenerate case)
        if (max_dist < 1e-9) max_dist = 1e-9;

        // Slightly expand to ensure endpoint weights > 0
        max_dist *= 1.0000001;

        // Step 3: Accumulate Weighted Normal Equations
        Array<const Real> x_win(x.ptr + left, (right - left + 1));
        Array<const Real> y_win(y.ptr + left, (right - left + 1));

        Real sums[8] = {0};
        detail::accumulate_loess_window_simd(x_win, y_win, target_x, max_dist, sums);

        // Step 4: Solve Local System
        Real A_sym[6] = { sums[0], sums[1], sums[2], sums[2], sums[3], sums[4] };
        Real B[3] = { sums[5], sums[6], sums[7] };
        Real X[3];
        detail::solve_sym_3x3_static(A_sym, B, X);

        // Step 5: Evaluate at Target
        fitted[i] = detail::poly_eval<3>(X, target_x);
    });
}

} // namespace scl::math::regression
