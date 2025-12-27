#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/simd.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>

// =============================================================================
// FILE: scl/kernel/sparse_opt.hpp
// BRIEF: Sparse optimization methods
//
// STRATEGIC POSITION: Sparse + Nonlinear (Tier 4 - Advanced Foundation)
// - Optimization that maintains sparsity
// - Foundation for advanced methods (Lasso, elastic net)
// - Nonlinear optimization algorithms
//
// APPLICATIONS:
// - Sparse regression (Lasso, Elastic Net)
// - Feature selection
// - Sparse PCA
// - Compressed sensing
//
// KEY ALGORITHMS:
// - Coordinate descent for L1 regularization
// - Proximal gradient methods
// - Iterative soft/hard thresholding
// =============================================================================

namespace scl::kernel::sparse_opt {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_ALPHA = Real(1.0);       // Regularization strength
    constexpr Real DEFAULT_L1_RATIO = Real(1.0);    // 1.0 = Lasso, 0.0 = Ridge
    constexpr Index DEFAULT_MAX_ITER = 1000;
    constexpr Real DEFAULT_TOL = Real(1e-4);
}

// =============================================================================
// Regularization Types
// =============================================================================

enum class RegularizationType {
    L1,              // Lasso (sparsity-inducing)
    L2,              // Ridge (shrinkage)
    ELASTIC_NET,     // L1 + L2
    SCAD,            // Smoothly Clipped Absolute Deviation
    MCP              // Minimax Concave Penalty
};

// =============================================================================
// Coordinate Descent (TODO: Implementation)
// =============================================================================

// TODO: Coordinate descent for Lasso regression
template <typename T, bool IsCSR>
void lasso_coordinate_descent(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Real alpha,
    Array<Real> coefficients,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOL
);

// TODO: Elastic net via coordinate descent
template <typename T, bool IsCSR>
void elastic_net_coordinate_descent(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Real alpha,
    Real l1_ratio,
    Array<Real> coefficients,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOL
);

// TODO: Coordinate descent with feature screening
template <typename T, bool IsCSR>
void coordinate_descent_screening(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Real alpha,
    Array<Real> coefficients,
    Index max_iter = config::DEFAULT_MAX_ITER
);

// =============================================================================
// Proximal Methods (TODO: Implementation)
// =============================================================================

// TODO: Proximal gradient descent
template <typename T, bool IsCSR>
void proximal_gradient(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Real alpha,
    RegularizationType reg_type,
    Array<Real> coefficients,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOL
);

// TODO: Accelerated proximal gradient (FISTA)
template <typename T, bool IsCSR>
void fista(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Real alpha,
    Array<Real> coefficients,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOL
);

// TODO: Proximal operator for L1 (soft thresholding)
void prox_l1(
    Array<Real> x,
    Real lambda
);

// TODO: Proximal operator for elastic net
void prox_elastic_net(
    Array<Real> x,
    Real lambda,
    Real l1_ratio
);

// =============================================================================
// Iterative Thresholding (TODO: Implementation)
// =============================================================================

// TODO: Iterative soft thresholding (ISTA)
template <typename T, bool IsCSR>
void ista(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Real lambda,
    Array<Real> coefficients,
    Index max_iter = config::DEFAULT_MAX_ITER
);

// TODO: Iterative hard thresholding (IHT)
template <typename T, bool IsCSR>
void iht(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Index sparsity_level,                   // Number of non-zeros
    Array<Real> coefficients,
    Index max_iter = config::DEFAULT_MAX_ITER
);

// =============================================================================
// Regularization Path (TODO: Implementation)
// =============================================================================

// TODO: Compute full regularization path (multiple alphas)
template <typename T, bool IsCSR>
void lasso_path(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    const std::vector<Real>& alphas,
    std::vector<Array<Real>>& coefficient_paths,
    Index max_iter = config::DEFAULT_MAX_ITER
);

// TODO: Compute regularization path with cross-validation
template <typename T, bool IsCSR>
void lasso_path_cv(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Index n_folds,
    Array<Real> optimal_coefficients,
    Real& optimal_alpha
);

// =============================================================================
// Sparse Regression Variants (TODO: Implementation)
// =============================================================================

// TODO: Group Lasso (grouped sparsity)
template <typename T, bool IsCSR>
void group_lasso(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    const std::vector<std::vector<Index>>& groups,
    Real alpha,
    Array<Real> coefficients,
    Index max_iter = config::DEFAULT_MAX_ITER
);

// TODO: Sparse logistic regression
template <typename T, bool IsCSR>
void sparse_logistic_regression(
    const Sparse<T, IsCSR>& X,
    Array<const Index> y_binary,
    Real alpha,
    Array<Real> coefficients,
    Index max_iter = config::DEFAULT_MAX_ITER
);

// =============================================================================
// Helper Functions (TODO: Implementation)
// =============================================================================

namespace detail {

// TODO: Soft thresholding operator
SCL_FORCE_INLINE Real soft_threshold(
    Real x,
    Real lambda
);

// TODO: Hard thresholding operator
SCL_FORCE_INLINE Real hard_threshold(
    Real x,
    Real lambda
);

// TODO: Compute gradient for coordinate
template <typename T, bool IsCSR>
Real compute_coordinate_gradient(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Array<const Real> coefficients,
    Index coordinate,
    const Real* residuals
);

// TODO: Update residuals after coordinate update
template <typename T, bool IsCSR>
void update_residuals(
    const Sparse<T, IsCSR>& X,
    Index coordinate,
    Real delta_coef,
    Array<Real> residuals
);

// TODO: Check convergence via coefficient change
bool check_convergence_coef(
    const Real* coef_old,
    const Real* coef_new,
    Size n,
    Real tol
);

// TODO: Lipschitz constant estimation
template <typename T, bool IsCSR>
Real estimate_lipschitz_constant(
    const Sparse<T, IsCSR>& X
);

} // namespace detail

} // namespace scl::kernel::sparse_opt

