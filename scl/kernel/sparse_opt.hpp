#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/sparse_opt.hpp
// BRIEF: Sparse optimization methods with SIMD optimization
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
    constexpr Real DEFAULT_ALPHA = Real(1.0);
    constexpr Real DEFAULT_L1_RATIO = Real(1.0);
    constexpr Index DEFAULT_MAX_ITER = 1000;
    constexpr Real DEFAULT_TOL = Real(1e-4);
    constexpr Size PREFETCH_DISTANCE = 8;
    constexpr Size SIMD_THRESHOLD = 16;
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size UNROLL_FACTOR = 4;
    constexpr Real EPS = Real(1e-12);
    constexpr Real LIPSCHITZ_SCALING = Real(1.5);
}

// =============================================================================
// Regularization Types
// =============================================================================

enum class RegularizationType {
    L1,
    L2,
    ELASTIC_NET,
    SCAD,
    MCP
};

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// SIMD-optimized soft thresholding operator: sign(x) * max(|x| - lambda, 0)
SCL_FORCE_INLINE SCL_HOT Real soft_threshold(Real x, Real lambda) {
    if (x > lambda) {
        return x - lambda;
    } else if (x < -lambda) {
        return x + lambda;
    }
    return Real(0);
}

// Hard thresholding operator: x if |x| > lambda, 0 otherwise
SCL_FORCE_INLINE SCL_HOT Real hard_threshold(Real x, Real lambda) {
    return (x > lambda || x < -lambda) ? x : Real(0);
}

// SCAD penalty proximal operator
SCL_FORCE_INLINE SCL_HOT Real prox_scad(Real x, Real lambda, Real a = Real(3.7)) {
    Real abs_x = (x >= Real(0)) ? x : -x;
    Real sign_x = (x >= Real(0)) ? Real(1) : Real(-1);
    
    if (abs_x <= lambda) {
        return Real(0);
    } else if (abs_x <= Real(2) * lambda) {
        return sign_x * (abs_x - lambda);
    } else if (abs_x <= a * lambda) {
        return sign_x * ((a - Real(1)) * abs_x - a * lambda) / (a - Real(2));
    }
    return x;
}

// MCP penalty proximal operator
SCL_FORCE_INLINE SCL_HOT Real prox_mcp(Real x, Real lambda, Real gamma = Real(3.0)) {
    Real abs_x = (x >= Real(0)) ? x : -x;
    Real sign_x = (x >= Real(0)) ? Real(1) : Real(-1);
    
    if (abs_x <= gamma * lambda) {
        return sign_x * scl::algo::max2(abs_x - lambda, Real(0)) * gamma / (gamma - Real(1));
    }
    return x;
}

// SIMD-optimized sparse dot product: x^T * sparse_column
template <typename T, bool IsCSR>
SCL_FORCE_INLINE SCL_HOT T sparse_dot_dense(
    const Sparse<T, IsCSR>& X,
    Index col_idx,
    const Real* SCL_RESTRICT dense_vec
) {
    if constexpr (IsCSR) {
        // For CSR, need to iterate all rows
        const Index n_rows = X.rows();
        T result = T(0);
        
        for (Index i = 0; i < n_rows; ++i) {
            const Index len = X.primary_length_unsafe(i);
            if (len == 0) continue;
            
            auto indices = X.primary_indices_unsafe(i);
            auto values = X.primary_values_unsafe(i);
            
            // Binary search for col_idx
            const Index* pos = scl::algo::lower_bound(
                indices.ptr, indices.ptr + len, col_idx);
            
            if (pos != indices.ptr + len && *pos == col_idx) {
                Index k = static_cast<Index>(pos - indices.ptr);
                result += values[k] * static_cast<T>(dense_vec[i]);
            }
        }
        return result;
    } else {
        // For CSC, direct column access
        const Index len = X.primary_length_unsafe(col_idx);
        if (len == 0) return T(0);
        
        auto indices = X.primary_indices_unsafe(col_idx);
        auto values = X.primary_values_unsafe(col_idx);
        const Size len_sz = static_cast<Size>(len);
        
        T acc0 = T(0), acc1 = T(0), acc2 = T(0), acc3 = T(0);
        Size k = 0;
        
        for (; k + 4 <= len_sz; k += 4) {
            if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len_sz)) {
                SCL_PREFETCH_READ(&dense_vec[indices[k + config::PREFETCH_DISTANCE]], 0);
            }
            acc0 += values[k + 0] * static_cast<T>(dense_vec[indices[k + 0]]);
            acc1 += values[k + 1] * static_cast<T>(dense_vec[indices[k + 1]]);
            acc2 += values[k + 2] * static_cast<T>(dense_vec[indices[k + 2]]);
            acc3 += values[k + 3] * static_cast<T>(dense_vec[indices[k + 3]]);
        }
        
        T result = acc0 + acc1 + acc2 + acc3;
        for (; k < len_sz; ++k) {
            result += values[k] * static_cast<T>(dense_vec[indices[k]]);
        }
        
        return result;
    }
}

// Compute X^T * X for a single column (diagonal element)
template <typename T, bool IsCSR>
SCL_FORCE_INLINE SCL_HOT T column_squared_norm(
    const Sparse<T, IsCSR>& X,
    Index col_idx
) {
    if constexpr (!IsCSR) {
        // CSC: direct column access
        const Index len = X.primary_length_unsafe(col_idx);
        if (len == 0) return T(0);
        
        auto values = X.primary_values_unsafe(col_idx);
        return scl::vectorize::sum_squared(Array<const T>(values.ptr, static_cast<Size>(len)));
    } else {
        // CSR: iterate all rows
        const Index n_rows = X.rows();
        T result = T(0);
        
        for (Index i = 0; i < n_rows; ++i) {
            const Index len = X.primary_length_unsafe(i);
            if (len == 0) continue;
            
            auto indices = X.primary_indices_unsafe(i);
            auto values = X.primary_values_unsafe(i);
            
            const Index* pos = scl::algo::lower_bound(
                indices.ptr, indices.ptr + len, col_idx);
            
            if (pos != indices.ptr + len && *pos == col_idx) {
                Index k = static_cast<Index>(pos - indices.ptr);
                result += values[k] * values[k];
            }
        }
        return result;
    }
}

// Update residuals: r = r - delta * X[:, j]
template <typename T, bool IsCSR>
SCL_FORCE_INLINE void update_residuals_column(
    const Sparse<T, IsCSR>& X,
    Index col_idx,
    T delta,
    Real* SCL_RESTRICT residuals
) {
    if (SCL_UNLIKELY(delta == T(0))) return;
    
    if constexpr (!IsCSR) {
        // CSC: direct column access
        const Index len = X.primary_length_unsafe(col_idx);
        if (len == 0) return;
        
        auto indices = X.primary_indices_unsafe(col_idx);
        auto values = X.primary_values_unsafe(col_idx);
        const Size len_sz = static_cast<Size>(len);
        
        Size k = 0;
        for (; k + 4 <= len_sz; k += 4) {
            residuals[indices[k + 0]] -= static_cast<Real>(delta * values[k + 0]);
            residuals[indices[k + 1]] -= static_cast<Real>(delta * values[k + 1]);
            residuals[indices[k + 2]] -= static_cast<Real>(delta * values[k + 2]);
            residuals[indices[k + 3]] -= static_cast<Real>(delta * values[k + 3]);
        }
        for (; k < len_sz; ++k) {
            residuals[indices[k]] -= static_cast<Real>(delta * values[k]);
        }
    } else {
        // CSR: iterate all rows
        const Index n_rows = X.rows();
        for (Index i = 0; i < n_rows; ++i) {
            const Index len = X.primary_length_unsafe(i);
            if (len == 0) continue;
            
            auto indices = X.primary_indices_unsafe(i);
            auto values = X.primary_values_unsafe(i);
            
            const Index* pos = scl::algo::lower_bound(
                indices.ptr, indices.ptr + len, col_idx);
            
            if (pos != indices.ptr + len && *pos == col_idx) {
                Index k = static_cast<Index>(pos - indices.ptr);
                residuals[i] -= static_cast<Real>(delta * values[k]);
            }
        }
    }
}

// Compute X * coef (sparse matrix-vector multiply)
template <typename T, bool IsCSR>
void sparse_matvec(
    const Sparse<T, IsCSR>& X,
    const Real* SCL_RESTRICT coef,
    Real* SCL_RESTRICT output,
    Size n_output
) {
    scl::algo::zero(output, n_output);
    
    if constexpr (IsCSR) {
        // CSR: parallel over rows
        const Index n_rows = X.rows();
        
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_rows), [&](size_t i) {
            const Index idx = static_cast<Index>(i);
            const Index len = X.primary_length_unsafe(idx);
            if (len == 0) return;
            
            auto indices = X.primary_indices_unsafe(idx);
            auto values = X.primary_values_unsafe(idx);
            const Size len_sz = static_cast<Size>(len);
            
            Real acc0 = Real(0), acc1 = Real(0), acc2 = Real(0), acc3 = Real(0);
            Size k = 0;
            
            for (; k + 4 <= len_sz; k += 4) {
                acc0 += static_cast<Real>(values[k + 0]) * coef[indices[k + 0]];
                acc1 += static_cast<Real>(values[k + 1]) * coef[indices[k + 1]];
                acc2 += static_cast<Real>(values[k + 2]) * coef[indices[k + 2]];
                acc3 += static_cast<Real>(values[k + 3]) * coef[indices[k + 3]];
            }
            
            Real result = acc0 + acc1 + acc2 + acc3;
            for (; k < len_sz; ++k) {
                result += static_cast<Real>(values[k]) * coef[indices[k]];
            }
            
            output[i] = result;
        });
    } else {
        // CSC: accumulate column contributions
        const Index n_cols = X.cols();
        
        for (Index j = 0; j < n_cols; ++j) {
            const Index len = X.primary_length_unsafe(j);
            if (len == 0 || coef[j] == Real(0)) continue;
            
            auto indices = X.primary_indices_unsafe(j);
            auto values = X.primary_values_unsafe(j);
            const Size len_sz = static_cast<Size>(len);
            const Real c = coef[j];
            
            for (Size k = 0; k < len_sz; ++k) {
                output[indices[k]] += static_cast<Real>(values[k]) * c;
            }
        }
    }
}

// Check convergence via coefficient change
SCL_FORCE_INLINE bool check_convergence(
    const Real* SCL_RESTRICT coef_old,
    const Real* SCL_RESTRICT coef_new,
    Size n,
    Real tol
) {
    Real max_diff = Real(0);
    Real max_val = Real(0);
    
    for (Size i = 0; i < n; ++i) {
        Real diff = (coef_new[i] - coef_old[i]);
        diff = (diff >= Real(0)) ? diff : -diff;
        max_diff = scl::algo::max2(max_diff, diff);
        
        Real val = (coef_new[i] >= Real(0)) ? coef_new[i] : -coef_new[i];
        max_val = scl::algo::max2(max_val, val);
    }
    
    if (max_val < config::EPS) return true;
    return (max_diff / max_val) < tol;
}

// Estimate Lipschitz constant (largest eigenvalue of X^T * X)
template <typename T, bool IsCSR>
Real estimate_lipschitz_constant(
    const Sparse<T, IsCSR>& X,
    Index max_power_iter = 20
) {
    const Index n_cols = X.cols();
    const Index n_rows = X.rows();
    const Size n_cols_sz = static_cast<Size>(n_cols);
    const Size n_rows_sz = static_cast<Size>(n_rows);
    
    // Power iteration to estimate largest singular value
    Real* v = scl::memory::aligned_alloc<Real>(n_cols_sz, SCL_ALIGNMENT);
    Real* u = scl::memory::aligned_alloc<Real>(n_rows_sz, SCL_ALIGNMENT);
    Real* Xv = scl::memory::aligned_alloc<Real>(n_rows_sz, SCL_ALIGNMENT);
    
    // Initialize v randomly
    for (Size i = 0; i < n_cols_sz; ++i) {
        v[i] = Real(1.0) / std::sqrt(static_cast<Real>(n_cols_sz));
    }
    
    Real sigma = Real(1.0);
    
    for (Index iter = 0; iter < max_power_iter; ++iter) {
        // Xv = X * v
        sparse_matvec(X, v, Xv, n_rows_sz);
        
        // u = Xv / ||Xv||
        Real norm_Xv = std::sqrt(scl::vectorize::sum_squared(
            Array<const Real>(Xv, n_rows_sz)));
        if (norm_Xv < config::EPS) break;
        
        for (Size i = 0; i < n_rows_sz; ++i) {
            u[i] = Xv[i] / norm_Xv;
        }
        
        // v_new = X^T * u
        scl::algo::zero(v, n_cols_sz);
        if constexpr (!IsCSR) {
            // CSC: parallel over columns
            for (Index j = 0; j < n_cols; ++j) {
                v[j] = static_cast<Real>(sparse_dot_dense(X, j, u));
            }
        } else {
            // CSR: iterate rows
            for (Index i = 0; i < n_rows; ++i) {
                const Index len = X.primary_length_unsafe(i);
                if (len == 0) continue;
                
                auto indices = X.primary_indices_unsafe(i);
                auto values = X.primary_values_unsafe(i);
                
                for (Index k = 0; k < len; ++k) {
                    v[indices[k]] += static_cast<Real>(values[k]) * u[i];
                }
            }
        }
        
        // sigma = ||v||
        sigma = std::sqrt(scl::vectorize::sum_squared(
            Array<const Real>(v, n_cols_sz)));
        
        if (sigma < config::EPS) break;
        
        // Normalize v
        Real inv_sigma = Real(1.0) / sigma;
        for (Size i = 0; i < n_cols_sz; ++i) {
            v[i] *= inv_sigma;
        }
    }
    
    scl::memory::aligned_free(v, SCL_ALIGNMENT);
    scl::memory::aligned_free(u, SCL_ALIGNMENT);
    scl::memory::aligned_free(Xv, SCL_ALIGNMENT);
    
    return sigma * sigma * config::LIPSCHITZ_SCALING;
}

} // namespace detail

// =============================================================================
// Proximal Operators
// =============================================================================

// Proximal operator for L1 (soft thresholding)
void prox_l1(Array<Real> x, Real lambda) {
    const Size n = x.len;
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::Lanes(d);
    
    const auto v_lambda = s::Set(d, lambda);
    const auto v_neg_lambda = s::Set(d, -lambda);
    const auto v_zero = s::Zero(d);
    
    Size i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto v = s::Load(d, x.ptr + i);
        
        // SIMD soft thresholding
        auto pos_mask = s::Gt(v, v_lambda);
        auto neg_mask = s::Lt(v, v_neg_lambda);
        
        auto pos_result = s::Sub(v, v_lambda);
        auto neg_result = s::Add(v, v_lambda);
        
        auto result = s::IfThenElse(pos_mask, pos_result,
                      s::IfThenElse(neg_mask, neg_result, v_zero));
        
        s::Store(result, d, x.ptr + i);
    }
    
    for (; i < n; ++i) {
        x[i] = detail::soft_threshold(x[i], lambda);
    }
}

// Proximal operator for elastic net
void prox_elastic_net(Array<Real> x, Real lambda, Real l1_ratio) {
    const Real l1_lambda = lambda * l1_ratio;
    const Real l2_scale = Real(1.0) / (Real(1.0) + lambda * (Real(1.0) - l1_ratio));
    const Size n = x.len;
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::Lanes(d);
    
    const auto v_l1_lambda = s::Set(d, l1_lambda);
    const auto v_neg_l1_lambda = s::Set(d, -l1_lambda);
    const auto v_l2_scale = s::Set(d, l2_scale);
    const auto v_zero = s::Zero(d);
    
    Size i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto v = s::Load(d, x.ptr + i);
        
        // Soft thresholding then L2 scaling
        auto pos_mask = s::Gt(v, v_l1_lambda);
        auto neg_mask = s::Lt(v, v_neg_l1_lambda);
        
        auto pos_result = s::Sub(v, v_l1_lambda);
        auto neg_result = s::Add(v, v_l1_lambda);
        
        auto soft = s::IfThenElse(pos_mask, pos_result,
                    s::IfThenElse(neg_mask, neg_result, v_zero));
        
        auto result = s::Mul(soft, v_l2_scale);
        s::Store(result, d, x.ptr + i);
    }
    
    for (; i < n; ++i) {
        Real soft = detail::soft_threshold(x[i], l1_lambda);
        x[i] = soft * l2_scale;
    }
}

// =============================================================================
// Coordinate Descent for Lasso
// =============================================================================

// Lasso regression via coordinate descent
// minimize (1/2n) * ||y - X*coef||^2 + alpha * ||coef||_1
template <typename T, bool IsCSR>
void lasso_coordinate_descent(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Real alpha,
    Array<Real> coefficients,
    Index max_iter,
    Real tol
) {
    const Index n_samples = X.rows();
    const Index n_features = X.cols();
    const Size n_samples_sz = static_cast<Size>(n_samples);
    const Size n_features_sz = static_cast<Size>(n_features);
    
    SCL_CHECK_DIM(y.len >= n_samples_sz, "y size mismatch");
    SCL_CHECK_DIM(coefficients.len >= n_features_sz, "coefficients size mismatch");
    SCL_CHECK_ARG(alpha >= Real(0), "alpha must be non-negative");
    SCL_CHECK_ARG(max_iter > 0, "max_iter must be positive");
    
    // Initialize coefficients to zero
    scl::algo::zero(coefficients.ptr, n_features_sz);
    
    // Allocate residuals: r = y - X * coef (initially r = y)
    Real* residuals = scl::memory::aligned_alloc<Real>(n_samples_sz, SCL_ALIGNMENT);
    scl::algo::copy(y.ptr, residuals, n_samples_sz);
    
    // Precompute column squared norms (X[:, j]^T * X[:, j])
    Real* col_norms_sq = scl::memory::aligned_alloc<Real>(n_features_sz, SCL_ALIGNMENT);
    
    for (Size j = 0; j < n_features_sz; ++j) {
        col_norms_sq[j] = static_cast<Real>(detail::column_squared_norm(X, static_cast<Index>(j)));
    }
    
    // Old coefficients for convergence check
    Real* coef_old = scl::memory::aligned_alloc<Real>(n_features_sz, SCL_ALIGNMENT);
    
    const Real n_inv = Real(1.0) / static_cast<Real>(n_samples);
    const Real lambda = alpha * static_cast<Real>(n_samples);
    
    for (Index iter = 0; iter < max_iter; ++iter) {
        // Save old coefficients
        scl::algo::copy(coefficients.ptr, coef_old, n_features_sz);
        
        // Coordinate descent sweep
        for (Index j = 0; j < n_features; ++j) {
            const Real norm_sq = col_norms_sq[j];
            if (SCL_UNLIKELY(norm_sq < config::EPS)) continue;
            
            // Compute X[:, j]^T * residuals + norm_sq * coef[j]
            Real rho = static_cast<Real>(detail::sparse_dot_dense(X, j, residuals));
            rho += norm_sq * coefficients[j];
            
            // Soft thresholding
            Real coef_new = detail::soft_threshold(rho, lambda) / norm_sq;
            
            // Update residuals if coefficient changed
            Real delta = coef_new - coefficients[j];
            if (delta != Real(0)) {
                detail::update_residuals_column(X, j, static_cast<T>(delta), residuals);
                coefficients[j] = coef_new;
            }
        }
        
        // Check convergence
        if (detail::check_convergence(coef_old, coefficients.ptr, n_features_sz, tol)) {
            break;
        }
    }
    
    scl::memory::aligned_free(residuals, SCL_ALIGNMENT);
    scl::memory::aligned_free(col_norms_sq, SCL_ALIGNMENT);
    scl::memory::aligned_free(coef_old, SCL_ALIGNMENT);
}

// =============================================================================
// Elastic Net Coordinate Descent
// =============================================================================

// Elastic net via coordinate descent
// minimize (1/2n) * ||y - X*coef||^2 + alpha * (l1_ratio * ||coef||_1 + (1-l1_ratio)/2 * ||coef||_2^2)
template <typename T, bool IsCSR>
void elastic_net_coordinate_descent(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Real alpha,
    Real l1_ratio,
    Array<Real> coefficients,
    Index max_iter,
    Real tol
) {
    const Index n_samples = X.rows();
    const Index n_features = X.cols();
    const Size n_samples_sz = static_cast<Size>(n_samples);
    const Size n_features_sz = static_cast<Size>(n_features);
    
    SCL_CHECK_DIM(y.len >= n_samples_sz, "y size mismatch");
    SCL_CHECK_DIM(coefficients.len >= n_features_sz, "coefficients size mismatch");
    SCL_CHECK_ARG(alpha >= Real(0), "alpha must be non-negative");
    SCL_CHECK_ARG(l1_ratio >= Real(0) && l1_ratio <= Real(1), "l1_ratio must be in [0, 1]");
    
    scl::algo::zero(coefficients.ptr, n_features_sz);
    
    Real* residuals = scl::memory::aligned_alloc<Real>(n_samples_sz, SCL_ALIGNMENT);
    scl::algo::copy(y.ptr, residuals, n_samples_sz);
    
    Real* col_norms_sq = scl::memory::aligned_alloc<Real>(n_features_sz, SCL_ALIGNMENT);
    for (Size j = 0; j < n_features_sz; ++j) {
        col_norms_sq[j] = static_cast<Real>(detail::column_squared_norm(X, static_cast<Index>(j)));
    }
    
    Real* coef_old = scl::memory::aligned_alloc<Real>(n_features_sz, SCL_ALIGNMENT);
    
    const Real n_samples_real = static_cast<Real>(n_samples);
    const Real l1_lambda = alpha * l1_ratio * n_samples_real;
    const Real l2_lambda = alpha * (Real(1.0) - l1_ratio) * n_samples_real;
    
    for (Index iter = 0; iter < max_iter; ++iter) {
        scl::algo::copy(coefficients.ptr, coef_old, n_features_sz);
        
        for (Index j = 0; j < n_features; ++j) {
            const Real norm_sq = col_norms_sq[j];
            if (SCL_UNLIKELY(norm_sq < config::EPS)) continue;
            
            Real rho = static_cast<Real>(detail::sparse_dot_dense(X, j, residuals));
            rho += norm_sq * coefficients[j];
            
            // Elastic net update
            Real denom = norm_sq + l2_lambda;
            Real coef_new = detail::soft_threshold(rho, l1_lambda) / denom;
            
            Real delta = coef_new - coefficients[j];
            if (delta != Real(0)) {
                detail::update_residuals_column(X, j, static_cast<T>(delta), residuals);
                coefficients[j] = coef_new;
            }
        }
        
        if (detail::check_convergence(coef_old, coefficients.ptr, n_features_sz, tol)) {
            break;
        }
    }
    
    scl::memory::aligned_free(residuals, SCL_ALIGNMENT);
    scl::memory::aligned_free(col_norms_sq, SCL_ALIGNMENT);
    scl::memory::aligned_free(coef_old, SCL_ALIGNMENT);
}

// =============================================================================
// Proximal Gradient Methods
// =============================================================================

// Proximal gradient descent (ISTA)
template <typename T, bool IsCSR>
void proximal_gradient(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Real alpha,
    RegularizationType reg_type,
    Array<Real> coefficients,
    Index max_iter,
    Real tol
) {
    const Index n_samples = X.rows();
    const Index n_features = X.cols();
    const Size n_samples_sz = static_cast<Size>(n_samples);
    const Size n_features_sz = static_cast<Size>(n_features);
    
    SCL_CHECK_DIM(y.len >= n_samples_sz, "y size mismatch");
    SCL_CHECK_DIM(coefficients.len >= n_features_sz, "coefficients size mismatch");
    
    scl::algo::zero(coefficients.ptr, n_features_sz);
    
    // Estimate step size (1 / Lipschitz constant)
    Real L = detail::estimate_lipschitz_constant(X);
    Real step_size = Real(1.0) / L;
    
    Real* residuals = scl::memory::aligned_alloc<Real>(n_samples_sz, SCL_ALIGNMENT);
    Real* gradient = scl::memory::aligned_alloc<Real>(n_features_sz, SCL_ALIGNMENT);
    Real* coef_old = scl::memory::aligned_alloc<Real>(n_features_sz, SCL_ALIGNMENT);
    
    for (Index iter = 0; iter < max_iter; ++iter) {
        scl::algo::copy(coefficients.ptr, coef_old, n_features_sz);
        
        // Compute residuals: r = X * coef - y
        detail::sparse_matvec(X, coefficients.ptr, residuals, n_samples_sz);
        for (Size i = 0; i < n_samples_sz; ++i) {
            residuals[i] -= y[i];
        }
        
        // Compute gradient: grad = X^T * residuals
        scl::algo::zero(gradient, n_features_sz);
        if constexpr (!IsCSR) {
            for (Index j = 0; j < n_features; ++j) {
                gradient[j] = static_cast<Real>(detail::sparse_dot_dense(X, j, residuals));
            }
        } else {
            for (Index i = 0; i < n_samples; ++i) {
                const Index len = X.primary_length_unsafe(i);
                if (len == 0) continue;
                
                auto indices = X.primary_indices_unsafe(i);
                auto values = X.primary_values_unsafe(i);
                
                for (Index k = 0; k < len; ++k) {
                    gradient[indices[k]] += static_cast<Real>(values[k]) * residuals[i];
                }
            }
        }
        
        // Gradient step
        for (Size j = 0; j < n_features_sz; ++j) {
            coefficients[j] -= step_size * gradient[j];
        }
        
        // Proximal step
        const Real prox_lambda = alpha * step_size;
        
        switch (reg_type) {
            case RegularizationType::L1:
                prox_l1(coefficients, prox_lambda);
                break;
            case RegularizationType::ELASTIC_NET:
                prox_elastic_net(coefficients, prox_lambda, Real(0.5));
                break;
            case RegularizationType::SCAD:
                for (Size j = 0; j < n_features_sz; ++j) {
                    coefficients[j] = detail::prox_scad(coefficients[j], prox_lambda);
                }
                break;
            case RegularizationType::MCP:
                for (Size j = 0; j < n_features_sz; ++j) {
                    coefficients[j] = detail::prox_mcp(coefficients[j], prox_lambda);
                }
                break;
            case RegularizationType::L2:
                // L2 doesn't induce sparsity, just shrinkage
                {
                    Real scale = Real(1.0) / (Real(1.0) + prox_lambda);
                    for (Size j = 0; j < n_features_sz; ++j) {
                        coefficients[j] *= scale;
                    }
                }
                break;
        }
        
        if (detail::check_convergence(coef_old, coefficients.ptr, n_features_sz, tol)) {
            break;
        }
    }
    
    scl::memory::aligned_free(residuals, SCL_ALIGNMENT);
    scl::memory::aligned_free(gradient, SCL_ALIGNMENT);
    scl::memory::aligned_free(coef_old, SCL_ALIGNMENT);
}

// =============================================================================
// FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
// =============================================================================

// Accelerated proximal gradient (FISTA) for Lasso
template <typename T, bool IsCSR>
void fista(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Real alpha,
    Array<Real> coefficients,
    Index max_iter,
    Real tol
) {
    const Index n_samples = X.rows();
    const Index n_features = X.cols();
    const Size n_samples_sz = static_cast<Size>(n_samples);
    const Size n_features_sz = static_cast<Size>(n_features);
    
    SCL_CHECK_DIM(y.len >= n_samples_sz, "y size mismatch");
    SCL_CHECK_DIM(coefficients.len >= n_features_sz, "coefficients size mismatch");
    
    scl::algo::zero(coefficients.ptr, n_features_sz);
    
    Real L = detail::estimate_lipschitz_constant(X);
    Real step_size = Real(1.0) / L;
    
    Real* residuals = scl::memory::aligned_alloc<Real>(n_samples_sz, SCL_ALIGNMENT);
    Real* gradient = scl::memory::aligned_alloc<Real>(n_features_sz, SCL_ALIGNMENT);
    Real* coef_old = scl::memory::aligned_alloc<Real>(n_features_sz, SCL_ALIGNMENT);
    Real* z = scl::memory::aligned_alloc<Real>(n_features_sz, SCL_ALIGNMENT);  // Momentum term
    
    scl::algo::zero(z, n_features_sz);
    
    Real t = Real(1.0);
    
    for (Index iter = 0; iter < max_iter; ++iter) {
        scl::algo::copy(coefficients.ptr, coef_old, n_features_sz);
        
        // Compute residuals: r = X * z - y
        detail::sparse_matvec(X, z, residuals, n_samples_sz);
        for (Size i = 0; i < n_samples_sz; ++i) {
            residuals[i] -= y[i];
        }
        
        // Compute gradient: grad = X^T * residuals
        scl::algo::zero(gradient, n_features_sz);
        if constexpr (!IsCSR) {
            for (Index j = 0; j < n_features; ++j) {
                gradient[j] = static_cast<Real>(detail::sparse_dot_dense(X, j, residuals));
            }
        } else {
            for (Index i = 0; i < n_samples; ++i) {
                const Index len = X.primary_length_unsafe(i);
                if (len == 0) continue;
                
                auto indices = X.primary_indices_unsafe(i);
                auto values = X.primary_values_unsafe(i);
                
                for (Index k = 0; k < len; ++k) {
                    gradient[indices[k]] += static_cast<Real>(values[k]) * residuals[i];
                }
            }
        }
        
        // Gradient step on z
        for (Size j = 0; j < n_features_sz; ++j) {
            coefficients[j] = z[j] - step_size * gradient[j];
        }
        
        // Proximal step (soft thresholding)
        prox_l1(coefficients, alpha * step_size);
        
        // FISTA momentum update
        Real t_new = (Real(1.0) + std::sqrt(Real(1.0) + Real(4.0) * t * t)) / Real(2.0);
        Real momentum = (t - Real(1.0)) / t_new;
        
        for (Size j = 0; j < n_features_sz; ++j) {
            z[j] = coefficients[j] + momentum * (coefficients[j] - coef_old[j]);
        }
        
        t = t_new;
        
        if (detail::check_convergence(coef_old, coefficients.ptr, n_features_sz, tol)) {
            break;
        }
    }
    
    scl::memory::aligned_free(residuals, SCL_ALIGNMENT);
    scl::memory::aligned_free(gradient, SCL_ALIGNMENT);
    scl::memory::aligned_free(coef_old, SCL_ALIGNMENT);
    scl::memory::aligned_free(z, SCL_ALIGNMENT);
}

// =============================================================================
// Iterative Hard Thresholding
// =============================================================================

// IHT for sparse recovery with fixed sparsity level
template <typename T, bool IsCSR>
void iht(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Index sparsity_level,
    Array<Real> coefficients,
    Index max_iter
) {
    const Index n_samples = X.rows();
    const Index n_features = X.cols();
    const Size n_samples_sz = static_cast<Size>(n_samples);
    const Size n_features_sz = static_cast<Size>(n_features);
    
    SCL_CHECK_DIM(y.len >= n_samples_sz, "y size mismatch");
    SCL_CHECK_DIM(coefficients.len >= n_features_sz, "coefficients size mismatch");
    SCL_CHECK_ARG(sparsity_level > 0 && sparsity_level <= n_features, 
                  "sparsity_level must be in (0, n_features]");
    
    scl::algo::zero(coefficients.ptr, n_features_sz);
    
    Real L = detail::estimate_lipschitz_constant(X);
    Real step_size = Real(1.0) / L;
    
    Real* residuals = scl::memory::aligned_alloc<Real>(n_samples_sz, SCL_ALIGNMENT);
    Real* gradient = scl::memory::aligned_alloc<Real>(n_features_sz, SCL_ALIGNMENT);
    Real* abs_coef = scl::memory::aligned_alloc<Real>(n_features_sz, SCL_ALIGNMENT);
    Index* sorted_idx = scl::memory::aligned_alloc<Index>(n_features_sz, SCL_ALIGNMENT);
    
    for (Index iter = 0; iter < max_iter; ++iter) {
        // Compute residuals: r = X * coef - y
        detail::sparse_matvec(X, coefficients.ptr, residuals, n_samples_sz);
        for (Size i = 0; i < n_samples_sz; ++i) {
            residuals[i] -= y[i];
        }
        
        // Compute gradient
        scl::algo::zero(gradient, n_features_sz);
        if constexpr (!IsCSR) {
            for (Index j = 0; j < n_features; ++j) {
                gradient[j] = static_cast<Real>(detail::sparse_dot_dense(X, j, residuals));
            }
        } else {
            for (Index i = 0; i < n_samples; ++i) {
                const Index len = X.primary_length_unsafe(i);
                if (len == 0) continue;
                
                auto indices = X.primary_indices_unsafe(i);
                auto values = X.primary_values_unsafe(i);
                
                for (Index k = 0; k < len; ++k) {
                    gradient[indices[k]] += static_cast<Real>(values[k]) * residuals[i];
                }
            }
        }
        
        // Gradient step
        for (Size j = 0; j < n_features_sz; ++j) {
            coefficients[j] -= step_size * gradient[j];
        }
        
        // Hard thresholding: keep only top-k by absolute value
        for (Size j = 0; j < n_features_sz; ++j) {
            abs_coef[j] = (coefficients[j] >= Real(0)) ? coefficients[j] : -coefficients[j];
            sorted_idx[j] = static_cast<Index>(j);
        }
        
        // Partial sort to find k-th largest
        const Size k = static_cast<Size>(sparsity_level);
        scl::algo::nth_element(abs_coef, abs_coef + k, abs_coef + n_features_sz);
        Real threshold = abs_coef[k];
        
        // Zero out elements below threshold
        for (Size j = 0; j < n_features_sz; ++j) {
            Real abs_val = (coefficients[j] >= Real(0)) ? coefficients[j] : -coefficients[j];
            if (abs_val < threshold) {
                coefficients[j] = Real(0);
            }
        }
    }
    
    scl::memory::aligned_free(residuals, SCL_ALIGNMENT);
    scl::memory::aligned_free(gradient, SCL_ALIGNMENT);
    scl::memory::aligned_free(abs_coef, SCL_ALIGNMENT);
    scl::memory::aligned_free(sorted_idx, SCL_ALIGNMENT);
}

// =============================================================================
// Regularization Path
// =============================================================================

// Compute Lasso solution path for multiple alpha values
template <typename T, bool IsCSR>
void lasso_path(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Array<const Real> alphas,
    Real* coefficient_paths,
    Index max_iter
) {
    const Index n_alphas = static_cast<Index>(alphas.len);
    const Index n_features = X.cols();
    const Size n_features_sz = static_cast<Size>(n_features);
    
    // Warm start from previous solution
    Real* current_coef = scl::memory::aligned_alloc<Real>(n_features_sz, SCL_ALIGNMENT);
    scl::algo::zero(current_coef, n_features_sz);
    
    for (Index a = 0; a < n_alphas; ++a) {
        // Use previous solution as warm start
        Array<Real> coef_out(coefficient_paths + a * n_features_sz, n_features_sz);
        scl::algo::copy(current_coef, coef_out.ptr, n_features_sz);
        
        lasso_coordinate_descent(X, y, alphas[a], coef_out, max_iter, config::DEFAULT_TOL);
        
        // Save for warm start
        scl::algo::copy(coef_out.ptr, current_coef, n_features_sz);
    }
    
    scl::memory::aligned_free(current_coef, SCL_ALIGNMENT);
}

// =============================================================================
// Group Lasso
// =============================================================================

// Group Lasso via block coordinate descent
template <typename T, bool IsCSR>
void group_lasso(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    const Index* group_indices,
    const Index* group_offsets,
    Index n_groups,
    Real alpha,
    Array<Real> coefficients,
    Index max_iter
) {
    const Index n_samples = X.rows();
    const Index n_features = X.cols();
    const Size n_samples_sz = static_cast<Size>(n_samples);
    const Size n_features_sz = static_cast<Size>(n_features);
    
    SCL_CHECK_DIM(y.len >= n_samples_sz, "y size mismatch");
    SCL_CHECK_DIM(coefficients.len >= n_features_sz, "coefficients size mismatch");
    
    scl::algo::zero(coefficients.ptr, n_features_sz);
    
    Real* residuals = scl::memory::aligned_alloc<Real>(n_samples_sz, SCL_ALIGNMENT);
    scl::algo::copy(y.ptr, residuals, n_samples_sz);
    
    Real* coef_old = scl::memory::aligned_alloc<Real>(n_features_sz, SCL_ALIGNMENT);
    
    for (Index iter = 0; iter < max_iter; ++iter) {
        scl::algo::copy(coefficients.ptr, coef_old, n_features_sz);
        
        for (Index g = 0; g < n_groups; ++g) {
            const Index group_start = group_offsets[g];
            const Index group_end = group_offsets[g + 1];
            const Size group_size = static_cast<Size>(group_end - group_start);
            
            if (group_size == 0) continue;
            
            // Compute group gradient
            Real* group_grad = scl::memory::aligned_alloc<Real>(group_size, SCL_ALIGNMENT);
            Real* group_coef = scl::memory::aligned_alloc<Real>(group_size, SCL_ALIGNMENT);
            
            for (Size j = 0; j < group_size; ++j) {
                Index feat_idx = group_indices[group_start + j];
                group_coef[j] = coefficients[feat_idx];
                
                Real rho = static_cast<Real>(detail::sparse_dot_dense(X, feat_idx, residuals));
                Real norm_sq = static_cast<Real>(detail::column_squared_norm(X, feat_idx));
                group_grad[j] = rho + norm_sq * group_coef[j];
            }
            
            // Group L2 norm of gradient
            Real grad_norm = std::sqrt(scl::vectorize::sum_squared(
                Array<const Real>(group_grad, group_size)));
            
            // Group soft thresholding
            Real sqrt_group_size = std::sqrt(static_cast<Real>(group_size));
            Real threshold = alpha * sqrt_group_size;
            
            if (grad_norm > threshold) {
                Real scale = (grad_norm - threshold) / grad_norm;
                
                for (Size j = 0; j < group_size; ++j) {
                    Index feat_idx = group_indices[group_start + j];
                    Real norm_sq = static_cast<Real>(detail::column_squared_norm(X, feat_idx));
                    Real new_coef = (scale * group_grad[j]) / scl::algo::max2(norm_sq, config::EPS);
                    
                    Real delta = new_coef - coefficients[feat_idx];
                    if (delta != Real(0)) {
                        detail::update_residuals_column(X, feat_idx, static_cast<T>(delta), residuals);
                        coefficients[feat_idx] = new_coef;
                    }
                }
            } else {
                // Zero out entire group
                for (Size j = 0; j < group_size; ++j) {
                    Index feat_idx = group_indices[group_start + j];
                    Real delta = -coefficients[feat_idx];
                    if (delta != Real(0)) {
                        detail::update_residuals_column(X, feat_idx, static_cast<T>(delta), residuals);
                        coefficients[feat_idx] = Real(0);
                    }
                }
            }
            
            scl::memory::aligned_free(group_grad, SCL_ALIGNMENT);
            scl::memory::aligned_free(group_coef, SCL_ALIGNMENT);
        }
        
        if (detail::check_convergence(coef_old, coefficients.ptr, n_features_sz, config::DEFAULT_TOL)) {
            break;
        }
    }
    
    scl::memory::aligned_free(residuals, SCL_ALIGNMENT);
    scl::memory::aligned_free(coef_old, SCL_ALIGNMENT);
}

// =============================================================================
// Sparse Logistic Regression
// =============================================================================

// L1-regularized logistic regression via coordinate descent
template <typename T, bool IsCSR>
void sparse_logistic_regression(
    const Sparse<T, IsCSR>& X,
    Array<const Index> y_binary,
    Real alpha,
    Array<Real> coefficients,
    Index max_iter
) {
    const Index n_samples = X.rows();
    const Index n_features = X.cols();
    const Size n_samples_sz = static_cast<Size>(n_samples);
    const Size n_features_sz = static_cast<Size>(n_features);
    
    SCL_CHECK_DIM(y_binary.len >= n_samples_sz, "y size mismatch");
    SCL_CHECK_DIM(coefficients.len >= n_features_sz, "coefficients size mismatch");
    
    scl::algo::zero(coefficients.ptr, n_features_sz);
    
    // Intercept term (not regularized)
    Real intercept = Real(0);
    
    // Working response and weights
    Real* linear_pred = scl::memory::aligned_alloc<Real>(n_samples_sz, SCL_ALIGNMENT);
    Real* prob = scl::memory::aligned_alloc<Real>(n_samples_sz, SCL_ALIGNMENT);
    Real* weights = scl::memory::aligned_alloc<Real>(n_samples_sz, SCL_ALIGNMENT);
    Real* working_response = scl::memory::aligned_alloc<Real>(n_samples_sz, SCL_ALIGNMENT);
    Real* coef_old = scl::memory::aligned_alloc<Real>(n_features_sz, SCL_ALIGNMENT);
    
    scl::algo::zero(linear_pred, n_samples_sz);
    
    for (Index iter = 0; iter < max_iter; ++iter) {
        scl::algo::copy(coefficients.ptr, coef_old, n_features_sz);
        
        // Compute probabilities and working quantities
        for (Size i = 0; i < n_samples_sz; ++i) {
            // Sigmoid function with numerical stability
            Real z = linear_pred[i] + intercept;
            if (z > Real(20)) {
                prob[i] = Real(1.0);
            } else if (z < Real(-20)) {
                prob[i] = Real(0.0);
            } else {
                prob[i] = Real(1.0) / (Real(1.0) + std::exp(-z));
            }
            
            // Weight = p(1-p), clipped for stability
            weights[i] = scl::algo::max2(prob[i] * (Real(1.0) - prob[i]), config::EPS);
            
            // Working response
            Real y_val = static_cast<Real>(y_binary[i]);
            working_response[i] = z + (y_val - prob[i]) / weights[i];
        }
        
        // Update intercept (unpenalized)
        Real sum_w = Real(0), sum_wz = Real(0);
        for (Size i = 0; i < n_samples_sz; ++i) {
            sum_w += weights[i];
            sum_wz += weights[i] * working_response[i];
        }
        intercept = sum_wz / scl::algo::max2(sum_w, config::EPS);
        
        // Coordinate descent on features
        for (Index j = 0; j < n_features; ++j) {
            // Weighted column norm
            Real weighted_norm_sq = Real(0);
            Real rho = Real(0);
            
            if constexpr (!IsCSR) {
                const Index len = X.primary_length_unsafe(j);
                if (len == 0) continue;
                
                auto indices = X.primary_indices_unsafe(j);
                auto values = X.primary_values_unsafe(j);
                
                for (Index k = 0; k < len; ++k) {
                    Index i = indices[k];
                    Real x_ij = static_cast<Real>(values[k]);
                    weighted_norm_sq += weights[i] * x_ij * x_ij;
                    rho += weights[i] * x_ij * (working_response[i] - intercept);
                }
            } else {
                for (Index i = 0; i < n_samples; ++i) {
                    const Index len = X.primary_length_unsafe(i);
                    if (len == 0) continue;
                    
                    auto indices = X.primary_indices_unsafe(i);
                    auto values = X.primary_values_unsafe(i);
                    
                    const Index* pos = scl::algo::lower_bound(
                        indices.ptr, indices.ptr + len, j);
                    
                    if (pos != indices.ptr + len && *pos == j) {
                        Index k = static_cast<Index>(pos - indices.ptr);
                        Real x_ij = static_cast<Real>(values[k]);
                        weighted_norm_sq += weights[i] * x_ij * x_ij;
                        rho += weights[i] * x_ij * (working_response[i] - intercept);
                    }
                }
            }
            
            if (weighted_norm_sq < config::EPS) continue;
            
            rho += weighted_norm_sq * coefficients[j];
            
            // Soft thresholding
            Real coef_new = detail::soft_threshold(rho, alpha) / weighted_norm_sq;
            Real delta = coef_new - coefficients[j];
            
            if (delta != Real(0)) {
                // Update linear predictor
                if constexpr (!IsCSR) {
                    const Index len = X.primary_length_unsafe(j);
                    auto indices = X.primary_indices_unsafe(j);
                    auto values = X.primary_values_unsafe(j);
                    
                    for (Index k = 0; k < len; ++k) {
                        linear_pred[indices[k]] += delta * static_cast<Real>(values[k]);
                    }
                } else {
                    for (Index i = 0; i < n_samples; ++i) {
                        const Index len = X.primary_length_unsafe(i);
                        if (len == 0) continue;
                        
                        auto indices = X.primary_indices_unsafe(i);
                        auto values = X.primary_values_unsafe(i);
                        
                        const Index* pos = scl::algo::lower_bound(
                            indices.ptr, indices.ptr + len, j);
                        
                        if (pos != indices.ptr + len && *pos == j) {
                            Index k = static_cast<Index>(pos - indices.ptr);
                            linear_pred[i] += delta * static_cast<Real>(values[k]);
                        }
                    }
                }
                
                coefficients[j] = coef_new;
            }
        }
        
        if (detail::check_convergence(coef_old, coefficients.ptr, n_features_sz, config::DEFAULT_TOL)) {
            break;
        }
    }
    
    scl::memory::aligned_free(linear_pred, SCL_ALIGNMENT);
    scl::memory::aligned_free(prob, SCL_ALIGNMENT);
    scl::memory::aligned_free(weights, SCL_ALIGNMENT);
    scl::memory::aligned_free(working_response, SCL_ALIGNMENT);
    scl::memory::aligned_free(coef_old, SCL_ALIGNMENT);
}

} // namespace scl::kernel::sparse_opt

