#pragma once

// =============================================================================
// SCL Core - Numerical Precision Comparison Utilities
// =============================================================================
//
// Provides robust numerical comparison with configurable tolerances.
//
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <cmath>
#include <algorithm>
#include <limits>

namespace scl::test::precision {

// =============================================================================
// Tolerance Presets
// =============================================================================

struct Tolerance {
    double rtol;  // Relative tolerance
    double atol;  // Absolute tolerance
    
    // Presets for different precision requirements
    static constexpr Tolerance strict() { return {1e-12, 1e-15}; }
    static constexpr Tolerance normal() { return {1e-9, 1e-12}; }
    static constexpr Tolerance relaxed() { return {1e-6, 1e-9}; }
    static constexpr Tolerance loose() { return {1e-3, 1e-6}; }
    
    // Algorithm-specific presets
    static constexpr Tolerance iterative() { return {1e-6, 1e-9}; }      // For iterative solvers
    static constexpr Tolerance statistical() { return {1e-4, 1e-6}; }    // For statistical methods
    static constexpr Tolerance approximate() { return {1e-2, 1e-4}; }    // For approximation algorithms
};

// =============================================================================
// Scalar Comparison
// =============================================================================

/// Check if two scalars are approximately equal
inline bool approx_equal(
    scl_real_t a,
    scl_real_t b,
    const Tolerance& tol = Tolerance::normal()
) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) && std::isnan(b);
    }
    
    if (std::isinf(a) || std::isinf(b)) {
        return a == b;  // Both must be same infinity
    }
    
    double diff = std::abs(a - b);
    double max_val = std::max(std::abs(a), std::abs(b));
    
    return diff <= tol.atol + tol.rtol * max_val;
}

/// Check if scalar is approximately zero
inline bool approx_zero(
    scl_real_t value,
    double atol = 1e-12
) {
    return std::abs(value) <= atol;
}

// =============================================================================
// Vector Comparison
// =============================================================================

/// Check if two vectors are approximately equal
template<typename VecA, typename VecB>
inline bool vectors_equal(
    const VecA& a,
    const VecB& b,
    const Tolerance& tol = Tolerance::normal()
) {
    if (a.size() != b.size()) {
        return false;
    }
    
    double max_val = 0;
    double max_diff = 0;
    
    for (size_t i = 0; i < a.size(); ++i) {
        max_val = std::max(max_val, std::max(std::abs(a[i]), std::abs(b[i])));
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    
    return max_diff <= tol.atol + tol.rtol * max_val;
}

/// Compute relative error between vectors
template<typename VecA, typename VecB>
inline double relative_error(const VecA& a, const VecB& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<double>::infinity();
    }
    
    double max_val = 0;
    double max_diff = 0;
    
    for (size_t i = 0; i < a.size(); ++i) {
        max_val = std::max(max_val, std::max(std::abs(a[i]), std::abs(b[i])));
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    
    if (max_val < 1e-15) {
        return max_diff;  // Both near zero
    }
    
    return max_diff / max_val;
}

// =============================================================================
// Matrix Comparison
// =============================================================================

/// Check if two dense matrices are approximately equal (Eigen)
template<typename MatA, typename MatB>
inline bool matrices_equal(
    const Eigen::MatrixBase<MatA>& A,
    const Eigen::MatrixBase<MatB>& B,
    const Tolerance& tol = Tolerance::normal()
) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        return false;
    }
    
    double max_val = std::max(A.cwiseAbs().maxCoeff(), B.cwiseAbs().maxCoeff());
    double max_diff = (A - B).cwiseAbs().maxCoeff();
    
    return max_diff <= tol.atol + tol.rtol * max_val;
}

/// Compute relative error between matrices
template<typename MatA, typename MatB>
inline double relative_error(
    const Eigen::MatrixBase<MatA>& A,
    const Eigen::MatrixBase<MatB>& B
) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        return std::numeric_limits<double>::infinity();
    }
    
    double max_val = std::max(A.cwiseAbs().maxCoeff(), B.cwiseAbs().maxCoeff());
    double max_diff = (A - B).cwiseAbs().maxCoeff();
    
    if (max_val < 1e-15) {
        return max_diff;
    }
    
    return max_diff / max_val;
}

// =============================================================================
// Sparse Matrix Comparison
// =============================================================================

/// Check if two sparse matrices are approximately equal
template<typename Scalar, int Options, typename Index>
inline bool sparse_matrices_equal(
    const Eigen::SparseMatrix<Scalar, Options, Index>& A,
    const Eigen::SparseMatrix<Scalar, Options, Index>& B,
    const Tolerance& tol = Tolerance::normal()
) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        return false;
    }
    
    if (A.nonZeros() != B.nonZeros()) {
        return false;
    }
    
    auto diff = A - B;
    
    double max_val = 0;
    for (int k = 0; k < A.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<Scalar, Options, Index>::InnerIterator it(A, k); it; ++it) {
            max_val = std::max(max_val, std::abs(it.value()));
        }
        for (typename Eigen::SparseMatrix<Scalar, Options, Index>::InnerIterator it(B, k); it; ++it) {
            max_val = std::max(max_val, std::abs(it.value()));
        }
    }
    
    double max_diff = 0;
    for (int k = 0; k < diff.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<Scalar, Options, Index>::InnerIterator it(diff, k); it; ++it) {
            max_diff = std::max(max_diff, std::abs(it.value()));
        }
    }
    
    return max_diff <= tol.atol + tol.rtol * max_val;
}

// =============================================================================
// Statistical Comparison (for Monte Carlo tests)
// =============================================================================

struct Statistics {
    double mean;
    double stddev;
    double min;
    double max;
    size_t count;
};

/// Compute statistics for a sequence of values
template<typename Container>
inline Statistics compute_statistics(const Container& values) {
    Statistics stats;
    stats.count = values.size();
    
    if (values.empty()) {
        stats.mean = stats.stddev = stats.min = stats.max = 0;
        return stats;
    }
    
    // Mean
    double sum = 0;
    for (auto v : values) {
        sum += v;
    }
    stats.mean = sum / values.size();
    
    // Min/Max
    stats.min = *std::min_element(values.begin(), values.end());
    stats.max = *std::max_element(values.begin(), values.end());
    
    // Stddev
    double var = 0;
    for (auto v : values) {
        double diff = v - stats.mean;
        var += diff * diff;
    }
    stats.stddev = std::sqrt(var / values.size());
    
    return stats;
}

/// Check if error statistics are within acceptable range
inline bool error_stats_acceptable(
    const Statistics& error_stats,
    const Tolerance& tol = Tolerance::normal()
) {
    // Mean error should be small
    if (error_stats.mean > tol.atol + tol.rtol) {
        return false;
    }
    
    // Max error should be reasonable (allow some outliers)
    if (error_stats.max > tol.atol + tol.rtol * 10) {
        return false;
    }
    
    return true;
}

} // namespace scl::test::precision

