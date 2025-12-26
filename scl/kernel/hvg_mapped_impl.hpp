#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/argsort.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"
#include "scl/kernel/feature_mapped_impl.hpp"

// =============================================================================
/// @file hvg_mapped_impl.hpp
/// @brief HVG Selection for Mapped Sparse Matrices
///
/// HVG is primarily sort-bound (O(n log n)), not memory-bound.
/// For Mapped matrices, we:
/// 1. Use feature_mapped_impl for statistics computation
/// 2. Apply VQSort for ranking
///
/// Operations:
/// - compute_hvg_stats_mapped: Compute mean/variance for HVG
// =============================================================================

namespace scl::kernel::hvg::mapped {

// =============================================================================
// HVG Statistics - Uses feature_mapped_impl
// =============================================================================

/// @brief Compute HVG statistics from mapped matrix (MappedCustomSparse)
///
/// This wraps feature_mapped_impl to compute mean and variance
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_hvg_stats_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_variances
) {
    // Use feature statistics from feature_mapped_impl
    scl::kernel::feature::mapped::compute_feature_stats_mapped(
        matrix, out_means, out_variances
    );
}

/// @brief Compute HVG statistics from mapped matrix (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_hvg_stats_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_variances
) {
    // Use feature statistics from feature_mapped_impl
    scl::kernel::feature::mapped::compute_feature_stats_mapped(
        matrix, out_means, out_variances
    );
}

// =============================================================================
// Normalized Dispersion Computation
// =============================================================================

/// @brief Compute normalized dispersion for HVG ranking
///
/// dispersion = variance / mean (coefficient of variation squared)
/// normalized_dispersion = (dispersion - mean(dispersion)) / std(dispersion)
void compute_normalized_dispersion(
    Array<const Real> means,
    Array<const Real> variances,
    Array<Real> out_dispersions,
    Real min_mean = 0.0125,
    Real max_mean = 3.0,
    Real min_disp = 0.5
) {
    const Size n_features = means.len;

    SCL_CHECK_DIM(variances.len == n_features, "Variances size mismatch");
    SCL_CHECK_DIM(out_dispersions.len == n_features, "Dispersions size mismatch");

    // Compute raw dispersion (variance / mean)
    for (Size i = 0; i < n_features; ++i) {
        Real mean = means[i];
        Real var = variances[i];

        if (mean > 0 && mean >= min_mean && mean <= max_mean) {
            out_dispersions[i] = var / mean;
        } else {
            out_dispersions[i] = 0.0;  // Will be filtered out
        }
    }

    // Compute mean and std of dispersions for normalization
    Real disp_sum = 0.0;
    Real disp_sum_sq = 0.0;
    Size valid_count = 0;

    for (Size i = 0; i < n_features; ++i) {
        Real d = out_dispersions[i];
        if (d > 0) {
            disp_sum += d;
            disp_sum_sq += d * d;
            valid_count++;
        }
    }

    if (valid_count == 0) return;

    Real disp_mean = disp_sum / static_cast<Real>(valid_count);
    Real disp_var = (disp_sum_sq / static_cast<Real>(valid_count)) - (disp_mean * disp_mean);
    Real disp_std = (disp_var > 0) ? std::sqrt(disp_var) : 1.0;

    // Normalize dispersions
    for (Size i = 0; i < n_features; ++i) {
        Real d = out_dispersions[i];
        if (d > 0) {
            out_dispersions[i] = (d - disp_mean) / disp_std;
        } else {
            out_dispersions[i] = -std::numeric_limits<Real>::infinity();
        }
    }
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void compute_hvg_stats_mapped_dispatch(
    const MatrixT& matrix,
    Array<Real> out_means,
    Array<Real> out_variances
) {
    compute_hvg_stats_mapped(matrix, out_means, out_variances);
}

} // namespace scl::kernel::hvg::mapped

