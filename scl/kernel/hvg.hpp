#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/argsort.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/feature.hpp"

#include <cmath>
#include <vector>

// =============================================================================
/// @file hvg.hpp
/// @brief Highly Variable Genes (HVG) Selection
///
/// Implements standardized HVG workflows:
/// 1. Seurat V3 (vst): Robust variance with clipping
/// 2. Dispersion-based: Fano factor selection
/// 3. Top-K selection by variance
///
/// Architecture: Wraps feature.hpp functions + ranking logic
// =============================================================================

namespace scl::kernel::hvg {

// =============================================================================
// Helper: Top-K Selection
// =============================================================================

/// @brief Select top K indices by descending scores
///
/// @param scores Input scores
/// @param k Number of top elements
/// @param out_indices Output indices [size >= k]
/// @param out_mask Output binary mask [size = n]
SCL_FORCE_INLINE void select_top_k(
    Array<const Real> scores,
    Size k,
    Array<Index> out_indices,
    Array<uint8_t> out_mask
) {
    const Size n = scores.size();
    
    SCL_CHECK_ARG(k <= n, "HVG: k exceeds number of elements");
    SCL_CHECK_DIM(out_indices.size() >= k, "HVG: Output indices too small");
    SCL_CHECK_DIM(out_mask.size() == n, "HVG: Output mask size mismatch");
    
    std::vector<Index> indices(n);
    std::vector<Real> score_copy(scores.ptr, scores.ptr + n);
    
    // Argsort descending
    scl::sort::argsort_inplace_descending(
        Array<Real>(score_copy.data(), n),
        Array<Index>(indices.data(), n)
    );
    
    // Zero mask
    for (Size i = 0; i < n; ++i) {
        out_mask[i] = 0;
    }
    
    // Mark top-K
    for (Size i = 0; i < k; ++i) {
        Index idx = indices[i];
        out_indices[i] = idx;
        out_mask[idx] = 1;
    }
}

// =============================================================================
// Method 1: Dispersion-Based Selection
// =============================================================================

/// @brief Select HVGs by dispersion (variance/mean)
///
/// @param matrix Input sparse matrix
/// @param n_top Number of HVGs to select
/// @param out_indices Output indices [size >= n_top]
/// @param out_mask Output mask [size = primary_dim]
/// @param out_dispersions Output dispersion values [size = primary_dim]
template <typename MatrixT>
    requires AnySparse<MatrixT>
void select_by_dispersion(
    const MatrixT& matrix,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask,
    Array<Real> out_dispersions
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(out_dispersions.size() == static_cast<Size>(primary_dim),
                  "HVG: Dispersions size mismatch");
    
    // Compute mean and variance
    std::vector<Real> means(primary_dim);
    std::vector<Real> vars(primary_dim);
    
    scl::kernel::feature::standard_moments(
        matrix,
        Array<Real>(means.data(), means.size()),
        Array<Real>(vars.data(), vars.size()),
        1
    );
    
    // Compute dispersion
    scl::kernel::feature::dispersion(
        Array<const Real>(means.data(), means.size()),
        Array<const Real>(vars.data(), vars.size()),
        out_dispersions
    );
    
    // Select top-K
    select_top_k(
        Array<const Real>(out_dispersions.ptr, out_dispersions.size()),
        n_top,
        out_indices,
        out_mask
    );
}

// =============================================================================
// Method 2: Seurat V3 (VST) - Clipped Variance
// =============================================================================

/// @brief Select HVGs using Seurat V3 method (variance stabilization)
///
/// @param matrix Input sparse matrix
/// @param clip_vals Clipping thresholds [size = primary_dim]
/// @param n_top Number of HVGs to select
/// @param out_indices Output indices [size >= n_top]
/// @param out_mask Output mask [size = primary_dim]
/// @param out_variances Output clipped variances [size = primary_dim]
template <typename MatrixT>
    requires AnySparse<MatrixT>
void select_by_vst(
    const MatrixT& matrix,
    Array<const Real> clip_vals,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask,
    Array<Real> out_variances
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(clip_vals.size() == static_cast<Size>(primary_dim),
                  "HVG: Clip vals size mismatch");
    SCL_CHECK_DIM(out_variances.size() == static_cast<Size>(primary_dim),
                  "HVG: Variances size mismatch");
    
    // Compute clipped moments
    std::vector<Real> means(primary_dim);
    
    scl::kernel::feature::clipped_moments(
        matrix,
        clip_vals,
        Array<Real>(means.data(), means.size()),
        out_variances
    );
    
    // Select top-K by variance
    select_top_k(
        Array<const Real>(out_variances.ptr, out_variances.size()),
        n_top,
        out_indices,
        out_mask
    );
}

// =============================================================================
// Method 3: Simple Variance-Based Selection
// =============================================================================

/// @brief Select HVGs by raw variance (simple method)
///
/// @param matrix Input sparse matrix
/// @param n_top Number of HVGs to select
/// @param out_indices Output indices [size >= n_top]
/// @param out_mask Output mask [size = primary_dim]
template <typename MatrixT>
    requires AnySparse<MatrixT>
void select_by_variance(
    const MatrixT& matrix,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    std::vector<Real> means(primary_dim);
    std::vector<Real> vars(primary_dim);
    
    scl::kernel::feature::standard_moments(
        matrix,
        Array<Real>(means.data(), means.size()),
        Array<Real>(vars.data(), vars.size()),
        1
    );
    
    select_top_k(
        Array<const Real>(vars.data(), vars.size()),
        n_top,
        out_indices,
        out_mask
    );
}

} // namespace scl::kernel::hvg
