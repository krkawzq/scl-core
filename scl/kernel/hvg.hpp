#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/argsort.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/feature.hpp"

#include <cmath>
#include <algorithm>
#include <vector>

// =============================================================================
/// @file hvg.hpp
/// @brief Highly Variable Genes (HVG) Selection Kernel
///
/// Implements standardized HVG selection workflows:
/// 1. Seurat V3 ("vst"): Robust variance with clipping
/// 2. CellRanger/Scanpy ("seurat"): Dispersion-based
/// 3. Mean-Variance Plot: Log-space fitting
///
/// Architecture:
///
/// This module wraps scl::kernel::feature functions and adds:
/// - Ranking logic
/// - Top-K selection
/// - Variance stabilization
/// - Binning strategies
///
/// Use Cases:
///
/// - Feature selection for dimensionality reduction (PCA, UMAP)
/// - Marker gene filtering
/// - Quality control (remove low-variance genes)
///
/// Performance:
///
/// - Complexity: O(n_genes log n_genes) dominated by sorting
/// - Parallelism: Statistics computation is parallel, sorting is serial
/// - Memory: O(n_genes) for intermediate arrays
// =============================================================================

namespace scl::kernel::hvg {

// =============================================================================
// Helper: Top-K Selection
// =============================================================================

/// @brief Select top K indices by descending order of scores.
///
/// @param scores Input scores [size = n]
/// @param k Number of top elements to select
/// @param out_indices Output: Indices of top-K elements [size >= k]
/// @param out_mask Output: Binary mask (1=selected, 0=not) [size = n]
SCL_FORCE_INLINE void select_top_k(
    Span<const Real> scores,
    Size k,
    MutableSpan<Index> out_indices,
    MutableSpan<uint8_t> out_mask
) {
    const Size n = scores.size;
    
    SCL_CHECK_ARG(k <= n, "HVG: k exceeds number of genes");
    SCL_CHECK_DIM(out_indices.size >= k, "HVG: Output indices too small");
    SCL_CHECK_DIM(out_mask.size == n, "HVG: Output mask size mismatch");
    
    // Create index array and score copy for VQSort-based argsort
    std::vector<Index> indices(n);
    std::vector<Real> score_copy(scores.ptr, scores.ptr + n);
    
    // Full argsort descending using VQSort (SIMD optimized)
    // Note: For top-K, full sort is still very fast with VQSort
    // and simpler than maintaining partial sort semantics
    scl::sort::argsort_inplace_descending(
        MutableSpan<Real>(score_copy.data(), n),
        MutableSpan<Index>(indices.data(), n)
    );
    
    // Zero out mask
    for (Size i = 0; i < n; ++i) {
        out_mask[i] = 0;
    }
    
    // Copy top-K and mark mask
    for (Size i = 0; i < k; ++i) {
        Index idx = indices[i];
        out_indices[i] = idx;
        out_mask[idx] = 1;
    }
}

// =============================================================================
// Method 1: Dispersion-Based (CellRanger/Seurat)
// =============================================================================

/// @brief Select HVGs using dispersion (variance/mean) ranking (CSC version).
///
/// Algorithm:
/// 1. Compute mean and variance per gene
/// 2. Compute dispersion = variance / mean
/// 3. Select top-K genes by dispersion
///
/// @tparam MatrixT Any CSC-like matrix type
/// @param matrix Input CSC-like matrix (cells x genes)
/// @param n_top Number of HVGs to select
/// @param out_indices Output: Indices of selected genes [size >= n_top]
/// @param out_mask Output: Binary mask [size = n_genes]
/// @param out_dispersions Output: Dispersion values [size = n_genes]
template <CSCLike MatrixT>
void select_by_dispersion(
    const MatrixT& matrix,
    Size n_top,
    MutableSpan<Index> out_indices,
    MutableSpan<uint8_t> out_mask,
    MutableSpan<Real> out_dispersions
) {
    const Index n_genes = matrix.cols;
    
    SCL_CHECK_DIM(out_dispersions.size == static_cast<Size>(n_genes),
                  "HVG: Dispersions size mismatch");
    
    // Compute statistics
    std::vector<Real> means(n_genes);
    std::vector<Real> vars(n_genes);
    
    scl::kernel::feature::standard_moments(
        matrix,
        {means.data(), static_cast<Size>(n_genes)},
        {vars.data(), static_cast<Size>(n_genes)}
    );
    
    // Compute dispersion
    scl::kernel::feature::dispersion(
        {means.data(), static_cast<Size>(n_genes)},
        {vars.data(), static_cast<Size>(n_genes)},
        out_dispersions
    );
    
    // Select top-K
    select_top_k(
        Span<const Real>(out_dispersions.ptr, out_dispersions.size), 
        n_top, 
        out_indices, 
        out_mask
    );
}

/// @brief Select HVGs using dispersion ranking (CSR version).
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param matrix Input CSR-like matrix (samples x features)
/// @param n_top Number of highly variable samples to select
/// @param out_indices Output: Indices of selected samples [size >= n_top]
/// @param out_mask Output: Binary mask [size = n_samples]
/// @param out_dispersions Output: Dispersion values [size = n_samples]
template <CSRLike MatrixT>
void select_by_dispersion(
    const MatrixT& matrix,
    Size n_top,
    MutableSpan<Index> out_indices,
    MutableSpan<uint8_t> out_mask,
    MutableSpan<Real> out_dispersions
) {
    const Index n_samples = matrix.rows;
    
    SCL_CHECK_DIM(out_dispersions.size == static_cast<Size>(n_samples),
                  "HVG: Dispersions size mismatch");
    
    // Compute statistics
    std::vector<Real> means(n_samples);
    std::vector<Real> vars(n_samples);
    
    scl::kernel::feature::standard_moments(
        matrix,
        {means.data(), static_cast<Size>(n_samples)},
        {vars.data(), static_cast<Size>(n_samples)}
    );
    
    // Compute dispersion
    scl::kernel::feature::dispersion(
        {means.data(), static_cast<Size>(n_samples)},
        {vars.data(), static_cast<Size>(n_samples)},
        out_dispersions
    );
    
    // Select top-K
    select_top_k(
        Span<const Real>(out_dispersions.ptr, out_dispersions.size), 
        n_top, 
        out_indices, 
        out_mask
    );
}

// =============================================================================
// Method 2: Simple Variance-Based
// =============================================================================

/// @brief Select HVGs by raw variance ranking (CSC version).
///
/// Simpler but less robust than dispersion-based methods.
///
/// @tparam MatrixT Any CSC-like matrix type
/// @param matrix Input CSC-like matrix (cells x genes)
/// @param n_top Number of HVGs to select
/// @param out_indices Output: Indices of selected genes [size >= n_top]
/// @param out_mask Output: Binary mask [size = n_genes]
template <CSCLike MatrixT>
void select_by_variance(
    const MatrixT& matrix,
    Size n_top,
    MutableSpan<Index> out_indices,
    MutableSpan<uint8_t> out_mask
) {
    const Index n_genes = matrix.cols;
    
    // Compute statistics
    std::vector<Real> means(n_genes);
    std::vector<Real> vars(n_genes);
    
    scl::kernel::feature::standard_moments(
        matrix,
        {means.data(), static_cast<Size>(n_genes)},
        {vars.data(), static_cast<Size>(n_genes)}
    );
    
    // Select top-K by variance
    select_top_k(
        Span<const Real>(vars.data(), static_cast<Size>(n_genes)), 
        n_top, 
        out_indices, 
        out_mask
    );
}

/// @brief Select highly variable samples by raw variance (CSR version).
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param matrix Input CSR-like matrix (samples x features)
/// @param n_top Number of highly variable samples to select
/// @param out_indices Output: Indices of selected samples [size >= n_top]
/// @param out_mask Output: Binary mask [size = n_samples]
template <CSRLike MatrixT>
void select_by_variance(
    const MatrixT& matrix,
    Size n_top,
    MutableSpan<Index> out_indices,
    MutableSpan<uint8_t> out_mask
) {
    const Index n_samples = matrix.rows;
    
    // Compute statistics
    std::vector<Real> means(n_samples);
    std::vector<Real> vars(n_samples);
    
    scl::kernel::feature::standard_moments(
        matrix,
        {means.data(), static_cast<Size>(n_samples)},
        {vars.data(), static_cast<Size>(n_samples)}
    );
    
    // Select top-K by variance
    select_top_k(
        Span<const Real>(vars.data(), static_cast<Size>(n_samples)), 
        n_top, 
        out_indices, 
        out_mask
    );
}

// =============================================================================
// Method 3: Detection Rate-Based
// =============================================================================

/// @brief Select genes by detection rate (fraction of expressing cells).
///
/// Useful for filtering lowly expressed genes.
///
/// @tparam MatrixT Any CSC-like matrix type
/// @param matrix Input CSC-like matrix (cells x genes)
/// @param min_cells Minimum fraction of cells (0.0 to 1.0)
/// @param out_mask Output: Binary mask [size = n_genes]
/// @return Number of selected genes
template <CSCLike MatrixT>
Size filter_by_detection_rate(
    const MatrixT& matrix,
    Real min_cells,
    MutableSpan<uint8_t> out_mask
) {
    const Index n_genes = matrix.cols;
    
    SCL_CHECK_ARG(min_cells >= 0.0 && min_cells <= 1.0,
                  "HVG: min_cells must be in [0, 1]");
    SCL_CHECK_DIM(out_mask.size == static_cast<Size>(n_genes),
                  "HVG: Output mask size mismatch");
    
    // Compute detection rates
    std::vector<Real> rates(n_genes);
    scl::kernel::feature::detection_rate(
        matrix,
        {rates.data(), static_cast<Size>(n_genes)}
    );
    
    // Apply threshold
    Size count = 0;
    for (Index j = 0; j < n_genes; ++j) {
        if (rates[j] >= min_cells) {
            out_mask[j] = 1;
            count++;
        } else {
            out_mask[j] = 0;
        }
    }
    
    return count;
}

/// @brief Filter samples by detection rate (CSR version).
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param matrix Input CSR-like matrix (samples x features)
/// @param min_features Minimum fraction of features (0.0 to 1.0)
/// @param out_mask Output: Binary mask [size = n_samples]
/// @return Number of selected samples
template <CSRLike MatrixT>
Size filter_by_detection_rate(
    const MatrixT& matrix,
    Real min_features,
    MutableSpan<uint8_t> out_mask
) {
    const Index n_samples = matrix.rows;
    
    SCL_CHECK_ARG(min_features >= 0.0 && min_features <= 1.0,
                  "HVG: min_features must be in [0, 1]");
    SCL_CHECK_DIM(out_mask.size == static_cast<Size>(n_samples),
                  "HVG: Output mask size mismatch");
    
    // Compute detection rates
    std::vector<Real> rates(n_samples);
    scl::kernel::feature::detection_rate(
        matrix,
        {rates.data(), static_cast<Size>(n_samples)}
    );
    
    // Apply threshold
    Size count = 0;
    for (Index i = 0; i < n_samples; ++i) {
        if (rates[i] >= min_features) {
            out_mask[i] = 1;
            count++;
        } else {
            out_mask[i] = 0;
        }
    }
    
    return count;
}

} // namespace scl::kernel::hvg

