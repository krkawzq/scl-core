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

namespace detail {

/// @brief Base implementation using ISparse interface (CSC version).
///
/// Works with any sparse matrix type that inherits from ICSC.
/// This is the fallback for custom types that use virtual inheritance.
///
/// @param matrix Input ICSC matrix (cells x genes)
/// @param n_top Number of HVGs to select
/// @param out_indices Output: Indices of selected genes [size >= n_top]
/// @param out_mask Output: Binary mask [size = n_genes]
/// @param out_dispersions Output: Dispersion values [size = n_genes]
template <typename T>
void select_by_dispersion_base_csc(
    const ICSC<T>& matrix,
    Size n_top,
    MutableSpan<Index> out_indices,
    MutableSpan<uint8_t> out_mask,
    MutableSpan<Real> out_dispersions
) {
    const Index n_genes = matrix.cols();
    
    SCL_CHECK_DIM(out_dispersions.size == static_cast<Size>(n_genes),
                  "HVG: Dispersions size mismatch");
    
    // Compute statistics using virtual interface
    std::vector<Real> means(n_genes);
    std::vector<Real> vars(n_genes);
    
    scl::threading::parallel_for(0, static_cast<size_t>(n_genes), [&](size_t j) {
        Index col_idx = static_cast<Index>(j);
        auto col_vals = matrix.col_values(col_idx);
        Index len = matrix.col_length(col_idx);
        
        if (len == 0) {
            means[j] = 0.0;
            vars[j] = 0.0;
            return;
        }
        
        // Compute mean
        Real sum = 0.0;
        for (Index k = 0; k < len; ++k) {
            sum += static_cast<Real>(col_vals[k]);
        }
        Real mean = sum / static_cast<Real>(matrix.rows());
        means[j] = mean;
        
        // Compute variance
        Real var_sum = 0.0;
        for (Index k = 0; k < len; ++k) {
            Real diff = static_cast<Real>(col_vals[k]) - mean;
            var_sum += diff * diff;
        }
        // Include zeros in variance calculation
        Real n_zeros = static_cast<Real>(matrix.rows() - len);
        var_sum += n_zeros * mean * mean;
        vars[j] = var_sum / static_cast<Real>(matrix.rows() - 1);
    });
    
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

/// @brief Base implementation using ISparse interface (CSR version).
///
/// @param matrix Input ICSR matrix (samples x features)
/// @param n_top Number of highly variable samples to select
/// @param out_indices Output: Indices of selected samples [size >= n_top]
/// @param out_mask Output: Binary mask [size = n_samples]
/// @param out_dispersions Output: Dispersion values [size = n_samples]
template <typename T>
void select_by_dispersion_base_csr(
    const ICSR<T>& matrix,
    Size n_top,
    MutableSpan<Index> out_indices,
    MutableSpan<uint8_t> out_mask,
    MutableSpan<Real> out_dispersions
) {
    const Index n_samples = matrix.rows();
    
    SCL_CHECK_DIM(out_dispersions.size == static_cast<Size>(n_samples),
                  "HVG: Dispersions size mismatch");
    
    // Compute statistics using virtual interface
    std::vector<Real> means(n_samples);
    std::vector<Real> vars(n_samples);
    
    scl::threading::parallel_for(0, static_cast<size_t>(n_samples), [&](size_t i) {
        Index row_idx = static_cast<Index>(i);
        auto row_vals = matrix.row_values(row_idx);
        Index len = matrix.row_length(row_idx);
        
        if (len == 0) {
            means[i] = 0.0;
            vars[i] = 0.0;
            return;
        }
        
        // Compute mean
        Real sum = 0.0;
        for (Index k = 0; k < len; ++k) {
            sum += static_cast<Real>(row_vals[k]);
        }
        Real mean = sum / static_cast<Real>(matrix.cols());
        means[i] = mean;
        
        // Compute variance
        Real var_sum = 0.0;
        for (Index k = 0; k < len; ++k) {
            Real diff = static_cast<Real>(row_vals[k]) - mean;
            var_sum += diff * diff;
        }
        Real n_zeros = static_cast<Real>(matrix.cols() - len);
        var_sum += n_zeros * mean * mean;
        vars[i] = var_sum / static_cast<Real>(matrix.cols() - 1);
    });
    
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

} // namespace detail

// =============================================================================
// Public Base Interface (ISparse/ICSC/ICSR)
// =============================================================================

/// @brief Select HVGs using dispersion ranking (ICSC base interface).
///
/// Works with any matrix type that inherits from ICSC.
/// This is the base implementation for virtual inheritance.
///
/// @param matrix Input ICSC matrix (cells x genes)
/// @param n_top Number of HVGs to select
/// @param out_indices Output: Indices of selected genes [size >= n_top]
/// @param out_mask Output: Binary mask [size = n_genes]
/// @param out_dispersions Output: Dispersion values [size = n_genes]
template <typename T>
void select_by_dispersion(
    const ICSC<T>& matrix,
    Size n_top,
    MutableSpan<Index> out_indices,
    MutableSpan<uint8_t> out_mask,
    MutableSpan<Real> out_dispersions
) {
    detail::select_by_dispersion_base_csc(matrix, n_top, out_indices, out_mask, out_dispersions);
}

/// @brief Select HVGs using dispersion ranking (ICSR base interface).
///
/// Works with any matrix type that inherits from ICSR.
///
/// @param matrix Input ICSR matrix (samples x features)
/// @param n_top Number of highly variable samples to select
/// @param out_indices Output: Indices of selected samples [size >= n_top]
/// @param out_mask Output: Binary mask [size = n_samples]
/// @param out_dispersions Output: Dispersion values [size = n_samples]
template <typename T>
void select_by_dispersion(
    const ICSR<T>& matrix,
    Size n_top,
    MutableSpan<Index> out_indices,
    MutableSpan<uint8_t> out_mask,
    MutableSpan<Real> out_dispersions
) {
    detail::select_by_dispersion_base_csr(matrix, n_top, out_indices, out_mask, out_dispersions);
}

// =============================================================================
// Layer 2: Concept-Based (CSCLike/CSRLike, Optimized for Custom/Virtual)
// =============================================================================

/// @brief Select HVGs using dispersion (variance/mean) ranking (Concept-based, Optimized, CSC).
///
/// High-performance implementation for CSCLike matrices.
/// Uses unified accessors for zero-overhead abstraction.
///
/// Algorithm:
/// 1. Compute mean and variance per gene
/// 2. Compute dispersion = variance / mean
/// 3. Select top-K genes by dispersion
///
/// @tparam MatrixT Any CSC-like matrix type (CustomSparse or VirtualSparse)
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
    const Index n_genes = scl::cols(matrix);
    
    SCL_CHECK_DIM(out_dispersions.size == static_cast<Size>(n_genes),
                  "HVG: Dispersions size mismatch");
    
    // Compute statistics using unified interface (works for both Custom and Virtual)
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

/// @brief Select HVGs using dispersion ranking (Concept-based, Optimized, CSR).
///
/// High-performance implementation for CSRLike matrices.
/// Uses unified accessors for zero-overhead abstraction.
///
/// @tparam MatrixT Any CSR-like matrix type (CustomSparse or VirtualSparse)
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
    const Index n_samples = scl::rows(matrix);
    
    SCL_CHECK_DIM(out_dispersions.size == static_cast<Size>(n_samples),
                  "HVG: Dispersions size mismatch");
    
    // Compute statistics using unified interface
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
// Specialized Implementations (CustomSparseLike/VirtualSparseLike, Optional)
// =============================================================================
//
// Note: The unified CSCLike/CSRLike versions above work for both Custom and Virtual.
// These specialized versions are kept for backward compatibility and potential
// future optimizations if performance profiling shows significant differences.
// =============================================================================

/// @brief Select HVGs using dispersion (variance/mean) ranking (Custom CSC version).
///
/// Specialized for CustomSparseLike matrices (contiguous storage).
/// Uses optimized feature::standard_moments for better performance.
///
/// @tparam MatrixT Any CustomCSC-like matrix type
/// @param matrix Input CSC-like matrix (cells x genes)
/// @param n_top Number of HVGs to select
/// @param out_indices Output: Indices of selected genes [size >= n_top]
/// @param out_mask Output: Binary mask [size = n_genes]
/// @param out_dispersions Output: Dispersion values [size = n_genes]
template <CustomCSCLike MatrixT>
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
    
    // Compute statistics (optimized for contiguous storage)
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

/// @brief Select HVGs using dispersion ranking (Virtual CSC version).
///
/// Optimized for VirtualSparseLike matrices (discontiguous storage).
/// Uses unified SparseLike interface for compatibility.
///
/// @tparam MatrixT Any VirtualCSC-like matrix type
/// @param matrix Input Virtual CSC matrix (cells x genes)
/// @param n_top Number of HVGs to select
/// @param out_indices Output: Indices of selected genes [size >= n_top]
/// @param out_mask Output: Binary mask [size = n_genes]
/// @param out_dispersions Output: Dispersion values [size = n_genes]
template <VirtualCSCLike MatrixT>
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
    
    // Virtual matrices can use feature::standard_moments (supports CSCLike)
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
/// This overload works with CustomSparseLike matrices (contiguous storage).
///
/// @tparam MatrixT Any CustomCSR-like matrix type
/// @param matrix Input CSR-like matrix (samples x features)
/// @param n_top Number of highly variable samples to select
/// @param out_indices Output: Indices of selected samples [size >= n_top]
/// @param out_mask Output: Binary mask [size = n_samples]
/// @param out_dispersions Output: Dispersion values [size = n_samples]
template <CustomCSRLike MatrixT>
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
    
    // Compute statistics (optimized for contiguous storage)
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

/// @brief Select HVGs using dispersion ranking (Virtual CSR version).
///
/// Optimized for VirtualSparseLike matrices (discontiguous storage).
/// Uses unified SparseLike interface for compatibility.
///
/// @tparam MatrixT Any VirtualCSR-like matrix type
/// @param matrix Input Virtual CSR matrix (samples x features)
/// @param n_top Number of highly variable samples to select
/// @param out_indices Output: Indices of selected samples [size >= n_top]
/// @param out_mask Output: Binary mask [size = n_samples]
/// @param out_dispersions Output: Dispersion values [size = n_samples]
template <VirtualCSRLike MatrixT>
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
    
    // Virtual matrices can use feature::standard_moments (supports CSRLike)
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

