#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/simd.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file feature.hpp
/// @brief Feature Selection & Gene Statistics Kernels
///
/// Implements high-performance estimators for identifying informative features
/// (Highly Variable Genes).
///
/// Supported Methods:
/// 1. Clipped Moments (Seurat V3 "vst"): Robust variance with outlier clipping.
/// 2. Standard Moments (CellRanger/LogMeanVar): Basic Mean/Var stats.
/// 3. Detection Rate: Fraction of cells expressing a gene (Dropout analysis).
/// 4. Dispersion: Fano factor calculation.
///
/// Note: Optimized for CSC Matrix layout (Gene-wise parallelization).
// =============================================================================

namespace scl::kernel::feature {

namespace detail {

// =============================================================================
// Generic Implementation (Tag Dispatch)
// =============================================================================

/// @brief Generic implementation for clipped moments.
template <AnySparse MatrixT>
SCL_FORCE_INLINE void clipped_moments_impl(
    MatrixT matrix,
    Span<const Real> clip_vals,
    MutableSpan<Real> out_means,
    MutableSpan<Real> out_vars
) {
    if constexpr (!tag_is_csr_v<typename MatrixT::Tag>) {
        SCL_CHECK_DIM(clip_vals.size == static_cast<Size>(scl::cols(matrix)), "Clip vals mismatch");
        SCL_CHECK_DIM(out_means.size == static_cast<Size>(scl::cols(matrix)), "Output mean mismatch");
        SCL_CHECK_DIM(out_vars.size == static_cast<Size>(scl::cols(matrix)), "Output var mismatch");
        
        const Real N = static_cast<Real>(scl::rows(matrix));
        const Real N_minus_1 = N - 1.0;
        
        scl::threading::parallel_for(0, static_cast<size_t>(scl::cols(matrix)), [&](size_t c) {
            Real clip = clip_vals[c];
            auto vals = scl::primary_values(matrix, static_cast<Index>(c));
            
            namespace s = scl::simd;
            const s::Tag d;
            const size_t lanes = s::lanes();
            
            auto v_clip = s::Set(d, clip);
            auto v_sum  = s::Zero(d);
            auto v_ssq  = s::Zero(d);
            
            size_t k = 0;
            for (; k + lanes <= vals.size; k += lanes) {
                auto v = s::Load(d, vals.ptr + k);
                v = s::Min(v, v_clip);
                v_sum = s::Add(v_sum, v);
                v_ssq = s::MulAdd(v, v, v_ssq);
            }
            
            Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
            Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));
            
            for (; k < vals.size; ++k) {
                Real v = vals[k];
                if (v > clip) v = clip;
                sum += v;
                sum_sq += v * v;
            }
            
            Real mu = sum / N;
            Real var = 0.0;
            
            if (N > 1.0) {
                var = (sum_sq - N * mu * mu) / N_minus_1;
            }
            
            if (var < 0) var = 0.0;
            
            out_means[c] = mu;
            out_vars[c] = var;
        });
    } else if constexpr (tag_is_csr_v<typename MatrixT::Tag>) {
        SCL_CHECK_DIM(clip_vals.size == static_cast<Size>(scl::rows(matrix)), "Clip vals mismatch");
        SCL_CHECK_DIM(out_means.size == static_cast<Size>(scl::rows(matrix)), "Output mean mismatch");
        SCL_CHECK_DIM(out_vars.size == static_cast<Size>(scl::rows(matrix)), "Output var mismatch");
        
        const Real N = static_cast<Real>(scl::cols(matrix));
        const Real N_minus_1 = N - 1.0;
        
        scl::threading::parallel_for(0, static_cast<size_t>(scl::rows(matrix)), [&](size_t r) {
            Real clip = clip_vals[r];
            auto vals = scl::primary_values(matrix, static_cast<Index>(r));
            
            namespace s = scl::simd;
            const s::Tag d;
            const size_t lanes = s::lanes();
            
            auto v_clip = s::Set(d, clip);
            auto v_sum  = s::Zero(d);
            auto v_ssq  = s::Zero(d);
            
            size_t k = 0;
            for (; k + lanes <= vals.size; k += lanes) {
                auto v = s::Load(d, vals.ptr + k);
                v = s::Min(v, v_clip);
                v_sum = s::Add(v_sum, v);
                v_ssq = s::MulAdd(v, v, v_ssq);
            }
            
            Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
            Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));
            
            for (; k < vals.size; ++k) {
                Real v = vals[k];
                if (v > clip) v = clip;
                sum += v;
                sum_sq += v * v;
            }
            
            Real mu = sum / N;
            Real var = 0.0;
            
            if (N > 1.0) {
                var = (sum_sq - N * mu * mu) / N_minus_1;
            }
            
            if (var < 0) var = 0.0;
            
            out_means[r] = mu;
            out_vars[r] = var;
        });
    }
}

/// @brief Generic implementation for standard moments.
template <AnySparse MatrixT>
SCL_FORCE_INLINE void standard_moments_impl(
    MatrixT matrix,
    MutableSpan<Real> out_means,
    MutableSpan<Real> out_vars,
    int ddof = 1
) {
    if constexpr (!tag_is_csr_v<typename MatrixT::Tag>) {
        SCL_CHECK_DIM(out_means.size == static_cast<Size>(scl::cols(matrix)), "Output mean mismatch");
        SCL_CHECK_DIM(out_vars.size == static_cast<Size>(scl::cols(matrix)), "Output var mismatch");
        
        const Real N = static_cast<Real>(scl::rows(matrix));
        const Real denom = N - static_cast<Real>(ddof);
        
        scl::threading::parallel_for(0, static_cast<size_t>(scl::cols(matrix)), [&](size_t c) {
            auto vals = scl::primary_values(matrix, static_cast<Index>(c));
            
            namespace s = scl::simd;
            const s::Tag d;
            const size_t lanes = s::lanes();
            
            auto v_sum = s::Zero(d);
            auto v_ssq = s::Zero(d);
            
            size_t k = 0;
            for (; k + lanes <= vals.size; k += lanes) {
                auto v = s::Load(d, vals.ptr + k);
                v_sum = s::Add(v_sum, v);
                v_ssq = s::MulAdd(v, v, v_ssq);
            }
            
            Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
            Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));
            
            for (; k < vals.size; ++k) {
                Real v = vals[k];
                sum += v;
                sum_sq += v * v;
            }
            
            Real mu = sum / N;
            Real var = 0.0;
            
            if (denom > 0) {
                var = (sum_sq - sum * mu) / denom;
            }
            
            if (var < 0) var = 0.0;
            
            out_means[c] = mu;
            out_vars[c] = var;
        });
    } else if constexpr (tag_is_csr_v<typename MatrixT::Tag>) {
        SCL_CHECK_DIM(out_means.size == static_cast<Size>(scl::rows(matrix)), "Output mean mismatch");
        SCL_CHECK_DIM(out_vars.size == static_cast<Size>(scl::rows(matrix)), "Output var mismatch");
        
        const Real N = static_cast<Real>(scl::cols(matrix));
        const Real denom = N - static_cast<Real>(ddof);
        
        scl::threading::parallel_for(0, static_cast<size_t>(scl::rows(matrix)), [&](size_t r) {
            auto vals = scl::primary_values(matrix, static_cast<Index>(r));
            
            namespace s = scl::simd;
            const s::Tag d;
            const size_t lanes = s::lanes();
            
            auto v_sum = s::Zero(d);
            auto v_ssq = s::Zero(d);
            
            size_t k = 0;
            for (; k + lanes <= vals.size; k += lanes) {
                auto v = s::Load(d, vals.ptr + k);
                v_sum = s::Add(v_sum, v);
                v_ssq = s::MulAdd(v, v, v_ssq);
            }
            
            Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
            Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));
            
            for (; k < vals.size; ++k) {
                Real v = vals[k];
                sum += v;
                sum_sq += v * v;
            }
            
            Real mu = sum / N;
            Real var = 0.0;
            
            if (denom > 0) {
                var = (sum_sq - sum * mu) / denom;
            }
            
            if (var < 0) var = 0.0;
            
            out_means[r] = mu;
            out_vars[r] = var;
        });
    }
}

/// @brief Generic implementation for detection rate.
template <AnySparse MatrixT>
SCL_FORCE_INLINE void detection_rate_impl(
    MatrixT matrix,
    MutableSpan<Real> out_rates
) {
    if constexpr (!tag_is_csr_v<typename MatrixT::Tag>) {
        SCL_CHECK_DIM(out_rates.size == static_cast<Size>(scl::cols(matrix)), "Output rates mismatch");
        
        const Real inv_N = 1.0 / static_cast<Real>(scl::rows(matrix));
        
        scl::threading::parallel_for(0, static_cast<size_t>(scl::cols(matrix)), [&](size_t c) {
            Index col_idx = static_cast<Index>(c);
            Index len = scl::primary_length(matrix, col_idx);
            
            Real nnz_count = static_cast<Real>(len);
            out_rates[c] = nnz_count * inv_N;
        });
    } else {
        SCL_CHECK_DIM(out_rates.size == static_cast<Size>(scl::rows(matrix)), "Output rates mismatch");
        
        const Real inv_N = 1.0 / static_cast<Real>(scl::cols(matrix));
        
        scl::threading::parallel_for(0, static_cast<size_t>(scl::rows(matrix)), [&](size_t r) {
            Index row_idx = static_cast<Index>(r);
            Index len = scl::primary_length(matrix, row_idx);
            
            Real nnz_count = static_cast<Real>(len);
            out_rates[r] = nnz_count * inv_N;
        });
    }
}

} // namespace detail

// =============================================================================
// Layer 1: Virtual Interface (ISparse-based, Generic but Slower)
// =============================================================================

/// @brief Compute clipped mean and variance (Virtual Interface, CSC).
///
/// Generic implementation using ISparse base class.
///
/// For each column j, computes statistics on values clipped at theta_j.
template <typename T>
SCL_FORCE_INLINE void clipped_moments(
    const ICSC<T>& matrix,
    Span<const Real> clip_vals,
    MutableSpan<Real> out_means,
    MutableSpan<Real> out_vars
) {
    SCL_CHECK_DIM(clip_vals.size == static_cast<Size>(matrix.cols()), "Clip vals mismatch");
    SCL_CHECK_DIM(out_means.size == static_cast<Size>(matrix.cols()), "Output mean mismatch");
    SCL_CHECK_DIM(out_vars.size == static_cast<Size>(matrix.cols()), "Output var mismatch");
    
    const Real N = static_cast<Real>(matrix.rows());
    const Real N_minus_1 = N - 1.0;
    
    scl::threading::parallel_for(0, static_cast<size_t>(matrix.cols()), [&](size_t c) {
        Real clip = clip_vals[c];
        auto vals = matrix.primary_values(static_cast<Index>(c));
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_clip = s::Set(d, clip);
        auto v_sum  = s::Zero(d);
        auto v_ssq  = s::Zero(d);
        
        size_t k = 0;
        for (; k + lanes <= vals.size; k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            v = s::Min(v, v_clip);
            v_sum = s::Add(v_sum, v);
            v_ssq = s::MulAdd(v, v, v_ssq);
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));
        
        for (; k < vals.size; ++k) {
            Real v = vals[k];
            if (v > clip) v = clip;
            sum += v;
            sum_sq += v * v;
        }
        
        Real mu = sum / N;
        Real var = 0.0;
        
        if (N > 1.0) {
            var = (sum_sq - N * mu * mu) / N_minus_1;
        }
        
        if (var < 0) var = 0.0;
        
        out_means[c] = mu;
        out_vars[c] = var;
    });
}

/// @brief Compute clipped mean and variance (Virtual Interface, CSR).
///
/// Generic implementation using ISparse base class.
///
/// For each row i, computes statistics on values clipped at theta_i.
template <typename T>
SCL_FORCE_INLINE void clipped_moments(
    const ICSR<T>& matrix,
    Span<const Real> clip_vals,
    MutableSpan<Real> out_means,
    MutableSpan<Real> out_vars
) {
    SCL_CHECK_DIM(clip_vals.size == static_cast<Size>(matrix.rows()), "Clip vals mismatch");
    SCL_CHECK_DIM(out_means.size == static_cast<Size>(matrix.rows()), "Output mean mismatch");
    SCL_CHECK_DIM(out_vars.size == static_cast<Size>(matrix.rows()), "Output var mismatch");
    
    const Real N = static_cast<Real>(matrix.cols());
    const Real N_minus_1 = N - 1.0;
    
    scl::threading::parallel_for(0, static_cast<size_t>(matrix.rows()), [&](size_t r) {
        Real clip = clip_vals[r];
        auto vals = matrix.primary_values(static_cast<Index>(r));
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_clip = s::Set(d, clip);
        auto v_sum  = s::Zero(d);
        auto v_ssq  = s::Zero(d);
        
        size_t k = 0;
        for (; k + lanes <= vals.size; k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            v = s::Min(v, v_clip);
            v_sum = s::Add(v_sum, v);
            v_ssq = s::MulAdd(v, v, v_ssq);
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));
        
        for (; k < vals.size; ++k) {
            Real v = vals[k];
            if (v > clip) v = clip;
            sum += v;
            sum_sq += v * v;
        }
        
        Real mu = sum / N;
        Real var = 0.0;
        
        if (N > 1.0) {
            var = (sum_sq - N * mu * mu) / N_minus_1;
        }
        
        if (var < 0) var = 0.0;
        
        out_means[r] = mu;
        out_vars[r] = var;
    });
}

// =============================================================================
// Layer 2: Concept-Based (CSCLike/CSRLike, Optimized for Custom/Virtual)
// =============================================================================

/// @brief Compute clipped mean and variance (Concept-based, Optimized, CSC).
///
/// High-performance implementation for CSCLike matrices.
/// Uses unified accessors for zero-overhead abstraction.
///
/// For each column j, computes statistics on values clipped at theta_j.
///
/// For each column j, computes statistics on values clipped at theta_j.
/// - mu_j = (1/N) * sum over i of min(X_{ij}, theta_j)
/// - sigma^2_j = (1/(N-1)) * (sum over i of min(X_{ij}, theta_j)^2 - N * mu_j^2)
///
/// @param matrix    CSC Matrix (Column-major is required for speed).
/// @param clip_vals Clipping thresholds theta_j for each column (size = cols).
/// @param out_means Output: Mean of clipped values (size = cols).
/// @param out_vars  Output: Variance of clipped values (size = cols).
template <CSCLike MatrixT>
SCL_FORCE_INLINE void clipped_moments(
    MatrixT matrix,
    Span<const Real> clip_vals,
    MutableSpan<Real> out_means,
    MutableSpan<Real> out_vars
) {
    detail::clipped_moments_impl(matrix, clip_vals, out_means, out_vars);
}

/// @brief Compute clipped mean and variance for rows (CSR version).
///
/// For each row i, computes statistics on values clipped at theta_i.
/// - mu_i = (1/N) * sum over j of min(X_{ij}, theta_i)
/// - sigma^2_i = (1/(N-1)) * (sum over j of min(X_{ij}, theta_i)^2 - N * mu_i^2)
///
/// @param matrix    CSR Matrix.
/// @param clip_vals Clipping thresholds theta_i for each row (size = rows).
/// @param out_means Output: Mean of clipped values (size = rows).
/// @param out_vars  Output: Variance of clipped values (size = rows).
template <CSRLike MatrixT>
SCL_FORCE_INLINE void clipped_moments(
    MatrixT matrix,
    Span<const Real> clip_vals,
    MutableSpan<Real> out_means,
    MutableSpan<Real> out_vars
) {
    detail::clipped_moments_impl(matrix, clip_vals, out_means, out_vars);
}

/// @brief Compute standard mean and variance (Virtual Interface, CSC).
///
/// Generic implementation using ISparse base class.
template <typename T>
SCL_FORCE_INLINE void standard_moments(
    const ICSC<T>& matrix,
    MutableSpan<Real> out_means,
    MutableSpan<Real> out_vars,
    int ddof = 1
) {
    SCL_CHECK_DIM(out_means.size == static_cast<Size>(matrix.cols()), "Output mean mismatch");
    SCL_CHECK_DIM(out_vars.size == static_cast<Size>(matrix.cols()), "Output var mismatch");
    
    const Real N = static_cast<Real>(matrix.rows());
    const Real denom = N - static_cast<Real>(ddof);
    
    scl::threading::parallel_for(0, static_cast<size_t>(matrix.cols()), [&](size_t c) {
        auto vals = matrix.primary_values(static_cast<Index>(c));
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_sum = s::Zero(d);
        auto v_ssq = s::Zero(d);
        
        size_t k = 0;
        for (; k + lanes <= vals.size; k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            v_sum = s::Add(v_sum, v);
            v_ssq = s::MulAdd(v, v, v_ssq);
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));
        
        for (; k < vals.size; ++k) {
            Real v = vals[k];
            sum += v;
            sum_sq += v * v;
        }
        
        Real mu = sum / N;
        Real var = 0.0;
        
        if (denom > 0) {
            var = (sum_sq - sum * mu) / denom;
        }
        
        if (var < 0) var = 0.0;
        
        out_means[c] = mu;
        out_vars[c] = var;
    });
}

/// @brief Compute standard mean and variance (Virtual Interface, CSR).
///
/// Generic implementation using ISparse base class.
template <typename T>
SCL_FORCE_INLINE void standard_moments(
    const ICSR<T>& matrix,
    MutableSpan<Real> out_means,
    MutableSpan<Real> out_vars,
    int ddof = 1
) {
    SCL_CHECK_DIM(out_means.size == static_cast<Size>(matrix.rows()), "Output mean mismatch");
    SCL_CHECK_DIM(out_vars.size == static_cast<Size>(matrix.rows()), "Output var mismatch");
    
    const Real N = static_cast<Real>(matrix.cols());
    const Real denom = N - static_cast<Real>(ddof);
    
    scl::threading::parallel_for(0, static_cast<size_t>(matrix.rows()), [&](size_t r) {
        auto vals = matrix.primary_values(static_cast<Index>(r));
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_sum = s::Zero(d);
        auto v_ssq = s::Zero(d);
        
        size_t k = 0;
        for (; k + lanes <= vals.size; k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            v_sum = s::Add(v_sum, v);
            v_ssq = s::MulAdd(v, v, v_ssq);
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));
        
        for (; k < vals.size; ++k) {
            Real v = vals[k];
            sum += v;
            sum_sq += v * v;
        }
        
        Real mu = sum / N;
        Real var = 0.0;
        
        if (denom > 0) {
            var = (sum_sq - sum * mu) / denom;
        }
        
        if (var < 0) var = 0.0;
        
        out_means[r] = mu;
        out_vars[r] = var;
    });
}

// =============================================================================
// 2. Standard Moments (Basic Mean/Var)
// =============================================================================

/// @brief Compute standard mean and variance per gene (Concept-based, Optimized, CSC).
/// 
/// Used for:
/// - "mean_var_plot" / "cell_ranger" flavor HVG selection.
/// - General QC metrics.
///
/// @param matrix    CSC Matrix.
/// @param out_means Output Means.
/// @param out_vars  Output Variances.
/// @param ddof      Delta Degrees of Freedom (usually 1 for sample variance).
template <CSCLike MatrixT>
SCL_FORCE_INLINE void standard_moments(
    MatrixT matrix,
    MutableSpan<Real> out_means,
    MutableSpan<Real> out_vars,
    int ddof = 1
) {
    detail::standard_moments_impl(matrix, out_means, out_vars, ddof);
}

/// @brief Compute standard mean and variance per row (CSR version).
/// 
/// Used for row-wise statistics.
///
/// @param matrix    CSR Matrix.
/// @param out_means Output Means (size = rows).
/// @param out_vars  Output Variances (size = rows).
/// @param ddof      Delta Degrees of Freedom (usually 1 for sample variance).
template <CSRLike MatrixT>
SCL_FORCE_INLINE void standard_moments(
    MatrixT matrix,
    MutableSpan<Real> out_means,
    MutableSpan<Real> out_vars,
    int ddof = 1
) {
    detail::standard_moments_impl(matrix, out_means, out_vars, ddof);
}

/// @brief Compute detection rate (Virtual Interface, CSC).
///
/// Generic implementation using ISparse base class.
template <typename T>
SCL_FORCE_INLINE void detection_rate(
    const ICSC<T>& matrix,
    MutableSpan<Real> out_rates
) {
    SCL_CHECK_DIM(out_rates.size == static_cast<Size>(matrix.cols()), "Output rates mismatch");
    
    const Real inv_N = 1.0 / static_cast<Real>(matrix.rows());
    
    scl::threading::parallel_for(0, static_cast<size_t>(matrix.cols()), [&](size_t c) {
        Index col_idx = static_cast<Index>(c);
        Index len = matrix.primary_length(col_idx);
        
        Real nnz_count = static_cast<Real>(len);
        out_rates[c] = nnz_count * inv_N;
    });
}

/// @brief Compute detection rate (Virtual Interface, CSR).
///
/// Generic implementation using ISparse base class.
template <typename T>
SCL_FORCE_INLINE void detection_rate(
    const ICSR<T>& matrix,
    MutableSpan<Real> out_rates
) {
    SCL_CHECK_DIM(out_rates.size == static_cast<Size>(matrix.rows()), "Output rates mismatch");
    
    const Real inv_N = 1.0 / static_cast<Real>(matrix.cols());
    
    scl::threading::parallel_for(0, static_cast<size_t>(matrix.rows()), [&](size_t r) {
        Index row_idx = static_cast<Index>(r);
        Index len = matrix.primary_length(row_idx);
        
        Real nnz_count = static_cast<Real>(len);
        out_rates[r] = nnz_count * inv_N;
    });
}

// =============================================================================
// 3. Detection Rate (Dropout)
// =============================================================================

/// @brief Compute the fraction of cells expressing each gene (Concept-based, Optimized, CSC).
///
/// Useful for filtering genes (e.g., min_cells > 3).
///
/// @param matrix CSC Matrix.
/// @param out_rates Output: fraction [0.0, 1.0] for each gene.
template <CSCLike MatrixT>
SCL_FORCE_INLINE void detection_rate(
    MatrixT matrix,
    MutableSpan<Real> out_rates
) {
    detail::detection_rate_impl(matrix, out_rates);
}

/// @brief Compute the fraction of genes expressed in each cell (CSR version).
///
/// Useful for filtering cells (e.g., min_genes > 200).
///
/// @param matrix CSR Matrix.
/// @param out_rates Output: fraction [0.0, 1.0] for each cell (row).
template <CSRLike MatrixT>
SCL_FORCE_INLINE void detection_rate(
    MatrixT matrix,
    MutableSpan<Real> out_rates
) {
    detail::detection_rate_impl(matrix, out_rates);
}

// =============================================================================
// 4. Dispersion (Fano Factor)
// =============================================================================

/// @brief Compute Dispersion (Variance / Mean).
///
/// Can be used after `standard_moments` or `clipped_moments`.
/// Operations are performed in blocks to maximize SIMD throughput.
///
/// @param means Input Means.
/// @param vars  Input Variances.
/// @param out_dispersion Output Dispersion values.
SCL_FORCE_INLINE void dispersion(
    Span<const Real> means,
    Span<const Real> vars,
    MutableSpan<Real> out_dispersion
) {
    SCL_CHECK_DIM(means.size == vars.size, "Mean/Var dim mismatch");
    SCL_CHECK_DIM(out_dispersion.size == means.size, "Output dim mismatch");

    // Block processing for cache locality and SIMD efficiency
    const size_t block_size = 4096;
    const size_t num_blocks = (means.size + block_size - 1) / block_size;

    scl::threading::parallel_for(0, num_blocks, [&](size_t b) {
        size_t start = b * block_size;
        size_t end = std::min(start + block_size, static_cast<size_t>(means.size));

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

        // Epsilon for zero-division check
        const auto v_eps = s::Set(d, 1e-12);
        const auto v_zero = s::Zero(d);

        size_t k = start;

        // --- SIMD Loop ---
        for (; k + lanes <= end; k += lanes) {
            auto v_mean = s::Load(d, means.ptr + k);
            auto v_var  = s::Load(d, vars.ptr + k);

            // Mask: mean > 1e-12
            auto mask = s::Gt(v_mean, v_eps);

            // Calculate division: var / mean
            // Note: We compute div even if mean is small (it might produce Inf/NaN), 
            // but we mask the result afterwards. This avoids branching.
            auto v_div = s::Div(v_var, v_mean);

            // Result = (mean > eps) ? (var / mean) : 0.0
            auto v_res = s::IfThenElse(mask, v_div, v_zero);

            s::Store(v_res, d, out_dispersion.ptr + k);
        }

        // --- Scalar Tail ---
        for (; k < end; ++k) {
            Real m = means[k];
            Real v = vars[k];

            if (m > 1e-12) {
                out_dispersion[k] = v / m;
            } else {
                out_dispersion[k] = 0.0;
            }
        }
    });
}

} // namespace scl::kernel::feature
