#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
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
/// @section Supported Methods
/// 1. **Clipped Moments** (Seurat V3 "vst"): Robust variance with outlier clipping.
/// 2. **Standard Moments** (CellRanger/LogMeanVar): Basic Mean/Var stats.
/// 3. **Detection Rate**: Fraction of cells expressing a gene (Dropout analysis).
/// 4. **Dispersion**: Fano factor calculation.
///
/// @note Optimized for CSC Matrix layout (Gene-wise parallelization).
// =============================================================================

namespace scl::kernel::feature {

// =============================================================================
// 1. Seurat V3: Clipped Moments (VST)
// =============================================================================

/// @brief Compute clipped mean and variance (Seurat V3 "vst" flavor).
///
/// For each column j, computes statistics on values clipped at theta_j.
/// - mu_j = (1/N) * sum over i of min(X_{ij}, theta_j)
/// - sigma^2_j = (1/(N-1)) * (sum over i of min(X_{ij}, theta_j)^2 - N * mu_j^2)
///
/// @param matrix    CSC Matrix (Column-major is required for speed).
/// @param clip_vals Clipping thresholds theta_j for each column (size = cols).
/// @param out_means Output: Mean of clipped values (size = cols).
/// @param out_vars  Output: Variance of clipped values (size = cols).
SCL_FORCE_INLINE void clipped_moments(
    CSCMatrix<Real> matrix,
    Span<const Real> clip_vals,
    MutableSpan<Real> out_means,
    MutableSpan<Real> out_vars
) {
    SCL_CHECK_DIM(clip_vals.size == static_cast<Size>(matrix.cols), "Clip vals mismatch");
    SCL_CHECK_DIM(out_means.size == static_cast<Size>(matrix.cols), "Output mean mismatch");
    SCL_CHECK_DIM(out_vars.size == static_cast<Size>(matrix.cols), "Output var mismatch");

    const Real N = static_cast<Real>(matrix.rows);
    const Real N_minus_1 = N - 1.0;

    // Parallelize over Features (Genes)
    scl::threading::parallel_for(0, matrix.cols, [&](size_t c) {
        Real clip = clip_vals[c];
        auto vals = matrix.col_values(static_cast<Index>(c));
        
        // --- SIMD Accumulation ---
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_clip = s::Set(d, clip);
        auto v_sum  = s::Zero(d);
        auto v_ssq  = s::Zero(d);

        size_t k = 0;
        for (; k + lanes <= vals.size; k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            
            // Clip: v = min(v, clip)
            v = s::Min(v, v_clip);
            
            // Accumulate
            v_sum = s::Add(v_sum, v);
            v_ssq = s::MulAdd(v, v, v_ssq); // FMA for speed
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));

        // --- Scalar Tail ---
        for (; k < vals.size; ++k) {
            Real v = vals[k];
            if (v > clip) v = clip;
            sum += v;
            sum_sq += v * v;
        }
        
        // --- Final Stats ---
        // Implicit zeros are handled because min(0, clip) = 0.
        Real mu = sum / N;
        Real var = 0.0;
        
        if (N > 1.0) {
            // Var = (SumSq - N*mu^2) / (N-1)
            var = (sum_sq - N * mu * mu) / N_minus_1;
        }
        
        if (var < 0) var = 0.0; // Fix floating point undershoot
        
        out_means[c] = mu;
        out_vars[c] = var;
    });
}

// =============================================================================
// 2. Standard Moments (Basic Mean/Var)
// =============================================================================

/// @brief Compute standard mean and variance per gene.
/// 
/// Used for:
/// - "mean_var_plot" / "cell_ranger" flavor HVG selection.
/// - General QC metrics.
///
/// @param matrix    CSC Matrix.
/// @param out_means Output Means.
/// @param out_vars  Output Variances.
/// @param ddof      Delta Degrees of Freedom (usually 1 for sample variance).
SCL_FORCE_INLINE void standard_moments(
    CSCMatrix<Real> matrix,
    MutableSpan<Real> out_means,
    MutableSpan<Real> out_vars,
    int ddof = 1
) {
    SCL_CHECK_DIM(out_means.size == static_cast<Size>(matrix.cols), "Output mean mismatch");
    SCL_CHECK_DIM(out_vars.size == static_cast<Size>(matrix.cols), "Output var mismatch");

    const Real N = static_cast<Real>(matrix.rows);
    const Real denom = N - static_cast<Real>(ddof);

    scl::threading::parallel_for(0, matrix.cols, [&](size_t c) {
        auto vals = matrix.col_values(static_cast<Index>(c));
        
        // --- SIMD Accumulation ---
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

        // --- Final Stats ---
        Real mu = sum / N;
        Real var = 0.0;
        
        if (denom > 0) {
            // Var = (SumSq - N*mu^2) / (N-ddof)
            // Equivalent to (SumSq - Sum*Mean) / (N-ddof)
            var = (sum_sq - sum * mu) / denom;
        }
        
        if (var < 0) var = 0.0;

        out_means[c] = mu;
        out_vars[c] = var;
    });
}

// =============================================================================
// 3. Detection Rate (Dropout)
// =============================================================================

/// @brief Compute the fraction of cells expressing each gene.
///
/// Useful for filtering genes (e.g., min_cells > 3).
///
/// @param matrix CSC Matrix.
/// @param out_rates Output: fraction [0.0, 1.0] for each gene.
SCL_FORCE_INLINE void detection_rate(
    CSCMatrix<Real> matrix,
    MutableSpan<Real> out_rates
) {
    SCL_CHECK_DIM(out_rates.size == static_cast<Size>(matrix.cols), "Output rates mismatch");

    const Real inv_N = 1.0 / static_cast<Real>(matrix.rows);

    scl::threading::parallel_for(0, matrix.cols, [&](size_t c) {
        // In CSC, the number of non-zeros is simply determined by the indptr difference.
        // This is O(1) per gene! Extremely fast.
        Index start = matrix.indptr[c];
        Index end   = matrix.indptr[c+1];
        
        Real nnz_count = static_cast<Real>(end - start);
        out_rates[c] = nnz_count * inv_N;
    });
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
