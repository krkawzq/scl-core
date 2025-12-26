#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file feature.hpp
/// @brief Feature Selection and Gene Statistics
///
/// Implements:
/// 1. Clipped Moments: Robust variance with outlier clipping (Seurat V3)
/// 2. Standard Moments: Basic mean/variance
/// 3. Detection Rate: Fraction of non-zero elements
/// 4. Dispersion: Fano factor (variance/mean)
///
/// All functions unified for CSR/CSC using AnySparse
// =============================================================================

namespace scl::kernel::feature {

// =============================================================================
// Core Algorithms (Unified for CSR/CSC)
// =============================================================================

/// @brief Compute clipped mean and variance (unified for CSR/CSC)
///
/// For each primary dimension element, computes statistics on values clipped at threshold.
/// Formula: mu = (1/N) * sum(min(X, theta))
///          var = (1/(N-ddof)) * (sum(min(X, theta)^2) - N*mu^2)
///
/// @param matrix Input sparse matrix
/// @param clip_vals Clipping thresholds [size = primary_dim]
/// @param out_means Output means [size = primary_dim]
/// @param out_vars Output variances [size = primary_dim]
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void clipped_moments(
    const MatrixT& matrix,
    Array<const Real> clip_vals,
    Array<Real> out_means,
    Array<Real> out_vars
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real N_minus_1 = N - 1.0;
    
    SCL_CHECK_DIM(clip_vals.size() == static_cast<Size>(primary_dim), "Clip vals mismatch");
    SCL_CHECK_DIM(out_means.size() == static_cast<Size>(primary_dim), "Output mean mismatch");
    SCL_CHECK_DIM(out_vars.size() == static_cast<Size>(primary_dim), "Output var mismatch");
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Real clip = clip_vals[p];
        auto vals = scl::primary_values(matrix, static_cast<Index>(p));
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_clip = s::Set(d, clip);
        auto v_sum  = s::Zero(d);
        auto v_ssq  = s::Zero(d);
        
        size_t k = 0;
        for (; k + lanes <= vals.size(); k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            v = s::Min(v, v_clip);
            v_sum = s::Add(v_sum, v);
            v_ssq = s::MulAdd(v, v, v_ssq);
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));
        
        for (; k < vals.size(); ++k) {
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
        
        out_means[p] = mu;
        out_vars[p] = var;
    });
}

/// @brief Compute standard mean and variance (unified for CSR/CSC)
///
/// @param matrix Input sparse matrix
/// @param out_means Output means [size = primary_dim]
/// @param out_vars Output variances [size = primary_dim]
/// @param ddof Delta degrees of freedom (default 1)
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void standard_moments(
    const MatrixT& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof = 1
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real denom = N - static_cast<Real>(ddof);
    
    SCL_CHECK_DIM(out_means.size() == static_cast<Size>(primary_dim), "Output mean mismatch");
    SCL_CHECK_DIM(out_vars.size() == static_cast<Size>(primary_dim), "Output var mismatch");
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        auto vals = scl::primary_values(matrix, static_cast<Index>(p));
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_sum = s::Zero(d);
        auto v_ssq = s::Zero(d);
        
        size_t k = 0;
        for (; k + lanes <= vals.size(); k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            v_sum = s::Add(v_sum, v);
            v_ssq = s::MulAdd(v, v, v_ssq);
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));
        
        for (; k < vals.size(); ++k) {
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
        
        out_means[p] = mu;
        out_vars[p] = var;
    });
}

/// @brief Compute detection rate (unified for CSR/CSC)
///
/// Computes fraction of non-zero elements for each primary dimension element.
///
/// @param matrix Input sparse matrix
/// @param out_rates Output rates [size = primary_dim], values in [0, 1]
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void detection_rate(
    const MatrixT& matrix,
    Array<Real> out_rates
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real inv_N = 1.0 / static_cast<Real>(scl::secondary_size(matrix));
    
    SCL_CHECK_DIM(out_rates.size() == static_cast<Size>(primary_dim), "Output rates mismatch");
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = scl::primary_length(matrix, static_cast<Index>(p));
        out_rates[p] = static_cast<Real>(len) * inv_N;
    });
}

/// @brief Compute dispersion (variance/mean) with SIMD
///
/// @param means Input means
/// @param vars Input variances
/// @param out_dispersion Output dispersion values
SCL_FORCE_INLINE void dispersion(
    Array<const Real> means,
    Array<const Real> vars,
    Array<Real> out_dispersion
) {
    SCL_CHECK_DIM(means.size() == vars.size(), "Mean/Var dim mismatch");
    SCL_CHECK_DIM(out_dispersion.size() == means.size(), "Output dim mismatch");

    const size_t n = means.size();
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    const auto v_eps = s::Set(d, 1e-12);
    const auto v_zero = s::Zero(d);
    
    size_t k = 0;
    
    for (; k + lanes <= n; k += lanes) {
        auto v_mean = s::Load(d, means.ptr + k);
        auto v_var  = s::Load(d, vars.ptr + k);
        
        auto mask = s::Gt(v_mean, v_eps);
        auto v_div = s::Div(v_var, v_mean);
        auto v_res = s::IfThenElse(mask, v_div, v_zero);
        
        s::Store(v_res, d, out_dispersion.ptr + k);
    }
    
    for (; k < n; ++k) {
        Real m = means[k];
        Real v = vars[k];
        out_dispersion[k] = (m > 1e-12) ? (v / m) : 0.0;
    }
}

} // namespace scl::kernel::feature
