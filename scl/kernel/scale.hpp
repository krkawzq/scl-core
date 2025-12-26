#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file scale.hpp
/// @brief Matrix Standardization (z-score transformation)
///
/// Implements: X' = (X - mean) / std
/// With optional clipping to [-max, max]
///
/// Note: For sparse matrices, only non-zero values are transformed.
/// This maintains sparsity but is mathematically approximate.
// =============================================================================

namespace scl::kernel::scale {

/// @brief Standardize matrix (unified for CSR/CSC)
///
/// @param matrix Input sparse matrix (modified in-place)
/// @param means Means for each primary dimension [size = primary_dim]
/// @param stds Standard deviations [size = primary_dim]
/// @param max_value Clipping threshold (0 = no clipping)
/// @param zero_center If true, subtract mean
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void standardize(
    MatrixT& matrix,
    Array<const Real> means,
    Array<const Real> stds,
    Real max_value = 0.0,
    bool zero_center = true
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(means.size() == static_cast<Size>(primary_dim), "Means dim mismatch");
    SCL_CHECK_DIM(stds.size() == static_cast<Size>(primary_dim), "Stds dim mismatch");
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Real mu = means[p];
        Real sigma = stds[p];
        Real inv_sigma = (sigma != 0.0) ? (1.0 / sigma) : 0.0;
        
        if (sigma == 0.0) return;

        auto vals = scl::primary_values(matrix, static_cast<Index>(p));
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        const auto v_mu = s::Set(d, mu);
        const auto v_inv_sigma = s::Set(d, inv_sigma);
        const auto v_max = s::Set(d, max_value);
        const auto v_min = s::Set(d, -max_value);
        const bool do_clip = (max_value > 0.0);

        size_t k = 0;
        for (; k + lanes <= vals.size(); k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            
            if (zero_center) {
                v = s::Sub(v, v_mu);
            }
            
            v = s::Mul(v, v_inv_sigma);
            
            if (do_clip) {
                v = s::Min(v, v_max);
                v = s::Max(v, v_min);
            }
            
            s::Store(v, d, vals.ptr + k);
        }
        
        for (; k < vals.size(); ++k) {
            Real v = vals[k];
            
            if (zero_center) {
                v -= mu;
            }
            
            v *= inv_sigma;
            
            if (do_clip) {
                if (v > max_value) v = max_value;
                if (v < -max_value) v = -max_value;
            }
            
            vals[k] = v;
        }
    });
}

} // namespace scl::kernel::scale
