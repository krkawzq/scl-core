#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file softmax.hpp
/// @brief Softmax Transformation
///
/// Implements numerically stable softmax: softmax(x) = exp(x) / sum(exp(x))
/// Uses max-subtraction trick: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
///
/// Performance: O(nnz), SIMD optimized
// =============================================================================

namespace scl::kernel::softmax {

/// @brief Softmax transformation (unified for CSR/CSC)
///
/// Applies softmax independently to each primary dimension element.
///
/// @param matrix Input sparse matrix (modified in-place)
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void softmax_inplace(MatrixT& matrix) {
    const Index primary_dim = scl::primary_size(matrix);
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        auto vals = scl::primary_values(matrix, static_cast<Index>(p));
        
        if (vals.size() == 0) return;
        
        // Find max (for numerical stability)
        Real max_val = vals[0];
        for (size_t k = 1; k < vals.size(); ++k) {
            if (vals[k] > max_val) {
                max_val = vals[k];
            }
        }
        
        // Compute exp(x - max) and sum
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        const auto v_max = s::Set(d, max_val);
        auto v_sum = s::Zero(d);
        
        size_t k = 0;
        for (; k + lanes <= vals.size(); k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            v = s::Sub(v, v_max);
            v = s::Exp(d, v);
            s::Store(v, d, vals.ptr + k);
            v_sum = s::Add(v_sum, v);
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        
        for (; k < vals.size(); ++k) {
            Real v = std::exp(vals[k] - max_val);
            vals[k] = v;
            sum += v;
        }
        
        // Normalize
        if (sum > 0) {
            Real inv_sum = 1.0 / sum;
            const auto v_inv_sum = s::Set(d, inv_sum);
            
            k = 0;
            for (; k + lanes <= vals.size(); k += lanes) {
                auto v = s::Load(d, vals.ptr + k);
                v = s::Mul(v, v_inv_sum);
                s::Store(v, d, vals.ptr + k);
            }
            
            for (; k < vals.size(); ++k) {
                vals[k] *= inv_sum;
            }
        }
    });
}

} // namespace scl::kernel::softmax
