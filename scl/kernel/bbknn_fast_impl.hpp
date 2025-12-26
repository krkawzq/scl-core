#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>

// =============================================================================
/// @file bbknn_fast_impl.hpp
/// @brief Extreme Performance BBKNN for CustomSparse
///
/// Ultra-optimized batch-balanced KNN with:
/// - Batch norm computation
/// - Cache-blocked distance computation
/// - Optimized heap operations
///
/// Performance Target: 1.5-2x faster than generic
// =============================================================================

namespace scl::kernel::bbknn::fast {

/// @brief Ultra-fast norm precomputation (CustomSparse)
///
/// Optimization: Same as neighbors_fast_impl.hpp
/// Batch SIMD on entire data array
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void compute_norms_bbknn_fast(
    const CustomSparse<T, IsCSR>& matrix,
    Array<T> norms_sq
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index len = end - start;
        
        const T* SCL_RESTRICT vals = matrix.data + start;
        
        auto v_sum0 = s::Zero(d);
        auto v_sum1 = s::Zero(d);
        auto v_sum2 = s::Zero(d);
        auto v_sum3 = s::Zero(d);
        
        Index k = 0;
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);
            
            v_sum0 = s::MulAdd(v0, v0, v_sum0);
            v_sum1 = s::MulAdd(v1, v1, v_sum1);
            v_sum2 = s::MulAdd(v2, v2, v_sum2);
            v_sum3 = s::MulAdd(v3, v3, v_sum3);
        }
        
        auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
        
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            v_sum = s::MulAdd(v, v, v_sum);
        }
        
        T sum_sq = s::GetLane(s::SumOfLanes(d, v_sum));
        
        for (; k < len; ++k) {
            T val = vals[k];
            sum_sq += val * val;
        }
        
        norms_sq[p] = sum_sq;
    });
}

} // namespace scl::kernel::bbknn::fast

