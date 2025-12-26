#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
/// @file feature_fast_impl.hpp
/// @brief Extreme Performance Feature Statistics
///
/// Unified fast path for CustomSparse and VirtualSparse:
/// - 访问模式相同: 逐行统计
/// - 优化策略相同: 行内SIMD + 4-way展开
/// - 性能提升相似: 2-3x
///
/// 因此使用统一实现,编译器会根据类型优化
// =============================================================================

namespace scl::kernel::feature::fast {

/// @brief Ultra-fast standard moments (统一fast path)
///
/// 适用CustomSparse和VirtualSparse
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void standard_moments_fast(
    const MatrixT& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real denom = N - static_cast<Real>(ddof);
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        auto vals = scl::primary_values(matrix, static_cast<Index>(p));
        
        // 4-way unrolled SIMD
        auto v_sum0 = s::Zero(d);
        auto v_sum1 = s::Zero(d);
        auto v_sum2 = s::Zero(d);
        auto v_sum3 = s::Zero(d);
        
        auto v_ssq0 = s::Zero(d);
        auto v_ssq1 = s::Zero(d);
        auto v_ssq2 = s::Zero(d);
        auto v_ssq3 = s::Zero(d);
        
        size_t k = 0;
        for (; k + 4 * lanes <= vals.len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals.ptr + k + 0 * lanes);
            auto v1 = s::Load(d, vals.ptr + k + 1 * lanes);
            auto v2 = s::Load(d, vals.ptr + k + 2 * lanes);
            auto v3 = s::Load(d, vals.ptr + k + 3 * lanes);
            
            v_sum0 = s::Add(v_sum0, v0);
            v_sum1 = s::Add(v_sum1, v1);
            v_sum2 = s::Add(v_sum2, v2);
            v_sum3 = s::Add(v_sum3, v3);
            
            v_ssq0 = s::MulAdd(v0, v0, v_ssq0);
            v_ssq1 = s::MulAdd(v1, v1, v_ssq1);
            v_ssq2 = s::MulAdd(v2, v2, v_ssq2);
            v_ssq3 = s::MulAdd(v3, v3, v_ssq3);
        }
        
        auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
        auto v_ssq = s::Add(s::Add(v_ssq0, v_ssq1), s::Add(v_ssq2, v_ssq3));
        
        for (; k + lanes <= vals.len; k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            v_sum = s::Add(v_sum, v);
            v_ssq = s::MulAdd(v, v, v_ssq);
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));
        
        for (; k < vals.len; ++k) {
            Real v = static_cast<Real>(vals[k]);
            sum += v;
            sum_sq += v * v;
        }
        
        Real mu = sum / N;
        Real var = (denom > 0) ? ((sum_sq - sum * mu) / denom) : 0.0;
        if (var < 0) var = 0.0;
        
        out_means[p] = mu;
        out_vars[p] = var;
    });
}

} // namespace scl::kernel::feature::fast
