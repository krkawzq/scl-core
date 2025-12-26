#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>

// =============================================================================
/// @file mmd_fast_impl.hpp
/// @brief Extreme Performance MMD
///
/// Ultra-optimized Maximum Mean Discrepancy with:
/// - Batch SIMD exp computation
/// - Cache-optimized kernel evaluation
/// - 4-way unrolling for distance computation
///
/// Note: These functions accept raw pointers and are generic for both
/// CustomSparse and VirtualSparse. The caller should extract the appropriate
/// data pointers before calling these functions.
///
/// Performance Target: 2-3x faster than generic
// =============================================================================

namespace scl::kernel::mmd::fast {

/// @brief Ultra-fast unary exp sum
///
/// Optimization: Batch SIMD on contiguous data
/// Generic for both CustomSparse and VirtualSparse (accepts raw pointers)
template <typename T>
SCL_FORCE_INLINE void unary_exp_sum_fast(
    const T* SCL_RESTRICT vals,
    Size nnz,
    T gamma,
    T* cache,
    T& out_sum
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    const auto v_gamma = s::Set(d, gamma);
    
    // 4-way unrolled SIMD loop
    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);
    
    size_t k = 0;
    for (; k + 4 * lanes <= nnz; k += 4 * lanes) {
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);
        
        auto v_sq0 = s::Mul(v0, v0);
        auto v_sq1 = s::Mul(v1, v1);
        auto v_sq2 = s::Mul(v2, v2);
        auto v_sq3 = s::Mul(v3, v3);
        
        auto v_exp0 = s::Exp(d, s::Neg(s::Mul(v_sq0, v_gamma)));
        auto v_exp1 = s::Exp(d, s::Neg(s::Mul(v_sq1, v_gamma)));
        auto v_exp2 = s::Exp(d, s::Neg(s::Mul(v_sq2, v_gamma)));
        auto v_exp3 = s::Exp(d, s::Neg(s::Mul(v_sq3, v_gamma)));
        
        s::Store(v_exp0, d, cache + k + 0 * lanes);
        s::Store(v_exp1, d, cache + k + 1 * lanes);
        s::Store(v_exp2, d, cache + k + 2 * lanes);
        s::Store(v_exp3, d, cache + k + 3 * lanes);
        
        v_sum0 = s::Add(v_sum0, v_exp0);
        v_sum1 = s::Add(v_sum1, v_exp1);
        v_sum2 = s::Add(v_sum2, v_exp2);
        v_sum3 = s::Add(v_sum3, v_exp3);
    }
    
    auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
    
    for (; k + lanes <= nnz; k += lanes) {
        auto v = s::Load(d, vals + k);
        auto v_sq = s::Mul(v, v);
        auto v_exp = s::Exp(d, s::Neg(s::Mul(v_sq, v_gamma)));
        s::Store(v_exp, d, cache + k);
        v_sum = s::Add(v_sum, v_exp);
    }
    
    T sum = s::GetLane(s::SumOfLanes(d, v_sum));
    
    for (; k < nnz; ++k) {
        T val = vals[k];
        T exp_term = std::exp(-gamma * val * val);
        cache[k] = exp_term;
        sum += exp_term;
    }
    
    out_sum = sum;
}

/// @brief Ultra-fast cross kernel sum
///
/// Optimization: 4-way unrolled distance computation
template <typename T>
SCL_FORCE_INLINE void cross_kernel_sum_fast(
    const T* SCL_RESTRICT vals_x, Size nnz_x,
    const T* SCL_RESTRICT vals_y, Size nnz_y,
    Size N_x, Size N_y,
    T gamma,
    T sum_x_unary,
    T sum_y_unary,
    T& out_sum
) {
    const Size zeros_x = N_x - nnz_x;
    const Size zeros_y = N_y - nnz_y;

    T sum = static_cast<T>(0.0);

    // Zero-Zero
    sum += static_cast<T>(zeros_x * zeros_y);

    // Zero-Val interactions
    if (zeros_x > 0) {
        sum += static_cast<T>(zeros_x) * sum_y_unary;
    }
    if (zeros_y > 0) {
        sum += static_cast<T>(zeros_y) * sum_x_unary;
    }

    // Val-Val with SIMD
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_gamma = s::Set(d, gamma);

    T cross_sum = static_cast<T>(0.0);

    // 4-way unrolled outer loop
    for (size_t i = 0; i < nnz_x; i += 4) {
        size_t batch_size = std::min(size_t(4), nnz_x - i);
        
        for (size_t b = 0; b < batch_size; ++b) {
            const T xi = vals_x[i + b];
            const auto v_xi = s::Set(d, xi);
            
            auto v_row_sum = s::Zero(d);
            size_t j = 0;

            for (; j + lanes <= nnz_y; j += lanes) {
                auto v_yj = s::Load(d, vals_y + j);
                auto v_diff = s::Sub(v_xi, v_yj);
                auto v_sq = s::Mul(v_diff, v_diff);
                auto v_exp = s::Exp(d, s::Neg(s::Mul(v_sq, v_gamma)));
                v_row_sum = s::Add(v_row_sum, v_exp);
            }
            
            cross_sum += s::GetLane(s::SumOfLanes(d, v_row_sum));

            for (; j < nnz_y; ++j) {
                T diff = xi - vals_y[j];
                cross_sum += std::exp(-gamma * diff * diff);
            }
        }
    }

    sum += cross_sum;
    out_sum = sum;
}

} // namespace scl::kernel::mmd::fast

