#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/gram.hpp"

#include <cmath>
#include <vector>

// =============================================================================
/// @file correlation.hpp
/// @brief Pearson Correlation Computation
///
/// Algorithm:
/// 1. Compute mean and std for each feature
/// 2. Compute Gram matrix G = X^T * X
/// 3. Transform: corr = (G/N - mu_i*mu_j) / (sigma_i*sigma_j)
///
/// Performance: O(nnz) + Gram computation
// =============================================================================

namespace scl::kernel::correlation {

namespace detail {

/// @brief Compute statistics for primary dimension (SIMD)
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void compute_stats(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> out_means,
    Array<typename MatrixT::ValueType> out_inv_stds
) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = scl::primary_size(matrix);
    const Size secondary_dim = static_cast<Size>(scl::secondary_size(matrix));
    const T inv_n = static_cast<T>(1.0) / static_cast<T>(secondary_dim);
    
    SCL_CHECK_DIM(out_means.size() == static_cast<Size>(primary_dim), "Means size mismatch");
    SCL_CHECK_DIM(out_inv_stds.size() == static_cast<Size>(primary_dim), "Inv_stds size mismatch");

    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index idx = static_cast<Index>(p);
        auto vals = scl::primary_values(matrix, idx);
        Index len = scl::primary_length(matrix, idx);
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

        auto v_sum = s::Zero(d);
        auto v_sq_sum = s::Zero(d);
        Index k = 0;

        for (; k + static_cast<Index>(lanes) <= len; k += static_cast<Index>(lanes)) {
            auto v = s::Load(d, vals.ptr + k);
            v_sum = s::Add(v_sum, v);
            v_sq_sum = s::Add(v_sq_sum, s::Mul(v, v));
        }
        
        T sum = s::GetLane(s::SumOfLanes(d, v_sum));
        T sq_sum = s::GetLane(s::SumOfLanes(d, v_sq_sum));

        for (; k < len; ++k) {
            T v = vals[k];
            sum += v;
            sq_sum += v * v;
        }

        T mean = sum * inv_n;
        T var = (sq_sum * inv_n) - (mean * mean);
        if (var < 0) var = 0;

        out_means[p] = mean;
        out_inv_stds[p] = (var > 0) ? (static_cast<T>(1.0) / std::sqrt(var)) : 0.0;
    });
}

} // namespace detail

/// @brief Compute Pearson correlation matrix (unified for CSR/CSC)
///
/// @param matrix Input sparse matrix
/// @param output Output dense correlation matrix [size = primary_dim^2]
template <typename MatrixT>
    requires AnySparse<MatrixT>
void pearson(const MatrixT& matrix, Array<typename MatrixT::ValueType> output) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = scl::primary_size(matrix);
    const Size secondary_dim = static_cast<Size>(scl::secondary_size(matrix));
    
    SCL_CHECK_DIM(output.size() == static_cast<Size>(primary_dim * primary_dim), 
                  "Pearson: Output size mismatch");

    // Allocate workspace for statistics
    std::vector<T> means(primary_dim);
    std::vector<T> inv_stds(primary_dim);

    // Compute statistics
    detail::compute_stats(
        matrix,
        Array<T>(means.data(), static_cast<Size>(primary_dim)),
        Array<T>(inv_stds.data(), static_cast<Size>(primary_dim))
    );

    // Compute Gram matrix
    scl::kernel::gram::gram(matrix, output);

    // Transform to correlation
    const T inv_n = static_cast<T>(1.0) / static_cast<T>(secondary_dim);

    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t i) {
        T* row_ptr = output.ptr + (i * static_cast<size_t>(primary_dim));
        T mu_i = means[i];
        T inv_sig_i = inv_stds[i];

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

        const auto v_inv_n = s::Set(d, inv_n);
        const auto v_mu_i = s::Set(d, mu_i);
        const auto v_inv_sig_i = s::Set(d, inv_sig_i);
        const auto v_one = s::Set(d, static_cast<T>(1.0));
        const auto v_neg_one = s::Set(d, static_cast<T>(-1.0));

        size_t j = 0;
        
        for (; j + lanes <= static_cast<size_t>(primary_dim); j += lanes) {
            auto v_g = s::Load(d, row_ptr + j);
            auto v_mu_j = s::Load(d, means.data() + j);
            auto v_inv_sig_j = s::Load(d, inv_stds.data() + j);

            auto v_cov = s::Sub(s::Mul(v_g, v_inv_n), s::Mul(v_mu_i, v_mu_j));
            auto v_norm = s::Mul(v_inv_sig_i, v_inv_sig_j);
            auto v_corr = s::Mul(v_cov, v_norm);

            v_corr = s::Min(v_corr, v_one);
            v_corr = s::Max(v_corr, v_neg_one);
            
            s::Store(v_corr, d, row_ptr + j);
        }

        for (; j < static_cast<size_t>(primary_dim); ++j) {
            T g_val = row_ptr[j];
            T mu_j = means[j];
            T inv_sig_j = inv_stds[j];

            T cov = (g_val * inv_n) - (mu_i * mu_j);
            T corr = cov * (inv_sig_i * inv_sig_j);

            if (corr > 1.0) corr = 1.0;
            if (corr < -1.0) corr = -1.0;
            if (inv_sig_i == 0.0 || inv_sig_j == 0.0) corr = 0.0;

            row_ptr[j] = corr;
        }
    });
}

} // namespace scl::kernel::correlation
