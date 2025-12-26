#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/simd.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <algorithm>

// =============================================================================
/// @file scale.hpp
/// @brief Matrix Scaling and Standardization Kernels
///
/// Implements `(X - mean) / std` transformation with optional clipping.
/// Supports Dense, CSR, and CSC layouts.
// =============================================================================

namespace scl::kernel::scale {

// =============================================================================
// Layer 1: Virtual Interface (ISparse-based, Generic but Slower)
// =============================================================================

/// @brief Standardize CSC matrix columns in-place (Virtual Interface).
///
/// Generic implementation using ISparse base class.
/// Works with any sparse matrix type but may have virtual call overhead.
///
/// Note: This modifies ONLY the stored non-zero elements.
/// If `zero_center=true`, this operation is mathematically approximate 
/// for sparse matrices (implicit zeros are not updated), 
/// which is standard behavior in Scanpy/Seurat to maintain sparsity.
///
/// Use `zero_center=false` for exact mathematical correctness on sparse data.
///
/// @param matrix CSC sparse matrix (via ISparse interface). Modified in-place.
/// @param means Column means (size = matrix.cols).
/// @param stds Column stds (size = matrix.cols).
/// @param max_value Clipping threshold (0 = no clipping).
/// @param zero_center If true, subtract mean. If false, only divide by std.
template <typename T>
SCL_FORCE_INLINE void standardize_csc(
    ICSC<T>& matrix,
    Span<const Real> means,
    Span<const Real> stds,
    Real max_value = 0.0,
    bool zero_center = true
) {
    const Index cols = matrix.cols();
    SCL_CHECK_DIM(means.size == static_cast<Size>(cols), "Means dim mismatch");
    
    scl::threading::parallel_for(0, static_cast<size_t>(cols), [&](size_t c) {
        Real mu = means[c];
        Real sigma = stds[c];
        Real inv_sigma = (sigma != 0.0) ? (1.0 / sigma) : 0.0;
        
        if (sigma == 0.0) return;

        auto vals = matrix.primary_values(static_cast<Index>(c));
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        const auto v_mu = s::Set(d, mu);
        const auto v_inv_sigma = s::Set(d, inv_sigma);
        const auto v_max = s::Set(d, max_value);
        const auto v_min = s::Set(d, -max_value);
        const bool do_clip = (max_value > 0.0);

        size_t k = 0;
        for (; k + lanes <= vals.size; k += lanes) {
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

        for (; k < vals.size; ++k) {
            Real v = vals[k];
            if (zero_center) v -= mu;
            v *= inv_sigma;
            if (do_clip) {
                if (v > max_value) v = max_value;
                else if (v < -max_value) v = -max_value;
            }
            vals[k] = v;
        }
    });
}

/// @brief Standardize CSR matrix columns in-place (Virtual Interface).
///
/// Generic implementation using ISparse base class.
/// Works with any sparse matrix type but may have virtual call overhead.
///
/// Performance Warning: Row-wise iteration accessing column statistics (means/stds)
/// causes random memory access (gather) on the stats arrays. 
/// Less efficient than CSC or Dense.
///
/// @param matrix CSR sparse matrix (via ISparse interface). Modified in-place.
/// @param means Column means (size = matrix.cols).
/// @param stds Column stds (size = matrix.cols).
/// @param max_value Clipping threshold (0 = no clipping).
/// @param zero_center If true, subtract mean. If false, only divide by std.
template <typename T>
SCL_FORCE_INLINE void standardize_csr(
    ICSR<T>& matrix,
    Span<const Real> means,
    Span<const Real> stds,
    Real max_value = 0.0,
    bool zero_center = true
) {
    const Index cols = matrix.cols();
    SCL_CHECK_DIM(means.size == static_cast<Size>(cols), "Means dim mismatch");

    const bool do_clip = (max_value > 0.0);
    const Index rows = matrix.rows();

    scl::threading::parallel_for(0, static_cast<size_t>(rows), [&](size_t i) {
        auto cols_idx = matrix.primary_indices(static_cast<Index>(i));
        auto vals = matrix.primary_values(static_cast<Index>(i));
        
        for (size_t k = 0; k < vals.size; ++k) {
            Index col_idx = cols_idx[k];
            Real val = vals[k];
            
            Real mu = means[col_idx];
            Real sigma = stds[col_idx];
            
            if (sigma == 0.0) continue;

            if (zero_center) val -= mu;
            val /= sigma;
            
            if (do_clip) {
                if (val > max_value) val = max_value;
                else if (val < -max_value) val = -max_value;
            }
            
            vals[k] = val;
        }
    });
}

// =============================================================================
// Layer 2: Concept-Based (CSCLike/CSRLike, Optimized for Custom/Virtual)
// =============================================================================

// =============================================================================
// 1. Dense Matrix Standardization
// =============================================================================

/// @brief Standardize dense matrix columns in-place (Generic Dense matrices).
///
/// Operation: X[i,j] = clamp((X[i,j] - mu[j]) * inv_std[j], -max, max)
///
/// @tparam MatrixT Any Dense-like matrix type
/// @param matrix      Dense matrix (Row-Major). Modified in-place.
/// @param means       Column means (size = matrix.cols).
/// @param stds        Column stds (size = matrix.cols).
/// @param max_value   Clipping threshold (0 = no clipping).
/// @param zero_center If true, subtract mean. If false, only divide by std.
template <DenseLike MatrixT>
SCL_FORCE_INLINE void standardize_dense(
    MatrixT matrix,
    Span<const Real> means,
    Span<const Real> stds,
    Real max_value = 0.0,
    bool zero_center = true
) {
    const Index cols = scl::cols(matrix);
    SCL_CHECK_DIM(means.size == static_cast<Size>(cols), "Means dim mismatch");
    SCL_CHECK_DIM(stds.size == static_cast<Size>(cols), "Stds dim mismatch");

    const bool do_clip = (max_value > 0.0);
    const Index rows = scl::rows(matrix);
    
    scl::threading::parallel_for(0, rows, [&](size_t i) {
        Real* row_ptr = matrix.ptr + (i * cols);
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        // Broadcast clip values
        const auto v_max = s::Set(d, max_value);
        const auto v_min = s::Set(d, -max_value);

        size_t j = 0;
        
        // SIMD Loop over columns
        for (; j + lanes <= static_cast<size_t>(cols); j += lanes) {
            auto v_val  = s::Load(d, row_ptr + j);
            auto v_mean = s::Load(d, means.ptr + j);
            auto v_std  = s::Load(d, stds.ptr + j);

            // 1. Center
            if (zero_center) {
                v_val = s::Sub(v_val, v_mean);
            }

            // 2. Scale
            // Note: SIMD path assumes stds are sanitized (no zero stds).
            // Scalar tail handles std=0 explicitly for robustness.
            v_val = s::Div(v_val, v_std);

            // 3. Clip
            if (do_clip) {
                v_val = s::Min(v_val, v_max);
                v_val = s::Max(v_val, v_min);
            }

            s::Store(v_val, d, row_ptr + j);
        }

        // Scalar Tail
        for (; j < static_cast<size_t>(cols); ++j) {
            Real val = row_ptr[j];
            Real m = means[j];
            Real s = stds[j];

            if (zero_center) val -= m;
            if (s != 0) val /= s;

            if (do_clip) {
                if (val > max_value) val = max_value;
                else if (val < -max_value) val = -max_value;
            }
            row_ptr[j] = val;
        }
    });
}

// =============================================================================
// 2. CSC Matrix Standardization (Sparse)
// =============================================================================

/// @brief Standardize CSC matrix columns in-place (Concept-based, Optimized, CSC).
///
/// High-performance implementation for CSCLike matrices.
/// Uses unified accessors for zero-overhead abstraction.
///
/// Note: This modifies ONLY the stored non-zero elements.
/// If `zero_center=true`, this operation is mathematically approximate 
/// for sparse matrices (implicit zeros are not updated), 
/// which is standard behavior in Scanpy/Seurat to maintain sparsity.
///
/// Use `zero_center=false` for exact mathematical correctness on sparse data.
///
/// @tparam MatrixT Any CSC-like matrix type (CustomSparse or VirtualSparse)
template <CSCLike MatrixT>
SCL_FORCE_INLINE void standardize_csc(
    MatrixT matrix,
    Span<const Real> means,
    Span<const Real> stds,
    Real max_value = 0.0,
    bool zero_center = true
) {
    const Index cols = scl::cols(matrix);
    SCL_CHECK_DIM(means.size == static_cast<Size>(cols), "Means dim mismatch");
    
    // CSC is Column-Major. This is the most efficient layout for scaling columns.
    // We can parallelize over columns directly.
    
    scl::threading::parallel_for(0, static_cast<size_t>(cols), [&](size_t c) {
        Real mu = means[c];
        Real sigma = stds[c];
        Real inv_sigma = (sigma != 0.0) ? (1.0 / sigma) : 0.0;
        
        // Skip constant columns if sigma is 0
        if (sigma == 0.0) return;

        auto vals = matrix.col_values(c);
        
        // Process all non-zeros in this column
        // We can vectorize this loop easily because mu and sigma are constant for the whole column!
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        const auto v_mu = s::Set(d, mu);
        const auto v_inv_sigma = s::Set(d, inv_sigma);
        const auto v_max = s::Set(d, max_value);
        const auto v_min = s::Set(d, -max_value);
        const bool do_clip = (max_value > 0.0);

        size_t k = 0;
        for (; k + lanes <= vals.size; k += lanes) {
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

        // Tail
        for (; k < vals.size; ++k) {
            Real v = vals[k];
            if (zero_center) v -= mu;
            v *= inv_sigma;
            if (do_clip) {
                if (v > max_value) v = max_value;
                else if (v < -max_value) v = -max_value;
            }
            vals[k] = v;
        }
    });
}

// =============================================================================
// 3. CSR Matrix Standardization (Sparse)
// =============================================================================

/// @brief Standardize CSR matrix columns in-place (Concept-based, Optimized, CSR).
///
/// High-performance implementation for CSRLike matrices.
/// Uses unified accessors for zero-overhead abstraction.
///
/// Performance Warning: Row-wise iteration accessing column statistics (means/stds)
/// causes random memory access (gather) on the stats arrays. 
/// Less efficient than CSC or Dense.
///
/// @tparam MatrixT Any CSR-like matrix type (CustomSparse or VirtualSparse)
template <CSRLike MatrixT>
SCL_FORCE_INLINE void standardize_csr(
    MatrixT matrix,
    Span<const Real> means,
    Span<const Real> stds,
    Real max_value = 0.0,
    bool zero_center = true
) {
    const Index cols = scl::cols(matrix);
    SCL_CHECK_DIM(means.size == static_cast<Size>(cols), "Means dim mismatch");

    const bool do_clip = (max_value > 0.0);
    const Index rows = scl::rows(matrix);

    // Parallelize over rows
    scl::threading::parallel_for(0, static_cast<size_t>(rows), [&](size_t i) {
        auto cols = matrix.row_indices(i);
        auto vals = matrix.row_values(i);
        
        // Scalar loop is usually best here due to indirect addressing of means/stds
        for (size_t k = 0; k < vals.size; ++k) {
            Index col_idx = cols[k];
            Real val = vals[k];
            
            Real mu = means[col_idx];     // Random access!
            Real sigma = stds[col_idx];   // Random access!
            
            if (sigma == 0.0) continue;

            if (zero_center) val -= mu;
            val /= sigma;
            
            if (do_clip) {
                if (val > max_value) val = max_value;
                else if (val < -max_value) val = -max_value;
            }
            
            vals[k] = val;
        }
    });
}

} // namespace scl::kernel::scale
