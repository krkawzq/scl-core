#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file sparse.hpp
/// @brief Sparse Matrix Statistics and Aggregations
///
/// Implements:
/// - Row/column sums
/// - Row/column means
/// - Row/column variances
/// - NNZ counts
///
/// All functions unified for CSR/CSC
// =============================================================================

namespace scl::kernel::sparse {

/// @brief Compute sums for primary dimension (unified for CSR/CSC)
///
/// @param matrix Input sparse matrix
/// @param output Output sums [size = primary_dim]
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void primary_sums(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(output.size() == static_cast<Size>(primary_dim), 
                  "Output size mismatch");
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        auto vals = scl::primary_values(matrix, static_cast<Index>(p));
        
        T sum = static_cast<T>(0);
        for (Size k = 0; k < vals.size(); ++k) {
            sum += vals[k];
        }
        output[p] = sum;
    });
}

/// @brief Compute means for primary dimension
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void primary_means(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = scl::primary_size(matrix);
    const T inv_n = static_cast<T>(1.0) / static_cast<T>(scl::secondary_size(matrix));
    
    SCL_CHECK_DIM(output.size() == static_cast<Size>(primary_dim), 
                  "Output size mismatch");
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        auto vals = scl::primary_values(matrix, static_cast<Index>(p));
        
        T sum = static_cast<T>(0);
        for (Size k = 0; k < vals.size(); ++k) {
            sum += vals[k];
        }
        output[p] = sum * inv_n;
    });
}

/// @brief Compute variances for primary dimension
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void primary_variances(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output,
    int ddof = 1
) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = scl::primary_size(matrix);
    const T N = static_cast<T>(scl::secondary_size(matrix));
    const T denom = N - static_cast<T>(ddof);
    
    SCL_CHECK_DIM(output.size() == static_cast<Size>(primary_dim), 
                  "Output size mismatch");
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        auto vals = scl::primary_values(matrix, static_cast<Index>(p));
        
        T sum = static_cast<T>(0);
        T sum_sq = static_cast<T>(0);
        
        for (Size k = 0; k < vals.size(); ++k) {
            T v = vals[k];
            sum += v;
            sum_sq += v * v;
        }
        
        T mu = sum / N;
        T var = 0.0;
        
        if (denom > 0) {
            var = (sum_sq - sum * mu) / denom;
        }
        
        if (var < 0) var = 0.0;
        
        output[p] = var;
    });
}

/// @brief Count non-zeros for primary dimension
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void primary_nnz_counts(
    const MatrixT& matrix,
    Array<Index> output
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(output.size() == static_cast<Size>(primary_dim), 
                  "Output size mismatch");
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = scl::primary_length(matrix, static_cast<Index>(p));
        output[p] = len;
    });
}

} // namespace scl::kernel::sparse
