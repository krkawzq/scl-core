#pragma once

// =============================================================================
/// @file math.hpp
/// @brief Mathematical Operations on Memory-Mapped Matrices
///
/// This file provides statistical and mathematical operations optimized for
/// memory-mapped sparse matrices with on-demand paging.
///
/// Operations:
///
/// 1. Row-wise Statistics: sum, mean, var, max, min
/// 2. Column-wise Statistics: sum, mean, nnz (for CSR)
/// 3. Global Statistics: sum, mean
/// 4. Normalization: L1, L2, max
/// 5. Element-wise: log1p, scale
/// 6. Linear Algebra: SpMV, dot products
/// 7. Filtering: threshold, top-k
///
/// Performance Optimizations:
///
/// - Parallel processing via scl::threading::parallel_for
/// - SIMD-friendly loop structures
/// - Minimized cache invalidation
/// - Thread-local accumulators for reductions
// =============================================================================

#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/mmap/configuration.hpp"
#include "scl/mmap/matrix.hpp"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <atomic>

namespace scl::mmap {

// =============================================================================
// SECTION 1: Row-wise Statistics (Parallel)
// =============================================================================

/// @brief Compute sum of each row (parallel)
template <typename T, bool IsCSR>
void row_sum(const MappedSparse<T, IsCSR>& mat, T* SCL_RESTRICT out) {
    const scl::Index rows = mat.rows();
    
    scl::threading::parallel_for(0, static_cast<std::size_t>(rows), [&](std::size_t i) {
        const auto idx = static_cast<scl::Index>(i);
        scl::Array<T> vals;
        
        if constexpr (IsCSR) {
            vals = mat.row_values(idx);
        } else {
            vals = mat.col_values(idx);
        }
        
        T sum = T(0);
        const T* SCL_RESTRICT ptr = vals.data();
        const scl::Size n = vals.size();
        
        // SIMD-friendly reduction
        for (scl::Size k = 0; k < n; ++k) {
            sum += ptr[k];
        }
        
        out[i] = sum;
    });
}

/// @brief Compute mean of each row (parallel)
template <typename T, bool IsCSR>
void row_mean(const MappedSparse<T, IsCSR>& mat, T* SCL_RESTRICT out, 
              bool count_zeros = true) {
    const scl::Index rows = mat.rows();
    const scl::Index cols = mat.cols();
    
    scl::threading::parallel_for(0, static_cast<std::size_t>(rows), [&](std::size_t i) {
        const auto idx = static_cast<scl::Index>(i);
        scl::Array<T> vals;
        
        if constexpr (IsCSR) {
            vals = mat.row_values(idx);
        } else {
            vals = mat.col_values(idx);
        }
        
        T sum = T(0);
        const T* SCL_RESTRICT ptr = vals.data();
        const scl::Size n = vals.size();
        
        for (scl::Size k = 0; k < n; ++k) {
            sum += ptr[k];
        }
        
        const scl::Index divisor = count_zeros ? cols : static_cast<scl::Index>(n);
        out[i] = divisor > 0 ? sum / static_cast<T>(divisor) : T(0);
    });
}

/// @brief Compute variance of each row (parallel)
template <typename T, bool IsCSR>
void row_var(const MappedSparse<T, IsCSR>& mat, T* SCL_RESTRICT out,
             const T* means = nullptr, bool count_zeros = true) {
    const scl::Index rows = mat.rows();
    const scl::Index cols = mat.cols();
    
    // Compute means if not provided
    std::vector<T> mean_buf;
    if (!means) {
        mean_buf.resize(static_cast<std::size_t>(rows));
        row_mean<T, IsCSR>(mat, mean_buf.data(), count_zeros);
        means = mean_buf.data();
    }
    
    scl::threading::parallel_for(0, static_cast<std::size_t>(rows), [&](std::size_t i) {
        const auto idx = static_cast<scl::Index>(i);
        scl::Array<T> vals;
        
        if constexpr (IsCSR) {
            vals = mat.row_values(idx);
        } else {
            vals = mat.col_values(idx);
        }
        
        const T mean = means[i];
        T sum_sq = T(0);
        const T* SCL_RESTRICT ptr = vals.data();
        const scl::Size n = vals.size();
        
        for (scl::Size k = 0; k < n; ++k) {
            const T diff = ptr[k] - mean;
            sum_sq += diff * diff;
        }
        
        // Add zero contribution
        if (count_zeros) {
            const scl::Index num_zeros = cols - static_cast<scl::Index>(n);
            sum_sq += static_cast<T>(num_zeros) * mean * mean;
        }
        
        const scl::Index divisor = count_zeros ? cols : static_cast<scl::Index>(n);
        out[i] = divisor > 0 ? sum_sq / static_cast<T>(divisor) : T(0);
    });
}

/// @brief Compute max of each row (parallel)
template <typename T, bool IsCSR>
void row_max(const MappedSparse<T, IsCSR>& mat, T* SCL_RESTRICT out, 
             bool consider_zeros = true) {
    const scl::Index rows = mat.rows();
    
    scl::threading::parallel_for(0, static_cast<std::size_t>(rows), [&](std::size_t i) {
        const auto idx = static_cast<scl::Index>(i);
        scl::Array<T> vals;
        scl::Index len;
        
        if constexpr (IsCSR) {
            vals = mat.row_values(idx);
            len = mat.row_length(idx);
        } else {
            vals = mat.col_values(idx);
            len = mat.col_length(idx);
        }
        
        if (vals.size() == 0) {
            out[i] = T(0);
            return;
        }
        
        T max_val = vals[0];
        const T* SCL_RESTRICT ptr = vals.data();
        const scl::Size n = vals.size();
        
        for (scl::Size k = 1; k < n; ++k) {
            max_val = std::max(max_val, ptr[k]);
        }
        
        if (consider_zeros && len < mat.cols()) {
            max_val = std::max(max_val, T(0));
        }
        
        out[i] = max_val;
    });
}

/// @brief Compute min of each row (parallel)
template <typename T, bool IsCSR>
void row_min(const MappedSparse<T, IsCSR>& mat, T* SCL_RESTRICT out,
             bool consider_zeros = true) {
    const scl::Index rows = mat.rows();
    
    scl::threading::parallel_for(0, static_cast<std::size_t>(rows), [&](std::size_t i) {
        const auto idx = static_cast<scl::Index>(i);
        scl::Array<T> vals;
        
        if constexpr (IsCSR) {
            vals = mat.row_values(idx);
        } else {
            vals = mat.col_values(idx);
        }
        
        if (vals.size() == 0) {
            out[i] = T(0);
            return;
        }
        
        T min_val = vals[0];
        const T* SCL_RESTRICT ptr = vals.data();
        const scl::Size n = vals.size();
        
        for (scl::Size k = 1; k < n; ++k) {
            min_val = std::min(min_val, ptr[k]);
        }
        
        if (consider_zeros && static_cast<scl::Index>(n) < mat.cols()) {
            min_val = std::min(min_val, T(0));
        }
        
        out[i] = min_val;
    });
}

// =============================================================================
// SECTION 2: Column-wise Statistics (CSR)
// =============================================================================

/// @brief Compute sum of each column (CSR format, parallel reduction)
template <typename T>
void col_sum_csr(const MappedSparse<T, true>& mat, T* SCL_RESTRICT out) {
    const scl::Index rows = mat.rows();
    const scl::Index cols = mat.cols();
    
    // Initialize output
    std::memset(out, 0, static_cast<std::size_t>(cols) * sizeof(T));
    
    // Sequential accumulation (column updates have dependencies)
    // Could use thread-local accumulators + merge, but overhead often not worth it
    for (scl::Index i = 0; i < rows; ++i) {
        auto vals = mat.row_values(i);
        auto inds = mat.row_indices(i);
        
        const T* SCL_RESTRICT vptr = vals.data();
        const scl::Index* SCL_RESTRICT iptr = inds.data();
        const scl::Size n = vals.size();
        
        for (scl::Size k = 0; k < n; ++k) {
            out[iptr[k]] += vptr[k];
        }
    }
}

/// @brief Compute mean of each column (CSR)
template <typename T>
void col_mean_csr(const MappedSparse<T, true>& mat, T* SCL_RESTRICT out,
                  bool count_zeros = true) {
    const scl::Index rows = mat.rows();
    const scl::Index cols = mat.cols();
    
    col_sum_csr(mat, out);
    
    const T divisor = count_zeros ? static_cast<T>(rows) : T(1);
    
    scl::threading::parallel_for(0, static_cast<std::size_t>(cols), [&](std::size_t j) {
        out[j] /= divisor;
    });
}

/// @brief Count non-zeros per column (CSR)
template <typename T>
void col_nnz_csr(const MappedSparse<T, true>& mat, scl::Index* SCL_RESTRICT out) {
    const scl::Index rows = mat.rows();
    const scl::Index cols = mat.cols();
    
    std::memset(out, 0, static_cast<std::size_t>(cols) * sizeof(scl::Index));
    
    for (scl::Index i = 0; i < rows; ++i) {
        auto inds = mat.row_indices(i);
        const scl::Index* SCL_RESTRICT iptr = inds.data();
        const scl::Size n = inds.size();
        
        for (scl::Size k = 0; k < n; ++k) {
            ++out[iptr[k]];
        }
    }
}

// =============================================================================
// SECTION 3: Global Statistics
// =============================================================================

/// @brief Compute global sum (parallel with reduction)
template <typename T, bool IsCSR>
T global_sum(const MappedSparse<T, IsCSR>& mat) {
    const scl::Index n = IsCSR ? mat.rows() : mat.cols();
    
    // Thread-local sums
    std::atomic<T> total{T(0)};
    
    scl::threading::parallel_for(0, static_cast<std::size_t>(n), [&](std::size_t i) {
        const auto idx = static_cast<scl::Index>(i);
        scl::Array<T> vals;
        
        if constexpr (IsCSR) {
            vals = mat.row_values(idx);
        } else {
            vals = mat.col_values(idx);
        }
        
        T local_sum = T(0);
        const T* SCL_RESTRICT ptr = vals.data();
        const scl::Size len = vals.size();
        
        for (scl::Size k = 0; k < len; ++k) {
            local_sum += ptr[k];
        }
        
        // Atomic add (could use thread-local + final merge for better perf)
        T expected = total.load(std::memory_order_relaxed);
        while (!total.compare_exchange_weak(expected, expected + local_sum,
                                            std::memory_order_relaxed)) {}
    });
    
    return total.load(std::memory_order_relaxed);
}

/// @brief Compute global mean
template <typename T, bool IsCSR>
T global_mean(const MappedSparse<T, IsCSR>& mat, bool count_zeros = true) {
    const T sum = global_sum<T, IsCSR>(mat);
    const std::size_t count = count_zeros
        ? static_cast<std::size_t>(mat.rows()) * static_cast<std::size_t>(mat.cols())
        : static_cast<std::size_t>(mat.nnz());
    return count > 0 ? sum / static_cast<T>(count) : T(0);
}

// =============================================================================
// SECTION 4: Normalization (Parallel)
// =============================================================================

/// @brief L1 row normalization (parallel)
template <typename T, bool IsCSR>
void normalize_l1_rows(const MappedSparse<T, IsCSR>& mat,
                       T* SCL_RESTRICT data_out,
                       scl::Index* SCL_RESTRICT indices_out,
                       scl::Index* SCL_RESTRICT indptr_out) {
    const scl::Index n = IsCSR ? mat.rows() : mat.cols();
    
    // First pass: compute indptr (sequential for prefix sum)
    indptr_out[0] = 0;
    for (scl::Index i = 0; i < n; ++i) {
        scl::Index len;
        if constexpr (IsCSR) {
            len = mat.row_length(i);
        } else {
            len = mat.col_length(i);
        }
        indptr_out[i + 1] = indptr_out[i] + len;
    }
    
    // Second pass: normalize rows (parallel)
    scl::threading::parallel_for(0, static_cast<std::size_t>(n), [&](std::size_t i) {
        const auto idx = static_cast<scl::Index>(i);
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;
        
        if constexpr (IsCSR) {
            vals = mat.row_values(idx);
            inds = mat.row_indices(idx);
        } else {
            vals = mat.col_values(idx);
            inds = mat.col_indices(idx);
        }
        
        const T* SCL_RESTRICT vptr = vals.data();
        const scl::Index* SCL_RESTRICT iptr = inds.data();
        const scl::Size len = vals.size();
        
        // Compute L1 norm
        T norm = T(0);
        for (scl::Size k = 0; k < len; ++k) {
            norm += std::abs(vptr[k]);
        }
        
        const T scale = norm > T(0) ? T(1) / norm : T(0);
        const scl::Index offset = indptr_out[i];
        
        for (scl::Size k = 0; k < len; ++k) {
            data_out[offset + static_cast<scl::Index>(k)] = vptr[k] * scale;
            indices_out[offset + static_cast<scl::Index>(k)] = iptr[k];
        }
    });
}

/// @brief L2 row normalization (parallel)
template <typename T, bool IsCSR>
void normalize_l2_rows(const MappedSparse<T, IsCSR>& mat,
                       T* SCL_RESTRICT data_out,
                       scl::Index* SCL_RESTRICT indices_out,
                       scl::Index* SCL_RESTRICT indptr_out) {
    const scl::Index n = IsCSR ? mat.rows() : mat.cols();
    
    // First pass: compute indptr
    indptr_out[0] = 0;
    for (scl::Index i = 0; i < n; ++i) {
        scl::Index len;
        if constexpr (IsCSR) {
            len = mat.row_length(i);
        } else {
            len = mat.col_length(i);
        }
        indptr_out[i + 1] = indptr_out[i] + len;
    }
    
    // Second pass: normalize
    scl::threading::parallel_for(0, static_cast<std::size_t>(n), [&](std::size_t i) {
        const auto idx = static_cast<scl::Index>(i);
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;
        
        if constexpr (IsCSR) {
            vals = mat.row_values(idx);
            inds = mat.row_indices(idx);
        } else {
            vals = mat.col_values(idx);
            inds = mat.col_indices(idx);
        }
        
        const T* SCL_RESTRICT vptr = vals.data();
        const scl::Index* SCL_RESTRICT iptr = inds.data();
        const scl::Size len = vals.size();
        
        // Compute L2 norm
        T sum_sq = T(0);
        for (scl::Size k = 0; k < len; ++k) {
            sum_sq += vptr[k] * vptr[k];
        }
        const T norm = std::sqrt(sum_sq);
        const T scale = norm > T(0) ? T(1) / norm : T(0);
        
        const scl::Index offset = indptr_out[i];
        
        for (scl::Size k = 0; k < len; ++k) {
            data_out[offset + static_cast<scl::Index>(k)] = vptr[k] * scale;
            indices_out[offset + static_cast<scl::Index>(k)] = iptr[k];
        }
    });
}

/// @brief Max row normalization (parallel)
template <typename T, bool IsCSR>
void normalize_max_rows(const MappedSparse<T, IsCSR>& mat,
                        T* SCL_RESTRICT data_out,
                        scl::Index* SCL_RESTRICT indices_out,
                        scl::Index* SCL_RESTRICT indptr_out) {
    const scl::Index n = IsCSR ? mat.rows() : mat.cols();
    
    indptr_out[0] = 0;
    for (scl::Index i = 0; i < n; ++i) {
        scl::Index len;
        if constexpr (IsCSR) {
            len = mat.row_length(i);
        } else {
            len = mat.col_length(i);
        }
        indptr_out[i + 1] = indptr_out[i] + len;
    }
    
    scl::threading::parallel_for(0, static_cast<std::size_t>(n), [&](std::size_t i) {
        const auto idx = static_cast<scl::Index>(i);
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;
        
        if constexpr (IsCSR) {
            vals = mat.row_values(idx);
            inds = mat.row_indices(idx);
        } else {
            vals = mat.col_values(idx);
            inds = mat.col_indices(idx);
        }
        
        const T* SCL_RESTRICT vptr = vals.data();
        const scl::Index* SCL_RESTRICT iptr = inds.data();
        const scl::Size len = vals.size();
        
        T max_val = T(0);
        for (scl::Size k = 0; k < len; ++k) {
            max_val = std::max(max_val, std::abs(vptr[k]));
        }
        
        const T scale = max_val > T(0) ? T(1) / max_val : T(0);
        const scl::Index offset = indptr_out[i];
        
        for (scl::Size k = 0; k < len; ++k) {
            data_out[offset + static_cast<scl::Index>(k)] = vptr[k] * scale;
            indices_out[offset + static_cast<scl::Index>(k)] = iptr[k];
        }
    });
}

// =============================================================================
// SECTION 5: Element-wise Operations (Parallel)
// =============================================================================

/// @brief log1p transform (parallel)
template <typename T, bool IsCSR>
void log1p_transform(const MappedSparse<T, IsCSR>& mat,
                     T* SCL_RESTRICT data_out,
                     scl::Index* SCL_RESTRICT indices_out,
                     scl::Index* SCL_RESTRICT indptr_out) {
    const scl::Index n = IsCSR ? mat.rows() : mat.cols();
    
    indptr_out[0] = 0;
    for (scl::Index i = 0; i < n; ++i) {
        scl::Index len;
        if constexpr (IsCSR) {
            len = mat.row_length(i);
        } else {
            len = mat.col_length(i);
        }
        indptr_out[i + 1] = indptr_out[i] + len;
    }
    
    scl::threading::parallel_for(0, static_cast<std::size_t>(n), [&](std::size_t i) {
        const auto idx = static_cast<scl::Index>(i);
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;
        
        if constexpr (IsCSR) {
            vals = mat.row_values(idx);
            inds = mat.row_indices(idx);
        } else {
            vals = mat.col_values(idx);
            inds = mat.col_indices(idx);
        }
        
        const T* SCL_RESTRICT vptr = vals.data();
        const scl::Index* SCL_RESTRICT iptr = inds.data();
        const scl::Size len = vals.size();
        const scl::Index offset = indptr_out[i];
        
        for (scl::Size k = 0; k < len; ++k) {
            data_out[offset + static_cast<scl::Index>(k)] = std::log1p(vptr[k]);
            indices_out[offset + static_cast<scl::Index>(k)] = iptr[k];
        }
    });
}

/// @brief Scale transform (parallel)
template <typename T, bool IsCSR>
void scale_transform(const MappedSparse<T, IsCSR>& mat,
                     T scale_factor,
                     T* SCL_RESTRICT data_out,
                     scl::Index* SCL_RESTRICT indices_out,
                     scl::Index* SCL_RESTRICT indptr_out) {
    const scl::Index n = IsCSR ? mat.rows() : mat.cols();
    
    indptr_out[0] = 0;
    for (scl::Index i = 0; i < n; ++i) {
        scl::Index len;
        if constexpr (IsCSR) {
            len = mat.row_length(i);
        } else {
            len = mat.col_length(i);
        }
        indptr_out[i + 1] = indptr_out[i] + len;
    }
    
    scl::threading::parallel_for(0, static_cast<std::size_t>(n), [&](std::size_t i) {
        const auto idx = static_cast<scl::Index>(i);
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;
        
        if constexpr (IsCSR) {
            vals = mat.row_values(idx);
            inds = mat.row_indices(idx);
        } else {
            vals = mat.col_values(idx);
            inds = mat.col_indices(idx);
        }
        
        const T* SCL_RESTRICT vptr = vals.data();
        const scl::Index* SCL_RESTRICT iptr = inds.data();
        const scl::Size len = vals.size();
        const scl::Index offset = indptr_out[i];
        
        for (scl::Size k = 0; k < len; ++k) {
            data_out[offset + static_cast<scl::Index>(k)] = vptr[k] * scale_factor;
            indices_out[offset + static_cast<scl::Index>(k)] = iptr[k];
        }
    });
}

/// @brief Scale rows by per-row factors (parallel)
template <typename T, bool IsCSR>
void scale_rows(const MappedSparse<T, IsCSR>& mat,
                const T* SCL_RESTRICT row_factors,
                T* SCL_RESTRICT data_out,
                scl::Index* SCL_RESTRICT indices_out,
                scl::Index* SCL_RESTRICT indptr_out) {
    const scl::Index n = IsCSR ? mat.rows() : mat.cols();
    
    indptr_out[0] = 0;
    for (scl::Index i = 0; i < n; ++i) {
        scl::Index len;
        if constexpr (IsCSR) {
            len = mat.row_length(i);
        } else {
            len = mat.col_length(i);
        }
        indptr_out[i + 1] = indptr_out[i] + len;
    }
    
    scl::threading::parallel_for(0, static_cast<std::size_t>(n), [&](std::size_t i) {
        const auto idx = static_cast<scl::Index>(i);
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;
        
        if constexpr (IsCSR) {
            vals = mat.row_values(idx);
            inds = mat.row_indices(idx);
        } else {
            vals = mat.col_values(idx);
            inds = mat.col_indices(idx);
        }
        
        const T factor = row_factors[i];
        const T* SCL_RESTRICT vptr = vals.data();
        const scl::Index* SCL_RESTRICT iptr = inds.data();
        const scl::Size len = vals.size();
        const scl::Index offset = indptr_out[i];
        
        for (scl::Size k = 0; k < len; ++k) {
            data_out[offset + static_cast<scl::Index>(k)] = vptr[k] * factor;
            indices_out[offset + static_cast<scl::Index>(k)] = iptr[k];
        }
    });
}

/// @brief Scale columns by per-column factors (CSR, parallel)
template <typename T>
void scale_cols_csr(const MappedSparse<T, true>& mat,
                    const T* SCL_RESTRICT col_factors,
                    T* SCL_RESTRICT data_out,
                    scl::Index* SCL_RESTRICT indices_out,
                    scl::Index* SCL_RESTRICT indptr_out) {
    const scl::Index rows = mat.rows();
    
    indptr_out[0] = 0;
    for (scl::Index i = 0; i < rows; ++i) {
        indptr_out[i + 1] = indptr_out[i] + mat.row_length(i);
    }
    
    scl::threading::parallel_for(0, static_cast<std::size_t>(rows), [&](std::size_t i) {
        auto vals = mat.row_values(static_cast<scl::Index>(i));
        auto inds = mat.row_indices(static_cast<scl::Index>(i));
        
        const T* SCL_RESTRICT vptr = vals.data();
        const scl::Index* SCL_RESTRICT iptr = inds.data();
        const scl::Size len = vals.size();
        const scl::Index offset = indptr_out[i];
        
        for (scl::Size k = 0; k < len; ++k) {
            const scl::Index j = iptr[k];
            data_out[offset + static_cast<scl::Index>(k)] = vptr[k] * col_factors[j];
            indices_out[offset + static_cast<scl::Index>(k)] = j;
        }
    });
}

// =============================================================================
// SECTION 6: Sparse-Dense Operations (Parallel)
// =============================================================================

/// @brief Sparse matrix-vector multiply: y = A * x (parallel)
template <typename T, bool IsCSR>
void spmv(const MappedSparse<T, IsCSR>& mat,
          const T* SCL_RESTRICT x,
          T* SCL_RESTRICT y) {
    if constexpr (IsCSR) {
        const scl::Index rows = mat.rows();
        
        scl::threading::parallel_for(0, static_cast<std::size_t>(rows), [&](std::size_t i) {
            auto vals = mat.row_values(static_cast<scl::Index>(i));
            auto inds = mat.row_indices(static_cast<scl::Index>(i));
            
            const T* SCL_RESTRICT vptr = vals.data();
            const scl::Index* SCL_RESTRICT iptr = inds.data();
            const scl::Size n = vals.size();
            
            T sum = T(0);
            for (scl::Size k = 0; k < n; ++k) {
                sum += vptr[k] * x[iptr[k]];
            }
            y[i] = sum;
        });
    } else {
        // CSC: y = A * x (column-wise)
        const scl::Index rows = mat.rows();
        const scl::Index cols = mat.cols();
        
        std::memset(y, 0, static_cast<std::size_t>(rows) * sizeof(T));
        
        // Sequential for column accumulation (thread-safety)
        for (scl::Index j = 0; j < cols; ++j) {
            auto vals = mat.col_values(j);
            auto inds = mat.col_indices(j);
            
            const T* SCL_RESTRICT vptr = vals.data();
            const scl::Index* SCL_RESTRICT iptr = inds.data();
            const scl::Size n = vals.size();
            const T xj = x[j];
            
            for (scl::Size k = 0; k < n; ++k) {
                y[iptr[k]] += vptr[k] * xj;
            }
        }
    }
}

// =============================================================================
// SECTION 7: Dot Products
// =============================================================================

/// @brief Dot product of two rows
template <typename T, bool IsCSR>
T row_dot(const MappedSparse<T, IsCSR>& mat, scl::Index i, scl::Index j) {
    scl::Array<T> vals_i, vals_j;
    scl::Array<scl::Index> inds_i, inds_j;
    
    if constexpr (IsCSR) {
        vals_i = mat.row_values(i);
        inds_i = mat.row_indices(i);
        vals_j = mat.row_values(j);
        inds_j = mat.row_indices(j);
    } else {
        vals_i = mat.col_values(i);
        inds_i = mat.col_indices(i);
        vals_j = mat.col_values(j);
        inds_j = mat.col_indices(j);
    }
    
    // Merge-based dot product
    T dot = T(0);
    scl::Size pi = 0, pj = 0;
    
    while (pi < inds_i.size() && pj < inds_j.size()) {
        const scl::Index ci = inds_i[static_cast<scl::Index>(pi)];
        const scl::Index cj = inds_j[static_cast<scl::Index>(pj)];
        
        if (ci == cj) {
            dot += vals_i[static_cast<scl::Index>(pi)] * 
                   vals_j[static_cast<scl::Index>(pj)];
            ++pi;
            ++pj;
        } else if (ci < cj) {
            ++pi;
        } else {
            ++pj;
        }
    }
    
    return dot;
}

/// @brief L2 norm of a row
template <typename T, bool IsCSR>
T row_norm(const MappedSparse<T, IsCSR>& mat, scl::Index i) {
    scl::Array<T> vals;
    if constexpr (IsCSR) {
        vals = mat.row_values(i);
    } else {
        vals = mat.col_values(i);
    }
    
    T sum_sq = T(0);
    const T* SCL_RESTRICT ptr = vals.data();
    const scl::Size n = vals.size();
    
    for (scl::Size k = 0; k < n; ++k) {
        sum_sq += ptr[k] * ptr[k];
    }
    
    return std::sqrt(sum_sq);
}

// =============================================================================
// SECTION 8: Filtering (Parallel)
// =============================================================================

/// @brief Filter elements below threshold
/// Note: Output nnz may differ from input; returns actual output nnz
template <typename T, bool IsCSR>
scl::Index filter_threshold(const MappedSparse<T, IsCSR>& mat,
                            T threshold,
                            T* SCL_RESTRICT data_out,
                            scl::Index* SCL_RESTRICT indices_out,
                            scl::Index* SCL_RESTRICT indptr_out) {
    const scl::Index n = IsCSR ? mat.rows() : mat.cols();
    
    // Cannot parallelize indptr computation due to prefix sum
    indptr_out[0] = 0;
    scl::Index offset = 0;
    
    for (scl::Index i = 0; i < n; ++i) {
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;
        
        if constexpr (IsCSR) {
            vals = mat.row_values(i);
            inds = mat.row_indices(i);
        } else {
            vals = mat.col_values(i);
            inds = mat.col_indices(i);
        }
        
        const T* SCL_RESTRICT vptr = vals.data();
        const scl::Index* SCL_RESTRICT iptr = inds.data();
        const scl::Size len = vals.size();
        
        for (scl::Size k = 0; k < len; ++k) {
            if (std::abs(vptr[k]) >= threshold) {
                data_out[offset] = vptr[k];
                indices_out[offset] = iptr[k];
                ++offset;
            }
        }
        
        indptr_out[i + 1] = offset;
    }
    
    return offset;
}

/// @brief Keep top-k elements per row (parallel per row, sequential write)
template <typename T, bool IsCSR>
void top_k_per_row(const MappedSparse<T, IsCSR>& mat,
                   scl::Index k,
                   T* SCL_RESTRICT data_out,
                   scl::Index* SCL_RESTRICT indices_out,
                   scl::Index* SCL_RESTRICT indptr_out) {
    const scl::Index n = IsCSR ? mat.rows() : mat.cols();
    
    indptr_out[0] = 0;
    scl::Index offset = 0;
    
    for (scl::Index i = 0; i < n; ++i) {
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;
        
        if constexpr (IsCSR) {
            vals = mat.row_values(i);
            inds = mat.row_indices(i);
        } else {
            vals = mat.col_values(i);
            inds = mat.col_indices(i);
        }
        
        const scl::Size len = vals.size();
        
        // Collect (value, index) pairs
        std::vector<std::pair<T, scl::Index>> pairs;
        pairs.reserve(len);
        
        for (scl::Size j = 0; j < len; ++j) {
            pairs.emplace_back(vals[static_cast<scl::Index>(j)],
                              inds[static_cast<scl::Index>(j)]);
        }
        
        // Partial sort for top-k
        const scl::Index take = std::min(k, static_cast<scl::Index>(pairs.size()));
        std::partial_sort(pairs.begin(), 
                         pairs.begin() + take, 
                         pairs.end(),
                         [](const auto& a, const auto& b) {
                             return std::abs(a.first) > std::abs(b.first);
                         });
        
        // Sort by column index for output
        std::sort(pairs.begin(), pairs.begin() + take,
                 [](const auto& a, const auto& b) { return a.second < b.second; });
        
        for (scl::Index j = 0; j < take; ++j) {
            data_out[offset] = pairs[static_cast<std::size_t>(j)].first;
            indices_out[offset] = pairs[static_cast<std::size_t>(j)].second;
            ++offset;
        }
        
        indptr_out[i + 1] = offset;
    }
}

} // namespace scl::mmap
