#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/core/registry.hpp"
#include "scl/core/memory.hpp"
#include "scl/threading/parallel_for.hpp"
#include <cstring>
#include <cmath>
#include <span>
#include <vector>

// =============================================================================
// FILE: scl/kernel/sparse.hpp
// BRIEF: Sparse matrix statistics with SIMD optimization
// =============================================================================

namespace scl::kernel::sparse {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Size PARALLEL_THRESHOLD = 1024;
    constexpr Size BATCH_SIZE = 64;
}

// =============================================================================
// SIMD Helpers
// =============================================================================

namespace detail {

template <typename T>
SCL_FORCE_INLINE void simd_sum_sumsq_fused(
    const T* SCL_RESTRICT vals,
    Size len,
    T& out_sum,
    T& out_sumsq
) {
    namespace s = scl::simd;
    auto d = s::SimdTagFor<T>::d;
    const Size lanes = static_cast<Size>(s::Lanes(d));

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_ssq0 = s::Zero(d);
    auto v_ssq1 = s::Zero(d);

    Size k = 0;
    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE * lanes < len)) {
            SCL_PREFETCH_READ(vals + k + config::PREFETCH_DISTANCE * lanes, 0);
        }

        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        v_sum0 = s::Add(v_sum0, v0);
        v_sum1 = s::Add(v_sum1, v1);
        v_sum0 = s::Add(v_sum0, v2);
        v_sum1 = s::Add(v_sum1, v3);

        v_ssq0 = s::MulAdd(v0, v0, v_ssq0);
        v_ssq1 = s::MulAdd(v1, v1, v_ssq1);
        v_ssq0 = s::MulAdd(v2, v2, v_ssq0);
        v_ssq1 = s::MulAdd(v3, v3, v_ssq1);
    }

    auto v_sum = s::Add(v_sum0, v_sum1);
    auto v_ssq = s::Add(v_ssq0, v_ssq1);

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_sum = s::Add(v_sum, v);
        v_ssq = s::MulAdd(v, v, v_ssq);
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));
    T sumsq = s::GetLane(s::SumOfLanes(d, v_ssq));

    for (; k < len; ++k) {
        T v = vals[k];
        sum += v;
        sumsq += v * v;
    }

    out_sum = sum;
    out_sumsq = sumsq;
}

template <typename T>
SCL_FORCE_INLINE T compute_variance(T sum, T sum_sq, T N, T denom) {
    if (denom <= T(0)) return T(0);

    T mu = sum / N;
    T var = (sum_sq - sum * mu) / denom;

    return (var < T(0)) ? T(0) : var;
}

} // namespace detail

// =============================================================================
// Transform Functions
// =============================================================================

template <typename T, bool IsCSR>
void primary_sums(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index primary_dim = matrix.primary_dim();

    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        const Size len_sz = static_cast<Size>(len);

        if (len_sz == 0) {
            output[p] = T(0);
            return;
        }

        auto values = matrix.primary_values_unsafe(idx);
        output[p] = scl::vectorize::sum(Array<const T>(values.ptr, len_sz));
    });
}

template <typename T, bool IsCSR>
void primary_means(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index primary_dim = matrix.primary_dim();
    const T inv_n = T(1) / static_cast<T>(matrix.secondary_dim());

    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        const Size len_sz = static_cast<Size>(len);

        if (len_sz == 0) {
            output[p] = T(0);
            return;
        }

        auto values = matrix.primary_values_unsafe(idx);
        output[p] = scl::vectorize::sum(Array<const T>(values.ptr, len_sz)) * inv_n;
    });
}

template <typename T, bool IsCSR>
void primary_variances(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output,
    int ddof = 1
) {
    const Index primary_dim = matrix.primary_dim();
    const T N = static_cast<T>(matrix.secondary_dim());
    const T denom = N - static_cast<T>(ddof);

    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        const Size len_sz = static_cast<Size>(len);

        T sum = T(0), sumsq = T(0);

        if (len_sz > 0) {
            auto values = matrix.primary_values_unsafe(idx);
            detail::simd_sum_sumsq_fused(values.ptr, len_sz, sum, sumsq);
        }

        output[p] = detail::compute_variance(sum, sumsq, N, denom);
    });
}

template <typename T, bool IsCSR>
void primary_nnz(
    const Sparse<T, IsCSR>& matrix,
    Array<Index> output
) {
    const Index primary_dim = matrix.primary_dim();
    const Size n = static_cast<Size>(primary_dim);

    SCL_CHECK_DIM(output.len == n, "Output size mismatch");

    // Sequential for small matrices (parallel overhead > computation)
    if (SCL_UNLIKELY(n < config::PARALLEL_THRESHOLD)) {
        for (Size p = 0; p < n; ++p) {
            output[static_cast<Index>(p)] = matrix.primary_length_unsafe(static_cast<Index>(p));
        }
        return;
    }

    // Parallel with batching for better cache utilization
    const Size num_batches = (n + config::BATCH_SIZE - 1) / config::BATCH_SIZE;

    scl::threading::parallel_for(Size(0), num_batches, [&](Size batch_idx) {
        const Size start = batch_idx * config::BATCH_SIZE;
        const Size end = scl::algo::min2(start + config::BATCH_SIZE, n);

        for (Size p = start; p < end; ++p) {
            output[static_cast<Index>(p)] = matrix.primary_length_unsafe(static_cast<Index>(p));
        }
    });
}

// =============================================================================
// Format Export Tools
// =============================================================================

template <typename T>
struct ContiguousArraysT {
    T* data;             // registry registered values array
    Index* indices;      // registry registered indices array
    Index* indptr;       // registry registered offset array
    Index nnz;
    Index primary_dim;
};

template <typename T>
struct COOArraysT {
    Index* row_indices;  // registry registered
    Index* col_indices;  // registry registered
    T* values;           // registry registered
    Index nnz;
};

// Type aliases for backward compatibility
using ContiguousArrays = ContiguousArraysT<Real>;
using COOArrays = COOArraysT<Real>;

template <typename T, bool IsCSR>
ContiguousArraysT<T> to_contiguous_arrays(const Sparse<T, IsCSR>& matrix) {
    ContiguousArraysT<T> result{};
    
    if (!matrix.valid()) {
        result.data = nullptr;
        result.indices = nullptr;
        result.indptr = nullptr;
        result.nnz = 0;
        result.primary_dim = matrix.primary_dim();
        return result;
    }

    const Index primary_dim = matrix.primary_dim();
    const Index nnz = matrix.nnz();
    
    auto& reg = get_registry();
    
    // Allocate arrays via registry (handle zero nnz case)
    T* data = (nnz > 0) ? reg.new_array<T>(static_cast<size_t>(nnz)) : nullptr;
    Index* indices = (nnz > 0) ? reg.new_array<Index>(static_cast<size_t>(nnz)) : nullptr;
    auto* indptr = reg.new_array<Index>(static_cast<size_t>(primary_dim + 1));
    
    if (!indptr || (nnz > 0 && (!data || !indices))) {
        // Cleanup on failure
        if (data) reg.unregister_ptr(data);
        if (indices) reg.unregister_ptr(indices);
        if (indptr) reg.unregister_ptr(indptr);
        return result;
    }
    
    // Build indptr and copy data
    indptr[0] = 0;
    Index offset = 0;
    
    for (Index i = 0; i < primary_dim; ++i) {
        const Index len = matrix.primary_length_unsafe(i);
        if (len > 0) {
            auto vals = matrix.primary_values_unsafe(i);
            auto idxs = matrix.primary_indices_unsafe(i);
            
            std::memcpy(data + offset, vals.ptr, len * sizeof(T));
            std::memcpy(indices + offset, idxs.ptr, len * sizeof(Index));
        }
        offset += len;
        indptr[i + 1] = offset;
    }
    
    result.data = data;
    result.indices = indices;
    result.indptr = indptr;
    result.nnz = nnz;
    result.primary_dim = primary_dim;
    
    return result;
}

template <typename T, bool IsCSR>
COOArraysT<T> to_coo_arrays(const Sparse<T, IsCSR>& matrix) {
    COOArraysT<T> result{};
    
    if (!matrix.valid()) {
        result.row_indices = nullptr;
        result.col_indices = nullptr;
        result.values = nullptr;
        result.nnz = 0;
        return result;
    }

    const Index nnz = matrix.nnz();
    
    if (nnz == 0) {
        result.row_indices = nullptr;
        result.col_indices = nullptr;
        result.values = nullptr;
        result.nnz = 0;
        return result;
    }
    
    auto& reg = get_registry();
    
    auto* row_indices = reg.new_array<Index>(static_cast<size_t>(nnz));
    auto* col_indices = reg.new_array<Index>(static_cast<size_t>(nnz));
    T* values = reg.new_array<T>(static_cast<size_t>(nnz));
    
    if (!row_indices || !col_indices || !values) {
        if (row_indices) reg.unregister_ptr(row_indices);
        if (col_indices) reg.unregister_ptr(col_indices);
        if (values) reg.unregister_ptr(values);
        return result;
    }
    
    const Index primary_dim = matrix.primary_dim();
    
    // Compute offsets for each primary slice
    std::vector<Index> offsets(primary_dim + 1);
    offsets[0] = 0;
    for (Index i = 0; i < primary_dim; ++i) {
        offsets[i + 1] = offsets[i] + matrix.primary_length_unsafe(i);
    }
    
    // Parallel conversion to COO format
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        const auto i = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(i);
        if (len > 0) {
            auto vals = matrix.primary_values_unsafe(i);
            auto idxs = matrix.primary_indices_unsafe(i);
            const Index base_pos = offsets[i];
            
            for (Index k = 0; k < len; ++k) {
                const Index pos = base_pos + k;
                if constexpr (IsCSR) {
                    row_indices[pos] = i;
                    col_indices[pos] = idxs[k];
                } else {
                    row_indices[pos] = idxs[k];
                    col_indices[pos] = i;
                }
                values[pos] = vals[k];
            }
        }
    });
    
    result.row_indices = row_indices;
    result.col_indices = col_indices;
    result.values = values;
    result.nnz = nnz;
    
    return result;
}

/// @brief Create Sparse matrix from contiguous arrays
/// @param take_ownership If true, this function takes ownership of ALL three arrays:
///        - data and indices are registered with registry for lifecycle management
///        - indptr is freed immediately after use (only needed for offset calculation)
///        Caller MUST NOT use or free any of the arrays after this call.
/// @note When take_ownership=true, arrays are registered as buffer+aliases so
///       they participate in reference counting for slicing operations.
template <typename T, bool IsCSR>
Sparse<T, IsCSR> from_contiguous_arrays(
    T* data, Index* indices, Index* indptr,
    Index rows, Index cols, Index nnz,
    bool take_ownership = false
) {
    if (!data || !indices || !indptr) {
        return Sparse<T, IsCSR>{};
    }
    
    const Index primary_dim = IsCSR ? rows : cols;
    
    if (take_ownership) {
        auto& reg = get_registry();
        
        // Build alias lists for data and indices (one per non-empty row/column)
        std::vector<void*> data_aliases;
        std::vector<void*> indices_aliases;
        data_aliases.reserve(primary_dim);
        indices_aliases.reserve(primary_dim);
        
        for (Index i = 0; i < primary_dim; ++i) {
            Index start = indptr[i];
            Index len = indptr[i + 1] - start;
            if (len > 0) {
                data_aliases.push_back(data + start);
                indices_aliases.push_back(indices + start);
            }
        }
        
        // Register data array as buffer with aliases
        if (!data_aliases.empty()) {
            if (!reg.register_buffer_with_aliases(
                    data, 
                    static_cast<std::size_t>(nnz) * sizeof(T),
                    data_aliases, 
                    AllocType::AlignedAlloc)) {
                // Registration failed - free all memory and return empty
                // (acceptable leak in OOM scenarios per design)
                scl::memory::aligned_free(data);
                scl::memory::aligned_free(indices);
                scl::memory::aligned_free(indptr);
                return Sparse<T, IsCSR>{};
            }
        }
        
        // Register indices array as buffer with aliases
        if (!indices_aliases.empty()) {
            if (!reg.register_buffer_with_aliases(
                    indices,
                    static_cast<std::size_t>(nnz) * sizeof(Index),
                    indices_aliases,
                    AllocType::AlignedAlloc)) {
                // data already registered - will be managed by registry
                // free indptr and return empty
                scl::memory::aligned_free(indptr);
                return Sparse<T, IsCSR>{};
            }
        }
        
        // Note: indptr is NOT registered with registry
        // It's only used to compute offsets for wrap_traditional_unsafe
        // Will be freed after Sparse creation below
    }
    
    // Create Sparse using wrap_traditional (unsafe variant for performance since
    // we've already validated the arrays above or caller guarantees validity)
    std::span<const Index> indptr_span(indptr, static_cast<size_t>(primary_dim + 1));
    auto result = Sparse<T, IsCSR>::wrap_traditional_unsafe(rows, cols, data, indices, indptr_span);
    
    if (take_ownership) {
        // indptr is no longer needed - Sparse only stores dp/ip/len pointers
        // Free it now to prevent memory leak
        scl::memory::aligned_free(indptr);
    }
    
    return result;
}

// =============================================================================
// Data Cleanup Tools
// =============================================================================

template <typename T, bool IsCSR>
Sparse<T, IsCSR> eliminate_zeros(
    const Sparse<T, IsCSR>& matrix,
    T tolerance = T(0)
) {
    if (!matrix.valid()) return Sparse<T, IsCSR>{};
    
    const Index primary_dim = matrix.primary_dim();
    std::vector<Index> new_nnzs(primary_dim, 0);
    
    // Parallel count non-zero elements per row/column
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        const auto i = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(i);
        if (len > 0) {
            auto vals = matrix.primary_values_unsafe(i);
            Index count = 0;
            for (Index k = 0; k < len; ++k) {
                T abs_val = (vals[k] < T(0)) ? -vals[k] : vals[k];
                if (abs_val > tolerance) {
                    ++count;
                }
            }
            new_nnzs[i] = count;
        }
    });
    
    // Create result matrix
    Sparse<T, IsCSR> result = Sparse<T, IsCSR>::create(
        matrix.rows(), matrix.cols(), new_nnzs, BlockStrategy::contiguous());
    
    if (!result.valid()) return Sparse<T, IsCSR>{};
    
    // Parallel copy non-zero elements
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        const auto i = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(i);
        if (len > 0 && new_nnzs[i] > 0) {
            auto vals = matrix.primary_values_unsafe(i);
            auto idxs = matrix.primary_indices_unsafe(i);
            
            T* out_vals = static_cast<T*>(result.data_ptrs[i]);
            auto* out_idxs = static_cast<Index*>(result.indices_ptrs[i]);
            
            Index pos = 0;
            for (Index k = 0; k < len; ++k) {
                T abs_val = (vals[k] < T(0)) ? -vals[k] : vals[k];
                if (abs_val > tolerance) {
                    out_vals[pos] = vals[k];
                    out_idxs[pos] = idxs[k];
                    ++pos;
                }
            }
        }
    });
    
    return result;
}

template <typename T, bool IsCSR>
Sparse<T, IsCSR> prune(
    const Sparse<T, IsCSR>& matrix,
    T threshold,
    bool keep_structure = false
) {
    if (!matrix.valid()) return Sparse<T, IsCSR>{};
    
    if (keep_structure) {
        // Set small values to zero but keep structure (sparse format with explicit zeros)
        // Note: This creates a matrix that may contain explicit zeros, which is
        // unusual for sparse formats but useful for some algorithms
        Sparse<T, IsCSR> result = matrix.clone();
        if (!result.valid()) return Sparse<T, IsCSR>{};
        
        const Index primary_dim = result.primary_dim();
        for (Index i = 0; i < primary_dim; ++i) {
            const Index len = result.primary_length_unsafe(i);
            if (len > 0) {
                auto vals = result.primary_values_unsafe(i);
                for (Index k = 0; k < len; ++k) {
                    T abs_val = (vals[k] < T(0)) ? -vals[k] : vals[k];
                    if (abs_val < threshold) {
                        vals[k] = T(0);
                    }
                }
            }
        }
        
        return result;
    } else {
        // Remove small values entirely (compact structure)
        return eliminate_zeros(matrix, threshold);
    }
}

// =============================================================================
// Validation and Info Tools
// =============================================================================

struct ValidationResult {
    bool valid;
    const char* error_message;  // nullptr if valid
    Index error_index;          // -1 if valid
};

struct MemoryInfo {
    Size data_bytes;
    Size indices_bytes;
    Size metadata_bytes;
    Size total_bytes;
    Index block_count;
    bool is_contiguous;
};

template <typename T, bool IsCSR>
ValidationResult validate(const Sparse<T, IsCSR>& matrix) {
    ValidationResult result{true, nullptr, -1};
    
    if (!matrix.valid()) {
        result.valid = false;
        result.error_message = "Matrix is invalid (null pointers)";
        return result;
    }
    
    const Index primary_dim = matrix.primary_dim();
    Index total_nnz = 0;
    
    for (Index i = 0; i < primary_dim; ++i) {
        const Index len = matrix.primary_length_unsafe(i);
        total_nnz += len;
        
        if (len > 0) {
            auto idxs = matrix.primary_indices_unsafe(i);
            const Index secondary_dim = matrix.secondary_dim();
            
            // Check indices are in valid range
            for (Index k = 0; k < len; ++k) {
                if (idxs[k] < 0 || idxs[k] >= secondary_dim) {
                    result.valid = false;
                    result.error_message = "Index out of bounds";
                    result.error_index = i;
                    return result;
                }
            }
            
            // Check indices are sorted (CSR/CSC requirement)
            for (Index k = 1; k < len; ++k) {
                if (idxs[k] <= idxs[k-1]) {
                    result.valid = false;
                    result.error_message = "Indices not sorted";
                    result.error_index = i;
                    return result;
                }
            }
        }
    }
    
    // Check nnz consistency
    if (total_nnz != matrix.nnz()) {
        result.valid = false;
        result.error_message = "NNZ count mismatch";
        result.error_index = -1;
        return result;
    }
    
    return result;
}

template <typename T, bool IsCSR>
MemoryInfo memory_info(const Sparse<T, IsCSR>& matrix) {
    MemoryInfo info{};
    
    if (!matrix.valid()) {
        return info;
    }
    
    const Index nnz = matrix.nnz();
    const Index primary_dim = matrix.primary_dim();
    
    info.data_bytes = static_cast<Size>(nnz) * sizeof(T);
    info.indices_bytes = static_cast<Size>(nnz) * sizeof(Index);
    info.metadata_bytes = static_cast<Size>(primary_dim) * 
                         (2 * sizeof(Pointer) + sizeof(Index));
    info.total_bytes = info.data_bytes + info.indices_bytes + info.metadata_bytes;
    
    // Count blocks using layout info
    auto layout = matrix.layout_info();
    info.block_count = layout.data_block_count;
    info.is_contiguous = layout.is_contiguous;
    
    return info;
}

// =============================================================================
// Helper Conversion Tools
// =============================================================================

template <typename T, bool IsCSR>
Sparse<T, IsCSR> make_contiguous(const Sparse<T, IsCSR>& matrix) {
    if (!matrix.valid()) return Sparse<T, IsCSR>{};
    
    if (matrix.is_contiguous()) {
        return matrix.clone(BlockStrategy::contiguous());
    }
    
    return matrix.to_contiguous();
}

template <typename T, bool IsCSR>
void resize_secondary(Sparse<T, IsCSR>& matrix, Index new_secondary_dim) {
    if (!matrix.valid()) return;
    
    SCL_CHECK_ARG(new_secondary_dim >= 0, "new_secondary_dim must be non-negative");
    
    // This operation only updates the dimension metadata
    // It does not modify the actual data or indices
    // WARNING: Caller must ensure all indices are valid for new dimension
    
    // If shrinking, verify no indices are out of bounds (debug mode only)
#ifdef SCL_DEBUG
    if (new_secondary_dim < old_secondary_dim) {
        const Index primary_dim = matrix.primary_dim();
        for (Index i = 0; i < primary_dim; ++i) {
            const Index len = matrix.primary_length_unsafe(i);
            if (len > 0) {
                auto idxs = matrix.primary_indices_unsafe(i);
                for (Index k = 0; k < len; ++k) {
                    SCL_ASSERT(idxs[k] < new_secondary_dim,
                              "resize_secondary: index out of bounds for new dimension");
                }
            }
        }
    }
#endif
    
    if constexpr (IsCSR) {
        matrix.cols_ = new_secondary_dim;
    } else {
        matrix.rows_ = new_secondary_dim;
    }
}

} // namespace scl::kernel::sparse

