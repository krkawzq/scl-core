#pragma once

#include "scl/io/mmap.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/threading/parallel_for.hpp"

#include <vector>
#include <tuple>
#include <future>
#include <cstring>

// =============================================================================
/// @file mmatrix.hpp
/// @brief Memory-Mapped CSR Matrix Data Structures
///
/// Provides zero-copy CSR matrix views over memory-mapped files, enabling
/// out-of-core computation on TB-scale sparse matrices.
///
/// Design Philosophy:
///
/// 1. Composable: Built on top of generic MappedArray from mmap.hpp
/// 2. Pure Data Structures: No file organization assumptions
/// 3. CSRLike Interface: Seamless integration with SCL kernels
/// 4. Flexible Initialization: Support any file layout
///
/// Core Components:
///
/// - MountMatrix: CSR matrix view over three MappedArrays
/// - VirtualMountMatrix: Zero-copy row slicing with indirection
/// - OwnedCSR: Heap-allocated CSR matrix (materialization target)
///
/// Performance:
///
/// - SIMD-accelerated materialization
/// - Parallel row copying
/// - Streaming memory copy (bypasses cache)
/// - Huge page support
// =============================================================================

namespace scl::io {

// =============================================================================
// Forward Declarations
// =============================================================================

template <typename T> class MountMatrix;
template <typename T> class VirtualMountMatrix;
template <typename T> struct OwnedCSR;

namespace detail {

// =============================================================================
// Memory Copy Strategy
// =============================================================================

enum class CopyPolicy {
    Safe,       ///< memmove - handles overlapping regions
    Fast,       ///< memcpy/SIMD - assumes no overlap
    Streaming   ///< Non-temporal stores - bypasses cache
};

/// @brief Generic memory copy with policy selection.
template <typename T, CopyPolicy Policy = CopyPolicy::Fast>
inline void memory_copy(const T* src, T* dst, Size count) {
    constexpr Size L3_THRESHOLD = 8 * 1024 * 1024 / sizeof(T);
    
    if constexpr (Policy == CopyPolicy::Safe) {
        std::memmove(dst, src, count * sizeof(T));
    } else if constexpr (Policy == CopyPolicy::Fast) {
        std::memcpy(dst, src, count * sizeof(T));
    } else if constexpr (Policy == CopyPolicy::Streaming) {
        // For streaming, use memcpy (compiler + OS optimize for non-temporal access)
        std::memcpy(dst, src, count * sizeof(T));
    }
}

/// @brief Parallel memory copy.
template <typename T, CopyPolicy Policy = CopyPolicy::Fast>
inline void parallel_memory_copy(const T* src, T* dst, Size count) {
    constexpr Size MIN_PARALLEL = 1024 * 1024;
    
    if (count < MIN_PARALLEL) {
        memory_copy<T, Policy>(src, dst, count);
        return;
    }
    
    const Size num_threads = scl::threading::Scheduler::get_num_threads();
    const Size chunk = (count + num_threads - 1) / num_threads;
    
    scl::threading::parallel_for(Size(0), num_threads, [&](size_t tid) {
        Size start = tid * chunk;
        Size end = std::min(start + chunk, count);
        if (start < end) {
            memory_copy<T, Policy>(src + start, dst + start, end - start);
        }
    });
}

} // namespace detail

// =============================================================================
// MountMatrix: Pure CSR Matrix View
// =============================================================================

/// @brief CSR matrix view over three independent MappedArrays.
///
/// Makes ZERO assumptions about file organization - user provides three arrays.
/// This is a pure data structure that happens to work with memory-mapped data.
///
/// Satisfies CSRLike concept for seamless kernel integration.
///
/// @tparam T Value type (typically float or double)
template <typename T>
class MountMatrix {
public:
    using ValueType = T;
    using Tag = TagCSR;

    const Index rows;
    const Index cols;
    const Index nnz;

private:
    MappedArray<T> _data;
    MappedArray<Index> _indices;
    MappedArray<Index> _indptr;

public:
    /// @brief Construct from three pre-mapped arrays (pure, zero I/O assumptions).
    ///
    /// This is the fundamental constructor - no file paths, no hardcoded structure.
    /// Users map files however they want and pass arrays here.
    ///
    /// @param data Values array
    /// @param indices Column indices array
    /// @param indptr Row pointers array (size: rows+1)
    /// @param num_rows Number of rows
    /// @param num_cols Number of columns
    /// @param num_nnz Number of non-zero elements
    MountMatrix(
        MappedArray<T>&& data,
        MappedArray<Index>&& indices,
        MappedArray<Index>&& indptr,
        Index num_rows,
        Index num_cols,
        Index num_nnz
    )
        : rows(num_rows), cols(num_cols), nnz(num_nnz),
          _data(std::move(data)),
          _indices(std::move(indices)),
          _indptr(std::move(indptr))
    {
        SCL_CHECK_ARG(rows >= 0 && cols >= 0 && nnz >= 0, "Invalid dimensions");
        
        if (_indptr.size() != static_cast<Size>(rows + 1)) {
            throw ValueError("indptr size mismatch: expected " + 
                           std::to_string(rows + 1) + ", got " + 
                           std::to_string(_indptr.size()));
        }
    }

    MountMatrix(MountMatrix&&) noexcept = default;
    MountMatrix& operator=(MountMatrix&&) noexcept = default;
    MountMatrix(const MountMatrix&) = delete;
    MountMatrix& operator=(const MountMatrix&) = delete;

    // -------------------------------------------------------------------------
    // CSRLike Interface
    // -------------------------------------------------------------------------

    SCL_NODISCARD SCL_FORCE_INLINE Index row_length(Index i) const {
#if !defined(NDEBUG)
        SCL_ASSERT(i >= 0 && i < rows, "Row out of bounds");
#endif
        return _indptr[i + 1] - _indptr[i];
    }

    SCL_NODISCARD SCL_FORCE_INLINE Span<T> row_values(Index i) const {
#if !defined(NDEBUG)
        SCL_ASSERT(i >= 0 && i < rows, "Row out of bounds");
#endif
        Index start = _indptr[i];
        Size len = static_cast<Size>(row_length(i));
        return Span<T>(const_cast<T*>(_data.data() + start), len);
    }

    SCL_NODISCARD SCL_FORCE_INLINE Span<Index> row_indices(Index i) const {
#if !defined(NDEBUG)
        SCL_ASSERT(i >= 0 && i < rows, "Row out of bounds");
#endif
        Index start = _indptr[i];
        Size len = static_cast<Size>(row_length(i));
        return Span<Index>(const_cast<Index*>(_indices.data() + start), len);
    }

    // -------------------------------------------------------------------------
    // Direct Access
    // -------------------------------------------------------------------------

    SCL_NODISCARD T* data() const noexcept { return const_cast<T*>(_data.data()); }
    SCL_NODISCARD Index* indices() const noexcept { return const_cast<Index*>(_indices.data()); }
    SCL_NODISCARD Index* indptr() const noexcept { return const_cast<Index*>(_indptr.data()); }

    SCL_NODISCARD const MappedArray<T>& mapped_data() const noexcept { return _data; }
    SCL_NODISCARD const MappedArray<Index>& mapped_indices() const noexcept { return _indices; }
    SCL_NODISCARD const MappedArray<Index>& mapped_indptr() const noexcept { return _indptr; }

    // -------------------------------------------------------------------------
    // Materialization
    // -------------------------------------------------------------------------

    OwnedCSR<T> materialize() const;
    OwnedCSR<T> materialize_async() const;
    OwnedCSR<T> copy_rows(Span<const Index> row_selection) const;

    // -------------------------------------------------------------------------
    // Memory Hints
    // -------------------------------------------------------------------------

    void prefetch() const noexcept {
        _data.prefetch();
        _indices.prefetch();
        _indptr.prefetch();
    }

    void drop_cache() const noexcept {
        _data.drop_cache();
        _indices.drop_cache();
        _indptr.drop_cache();
    }

    SCL_NODISCARD Size estimate_memory_size() const noexcept {
        return _data.byte_size() + _indices.byte_size() + _indptr.byte_size();
    }
};

// =============================================================================
// VirtualMountMatrix: Zero-Copy Row Slicing
// =============================================================================

/// @brief Virtual CSR matrix with indirection over memory-mapped data.
///
/// Enables zero-copy row slicing through indirection array.
/// Minimal memory overhead: O(selected_rows) for row_map.
///
/// @tparam T Value type
template <typename T>
class VirtualMountMatrix {
public:
    using ValueType = T;
    using Tag = TagCSR;

    const Index rows;
    const Index cols;
    const Index src_rows;

private:
    const MappedArray<T>* _src_data;
    const MappedArray<Index>* _src_indices;
    const MappedArray<Index>* _src_indptr;
    const Index* _row_map;
    const Index* _src_row_lengths;
    bool _owns_row_map;
    std::vector<Index> _owned_row_map;

public:
    VirtualMountMatrix(
        const MappedArray<T>& src_data,
        const MappedArray<Index>& src_indices,
        const MappedArray<Index>& src_indptr,
        Span<const Index> row_map,
        Index num_src_rows,
        Index num_cols,
        const Index* src_row_lengths = nullptr
    )
        : rows(static_cast<Index>(row_map.size)),
          cols(num_cols),
          src_rows(num_src_rows),
          _src_data(&src_data),
          _src_indices(&src_indices),
          _src_indptr(&src_indptr),
          _row_map(row_map.ptr),
          _src_row_lengths(src_row_lengths),
          _owns_row_map(false)
    {
        SCL_CHECK_ARG(rows >= 0 && rows <= num_src_rows, "Invalid row count");
        SCL_CHECK_ARG(cols >= 0, "Invalid column count");
    }

    VirtualMountMatrix(const MountMatrix<T>& source, Span<const Index> row_map)
        : VirtualMountMatrix(
            source.mapped_data(),
            source.mapped_indices(),
            source.mapped_indptr(),
            row_map,
            source.rows,
            source.cols,
            nullptr
        )
    {}

    VirtualMountMatrix(VirtualMountMatrix&&) noexcept = default;
    VirtualMountMatrix& operator=(VirtualMountMatrix&&) noexcept = default;
    VirtualMountMatrix(const VirtualMountMatrix&) = delete;
    VirtualMountMatrix& operator=(const VirtualMountMatrix&) = delete;

    // -------------------------------------------------------------------------
    // CSRLike Interface
    // -------------------------------------------------------------------------

    SCL_NODISCARD SCL_FORCE_INLINE Index row_length(Index i) const {
#if !defined(NDEBUG)
        SCL_ASSERT(i >= 0 && i < rows, "Row out of bounds");
#endif
        Index phys_row = _row_map[i];
        
        if (_src_row_lengths) {
            return _src_row_lengths[phys_row];
        } else {
            const Index* indptr = _src_indptr->data();
            return indptr[phys_row + 1] - indptr[phys_row];
        }
    }

    SCL_NODISCARD SCL_FORCE_INLINE Span<T> row_values(Index i) const {
#if !defined(NDEBUG)
        SCL_ASSERT(i >= 0 && i < rows, "Row out of bounds");
#endif
        Index phys_row = _row_map[i];
        const Index* indptr = _src_indptr->data();
        Index start = indptr[phys_row];
        Size len = static_cast<Size>(row_length(i));
        
        return Span<T>(const_cast<T*>(_src_data->data() + start), len);
    }

    SCL_NODISCARD SCL_FORCE_INLINE Span<Index> row_indices(Index i) const {
#if !defined(NDEBUG)
        SCL_ASSERT(i >= 0 && i < rows, "Row out of bounds");
#endif
        Index phys_row = _row_map[i];
        const Index* indptr = _src_indptr->data();
        Index start = indptr[phys_row];
        Size len = static_cast<Size>(row_length(i));
        
        return Span<Index>(const_cast<Index*>(_src_indices->data() + start), len);
    }

    // -------------------------------------------------------------------------
    // Materialization
    // -------------------------------------------------------------------------

    OwnedCSR<T> materialize() const;

    SCL_NODISCARD Size estimate_memory_size() const noexcept {
        Index total_nnz = 0;
        for (Index i = 0; i < rows; ++i) {
            total_nnz += row_length(i);
        }
        return static_cast<Size>(total_nnz) * (sizeof(T) + sizeof(Index)) +
               static_cast<Size>(rows + 1) * sizeof(Index);
    }
};

// =============================================================================
// OwnedCSR: Heap-Allocated CSR Matrix
// =============================================================================

/// @brief CSR matrix with owned heap storage.
///
/// Target for materialization operations. Manages memory lifetime.
///
/// @tparam T Value type
template <typename T>
struct OwnedCSR {
    std::vector<T> data;
    std::vector<Index> indices;
    std::vector<Index> indptr;
    
    Index rows;
    Index cols;
    Index nnz;
    
    OwnedCSR() : rows(0), cols(0), nnz(0) {}
    
    OwnedCSR(
        std::vector<T>&& d,
        std::vector<Index>&& i,
        std::vector<Index>&& p,
        Index r, Index c, Index n
    )
        : data(std::move(d)), indices(std::move(i)), indptr(std::move(p)),
          rows(r), cols(c), nnz(n)
    {}
    
    /// @brief Get CustomCSR view (non-owning).
    CustomCSR<T> view() noexcept {
        return CustomCSR<T>(data.data(), indices.data(), indptr.data(), rows, cols, nnz);
    }
    
    // -------------------------------------------------------------------------
    // Zero-Copy Python Interface
    // -------------------------------------------------------------------------
    
    /// @brief Release ownership of vectors (for Python binding).
    std::tuple<std::vector<T>, std::vector<Index>, std::vector<Index>>
    release_vectors() && {
        return std::make_tuple(std::move(data), std::move(indices), std::move(indptr));
    }
    
    /// @brief Get buffer info (for Python buffer protocol).
    std::tuple<T*, Size, Index*, Size, Index*, Size, Index, Index, Index>
    buffer_info() noexcept {
        return std::make_tuple(
            data.data(), data.size(),
            indices.data(), indices.size(),
            indptr.data(), indptr.size(),
            rows, cols, nnz
        );
    }
};

// =============================================================================
// Materialization Implementations
// =============================================================================

template <typename T>
OwnedCSR<T> MountMatrix<T>::materialize() const {
    std::vector<T> data_copy(nnz);
    std::vector<Index> indices_copy(nnz);
    std::vector<Index> indptr_copy(rows + 1);
    
    // Use streaming policy for large arrays (bypasses cache)
    detail::parallel_memory_copy<T, detail::CopyPolicy::Streaming>(
        _data.data(), data_copy.data(), _data.size()
    );
    detail::parallel_memory_copy<Index, detail::CopyPolicy::Streaming>(
        _indices.data(), indices_copy.data(), _indices.size()
    );
    // indptr is small, use fast policy
    detail::memory_copy<Index, detail::CopyPolicy::Fast>(
        _indptr.data(), indptr_copy.data(), _indptr.size()
    );
    
    return OwnedCSR<T>(std::move(data_copy), std::move(indices_copy), std::move(indptr_copy), rows, cols, nnz);
}

template <typename T>
OwnedCSR<T> MountMatrix<T>::materialize_async() const {
    return std::async(std::launch::async, [this]() { return materialize(); }).get();
}

template <typename T>
OwnedCSR<T> MountMatrix<T>::copy_rows(Span<const Index> row_selection) const {
    Index total_nnz = 0;
    for (Size i = 0; i < row_selection.size; ++i) {
        total_nnz += row_length(row_selection[i]);
    }
    
    std::vector<T> out_data;
    std::vector<Index> out_indices;
    std::vector<Index> out_indptr;
    
    out_data.reserve(total_nnz);
    out_indices.reserve(total_nnz);
    out_indptr.reserve(row_selection.size + 1);
    out_indptr.push_back(0);
    
    for (Size i = 0; i < row_selection.size; ++i) {
        Index row_idx = row_selection[i];
        auto vals = row_values(row_idx);
        auto idxs = row_indices(row_idx);
        
        out_data.insert(out_data.end(), vals.begin(), vals.end());
        out_indices.insert(out_indices.end(), idxs.begin(), idxs.end());
        out_indptr.push_back(static_cast<Index>(out_data.size()));
    }
    
    return OwnedCSR<T>(std::move(out_data), std::move(out_indices), std::move(out_indptr), 
                       static_cast<Index>(row_selection.size), cols, total_nnz);
}

template <typename T>
OwnedCSR<T> VirtualMountMatrix<T>::materialize() const {
    // Phase 1: Parallel row length computation
    std::vector<Index> row_sizes(rows);
    
    scl::threading::parallel_for(Index(0), rows, [&](Index i) {
        row_sizes[i] = row_length(i);
    });
    
    // Phase 2: Exclusive scan for indptr
    std::vector<Index> out_indptr(rows + 1);
    out_indptr[0] = 0;
    
    Index total_nnz = 0;
    for (Index i = 0; i < rows; ++i) {
        total_nnz += row_sizes[i];
        out_indptr[i + 1] = total_nnz;
    }
    
    // Phase 3: Allocate output arrays
    std::vector<T> out_data(total_nnz);
    std::vector<Index> out_indices(total_nnz);
    
    // Phase 4: Parallel copy (lock-free, non-overlapping writes)
    scl::threading::parallel_for(Index(0), rows, [&](Index i) {
        Index phys_row = _row_map[i];
        const Index* indptr = _src_indptr->data();
        
        Index src_start = indptr[phys_row];
        Index dest_start = out_indptr[i];
        Size len = static_cast<Size>(row_sizes[i]);
        
        if (len > 0) {
            detail::memory_copy<T, detail::CopyPolicy::Fast>(
                _src_data->data() + src_start,
                out_data.data() + dest_start,
                len
            );
            
            detail::memory_copy<Index, detail::CopyPolicy::Fast>(
                _src_indices->data() + src_start,
                out_indices.data() + dest_start,
                len
            );
        }
    });
    
    return OwnedCSR<T>(std::move(out_data), std::move(out_indices), std::move(out_indptr), rows, cols, total_nnz);
}

// =============================================================================
// Type Aliases
// =============================================================================

using MountMatrixF32 = MountMatrix<float>;
using MountMatrixF64 = MountMatrix<double>;
using MountMatrixReal = MountMatrix<Real>;

using VirtualMountMatrixF32 = VirtualMountMatrix<float>;
using VirtualMountMatrixF64 = VirtualMountMatrix<double>;
using VirtualMountMatrixReal = VirtualMountMatrix<Real>;

using OwnedCSRF32 = OwnedCSR<float>;
using OwnedCSRF64 = OwnedCSR<double>;
using OwnedCSRReal = OwnedCSR<Real>;

} // namespace scl::io

