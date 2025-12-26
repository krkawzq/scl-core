#pragma once

#include "scl/io/mmap.hpp"
#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/simd.hpp"
#include "scl/threading/parallel_for.hpp"

#include <vector>
#include <tuple>
#include <cstring>

// =============================================================================
/// @file mmatrix.hpp
/// @brief Memory-Mapped Sparse Matrix
///
/// Provides zero-copy sparse matrix views over memory-mapped files.
/// Enables out-of-core computation on TB-scale sparse matrices.
///
/// Design:
/// - Built on MappedArray from mmap.hpp
/// - Satisfies SparseLike concept
/// - Zero-copy operations
/// - Parallel materialization
// =============================================================================

namespace scl::io {

// =============================================================================
// Forward Declarations
// =============================================================================

template <typename T, bool IsCSR> class MappedCustomSparse;
template <typename T, bool IsCSR> class MappedVirtualSparse;
template <typename T, bool IsCSR> struct OwnedSparse;

namespace detail {

/// @brief Parallel memory copy
template <typename T>
inline void parallel_copy(const T* src, T* dst, Size count) {
    constexpr Size MIN_PARALLEL = 1024 * 1024;
    
    if (count < MIN_PARALLEL) {
        std::memcpy(dst, src, count * sizeof(T));
        return;
    }
    
    const Size num_threads = scl::threading::Scheduler::get_num_threads();
    const Size chunk = (count + num_threads - 1) / num_threads;
    
    scl::threading::parallel_for(Size(0), num_threads, [&](size_t tid) {
        Size start = tid * chunk;
        Size end = std::min(start + chunk, count);
        if (start < end) {
            std::memcpy(dst + start, src + start, (end - start) * sizeof(T));
        }
    });
}

} // namespace detail

// =============================================================================
// MappedCustomSparse: Memory-Mapped Sparse Matrix
// =============================================================================

/// @brief Sparse matrix view over memory-mapped arrays
///
/// Satisfies SparseLike<IsCSR> concept.
/// Zero I/O assumptions - user provides mapped arrays.
template <typename T, bool IsCSR>
class MappedCustomSparse {
public:
    using ValueType = T;
    using Tag = TagSparse<IsCSR>;
    
    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;

    const Index rows;
    const Index cols;

private:
    MappedArray<T> _data;
    MappedArray<Index> _indices;
    MappedArray<Index> _indptr;

public:
    /// @brief Construct from three memory-mapped arrays
    MappedCustomSparse(
        MappedArray<T>&& data,
        MappedArray<Index>&& indices,
        MappedArray<Index>&& indptr,
        Index num_rows,
        Index num_cols
    )
        : rows(num_rows), cols(num_cols),
          _data(std::move(data)),
          _indices(std::move(indices)),
          _indptr(std::move(indptr))
    {
        SCL_CHECK_ARG(rows >= 0 && cols >= 0, "Invalid dimensions");
        
        Index expected_indptr_size = IsCSR ? (rows + 1) : (cols + 1);
        if (_indptr.size() != static_cast<Size>(expected_indptr_size)) {
            throw std::runtime_error("indptr size mismatch");
        }
    }

    MappedCustomSparse(MappedCustomSparse&&) noexcept = default;
    MappedCustomSparse& operator=(MappedCustomSparse&&) noexcept = default;
    MappedCustomSparse(const MappedCustomSparse&) = delete;
    MappedCustomSparse& operator=(const MappedCustomSparse&) = delete;

    // SparseLike Interface
    SCL_NODISCARD SCL_FORCE_INLINE Index nnz() const {
        Index primary_dim = IsCSR ? rows : cols;
        return _indptr[primary_dim];
    }

    // CSR interface
    SCL_NODISCARD SCL_FORCE_INLINE Array<T> row_values(Index i) const requires (IsCSR) {
        Index start = _indptr[i];
        Index end = _indptr[i + 1];
        return Array<T>(const_cast<T*>(_data.data() + start), static_cast<Size>(end - start));
    }

    SCL_NODISCARD SCL_FORCE_INLINE Array<Index> row_indices(Index i) const requires (IsCSR) {
        Index start = _indptr[i];
        Index end = _indptr[i + 1];
        return Array<Index>(const_cast<Index*>(_indices.data() + start), static_cast<Size>(end - start));
    }

    SCL_NODISCARD SCL_FORCE_INLINE Index row_length(Index i) const requires (IsCSR) {
        return _indptr[i + 1] - _indptr[i];
    }

    // CSC interface
    SCL_NODISCARD SCL_FORCE_INLINE Array<T> col_values(Index j) const requires (!IsCSR) {
        Index start = _indptr[j];
        Index end = _indptr[j + 1];
        return Array<T>(const_cast<T*>(_data.data() + start), static_cast<Size>(end - start));
    }

    SCL_NODISCARD SCL_FORCE_INLINE Array<Index> col_indices(Index j) const requires (!IsCSR) {
        Index start = _indptr[j];
        Index end = _indptr[j + 1];
        return Array<Index>(const_cast<Index*>(_indices.data() + start), static_cast<Size>(end - start));
    }

    SCL_NODISCARD SCL_FORCE_INLINE Index col_length(Index j) const requires (!IsCSR) {
        return _indptr[j + 1] - _indptr[j];
    }

    // Direct access
    SCL_NODISCARD T* data() const noexcept { return const_cast<T*>(_data.data()); }
    SCL_NODISCARD Index* indices() const noexcept { return const_cast<Index*>(_indices.data()); }
    SCL_NODISCARD Index* indptr() const noexcept { return const_cast<Index*>(_indptr.data()); }

    // Materialization
    OwnedSparse<T, IsCSR> materialize() const {
        Index total_nnz = nnz();
        Index primary_dim = IsCSR ? rows : cols;
        
        std::vector<T> data_copy(total_nnz);
        std::vector<Index> indices_copy(total_nnz);
        std::vector<Index> indptr_copy(primary_dim + 1);
        
        detail::parallel_copy(_data.data(), data_copy.data(), _data.size());
        detail::parallel_copy(_indices.data(), indices_copy.data(), _indices.size());
        std::memcpy(indptr_copy.data(), _indptr.data(), _indptr.size() * sizeof(Index));
        
        return OwnedSparse<T, IsCSR>(
            std::move(data_copy), 
            std::move(indices_copy), 
            std::move(indptr_copy),
            rows, cols
        );
    }

    // Memory hints
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
};

// =============================================================================
// MappedVirtualSparse: Zero-Copy Row/Col Slicing
// =============================================================================

/// @brief Virtual sparse matrix with indirection over memory-mapped data
///
/// Enables zero-copy slicing through indirection array.
/// Minimal overhead: O(selected_elements) for map.
template <typename T, bool IsCSR>
class MappedVirtualSparse {
public:
    using ValueType = T;
    using Tag = TagSparse<IsCSR>;
    
    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;

    const Index rows;
    const Index cols;

private:
    const MappedArray<T>* _src_data;
    const MappedArray<Index>* _src_indices;
    const MappedArray<Index>* _src_indptr;
    const Index* _map;
    Index _src_primary_dim;

public:
    MappedVirtualSparse(
        const MappedArray<T>& src_data,
        const MappedArray<Index>& src_indices,
        const MappedArray<Index>& src_indptr,
        Array<const Index> map,
        Index src_rows,
        Index src_cols
    )
        : rows(IsCSR ? static_cast<Index>(map.len) : src_rows),
          cols(IsCSR ? src_cols : static_cast<Index>(map.len)),
          _src_data(&src_data),
          _src_indices(&src_indices),
          _src_indptr(&src_indptr),
          _map(map.ptr),
          _src_primary_dim(IsCSR ? src_rows : src_cols)
    {}

    SCL_NODISCARD Index nnz() const {
        Index total = 0;
        Index primary_dim = IsCSR ? rows : cols;
        for (Index i = 0; i < primary_dim; ++i) {
            Index phys_idx = _map[i];
            total += _src_indptr->data()[phys_idx + 1] - _src_indptr->data()[phys_idx];
        }
        return total;
    }

    // CSR interface
    SCL_NODISCARD SCL_FORCE_INLINE Array<T> row_values(Index i) const requires (IsCSR) {
        Index phys_row = _map[i];
        const Index* indptr = _src_indptr->data();
        Index start = indptr[phys_row];
        Size len = static_cast<Size>(indptr[phys_row + 1] - start);
        return Array<T>(const_cast<T*>(_src_data->data() + start), len);
    }

    SCL_NODISCARD SCL_FORCE_INLINE Array<Index> row_indices(Index i) const requires (IsCSR) {
        Index phys_row = _map[i];
        const Index* indptr = _src_indptr->data();
        Index start = indptr[phys_row];
        Size len = static_cast<Size>(indptr[phys_row + 1] - start);
        return Array<Index>(const_cast<Index*>(_src_indices->data() + start), len);
    }

    SCL_NODISCARD SCL_FORCE_INLINE Index row_length(Index i) const requires (IsCSR) {
        Index phys_row = _map[i];
        const Index* indptr = _src_indptr->data();
        return indptr[phys_row + 1] - indptr[phys_row];
    }

    // CSC interface
    SCL_NODISCARD SCL_FORCE_INLINE Array<T> col_values(Index j) const requires (!IsCSR) {
        Index phys_col = _map[j];
        const Index* indptr = _src_indptr->data();
        Index start = indptr[phys_col];
        Size len = static_cast<Size>(indptr[phys_col + 1] - start);
        return Array<T>(const_cast<T*>(_src_data->data() + start), len);
    }

    SCL_NODISCARD SCL_FORCE_INLINE Array<Index> col_indices(Index j) const requires (!IsCSR) {
        Index phys_col = _map[j];
        const Index* indptr = _src_indptr->data();
        Index start = indptr[phys_col];
        Size len = static_cast<Size>(indptr[phys_col + 1] - start);
        return Array<Index>(const_cast<Index*>(_src_indices->data() + start), len);
    }

    SCL_NODISCARD SCL_FORCE_INLINE Index col_length(Index j) const requires (!IsCSR) {
        Index phys_col = _map[j];
        const Index* indptr = _src_indptr->data();
        return indptr[phys_col + 1] - indptr[phys_col];
    }

    // Materialization
    OwnedSparse<T, IsCSR> materialize() const {
        Index primary_dim = IsCSR ? rows : cols;
        Index total_nnz = nnz();
        
        std::vector<T> data_copy(total_nnz);
        std::vector<Index> indices_copy(total_nnz);
        std::vector<Index> indptr_copy(primary_dim + 1);
        
        indptr_copy[0] = 0;
        Index offset = 0;
        
        for (Index i = 0; i < primary_dim; ++i) {
            auto vals = IsCSR ? row_values(i) : col_values(i);
            auto inds = IsCSR ? row_indices(i) : col_indices(i);
            
            std::memcpy(data_copy.data() + offset, vals.ptr, vals.len * sizeof(T));
            std::memcpy(indices_copy.data() + offset, inds.ptr, inds.len * sizeof(Index));
            
            offset += vals.len;
            indptr_copy[i + 1] = offset;
        }
        
        return OwnedSparse<T, IsCSR>(
            std::move(data_copy),
            std::move(indices_copy),
            std::move(indptr_copy),
            rows, cols
        );
    }
};

// =============================================================================
// OwnedSparse: Heap-Allocated Sparse Matrix
// =============================================================================

/// @brief Sparse matrix with owned heap storage
///
/// Target for materialization. Manages memory lifetime.
template <typename T, bool IsCSR>
struct OwnedSparse {
    std::vector<T> data;
    std::vector<Index> indices;
    std::vector<Index> indptr;
    
    Index rows;
    Index cols;
    
    OwnedSparse() : rows(0), cols(0) {}
    
    OwnedSparse(
        std::vector<T>&& d,
        std::vector<Index>&& i,
        std::vector<Index>&& p,
        Index r, Index c
    )
        : data(std::move(d)), 
          indices(std::move(i)), 
          indptr(std::move(p)),
          rows(r), cols(c)
    {}
    
    /// @brief Get CustomSparse view (non-owning)
    CustomSparse<T, IsCSR> view() noexcept {
        return CustomSparse<T, IsCSR>(
            data.data(), 
            indices.data(), 
            indptr.data(), 
            rows, cols
        );
    }
    
    /// @brief Get nnz
    Index nnz() const {
        Index primary_dim = IsCSR ? rows : cols;
        return indptr[primary_dim];
    }
};

// =============================================================================
// Type Aliases
// =============================================================================

// MappedCustomSparse aliases
template <typename T>
using MappedCustomCSR = MappedCustomSparse<T, true>;

template <typename T>
using MappedCustomCSC = MappedCustomSparse<T, false>;

using MappedCustomCSRReal = MappedCustomCSR<Real>;
using MappedCustomCSCReal = MappedCustomCSC<Real>;

// MappedVirtualSparse aliases
template <typename T>
using MappedVirtualCSR = MappedVirtualSparse<T, true>;

template <typename T>
using MappedVirtualCSC = MappedVirtualSparse<T, false>;

using MappedVirtualCSRReal = MappedVirtualCSR<Real>;
using MappedVirtualCSCReal = MappedVirtualCSC<Real>;

// OwnedSparse aliases
template <typename T>
using OwnedCSR = OwnedSparse<T, true>;

template <typename T>
using OwnedCSC = OwnedSparse<T, false>;

using OwnedCSRReal = OwnedCSR<Real>;
using OwnedCSCReal = OwnedCSC<Real>;

// Concept verification
static_assert(SparseLike<MappedCustomCSR<Real>, true>);
static_assert(SparseLike<MappedCustomCSC<Real>, false>);
static_assert(SparseLike<MappedVirtualCSR<Real>, true>);
static_assert(SparseLike<MappedVirtualCSC<Real>, false>);

} // namespace scl::io
