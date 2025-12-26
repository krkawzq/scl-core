#pragma once

#include "scl/io/mmap.hpp"
#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/lifetime.hpp"
#include "scl/threading/parallel_for.hpp"

#include <vector>
#include <tuple>
#include <cstring>

// =============================================================================
/// @file mmatrix.hpp
/// @brief Memory-Mapped Sparse Matrix with Lifetime Management
///
/// Provides zero-copy sparse matrix views over memory-mapped files,
/// with proper lifetime management for conversions.
///
/// Key Types:
///
/// 1. **MappedArray<T>** - Memory-mapped array (file-backed)
/// 2. **OwnedArray<T>** - Heap-allocated array with RAII (uses MemHandle)
/// 3. **MappedCustomSparse<T, IsCSR>** - Memory-mapped sparse matrix
/// 4. **OwnedSparse<T, IsCSR>** - Heap-allocated sparse matrix
///
/// Conversion Functions:
///
/// - MappedArray → OwnedArray: `mapped.to_owned()`
/// - MappedArray → Array: `mapped.as_array()` (non-owning view)
/// - OwnedArray → Array: `owned.as_array()` (non-owning view)
/// - MappedCustomSparse → OwnedSparse: `mapped.materialize()`
/// - OwnedSparse → CustomSparse: `owned.view()` (non-owning view)
///
/// Design:
/// - Built on MappedArray from mmap.hpp
/// - Uses MemHandle from lifetime.hpp for memory management
/// - Satisfies SparseLike concept
/// - Zero-copy operations where possible
/// - Parallel materialization for large data
// =============================================================================

namespace scl::io {

// =============================================================================
// Forward Declarations
// =============================================================================

template <typename T> class OwnedArray;
template <typename T, bool IsCSR> class MappedCustomSparse;
template <typename T, bool IsCSR> class MappedVirtualSparse;
template <typename T, bool IsCSR> struct OwnedSparse;

namespace detail {

/// @brief Parallel memory copy for large arrays
template <typename T>
inline void parallel_copy(const T* src, T* dst, Size count) {
    constexpr Size MIN_PARALLEL = 1024 * 1024;  // 1M elements
    
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
// OwnedArray: Heap-Allocated Array with RAII
// =============================================================================

/// @brief Heap-allocated array with automatic memory management
///
/// Uses MemHandle from lifetime.hpp to manage memory.
/// Provides conversions to/from Array views.
///
/// Memory Ownership:
/// - OwnedArray owns its memory (freed on destruction)
/// - as_array() returns a non-owning view
/// - release() transfers ownership to caller
///
/// @tparam T Element type
template <typename T>
class OwnedArray {
public:
    using value_type = T;
    
private:
    core::MemHandle _handle;
    
public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------
    
    /// @brief Default constructor (empty)
    OwnedArray() = default;
    
    /// @brief Construct with allocated memory (uninitialized)
    explicit OwnedArray(Size count) 
        : _handle(core::mem::alloc_array<T>(count)) {}
    
    /// @brief Construct with allocated memory (zero-initialized)
    static OwnedArray zeros(Size count) {
        OwnedArray arr;
        arr._handle = core::mem::alloc_array_zero<T>(count);
        return arr;
    }
    
    /// @brief Construct with aligned memory (for SIMD)
    static OwnedArray aligned(Size count, Size alignment = 64) {
        OwnedArray arr;
        arr._handle = core::mem::alloc_array_aligned<T>(count, alignment);
        return arr;
    }
    
    /// @brief Construct from existing MemHandle (takes ownership)
    explicit OwnedArray(core::MemHandle&& handle) 
        : _handle(std::move(handle)) {}
    
    /// @brief Copy from Array (deep copy)
    static OwnedArray from_array(Array<const T> src) {
        if (src.len == 0) return OwnedArray();
        
        OwnedArray arr(src.len);
        detail::parallel_copy(src.ptr, arr.data(), src.len);
        return arr;
    }
    
    /// @brief Copy from raw pointer (deep copy)
    static OwnedArray from_ptr(const T* src, Size count) {
        return from_array(Array<const T>(src, count));
    }
    
    // -------------------------------------------------------------------------
    // Move Semantics (No Copy)
    // -------------------------------------------------------------------------
    
    OwnedArray(OwnedArray&&) noexcept = default;
    OwnedArray& operator=(OwnedArray&&) noexcept = default;
    
    OwnedArray(const OwnedArray&) = delete;
    OwnedArray& operator=(const OwnedArray&) = delete;
    
    /// @brief Deep copy (explicit)
    [[nodiscard]] OwnedArray clone() const {
        return from_array(as_array());
    }
    
    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------
    
    /// @brief Get raw pointer
    [[nodiscard]] T* data() noexcept { 
        return _handle.as<T>(); 
    }
    
    [[nodiscard]] const T* data() const noexcept { 
        return _handle.as<T>(); 
    }
    
    /// @brief Get number of elements
    [[nodiscard]] Size size() const noexcept { 
        return _handle.count<T>(); 
    }
    
    /// @brief Check if empty
    [[nodiscard]] bool empty() const noexcept { 
        return _handle.empty(); 
    }
    
    /// @brief Element access
    [[nodiscard]] T& operator[](Size i) noexcept { 
        return data()[i]; 
    }
    
    [[nodiscard]] const T& operator[](Size i) const noexcept { 
        return data()[i]; 
    }
    
    // -------------------------------------------------------------------------
    // Conversion to Array (Non-Owning View)
    // -------------------------------------------------------------------------
    
    /// @brief Get non-owning Array view
    ///
    /// WARNING: The returned Array is only valid while this OwnedArray exists!
    [[nodiscard]] Array<T> as_array() noexcept {
        return Array<T>(data(), size());
    }
    
    [[nodiscard]] Array<const T> as_array() const noexcept {
        return Array<const T>(data(), size());
    }
    
    /// @brief Implicit conversion to Array (non-owning view)
    operator Array<T>() noexcept { return as_array(); }
    operator Array<const T>() const noexcept { return as_array(); }
    
    // -------------------------------------------------------------------------
    // Lifetime Control
    // -------------------------------------------------------------------------
    
    /// @brief Release ownership (for handoff to external code)
    ///
    /// After calling release(), this OwnedArray becomes empty.
    /// The caller is responsible for freeing the memory!
    [[nodiscard]] T* release() noexcept {
        return _handle.release<T>();
    }
    
    /// @brief Get underlying MemHandle (moves ownership)
    [[nodiscard]] core::MemHandle release_handle() noexcept {
        return std::move(_handle);
    }
    
    /// @brief Free memory immediately
    void reset() noexcept {
        _handle.reset();
    }
};

// =============================================================================
// MappedArray Extensions (Conversion Methods)
// =============================================================================

// Note: These are free functions because MappedArray is defined in mmap.hpp

/// @brief Convert MappedArray to OwnedArray (deep copy)
///
/// Copies data from memory-mapped file to heap memory.
/// The returned OwnedArray owns the copied data.
///
/// @param mapped Source memory-mapped array
/// @return OwnedArray with copied data
template <typename T>
[[nodiscard]] inline OwnedArray<T> to_owned(const MappedArray<T>& mapped) {
    if (mapped.size() == 0) return OwnedArray<T>();
    
    OwnedArray<T> owned(mapped.size());
    detail::parallel_copy(mapped.data(), owned.data(), mapped.size());
    return owned;
}

/// @brief Convert MappedArray to Array view (non-owning, zero-copy)
///
/// WARNING: The returned Array is only valid while the MappedArray exists!
///
/// @param mapped Source memory-mapped array
/// @return Non-owning Array view
template <typename T>
[[nodiscard]] inline Array<const T> as_array(const MappedArray<T>& mapped) noexcept {
    return Array<const T>(mapped.data(), mapped.size());
}

/// @brief Convert MappedArray to mutable Array view
template <typename T>
[[nodiscard]] inline Array<T> as_mutable_array(MappedArray<T>& mapped) noexcept {
    return Array<T>(const_cast<T*>(mapped.data()), mapped.size());
}

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

    // -------------------------------------------------------------------------
    // SparseLike Interface
    // -------------------------------------------------------------------------

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

    // -------------------------------------------------------------------------
    // Direct Array Access
    // -------------------------------------------------------------------------

    SCL_NODISCARD T* data() const noexcept { return const_cast<T*>(_data.data()); }
    SCL_NODISCARD Index* indices() const noexcept { return const_cast<Index*>(_indices.data()); }
    SCL_NODISCARD Index* indptr() const noexcept { return const_cast<Index*>(_indptr.data()); }

    // -------------------------------------------------------------------------
    // Conversion Methods
    // -------------------------------------------------------------------------

    /// @brief Materialize to OwnedSparse (deep copy)
    ///
    /// Copies all data from memory-mapped files to heap memory.
    /// The returned OwnedSparse owns all the copied data.
    [[nodiscard]] OwnedSparse<T, IsCSR> materialize() const {
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

    /// @brief Get CustomSparse view (non-owning, zero-copy)
    ///
    /// WARNING: The returned CustomSparse is only valid while this
    /// MappedCustomSparse exists!
    [[nodiscard]] CustomSparse<T, IsCSR> as_view() const noexcept {
        return CustomSparse<T, IsCSR>(
            const_cast<T*>(_data.data()),
            const_cast<Index*>(_indices.data()),
            const_cast<Index*>(_indptr.data()),
            rows, cols
        );
    }

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

    // -------------------------------------------------------------------------
    // Conversion Methods
    // -------------------------------------------------------------------------

    /// @brief Materialize to OwnedSparse (deep copy with remapping)
    [[nodiscard]] OwnedSparse<T, IsCSR> materialize() const {
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
/// Target for materialization. Manages memory lifetime via std::vector.
/// Provides view() to get non-owning CustomSparse.
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
    ///
    /// WARNING: The returned CustomSparse is only valid while this
    /// OwnedSparse exists!
    [[nodiscard]] CustomSparse<T, IsCSR> view() noexcept {
        return CustomSparse<T, IsCSR>(
            data.data(), 
            indices.data(), 
            indptr.data(), 
            rows, cols
        );
    }
    
    [[nodiscard]] CustomSparse<T, IsCSR> view() const noexcept {
        return CustomSparse<T, IsCSR>(
            const_cast<T*>(data.data()), 
            const_cast<Index*>(indices.data()), 
            const_cast<Index*>(indptr.data()), 
            rows, cols
        );
    }
    
    /// @brief Get nnz
    [[nodiscard]] Index nnz() const {
        Index primary_dim = IsCSR ? rows : cols;
        return primary_dim > 0 ? indptr[primary_dim] : 0;
    }
    
    /// @brief Check if empty
    [[nodiscard]] bool empty() const noexcept {
        return rows == 0 && cols == 0;
    }
    
    /// @brief Deep copy
    [[nodiscard]] OwnedSparse clone() const {
        return OwnedSparse(
            std::vector<T>(data),
            std::vector<Index>(indices),
            std::vector<Index>(indptr),
            rows, cols
        );
    }
};

// =============================================================================
// Convenience Functions
// =============================================================================

/// @brief Create OwnedSparse from CustomSparse (deep copy)
template <typename T, bool IsCSR>
[[nodiscard]] inline OwnedSparse<T, IsCSR> to_owned(const CustomSparse<T, IsCSR>& sparse) {
    Index primary_dim = IsCSR ? sparse.rows : sparse.cols;
    Index total_nnz = sparse.indptr[primary_dim];
    
    std::vector<T> data_copy(sparse.data, sparse.data + total_nnz);
    std::vector<Index> indices_copy(sparse.indices, sparse.indices + total_nnz);
    std::vector<Index> indptr_copy(sparse.indptr, sparse.indptr + primary_dim + 1);
    
    return OwnedSparse<T, IsCSR>(
        std::move(data_copy),
        std::move(indices_copy),
        std::move(indptr_copy),
        sparse.rows, sparse.cols
    );
}

/// @brief Create OwnedSparse from VirtualSparse (deep copy with remapping)
template <typename T, bool IsCSR>
[[nodiscard]] inline OwnedSparse<T, IsCSR> to_owned(const VirtualSparse<T, IsCSR>& sparse) {
    Index primary_dim = IsCSR ? sparse.rows : sparse.cols;
    Index total_nnz = scl::nnz(sparse);
    
    std::vector<T> data_copy;
    std::vector<Index> indices_copy;
    std::vector<Index> indptr_copy;
    
    data_copy.reserve(total_nnz);
    indices_copy.reserve(total_nnz);
    indptr_copy.reserve(primary_dim + 1);
    indptr_copy.push_back(0);
    
    for (Index i = 0; i < primary_dim; ++i) {
        auto vals = scl::primary_values(sparse, i);
        auto inds = scl::primary_indices(sparse, i);
        
        data_copy.insert(data_copy.end(), vals.ptr, vals.ptr + vals.len);
        indices_copy.insert(indices_copy.end(), inds.ptr, inds.ptr + inds.len);
        indptr_copy.push_back(static_cast<Index>(data_copy.size()));
    }
    
    return OwnedSparse<T, IsCSR>(
        std::move(data_copy),
        std::move(indices_copy),
        std::move(indptr_copy),
        sparse.rows, sparse.cols
    );
}

// =============================================================================
// Type Aliases
// =============================================================================

// OwnedArray aliases
using OwnedArrayReal = OwnedArray<Real>;
using OwnedArrayIndex = OwnedArray<Index>;

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

// =============================================================================
// Concept Verification
// =============================================================================

static_assert(SparseLike<MappedCustomCSR<Real>, true>);
static_assert(SparseLike<MappedCustomCSC<Real>, false>);
static_assert(SparseLike<MappedVirtualCSR<Real>, true>);
static_assert(SparseLike<MappedVirtualCSC<Real>, false>);

} // namespace scl::io
