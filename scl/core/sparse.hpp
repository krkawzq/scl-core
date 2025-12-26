#pragma once

#include "scl/core/type.hpp"

// =============================================================================
/// @file sparse.hpp  
/// @brief Sparse Matrix Implementations
///
/// Contains:
/// - ISparse<T, IsCSR>: Virtual interface for polymorphic sparse matrices
/// - CustomSparse<T, IsCSR>: Contiguous array implementation
/// - VirtualSparse<T, IsCSR>: Pointer array implementation
///
/// Design: Template-based CSR/CSC unification
/// - Single template handles both layouts
/// - Compile-time dispatch via IsCSR parameter
/// - Zero code duplication
// =============================================================================

namespace scl {

// =============================================================================
// Virtual Interface: ISparse
// =============================================================================

/// @brief Abstract base for sparse matrices (unified CSR/CSC)
///
/// Template parameter:
/// - IsCSR = true: CSR layout
/// - IsCSR = false: CSC layout
///
/// Example:
///
/// class LazyCSR : public ISparse<float, true> {
///     Array<float> primary_values(Index i) const override {
///         return load_row_on_demand(i);
///     }
/// };
template <typename T, bool IsCSR>
struct ISparse {
    using ValueType = T;
    using Tag = TagSparse<IsCSR>;
    
    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;
    
    virtual ~ISparse() = default;
    
    virtual Index rows() const = 0;
    virtual Index cols() const = 0;
    virtual Index nnz() const = 0;
    
    // Unified primary access
    virtual Array<T> primary_values(Index i) const = 0;
    virtual Array<Index> primary_indices(Index i) const = 0;
    virtual Index primary_length(Index i) const {
        return static_cast<Index>(primary_values(i).size());
    }
    
    // CSR interface
    Array<T> row_values(Index i) const requires (IsCSR) { return primary_values(i); }
    Array<Index> row_indices(Index i) const requires (IsCSR) { return primary_indices(i); }
    Index row_length(Index i) const requires (IsCSR) { return primary_length(i); }
    
    // CSC interface
    Array<T> col_values(Index j) const requires (!IsCSR) { return primary_values(j); }
    Array<Index> col_indices(Index j) const requires (!IsCSR) { return primary_indices(j); }
    Index col_length(Index j) const requires (!IsCSR) { return primary_length(j); }
};

template <typename T>
using ICSR = ISparse<T, true>;

template <typename T>
using ICSC = ISparse<T, false>;

static_assert(CSRLike<ICSR<float>>);
static_assert(CSCLike<ICSC<float>>);
static_assert(AnySparse<ICSR<float>>);

// =============================================================================
// CustomSparse: Contiguous Array Implementation
// =============================================================================

/// @brief Sparse matrix with contiguous array storage
///
/// Memory Layout:
/// - data[nnz]: Values array
/// - indices[nnz]: Secondary dimension indices
/// - indptr[primary+1]: Cumulative offsets
///
/// Template Parameters:
/// - T: Element type
/// - IsCSR: true = CSR, false = CSC
template <typename T, bool IsCSR>
struct CustomSparse {
    using ValueType = T;
    using Tag = TagSparse<IsCSR>;
    
    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;
    
    // Public members (direct access for fast path)
    T* data;
    Index* indices;
    Index* indptr;
    Index rows;
    Index cols;
    
    constexpr CustomSparse() noexcept 
        : data(nullptr), indices(nullptr), indptr(nullptr), rows(0), cols(0) {}
    
    constexpr CustomSparse(T* d, Index* idx, Index* ptr, Index r, Index c) noexcept
        : data(d), indices(idx), indptr(ptr), rows(r), cols(c) {}
    
    // Derived properties
    SCL_NODISCARD SCL_FORCE_INLINE Index nnz() const {
        if constexpr (IsCSR) {
            return indptr[rows];
        } else {
            return indptr[cols];
        }
    }
    
    // CSR interface
    SCL_NODISCARD SCL_FORCE_INLINE Array<T> row_values(Index i) const requires (IsCSR) {
        Index start = indptr[i];
        Index end = indptr[i + 1];
        return Array<T>(data + start, static_cast<Size>(end - start));
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Array<Index> row_indices(Index i) const requires (IsCSR) {
        Index start = indptr[i];
        Index end = indptr[i + 1];
        return Array<Index>(indices + start, static_cast<Size>(end - start));
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Index row_length(Index i) const requires (IsCSR) {
        return indptr[i + 1] - indptr[i];
    }
    
    // CSC interface
    SCL_NODISCARD SCL_FORCE_INLINE Array<T> col_values(Index j) const requires (!IsCSR) {
        Index start = indptr[j];
        Index end = indptr[j + 1];
        return Array<T>(data + start, static_cast<Size>(end - start));
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Array<Index> col_indices(Index j) const requires (!IsCSR) {
        Index start = indptr[j];
        Index end = indptr[j + 1];
        return Array<Index>(indices + start, static_cast<Size>(end - start));
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Index col_length(Index j) const requires (!IsCSR) {
        return indptr[j + 1] - indptr[j];
    }
};

using CustomCSR = CustomSparse<Real, true>;
using CustomCSC = CustomSparse<Real, false>;

static_assert(CSRLike<CustomCSR>);
static_assert(CSCLike<CustomCSC>);
static_assert(CustomSparseLike<CustomCSR, true>);

// =============================================================================
// VirtualSparse: Pointer Array Implementation
// =============================================================================

/// @brief Sparse matrix with discontiguous storage (pointer arrays)
///
/// Memory Layout:
/// - data_ptrs[primary]: Pointers to each row/col's values
/// - indices_ptrs[primary]: Pointers to each row/col's indices
/// - lengths[primary]: Length of each row/col
/// - nnz: Explicitly stored total nnz
template <typename T, bool IsCSR>
struct VirtualSparse {
    using ValueType = T;
    using Tag = TagSparse<IsCSR>;
    
    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;
    
    Pointer* data_ptrs;
    Pointer* indices_ptrs;
    Index* lengths;
    Index rows;
    Index cols;
    Index nnz;
    
    constexpr VirtualSparse() noexcept
        : data_ptrs(nullptr), indices_ptrs(nullptr), lengths(nullptr),
          rows(0), cols(0), nnz(0) {}
    
    constexpr VirtualSparse(Pointer* dp, Pointer* ip, Index* len, 
                           Index r, Index c, Index n) noexcept
        : data_ptrs(dp), indices_ptrs(ip), lengths(len),
          rows(r), cols(c), nnz(n) {}
    
    // CSR interface
    SCL_NODISCARD SCL_FORCE_INLINE Array<T> row_values(Index i) const requires (IsCSR) {
        return Array<T>(static_cast<T*>(data_ptrs[i]), static_cast<Size>(lengths[i]));
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Array<Index> row_indices(Index i) const requires (IsCSR) {
        return Array<Index>(static_cast<Index*>(indices_ptrs[i]), static_cast<Size>(lengths[i]));
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Index row_length(Index i) const requires (IsCSR) {
        return lengths[i];
    }
    
    // CSC interface
    SCL_NODISCARD SCL_FORCE_INLINE Array<T> col_values(Index j) const requires (!IsCSR) {
        return Array<T>(static_cast<T*>(data_ptrs[j]), static_cast<Size>(lengths[j]));
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Array<Index> col_indices(Index j) const requires (!IsCSR) {
        return Array<Index>(static_cast<Index*>(indices_ptrs[j]), static_cast<Size>(lengths[j]));
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Index col_length(Index j) const requires (!IsCSR) {
        return lengths[j];
    }
};

using VirtualCSR = VirtualSparse<Real, true>;
using VirtualCSC = VirtualSparse<Real, false>;

static_assert(CSRLike<VirtualCSR>);
static_assert(CSCLike<VirtualCSC>);
static_assert(VirtualSparseLike<VirtualCSR, true>);

} // namespace scl

