#pragma once

#include "scl/config.hpp"
#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"

#include <concepts>
#include <type_traits>
#include <vector>
#include <stdexcept>

// =============================================================================
/// @file matrix.hpp
/// @brief SCL Matrix System: Unified Interface with Template-Based Layout
///
/// Key Innovation: CSR/CSC Unification via Template Parameter
///
/// Instead of separate TagCSR and TagCSC, we use:
///   TagSparse<true>  = CSR (row-major sparse)
///   TagSparse<false> = CSC (column-major sparse)
///
/// Benefits:
/// - Single implementation for CustomSparse<T, IsCSR>
/// - Single implementation for VirtualSparse<T, IsCSR>
/// - Eliminates code duplication
/// - Type-safe compile-time dispatch
///
/// Architecture:
///
/// Layer 0: Tags
///   TagDense, TagSparse<bool IsCSR>
///
/// Layer 1: Unified Accessors
///   scl::rows(), scl::cols(), scl::nnz()
///
/// Layer 2: Concepts
///   DenseLike, SparseLike<bool IsCSR>
///
/// Layer 3: Virtual Interfaces
///   IDense<T>, ISparse<T, bool IsCSR>
///
/// Layer 4: Utilities
///   primary_values(), primary_indices()
// =============================================================================

namespace scl {

// =============================================================================
// Layer 0: Type Tags (Template-Based)
// =============================================================================

/// @brief Tag for dense matrices.
struct TagDense {};

/// @brief Tag for sparse matrices (CSR or CSC).
///
/// Template parameter:
/// - IsCSR = true: Compressed Sparse Row (row-major)
/// - IsCSR = false: Compressed Sparse Column (column-major)
template <bool IsCSR>
struct TagSparse {
    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;
};

/// @brief Type alias for CSR tag.
using TagCSR = TagSparse<true>;

/// @brief Type alias for CSC tag.
using TagCSC = TagSparse<false>;

/// @brief Trait to check if tag is sparse.
template <typename Tag>
struct is_sparse_tag : std::false_type {};

template <bool IsCSR>
struct is_sparse_tag<TagSparse<IsCSR>> : std::true_type {};

template <typename Tag>
inline constexpr bool is_sparse_tag_v = is_sparse_tag<Tag>::value;

/// @brief Extract IsCSR from Tag.
template <typename Tag>
struct tag_is_csr;

template <bool IsCSR>
struct tag_is_csr<TagSparse<IsCSR>> : std::bool_constant<IsCSR> {};

template <typename Tag>
inline constexpr bool tag_is_csr_v = tag_is_csr<Tag>::value;

// =============================================================================
// Layer 1: Unified Accessors (POD + Virtual Support)
// =============================================================================

namespace detail {
    // Member variable detection
    template <typename T> 
    concept HasRowMember = requires(const T& t) { 
        { t.rows } -> std::convertible_to<Index>; 
    };
    
    template <typename T> 
    concept HasColMember = requires(const T& t) { 
        { t.cols } -> std::convertible_to<Index>; 
    };
    
    template <typename T> 
    concept HasNnzMember = requires(const T& t) { 
        { t.nnz } -> std::convertible_to<Index>; 
    };
    
    // Member function detection
    template <typename T> 
    concept HasRowMethod = requires(const T& t) { 
        { t.rows() } -> std::convertible_to<Index>; 
    };
    
    template <typename T> 
    concept HasColMethod = requires(const T& t) { 
        { t.cols() } -> std::convertible_to<Index>; 
    };
    
    template <typename T> 
    concept HasNnzMethod = requires(const T& t) { 
        { t.nnz() } -> std::convertible_to<Index>; 
    };
}

/// @brief Unified accessor for row count (POD member or virtual method).
template <typename M>
SCL_NODISCARD SCL_FORCE_INLINE constexpr Index rows(const M& m) {
    if constexpr (detail::HasRowMember<M>) {
        return m.rows;  // Quick path: zero-cost
    } else {
        return m.rows();  // Unified path: virtual call
    }
}

/// @brief Unified accessor for column count.
template <typename M>
SCL_NODISCARD SCL_FORCE_INLINE constexpr Index cols(const M& m) {
    if constexpr (detail::HasColMember<M>) {
        return m.cols;
    } else {
        return m.cols();
    }
}

/// @brief Unified accessor for non-zero count.
template <typename M>
SCL_NODISCARD SCL_FORCE_INLINE constexpr Index nnz(const M& m) {
    if constexpr (detail::HasNnzMember<M>) {
        return m.nnz;
    } else {
        return m.nnz();
    }
}

/// @brief Unified accessor for data pointer (dense matrices).
template <typename M>
SCL_NODISCARD SCL_FORCE_INLINE constexpr const typename M::ValueType* ptr(const M& m) {
    if constexpr (requires { { m.ptr } -> std::convertible_to<const typename M::ValueType*>; }) {
        return m.ptr;
    } else {
        return m.data();
    }
}

// =============================================================================
// Layer 2: Concepts (Interface Contracts)
// =============================================================================

/// @brief Concept for 1D array-like containers (vectors).
///
/// ArrayLike represents a contiguous 1D array with random access.
/// Used for storing sparse matrix data arrays (values, indices, indptr).
///
/// Requirements:
/// - value_type: Element type alias
/// - data(): Returns pointer to contiguous data
/// - size(): Returns number of elements
/// - operator[]: Random access by index
/// - begin()/end(): Iterator support for range-based loops
///
/// Examples: std::vector<T>, std::array<T>, MappedArray<T>, Span<T>
template <typename A>
concept ArrayLike = requires(const A& a, Index i) {
    typename A::value_type;
    { a.data() } -> std::convertible_to<const typename A::value_type*>;
    { a.size() } -> std::convertible_to<Size>;
    { a[i] } -> std::convertible_to<const typename A::value_type&>;
    { a.begin() } -> std::input_iterator;
    { a.end() } -> std::input_iterator;
};

/// @brief Concept for 2D dense matrices.
///
/// DenseLike represents a dense matrix with 2D indexing.
/// Supports row-major or column-major contiguous storage.
///
/// Requirements:
/// - ValueType: Element type
/// - Tag: TagDense
/// - rows(), cols(): Dimensions
/// - ptr(): Pointer to underlying contiguous data
/// - operator()(i, j): 2D element access
///
/// Examples: DenseArray<T>, Eigen::Matrix<T>, numpy array view
template <typename M>
concept DenseLike = requires(const M& m, Index i, Index j) {
    typename M::ValueType;
    typename M::Tag;
    requires std::is_same_v<typename M::Tag, TagDense>;
    
    { scl::rows(m) } -> std::convertible_to<Index>;
    { scl::cols(m) } -> std::convertible_to<Index>;
    { scl::ptr(m) } -> std::convertible_to<const typename M::ValueType*>;
    { m(i, j) } -> std::convertible_to<typename M::ValueType>;
};

/// @brief Unified concept for sparse matrices (CSR or CSC).
///
/// Template parameter IsCSR:
/// - true: CSR requirements (row_values, row_indices, row_length)
/// - false: CSC requirements (col_values, col_indices, col_length)
///
/// Usage:
///
/// template <SparseLike<true> M>   // CSR only
/// void csr_algo(M& mat);
///
/// template <bool IsCSR, SparseLike<IsCSR> M>  // CSR or CSC
/// void unified_algo(M& mat) {
///     if constexpr (IsCSR) {
///         // CSR path
///     } else {
///         // CSC path
///     }
/// }
template <bool IsCSR>
struct SparseLikeConcept {
    template <typename M>
    static constexpr bool value = requires(const M& m, Index i) {
        typename M::ValueType;
        typename M::Tag;
        requires is_sparse_tag_v<typename M::Tag>;
        requires tag_is_csr_v<typename M::Tag> == IsCSR;
        
        // Dimensions
        { scl::rows(m) } -> std::convertible_to<Index>;
        { scl::cols(m) } -> std::convertible_to<Index>;
        { scl::nnz(m) } -> std::convertible_to<Index>;
    } && (
        (IsCSR && requires(const M& m, Index i) {
            // CSR requirements
            { m.row_values(i) } -> std::convertible_to<Span<typename M::ValueType>>;
            { m.row_indices(i) } -> std::convertible_to<Span<Index>>;
            { m.row_length(i) } -> std::convertible_to<Index>;
        })
        ||
        (!IsCSR && requires(const M& m, Index j) {
            // CSC requirements
            { m.col_values(j) } -> std::convertible_to<Span<typename M::ValueType>>;
            { m.col_indices(j) } -> std::convertible_to<Span<Index>>;
            { m.col_length(j) } -> std::convertible_to<Index>;
        })
    );
};

/// @brief SparseLike concept with template parameter.
template <typename M, bool IsCSR>
concept SparseLike = SparseLikeConcept<IsCSR>::template value<M>;

/// @brief Any sparse matrix (CSR or CSC).
template <typename M>
concept AnySparse = SparseLike<M, true> || SparseLike<M, false>;

/// @brief CSR-specific matrices (convenience alias).
template <typename M>
concept CSRLike = SparseLike<M, true>;

/// @brief CSC-specific matrices (convenience alias).
template <typename M>
concept CSCLike = SparseLike<M, false>;

// =============================================================================
// Storage Pattern Concepts (Interface Contracts)
// =============================================================================
//
// These concepts define INTERFACE requirements, not implementations.
// They describe "what" a matrix should provide, not "how" it's implemented.
//
// Reference Implementations:
// - CustomSparse<T, IsCSR> in sparse.hpp (standard implementation)
// - VirtualSparse<T, IsCSR> in sparse.hpp (standard implementation)
// - MountMatrix<T> in io/mmatrix.hpp (mmap-based)
// - DequeCSR<T> in io/h5_tools.hpp (deque-based)
//
// =============================================================================

/// @brief Concept for sparse matrices with Custom pattern (Standard CSR/CSC).
///
/// Interface Contract - Three Contiguous Arrays:
///
/// Storage Requirements:
///   T* data;           // Values (size: computed from indptr[-1])
///   Index* indices;    // Secondary dimension indices (size: same as data)
///   Index* indptr;     // Primary dimension pointers (size: n+1, cumulative offsets)
///
/// Type Constraints (STRONG):
///   1. data: ValueType* (T by default, or Real)
///   2. indices: Index* (NOT Pointer*, must be Index for arithmetic)
///   3. indptr: Index* (NOT Pointer*, must be Index for cumulative offsets)
///
/// nnz Derivation:
///   - NOT explicitly stored as member
///   - Derived from: static_cast<Size>(indptr[primary_dimension] - 1)
///   - Requires Index to Size conversion
///
/// Memory Layout (Standard CSR/CSC):
///   data:    [val0, val1, val2, val3, val4, val5, ...]
///   indices: [col0, col1, col2, col3, col4, col5, ...]
///   indptr:  [0,    2,    2,    5,    8]  (n+1 elements)
///             row0  row1  row2  row3   end
///
///   nnz = indptr[n] = 8
///
/// Enables:
///   - SIMD batch operations on entire data array
///   - Zero-copy NumPy/SciPy interop
///   - Standard sparse matrix algorithms
///   - Optimal memory locality
///
/// Satisfied By:
///   - CustomSparse<T, IsCSR> (heap pointers)
///   - MountMatrix<T> (mmap pointers)
///   - OwnedCSR<T> (vector::data() pointers)
///
/// Not Satisfied By:
///   - VirtualSparse: Uses Pointer* arrays, explicit nnz
///   - DequeCSR: Discontiguous, pointer arrays
///   - ISparse: Virtual base class
///
/// Usage:
///
/// template <typename T, bool IsCSR, CustomSparseLike<IsCSR> M>
/// void simd_algo(M& mat) {
///     // Guaranteed contiguous
///     Size nnz_val = static_cast<Size>(mat.indptr[mat.rows] - 1);
///     simd::process(mat.data, nnz_val);
/// }
template <typename M, bool IsCSR>
concept CustomSparseLike = SparseLike<M, IsCSR> && requires(const M& m) {
    // Three contiguous arrays (STRONG type constraints)
    { m.data } -> std::convertible_to<typename M::ValueType*>;
    { m.indices } -> std::convertible_to<Index*>;    // STRONG: Index*, not Pointer*
    { m.indptr } -> std::convertible_to<Index*>;     // STRONG: Index*, n+1 elements
};

/// @brief Concept for sparse matrices with Virtual pattern (ArrayLike containers).
///
/// Interface Contract - Three ArrayLike Containers + Explicit nnz:
///
/// Storage Requirements:
///   ArrayLike<T> data_container;           // Values (can be vector, deque, etc.)
///   ArrayLike<Index> indices_container;    // Secondary indices
///   ArrayLike<Pointer> ptr_container;      // Pointers to row/col starts
///   Size nnz;                              // Explicitly stored total nnz
///
/// Type Constraints (STRONG):
///   1. data_container: Must be ArrayLike with value_type = T (or Real)
///   2. indices_container: Must be ArrayLike with value_type = Index
///   3. ptr_container: Must be ArrayLike with value_type = Pointer
///   4. nnz: Must be Size type (explicitly stored, NOT derived)
///
/// Container Semantics:
///   - data_container[n]: Can be any ArrayLike (std::vector, std::deque, custom)
///   - Enables nested concept support
///   - Fully flexible storage backend
///
/// Memory Pattern (Discontiguous):
///   Row 0: data @ ptr_container[0]     -> [val0, val1]
///   Row 1: data @ ptr_container[1]     -> [val2]
///   Row 2: data @ ptr_container[2]     -> [val3, val4, val5]
///
///   data_container: Can be segmented/discontiguous
///   indices_container: Can be segmented/discontiguous  
///   ptr_container: Pointers to each segment
///
///   nnz = 6 (explicitly stored)
///
/// Key Difference from Custom:
///   - Custom: Index* indptr (n+1 offsets), nnz derived
///   - Virtual: Pointer* ptr_container (n pointers), nnz explicit
///
/// Enables:
///   - Fully discontiguous storage (deque, memory-pool)
///   - Incremental construction (unknown final nnz)
///   - Nested concept composition (ArrayLike containers)
///   - Flexible memory management
///
/// Satisfied By:
///   - VirtualSparse<T, IsCSR> (pointer-array-based)
///   - DequeCSR<T> (deque-backed rows)
///   - SegmentedCSR<T> (memory-pool segments)
///
/// Not Satisfied By:
///   - CustomSparse: Uses Index* indptr, derived nnz
///   - ISparse: Virtual base, different mechanism
///
/// Performance:
///   - Direct pointer access (no offset computation)
///   - Supports arbitrary memory layouts
///   - nnz lookup: O(1) (stored explicitly)
///
/// Usage:
///
/// template <typename T, bool IsCSR, VirtualSparseLike<IsCSR> M>
/// void virtual_algo(M& mat) {
///     // Access explicit nnz
///     Size total = mat.nnz;  // O(1), not computed
///     
///     // Access via containers (ArrayLike)
///     // Implementation handles container details
/// }
template <typename M, bool IsCSR>
concept VirtualSparseLike = SparseLike<M, IsCSR> && 
    // Key distinction: HAS pointer arrays, NOT Index* indptr
    requires(const M& m) {
        { m.data_ptrs } -> std::convertible_to<Pointer*>;
        { m.indices_ptrs } -> std::convertible_to<Pointer*>;
        { m.lengths } -> std::convertible_to<Index*>;
    } &&
    // Negative constraint: Must NOT have Index* indptr (that's CustomSparseLike)
    !requires(const M& m) {
        { m.indptr } -> std::convertible_to<Index*>;
    };

/// @brief Refined concepts (single bool parameter).
template <typename M>
concept CustomCSRLike = CustomSparseLike<M, true>;

template <typename M>
concept VirtualCSRLike = VirtualSparseLike<M, true>;

template <typename M>
concept CustomCSCLike = CustomSparseLike<M, false>;

template <typename M>
concept VirtualCSCLike = VirtualSparseLike<M, false>;

// =============================================================================
// Layer 3: Virtual Interfaces (User Inheritance Support)
// =============================================================================

/// @brief Abstract base for dense matrices.
template <typename T>
struct IDense {
    using ValueType = T;
    using Tag = TagDense;
    
    virtual ~IDense() = default;
    
    virtual Index rows() const = 0;
    virtual Index cols() const = 0;
    
    // Const-correct data access
    virtual const T* data() const { return nullptr; }
    
    // Non-const overload for mutable access
    virtual T* data() { return nullptr; }
    
    virtual T operator()(Index i, Index j) const = 0;
    
    // Nested concept support: row returns DenseLike vector
    virtual std::vector<T> row(Index i) const {
        Index n_cols = cols();
        std::vector<T> result(n_cols);
        for (Index j = 0; j < n_cols; ++j) {
            result[j] = (*this)(i, j);
        }
        return result;
    }
};

/// @brief Abstract base for sparse matrices (unified CSR/CSC).
///
/// Template parameter:
/// - IsCSR = true: CSR layout
/// - IsCSR = false: CSC layout
///
/// Users can inherit to create custom sparse types:
///
/// Example: Lazy-loaded CSR
///
/// class LazyCSR : public ISparse<float, true> {
///     Span<float> primary_values(Index i) const override {
///         return load_row_on_demand(i);
///     }
/// };
/// static_assert(CSRLike<LazyCSR>);  // ✓
template <typename T, bool IsCSR>
struct ISparse {
    using ValueType = T;
    using Tag = TagSparse<IsCSR>;
    
    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;
    
    virtual ~ISparse() = default;
    
    // Dimensions
    virtual Index rows() const = 0;
    virtual Index cols() const = 0;
    virtual Index nnz() const = 0;
    
    // Unified primary access (row for CSR, col for CSC)
    virtual Span<T> primary_values(Index i) const = 0;
    virtual Span<Index> primary_indices(Index i) const = 0;
    virtual Index primary_length(Index i) const {
        return static_cast<Index>(primary_values(i).size);
    }
    
    // Layout-specific interface (delegates to primary_*)
    // CSR interface
    Span<T> row_values(Index i) const 
        requires (IsCSR) 
    { 
        return primary_values(i); 
    }
    
    Span<Index> row_indices(Index i) const 
        requires (IsCSR) 
    { 
        return primary_indices(i); 
    }
    
    Index row_length(Index i) const 
        requires (IsCSR) 
    { 
        return primary_length(i); 
    }
    
    // CSC interface
    Span<T> col_values(Index j) const 
        requires (!IsCSR) 
    { 
        return primary_values(j); 
    }
    
    Span<Index> col_indices(Index j) const 
        requires (!IsCSR) 
    { 
        return primary_indices(j); 
    }
    
    Index col_length(Index j) const 
        requires (!IsCSR) 
    { 
        return primary_length(j); 
    }
};

/// @brief Type aliases for user convenience.
template <typename T>
using ICSR = ISparse<T, true>;

template <typename T>
using ICSC = ISparse<T, false>;

// =============================================================================
// Concept Verification for Virtual Interfaces
// =============================================================================

static_assert(DenseLike<IDense<float>>, "IDense must satisfy DenseLike");
static_assert(CSRLike<ICSR<float>>, "ICSR must satisfy CSRLike");
static_assert(CSCLike<ICSC<float>>, "ICSC must satisfy CSCLike");
static_assert(AnySparse<ICSR<float>>, "ICSR must satisfy AnySparse");
static_assert(AnySparse<ICSC<float>>, "ICSC must satisfy AnySparse");

// =============================================================================
// Layer 4: Traits and Utilities
// =============================================================================

/// @brief Sparse matrix traits (works for all SparseLike types).
template <typename M>
    requires (AnySparse<M>)
struct sparse_traits {
    static constexpr bool is_csr = tag_is_csr_v<typename M::Tag>;
    static constexpr bool is_csc = !is_csr;
};

/// @brief Get primary dimension size.
///
/// CSR: rows (iterate over rows)
/// CSC: cols (iterate over cols)
template <typename M>
    requires (AnySparse<M>)
SCL_NODISCARD SCL_FORCE_INLINE Index primary_size(const M& mat) {
    if constexpr (tag_is_csr_v<typename M::Tag>) {
        return scl::rows(mat);
    } else {
        return scl::cols(mat);
    }
}

/// @brief Get secondary dimension size.
///
/// CSR: cols
/// CSC: rows
template <typename M>
    requires (AnySparse<M>)
SCL_NODISCARD SCL_FORCE_INLINE Index secondary_size(const M& mat) {
    if constexpr (tag_is_csr_v<typename M::Tag>) {
        return scl::cols(mat);
    } else {
        return scl::rows(mat);
    }
}

/// @brief Get values span for primary dimension i.
///
/// Unified accessor: Calls row_values() for CSR, col_values() for CSC.
template <typename M>
    requires (AnySparse<M>)
SCL_NODISCARD SCL_FORCE_INLINE auto primary_values(const M& mat, Index i) {
    if constexpr (tag_is_csr_v<typename M::Tag>) {
        return mat.row_values(i);
    } else {
        return mat.col_values(i);
    }
}

/// @brief Get indices span for primary dimension i.
template <typename M>
    requires (AnySparse<M>)
SCL_NODISCARD SCL_FORCE_INLINE auto primary_indices(const M& mat, Index i) {
    if constexpr (tag_is_csr_v<typename M::Tag>) {
        return mat.row_indices(i);
    } else {
        return mat.col_indices(i);
    }
}

/// @brief Get length for primary dimension i.
template <typename M>
    requires (AnySparse<M>)
SCL_NODISCARD SCL_FORCE_INLINE Index primary_length(const M& mat, Index i) {
    if constexpr (tag_is_csr_v<typename M::Tag>) {
        return mat.row_length(i);
    } else {
        return mat.col_length(i);
    }
}

// =============================================================================
// Advanced Utilities
// =============================================================================

/// @brief Check if matrix has contiguous storage (compile-time).
template <typename M>
inline constexpr bool has_contiguous_storage_v = 
    (AnySparse<M> && CustomSparseLike<M, true>) ||
    (AnySparse<M> && CustomSparseLike<M, false>);

/// @brief Check if matrix uses indirection (compile-time).
template <typename M>
inline constexpr bool has_indirection_v = 
    (AnySparse<M> && VirtualSparseLike<M, true>) ||
    (AnySparse<M> && VirtualSparseLike<M, false>);

/// @brief Get element count for dense matrix.
template <DenseLike M>
SCL_NODISCARD constexpr Size element_count(const M& mat) {
    return static_cast<Size>(scl::rows(mat)) * static_cast<Size>(scl::cols(mat));
}

/// @brief Check if indices are valid.
template <typename M>
SCL_NODISCARD inline bool is_valid_index(const M& mat, Index i, Index j) {
    return i >= 0 && i < scl::rows(mat) && j >= 0 && j < scl::cols(mat);
}

/// @brief Safe element access with bounds checking (dense).
template <DenseLike M>
SCL_NODISCARD inline typename M::ValueType safe_at(const M& mat, Index i, Index j) {
    if (!is_valid_index(mat, i, j)) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return mat(i, j);
}

// =============================================================================
// Backward Compatibility Aliases
// =============================================================================

/// @brief Legacy alias (prefer CustomSparseLike<M, true/false>).
template <typename M>
concept ContiguousData = CustomSparseLike<M, true> || CustomSparseLike<M, false>;

/// @brief Legacy alias (prefer CustomCSRLike).
template <typename M>
concept CSRContiguous = CustomCSRLike<M>;

/// @brief Legacy alias (prefer CustomCSCLike).
template <typename M>
concept CSCContiguous = CustomCSCLike<M>;

// =============================================================================
// Type Checks (Compile-Time Verification Helpers)
// =============================================================================

template <typename T>
inline constexpr bool is_csr_like_v = CSRLike<T>;

template <typename T>
inline constexpr bool is_csc_like_v = CSCLike<T>;

template <typename T>
inline constexpr bool is_any_sparse_v = AnySparse<T>;

template <typename T>
inline constexpr bool is_dense_like_v = DenseLike<T>;

template <typename T>
inline constexpr bool is_custom_csr_v = CustomCSRLike<T>;

template <typename T>
inline constexpr bool is_virtual_csr_v = VirtualCSRLike<T>;

} // namespace scl
