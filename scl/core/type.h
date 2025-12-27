// =============================================================================
// FILE: scl/core/type.h
// BRIEF: API reference for SCL unified type system
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/macros.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <concepts>

namespace scl {

/* =============================================================================
 * SECTION 1: BASIC TYPES
 * =============================================================================
 * SCL uses compile-time type selection via preprocessor macros to configure
 * precision and index types. This enables:
 * - Single codebase for float32/float64/float16
 * - Consistent type selection across the entire library
 * - Zero runtime overhead for type dispatching
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * TYPE: Real
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Primary floating-point type for numerical computation.
 *
 * CONFIGURATION:
 *     Selected at compile-time via one of:
 *     - SCL_USE_FLOAT32: Real = float (32-bit IEEE 754)
 *     - SCL_USE_FLOAT64: Real = double (64-bit IEEE 754)
 *     - SCL_USE_FLOAT16: Real = _Float16 (16-bit IEEE 754-2008)
 *
 * DTYPE METADATA:
 *     - DTYPE_CODE: Integer code (0=float32, 1=float64, 2=float16)
 *     - DTYPE_NAME: String name (const char*)
 *
 * PRECISION REQUIREMENTS:
 *     - float32: Standard single precision, good balance of speed/accuracy
 *     - float64: Double precision for high-accuracy requirements
 *     - float16: Half precision, requires GCC >= 12 or Clang >= 15
 *
 * USAGE GUIDELINES:
 *     - Use Real for all numerical computations
 *     - Do NOT hardcode float or double in library code
 *     - Precision is library-wide, cannot mix within single build
 *
 * PERFORMANCE:
 *     - float32: Fastest on most hardware, 2x throughput vs float64 on SIMD
 *     - float64: Slower but better numerical stability
 *     - float16: 4x throughput on modern GPUs, limited CPU support
 *
 * COMPATIBILITY:
 *     Exactly one of SCL_USE_FLOAT32, SCL_USE_FLOAT64, SCL_USE_FLOAT16
 *     must be defined, otherwise compilation error.
 * -------------------------------------------------------------------------- */
using Real = /* float | double | _Float16 */;

constexpr int DTYPE_CODE;         // 0, 1, or 2
constexpr const char* DTYPE_NAME; // "float32", "float64", or "float16"

/* -----------------------------------------------------------------------------
 * TYPE: Index
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Signed integer type for array indexing and dimensions.
 *
 * CONFIGURATION:
 *     Selected at compile-time via one of:
 *     - SCL_USE_INT16: Index = int16_t (16-bit signed)
 *     - SCL_USE_INT32: Index = int32_t (32-bit signed)
 *     - SCL_USE_INT64: Index = int64_t (64-bit signed)
 *
 * DTYPE METADATA:
 *     - INDEX_DTYPE_CODE: Integer code (0=int16, 1=int32, 2=int64)
 *     - INDEX_DTYPE_NAME: String name (const char*)
 *
 * RATIONALE FOR SIGNED:
 *     - Allows negative indices for reverse iteration
 *     - Simplifies loop bounds checking (i >= 0)
 *     - Compatible with BLAS/LAPACK conventions
 *     - Prevents unsigned underflow bugs in loops
 *
 * SIZE GUIDELINES:
 *     - int16: Matrices up to 32K x 32K, saves memory
 *     - int32: Standard choice, supports 2B x 2B matrices
 *     - int64: Very large matrices, increases memory footprint
 *
 * USAGE GUIDELINES:
 *     - Use Index for all array indexing, dimensions, loop counters
 *     - Do NOT use int or size_t for indexing
 *     - Use Size (unsigned) only for memory/byte sizes
 *
 * PERFORMANCE:
 *     Smaller indices reduce memory traffic and improve cache performance,
 *     especially for sparse matrix index arrays.
 *
 * COMPATIBILITY:
 *     Exactly one of SCL_USE_INT16, SCL_USE_INT32, SCL_USE_INT64 must be
 *     defined, otherwise compilation error.
 * -------------------------------------------------------------------------- */
using Index = /* int16_t | int32_t | int64_t */;

constexpr int INDEX_DTYPE_CODE;         // 0, 1, or 2
constexpr const char* INDEX_DTYPE_NAME; // "int16", "int32", or "int64"

/* -----------------------------------------------------------------------------
 * TYPE: Size
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Unsigned integer type for memory sizes and byte counts.
 *
 * DEFINITION:
 *     using Size = std::size_t;
 *
 * USAGE GUIDELINES:
 *     - Use for memory allocation sizes
 *     - Use for byte counts and buffer lengths
 *     - Do NOT use for array indexing (use Index instead)
 *     - Do NOT use for loop counters over array elements
 *
 * RATIONALE:
 *     - Matches C++ standard library convention (std::vector::size())
 *     - Guaranteed to represent any object size
 *     - Platform-dependent (32-bit on 32-bit systems, 64-bit on 64-bit)
 *
 * CONVERSION:
 *     When converting from Index to Size, ensure non-negative:
 *     Size n = static_cast<Size>(index);  // Only if index >= 0
 * -------------------------------------------------------------------------- */
using Size = std::size_t;

/* -----------------------------------------------------------------------------
 * TYPE: Byte
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Unsigned 8-bit integer for raw memory operations.
 *
 * DEFINITION:
 *     using Byte = std::uint8_t;
 *
 * USAGE GUIDELINES:
 *     - Use for raw memory buffers
 *     - Use for serialization/deserialization
 *     - Use for byte-level I/O operations
 *
 * TYPICAL USE CASES:
 *     - Memory-mapped file buffers
 *     - Binary data streams
 *     - Pointer arithmetic at byte level
 * -------------------------------------------------------------------------- */
using Byte = std::uint8_t;

/* -----------------------------------------------------------------------------
 * TYPE: Pointer
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Generic untyped pointer for discontiguous storage.
 *
 * DEFINITION:
 *     using Pointer = void*;
 *
 * USAGE GUIDELINES:
 *     - Use for type-erased pointers across C-ABI
 *     - Use for generic memory handles
 *     - Always cast to proper type before dereferencing
 *
 * WARNING:
 *     Void pointers bypass type safety. Use only at API boundaries where
 *     type erasure is necessary (e.g., Python bindings).
 * -------------------------------------------------------------------------- */
using Pointer = void*;

/* =============================================================================
 * SECTION 2: ARRAY VIEW
 * =============================================================================
 * Array<T> is a lightweight, non-owning view of contiguous memory. It serves
 * as the fundamental building block for all data structures in SCL.
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * STRUCT: Array<T>
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Lightweight, non-owning view of a contiguous 1D array.
 *
 * DESIGN PHILOSOPHY:
 *     - Non-owning: No destructor, no allocation, no deallocation
 *     - Zero-overhead: POD type with trivial copy (16 bytes on 64-bit)
 *     - Const-correct: Array<T> (mutable) vs Array<const T> (immutable)
 *     - Public members: Direct access for performance-critical code
 *     - Copyable: Shallow copy of view, not the underlying data
 *
 * TEMPLATE PARAMETERS:
 *     T - Element type (can be const-qualified)
 *
 * MEMBER TYPES:
 *     value_type - T (element type)
 *
 * PUBLIC MEMBERS:
 *     ptr - Pointer to first element (T*)
 *     len - Number of elements (Size)
 *
 * MEMORY LAYOUT:
 *     sizeof(Array<T>) = sizeof(T*) + sizeof(Size)
 *     Typically 16 bytes on 64-bit systems (8 + 8)
 *
 * OWNERSHIP:
 *     Array does NOT own the memory it points to. Lifetime of pointed-to
 *     memory must exceed the lifetime of the Array view.
 *
 * CONST-CORRECTNESS:
 *     Array<T>       - Mutable view (can modify elements)
 *     Array<const T> - Immutable view (read-only access)
 *     Conversion from Array<T> to Array<const T> is implicit.
 *
 * THREAD SAFETY:
 *     Array itself is trivially copyable (thread-safe).
 *     Concurrent access to underlying data requires external synchronization.
 *
 * PERFORMANCE:
 *     - All methods are force-inlined (zero function call overhead)
 *     - Public members enable compiler optimizations (auto-vectorization)
 *     - Trivially copyable (passed by value in registers)
 * -------------------------------------------------------------------------- */
template <typename T>
struct Array {
    using value_type = T;
    
    T* ptr;    // Pointer to first element
    Size len;  // Number of elements

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: Array() (default)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Default constructor, creates empty view.
     *
     * POSTCONDITIONS:
     *     ptr = nullptr
     *     len = 0
     *
     * MUTABILITY:
     *     CONST - initializes to empty state
     *
     * THREAD SAFETY:
     *     Safe - no shared state
     * ---------------------------------------------------------------------- */
    constexpr Array() noexcept;  // : ptr(nullptr), len(0)

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: Array(T* p, Size s)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Construct view from pointer and size.
     *
     * PARAMETERS:
     *     p [in] - Pointer to first element
     *     s [in] - Number of elements
     *
     * PRECONDITIONS:
     *     - If s > 0, p must point to valid array of at least s elements
     *     - If s == 0, p can be nullptr or any value (ignored)
     *
     * POSTCONDITIONS:
     *     ptr = p
     *     len = s
     *
     * MUTABILITY:
     *     CONST - creates view of existing data
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     *
     * THREAD SAFETY:
     *     Safe - no shared state during construction
     *
     * WARNING:
     *     Does not validate pointer or size. Caller responsible for ensuring
     *     validity.
     * ---------------------------------------------------------------------- */
    constexpr Array(T* p, Size s) noexcept;  // : ptr(p), len(s)

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: Array(const Array<U>&) (conversion)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Implicit conversion from Array<U> to Array<const U>.
     *
     * TEMPLATE CONSTRAINTS:
     *     - T must be const-qualified
     *     - U must be same as T without const
     *     - Enables: Array<int> -> Array<const int>
     *     - Disables: Array<const int> -> Array<int> (would remove const)
     *
     * POSTCONDITIONS:
     *     ptr = other.ptr
     *     len = other.len
     *
     * MUTABILITY:
     *     CONST - creates immutable view of mutable data
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     *
     * THREAD SAFETY:
     *     Safe
     * ---------------------------------------------------------------------- */
    template <typename U>
        requires (std::is_const_v<T> && 
                  std::is_same_v<std::remove_const_t<T>, U>)
    constexpr Array(const Array<U>& other) noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: operator[](Index i) const
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Access element at index i.
     *
     * PARAMETERS:
     *     i [in] - Index of element (0-based)
     *
     * PRECONDITIONS:
     *     0 <= i < len
     *
     * POSTCONDITIONS:
     *     Returns reference to element at index i
     *
     * MUTABILITY:
     *     CONST method, but returns mutable reference if T is non-const
     *
     * BOUNDS CHECKING:
     *     Debug builds (NDEBUG not defined): Assertion failure if out of bounds
     *     Release builds: No checking, undefined behavior if out of bounds
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     *
     * THREAD SAFETY:
     *     Safe for index operation
     *     Concurrent element access requires synchronization
     * ---------------------------------------------------------------------- */
    SCL_FORCE_INLINE constexpr T& operator[](Index i) const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: data() const
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get pointer to first element.
     *
     * POSTCONDITIONS:
     *     Returns ptr (may be nullptr if empty)
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     * ---------------------------------------------------------------------- */
    SCL_FORCE_INLINE constexpr T* data() const noexcept;  // return ptr;

    /* -------------------------------------------------------------------------
     * METHOD: size() const
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get number of elements.
     *
     * POSTCONDITIONS:
     *     Returns len
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     * ---------------------------------------------------------------------- */
    SCL_FORCE_INLINE constexpr Size size() const noexcept;  // return len;

    /* -------------------------------------------------------------------------
     * METHOD: empty() const
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Check if view is empty.
     *
     * POSTCONDITIONS:
     *     Returns true if len == 0, false otherwise
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     * ---------------------------------------------------------------------- */
    SCL_FORCE_INLINE constexpr bool empty() const noexcept;  // return len == 0;

    /* -------------------------------------------------------------------------
     * METHOD: begin() const
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get iterator to first element.
     *
     * POSTCONDITIONS:
     *     Returns ptr
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     * ---------------------------------------------------------------------- */
    SCL_FORCE_INLINE constexpr T* begin() const noexcept;  // return ptr;

    /* -------------------------------------------------------------------------
     * METHOD: end() const
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get iterator to one-past-last element.
     *
     * POSTCONDITIONS:
     *     Returns ptr + len
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     * ---------------------------------------------------------------------- */
    SCL_FORCE_INLINE constexpr T* end() const noexcept;  // return ptr + len;
};

/* =============================================================================
 * SECTION 3: ARRAYLIKE CONCEPT
 * =============================================================================
 * ArrayLike is a concept that defines the interface requirements for 1D
 * array-like containers. It enables generic algorithms to work with any
 * container that provides array-like access.
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * CONCEPT: ArrayLike<A>
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Concept for 1D array-like containers with random access.
 *
 * REQUIREMENTS:
 *     - value_type: Member type alias for element type
 *     - size(): Returns number of elements (convertible to Size)
 *     - operator[](Index): Random access to elements
 *     - begin(): Returns iterator to first element
 *     - end(): Returns iterator to one-past-last element
 *
 * NOT REQUIRED:
 *     - data(): Does NOT require contiguous storage
 *     - This allows support for std::deque and other non-contiguous containers
 *
 * SATISFIED BY:
 *     - scl::Array<T> (contiguous)
 *     - std::vector<T> (contiguous)
 *     - std::array<T, N> (contiguous)
 *     - std::deque<T> (non-contiguous, but random access)
 *     - std::span<T> (C++20, contiguous)
 *     - Custom containers with compatible interface
 *
 * USAGE:
 *     template <ArrayLike A>
 *     void process(const A& arr) {
 *         for (Index i = 0; i < arr.size(); ++i) {
 *             do_something(arr[i]);
 *         }
 *     }
 *
 * DESIGN RATIONALE:
 *     - Enables generic programming without virtual dispatch
 *     - Compile-time polymorphism (zero runtime overhead)
 *     - Compatible with standard library containers
 *     - Does NOT require contiguous storage (removed data() requirement)
 *     - Supports both contiguous and segmented data structures
 *
 * PERFORMANCE:
 *     Concept checks are resolved at compile-time, zero runtime cost.
 *     Enables inlining and optimization of generic algorithms.
 *     Note: Contiguous containers may enable better SIMD optimizations.
 *
 * THREAD SAFETY:
 *     Concept itself has no runtime semantics. Thread safety depends on
 *     the specific container implementation.
 *
 * CONTIGUITY CHECK:
 *     If contiguous storage is required for SIMD operations, use an
 *     additional concept or static_assert:
 *     static_assert(requires(A a) { a.data(); }, "Requires contiguous storage");
 * -------------------------------------------------------------------------- */
template <typename A>
concept ArrayLike = requires(const A& a, Index i) {
    typename A::value_type;
    { a.size() } -> std::convertible_to<Size>;
    { a[i] } -> std::convertible_to<const typename A::value_type&>;
    { a.begin() };
    { a.end() };
};

/* -----------------------------------------------------------------------------
 * STATIC ASSERTIONS
 * -----------------------------------------------------------------------------
 * Verify that Array<T> satisfies ArrayLike concept for common types.
 * These are compile-time checks that ensure API compatibility.
 * -------------------------------------------------------------------------- */
static_assert(ArrayLike<Array<Real>>);
static_assert(ArrayLike<Array<const Real>>);
static_assert(ArrayLike<Array<Index>>);

/* =============================================================================
 * SECTION 4: SPARSE MATRIX TAGS AND CONCEPTS
 * =============================================================================
 * Sparse matrix types are unified under a single template parameter IsCSR,
 * enabling compile-time selection between CSR and CSC formats without code
 * duplication.
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * STRUCT: TagSparse<IsCSR>
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Tag type for sparse matrix format selection.
 *
 * TEMPLATE PARAMETERS:
 *     IsCSR [bool] - true for CSR (row-major), false for CSC (column-major)
 *
 * STATIC MEMBERS:
 *     is_csr - constexpr bool, equals IsCSR
 *     is_csc - constexpr bool, equals !IsCSR
 *
 * PURPOSE:
 *     Enables compile-time dispatch and type traits for sparse matrices.
 * -------------------------------------------------------------------------- */
template <bool IsCSR>
struct TagSparse {
    static constexpr bool is_csr;  // = IsCSR
    static constexpr bool is_csc;  // = !IsCSR
};

/* -----------------------------------------------------------------------------
 * CONCEPT: CSRLike<M>
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Concept for CSR (Compressed Sparse Row) sparse matrices.
 *
 * REQUIREMENTS:
 *     - ValueType: Element type alias
 *     - Tag: Must be TagSparse<true>
 *     - is_csr: Static constexpr bool == true
 *     - rows(): Returns number of rows
 *     - cols(): Returns number of columns
 *     - nnz(): Returns total number of non-zeros
 *     - row_values(Index): Returns Array<ValueType> of values in row
 *     - row_indices(Index): Returns Array<Index> of column indices in row
 *     - row_length(Index): Returns number of non-zeros in row
 *
 * SATISFIED BY:
 *     - Sparse<T, true>
 *     - Custom CSR implementations with compatible interface
 *
 * USAGE:
 *     template <CSRLike M>
 *     void process_rows(const M& mat) {
 *         for (Index i = 0; i < mat.rows(); ++i) {
 *             auto vals = mat.row_values(i);
 *             // Process row values
 *         }
 *     }
 *
 * PERFORMANCE:
 *     Concept checks are compile-time only, zero runtime overhead.
 * -------------------------------------------------------------------------- */
template <typename M>
concept CSRLike = requires(const M& m, Index i) {
    typename M::ValueType;
    typename M::Tag;
    requires M::is_csr == true;
    { m.rows() } -> std::convertible_to<Index>;
    { m.cols() } -> std::convertible_to<Index>;
    { m.nnz() } -> std::convertible_to<Index>;
    { m.row_values(i) } -> std::convertible_to<Array<typename M::ValueType>>;
    { m.row_indices(i) } -> std::convertible_to<Array<Index>>;
    { m.row_length(i) } -> std::convertible_to<Index>;
};

/* -----------------------------------------------------------------------------
 * CONCEPT: CSCLike<M>
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Concept for CSC (Compressed Sparse Column) sparse matrices.
 *
 * REQUIREMENTS:
 *     - ValueType: Element type alias
 *     - Tag: Must be TagSparse<false>
 *     - is_csc: Static constexpr bool == true
 *     - rows(): Returns number of rows
 *     - cols(): Returns number of columns
 *     - nnz(): Returns total number of non-zeros
 *     - col_values(Index): Returns Array<ValueType> of values in column
 *     - col_indices(Index): Returns Array<Index> of row indices in column
 *     - col_length(Index): Returns number of non-zeros in column
 *
 * SATISFIED BY:
 *     - Sparse<T, false>
 *     - Custom CSC implementations with compatible interface
 *
 * USAGE:
 *     template <CSCLike M>
 *     void process_columns(const M& mat) {
 *         for (Index j = 0; j < mat.cols(); ++j) {
 *             auto vals = mat.col_values(j);
 *             // Process column values
 *         }
 *     }
 *
 * PERFORMANCE:
 *     Concept checks are compile-time only, zero runtime overhead.
 * -------------------------------------------------------------------------- */
template <typename M>
concept CSCLike = requires(const M& m, Index j) {
    typename M::ValueType;
    typename M::Tag;
    requires M::is_csc == true;
    { m.rows() } -> std::convertible_to<Index>;
    { m.cols() } -> std::convertible_to<Index>;
    { m.nnz() } -> std::convertible_to<Index>;
    { m.col_values(j) } -> std::convertible_to<Array<typename M::ValueType>>;
    { m.col_indices(j) } -> std::convertible_to<Array<Index>>;
    { m.col_length(j) } -> std::convertible_to<Index>;
};

/* -----------------------------------------------------------------------------
 * CONCEPT: SparseLike<M>
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Concept for any sparse matrix (CSR or CSC).
 *
 * DEFINITION:
 *     SparseLike<M> = CSRLike<M> || CSCLike<M>
 *
 * USAGE:
 *     template <SparseLike M>
 *     void generic_sparse_operation(const M& mat) {
 *         if constexpr (M::is_csr) {
 *             // CSR-specific path
 *         } else {
 *             // CSC-specific path
 *         }
 *     }
 *
 * DESIGN RATIONALE:
 *     Enables generic algorithms that work with both CSR and CSC formats,
 *     with compile-time dispatch for format-specific optimizations.
 * -------------------------------------------------------------------------- */
template <typename M>
concept SparseLike = CSRLike<M> || CSCLike<M>;

} // namespace scl

