// =============================================================================
// FILE: scl/core/sparse.h
// BRIEF: API reference for sparse matrix with discontiguous storage
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.h"
#include "scl/core/macros.hpp"

namespace scl {

/* =============================================================================
 * STRUCT: Sparse<T, IsCSR>
 * =============================================================================
 * SUMMARY:
 *     Sparse matrix with discontiguous storage using pointer arrays.
 *
 * DESIGN PHILOSOPHY:
 *     Unlike contiguous sparse formats (standard CSR/CSC with data, indices,
 *     indptr arrays), Sparse uses pointer arrays where each row/column can
 *     be stored in a separate allocation. This enables:
 *     - Wrapping heterogeneous data sources
 *     - Lazy loading of rows/columns on demand
 *     - Memory-mapped sparse matrices with row/column granularity
 *     - Integration with external data (Python/NumPy arrays)
 *
 * TEMPLATE PARAMETERS:
 *     T     - Element type (typically Real)
 *     IsCSR - Format selector (true = CSR row-major, false = CSC column-major)
 *
 * MEMBER TYPES:
 *     ValueType - T (element type)
 *     Tag       - TagSparse<IsCSR>
 *
 * STATIC MEMBERS:
 *     is_csr - constexpr bool == IsCSR
 *     is_csc - constexpr bool == !IsCSR
 *
 * PUBLIC MEMBERS:
 *     data_ptrs    - Pointer array to value arrays [primary_dim]
 *     indices_ptrs - Pointer array to index arrays [primary_dim]
 *     lengths      - Array of row/column lengths [primary_dim]
 *     rows_        - Number of rows
 *     cols_        - Number of columns
 *     nnz_         - Total number of non-zeros
 *
 * MEMORY LAYOUT (CSR example with 3 rows):
 *     data_ptrs    = [ptr_to_row0_vals, ptr_to_row1_vals, ptr_to_row2_vals]
 *     indices_ptrs = [ptr_to_row0_cols, ptr_to_row1_cols, ptr_to_row2_cols]
 *     lengths      = [len0, len1, len2]
 *
 * OWNERSHIP:
 *     Sparse is a NON-OWNING view. It does not manage the lifetime of:
 *     - The pointer arrays (data_ptrs, indices_ptrs, lengths)
 *     - The actual data arrays they point to
 *
 *     Use factory method new_registered() to create matrices with
 *     HandlerRegistry-managed metadata arrays.
 *
 * PERFORMANCE CHARACTERISTICS:
 *     - Row/column access: O(1) but may have cache misses
 *     - Less cache-friendly than contiguous formats for sequential access
 *     - More flexible for heterogeneous or lazy-loaded data
 *     - Memory: (2*primary_dim + 1) pointers + nnz elements
 *
 * THREAD SAFETY:
 *     Read operations are safe for concurrent access.
 *     Write operations require external synchronization.
 *
 * SATISFIES CONCEPTS:
 *     - CSRLike (when IsCSR == true)
 *     - CSCLike (when IsCSR == false)
 *     - SparseLike (always)
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
struct Sparse {
    using ValueType = T;
    using Tag = TagSparse<IsCSR>;

    static constexpr bool is_csr;  // = IsCSR
    static constexpr bool is_csc;  // = !IsCSR

    Pointer* data_ptrs;      // Pointers to value arrays [primary_dim]
    Pointer* indices_ptrs;   // Pointers to index arrays [primary_dim]
    Index* lengths;          // Length of each row/column [primary_dim]
    Index rows_;             // Number of rows
    Index cols_;             // Number of columns
    Index nnz_;              // Total non-zeros

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: Sparse() (default)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Default constructor, creates empty/null matrix.
     *
     * POSTCONDITIONS:
     *     - All pointers are nullptr
     *     - All dimensions are 0
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     * ---------------------------------------------------------------------- */
    constexpr Sparse() noexcept;

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: Sparse(dp, ip, len, r, c, n)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Construct from pre-allocated pointer arrays (non-owning view).
     *
     * PARAMETERS:
     *     dp  [in] - Pointer to data pointer array [primary_dim]
     *     ip  [in] - Pointer to indices pointer array [primary_dim]
     *     len [in] - Pointer to lengths array [primary_dim]
     *     r   [in] - Number of rows (must be >= 0)
     *     c   [in] - Number of columns (must be >= 0)
     *     n   [in] - Total number of non-zeros (must be >= 0)
     *
     * PRECONDITIONS:
     *     - r >= 0, c >= 0, n >= 0
     *     - If r,c > 0: dp, ip, len must point to valid arrays
     *     - primary_dim = IsCSR ? r : c
     *
     * POSTCONDITIONS:
     *     - Matrix view is created pointing to provided arrays
     *     - No memory is allocated or copied
     *
     * MUTABILITY:
     *     CONST - creates view of existing data
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     *
     * ASSERTIONS:
     *     Debug builds check that r, c, n are non-negative
     *
     * WARNING:
     *     Caller must ensure underlying memory outlives this Sparse object.
     * ---------------------------------------------------------------------- */
    constexpr Sparse(
        Pointer* dp, 
        Pointer* ip, 
        Index* len,
        Index r, 
        Index c, 
        Index n
    ) noexcept;

    /* -------------------------------------------------------------------------
     * DESTRUCTOR: ~Sparse()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Destructor is no-op (non-owning view).
     *
     * POSTCONDITIONS:
     *     No memory is freed.
     * ---------------------------------------------------------------------- */
    ~Sparse();

    /* -------------------------------------------------------------------------
     * COPY/MOVE OPERATIONS
     * -------------------------------------------------------------------------
     * All defaulted - just copies pointers (shallow copy of view).
     * ---------------------------------------------------------------------- */
    Sparse(const Sparse&);
    Sparse& operator=(const Sparse&);
    Sparse(Sparse&&) noexcept;
    Sparse& operator=(Sparse&&) noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: valid() const
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Check if matrix view is valid (has non-null pointers).
     *
     * POSTCONDITIONS:
     *     Returns true if all pointer arrays are non-null, false otherwise.
     *
     * COMPLEXITY:
     *     Time:  O(1)
     * ---------------------------------------------------------------------- */
    SCL_NODISCARD SCL_FORCE_INLINE bool valid() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: operator bool() const
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Boolean conversion, equivalent to valid().
     *
     * USAGE:
     *     if (mat) { ... }  // Check if matrix is valid
     * ---------------------------------------------------------------------- */
    SCL_NODISCARD SCL_FORCE_INLINE explicit operator bool() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: rows() const
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get number of rows.
     *
     * COMPLEXITY:
     *     Time:  O(1)
     * ---------------------------------------------------------------------- */
    SCL_NODISCARD SCL_FORCE_INLINE Index rows() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: cols() const
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get number of columns.
     *
     * COMPLEXITY:
     *     Time:  O(1)
     * ---------------------------------------------------------------------- */
    SCL_NODISCARD SCL_FORCE_INLINE Index cols() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: nnz() const
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get total number of non-zero elements.
     *
     * COMPLEXITY:
     *     Time:  O(1)
     * ---------------------------------------------------------------------- */
    SCL_NODISCARD SCL_FORCE_INLINE Index nnz() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: primary_dim() const
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get primary dimension size.
     *
     * POSTCONDITIONS:
     *     Returns rows_ if IsCSR, cols_ otherwise.
     *
     * COMPLEXITY:
     *     Time:  O(1)
     * ---------------------------------------------------------------------- */
    SCL_NODISCARD SCL_FORCE_INLINE Index primary_dim() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: secondary_dim() const
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get secondary dimension size.
     *
     * POSTCONDITIONS:
     *     Returns cols_ if IsCSR, rows_ otherwise.
     *
     * COMPLEXITY:
     *     Time:  O(1)
     * ---------------------------------------------------------------------- */
    SCL_NODISCARD SCL_FORCE_INLINE Index secondary_dim() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: empty() const
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Check if matrix is empty (no elements).
     *
     * POSTCONDITIONS:
     *     Returns true if rows == 0 OR cols == 0 OR nnz == 0.
     *
     * COMPLEXITY:
     *     Time:  O(1)
     * ---------------------------------------------------------------------- */
    SCL_NODISCARD SCL_FORCE_INLINE bool empty() const noexcept;

    /* -------------------------------------------------------------------------
     * CSR INTERFACE (enabled when IsCSR == true)
     * ------------------------------------------------------------------------- */

    /* -------------------------------------------------------------------------
     * METHOD: row_values(Index i) const [CSR only]
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get array view of values in row i.
     *
     * PARAMETERS:
     *     i [in] - Row index (must be in [0, rows_))
     *
     * PRECONDITIONS:
     *     - 0 <= i < rows_
     *     - data_ptrs != nullptr
     *
     * POSTCONDITIONS:
     *     Returns Array<T> view of values in row i.
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     *
     * THREAD SAFETY:
     *     Safe for concurrent reads
     *
     * ASSERTIONS:
     *     Debug builds check bounds and null pointers
     * ---------------------------------------------------------------------- */
    SCL_NODISCARD SCL_FORCE_INLINE Array<T> row_values(Index i) const noexcept 
        requires (IsCSR);

    /* -------------------------------------------------------------------------
     * METHOD: row_indices(Index i) const [CSR only]
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get array view of column indices in row i.
     *
     * PARAMETERS:
     *     i [in] - Row index (must be in [0, rows_))
     *
     * PRECONDITIONS:
     *     - 0 <= i < rows_
     *     - indices_ptrs != nullptr
     *
     * POSTCONDITIONS:
     *     Returns Array<Index> view of column indices in row i.
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     *
     * ASSERTIONS:
     *     Debug builds check bounds and null pointers
     * ---------------------------------------------------------------------- */
    SCL_NODISCARD SCL_FORCE_INLINE Array<Index> row_indices(Index i) const noexcept 
        requires (IsCSR);

    /* -------------------------------------------------------------------------
     * METHOD: row_length(Index i) const [CSR only]
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get number of non-zeros in row i.
     *
     * PARAMETERS:
     *     i [in] - Row index (must be in [0, rows_))
     *
     * PRECONDITIONS:
     *     - 0 <= i < rows_
     *     - lengths != nullptr
     *
     * POSTCONDITIONS:
     *     Returns lengths[i].
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     *
     * ASSERTIONS:
     *     Debug builds check bounds and null pointers
     * ---------------------------------------------------------------------- */
    SCL_NODISCARD SCL_FORCE_INLINE Index row_length(Index i) const noexcept 
        requires (IsCSR);

    /* -------------------------------------------------------------------------
     * CSC INTERFACE (enabled when IsCSR == false)
     * ------------------------------------------------------------------------- */

    /* -------------------------------------------------------------------------
     * METHOD: col_values(Index j) const [CSC only]
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get array view of values in column j.
     *
     * PARAMETERS:
     *     j [in] - Column index (must be in [0, cols_))
     *
     * PRECONDITIONS:
     *     - 0 <= j < cols_
     *     - data_ptrs != nullptr
     *
     * POSTCONDITIONS:
     *     Returns Array<T> view of values in column j.
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     *
     * ASSERTIONS:
     *     Debug builds check bounds and null pointers
     * ---------------------------------------------------------------------- */
    SCL_NODISCARD SCL_FORCE_INLINE Array<T> col_values(Index j) const noexcept 
        requires (!IsCSR);

    /* -------------------------------------------------------------------------
     * METHOD: col_indices(Index j) const [CSC only]
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get array view of row indices in column j.
     *
     * PARAMETERS:
     *     j [in] - Column index (must be in [0, cols_))
     *
     * PRECONDITIONS:
     *     - 0 <= j < cols_
     *     - indices_ptrs != nullptr
     *
     * POSTCONDITIONS:
     *     Returns Array<Index> view of row indices in column j.
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     *
     * ASSERTIONS:
     *     Debug builds check bounds and null pointers
     * ---------------------------------------------------------------------- */
    SCL_NODISCARD SCL_FORCE_INLINE Array<Index> col_indices(Index j) const noexcept 
        requires (!IsCSR);

    /* -------------------------------------------------------------------------
     * METHOD: col_length(Index j) const [CSC only]
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get number of non-zeros in column j.
     *
     * PARAMETERS:
     *     j [in] - Column index (must be in [0, cols_))
     *
     * PRECONDITIONS:
     *     - 0 <= j < cols_
     *     - lengths != nullptr
     *
     * POSTCONDITIONS:
     *     Returns lengths[j].
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     *
     * ASSERTIONS:
     *     Debug builds check bounds and null pointers
     * ---------------------------------------------------------------------- */
    SCL_NODISCARD SCL_FORCE_INLINE Index col_length(Index j) const noexcept 
        requires (!IsCSR);

    /* -------------------------------------------------------------------------
     * STATIC METHOD: new_registered(rows, cols, total_nnz)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create Sparse with HandlerRegistry-managed metadata arrays.
     *
     * PARAMETERS:
     *     rows      [in] - Number of rows (must be >= 0)
     *     cols      [in] - Number of columns (must be >= 0)
     *     total_nnz [in] - Total number of non-zeros (must be >= 0)
     *
     * PRECONDITIONS:
     *     - rows >= 0, cols >= 0, total_nnz >= 0
     *
     * POSTCONDITIONS:
     *     - Returns Sparse view with allocated metadata arrays
     *     - Returns empty Sparse on allocation failure
     *     - Metadata arrays are registered with HandlerRegistry
     *     - Actual row/column data arrays must be allocated separately
     *
     * MUTABILITY:
     *     ALLOCATES - creates metadata arrays (data_ptrs, indices_ptrs, lengths)
     *
     * CLEANUP:
     *     Call unregister_metadata() before transferring ownership to Python.
     *
     * COMPLEXITY:
     *     Time:  O(primary_dim) for zero-initialization
     *     Space: O(primary_dim) where primary_dim = IsCSR ? rows : cols
     *
     * THREAD SAFETY:
     *     Safe - allocation is synchronized by HandlerRegistry
     *
     * RETURNS:
     *     Sparse object with valid metadata arrays, or empty Sparse on failure.
     *
     * WARNING:
     *     This only allocates the metadata arrays (pointers and lengths).
     *     The actual data arrays for each row/column must be allocated
     *     separately and assigned to data_ptrs[i] and indices_ptrs[i].
     * ---------------------------------------------------------------------- */
    [[nodiscard]] static Sparse new_registered(
        Index rows, 
        Index cols, 
        Index total_nnz
    );

    /* -------------------------------------------------------------------------
     * METHOD: unregister_metadata()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Unregister metadata arrays from HandlerRegistry.
     *
     * PRECONDITIONS:
     *     None - safe to call even if not valid
     *
     * POSTCONDITIONS:
     *     - data_ptrs, indices_ptrs, lengths are unregistered
     *     - All three pointers are set to nullptr
     *     - Matrix becomes invalid
     *
     * MUTABILITY:
     *     INPLACE - modifies internal state, invalidates the object
     *
     * USAGE:
     *     Call this before transferring ownership of metadata arrays to
     *     Python or other external code.
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     *
     * THREAD SAFETY:
     *     Safe - uses synchronized HandlerRegistry operations
     *
     * WARNING:
     *     This only unregisters the metadata arrays (pointers and lengths).
     *     The actual data arrays they point to must be unregistered separately
     *     if they were also registered.
     * ---------------------------------------------------------------------- */
    void unregister_metadata();
};

/* -----------------------------------------------------------------------------
 * TYPE ALIASES
 * -------------------------------------------------------------------------- */

/* CSR matrix with Real values (row-major sparse) */
using CSR = Sparse<Real, true>;

/* CSC matrix with Real values (column-major sparse) */
using CSC = Sparse<Real, false>;

/* -----------------------------------------------------------------------------
 * STATIC ASSERTIONS
 * -----------------------------------------------------------------------------
 * Verify that Sparse satisfies the sparse matrix concepts.
 * -------------------------------------------------------------------------- */
static_assert(CSRLike<CSR>);
static_assert(CSCLike<CSC>);
static_assert(SparseLike<CSR>);
static_assert(SparseLike<CSC>);

} // namespace scl

