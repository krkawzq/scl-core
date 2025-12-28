// =============================================================================
// FILE: scl/mmap/integration/sparse_adapter.h
// BRIEF: API reference for sparse matrix adapter with mmap backend
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "view.h"
#include "../cache/tiered.h"
#include "../configuration.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <memory>

namespace scl::mmap::integration {

/* =============================================================================
 * STRUCT: SparseFormat
 * =============================================================================
 * SUMMARY:
 *     Sparse matrix storage format.
 *
 * VALUES:
 *     CSR - Compressed Sparse Row
 *     CSC - Compressed Sparse Column
 *     COO - Coordinate format
 * -------------------------------------------------------------------------- */
enum class SparseFormat : std::uint8_t {
    CSR,
    CSC,
    COO
};

/* =============================================================================
 * STRUCT: SparseMatrixInfo
 * =============================================================================
 * SUMMARY:
 *     Metadata about a sparse matrix.
 *
 * FIELDS:
 *     rows     - Number of rows
 *     cols     - Number of columns
 *     nnz      - Number of non-zero elements
 *     format   - Storage format (CSR/CSC/COO)
 *     sorted   - True if indices are sorted within each row/column
 *     has_dups - True if duplicate indices may exist
 * -------------------------------------------------------------------------- */
struct SparseMatrixInfo {
    std::size_t rows;
    std::size_t cols;
    std::size_t nnz;
    SparseFormat format;
    bool sorted;
    bool has_dups;
};

/* =============================================================================
 * CLASS: MmapSparseAdapter
 * =============================================================================
 * SUMMARY:
 *     Adapter for accessing sparse matrices stored in mmap arrays.
 *
 * DESIGN PURPOSE:
 *     Provides CSR/CSC-compatible interface to mmap-backed sparse data:
 *     - Lazy loading of row/column slices
 *     - Zero-copy access via ZeroCopyView
 *     - Compatible with scl::kernel sparse concepts
 *
 * CONCEPT COMPATIBILITY:
 *     Satisfies CSRLike and CSCLike concepts from scl::kernel:
 *     - rows() / cols() / nnz()
 *     - indptr() / indices() / data()
 *     - row_values() / row_indices() for CSR
 *     - col_values() / col_indices() for CSC
 *
 * MEMORY LAYOUT (CSR):
 *     indptr:  [rows+1] - Row pointers
 *     indices: [nnz]    - Column indices
 *     data:    [nnz]    - Values
 *
 * MEMORY LAYOUT (CSC):
 *     indptr:  [cols+1] - Column pointers
 *     indices: [nnz]    - Row indices
 *     data:    [nnz]    - Values
 *
 * THREAD SAFETY:
 *     NOT thread-safe. Create separate adapters per thread.
 * -------------------------------------------------------------------------- */
template <typename T, SparseFormat Format = SparseFormat::CSR>
class MmapSparseAdapter {
public:
    using value_type = T;
    using index_type = std::int64_t;
    using size_type = std::size_t;

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: MmapSparseAdapter
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create adapter from mmap arrays.
     *
     * PARAMETERS:
     *     indptr_array  [in] - Row/column pointer array
     *     indices_array [in] - Column/row indices array
     *     data_array    [in] - Values array
     *     cache         [in] - Tiered cache for page access
     *     info          [in] - Matrix metadata
     *
     * PRECONDITIONS:
     *     - indptr_array.size() == info.rows + 1 (CSR) or info.cols + 1 (CSC)
     *     - indices_array.size() == info.nnz
     *     - data_array.size() == info.nnz
     *
     * POSTCONDITIONS:
     *     Adapter ready for row/column access.
     * ---------------------------------------------------------------------- */
    template <typename IndptrArray, typename IndicesArray, typename DataArray>
    MmapSparseAdapter(
        IndptrArray& indptr_array,         // Row/column pointers
        IndicesArray& indices_array,       // Column/row indices
        DataArray& data_array,             // Values
        cache::TieredCache& cache,         // Page cache
        SparseMatrixInfo info              // Matrix metadata
    );

    ~MmapSparseAdapter();

    MmapSparseAdapter(const MmapSparseAdapter&) = delete;
    MmapSparseAdapter& operator=(const MmapSparseAdapter&) = delete;
    MmapSparseAdapter(MmapSparseAdapter&&) noexcept;
    MmapSparseAdapter& operator=(MmapSparseAdapter&&) noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: rows
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Number of rows in matrix.
     * ---------------------------------------------------------------------- */
    size_type rows() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: cols
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Number of columns in matrix.
     * ---------------------------------------------------------------------- */
    size_type cols() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: nnz
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Number of non-zero elements.
     * ---------------------------------------------------------------------- */
    size_type nnz() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: format
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Storage format (CSR/CSC).
     * ---------------------------------------------------------------------- */
    SparseFormat format() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: info
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Reference to matrix metadata.
     * ---------------------------------------------------------------------- */
    const SparseMatrixInfo& info() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: row_begin / row_end (CSR only)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get range of non-zeros for a row.
     *
     * PARAMETERS:
     *     row [in] - Row index
     *
     * RETURNS:
     *     Start/end index in indices/data arrays.
     *
     * PRECONDITIONS:
     *     row < rows()
     * ---------------------------------------------------------------------- */
    index_type row_begin(size_type row) const;
    index_type row_end(size_type row) const;

    /* -------------------------------------------------------------------------
     * METHOD: row_nnz (CSR only)
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Number of non-zeros in row.
     * ---------------------------------------------------------------------- */
    size_type row_nnz(size_type row) const;

    /* -------------------------------------------------------------------------
     * METHOD: row_indices (CSR only)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get column indices for a row.
     *
     * PARAMETERS:
     *     row [in] - Row index
     *
     * RETURNS:
     *     ZeroCopyView of column indices.
     *
     * LIFETIME:
     *     View valid until adapter destroyed or row changes.
     * ---------------------------------------------------------------------- */
    ZeroCopyView<index_type> row_indices(size_type row) const;

    /* -------------------------------------------------------------------------
     * METHOD: row_values (CSR only)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get values for a row.
     *
     * PARAMETERS:
     *     row [in] - Row index
     *
     * RETURNS:
     *     ZeroCopyView of row values.
     * ---------------------------------------------------------------------- */
    ZeroCopyView<T> row_values(size_type row) const;

    /* -------------------------------------------------------------------------
     * METHOD: col_begin / col_end (CSC only)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get range of non-zeros for a column.
     *
     * PARAMETERS:
     *     col [in] - Column index
     *
     * RETURNS:
     *     Start/end index in indices/data arrays.
     * ---------------------------------------------------------------------- */
    index_type col_begin(size_type col) const;
    index_type col_end(size_type col) const;

    /* -------------------------------------------------------------------------
     * METHOD: col_nnz (CSC only)
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Number of non-zeros in column.
     * ---------------------------------------------------------------------- */
    size_type col_nnz(size_type col) const;

    /* -------------------------------------------------------------------------
     * METHOD: col_indices (CSC only)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get row indices for a column.
     *
     * PARAMETERS:
     *     col [in] - Column index
     *
     * RETURNS:
     *     ZeroCopyView of row indices.
     * ---------------------------------------------------------------------- */
    ZeroCopyView<index_type> col_indices(size_type col) const;

    /* -------------------------------------------------------------------------
     * METHOD: col_values (CSC only)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get values for a column.
     *
     * PARAMETERS:
     *     col [in] - Column index
     *
     * RETURNS:
     *     ZeroCopyView of column values.
     * ---------------------------------------------------------------------- */
    ZeroCopyView<T> col_values(size_type col) const;

    /* -------------------------------------------------------------------------
     * METHOD: indptr
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get view of entire indptr array.
     *
     * RETURNS:
     *     ZeroCopyView of row/column pointers.
     *
     * NOTE:
     *     Loading entire array may require many pages.
     * ---------------------------------------------------------------------- */
    ZeroCopyView<index_type> indptr() const;

    /* -------------------------------------------------------------------------
     * METHOD: indices
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get view of entire indices array.
     *
     * RETURNS:
     *     ZeroCopyView of column/row indices.
     * ---------------------------------------------------------------------- */
    ZeroCopyView<index_type> indices() const;

    /* -------------------------------------------------------------------------
     * METHOD: data
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get view of entire data array.
     *
     * RETURNS:
     *     ZeroCopyView of values.
     * ---------------------------------------------------------------------- */
    ZeroCopyView<T> data() const;

    /* -------------------------------------------------------------------------
     * METHOD: get_value
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get value at specific row, column (slow for CSR/CSC).
     *
     * PARAMETERS:
     *     row [in] - Row index
     *     col [in] - Column index
     *
     * RETURNS:
     *     Value at (row, col), or 0 if not present.
     *
     * COMPLEXITY:
     *     O(nnz_in_row) for CSR, O(nnz_in_col) for CSC.
     * ---------------------------------------------------------------------- */
    T get_value(size_type row, size_type col) const;

    /* -------------------------------------------------------------------------
     * METHOD: prefetch_row (CSR only)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Hint that row will be accessed soon.
     *
     * PARAMETERS:
     *     row [in] - Row index to prefetch
     * ---------------------------------------------------------------------- */
    void prefetch_row(size_type row) const;

    /* -------------------------------------------------------------------------
     * METHOD: prefetch_col (CSC only)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Hint that column will be accessed soon.
     *
     * PARAMETERS:
     *     col [in] - Column index to prefetch
     * ---------------------------------------------------------------------- */
    void prefetch_col(size_type col) const;

    /* -------------------------------------------------------------------------
     * METHOD: prefetch_rows (CSR only)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Prefetch multiple rows.
     *
     * PARAMETERS:
     *     rows [in] - Row indices to prefetch
     * ---------------------------------------------------------------------- */
    void prefetch_rows(std::span<const size_type> rows) const;

    /* -------------------------------------------------------------------------
     * METHOD: prefetch_cols (CSC only)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Prefetch multiple columns.
     *
     * PARAMETERS:
     *     cols [in] - Column indices to prefetch
     * ---------------------------------------------------------------------- */
    void prefetch_cols(std::span<const size_type> cols) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Convenience type aliases
template <typename T>
using MmapCSR = MmapSparseAdapter<T, SparseFormat::CSR>;

template <typename T>
using MmapCSC = MmapSparseAdapter<T, SparseFormat::CSC>;

/* =============================================================================
 * FUNCTION: sparse_format_name
 * =============================================================================
 * SUMMARY:
 *     Convert SparseFormat enum to human-readable string.
 *
 * PARAMETERS:
 *     format [in] - Sparse format enum value
 *
 * RETURNS:
 *     Null-terminated C string (never nullptr).
 * -------------------------------------------------------------------------- */
const char* sparse_format_name(SparseFormat format) noexcept;

/* =============================================================================
 * CONCEPT: MmapCSRLike
 * =============================================================================
 * SUMMARY:
 *     Concept for types that behave like CSR sparse matrices.
 * -------------------------------------------------------------------------- */
template <typename M>
concept MmapCSRLike = requires(const M& m, typename M::size_type i) {
    { m.rows() } -> std::convertible_to<typename M::size_type>;
    { m.cols() } -> std::convertible_to<typename M::size_type>;
    { m.nnz() } -> std::convertible_to<typename M::size_type>;
    { m.row_begin(i) } -> std::convertible_to<typename M::index_type>;
    { m.row_end(i) } -> std::convertible_to<typename M::index_type>;
    { m.row_indices(i) };
    { m.row_values(i) };
};

/* =============================================================================
 * CONCEPT: MmapCSCLike
 * =============================================================================
 * SUMMARY:
 *     Concept for types that behave like CSC sparse matrices.
 * -------------------------------------------------------------------------- */
template <typename M>
concept MmapCSCLike = requires(const M& m, typename M::size_type i) {
    { m.rows() } -> std::convertible_to<typename M::size_type>;
    { m.cols() } -> std::convertible_to<typename M::size_type>;
    { m.nnz() } -> std::convertible_to<typename M::size_type>;
    { m.col_begin(i) } -> std::convertible_to<typename M::index_type>;
    { m.col_end(i) } -> std::convertible_to<typename M::index_type>;
    { m.col_indices(i) };
    { m.col_values(i) };
};

} // namespace scl::mmap::integration
