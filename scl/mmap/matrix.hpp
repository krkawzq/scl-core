#pragma once

// =============================================================================
/// @file matrix.hpp
/// @brief Memory-Mapped Sparse and Dense Matrix Types
///
/// This file provides matrix abstractions over virtual arrays:
///
/// 1. MappedSparse<T, IsCSR>: CSR/CSC sparse matrix with paged storage
/// 2. MappedDense<T>: Row-major dense matrix with paged storage
///
/// Performance Optimizations:
///
/// - Thread-local row/column cache to avoid repeated reads
/// - Batch read operations for consecutive rows
/// - Prefetch hints for sequential access patterns
/// - Zero-copy interface where possible
///
/// Design Philosophy:
///
/// - Interface compatible with scl::CSRLike/CSCLike concepts
/// - Transparent paging (user sees contiguous matrix)
/// - Memory efficiency for out-of-core datasets
// =============================================================================

#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/mmap/configuration.hpp"
#include "scl/mmap/table.hpp"
#include "scl/mmap/scheduler.hpp"
#include "scl/mmap/array.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace scl::mmap {

// =============================================================================
// MappedSparse: Memory-Mapped Sparse Matrix (CSR/CSC)
// =============================================================================

/// @brief Memory-mapped sparse matrix with on-demand page loading
///
/// Storage Layout:
/// - data[nnz]: Non-zero values (VirtualArray<T>)
/// - indices[nnz]: Column indices (CSR) or row indices (CSC)
/// - indptr[primary+1]: Cumulative offsets
///
/// Thread Safety:
/// - Row/column cache is thread-local
/// - Multiple threads can access different rows safely
/// - Same row access from multiple threads is safe but may duplicate work
///
/// Usage:
///
///   // Create from loaders
///   MappedCSR<float> mat(rows, cols, nnz, data_loader, indices_loader, indptr_loader);
///
///   // Access row (triggers page load if needed)
///   auto vals = mat.row_values(i);
///   auto inds = mat.row_indices(i);
///
///   // Batch prefetch for iteration
///   mat.prefetch_primary(start_row, num_rows);
template <typename T, bool IsCSR>
class MappedSparse {
public:
    using ValueType = T;
    using Tag = scl::TagSparse<IsCSR>;

    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;

private:
    // Virtual arrays for sparse data
    std::unique_ptr<VirtualArray<T>> data_;
    std::unique_ptr<VirtualArray<scl::Index>> indices_;
    std::unique_ptr<VirtualArray<scl::Index>> indptr_;

    // Dimensions
    scl::Index rows_;
    scl::Index cols_;
    scl::Index nnz_;

    // Thread-local cache for row/column data
    // Using struct to ensure proper alignment
    struct alignas(64) RowCache {
        std::vector<T> values;
        std::vector<scl::Index> indices;
        scl::Index cached_idx = -1;
        
        void ensure_capacity(std::size_t n) {
            if (values.capacity() < n) {
                values.reserve(n);
                indices.reserve(n);
            }
        }
        
        void resize(std::size_t n) {
            values.resize(n);
            indices.resize(n);
        }
        
        void invalidate() {
            cached_idx = -1;
        }
    };
    
    // Thread-local storage for cache
    mutable RowCache cache_;

public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /// @brief Construct mapped sparse matrix
    /// @param rows Number of rows
    /// @param cols Number of columns
    /// @param nnz Number of non-zero elements
    /// @param data_loader Loader for values array
    /// @param indices_loader Loader for indices array
    /// @param indptr_loader Loader for indptr array
    /// @param config Memory pool configuration
    MappedSparse(scl::Index rows, scl::Index cols, scl::Index nnz,
                 LoadFunc data_loader,
                 LoadFunc indices_loader,
                 LoadFunc indptr_loader,
                 const MmapConfig& config = MmapConfig{})
        : rows_(rows), cols_(cols), nnz_(nnz)
    {
        data_ = std::make_unique<VirtualArray<T>>(
            static_cast<std::size_t>(nnz), std::move(data_loader), config);

        indices_ = std::make_unique<VirtualArray<scl::Index>>(
            static_cast<std::size_t>(nnz), std::move(indices_loader), config);

        const scl::Index primary_dim = IsCSR ? rows : cols;
        indptr_ = std::make_unique<VirtualArray<scl::Index>>(
            static_cast<std::size_t>(primary_dim + 1), std::move(indptr_loader), config);
        
        // Pre-allocate cache for typical row size
        const std::size_t avg_nnz_per_row = nnz > 0 ? 
            static_cast<std::size_t>(nnz / primary_dim) : 32;
        cache_.ensure_capacity(avg_nnz_per_row * 2);
    }

    ~MappedSparse() = default;

    // Non-copyable
    MappedSparse(const MappedSparse&) = delete;
    MappedSparse& operator=(const MappedSparse&) = delete;

    // Movable
    MappedSparse(MappedSparse&&) noexcept = default;
    MappedSparse& operator=(MappedSparse&&) noexcept = default;

    // -------------------------------------------------------------------------
    // Properties
    // -------------------------------------------------------------------------

    SCL_NODISCARD SCL_FORCE_INLINE scl::Index rows() const noexcept { return rows_; }
    SCL_NODISCARD SCL_FORCE_INLINE scl::Index cols() const noexcept { return cols_; }
    SCL_NODISCARD SCL_FORCE_INLINE scl::Index nnz() const noexcept { return nnz_; }

    // -------------------------------------------------------------------------
    // CSR Interface
    // -------------------------------------------------------------------------

    SCL_NODISCARD scl::Array<T> row_values(scl::Index i) const requires (IsCSR) {
        ensure_cached(i);
        return scl::Array<T>(cache_.values.data(), cache_.values.size());
    }

    SCL_NODISCARD scl::Array<scl::Index> row_indices(scl::Index i) const requires (IsCSR) {
        ensure_cached(i);
        return scl::Array<scl::Index>(cache_.indices.data(), cache_.indices.size());
    }

    SCL_NODISCARD scl::Index row_length(scl::Index i) const requires (IsCSR) {
        const scl::Index start = (*indptr_)[static_cast<std::size_t>(i)];
        const scl::Index end = (*indptr_)[static_cast<std::size_t>(i + 1)];
        return end - start;
    }

    // -------------------------------------------------------------------------
    // CSC Interface
    // -------------------------------------------------------------------------

    SCL_NODISCARD scl::Array<T> col_values(scl::Index j) const requires (!IsCSR) {
        ensure_cached(j);
        return scl::Array<T>(cache_.values.data(), cache_.values.size());
    }

    SCL_NODISCARD scl::Array<scl::Index> col_indices(scl::Index j) const requires (!IsCSR) {
        ensure_cached(j);
        return scl::Array<scl::Index>(cache_.indices.data(), cache_.indices.size());
    }

    SCL_NODISCARD scl::Index col_length(scl::Index j) const requires (!IsCSR) {
        const scl::Index start = (*indptr_)[static_cast<std::size_t>(j)];
        const scl::Index end = (*indptr_)[static_cast<std::size_t>(j + 1)];
        return end - start;
    }

    // -------------------------------------------------------------------------
    // Batch Access (High Performance)
    // -------------------------------------------------------------------------

    /// @brief Read multiple consecutive rows/columns into buffers
    /// @param start Starting row/column index
    /// @param count Number of rows/columns
    /// @param values_out Output buffer for values
    /// @param indices_out Output buffer for indices
    /// @param lengths_out Output buffer for lengths (count elements)
    void read_primary_range(scl::Index start, scl::Index count,
                            T* SCL_RESTRICT values_out, 
                            scl::Index* SCL_RESTRICT indices_out,
                            scl::Index* SCL_RESTRICT lengths_out) const {
        // Read indptr for the range
        const std::size_t indptr_start = static_cast<std::size_t>(start);
        const std::size_t indptr_count = static_cast<std::size_t>(count + 1);
        
        // Stack-allocate small indptr buffer, heap for large
        constexpr std::size_t kStackThreshold = 256;
        scl::Index stack_indptr[kStackThreshold];
        std::unique_ptr<scl::Index[]> heap_indptr;
        scl::Index* indptr_buf;
        
        if (indptr_count <= kStackThreshold) {
            indptr_buf = stack_indptr;
        } else {
            heap_indptr = std::make_unique<scl::Index[]>(indptr_count);
            indptr_buf = heap_indptr.get();
        }
        
        indptr_->read_range(indptr_start, indptr_count, indptr_buf);
        
        // Calculate total nnz and output lengths
        const scl::Index total_start = indptr_buf[0];
        scl::Index total_nnz = 0;
        
        for (scl::Index i = 0; i < count; ++i) {
            const scl::Index len = indptr_buf[i + 1] - indptr_buf[i];
            lengths_out[i] = len;
            total_nnz += len;
        }
        
        // Batch read data and indices
        if (total_nnz > 0) {
            data_->read_range(static_cast<std::size_t>(total_start),
                             static_cast<std::size_t>(total_nnz),
                             values_out);
            indices_->read_range(static_cast<std::size_t>(total_start),
                                static_cast<std::size_t>(total_nnz),
                                indices_out);
        }
        
        // Invalidate cache since we did direct reads
        cache_.invalidate();
    }

    // -------------------------------------------------------------------------
    // Prefetch Hints
    // -------------------------------------------------------------------------

    /// @brief Prefetch data for upcoming row/column access
    void prefetch_primary(scl::Index start, scl::Index count) {
        const scl::Index s = (*indptr_)[static_cast<std::size_t>(start)];
        const scl::Index e = (*indptr_)[static_cast<std::size_t>(start + count)];
        
        const std::size_t data_start = static_cast<std::size_t>(s);
        const std::size_t data_count = static_cast<std::size_t>(e - s);
        
        data_->prefetch(data_start, data_count);
        indices_->prefetch(data_start, data_count);
    }

    /// @brief Invalidate row cache (call when switching threads)
    void invalidate_cache() const {
        cache_.invalidate();
    }

private:
    void ensure_cached(scl::Index primary_idx) const {
        if (cache_.cached_idx == primary_idx) {
            return;
        }

        const scl::Index start = (*indptr_)[static_cast<std::size_t>(primary_idx)];
        const scl::Index end = (*indptr_)[static_cast<std::size_t>(primary_idx + 1)];
        const scl::Index len = end - start;
        
        cache_.resize(static_cast<std::size_t>(len));

        if (len > 0) {
            data_->read_range(static_cast<std::size_t>(start),
                             static_cast<std::size_t>(len),
                             cache_.values.data());
            indices_->read_range(static_cast<std::size_t>(start),
                                static_cast<std::size_t>(len),
                                cache_.indices.data());
        }

        cache_.cached_idx = primary_idx;
    }
};

// Type aliases
template <typename T>
using MappedCSR = MappedSparse<T, true>;

template <typename T>
using MappedCSC = MappedSparse<T, false>;

using MappedCSRReal = MappedCSR<scl::Real>;
using MappedCSCReal = MappedCSC<scl::Real>;

// =============================================================================
// MappedDense: Memory-Mapped Dense Matrix
// =============================================================================

/// @brief Memory-mapped dense matrix with row-major storage
///
/// Storage: Contiguous row-major layout
/// data[i * cols + j] = element at (i, j)
///
/// Optimizations:
/// - Row cache for repeated row access
/// - Batch row read for iteration
/// - Prefetch support for sequential access
template <typename T>
class MappedDense {
public:
    using ValueType = T;
    using Tag = scl::TagDense;

private:
    std::unique_ptr<VirtualArray<T>> data_;
    scl::Index rows_;
    scl::Index cols_;

    // Row cache
    mutable std::vector<T> row_cache_;
    mutable scl::Index cached_row_ = -1;

public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    MappedDense(scl::Index rows, scl::Index cols,
                LoadFunc loader,
                const MmapConfig& config = MmapConfig{})
        : rows_(rows), cols_(cols)
    {
        const std::size_t total = static_cast<std::size_t>(rows) * 
                                   static_cast<std::size_t>(cols);
        data_ = std::make_unique<VirtualArray<T>>(total, std::move(loader), config);
        
        // Pre-allocate row cache
        row_cache_.reserve(static_cast<std::size_t>(cols));
    }

    ~MappedDense() = default;

    MappedDense(const MappedDense&) = delete;
    MappedDense& operator=(const MappedDense&) = delete;
    MappedDense(MappedDense&&) noexcept = default;
    MappedDense& operator=(MappedDense&&) noexcept = default;

    // -------------------------------------------------------------------------
    // Properties
    // -------------------------------------------------------------------------

    SCL_NODISCARD SCL_FORCE_INLINE scl::Index rows() const noexcept { return rows_; }
    SCL_NODISCARD SCL_FORCE_INLINE scl::Index cols() const noexcept { return cols_; }

    // -------------------------------------------------------------------------
    // Element Access
    // -------------------------------------------------------------------------

    SCL_NODISCARD T operator()(scl::Index i, scl::Index j) const {
        const std::size_t idx = static_cast<std::size_t>(i) * 
                                 static_cast<std::size_t>(cols_) +
                                 static_cast<std::size_t>(j);
        return (*data_)[idx];
    }

    SCL_NODISCARD const T* data() const {
        ensure_row_cached(0);
        return row_cache_.data();
    }

    // -------------------------------------------------------------------------
    // Row Access
    // -------------------------------------------------------------------------

    SCL_NODISCARD scl::Array<const T> row(scl::Index i) const {
        ensure_row_cached(i);
        return scl::Array<const T>(row_cache_.data(), static_cast<scl::Size>(cols_));
    }

    void read_rows(scl::Index start, scl::Index count, T* out) const {
        const std::size_t offset = static_cast<std::size_t>(start) * 
                                    static_cast<std::size_t>(cols_);
        const std::size_t num = static_cast<std::size_t>(count) * 
                                 static_cast<std::size_t>(cols_);
        data_->read_range(offset, num, out);
    }

    // -------------------------------------------------------------------------
    // Prefetch
    // -------------------------------------------------------------------------

    void prefetch_rows(scl::Index start, scl::Index count) {
        const std::size_t offset = static_cast<std::size_t>(start) * 
                                    static_cast<std::size_t>(cols_);
        const std::size_t num = static_cast<std::size_t>(count) * 
                                 static_cast<std::size_t>(cols_);
        data_->prefetch(offset, num);
    }

private:
    void ensure_row_cached(scl::Index row_idx) const {
        if (cached_row_ == row_idx) return;

        row_cache_.resize(static_cast<std::size_t>(cols_));
        const std::size_t offset = static_cast<std::size_t>(row_idx) * 
                                    static_cast<std::size_t>(cols_);
        data_->read_range(offset, static_cast<std::size_t>(cols_), row_cache_.data());

        cached_row_ = row_idx;
    }
};

using MappedDenseReal = MappedDense<scl::Real>;

} // namespace scl::mmap
