// =============================================================================
// FILE: scl/mmap/integration/sparse_adapter.hpp
// BRIEF: Sparse matrix adapter implementation
// =============================================================================
#pragma once

#include "sparse_adapter.h"
#include "view.hpp"
#include "../cache/tiered.hpp"
#include "../configuration.hpp"
#include "scl/core/macros.hpp"

#include <algorithm>
#include <vector>

namespace scl::mmap::integration {

// =============================================================================
// MmapSparseAdapter::Impl
// =============================================================================

template <typename T, SparseFormat Format>
struct MmapSparseAdapter<T, Format>::Impl {
    SparseMatrixInfo info_;
    ViewBuilder view_builder;

    // Type-erased array references (stored as void* with size info)
    void* indptr_array_ = nullptr;
    void* indices_array_ = nullptr;
    void* data_array_ = nullptr;

    // Function pointers for array access
    std::function<ZeroCopyView<index_type>(std::size_t, std::size_t)> get_indptr_view;
    std::function<ZeroCopyView<index_type>(std::size_t, std::size_t)> get_indices_view;
    std::function<ZeroCopyView<T>(std::size_t, std::size_t)> get_data_view;
    std::function<void(std::span<const std::size_t>)> prefetch_indptr;
    std::function<void(std::span<const std::size_t>)> prefetch_indices;
    std::function<void(std::span<const std::size_t>)> prefetch_data;

    // Cached indptr values for fast row/col bounds lookup
    mutable std::vector<index_type> indptr_cache;
    mutable bool indptr_cached = false;

    Impl(cache::TieredCache& cache, SparseMatrixInfo info)
        : info_(info)
        , view_builder(cache, ViewConfig::streaming())
    {}

    void ensure_indptr_cached() const {
        if (indptr_cached) return;

        std::size_t ptr_count = (Format == SparseFormat::CSR)
            ? info_.rows + 1
            : info_.cols + 1;

        indptr_cache.resize(ptr_count);

        // Load indptr array
        auto view = get_indptr_view(0, ptr_count);
        if (view.valid()) {
            view.copy_to(indptr_cache.data());
        }

        indptr_cached = true;
    }

    index_type get_ptr(std::size_t idx) const {
        ensure_indptr_cached();
        return indptr_cache[idx];
    }
};

// =============================================================================
// MmapSparseAdapter Implementation
// =============================================================================

template <typename T, SparseFormat Format>
template <typename IndptrArray, typename IndicesArray, typename DataArray>
MmapSparseAdapter<T, Format>::MmapSparseAdapter(
    IndptrArray& indptr_array,
    IndicesArray& indices_array,
    DataArray& data_array,
    cache::TieredCache& cache,
    SparseMatrixInfo info
) : impl_(std::make_unique<Impl>(cache, info))
{
    // Store array references and create view functions
    impl_->indptr_array_ = &indptr_array;
    impl_->indices_array_ = &indices_array;
    impl_->data_array_ = &data_array;

    // Create view functions that capture array references
    impl_->get_indptr_view = [this, &indptr_array](std::size_t offset, std::size_t count) {
        return impl_->view_builder.build<index_type>(indptr_array, offset, count);
    };

    impl_->get_indices_view = [this, &indices_array](std::size_t offset, std::size_t count) {
        return impl_->view_builder.build<index_type>(indices_array, offset, count);
    };

    impl_->get_data_view = [this, &data_array](std::size_t offset, std::size_t count) {
        return impl_->view_builder.build<T>(data_array, offset, count);
    };

    // Create prefetch functions
    impl_->prefetch_indptr = [&indptr_array, &cache](std::span<const std::size_t> pages) {
        cache.prefetch(pages, &indptr_array.backend());
    };

    impl_->prefetch_indices = [&indices_array, &cache](std::span<const std::size_t> pages) {
        cache.prefetch(pages, &indices_array.backend());
    };

    impl_->prefetch_data = [&data_array, &cache](std::span<const std::size_t> pages) {
        cache.prefetch(pages, &data_array.backend());
    };
}

template <typename T, SparseFormat Format>
MmapSparseAdapter<T, Format>::~MmapSparseAdapter() = default;

template <typename T, SparseFormat Format>
MmapSparseAdapter<T, Format>::MmapSparseAdapter(MmapSparseAdapter&& other) noexcept = default;

template <typename T, SparseFormat Format>
MmapSparseAdapter<T, Format>&
MmapSparseAdapter<T, Format>::operator=(MmapSparseAdapter&& other) noexcept = default;

template <typename T, SparseFormat Format>
typename MmapSparseAdapter<T, Format>::size_type
MmapSparseAdapter<T, Format>::rows() const noexcept {
    return impl_->info_.rows;
}

template <typename T, SparseFormat Format>
typename MmapSparseAdapter<T, Format>::size_type
MmapSparseAdapter<T, Format>::cols() const noexcept {
    return impl_->info_.cols;
}

template <typename T, SparseFormat Format>
typename MmapSparseAdapter<T, Format>::size_type
MmapSparseAdapter<T, Format>::nnz() const noexcept {
    return impl_->info_.nnz;
}

template <typename T, SparseFormat Format>
SparseFormat MmapSparseAdapter<T, Format>::format() const noexcept {
    return Format;
}

template <typename T, SparseFormat Format>
const SparseMatrixInfo& MmapSparseAdapter<T, Format>::info() const noexcept {
    return impl_->info_;
}

// CSR methods
template <typename T, SparseFormat Format>
typename MmapSparseAdapter<T, Format>::index_type
MmapSparseAdapter<T, Format>::row_begin(size_type row) const {
    static_assert(Format == SparseFormat::CSR, "row_begin only for CSR");
    return impl_->get_ptr(row);
}

template <typename T, SparseFormat Format>
typename MmapSparseAdapter<T, Format>::index_type
MmapSparseAdapter<T, Format>::row_end(size_type row) const {
    static_assert(Format == SparseFormat::CSR, "row_end only for CSR");
    return impl_->get_ptr(row + 1);
}

template <typename T, SparseFormat Format>
typename MmapSparseAdapter<T, Format>::size_type
MmapSparseAdapter<T, Format>::row_nnz(size_type row) const {
    static_assert(Format == SparseFormat::CSR, "row_nnz only for CSR");
    return static_cast<size_type>(row_end(row) - row_begin(row));
}

template <typename T, SparseFormat Format>
ZeroCopyView<typename MmapSparseAdapter<T, Format>::index_type>
MmapSparseAdapter<T, Format>::row_indices(size_type row) const {
    static_assert(Format == SparseFormat::CSR, "row_indices only for CSR");
    index_type start = row_begin(row);
    index_type end = row_end(row);
    return impl_->get_indices_view(
        static_cast<std::size_t>(start),
        static_cast<std::size_t>(end - start)
    );
}

template <typename T, SparseFormat Format>
ZeroCopyView<T>
MmapSparseAdapter<T, Format>::row_values(size_type row) const {
    static_assert(Format == SparseFormat::CSR, "row_values only for CSR");
    index_type start = row_begin(row);
    index_type end = row_end(row);
    return impl_->get_data_view(
        static_cast<std::size_t>(start),
        static_cast<std::size_t>(end - start)
    );
}

// CSC methods
template <typename T, SparseFormat Format>
typename MmapSparseAdapter<T, Format>::index_type
MmapSparseAdapter<T, Format>::col_begin(size_type col) const {
    static_assert(Format == SparseFormat::CSC, "col_begin only for CSC");
    return impl_->get_ptr(col);
}

template <typename T, SparseFormat Format>
typename MmapSparseAdapter<T, Format>::index_type
MmapSparseAdapter<T, Format>::col_end(size_type col) const {
    static_assert(Format == SparseFormat::CSC, "col_end only for CSC");
    return impl_->get_ptr(col + 1);
}

template <typename T, SparseFormat Format>
typename MmapSparseAdapter<T, Format>::size_type
MmapSparseAdapter<T, Format>::col_nnz(size_type col) const {
    static_assert(Format == SparseFormat::CSC, "col_nnz only for CSC");
    return static_cast<size_type>(col_end(col) - col_begin(col));
}

template <typename T, SparseFormat Format>
ZeroCopyView<typename MmapSparseAdapter<T, Format>::index_type>
MmapSparseAdapter<T, Format>::col_indices(size_type col) const {
    static_assert(Format == SparseFormat::CSC, "col_indices only for CSC");
    index_type start = col_begin(col);
    index_type end = col_end(col);
    return impl_->get_indices_view(
        static_cast<std::size_t>(start),
        static_cast<std::size_t>(end - start)
    );
}

template <typename T, SparseFormat Format>
ZeroCopyView<T>
MmapSparseAdapter<T, Format>::col_values(size_type col) const {
    static_assert(Format == SparseFormat::CSC, "col_values only for CSC");
    index_type start = col_begin(col);
    index_type end = col_end(col);
    return impl_->get_data_view(
        static_cast<std::size_t>(start),
        static_cast<std::size_t>(end - start)
    );
}

// Full array views
template <typename T, SparseFormat Format>
ZeroCopyView<typename MmapSparseAdapter<T, Format>::index_type>
MmapSparseAdapter<T, Format>::indptr() const {
    std::size_t count = (Format == SparseFormat::CSR)
        ? impl_->info_.rows + 1
        : impl_->info_.cols + 1;
    return impl_->get_indptr_view(0, count);
}

template <typename T, SparseFormat Format>
ZeroCopyView<typename MmapSparseAdapter<T, Format>::index_type>
MmapSparseAdapter<T, Format>::indices() const {
    return impl_->get_indices_view(0, impl_->info_.nnz);
}

template <typename T, SparseFormat Format>
ZeroCopyView<T>
MmapSparseAdapter<T, Format>::data() const {
    return impl_->get_data_view(0, impl_->info_.nnz);
}

template <typename T, SparseFormat Format>
T MmapSparseAdapter<T, Format>::get_value(size_type row, size_type col) const {
    if constexpr (Format == SparseFormat::CSR) {
        auto indices_view = row_indices(row);
        auto values_view = row_values(row);

        if (!indices_view.valid() || !values_view.valid()) {
            return T{};
        }

        // Binary search if sorted
        if (impl_->info_.sorted) {
            auto it = std::lower_bound(
                indices_view.begin(), indices_view.end(),
                static_cast<index_type>(col)
            );
            if (it != indices_view.end() && *it == static_cast<index_type>(col)) {
                std::size_t idx = std::distance(indices_view.begin(), it);
                return values_view[idx];
            }
        } else {
            // Linear search
            for (std::size_t i = 0; i < indices_view.size(); ++i) {
                if (indices_view[i] == static_cast<index_type>(col)) {
                    return values_view[i];
                }
            }
        }
    } else {
        auto indices_view = col_indices(col);
        auto values_view = col_values(col);

        if (!indices_view.valid() || !values_view.valid()) {
            return T{};
        }

        if (impl_->info_.sorted) {
            auto it = std::lower_bound(
                indices_view.begin(), indices_view.end(),
                static_cast<index_type>(row)
            );
            if (it != indices_view.end() && *it == static_cast<index_type>(row)) {
                std::size_t idx = std::distance(indices_view.begin(), it);
                return values_view[idx];
            }
        } else {
            for (std::size_t i = 0; i < indices_view.size(); ++i) {
                if (indices_view[i] == static_cast<index_type>(row)) {
                    return values_view[i];
                }
            }
        }
    }

    return T{};
}

template <typename T, SparseFormat Format>
void MmapSparseAdapter<T, Format>::prefetch_row(size_type row) const {
    static_assert(Format == SparseFormat::CSR, "prefetch_row only for CSR");

    index_type start = row_begin(row);
    index_type end = row_end(row);

    if (start >= end) return;

    // Calculate page indices
    std::size_t start_byte = static_cast<std::size_t>(start) * sizeof(index_type);
    std::size_t end_byte = static_cast<std::size_t>(end) * sizeof(index_type);

    std::size_t start_page = start_byte / kPageSize;
    std::size_t end_page = end_byte / kPageSize;

    std::vector<std::size_t> pages;
    for (std::size_t p = start_page; p <= end_page; ++p) {
        pages.push_back(p);
    }

    impl_->prefetch_indices(pages);
    impl_->prefetch_data(pages);
}

template <typename T, SparseFormat Format>
void MmapSparseAdapter<T, Format>::prefetch_col(size_type col) const {
    static_assert(Format == SparseFormat::CSC, "prefetch_col only for CSC");

    index_type start = col_begin(col);
    index_type end = col_end(col);

    if (start >= end) return;

    std::size_t start_byte = static_cast<std::size_t>(start) * sizeof(index_type);
    std::size_t end_byte = static_cast<std::size_t>(end) * sizeof(index_type);

    std::size_t start_page = start_byte / kPageSize;
    std::size_t end_page = end_byte / kPageSize;

    std::vector<std::size_t> pages;
    for (std::size_t p = start_page; p <= end_page; ++p) {
        pages.push_back(p);
    }

    impl_->prefetch_indices(pages);
    impl_->prefetch_data(pages);
}

template <typename T, SparseFormat Format>
void MmapSparseAdapter<T, Format>::prefetch_rows(std::span<const size_type> rows) const {
    static_assert(Format == SparseFormat::CSR, "prefetch_rows only for CSR");

    std::vector<std::size_t> pages;

    for (size_type row : rows) {
        if (row >= impl_->info_.rows) continue;

        index_type start = row_begin(row);
        index_type end = row_end(row);

        if (start >= end) continue;

        std::size_t start_page = (static_cast<std::size_t>(start) * sizeof(index_type)) / kPageSize;
        std::size_t end_page = (static_cast<std::size_t>(end) * sizeof(index_type)) / kPageSize;

        for (std::size_t p = start_page; p <= end_page; ++p) {
            pages.push_back(p);
        }
    }

    // Remove duplicates
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());

    impl_->prefetch_indices(pages);
    impl_->prefetch_data(pages);
}

template <typename T, SparseFormat Format>
void MmapSparseAdapter<T, Format>::prefetch_cols(std::span<const size_type> cols) const {
    static_assert(Format == SparseFormat::CSC, "prefetch_cols only for CSC");

    std::vector<std::size_t> pages;

    for (size_type col : cols) {
        if (col >= impl_->info_.cols) continue;

        index_type start = col_begin(col);
        index_type end = col_end(col);

        if (start >= end) continue;

        std::size_t start_page = (static_cast<std::size_t>(start) * sizeof(index_type)) / kPageSize;
        std::size_t end_page = (static_cast<std::size_t>(end) * sizeof(index_type)) / kPageSize;

        for (std::size_t p = start_page; p <= end_page; ++p) {
            pages.push_back(p);
        }
    }

    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());

    impl_->prefetch_indices(pages);
    impl_->prefetch_data(pages);
}

// =============================================================================
// Free Functions
// =============================================================================

inline const char* sparse_format_name(SparseFormat format) noexcept {
    switch (format) {
        case SparseFormat::CSR: return "CSR";
        case SparseFormat::CSC: return "CSC";
        case SparseFormat::COO: return "COO";
        default:                return "Unknown";
    }
}

} // namespace scl::mmap::integration
