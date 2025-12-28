// =============================================================================
// FILE: scl/mmap/integration/view.hpp
// BRIEF: Zero-copy view implementation
// =============================================================================
#pragma once

#include "view.h"
#include "../cache/tiered.hpp"
#include "../configuration.hpp"
#include "scl/core/macros.hpp"

#include <vector>
#include <stdexcept>
#include <cstring>

namespace scl::mmap::integration {

// =============================================================================
// ViewConfig Implementation
// =============================================================================

constexpr ViewConfig ViewConfig::pinned() noexcept {
    return ViewConfig{
        .pin_pages = true,
        .prefetch_adjacent = false,
        .prefetch_count = 0,
        .allow_partial = true
    };
}

constexpr ViewConfig ViewConfig::streaming() noexcept {
    return ViewConfig{
        .pin_pages = true,
        .prefetch_adjacent = true,
        .prefetch_count = 8,
        .allow_partial = true
    };
}

constexpr ViewConfig ViewConfig::random_access() noexcept {
    return ViewConfig{
        .pin_pages = true,
        .prefetch_adjacent = false,
        .prefetch_count = 1,
        .allow_partial = true
    };
}

// =============================================================================
// ZeroCopyView::Impl
// =============================================================================

template <typename T>
struct ZeroCopyView<T>::Impl {
    std::vector<cache::PageHandle> handles;
    T* data_ptr = nullptr;
    std::size_t element_count = 0;
    std::size_t byte_offset = 0;  // Offset within first page
    bool contiguous = true;
    bool dirty = false;

    Impl() = default;

    Impl(const Impl& other)
        : handles(other.handles)  // Shared handles
        , data_ptr(other.data_ptr)
        , element_count(other.element_count)
        , byte_offset(other.byte_offset)
        , contiguous(other.contiguous)
        , dirty(other.dirty)
    {}

    ~Impl() {
        release_handles();
    }

    void release_handles() {
        for (auto& handle : handles) {
            if (dirty && handle.valid()) {
                handle.mark_dirty();
            }
            handle.release();
        }
        handles.clear();
        data_ptr = nullptr;
        element_count = 0;
    }

    bool is_valid() const noexcept {
        return data_ptr != nullptr && element_count > 0;
    }
};

// =============================================================================
// ZeroCopyView Implementation
// =============================================================================

template <typename T>
ZeroCopyView<T>::ZeroCopyView() noexcept
    : impl_(std::make_shared<Impl>()) {}

template <typename T>
ZeroCopyView<T>::ZeroCopyView(const ZeroCopyView& other)
    : impl_(other.impl_ ? std::make_shared<Impl>(*other.impl_) : nullptr) {}

template <typename T>
ZeroCopyView<T>& ZeroCopyView<T>::operator=(const ZeroCopyView& other) {
    if (this != &other) {
        impl_ = other.impl_ ? std::make_shared<Impl>(*other.impl_) : nullptr;
    }
    return *this;
}

template <typename T>
ZeroCopyView<T>::ZeroCopyView(ZeroCopyView&& other) noexcept = default;

template <typename T>
ZeroCopyView<T>& ZeroCopyView<T>::operator=(ZeroCopyView&& other) noexcept = default;

template <typename T>
ZeroCopyView<T>::~ZeroCopyView() = default;

template <typename T>
ZeroCopyView<T>::ZeroCopyView(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

template <typename T>
const T* ZeroCopyView<T>::data() const noexcept {
    return impl_ ? impl_->data_ptr : nullptr;
}

template <typename T>
T* ZeroCopyView<T>::data() noexcept {
    return impl_ ? impl_->data_ptr : nullptr;
}

template <typename T>
typename ZeroCopyView<T>::size_type ZeroCopyView<T>::size() const noexcept {
    return impl_ ? impl_->element_count : 0;
}

template <typename T>
typename ZeroCopyView<T>::size_type ZeroCopyView<T>::size_bytes() const noexcept {
    return size() * sizeof(T);
}

template <typename T>
bool ZeroCopyView<T>::empty() const noexcept {
    return size() == 0;
}

template <typename T>
bool ZeroCopyView<T>::valid() const noexcept {
    return impl_ && impl_->is_valid();
}

template <typename T>
ZeroCopyView<T>::operator bool() const noexcept {
    return valid();
}

template <typename T>
bool ZeroCopyView<T>::is_contiguous() const noexcept {
    return impl_ ? impl_->contiguous : false;
}

template <typename T>
typename ZeroCopyView<T>::const_reference
ZeroCopyView<T>::operator[](size_type idx) const noexcept {
    return impl_->data_ptr[idx];
}

template <typename T>
typename ZeroCopyView<T>::reference
ZeroCopyView<T>::operator[](size_type idx) noexcept {
    return impl_->data_ptr[idx];
}

template <typename T>
typename ZeroCopyView<T>::const_reference
ZeroCopyView<T>::at(size_type idx) const {
    if (idx >= size()) {
        throw std::out_of_range("ZeroCopyView::at: index out of range");
    }
    return impl_->data_ptr[idx];
}

template <typename T>
typename ZeroCopyView<T>::reference
ZeroCopyView<T>::at(size_type idx) {
    if (idx >= size()) {
        throw std::out_of_range("ZeroCopyView::at: index out of range");
    }
    return impl_->data_ptr[idx];
}

template <typename T>
typename ZeroCopyView<T>::const_reference ZeroCopyView<T>::front() const noexcept {
    return impl_->data_ptr[0];
}

template <typename T>
typename ZeroCopyView<T>::reference ZeroCopyView<T>::front() noexcept {
    return impl_->data_ptr[0];
}

template <typename T>
typename ZeroCopyView<T>::const_reference ZeroCopyView<T>::back() const noexcept {
    return impl_->data_ptr[impl_->element_count - 1];
}

template <typename T>
typename ZeroCopyView<T>::reference ZeroCopyView<T>::back() noexcept {
    return impl_->data_ptr[impl_->element_count - 1];
}

template <typename T>
typename ZeroCopyView<T>::const_iterator ZeroCopyView<T>::begin() const noexcept {
    return impl_ ? impl_->data_ptr : nullptr;
}

template <typename T>
typename ZeroCopyView<T>::iterator ZeroCopyView<T>::begin() noexcept {
    return impl_ ? impl_->data_ptr : nullptr;
}

template <typename T>
typename ZeroCopyView<T>::const_iterator ZeroCopyView<T>::end() const noexcept {
    return impl_ ? impl_->data_ptr + impl_->element_count : nullptr;
}

template <typename T>
typename ZeroCopyView<T>::iterator ZeroCopyView<T>::end() noexcept {
    return impl_ ? impl_->data_ptr + impl_->element_count : nullptr;
}

template <typename T>
typename ZeroCopyView<T>::const_iterator ZeroCopyView<T>::cbegin() const noexcept {
    return begin();
}

template <typename T>
typename ZeroCopyView<T>::const_iterator ZeroCopyView<T>::cend() const noexcept {
    return end();
}

template <typename T>
std::span<const T> ZeroCopyView<T>::as_span() const noexcept {
    return std::span<const T>(data(), size());
}

template <typename T>
std::span<T> ZeroCopyView<T>::as_span() noexcept {
    return std::span<T>(data(), size());
}

template <typename T>
ZeroCopyView<T> ZeroCopyView<T>::subview(size_type offset, size_type count) const {
    if (!impl_ || offset > impl_->element_count) {
        return ZeroCopyView<T>();
    }

    if (count == npos || offset + count > impl_->element_count) {
        count = impl_->element_count - offset;
    }

    auto sub_impl = std::make_shared<Impl>();
    sub_impl->handles = impl_->handles;  // Share handles
    sub_impl->data_ptr = impl_->data_ptr + offset;
    sub_impl->element_count = count;
    sub_impl->byte_offset = impl_->byte_offset + offset * sizeof(T);
    sub_impl->contiguous = impl_->contiguous;
    sub_impl->dirty = impl_->dirty;

    return ZeroCopyView<T>(sub_impl);
}

template <typename T>
typename ZeroCopyView<T>::size_type ZeroCopyView<T>::copy_to(T* dest) const {
    if (!valid() || !dest) return 0;

    if (is_contiguous()) {
        std::memcpy(dest, impl_->data_ptr, size_bytes());
    } else {
        // Element-by-element copy for non-contiguous
        for (size_type i = 0; i < size(); ++i) {
            dest[i] = (*this)[i];
        }
    }
    return size();
}

template <typename T>
typename ZeroCopyView<T>::size_type ZeroCopyView<T>::copy_to(std::span<T> dest) const {
    size_type to_copy = std::min(size(), dest.size());
    if (is_contiguous()) {
        std::memcpy(dest.data(), impl_->data_ptr, to_copy * sizeof(T));
    } else {
        for (size_type i = 0; i < to_copy; ++i) {
            dest[i] = (*this)[i];
        }
    }
    return to_copy;
}

template <typename T>
typename ZeroCopyView<T>::size_type ZeroCopyView<T>::page_count() const noexcept {
    return impl_ ? impl_->handles.size() : 0;
}

template <typename T>
void ZeroCopyView<T>::mark_dirty() noexcept {
    if (impl_) {
        impl_->dirty = true;
        for (auto& handle : impl_->handles) {
            handle.mark_dirty();
        }
    }
}

template <typename T>
void ZeroCopyView<T>::release() noexcept {
    if (impl_) {
        impl_->release_handles();
    }
}

// =============================================================================
// ViewBuilder::Impl
// =============================================================================

struct ViewBuilder::Impl {
    cache::TieredCache& cache;
    ViewConfig config;

    Impl(cache::TieredCache& c, ViewConfig cfg)
        : cache(c), config(cfg) {}
};

// =============================================================================
// ViewBuilder Implementation
// =============================================================================

ViewBuilder::ViewBuilder(cache::TieredCache& cache, ViewConfig config)
    : impl_(std::make_unique<Impl>(cache, config)) {}

ViewBuilder::~ViewBuilder() = default;

template <typename T, typename ArrayT>
ZeroCopyView<T> ViewBuilder::build(
    ArrayT& array,
    std::size_t offset,
    std::size_t count
) {
    if (count == 0 || offset >= array.size()) {
        return ZeroCopyView<T>();
    }

    count = std::min(count, array.size() - offset);

    // Calculate byte range
    std::size_t byte_start = offset * sizeof(T);
    std::size_t byte_end = byte_start + count * sizeof(T);

    // Calculate page range
    std::size_t first_page = byte_start / kPageSize;
    std::size_t last_page = (byte_end - 1) / kPageSize;
    std::size_t num_pages = last_page - first_page + 1;

    auto view_impl = std::make_shared<typename ZeroCopyView<T>::Impl>();
    view_impl->handles.reserve(num_pages);
    view_impl->element_count = count;
    view_impl->byte_offset = byte_start % kPageSize;

    // Load all required pages
    for (std::size_t page_idx = first_page; page_idx <= last_page; ++page_idx) {
        auto handle = impl_->cache.get(page_idx, &array.backend());
        if (!handle.valid()) {
            return ZeroCopyView<T>();  // Failed to load page
        }
        view_impl->handles.push_back(std::move(handle));
    }

    // Set data pointer (offset into first page)
    view_impl->data_ptr = reinterpret_cast<T*>(
        view_impl->handles[0].data() + view_impl->byte_offset
    );

    // Check contiguity
    view_impl->contiguous = (num_pages == 1) ||
        (view_impl->byte_offset == 0 && byte_end % kPageSize == 0);

    // Prefetch adjacent pages if configured
    if (impl_->config.prefetch_adjacent && impl_->config.prefetch_count > 0) {
        std::vector<std::size_t> prefetch_pages;
        for (std::size_t i = 1; i <= impl_->config.prefetch_count; ++i) {
            if (last_page + i < array.num_pages()) {
                prefetch_pages.push_back(last_page + i);
            }
        }
        if (!prefetch_pages.empty()) {
            impl_->cache.prefetch(prefetch_pages, &array.backend());
        }
    }

    return ZeroCopyView<T>(view_impl);
}

template <typename T, typename ArrayT>
ZeroCopyView<T> ViewBuilder::build_all(ArrayT& array) {
    return build<T>(array, 0, array.size());
}

const ViewConfig& ViewBuilder::config() const noexcept {
    return impl_->config;
}

// =============================================================================
// Free Functions
// =============================================================================

template <typename T, typename ArrayT>
ZeroCopyView<T> make_view(
    cache::TieredCache& cache,
    ArrayT& array,
    std::size_t offset,
    std::size_t count
) {
    ViewBuilder builder(cache);
    return builder.build<T>(array, offset, count);
}

template <typename T, typename ArrayT>
ZeroCopyView<T> make_view_all(
    cache::TieredCache& cache,
    ArrayT& array
) {
    ViewBuilder builder(cache);
    return builder.build_all<T>(array);
}

} // namespace scl::mmap::integration
