// =============================================================================
// FILE: scl/mmap/backend/backend.hpp
// BRIEF: Storage backend CRTP base class implementation
// =============================================================================
#pragma once

#include "backend.h"
#include "../configuration.hpp"

#include <atomic>
#include <cassert>

namespace scl::mmap::backend {

// =============================================================================
// BackendCapabilities Implementation
// =============================================================================

constexpr BackendCapabilities BackendCapabilities::local_file() noexcept {
    return BackendCapabilities{
        .supports_random_access = true,
        .supports_concurrent_io = true,
        .supports_prefetch_hint = true,
        .supports_writeback = true,
        .is_compressed = false,
        .requires_decompression = false,
        .optimal_io_size = kPageSize,
        .max_concurrent_ops = 16,
        .typical_latency = std::chrono::microseconds(100)
    };
}

constexpr BackendCapabilities BackendCapabilities::compressed() noexcept {
    return BackendCapabilities{
        .supports_random_access = true,
        .supports_concurrent_io = true,
        .supports_prefetch_hint = false,
        .supports_writeback = false,
        .is_compressed = true,
        .requires_decompression = true,
        .optimal_io_size = kPageSize,
        .max_concurrent_ops = 8,
        .typical_latency = std::chrono::microseconds(500)
    };
}

constexpr BackendCapabilities BackendCapabilities::network() noexcept {
    return BackendCapabilities{
        .supports_random_access = true,
        .supports_concurrent_io = true,
        .supports_prefetch_hint = true,
        .supports_writeback = false,
        .is_compressed = false,
        .requires_decompression = false,
        .optimal_io_size = kPageSize * 4,
        .max_concurrent_ops = 32,
        .typical_latency = std::chrono::milliseconds(50)
    };
}

constexpr BackendCapabilities BackendCapabilities::memory() noexcept {
    return BackendCapabilities{
        .supports_random_access = true,
        .supports_concurrent_io = true,
        .supports_prefetch_hint = false,
        .supports_writeback = true,
        .is_compressed = false,
        .requires_decompression = false,
        .optimal_io_size = kPageSize,
        .max_concurrent_ops = 64,
        .typical_latency = std::chrono::nanoseconds(100)
    };
}

// =============================================================================
// IORequest Implementation
// =============================================================================

constexpr IORequest::IORequest(
    std::size_t page_idx,
    std::byte* dest,
    int priority,
    void* user_data
) noexcept
    : page_idx(page_idx)
    , dest(dest)
    , priority(priority)
    , user_data(user_data)
{}

// =============================================================================
// IOResult Implementation
// =============================================================================

constexpr bool IOResult::success() const noexcept {
    return error == 0;
}

// =============================================================================
// StorageBackend<Derived> Implementation
// =============================================================================

template <typename Derived>
Derived& StorageBackend<Derived>::derived() noexcept {
    return static_cast<Derived&>(*this);
}

template <typename Derived>
const Derived& StorageBackend<Derived>::derived() const noexcept {
    return static_cast<const Derived&>(*this);
}

template <typename Derived>
BackendCapabilities StorageBackend<Derived>::capabilities() const noexcept {
    return derived().capabilities_impl();
}

template <typename Derived>
std::size_t StorageBackend<Derived>::load_page(
    std::size_t page_idx,
    std::byte* dest
) {
    assert(page_idx < num_pages() && "page_idx out of bounds");
    assert(dest != nullptr && "dest is null");
    return derived().load_page_impl(page_idx, dest);
}

template <typename Derived>
std::size_t StorageBackend<Derived>::write_page(
    std::size_t page_idx,
    const std::byte* src
) {
    assert(page_idx < num_pages() && "page_idx out of bounds");
    assert(src != nullptr && "src is null");
    assert(capabilities().supports_writeback && "backend does not support writeback");
    return derived().write_page_impl(page_idx, src);
}

template <typename Derived>
std::size_t StorageBackend<Derived>::load_pages_async(
    std::span<const IORequest> requests,
    IOCallback callback
) {
    assert(callback != nullptr && "callback is null");
    return derived().load_pages_async_impl(requests, std::move(callback));
}

template <typename Derived>
void StorageBackend<Derived>::prefetch_hint(
    std::span<const std::size_t> pages
) {
    if (capabilities().supports_prefetch_hint) {
        derived().prefetch_hint_impl(pages);
    }
}

template <typename Derived>
std::size_t StorageBackend<Derived>::num_pages() const noexcept {
    return derived().num_pages_impl();
}

template <typename Derived>
std::size_t StorageBackend<Derived>::total_bytes() const noexcept {
    return derived().total_bytes_impl();
}

template <typename Derived>
std::size_t StorageBackend<Derived>::file_id() const noexcept {
    return derived().file_id_impl();
}

template <typename Derived>
BackendType StorageBackend<Derived>::type() const noexcept {
    return derived().type_impl();
}

// =============================================================================
// Free Functions
// =============================================================================

inline std::size_t generate_file_id() noexcept {
    static std::atomic<std::size_t> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

inline const char* backend_type_name(BackendType type) noexcept {
    switch (type) {
        case BackendType::LocalFile:  return "LocalFile";
        case BackendType::Compressed: return "Compressed";
        case BackendType::Network:    return "Network";
        case BackendType::Memory:     return "Memory";
        case BackendType::Custom:     return "Custom";
        default:                      return "Unknown";
    }
}

} // namespace scl::mmap::backend
