// =============================================================================
// FILE: scl/mmap/cache/tiered.hpp
// BRIEF: Tiered cache (L1/L2) implementation
// =============================================================================
#pragma once

#include "tiered.h"
#include "eviction.hpp"
#include "../backend/backend.hpp"
#include "../memory/numa.hpp"
#include "../configuration.hpp"
#include "scl/core/macros.hpp"

#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <queue>

namespace scl::mmap::cache {

// =============================================================================
// TieredCacheConfig Implementation
// =============================================================================

constexpr TieredCacheConfig TieredCacheConfig::simple(std::size_t capacity) noexcept {
    return TieredCacheConfig{
        .l1 = {
            .capacity_pages = capacity,
            .eviction = EvictionConfig::lru(),
            .numa_node = -1,
            .use_huge_pages = false
        },
        .l2 = {
            .capacity_pages = 0,
            .eviction = EvictionConfig::lru(),
            .numa_node = -1,
            .use_huge_pages = false
        },
        .enable_l2 = false,
        .writeback_on_evict = true,
        .prefetch_to_l2 = false,
        .promote_threshold = 1,
        .stats_sample_rate = 1
    };
}

constexpr TieredCacheConfig TieredCacheConfig::balanced() noexcept {
    return TieredCacheConfig{
        .l1 = {
            .capacity_pages = 64,
            .eviction = EvictionConfig::lru(),
            .numa_node = -1,
            .use_huge_pages = false
        },
        .l2 = {
            .capacity_pages = 512,
            .eviction = EvictionConfig::arc(),
            .numa_node = -1,
            .use_huge_pages = false
        },
        .enable_l2 = true,
        .writeback_on_evict = true,
        .prefetch_to_l2 = true,
        .promote_threshold = 2,
        .stats_sample_rate = 1
    };
}

constexpr TieredCacheConfig TieredCacheConfig::large_working_set() noexcept {
    return TieredCacheConfig{
        .l1 = {
            .capacity_pages = 128,
            .eviction = EvictionConfig::clock(),
            .numa_node = -1,
            .use_huge_pages = true
        },
        .l2 = {
            .capacity_pages = 2048,
            .eviction = EvictionConfig::arc(),
            .numa_node = -1,
            .use_huge_pages = true
        },
        .enable_l2 = true,
        .writeback_on_evict = true,
        .prefetch_to_l2 = true,
        .promote_threshold = 3,
        .stats_sample_rate = 4
    };
}

constexpr TieredCacheConfig TieredCacheConfig::low_latency() noexcept {
    return TieredCacheConfig{
        .l1 = {
            .capacity_pages = 256,
            .eviction = EvictionConfig::clock(),
            .numa_node = -1,
            .use_huge_pages = false
        },
        .l2 = {
            .capacity_pages = 256,
            .eviction = EvictionConfig::lru(),
            .numa_node = -1,
            .use_huge_pages = false
        },
        .enable_l2 = true,
        .writeback_on_evict = false,
        .prefetch_to_l2 = false,
        .promote_threshold = 1,
        .stats_sample_rate = 8
    };
}

// =============================================================================
// CacheStats Implementation
// =============================================================================

// Statistics relationship:
// - total_requests = l1_hits + l1_misses
// - l1_misses = l2_accesses = l2_hits + l2_misses (when L2 enabled)
// - l2_misses ~= backend_reads (when L2 enabled)
// - cache_hits = l1_hits + l2_hits

double CacheStats::hit_rate() const noexcept {
    // Overall cache hit rate: (L1 hits + L2 hits) / total requests
    std::size_t total = l1_hits + l1_misses;
    if (total == 0) return 0.0;
    return static_cast<double>(l1_hits + l2_hits) / static_cast<double>(total);
}

double CacheStats::l1_hit_rate() const noexcept {
    // L1 hit rate: L1 hits / total requests
    std::size_t total = l1_hits + l1_misses;
    if (total == 0) return 0.0;
    return static_cast<double>(l1_hits) / static_cast<double>(total);
}

double CacheStats::l2_hit_rate() const noexcept {
    // L2 hit rate: L2 hits / L2 accesses (only counts when L2 is actually accessed)
    std::size_t total = l2_hits + l2_misses;
    if (total == 0) return 0.0;
    return static_cast<double>(l2_hits) / static_cast<double>(total);
}

std::chrono::nanoseconds CacheStats::avg_latency() const noexcept {
    std::size_t total_accesses = l1_hits + l2_hits + backend_reads;
    if (total_accesses == 0) return std::chrono::nanoseconds(0);

    auto total = total_l1_latency + total_l2_latency + total_backend_latency;
    return total / total_accesses;
}

// =============================================================================
// PageHandle Implementation
// =============================================================================

struct PageHandle::Impl {
    std::byte* data = nullptr;
    std::size_t page_idx = std::numeric_limits<std::size_t>::max();
    std::size_t file_id = 0;
    std::size_t slot_idx = 0;
    bool is_dirty = false;
    bool is_l1 = true;
    TieredCache::Impl* cache_impl = nullptr;

    void unpin();
};

PageHandle::PageHandle() noexcept = default;

PageHandle::PageHandle(PageHandle&& other) noexcept = default;

PageHandle& PageHandle::operator=(PageHandle&& other) noexcept {
    if (this != &other) {
        if (impl_) {
            impl_->unpin();
        }
        impl_ = std::move(other.impl_);
    }
    return *this;
}

PageHandle::~PageHandle() {
    if (impl_) {
        impl_->unpin();
    }
}

PageHandle::PageHandle(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

const std::byte* PageHandle::data() const noexcept {
    return impl_ ? impl_->data : nullptr;
}

std::byte* PageHandle::data() noexcept {
    return impl_ ? impl_->data : nullptr;
}

std::size_t PageHandle::size() const noexcept {
    return impl_ ? kPageSize : 0;
}

std::size_t PageHandle::page_idx() const noexcept {
    return impl_ ? impl_->page_idx : std::numeric_limits<std::size_t>::max();
}

bool PageHandle::valid() const noexcept {
    return impl_ && impl_->data != nullptr;
}

PageHandle::operator bool() const noexcept {
    return valid();
}

void PageHandle::mark_dirty() noexcept {
    if (impl_) {
        impl_->is_dirty = true;
    }
}

bool PageHandle::is_dirty() const noexcept {
    return impl_ && impl_->is_dirty;
}

void PageHandle::release() noexcept {
    if (impl_) {
        impl_->unpin();
        impl_.reset();
    }
}

// =============================================================================
// Internal Cache Structures
// =============================================================================

struct CacheKey {
    std::size_t page_idx;
    std::size_t file_id;

    bool operator==(const CacheKey& other) const noexcept {
        return page_idx == other.page_idx && file_id == other.file_id;
    }
};

struct CacheKeyHash {
    std::size_t operator()(const CacheKey& key) const noexcept {
        return key.page_idx ^ (key.file_id << 16) ^ (key.file_id >> 48);
    }
};

struct CacheSlot {
    std::byte* data = nullptr;
    CacheKey key{};
    PageMetadata metadata{};
    std::atomic<int> pin_count{0};
    std::atomic<bool> is_valid{false};
};

// =============================================================================
// TieredCache::Impl
// =============================================================================

struct TieredCache::Impl {
    TieredCacheConfig config;
    memory::NUMAAllocator numa_alloc;

    // L1 cache
    std::vector<CacheSlot> l1_slots;
    std::unordered_map<CacheKey, std::size_t, CacheKeyHash> l1_map;
    std::unique_ptr<EvictionTracker> l1_tracker;
    mutable std::shared_mutex l1_mutex;
    std::condition_variable_any l1_unpin_cv;  // Notified when a page is unpinned

    // L2 cache
    std::vector<CacheSlot> l2_slots;
    std::unordered_map<CacheKey, std::size_t, CacheKeyHash> l2_map;
    std::unique_ptr<EvictionTracker> l2_tracker;
    mutable std::shared_mutex l2_mutex;
    std::condition_variable_any l2_unpin_cv;  // Notified when a page is unpinned

    // Statistics
    std::atomic<std::size_t> l1_hits{0};
    std::atomic<std::size_t> l1_misses{0};
    std::atomic<std::size_t> l2_hits{0};
    std::atomic<std::size_t> l2_misses{0};
    std::atomic<std::size_t> backend_reads{0};
    std::atomic<std::size_t> l1_evictions{0};
    std::atomic<std::size_t> l2_evictions{0};
    std::atomic<std::size_t> dirty_writebacks{0};
    std::atomic<std::size_t> l2_to_l1_promotions{0};
    std::atomic<std::size_t> l1_to_l2_demotions{0};
    std::atomic<std::size_t> prefetch_requests{0};
    std::atomic<std::size_t> prefetch_hits{0};
    std::atomic<std::size_t> total_l1_latency_ns{0};
    std::atomic<std::size_t> total_l2_latency_ns{0};
    std::atomic<std::size_t> total_backend_latency_ns{0};

    // Free slot tracking
    std::queue<std::size_t> l1_free_slots;
    std::queue<std::size_t> l2_free_slots;

    explicit Impl(TieredCacheConfig cfg)
        : config(cfg)
        , numa_alloc(memory::NUMAConfig::preferred(cfg.l1.numa_node))
    {
        // Initialize L1
        l1_slots.resize(config.l1.capacity_pages);
        l1_tracker = std::make_unique<EvictionTracker>(
            config.l1.capacity_pages, config.l1.eviction
        );

        for (std::size_t i = 0; i < l1_slots.size(); ++i) {
            l1_slots[i].data = static_cast<std::byte*>(
                numa_alloc.allocate(kPageSize, config.l1.numa_node)
            );
            l1_free_slots.push(i);
        }

        // Initialize L2
        if (config.enable_l2) {
            l2_slots.resize(config.l2.capacity_pages);
            l2_tracker = std::make_unique<EvictionTracker>(
                config.l2.capacity_pages, config.l2.eviction
            );

            for (std::size_t i = 0; i < l2_slots.size(); ++i) {
                l2_slots[i].data = static_cast<std::byte*>(
                    numa_alloc.allocate(kPageSize, config.l2.numa_node)
                );
                l2_free_slots.push(i);
            }
        }
    }

    ~Impl() {
        // Free all allocated pages
        for (auto& slot : l1_slots) {
            if (slot.data) {
                numa_alloc.deallocate(slot.data, kPageSize);
            }
        }
        for (auto& slot : l2_slots) {
            if (slot.data) {
                numa_alloc.deallocate(slot.data, kPageSize);
            }
        }
    }

    // Try to find page in L1
    std::size_t find_l1(const CacheKey& key) const noexcept {
        std::shared_lock lock(l1_mutex);
        auto it = l1_map.find(key);
        if (it != l1_map.end() && l1_slots[it->second].is_valid.load()) {
            return it->second;
        }
        return std::numeric_limits<std::size_t>::max();
    }

    // Find and pin page in L1 atomically (fixes TOCTOU race)
    // Returns slot index and increments pin_count while holding lock
    std::size_t find_and_pin_l1(const CacheKey& key) {
        std::unique_lock lock(l1_mutex);
        auto it = l1_map.find(key);
        if (it != l1_map.end() && l1_slots[it->second].is_valid.load()) {
            std::size_t slot_idx = it->second;
            l1_slots[slot_idx].pin_count.fetch_add(1, std::memory_order_acquire);
            l1_tracker->pin(slot_idx);
            return slot_idx;
        }
        return std::numeric_limits<std::size_t>::max();
    }

    // Try to find page in L2
    std::size_t find_l2(const CacheKey& key) const noexcept {
        if (!config.enable_l2) return std::numeric_limits<std::size_t>::max();

        std::shared_lock lock(l2_mutex);
        auto it = l2_map.find(key);
        if (it != l2_map.end() && l2_slots[it->second].is_valid.load()) {
            return it->second;
        }
        return std::numeric_limits<std::size_t>::max();
    }

    // Find and pin page in L2 atomically (fixes TOCTOU race)
    std::size_t find_and_pin_l2(const CacheKey& key) {
        if (!config.enable_l2) return std::numeric_limits<std::size_t>::max();

        std::unique_lock lock(l2_mutex);
        auto it = l2_map.find(key);
        if (it != l2_map.end() && l2_slots[it->second].is_valid.load()) {
            std::size_t slot_idx = it->second;
            l2_slots[slot_idx].pin_count.fetch_add(1, std::memory_order_acquire);
            l2_tracker->pin(slot_idx);
            return slot_idx;
        }
        return std::numeric_limits<std::size_t>::max();
    }

    // Get a free L1 slot, evicting if necessary
    std::size_t get_free_l1_slot() {
        std::unique_lock lock(l1_mutex);

        if (!l1_free_slots.empty()) {
            std::size_t slot = l1_free_slots.front();
            l1_free_slots.pop();
            return slot;
        }

        // Need to evict
        std::size_t victim = l1_tracker->select_victim(true);
        if (victim == std::numeric_limits<std::size_t>::max()) {
            return victim;
        }

        auto& slot = l1_slots[victim];

        // Wait for pin to release using condition variable with timeout
        constexpr auto kMaxWait = std::chrono::milliseconds(kEvictionWaitTimeoutMs);
        while (slot.pin_count.load() > 0) {
            if (l1_unpin_cv.wait_for(lock, kMaxWait) == std::cv_status::timeout) {
                // Timeout - try another victim or return failure
                std::size_t alt_victim = l1_tracker->select_victim(true);
                if (alt_victim != victim && alt_victim != std::numeric_limits<std::size_t>::max()) {
                    if (l1_slots[alt_victim].pin_count.load() == 0) {
                        victim = alt_victim;
                        break;
                    }
                }
                // Continue waiting on original victim
            }
        }

        // Demote to L2 if enabled and dirty
        if (config.enable_l2 && slot.metadata.is_dirty) {
            demote_to_l2(victim, lock);
        }

        // Remove from map
        l1_map.erase(slot.key);
        l1_tracker->on_evict(victim);
        slot.is_valid.store(false);
        l1_evictions.fetch_add(1);

        return victim;
    }

    // Get a free L2 slot, evicting if necessary
    std::size_t get_free_l2_slot() {
        if (!config.enable_l2) return std::numeric_limits<std::size_t>::max();

        std::unique_lock lock(l2_mutex);

        if (!l2_free_slots.empty()) {
            std::size_t slot = l2_free_slots.front();
            l2_free_slots.pop();
            return slot;
        }

        // Need to evict
        std::size_t victim = l2_tracker->select_victim(true);
        if (victim == std::numeric_limits<std::size_t>::max()) {
            return victim;
        }

        auto& slot = l2_slots[victim];

        // Wait for pin using condition variable with timeout
        constexpr auto kMaxWait = std::chrono::milliseconds(kEvictionWaitTimeoutMs);
        while (slot.pin_count.load() > 0) {
            if (l2_unpin_cv.wait_for(lock, kMaxWait) == std::cv_status::timeout) {
                // Timeout - try another victim
                std::size_t alt_victim = l2_tracker->select_victim(true);
                if (alt_victim != victim && alt_victim != std::numeric_limits<std::size_t>::max()) {
                    if (l2_slots[alt_victim].pin_count.load() == 0) {
                        victim = alt_victim;
                        break;
                    }
                }
            }
        }

        // Writeback if dirty
        if (slot.metadata.is_dirty && config.writeback_on_evict) {
            dirty_writebacks.fetch_add(1);
            // Note: actual writeback would happen here via backend
        }

        l2_map.erase(slot.key);
        l2_tracker->on_evict(victim);
        slot.is_valid.store(false);
        l2_evictions.fetch_add(1);

        return victim;
    }

    // Demote L1 page to L2
    void demote_to_l2(std::size_t l1_slot, std::unique_lock<std::shared_mutex>& l1_lock) {
        if (!config.enable_l2) return;

        auto& src = l1_slots[l1_slot];
        CacheKey key = src.key;
        PageMetadata meta = src.metadata;

        // Copy data while holding lock to avoid data race
        alignas(64) std::byte temp_buffer[kPageSize];
        std::memcpy(temp_buffer, src.data, kPageSize);

        l1_lock.unlock();

        std::size_t l2_slot = get_free_l2_slot();
        if (l2_slot == std::numeric_limits<std::size_t>::max()) {
            l1_lock.lock();
            return;
        }

        {
            std::unique_lock l2_lock(l2_mutex);
            auto& dst = l2_slots[l2_slot];

            // Copy from temp buffer (safe, no race)
            std::memcpy(dst.data, temp_buffer, kPageSize);
            dst.key = key;
            dst.metadata = meta;
            dst.is_valid.store(true);

            l2_map[key] = l2_slot;
            l2_tracker->on_access(l2_slot, meta, false);
        }

        l1_to_l2_demotions.fetch_add(1);
        l1_lock.lock();
    }

    // Promote L2 page to L1
    bool promote_to_l1(std::size_t l2_slot) {
        std::size_t l1_slot = get_free_l1_slot();
        if (l1_slot == std::numeric_limits<std::size_t>::max()) {
            return false;
        }

        std::unique_lock l2_lock(l2_mutex);
        auto& src = l2_slots[l2_slot];

        if (!src.is_valid.load()) {
            // Return l1_slot to free list (must hold l1_mutex)
            std::unique_lock l1_lock(l1_mutex);
            l1_free_slots.push(l1_slot);
            return false;
        }

        CacheKey key = src.key;
        PageMetadata meta = src.metadata;

        // Copy to L1
        {
            std::unique_lock l1_lock(l1_mutex);
            auto& dst = l1_slots[l1_slot];

            std::memcpy(dst.data, src.data, kPageSize);
            dst.key = key;
            dst.metadata = meta;
            dst.is_valid.store(true);

            l1_map[key] = l1_slot;
            l1_tracker->on_access(l1_slot, meta, false);
        }

        // Remove from L2
        l2_map.erase(key);
        l2_tracker->on_evict(l2_slot);
        src.is_valid.store(false);
        l2_free_slots.push(l2_slot);

        l2_to_l1_promotions.fetch_add(1);
        return true;
    }

    void unpin_slot(std::size_t slot_idx, bool is_l1, bool is_dirty) {
        if (is_l1) {
            auto& slot = l1_slots[slot_idx];
            if (is_dirty) {
                slot.metadata.is_dirty = true;
            }
            int prev_count = slot.pin_count.fetch_sub(1);
            l1_tracker->unpin(slot_idx);
            // Notify waiting eviction threads if this was the last pin
            if (prev_count == 1) {
                l1_unpin_cv.notify_all();
            }
        } else {
            auto& slot = l2_slots[slot_idx];
            if (is_dirty) {
                slot.metadata.is_dirty = true;
            }
            int prev_count = slot.pin_count.fetch_sub(1);
            l2_tracker->unpin(slot_idx);
            // Notify waiting eviction threads if this was the last pin
            if (prev_count == 1) {
                l2_unpin_cv.notify_all();
            }
        }
    }

    CacheStats get_stats() const {
        return CacheStats{
            .l1_hits = l1_hits.load(),
            .l1_misses = l1_misses.load(),
            .l2_hits = l2_hits.load(),
            .l2_misses = l2_misses.load(),
            .backend_reads = backend_reads.load(),
            .l1_evictions = l1_evictions.load(),
            .l2_evictions = l2_evictions.load(),
            .dirty_writebacks = dirty_writebacks.load(),
            .l2_to_l1_promotions = l2_to_l1_promotions.load(),
            .l1_to_l2_demotions = l1_to_l2_demotions.load(),
            .prefetch_requests = prefetch_requests.load(),
            .prefetch_hits = prefetch_hits.load(),
            .total_l1_latency = std::chrono::nanoseconds(total_l1_latency_ns.load()),
            .total_l2_latency = std::chrono::nanoseconds(total_l2_latency_ns.load()),
            .total_backend_latency = std::chrono::nanoseconds(total_backend_latency_ns.load())
        };
    }

    void reset_stats() {
        l1_hits.store(0);
        l1_misses.store(0);
        l2_hits.store(0);
        l2_misses.store(0);
        backend_reads.store(0);
        l1_evictions.store(0);
        l2_evictions.store(0);
        dirty_writebacks.store(0);
        l2_to_l1_promotions.store(0);
        l1_to_l2_demotions.store(0);
        prefetch_requests.store(0);
        prefetch_hits.store(0);
        total_l1_latency_ns.store(0);
        total_l2_latency_ns.store(0);
        total_backend_latency_ns.store(0);
    }
};

void PageHandle::Impl::unpin() {
    if (cache_impl) {
        cache_impl->unpin_slot(slot_idx, is_l1, is_dirty);
    }
}

// =============================================================================
// TieredCache Implementation
// =============================================================================

TieredCache::TieredCache(TieredCacheConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

TieredCache::~TieredCache() = default;

TieredCache::TieredCache(TieredCache&& other) noexcept = default;
TieredCache& TieredCache::operator=(TieredCache&& other) noexcept = default;

template <typename BackendT>
PageHandle TieredCache::get(std::size_t page_idx, BackendT* backend) {
    SCL_CHECK_ARG(backend != nullptr, "backend is null");

    CacheKey key{page_idx, backend->file_id()};
    auto now = std::chrono::steady_clock::now();

    // Check L1 using atomic find_and_pin (fixes TOCTOU race)
    auto start = std::chrono::steady_clock::now();
    std::size_t l1_slot = impl_->find_and_pin_l1(key);

    if (l1_slot != std::numeric_limits<std::size_t>::max()) {
        // L1 hit - slot is already pinned by find_and_pin_l1
        auto& slot = impl_->l1_slots[l1_slot];
        slot.metadata.access_count++;
        slot.metadata.last_access = now;

        impl_->l1_tracker->on_access(l1_slot, slot.metadata, true);

        auto latency = std::chrono::steady_clock::now() - start;
        impl_->l1_hits.fetch_add(1);
        impl_->total_l1_latency_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(latency).count()
        );

        auto handle_impl = std::make_unique<PageHandle::Impl>();
        handle_impl->data = slot.data;
        handle_impl->page_idx = page_idx;
        handle_impl->file_id = backend->file_id();
        handle_impl->slot_idx = l1_slot;
        handle_impl->is_l1 = true;
        handle_impl->cache_impl = impl_.get();

        return PageHandle(std::move(handle_impl));
    }

    impl_->l1_misses.fetch_add(1);

    // Check L2 using atomic find_and_pin
    if (impl_->config.enable_l2) {
        std::size_t l2_slot = impl_->find_and_pin_l2(key);

        if (l2_slot != std::numeric_limits<std::size_t>::max()) {
            // L2 hit - slot is already pinned by find_and_pin_l2
            auto& slot = impl_->l2_slots[l2_slot];
            slot.metadata.access_count++;
            slot.metadata.last_access = now;

            impl_->l2_tracker->on_access(l2_slot, slot.metadata, true);

            auto latency = std::chrono::steady_clock::now() - start;
            impl_->l2_hits.fetch_add(1);
            impl_->total_l2_latency_ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(latency).count()
            );

            // Check if should promote to L1
            if (slot.metadata.access_count >= impl_->config.promote_threshold) {
                // Unpin L2 before promotion (promote_to_l1 will handle its own pinning)
                impl_->unpin_slot(l2_slot, false, false);

                if (impl_->promote_to_l1(l2_slot)) {
                    // Re-lookup in L1 using atomic find_and_pin
                    l1_slot = impl_->find_and_pin_l1(key);
                    if (l1_slot != std::numeric_limits<std::size_t>::max()) {
                        auto& l1_s = impl_->l1_slots[l1_slot];

                        auto handle_impl = std::make_unique<PageHandle::Impl>();
                        handle_impl->data = l1_s.data;
                        handle_impl->page_idx = page_idx;
                        handle_impl->file_id = backend->file_id();
                        handle_impl->slot_idx = l1_slot;
                        handle_impl->is_l1 = true;
                        handle_impl->cache_impl = impl_.get();

                        return PageHandle(std::move(handle_impl));
                    }
                }
                // Promotion failed, but we already unpinned L2
                // Fall through to backend read
            } else {
                // Return L2 handle (already pinned)
                auto handle_impl = std::make_unique<PageHandle::Impl>();
                handle_impl->data = slot.data;
                handle_impl->page_idx = page_idx;
                handle_impl->file_id = backend->file_id();
                handle_impl->slot_idx = l2_slot;
                handle_impl->is_l1 = false;
                handle_impl->cache_impl = impl_.get();

                return PageHandle(std::move(handle_impl));
            }
        }

        impl_->l2_misses.fetch_add(1);
    }

    // Cache miss - load from backend
    start = std::chrono::steady_clock::now();

    l1_slot = impl_->get_free_l1_slot();
    if (l1_slot == std::numeric_limits<std::size_t>::max()) {
        // No space available
        return PageHandle();
    }

    auto& slot = impl_->l1_slots[l1_slot];

    // Load from backend
    std::size_t bytes = backend->load_page(page_idx, slot.data);
    if (bytes == 0) {
        std::unique_lock lock(impl_->l1_mutex);
        impl_->l1_free_slots.push(l1_slot);
        return PageHandle();
    }

    auto latency = std::chrono::steady_clock::now() - start;
    impl_->backend_reads.fetch_add(1);
    impl_->total_backend_latency_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(latency).count()
    );

    // Setup slot
    slot.key = key;
    slot.metadata = PageMetadata{
        .page_idx = page_idx,
        .file_id = backend->file_id(),
        .access_count = 1,
        .last_access = now,
        .load_latency = std::chrono::duration_cast<std::chrono::nanoseconds>(latency),
        .is_dirty = false,
        .is_pinned = true,
        .priority = 0
    };
    slot.pin_count.store(1);
    slot.is_valid.store(true);

    {
        std::unique_lock lock(impl_->l1_mutex);
        impl_->l1_map[key] = l1_slot;
        impl_->l1_tracker->on_access(l1_slot, slot.metadata, false);
        impl_->l1_tracker->pin(l1_slot);
    }

    auto handle_impl = std::make_unique<PageHandle::Impl>();
    handle_impl->data = slot.data;
    handle_impl->page_idx = page_idx;
    handle_impl->file_id = backend->file_id();
    handle_impl->slot_idx = l1_slot;
    handle_impl->is_l1 = true;
    handle_impl->cache_impl = impl_.get();

    return PageHandle(std::move(handle_impl));
}

template <typename BackendT>
void TieredCache::prefetch(std::span<const std::size_t> pages, BackendT* backend) {
    if (!backend || pages.empty()) return;

    impl_->prefetch_requests.fetch_add(pages.size());

    std::vector<std::size_t> to_load;
    to_load.reserve(pages.size());

    for (std::size_t page_idx : pages) {
        CacheKey key{page_idx, backend->file_id()};

        // Check if already cached
        if (impl_->find_l1(key) != std::numeric_limits<std::size_t>::max() ||
            impl_->find_l2(key) != std::numeric_limits<std::size_t>::max()) {
            impl_->prefetch_hits.fetch_add(1);
            continue;
        }

        to_load.push_back(page_idx);
    }

    // Issue backend prefetch hint
    if (!to_load.empty()) {
        backend->prefetch_hint(to_load);
    }

    // Load into L2 (or L1 if L2 disabled)
    auto now = std::chrono::steady_clock::now();

    for (std::size_t page_idx : to_load) {
        CacheKey key{page_idx, backend->file_id()};

        std::size_t slot_idx;
        bool is_l1;

        if (impl_->config.enable_l2 && impl_->config.prefetch_to_l2) {
            slot_idx = impl_->get_free_l2_slot();
            is_l1 = false;
            if (slot_idx == std::numeric_limits<std::size_t>::max()) {
                continue;
            }
        } else {
            slot_idx = impl_->get_free_l1_slot();
            is_l1 = true;
            if (slot_idx == std::numeric_limits<std::size_t>::max()) {
                continue;
            }
        }

        auto& slot = is_l1 ? impl_->l1_slots[slot_idx] : impl_->l2_slots[slot_idx];

        auto start = std::chrono::steady_clock::now();
        std::size_t bytes = backend->load_page(page_idx, slot.data);
        auto latency = std::chrono::steady_clock::now() - start;

        if (bytes == 0) {
            if (is_l1) {
                std::unique_lock lock(impl_->l1_mutex);
                impl_->l1_free_slots.push(slot_idx);
            } else {
                std::unique_lock lock(impl_->l2_mutex);
                impl_->l2_free_slots.push(slot_idx);
            }
            continue;
        }

        slot.key = key;
        slot.metadata = PageMetadata{
            .page_idx = page_idx,
            .file_id = backend->file_id(),
            .access_count = 0,
            .last_access = now,
            .load_latency = std::chrono::duration_cast<std::chrono::nanoseconds>(latency),
            .is_dirty = false,
            .is_pinned = false,
            .priority = kPrefetchPagePriority  // Lower priority for prefetched pages
        };
        slot.is_valid.store(true);

        if (is_l1) {
            std::unique_lock lock(impl_->l1_mutex);
            impl_->l1_map[key] = slot_idx;
            impl_->l1_tracker->on_access(slot_idx, slot.metadata, false);
        } else {
            std::unique_lock lock(impl_->l2_mutex);
            impl_->l2_map[key] = slot_idx;
            impl_->l2_tracker->on_access(slot_idx, slot.metadata, false);
        }

        impl_->backend_reads.fetch_add(1);
    }
}

bool TieredCache::contains(std::size_t page_idx, std::size_t file_id) const noexcept {
    CacheKey key{page_idx, file_id};
    return impl_->find_l1(key) != std::numeric_limits<std::size_t>::max() ||
           impl_->find_l2(key) != std::numeric_limits<std::size_t>::max();
}

void TieredCache::invalidate(std::size_t page_idx, std::size_t file_id) {
    CacheKey key{page_idx, file_id};

    // Remove from L1
    {
        std::unique_lock lock(impl_->l1_mutex);
        auto it = impl_->l1_map.find(key);
        if (it != impl_->l1_map.end()) {
            auto& slot = impl_->l1_slots[it->second];

            // Wait for unpin using condition variable with timeout
            constexpr auto kMaxWait = std::chrono::milliseconds(kEvictionWaitTimeoutMs);
            while (slot.pin_count.load() > 0) {
                if (impl_->l1_unpin_cv.wait_for(lock, kMaxWait) == std::cv_status::timeout) {
                    // Still pinned after timeout - continue waiting
                    // (invalidate must succeed, so we keep waiting)
                }
            }

            if (slot.metadata.is_dirty && impl_->config.writeback_on_evict) {
                impl_->dirty_writebacks.fetch_add(1);
            }

            impl_->l1_tracker->on_evict(it->second);
            slot.is_valid.store(false);
            impl_->l1_free_slots.push(it->second);
            impl_->l1_map.erase(it);
        }
    }

    // Remove from L2
    if (impl_->config.enable_l2) {
        std::unique_lock lock(impl_->l2_mutex);
        auto it = impl_->l2_map.find(key);
        if (it != impl_->l2_map.end()) {
            auto& slot = impl_->l2_slots[it->second];

            // Wait for unpin using condition variable with timeout
            constexpr auto kMaxWait = std::chrono::milliseconds(kEvictionWaitTimeoutMs);
            while (slot.pin_count.load() > 0) {
                if (impl_->l2_unpin_cv.wait_for(lock, kMaxWait) == std::cv_status::timeout) {
                    // Still pinned after timeout - continue waiting
                }
            }

            if (slot.metadata.is_dirty && impl_->config.writeback_on_evict) {
                impl_->dirty_writebacks.fetch_add(1);
            }

            impl_->l2_tracker->on_evict(it->second);
            slot.is_valid.store(false);
            impl_->l2_free_slots.push(it->second);
            impl_->l2_map.erase(it);
        }
    }
}

void TieredCache::invalidate_all(std::size_t file_id) {
    // Collect keys to invalidate
    std::vector<CacheKey> to_invalidate;

    {
        std::shared_lock lock(impl_->l1_mutex);
        for (const auto& [key, _] : impl_->l1_map) {
            if (key.file_id == file_id) {
                to_invalidate.push_back(key);
            }
        }
    }

    if (impl_->config.enable_l2) {
        std::shared_lock lock(impl_->l2_mutex);
        for (const auto& [key, _] : impl_->l2_map) {
            if (key.file_id == file_id) {
                to_invalidate.push_back(key);
            }
        }
    }

    for (const auto& key : to_invalidate) {
        invalidate(key.page_idx, key.file_id);
    }
}

void TieredCache::flush() {
    // Write back all dirty pages (placeholder - needs backend reference)
    // In practice, we'd store a backend reference per page

    std::size_t dirty_count = 0;

    {
        std::shared_lock lock(impl_->l1_mutex);
        for (const auto& slot : impl_->l1_slots) {
            if (slot.is_valid.load() && slot.metadata.is_dirty) {
                dirty_count++;
            }
        }
    }

    if (impl_->config.enable_l2) {
        std::shared_lock lock(impl_->l2_mutex);
        for (const auto& slot : impl_->l2_slots) {
            if (slot.is_valid.load() && slot.metadata.is_dirty) {
                dirty_count++;
            }
        }
    }

    impl_->dirty_writebacks.fetch_add(dirty_count);
}

void TieredCache::clear() {
    flush();

    {
        std::unique_lock lock(impl_->l1_mutex);
        impl_->l1_map.clear();
        impl_->l1_tracker->clear();
        while (!impl_->l1_free_slots.empty()) {
            impl_->l1_free_slots.pop();
        }
        for (std::size_t i = 0; i < impl_->l1_slots.size(); ++i) {
            impl_->l1_slots[i].is_valid.store(false);
            impl_->l1_free_slots.push(i);
        }
    }

    if (impl_->config.enable_l2) {
        std::unique_lock lock(impl_->l2_mutex);
        impl_->l2_map.clear();
        impl_->l2_tracker->clear();
        while (!impl_->l2_free_slots.empty()) {
            impl_->l2_free_slots.pop();
        }
        for (std::size_t i = 0; i < impl_->l2_slots.size(); ++i) {
            impl_->l2_slots[i].is_valid.store(false);
            impl_->l2_free_slots.push(i);
        }
    }
}

void TieredCache::resize(std::size_t l1_capacity, std::size_t l2_capacity) {
    // Update config - this controls soft capacity (how many pages we keep)
    // Note: This does NOT resize the underlying slot arrays, only controls eviction behavior
    impl_->config.l1.capacity_pages = std::min(l1_capacity, impl_->l1_slots.size());
    impl_->config.l2.capacity_pages = std::min(l2_capacity, impl_->l2_slots.size());

    // Evict excess pages from L1 if current usage exceeds new capacity
    {
        std::unique_lock lock(impl_->l1_mutex);
        while (impl_->l1_map.size() > impl_->config.l1.capacity_pages) {
            std::size_t victim = impl_->l1_tracker->select_victim(false);
            if (victim == std::numeric_limits<std::size_t>::max()) {
                break;
            }

            auto& slot = impl_->l1_slots[victim];
            if (slot.pin_count.load() > 0) {
                // Skip pinned pages
                continue;
            }

            // Demote to L2 if dirty and L2 enabled
            if (impl_->config.enable_l2 && slot.metadata.is_dirty) {
                lock.unlock();
                std::size_t l2_slot = impl_->get_free_l2_slot();
                if (l2_slot != std::numeric_limits<std::size_t>::max()) {
                    std::unique_lock l2_lock(impl_->l2_mutex);
                    auto& dst = impl_->l2_slots[l2_slot];
                    std::memcpy(dst.data, slot.data, kPageSize);
                    dst.key = slot.key;
                    dst.metadata = slot.metadata;
                    dst.is_valid.store(true);
                    impl_->l2_map[slot.key] = l2_slot;
                    impl_->l2_tracker->on_access(l2_slot, slot.metadata, false);
                    impl_->l1_to_l2_demotions.fetch_add(1);
                }
                lock.lock();
            }

            impl_->l1_map.erase(slot.key);
            impl_->l1_tracker->on_evict(victim);
            slot.is_valid.store(false);
            impl_->l1_free_slots.push(victim);
            impl_->l1_evictions.fetch_add(1);
        }
    }

    // Evict excess pages from L2
    if (impl_->config.enable_l2) {
        std::unique_lock lock(impl_->l2_mutex);
        while (impl_->l2_map.size() > impl_->config.l2.capacity_pages) {
            std::size_t victim = impl_->l2_tracker->select_victim(false);
            if (victim == std::numeric_limits<std::size_t>::max()) {
                break;
            }

            auto& slot = impl_->l2_slots[victim];
            if (slot.pin_count.load() > 0) {
                continue;
            }

            if (slot.metadata.is_dirty && impl_->config.writeback_on_evict) {
                impl_->dirty_writebacks.fetch_add(1);
            }

            impl_->l2_map.erase(slot.key);
            impl_->l2_tracker->on_evict(victim);
            slot.is_valid.store(false);
            impl_->l2_free_slots.push(victim);
            impl_->l2_evictions.fetch_add(1);
        }
    }
}

const TieredCacheConfig& TieredCache::config() const noexcept {
    return impl_->config;
}

CacheStats TieredCache::stats() const noexcept {
    return impl_->get_stats();
}

void TieredCache::reset_stats() noexcept {
    impl_->reset_stats();
}

std::size_t TieredCache::l1_size() const noexcept {
    std::shared_lock lock(impl_->l1_mutex);
    return impl_->l1_map.size();
}

std::size_t TieredCache::l2_size() const noexcept {
    if (!impl_->config.enable_l2) return 0;
    std::shared_lock lock(impl_->l2_mutex);
    return impl_->l2_map.size();
}

std::size_t TieredCache::memory_usage() const noexcept {
    std::size_t usage = impl_->l1_slots.size() * kPageSize;
    if (impl_->config.enable_l2) {
        usage += impl_->l2_slots.size() * kPageSize;
    }
    return usage;
}

// =============================================================================
// Free Functions
// =============================================================================

inline const char* cache_tier_name(bool is_l1) noexcept {
    return is_l1 ? "L1" : "L2";
}

} // namespace scl::mmap::cache
