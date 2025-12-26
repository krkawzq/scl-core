#pragma once

// =============================================================================
/// @file scheduler.hpp
/// @brief Page Scheduling and Eviction Algorithms
///
/// This file provides memory management for virtual arrays:
///
/// 1. PagePool: Fixed-size page allocator with free list
/// 2. Scheduler: Abstract base for page scheduling strategies
/// 3. StandardScheduler: Clock algorithm for general workloads
/// 4. SequentialScheduler: Optimized for sequential scan with prefetch
///
/// Performance Optimizations:
///
/// - Clock algorithm replaces LRU linked list (O(1) eviction, less overhead)
/// - Spin-lock for short critical sections (better than mutex for fast paths)
/// - Lock-free fast path for cache hits
/// - Batch prefetch for sequential access
///
/// Clock Algorithm:
///
/// The Clock algorithm approximates LRU with much lower overhead:
/// - Each page has an "accessed" bit
/// - Clock hand sweeps through pages on eviction
/// - If accessed=1, clear it and move on
/// - If accessed=0 and not pinned, evict
/// - O(1) amortized eviction cost
// =============================================================================

#include "scl/core/macros.hpp"
#include "scl/mmap/configuration.hpp"
#include "scl/mmap/table.hpp"

#include <cstddef>
#include <vector>
#include <functional>
#include <memory>
#include <atomic>
#include <stdexcept>
#include <new>

namespace scl::mmap {

// =============================================================================
// SpinLock: Lightweight Lock for Short Critical Sections
// =============================================================================

/// @brief Simple spin lock for fast critical sections
///
/// Advantages over std::mutex:
/// - No system call overhead
/// - Better for very short critical sections (<1000 cycles)
/// - Cache-friendly (single cache line)
///
/// Use only when:
/// - Critical section is very short
/// - Contention is low to moderate
/// - No blocking operations inside critical section
class SpinLock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;

public:
    SCL_FORCE_INLINE void lock() noexcept {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // Spin with pause hint for better CPU utilization
            #if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
            #elif defined(__aarch64__)
                asm volatile("yield" ::: "memory");
            #endif
        }
    }

    SCL_FORCE_INLINE void unlock() noexcept {
        flag_.clear(std::memory_order_release);
    }

    SCL_FORCE_INLINE bool try_lock() noexcept {
        return !flag_.test_and_set(std::memory_order_acquire);
    }
};

/// @brief RAII lock guard for SpinLock
class SpinGuard {
    SpinLock& lock_;
public:
    explicit SpinGuard(SpinLock& lock) noexcept : lock_(lock) { lock_.lock(); }
    ~SpinGuard() { lock_.unlock(); }
    SpinGuard(const SpinGuard&) = delete;
    SpinGuard& operator=(const SpinGuard&) = delete;
};

// =============================================================================
// PagePool: Fixed-Size Page Allocator
// =============================================================================

/// @brief Thread-safe fixed-size page pool
///
/// Design:
/// - Pre-allocated page array (no runtime allocation)
/// - Stack-based free list for O(1) alloc/free
/// - Spin lock for thread safety (short critical section)
///
/// Memory Layout:
/// - All pages allocated contiguously for cache efficiency
/// - 64-byte aligned for SIMD and cache line optimization
class PagePool {
    std::unique_ptr<Page[]> pages_;
    std::vector<Page*> free_list_;
    std::size_t capacity_;
    mutable SpinLock lock_;

public:
    explicit PagePool(std::size_t num_pages)
        : capacity_(num_pages)
    {
        // Allocate aligned page array
        pages_ = std::make_unique<Page[]>(num_pages);
        
        // Initialize free list (reverse order for stack behavior)
        free_list_.reserve(num_pages);
        for (std::size_t i = num_pages; i > 0; --i) {
            free_list_.push_back(&pages_[i - 1]);
        }
    }

    // Non-copyable, non-movable
    PagePool(const PagePool&) = delete;
    PagePool& operator=(const PagePool&) = delete;

    /// @brief Allocate a free page (O(1))
    /// @return Page pointer or nullptr if pool exhausted
    Page* alloc() noexcept {
        SpinGuard guard(lock_);
        if (SCL_UNLIKELY(free_list_.empty())) {
            return nullptr;
        }
        Page* p = free_list_.back();
        free_list_.pop_back();
        return p;
    }

    /// @brief Return page to pool (O(1))
    void free(Page* p) noexcept {
        if (SCL_UNLIKELY(!p)) return;
        SpinGuard guard(lock_);
        free_list_.push_back(p);
    }

    /// @brief Get number of free pages
    SCL_NODISCARD std::size_t free_count() const noexcept {
        SpinGuard guard(lock_);
        return free_list_.size();
    }

    /// @brief Get total capacity
    SCL_NODISCARD std::size_t capacity() const noexcept { return capacity_; }
};

// =============================================================================
// LoadFunc: Page Load Callback
// =============================================================================

/// @brief Callback for loading page data from storage
/// @param page_idx Logical page index
/// @param dest Destination buffer (kPageSize bytes)
using LoadFunc = std::function<void(std::size_t, std::byte*)>;

// =============================================================================
// Scheduler: Abstract Base Class
// =============================================================================

/// @brief Abstract page scheduler interface
///
/// Responsibilities:
/// - Handle page requests (load on miss)
/// - Manage page eviction when pool is full
/// - Provide prefetch hints for sequential access
class Scheduler {
protected:
    PageTable* table_;
    PagePool* pool_;
    LoadFunc load_func_;

public:
    Scheduler(PageTable* table, PagePool* pool, LoadFunc load_func)
        : table_(table), pool_(pool), load_func_(std::move(load_func)) {}

    virtual ~Scheduler() = default;

    /// @brief Request a page (may trigger load or eviction)
    /// @param page_idx Logical page index
    /// @return PageHandle with pinned page
    /// @throws std::runtime_error on OOM
    virtual PageHandle request(std::size_t page_idx) = 0;

    /// @brief Prefetch hint for upcoming pages
    /// @param page_idx Starting page index
    /// @param count Number of pages to prefetch
    virtual void prefetch(std::size_t page_idx, std::size_t count) {
        (void)page_idx;
        (void)count;
    }

protected:
    /// @brief Load page data from storage (no lock held)
    void perform_load(std::size_t page_idx, Page* page) {
        load_func_(page_idx, page->data);
        table_->install(page_idx, page);
    }
};

// =============================================================================
// StandardScheduler: Clock Algorithm
// =============================================================================

/// @brief Standard scheduler using Clock eviction algorithm
///
/// Clock Algorithm Advantages:
/// - O(1) amortized eviction (vs O(n) for naive LRU)
/// - No linked list maintenance overhead
/// - Better cache behavior (sequential scan through entries)
/// - Simple implementation with atomic operations
///
/// Operation:
/// 1. On hit: Set accessed bit, return page
/// 2. On miss: Try allocate from pool
/// 3. If pool empty: Run clock hand to find victim
/// 4. Load page data and install
class StandardScheduler : public Scheduler {
    std::atomic<std::size_t> clock_hand_{0};
    mutable SpinLock evict_lock_;

public:
    using Scheduler::Scheduler;

    PageHandle request(std::size_t page_idx) override {
        // Fast path: Cache hit (lock-free)
        if (Page* page = table_->get(page_idx)) {
            table_->touch(page_idx);
            return PageHandle(table_, page_idx, page);
        }

        // Slow path: Cache miss
        return handle_miss(page_idx);
    }

private:
    PageHandle handle_miss(std::size_t page_idx) {
        // Try allocate from pool
        Page* page = pool_->alloc();

        // If pool empty, evict using clock algorithm
        if (!page) {
            page = evict_clock();
        }

        // OOM: All pages pinned
        if (SCL_UNLIKELY(!page)) {
            throw std::runtime_error("Mmap OOM: All pages are pinned");
        }

        // Load page data
        try {
            perform_load(page_idx, page);
        } catch (...) {
            pool_->free(page);
            throw;
        }

        return PageHandle(table_, page_idx, page);
    }

    /// @brief Clock algorithm eviction
    Page* evict_clock() {
        SpinGuard guard(evict_lock_);
        
        const std::size_t num_pages = table_->num_pages();
        if (num_pages == 0) return nullptr;

        // Maximum iterations = 2 * num_pages (worst case: all accessed once)
        const std::size_t max_iterations = num_pages * 2;
        
        for (std::size_t iter = 0; iter < max_iterations; ++iter) {
            std::size_t idx = clock_hand_.fetch_add(1, std::memory_order_relaxed) % num_pages;
            PageEntry& entry = table_->entry(idx);

            // Skip unloaded pages
            if (!entry.is_loaded()) {
                continue;
            }

            // Skip pinned pages
            if (entry.is_pinned()) {
                continue;
            }

            // Check accessed bit
            if (entry.check_and_clear_accessed()) {
                // Give second chance
                continue;
            }

            // Found victim - evict
            Page* victim = table_->uninstall(idx);
            if (victim) {
                return victim;
            }
        }

        return nullptr;  // All pages pinned or accessed
    }
};

// =============================================================================
// SequentialScheduler: Optimized for Sequential Access
// =============================================================================

/// @brief Scheduler optimized for sequential scan with prefetch
///
/// Optimizations:
/// - Batch prefetch of upcoming pages
/// - FIFO-like eviction (oldest pages first)
/// - Best-effort prefetch (never blocks for prefetch)
///
/// Use Case:
/// - Row-by-row matrix iteration
/// - Streaming data processing
/// - Sequential file reads
class SequentialScheduler : public Scheduler {
    std::size_t prefetch_depth_;
    std::atomic<std::size_t> clock_hand_{0};
    std::atomic<std::size_t> last_requested_{0};
    mutable SpinLock evict_lock_;

public:
    SequentialScheduler(PageTable* table, PagePool* pool, LoadFunc load_func,
                        std::size_t depth = 4)
        : Scheduler(table, pool, std::move(load_func))
        , prefetch_depth_(depth) {}

    PageHandle request(std::size_t page_idx) override {
        // Update access tracking
        last_requested_.store(page_idx, std::memory_order_relaxed);
        
        // Try prefetch ahead (best effort)
        try_prefetch_ahead(page_idx);

        // Fast path: Cache hit
        if (Page* page = table_->get(page_idx)) {
            table_->touch(page_idx);
            return PageHandle(table_, page_idx, page);
        }

        // Slow path: Cache miss
        return handle_miss(page_idx);
    }

    void prefetch(std::size_t start_page, std::size_t count) override {
        const std::size_t num_pages = table_->num_pages();
        
        for (std::size_t i = 0; i < count; ++i) {
            std::size_t target = start_page + i;
            if (target >= num_pages) break;
            
            // Skip already loaded
            if (table_->is_loaded(target)) continue;
            
            // Try allocate (don't evict for prefetch)
            Page* page = pool_->alloc();
            if (!page) break;
            
            try {
                perform_load(target, page);
            } catch (...) {
                pool_->free(page);
                break;
            }
        }
    }

private:
    PageHandle handle_miss(std::size_t page_idx) {
        Page* page = pool_->alloc();

        if (!page) {
            page = evict_oldest();
        }

        if (SCL_UNLIKELY(!page)) {
            throw std::runtime_error("Mmap OOM: Sequential scan blocked");
        }

        try {
            perform_load(page_idx, page);
        } catch (...) {
            pool_->free(page);
            throw;
        }

        return PageHandle(table_, page_idx, page);
    }

    void try_prefetch_ahead(std::size_t current_page) {
        const std::size_t num_pages = table_->num_pages();
        
        for (std::size_t i = 1; i <= prefetch_depth_; ++i) {
            std::size_t ahead = current_page + i;
            if (ahead >= num_pages) break;
            
            // Skip already loaded
            if (table_->is_loaded(ahead)) continue;
            
            // Best effort: Only prefetch if pool has free pages
            Page* page = pool_->alloc();
            if (!page) break;
            
            try {
                perform_load(ahead, page);
            } catch (...) {
                pool_->free(page);
                break;
            }
        }
    }

    /// @brief Evict oldest pages (FIFO-like for sequential access)
    Page* evict_oldest() {
        SpinGuard guard(evict_lock_);
        
        const std::size_t num_pages = table_->num_pages();
        if (num_pages == 0) return nullptr;

        // Start from clock hand (tracks oldest region)
        std::size_t start = clock_hand_.load(std::memory_order_relaxed);
        
        for (std::size_t i = 0; i < num_pages; ++i) {
            std::size_t idx = (start + i) % num_pages;
            PageEntry& entry = table_->entry(idx);

            if (!entry.is_loaded() || entry.is_pinned()) {
                continue;
            }

            // For sequential: Prefer pages far behind current access
            std::size_t current = last_requested_.load(std::memory_order_relaxed);
            if (idx < current || (current < num_pages / 2 && idx > current + num_pages / 2)) {
                Page* victim = table_->uninstall(idx);
                if (victim) {
                    clock_hand_.store((idx + 1) % num_pages, std::memory_order_relaxed);
                    return victim;
                }
            }
        }

        // Fallback: Evict any unpinned page
        for (std::size_t i = 0; i < num_pages; ++i) {
            std::size_t idx = (start + i) % num_pages;
            PageEntry& entry = table_->entry(idx);

            if (!entry.is_loaded() || entry.is_pinned()) {
                continue;
            }

            Page* victim = table_->uninstall(idx);
            if (victim) {
                clock_hand_.store((idx + 1) % num_pages, std::memory_order_relaxed);
                return victim;
            }
        }

        return nullptr;
    }
};

} // namespace scl::mmap
