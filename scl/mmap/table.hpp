#pragma once

// =============================================================================
/// @file table.hpp
/// @brief Page Table and Handle Management for Memory-Mapped Arrays
///
/// This file provides the core data structures for virtual memory paging:
///
/// 1. Page: Fixed-size aligned memory block
/// 2. PageEntry: Atomic page table entry with pin counting
/// 3. PageTable: Virtual-to-physical address translation
/// 4. PageHandle: RAII wrapper for page pinning
///
/// Performance Optimizations:
///
/// - Bit-shift address translation (no division/modulo)
/// - Relaxed atomics for non-critical paths
/// - Cache-line aligned Page structure
/// - Minimal atomic operations in hot paths
///
/// Memory Ordering Strategy:
///
/// - acquire: When reading page pointer for data access
/// - release: When installing new page (ensures data visible)
/// - relaxed: For pin_count and hint checks
// =============================================================================

#include "scl/core/macros.hpp"
#include "scl/mmap/configuration.hpp"

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <utility>
#include <memory>

namespace scl::mmap {

// =============================================================================
// Page: Cache-Line Aligned Memory Block
// =============================================================================

/// @brief Fixed-size memory page with cache-line alignment
///
/// Alignment ensures optimal memory access patterns:
/// - No false sharing between pages
/// - SIMD-friendly data layout
/// - Efficient DMA transfers
struct alignas(64) Page {
    std::byte data[kPageSize];
};

static_assert(sizeof(Page) == kPageSize, "Page size mismatch");
static_assert(alignof(Page) == 64, "Page alignment must be 64 bytes");

// =============================================================================
// PageEntry: Atomic Page Table Entry
// =============================================================================

/// @brief Lightweight atomic page table entry
///
/// Memory Layout (16 bytes on 64-bit):
/// - ptr: 8 bytes (atomic pointer)
/// - pin_count: 4 bytes (atomic counter)
/// - flags: 4 bytes (atomic flags for dirty/accessed)
///
/// Thread Safety:
/// - Lock-free reads for fast path checks
/// - Atomic pin/unpin for concurrent access
/// - Memory ordering matches access patterns
struct PageEntry {
    /// Page pointer (nullptr = not loaded)
    std::atomic<Page*> ptr{nullptr};

    /// Reference count (>0 prevents eviction)
    std::atomic<uint32_t> pin_count{0};
    
    /// Flags: bit 0 = dirty, bit 1 = accessed (for Clock algorithm)
    std::atomic<uint32_t> flags{0};
    
    // Flag bit positions
    static constexpr uint32_t kFlagDirty = 1 << 0;
    static constexpr uint32_t kFlagAccessed = 1 << 1;

    PageEntry() = default;
    
    // Non-copyable (atomics)
    PageEntry(const PageEntry&) = delete;
    PageEntry& operator=(const PageEntry&) = delete;
    
    // Move semantics for vector resize
    PageEntry(PageEntry&& other) noexcept 
        : ptr(other.ptr.load(std::memory_order_relaxed))
        , pin_count(other.pin_count.load(std::memory_order_relaxed))
        , flags(other.flags.load(std::memory_order_relaxed)) 
    {
        other.ptr.store(nullptr, std::memory_order_relaxed);
        other.pin_count.store(0, std::memory_order_relaxed);
        other.flags.store(0, std::memory_order_relaxed);
    }

    // -------------------------------------------------------------------------
    // Fast Path Accessors (Inline)
    // -------------------------------------------------------------------------

    /// @brief Get page pointer with acquire semantics
    SCL_FORCE_INLINE Page* get() const noexcept {
        return ptr.load(std::memory_order_acquire);
    }

    /// @brief Fast check if page is loaded (relaxed for hints)
    SCL_FORCE_INLINE bool is_loaded() const noexcept {
        return ptr.load(std::memory_order_relaxed) != nullptr;
    }

    /// @brief Fast check if page is pinned
    SCL_FORCE_INLINE bool is_pinned() const noexcept {
        return pin_count.load(std::memory_order_relaxed) > 0;
    }

    /// @brief Check dirty flag
    SCL_FORCE_INLINE bool is_dirty() const noexcept {
        return (flags.load(std::memory_order_relaxed) & kFlagDirty) != 0;
    }
    
    /// @brief Check and clear accessed flag (for Clock algorithm)
    SCL_FORCE_INLINE bool check_and_clear_accessed() noexcept {
        uint32_t old_flags = flags.fetch_and(~kFlagAccessed, std::memory_order_relaxed);
        return (old_flags & kFlagAccessed) != 0;
    }

    /// @brief Set accessed flag
    SCL_FORCE_INLINE void set_accessed() noexcept {
        flags.fetch_or(kFlagAccessed, std::memory_order_relaxed);
    }

    /// @brief Set dirty flag
    SCL_FORCE_INLINE void set_dirty() noexcept {
        flags.fetch_or(kFlagDirty, std::memory_order_relaxed);
    }

    /// @brief Clear all flags
    SCL_FORCE_INLINE void clear_flags() noexcept {
        flags.store(0, std::memory_order_relaxed);
    }
};

// =============================================================================
// PageTable: Virtual Address Translation
// =============================================================================

/// @brief Virtual-to-physical page address translation table
///
/// Design:
/// - Contiguous entry array for cache efficiency
/// - Bit-shift address conversion (zero division overhead)
/// - Lock-free read path for concurrent access
///
/// Address Translation:
/// - page_idx = byte_offset >> kPageShift
/// - page_offset = byte_offset & kPageMask
class PageTable {
private:
    std::unique_ptr<PageEntry[]> entries_;
    std::size_t num_pages_;
    std::size_t total_bytes_;

public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    explicit PageTable(std::size_t total_bytes)
        : total_bytes_(total_bytes)
    {
        num_pages_ = bytes_to_pages(total_bytes);
        entries_ = std::make_unique<PageEntry[]>(num_pages_);
    }

    ~PageTable() = default;

    // Non-copyable, non-movable
    PageTable(const PageTable&) = delete;
    PageTable& operator=(const PageTable&) = delete;

    // -------------------------------------------------------------------------
    // Address Translation (Hot Path - Bit Operations)
    // -------------------------------------------------------------------------

    /// @brief Convert byte offset to page index
    SCL_NODISCARD SCL_FORCE_INLINE std::size_t to_page_idx(std::size_t byte_offset) const noexcept {
        return byte_to_page_idx(byte_offset);
    }

    /// @brief Convert byte offset to offset within page
    SCL_NODISCARD SCL_FORCE_INLINE std::size_t to_page_offset(std::size_t byte_offset) const noexcept {
        return byte_to_page_offset(byte_offset);
    }

    // -------------------------------------------------------------------------
    // Lock-Free Queries
    // -------------------------------------------------------------------------

    /// @brief Check if page is loaded (relaxed for hints)
    SCL_NODISCARD SCL_FORCE_INLINE bool is_loaded(std::size_t page_idx) const noexcept {
        return entries_[page_idx].is_loaded();
    }

    /// @brief Check if page is pinned
    SCL_NODISCARD SCL_FORCE_INLINE bool is_pinned(std::size_t page_idx) const noexcept {
        return entries_[page_idx].is_pinned();
    }

    /// @brief Get page pointer (acquire semantics)
    SCL_NODISCARD SCL_FORCE_INLINE Page* get(std::size_t page_idx) const noexcept {
        return entries_[page_idx].get();
    }

    /// @brief Direct entry access (for scheduler)
    SCL_NODISCARD SCL_FORCE_INLINE PageEntry& entry(std::size_t page_idx) noexcept {
        return entries_[page_idx];
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE const PageEntry& entry(std::size_t page_idx) const noexcept {
        return entries_[page_idx];
    }

    // -------------------------------------------------------------------------
    // Page Management (Scheduler Interface)
    // -------------------------------------------------------------------------

    /// @brief Install page with release semantics (data visible after)
    void install(std::size_t page_idx, Page* page) noexcept {
        entries_[page_idx].ptr.store(page, std::memory_order_release);
        entries_[page_idx].set_accessed();
    }

    /// @brief Uninstall page and return it
    Page* uninstall(std::size_t page_idx) noexcept {
        Page* old = entries_[page_idx].ptr.exchange(nullptr, std::memory_order_acquire);
        entries_[page_idx].clear_flags();
        return old;
    }

    // -------------------------------------------------------------------------
    // Pin/Unpin (Atomic, Relaxed)
    // -------------------------------------------------------------------------

    /// @brief Increment pin count
    SCL_FORCE_INLINE void pin(std::size_t page_idx) noexcept {
        entries_[page_idx].pin_count.fetch_add(1, std::memory_order_relaxed);
    }

    /// @brief Decrement pin count
    SCL_FORCE_INLINE void unpin(std::size_t page_idx) noexcept {
        entries_[page_idx].pin_count.fetch_sub(1, std::memory_order_relaxed);
    }

    /// @brief Mark page as accessed (for LRU/Clock)
    SCL_FORCE_INLINE void touch(std::size_t page_idx) noexcept {
        entries_[page_idx].set_accessed();
    }

    // -------------------------------------------------------------------------
    // Properties
    // -------------------------------------------------------------------------

    SCL_NODISCARD std::size_t num_pages() const noexcept { return num_pages_; }
    SCL_NODISCARD std::size_t total_bytes() const noexcept { return total_bytes_; }
};

// =============================================================================
// PageHandle: RAII Pin Management
// =============================================================================

/// @brief Move-only RAII wrapper for page pinning
///
/// Guarantees:
/// - Page remains in memory while handle is valid
/// - Pin count decremented on destruction
/// - Zero-overhead when empty (null check only)
///
/// Usage:
///   PageHandle handle = scheduler.request(page_idx);
///   if (handle) {
///       const std::byte* data = handle.data();
///       // ... use data safely ...
///   }  // Auto-unpin on scope exit
class PageHandle {
    PageTable* table_;
    std::size_t page_idx_;
    Page* page_;

public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /// @brief Default constructor (empty handle)
    PageHandle() noexcept : table_(nullptr), page_idx_(0), page_(nullptr) {}

    /// @brief Construct and pin page
    PageHandle(PageTable* table, std::size_t page_idx, Page* page) noexcept
        : table_(table), page_idx_(page_idx), page_(page)
    {
        if (SCL_LIKELY(table_ && page_)) {
            table_->pin(page_idx_);
        }
    }

    /// @brief Destructor - unpin if valid
    ~PageHandle() {
        if (table_ && page_) {
            table_->unpin(page_idx_);
        }
    }

    // -------------------------------------------------------------------------
    // Move Semantics
    // -------------------------------------------------------------------------

    PageHandle(PageHandle&& o) noexcept
        : table_(o.table_), page_idx_(o.page_idx_), page_(o.page_)
    {
        o.table_ = nullptr;
        o.page_ = nullptr;
    }

    PageHandle& operator=(PageHandle&& o) noexcept {
        if (SCL_LIKELY(this != &o)) {
            // Unpin current
            if (table_ && page_) {
                table_->unpin(page_idx_);
            }
            // Steal resources
            table_ = o.table_;
            page_idx_ = o.page_idx_;
            page_ = o.page_;
            // Invalidate source
            o.table_ = nullptr;
            o.page_ = nullptr;
        }
        return *this;
    }

    // Non-copyable
    PageHandle(const PageHandle&) = delete;
    PageHandle& operator=(const PageHandle&) = delete;

    // -------------------------------------------------------------------------
    // Data Access
    // -------------------------------------------------------------------------

    /// @brief Get raw data pointer (const)
    SCL_NODISCARD SCL_FORCE_INLINE const std::byte* data() const noexcept {
        return page_->data;
    }

    /// @brief Get raw data pointer (mutable)
    SCL_NODISCARD SCL_FORCE_INLINE std::byte* data() noexcept {
        return page_->data;
    }

    /// @brief Typed access helper
    template <typename T>
    SCL_NODISCARD SCL_FORCE_INLINE const T* as() const noexcept {
        return reinterpret_cast<const T*>(page_->data);
    }

    template <typename T>
    SCL_NODISCARD SCL_FORCE_INLINE T* as() noexcept {
        return reinterpret_cast<T*>(page_->data);
    }

    // -------------------------------------------------------------------------
    // State Queries
    // -------------------------------------------------------------------------

    SCL_NODISCARD SCL_FORCE_INLINE std::size_t page_idx() const noexcept { return page_idx_; }
    SCL_NODISCARD SCL_FORCE_INLINE bool valid() const noexcept { return page_ != nullptr; }
    SCL_FORCE_INLINE explicit operator bool() const noexcept { return valid(); }
};

} // namespace scl::mmap
