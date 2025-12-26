#pragma once

#include "scl/core/macros.hpp"
#include "scl/mmap/configuration.hpp"

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <utility>
#include <vector>

namespace scl::mmap {

// =============================================================================
// Page: 固定大小内存块 (Aligned for Cache Line / SIMD)
// =============================================================================

struct alignas(64) Page {
    std::byte data[SCL_MMAP_PAGE_SIZE];
};

static_assert(sizeof(Page) == SCL_MMAP_PAGE_SIZE, "Page size mismatch");

// =============================================================================
// PageEntry: 页表项 (轻量级，原子化)
// =============================================================================

struct PageEntry {
    // 使用原子指针，支持无锁读取检查
    std::atomic<Page*> ptr {nullptr};

    // 引用计数，>0 时禁止驱逐
    std::atomic<uint32_t> pin_count {0};

    // 脏页标记 (可选，用于回写策略)
    std::atomic<bool> dirty {false};

    // 默认构造
    PageEntry() = default;

    // 原子读取辅助
    Page* get() const { return ptr.load(std::memory_order_acquire); }
    bool is_loaded() const { return get() != nullptr; }
    bool is_pinned() const { return pin_count.load(std::memory_order_relaxed) > 0; }
};

// =============================================================================
// PageTable: 虚拟连续内存映射表
// =============================================================================

class PageTable {
public:
    static constexpr std::size_t page_size = SCL_MMAP_PAGE_SIZE;

private:
    // 使用 vector 管理内存，比 new[] 更安全且方便
    // PageEntry 很小 (16B)，连续存储对 CPU Cache 友好
    std::vector<PageEntry> entries_;
    std::size_t num_pages_;
    std::size_t total_bytes_;

public:
    // -------------------------------------------------------------------------
    // 构造/析构
    // -------------------------------------------------------------------------

    explicit PageTable(std::size_t total_bytes)
        : total_bytes_(total_bytes)
    {
        num_pages_ = (total_bytes + page_size - 1) / page_size;
        entries_.resize(num_pages_);
    }

    ~PageTable() = default;

    // Non-copyable, Non-movable (Table usually has long lifecycle)
    PageTable(const PageTable&) = delete;
    PageTable& operator=(const PageTable&) = delete;

    // -------------------------------------------------------------------------
    // 地址转换 (Hot Path - Inline)
    // -------------------------------------------------------------------------

    SCL_NODISCARD SCL_FORCE_INLINE std::size_t to_page_idx(std::size_t byte_offset) const {
        return byte_offset / page_size;
    }

    SCL_NODISCARD SCL_FORCE_INLINE std::size_t to_page_offset(std::size_t byte_offset) const {
        return byte_offset % page_size;
    }

    // -------------------------------------------------------------------------
    // 查询 (Lock-Free Read)
    // -------------------------------------------------------------------------

    SCL_NODISCARD SCL_FORCE_INLINE bool is_loaded(std::size_t page_idx) const {
        // Relaxed check is usually enough for "hint", acquire for actual usage
        return entries_[page_idx].ptr.load(std::memory_order_relaxed) != nullptr;
    }

    SCL_NODISCARD SCL_FORCE_INLINE bool is_pinned(std::size_t page_idx) const {
        return entries_[page_idx].is_pinned();
    }

    SCL_NODISCARD SCL_FORCE_INLINE Page* get(std::size_t page_idx) const {
        return entries_[page_idx].get();
    }

    SCL_NODISCARD SCL_FORCE_INLINE PageEntry& entry(std::size_t page_idx) {
        return entries_[page_idx];
    }

    // -------------------------------------------------------------------------
    // 页面管理 (供 Scheduler 调用)
    // -------------------------------------------------------------------------

    void install(std::size_t page_idx, Page* page) {
        // Release 语义保证 data 写入完成后指针才可见
        entries_[page_idx].ptr.store(page, std::memory_order_release);
    }

    Page* uninstall(std::size_t page_idx) {
        // Acquire 语义保证看到最新的 ptr
        Page* old = entries_[page_idx].ptr.exchange(nullptr, std::memory_order_acquire);
        return old;
    }

    // -------------------------------------------------------------------------
    // Pin/Unpin (Atomic)
    // -------------------------------------------------------------------------

    void pin(std::size_t page_idx) {
        // Relaxed ordering is usually sufficient for increments
        // as long as eviction checks correct count
        entries_[page_idx].pin_count.fetch_add(1, std::memory_order_relaxed);
    }

    void unpin(std::size_t page_idx) {
        entries_[page_idx].pin_count.fetch_sub(1, std::memory_order_relaxed);
    }

    // 通知访问 (用于更新 Scheduler 的 LRU)
    // 这里不再内部维护时钟，而是提供 hook 给外部
    void touch(std::size_t page_idx) {
        // Placeholder: Scheduler usually calls this explicitly via its own tracking
    }

    // -------------------------------------------------------------------------
    // 属性
    // -------------------------------------------------------------------------

    std::size_t num_pages() const { return num_pages_; }
    std::size_t total_bytes() const { return total_bytes_; }
};

// =============================================================================
// PageHandle: RAII Pin 管理 (Move-Only)
// =============================================================================

class PageHandle {
    PageTable* table_;
    std::size_t page_idx_;
    Page* page_;

public:
    PageHandle() : table_(nullptr), page_idx_(0), page_(nullptr) {}

    PageHandle(PageTable* table, std::size_t page_idx, Page* page)
        : table_(table), page_idx_(page_idx), page_(page)
    {
        if (table_) table_->pin(page_idx_);
    }

    ~PageHandle() {
        if (table_) table_->unpin(page_idx_);
    }

    // Move Constructor
    PageHandle(PageHandle&& o) noexcept
        : table_(o.table_), page_idx_(o.page_idx_), page_(o.page_)
    {
        o.table_ = nullptr;
        o.page_ = nullptr;
    }

    // Move Assignment
    PageHandle& operator=(PageHandle&& o) noexcept {
        if (this != &o) {
            // Unpin current if holding one
            if (table_) table_->unpin(page_idx_);

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

    // Delete Copy
    PageHandle(const PageHandle&) = delete;
    PageHandle& operator=(const PageHandle&) = delete;

    // 访问接口
    SCL_NODISCARD const std::byte* data() const { return page_->data; }
    SCL_NODISCARD std::byte* data() { return page_->data; }

    // 类型化访问辅助
    template <typename T>
    SCL_NODISCARD const T* as() const { return reinterpret_cast<const T*>(page_->data); }

    template <typename T>
    SCL_NODISCARD T* as() { return reinterpret_cast<T*>(page_->data); }

    std::size_t page_idx() const { return page_idx_; }
    bool valid() const { return page_ != nullptr; }
    explicit operator bool() const { return valid(); }
};

} // namespace scl::mmap
