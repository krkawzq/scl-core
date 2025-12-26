
#pragma once

#include "scl/mmap/configuration.hpp"
#include "scl/mmap/table.hpp"

#include <cstddef>
#include <vector>
#include <functional>
#include <list>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <atomic>
#include <stdexcept>

namespace scl::mmap {

// =============================================================================
// PagePool: 固定大小页面池 (Thread-Safe)
// =============================================================================

class PagePool {
    std::vector<std::unique_ptr<Page>> pages_;    // 所有页面存储 (Unique ownership)
    std::vector<Page*> free_list_;                // 空闲页面列表
    std::size_t capacity_;
    mutable std::mutex mutex_;

public:
    explicit PagePool(std::size_t num_pages, std::size_t /* page_size */ = SCL_MMAP_PAGE_SIZE)
        : capacity_(num_pages)
    {
        pages_.reserve(num_pages);
        free_list_.reserve(num_pages);
        for (std::size_t i = 0; i < num_pages; ++i) {
            // Allocate 64-byte aligned page
            auto p = std::make_unique<Page>(); 
            free_list_.push_back(p.get());
            pages_.push_back(std::move(p));
        }
    }

    // 禁止拷贝移动 (Pool 是资源持有者)
    PagePool(const PagePool&) = delete;
    PagePool& operator=(const PagePool&) = delete;

    /// @brief 分配一个空闲页 (O(1))
    Page* alloc() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_list_.empty()) return nullptr;
        Page* p = free_list_.back();
        free_list_.pop_back();
        return p;
    }

    /// @brief 回收页面 (O(1))
    void free(Page* p) {
        if (!p) return;
        std::lock_guard<std::mutex> lock(mutex_);
        free_list_.push_back(p);
    }

    /// @brief 获取当前空闲页数
    std::size_t free_count() const { 
        std::lock_guard<std::mutex> lock(mutex_);
        return free_list_.size(); 
    }
    
    /// @brief 获取总容量
    std::size_t capacity() const { return capacity_; }
};

// =============================================================================
// LoadFunc: 加载回调
// =============================================================================

/// @brief 数据加载回调: (page_idx, dest_buffer) -> void
/// @note 用户负责实现具体的 IO 逻辑 (pread, mmap copy, unzip, network fetch...)
using LoadFunc = std::function<void(std::size_t, std::byte*)>;

// =============================================================================
// Scheduler: 通用调度器基类 (Abstract Base)
// =============================================================================

class Scheduler {
protected:
    PageTable* table_;
    PagePool* pool_;
    LoadFunc load_func_;
    
    // 保护 LRU 状态和 Eviction 逻辑
    // 子类如果需要实现复杂原子操作，可以 lock 这个 mutex
    mutable std::mutex mutex_; 

    // LRU Tracking (O(1) access)
    // List: page_idx ordered by usage (Back = MRU, Front = LRU)
    std::list<std::size_t> lru_list_;
    // Map: page_idx -> List Iterator
    std::unordered_map<std::size_t, std::list<std::size_t>::iterator> lru_map_;

public:
    Scheduler(PageTable* table, PagePool* pool, LoadFunc load_func)
        : table_(table), pool_(pool), load_func_(std::move(load_func)) {}

    virtual ~Scheduler() = default;

    // -------------------------------------------------------------------------
    // 核心接口 (需重载)
    // -------------------------------------------------------------------------

    /// @brief 请求页面核心入口
    /// @param page_idx 逻辑页号
    /// @return PageHandle (RAII pin wrapper)
    /// @throw std::runtime_error 如果 OOM
    virtual PageHandle request(std::size_t page_idx) = 0;
    
    /// @brief 预取提示 (可选重载)
    /// @param page_idx 起始页号
    /// @param count 连续页数
    virtual void prefetch(std::size_t /* page_idx */, std::size_t /* count */) {}

protected:
    // -------------------------------------------------------------------------
    // 基础原语 (供子类使用)
    // -------------------------------------------------------------------------

    /// @brief 标记页面被访问 (更新 LRU 链表)
    /// @note 线程安全
    void touch_lru(std::size_t page_idx) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = lru_map_.find(page_idx);
        if (it != lru_map_.end()) {
            // Move to back (MRU)
            lru_list_.splice(lru_list_.end(), lru_list_, it->second);
        } else {
            // Add new
            lru_list_.push_back(page_idx);
            lru_map_[page_idx] = std::prev(lru_list_.end());
        }
        // Notify table for timestamp update (optional)
        table_->touch(page_idx);
    }

    /// @brief 移除 LRU 记录 (当页面被手动释放或错误回滚时调用)
    void remove_lru(std::size_t page_idx) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = lru_map_.find(page_idx);
        if (it != lru_map_.end()) {
            lru_list_.erase(it->second);
            lru_map_.erase(it);
        }
    }

    /// @brief 执行 LRU 驱逐策略
    /// @return 被驱逐的 Page* (已从 Table 卸载)，如果无页可驱逐则返回 nullptr
    Page* evict_lru() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Find victim from front (LRU)
        for (auto it = lru_list_.begin(); it != lru_list_.end(); ) {
            std::size_t victim_idx = *it;
            
            // Critical Check: Is it pinned?
            // If pinned (being used by another thread), we cannot evict.
            if (table_->entry(victim_idx).is_pinned()) {
                ++it; // Skip pinned pages
                continue;
            }

            // Perform Eviction
            Page* p = table_->uninstall(victim_idx);
            
            // Cleanup tracking
            lru_map_.erase(victim_idx);
            lru_list_.erase(it);
            
            return p;
        }
        return nullptr; // No evictable page found (All pinned?)
    }

    /// @brief 执行物理加载 (无锁操作)
    /// @note IO 操作通常很慢，不应持有锁
    void perform_load(std::size_t page_idx, Page* page) {
        try {
            load_func_(page_idx, page->data);
            table_->install(page_idx, page);
        } catch (...) {
            // 加载失败，无需回滚 table (因为还未 install)，但需抛出异常
            throw;
        }
    }
};

// =============================================================================
// StandardScheduler: 标准 LRU 调度器
// =============================================================================

class StandardScheduler : public Scheduler {
public:
    using Scheduler::Scheduler;

    PageHandle request(std::size_t page_idx) override {
        // 1. Fast path: 命中缓存
        if (this->table_->is_loaded(page_idx)) {
            this->touch_lru(page_idx);
            return PageHandle(this->table_, page_idx, this->table_->get(page_idx));
        }

        // 2. Slow path: 缺页处理
        Page* page = this->pool_->alloc();

        // 3. 内存已满，执行驱逐
        if (!page) {
            page = this->evict_lru();
            if (page) {
                // 将脏页归还池中重置 (清理数据等)
                // 这里的 free+alloc 模式比直接复用更安全，防止数据污染
                this->pool_->free(page);
                page = this->pool_->alloc();
            }
        }

        // 4. 彻底 OOM (所有页面都被 Pin 住)
        if (!page) {
            throw std::runtime_error("Mmap OOM: All pages are pinned in memory");
        }

        // 5. 加载数据 & 更新状态
        try {
            this->perform_load(page_idx, page);
            this->touch_lru(page_idx);
        } catch (...) {
            this->pool_->free(page); // 异常安全：归还页面
            throw;
        }

        return PageHandle(this->table_, page_idx, page);
    }
};

// =============================================================================
// SequentialScheduler: 顺序扫描优化调度器 (带预取)
// =============================================================================

class SequentialScheduler : public Scheduler {
    std::size_t prefetch_depth_;

public:
    SequentialScheduler(PageTable* table, PagePool* pool, LoadFunc load_func, 
                       std::size_t depth = 4)
        : Scheduler(table, pool, std::move(load_func))
        , prefetch_depth_(depth) {}

    PageHandle request(std::size_t page_idx) override {
        // 1. 尝试触发预取 (Best Effort)
        try_prefetch(page_idx);

        // 2. 标准加载逻辑 (复用 LRU 逻辑)
        // 顺序访问中，刚刚访问过的页面也是 LRU 保护的对象
        if (this->table_->is_loaded(page_idx)) {
            this->touch_lru(page_idx);
            return PageHandle(this->table_, page_idx, this->table_->get(page_idx));
        }

        Page* page = this->pool_->alloc();
        if (!page) {
            // 对于顺序扫描，LRU 策略 (驱逐最老的页) 恰好就是驱逐已经扫描完的页
            // 所以这里直接复用 evict_lru 是完全正确的
            page = this->evict_lru();
            if (page) {
                this->pool_->free(page);
                page = this->pool_->alloc();
            }
        }
        
        if (!page) throw std::runtime_error("Mmap OOM: Sequential scan blocked");

        try {
            this->perform_load(page_idx, page);
            this->touch_lru(page_idx);
        } catch (...) {
            this->pool_->free(page);
            throw;
        }

        return PageHandle(this->table_, page_idx, page);
    }

    // 允许手动触发预取
    void prefetch(std::size_t page_idx, std::size_t count) override {
        for (std::size_t i = 0; i < count; ++i) {
            // 简单的同步预取尝试，不强制
            // 如果已经在内存，就不管了；如果不在，尝试分配
            std::size_t target = page_idx + i;
            if (target >= this->table_->num_pages()) break;
            
            if (!this->table_->is_loaded(target)) {
                 Page* p = this->pool_->alloc();
                 if (p) {
                     this->perform_load(target, p);
                     this->touch_lru(target);
                 } else {
                     break; // Pool full, stop prefetching
                 }
            }
        }
    }

private:
    void try_prefetch(std::size_t current_idx) {
        // 简单的 Look-ahead 预取
        for (std::size_t i = 1; i <= prefetch_depth_; ++i) {
            std::size_t ahead = current_idx + i;
            if (ahead >= this->table_->num_pages()) break;
            
            // 只在有空闲页时预取，绝不为了预取而驱逐现有的页 (防止 Thrashing)
            if (!this->table_->is_loaded(ahead)) {
                Page* p = this->pool_->alloc();
                if (p) {
                    this->perform_load(ahead, p);
                    this->touch_lru(ahead);
                } else {
                    break; // Pool full
                }
            }
        }
    }
};

} // namespace scl::mmap
