#pragma once

#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"
#include "scl/mmap/page.hpp"
#include "scl/mmap/configuration.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <shared_mutex>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

// =============================================================================
// FILE: scl/mmap/scheduler.hpp
// BRIEF: Event-driven prefetch scheduler with pluggable policies
// =============================================================================

namespace scl::mmap {

// =============================================================================
// Forward Declarations
// =============================================================================

class PrefetchScheduler;
class SchedulePolicy;

// =============================================================================
// Time Utilities
// =============================================================================

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::nanoseconds;
using TimePoint = Clock::time_point;

// =============================================================================
// Scheduler Statistics
// =============================================================================

struct SchedulerStats {
    std::atomic<std::size_t> total_fetches{0};
    std::atomic<std::size_t> cache_hits{0};
    std::atomic<std::size_t> cache_misses{0};
    std::atomic<std::size_t> evictions{0};
    std::atomic<std::uint64_t> total_fetch_latency_ns{0};
    
    SCL_NODISCARD double hit_rate() const noexcept {
        std::size_t hits = cache_hits.load(std::memory_order_relaxed);
        std::size_t misses = cache_misses.load(std::memory_order_relaxed);
        std::size_t total = hits + misses;
        if (total == 0) return 1.0;
        return static_cast<double>(hits) / static_cast<double>(total);
    }
    
    SCL_NODISCARD Duration avg_fetch_latency() const noexcept {
        std::size_t fetches = total_fetches.load(std::memory_order_relaxed);
        if (fetches == 0) return Duration{0};
        return Duration{total_fetch_latency_ns.load(std::memory_order_relaxed) / fetches};
    }
    
    void reset() noexcept {
        total_fetches.store(0, std::memory_order_relaxed);
        cache_hits.store(0, std::memory_order_relaxed);
        cache_misses.store(0, std::memory_order_relaxed);
        evictions.store(0, std::memory_order_relaxed);
        total_fetch_latency_ns.store(0, std::memory_order_relaxed);
    }
};

// =============================================================================
// Scheduler State
// =============================================================================

struct SchedulerState {
    std::size_t current_row;
    std::size_t total_rows;
    std::size_t resident_pages;
    std::size_t max_resident;
    std::size_t pending_fetches;
    double hit_rate;
    Duration avg_fetch_latency;
    std::span<const std::size_t> row_lengths;
    
    SCL_NODISCARD bool has_capacity() const noexcept {
        return resident_pages + pending_fetches < max_resident;
    }
    
    SCL_NODISCARD std::size_t available_slots() const noexcept {
        std::size_t used = resident_pages + pending_fetches;
        return (used < max_resident) ? (max_resident - used) : 0;
    }
};

// =============================================================================
// Blocking Mode
// =============================================================================

enum class BlockMode : std::uint8_t {
    SpinWait,
    ConditionWait,
    Hybrid,
    Callback
};

// =============================================================================
// Blocking Primitive
// =============================================================================

class BlockingPrimitive {
private:
    BlockMode mode_;
    std::condition_variable cv_;
    mutable std::mutex mutex_;
    std::atomic<bool> ready_{false};
    
    static constexpr std::size_t kSpinIterations = 1000;

public:
    explicit BlockingPrimitive(BlockMode mode = BlockMode::Hybrid) noexcept
        : mode_(mode) {}
    
    void wait() {
        switch (mode_) {
        case BlockMode::SpinWait:
            spin_wait();
            break;
        case BlockMode::ConditionWait:
            condition_wait();
            break;
        case BlockMode::Hybrid:
            hybrid_wait();
            break;
        case BlockMode::Callback:
            break;
        }
    }
    
    template <typename Predicate>
    void wait_until(Predicate pred) {
        switch (mode_) {
        case BlockMode::SpinWait:
            while (!pred()) {
                spin_pause();
            }
            break;
            
        case BlockMode::ConditionWait:
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [&pred] { return pred(); });
            }
            break;
            
        case BlockMode::Hybrid:
            for (std::size_t i = 0; i < kSpinIterations; ++i) {
                if (pred()) return;
                spin_pause();
            }
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [&pred] { return pred(); });
            }
            break;
            
        case BlockMode::Callback:
            break;
        }
    }
    
    template <typename Predicate, typename Rep, typename Period>
    bool wait_until_for(Predicate pred, std::chrono::duration<Rep, Period> timeout) {
        switch (mode_) {
        case BlockMode::SpinWait:
            {
                auto deadline = Clock::now() + timeout;
                while (!pred()) {
                    if (Clock::now() >= deadline) return false;
                    spin_pause();
                }
                return true;
            }
            
        case BlockMode::ConditionWait:
        case BlockMode::Hybrid:
            {
                std::unique_lock lock(mutex_);
                return cv_.wait_for(lock, timeout, [&pred] { return pred(); });
            }
            
        case BlockMode::Callback:
            return pred();
        }
        return false;
    }
    
    void notify_all() {
        ready_.store(true, std::memory_order_release);
        if (mode_ != BlockMode::SpinWait) {
            std::lock_guard lock(mutex_);
            cv_.notify_all();
        }
    }
    
    void notify_one() {
        ready_.store(true, std::memory_order_release);
        if (mode_ != BlockMode::SpinWait) {
            std::lock_guard lock(mutex_);
            cv_.notify_one();
        }
    }
    
    void reset() noexcept {
        ready_.store(false, std::memory_order_release);
    }
    
    void set_mode(BlockMode mode) noexcept {
        mode_ = mode;
    }

private:
    void spin_wait() {
        while (!ready_.load(std::memory_order_acquire)) {
            spin_pause();
        }
    }
    
    void condition_wait() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { 
            return ready_.load(std::memory_order_relaxed); 
        });
    }
    
    void hybrid_wait() {
        for (std::size_t i = 0; i < kSpinIterations; ++i) {
            if (ready_.load(std::memory_order_acquire)) return;
            spin_pause();
        }
        condition_wait();
    }
    
    static void spin_pause() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#else
        std::this_thread::yield();
#endif
    }
};

// =============================================================================
// Prefetch Task
// =============================================================================

struct PrefetchTask {
    std::size_t page_idx;
    PageStore* store;
    int priority;
    TimePoint submit_time;
    
    bool operator<(const PrefetchTask& other) const noexcept {
        return priority < other.priority;
    }
};

// =============================================================================
// Policy Decision
// =============================================================================

struct PolicyDecision {
    std::vector<std::size_t> pages_to_fetch;
    int priority{0};
    bool should_wait{false};
};

// =============================================================================
// Schedule Policy Interface
// =============================================================================

class SchedulePolicy {
public:
    virtual ~SchedulePolicy() = default;
    
    virtual PolicyDecision decide(const SchedulerState& state) = 0;
    
    virtual void on_fetch_complete(std::size_t page, Duration latency) { 
        (void)page; (void)latency; 
    }
    virtual void on_cache_hit(std::size_t page) { (void)page; }
    virtual void on_cache_miss(std::size_t page) { (void)page; }
    virtual void on_eviction(std::size_t page) { (void)page; }
    virtual void on_row_begin(std::size_t row) { (void)row; }
    virtual void on_row_end(std::size_t row) { (void)row; }
    
    virtual void set_lookahead(std::size_t lookahead) { (void)lookahead; }
    virtual void set_batch_size(std::size_t batch) { (void)batch; }
    
    SCL_NODISCARD virtual const char* name() const noexcept { return "BasePolicy"; }
};

// =============================================================================
// Lookahead Policy
// =============================================================================

class LookaheadPolicy : public SchedulePolicy {
private:
    std::atomic<std::size_t> lookahead_;
    std::atomic<std::size_t> batch_size_;
    std::size_t min_lookahead_;
    std::size_t max_lookahead_;
    Duration target_latency_;
    std::size_t total_pages_;
    
    std::atomic<std::size_t> current_row_{0};
    std::unique_ptr<std::atomic<bool>[]> loaded_flags_;

public:
    explicit LookaheadPolicy(
        std::size_t total_pages,
        std::size_t lookahead = 32,
        std::size_t batch_size = 8,
        Duration target_latency = std::chrono::microseconds(100)
    )
        : lookahead_(lookahead)
        , batch_size_(batch_size)
        , min_lookahead_(8)
        , max_lookahead_(128)
        , target_latency_(target_latency)
        , total_pages_(total_pages)
        , loaded_flags_(std::make_unique<std::atomic<bool>[]>(total_pages))
    {
        for (std::size_t i = 0; i < total_pages; ++i) {
            loaded_flags_[i].store(false, std::memory_order_relaxed);
        }
    }
    
    PolicyDecision decide(const SchedulerState& state) override {
        PolicyDecision d;
        
        if (!state.has_capacity()) {
            d.should_wait = true;
            return d;
        }
        
        std::size_t lookahead = lookahead_.load(std::memory_order_relaxed);
        std::size_t batch_size = batch_size_.load(std::memory_order_relaxed);
        
        std::size_t start = state.current_row;
        std::size_t end = std::min(start + lookahead, state.total_rows);
        std::size_t available = state.available_slots();
        std::size_t batch = std::min(batch_size, available);
        
        d.pages_to_fetch.reserve(batch);
        
        for (std::size_t page = start; page < end && d.pages_to_fetch.size() < batch; ++page) {
            if (page < total_pages_ && !loaded_flags_[page].load(std::memory_order_acquire)) {
                d.pages_to_fetch.push_back(page);
                d.priority = static_cast<int>(lookahead) - static_cast<int>(page - start);
            }
        }
        
        return d;
    }
    
    void on_fetch_complete(std::size_t page, Duration latency) override {
        if (page < total_pages_) {
            loaded_flags_[page].store(true, std::memory_order_release);
        }
        
        if (latency > target_latency_ * 2) {
            std::size_t current = lookahead_.load(std::memory_order_relaxed);
            if (current < max_lookahead_) {
                std::size_t desired = std::min(current + 4, max_lookahead_);
                lookahead_.compare_exchange_weak(current, desired,
                    std::memory_order_relaxed, std::memory_order_relaxed);
            }
        } else if (latency < target_latency_ / 2) {
            std::size_t current = lookahead_.load(std::memory_order_relaxed);
            if (current > min_lookahead_) {
                std::size_t desired = std::max(current - 1, min_lookahead_);
                lookahead_.compare_exchange_weak(current, desired,
                    std::memory_order_relaxed, std::memory_order_relaxed);
            }
        }
    }
    
    void on_eviction(std::size_t page) override {
        if (page < total_pages_) {
            loaded_flags_[page].store(false, std::memory_order_release);
        }
    }
    
    void on_row_begin(std::size_t row) override {
        current_row_.store(row, std::memory_order_release);
    }
    
    void set_lookahead(std::size_t lookahead) override {
        lookahead_.store(std::clamp(lookahead, min_lookahead_, max_lookahead_),
                        std::memory_order_relaxed);
    }
    
    void set_batch_size(std::size_t batch) override {
        batch_size_.store(std::max(batch, std::size_t{1}), std::memory_order_relaxed);
    }
    
    SCL_NODISCARD const char* name() const noexcept override { return "LookaheadPolicy"; }
};

// =============================================================================
// Compute Hooks Interface
// =============================================================================

class ComputeHooks {
public:
    virtual ~ComputeHooks() = default;
    
    virtual void on_computation_begin(std::size_t total_rows) { (void)total_rows; }
    virtual void on_computation_end() {}
    virtual void on_row_begin(std::size_t row) { (void)row; }
    virtual void on_row_end(std::size_t row) { (void)row; }
    virtual void sync_barrier() {}
    virtual bool sync_fence(std::size_t up_to_row, 
                           std::chrono::milliseconds timeout = std::chrono::milliseconds{5000}) {
        (void)up_to_row; (void)timeout;
        return true;
    }
    virtual void hint_access_pattern(std::span<const std::size_t> rows) { (void)rows; }
    virtual void hint_priority_boost(std::size_t row) { (void)row; }
};

// =============================================================================
// Cache Entry
// =============================================================================

struct alignas(64) CacheEntry {
    std::atomic<Page*> page{nullptr};
    std::atomic<std::uint32_t> pin_count{0};
    std::atomic<std::uint8_t> access_bit{0};
    std::atomic<std::uint8_t> load_state{0};
    
    std::mutex load_mutex;
    std::condition_variable load_cv;
    
    static constexpr std::uint8_t STATE_IDLE = 0;
    static constexpr std::uint8_t STATE_LOADING = 1;
    static constexpr std::uint8_t STATE_LOADED = 2;
    
    SCL_NODISCARD bool is_loaded() const noexcept {
        return page.load(std::memory_order_acquire) != nullptr;
    }
    
    SCL_NODISCARD bool is_pinned() const noexcept {
        return pin_count.load(std::memory_order_acquire) > 0;
    }
    
    void pin() noexcept {
        pin_count.fetch_add(1, std::memory_order_acq_rel);
    }
    
    bool unpin() noexcept {
        std::uint32_t prev = pin_count.fetch_sub(1, std::memory_order_acq_rel);
        return prev == 1;
    }
    
    void mark_accessed() noexcept {
        access_bit.store(1, std::memory_order_release);
    }
    
    bool clear_access() noexcept {
        std::uint8_t expected = 1;
        return access_bit.compare_exchange_strong(expected, 0,
            std::memory_order_acq_rel, std::memory_order_acquire);
    }
    
    bool try_acquire_load() noexcept {
        std::uint8_t expected = STATE_IDLE;
        return load_state.compare_exchange_strong(expected, STATE_LOADING,
            std::memory_order_acq_rel, std::memory_order_acquire);
    }
    
    void finish_load() {
        {
            std::lock_guard lock(load_mutex);
            load_state.store(STATE_LOADED, std::memory_order_release);
        }
        load_cv.notify_all();
    }
    
    void abort_load() {
        {
            std::lock_guard lock(load_mutex);
            load_state.store(STATE_IDLE, std::memory_order_release);
        }
        load_cv.notify_all();
    }
    
    enum class WaitResult {
        Loaded,
        Aborted,
        Timeout
    };
    
    WaitResult wait_for_load(std::chrono::milliseconds timeout) {
        std::unique_lock lock(load_mutex);
        bool success = load_cv.wait_for(lock, timeout, [this] {
            std::uint8_t state = load_state.load(std::memory_order_acquire);
            return state == STATE_LOADED || 
                   state == STATE_IDLE ||
                   page.load(std::memory_order_acquire) != nullptr;
        });
        
        if (!success) {
            return WaitResult::Timeout;
        }
        
        if (page.load(std::memory_order_acquire) != nullptr ||
            load_state.load(std::memory_order_acquire) == STATE_LOADED) {
            return WaitResult::Loaded;
        }
        
        return WaitResult::Aborted;
    }
    
    void reset_load_state() noexcept {
        load_state.store(STATE_IDLE, std::memory_order_release);
    }
};

static_assert(sizeof(CacheEntry) <= 256, "CacheEntry too large");

// =============================================================================
// Page Handle (RAII)
// =============================================================================

class PageHandle {
    PrefetchScheduler* scheduler_;
    std::size_t page_idx_;
    Page* page_;
    bool owns_pin_;

public:
    PageHandle() noexcept 
        : scheduler_(nullptr), page_idx_(0), page_(nullptr), owns_pin_(false) {}
    
    PageHandle(PrefetchScheduler* scheduler, std::size_t idx, Page* page) noexcept;
    PageHandle(PrefetchScheduler* scheduler, std::size_t idx, Page* page, bool already_pinned) noexcept;
    
    ~PageHandle();
    
    PageHandle(PageHandle&& other) noexcept
        : scheduler_(other.scheduler_)
        , page_idx_(other.page_idx_)
        , page_(other.page_)
        , owns_pin_(other.owns_pin_)
    {
        other.scheduler_ = nullptr;
        other.page_ = nullptr;
        other.owns_pin_ = false;
    }
    
    PageHandle& operator=(PageHandle&& other) noexcept;
    
    PageHandle(const PageHandle&) = delete;
    PageHandle& operator=(const PageHandle&) = delete;
    
    SCL_NODISCARD const std::byte* data() const noexcept {
        return page_ ? page_->data : nullptr;
    }
    
    SCL_NODISCARD std::byte* data() noexcept {
        return page_ ? page_->data : nullptr;
    }
    
    template <typename T>
    SCL_NODISCARD const T* as() const noexcept {
        return page_ ? page_->as<T>() : nullptr;
    }
    
    template <typename T>
    SCL_NODISCARD T* as() noexcept {
        return page_ ? page_->as<T>() : nullptr;
    }
    
    SCL_NODISCARD bool valid() const noexcept { return page_ != nullptr; }
    SCL_NODISCARD explicit operator bool() const noexcept { return valid(); }
    SCL_NODISCARD std::size_t index() const noexcept { return page_idx_; }
    
    void mark_dirty() noexcept;
};

// =============================================================================
// Prefetch Scheduler
// =============================================================================

class PrefetchScheduler : public ComputeHooks {
private:
    std::unique_ptr<CacheEntry[]> entries_;
    std::size_t num_pages_;
    std::size_t max_resident_;
    std::atomic<std::size_t> current_resident_{0};
    
    std::atomic<std::size_t> clock_hand_{0};
    
    std::unique_ptr<SchedulePolicy> policy_;
    mutable std::shared_mutex policy_lock_;
    
    std::priority_queue<PrefetchTask> task_queue_;
    std::mutex queue_lock_;
    std::condition_variable queue_cv_;
    
    std::atomic<std::size_t> current_row_{0};
    std::atomic<std::size_t> total_rows_{0};
    BlockingPrimitive sync_primitive_;
    
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{false};
    std::size_t num_workers_;
    
    std::vector<PageStore*> stores_;
    std::unordered_map<std::size_t, PageStore*> stores_by_file_id_;
    mutable std::shared_mutex stores_lock_;
    
    SchedulerStats stats_;
    std::atomic<std::size_t> pending_fetches_{0};
    
    static constexpr int kMaxLoadRetries = 3;
    static constexpr auto kLoadTimeout = std::chrono::milliseconds{5000};

public:
    explicit PrefetchScheduler(
        std::size_t num_pages,
        std::size_t max_resident,
        std::size_t num_workers = 0,
        BlockMode block_mode = BlockMode::Hybrid
    )
        : num_pages_(num_pages)
        , max_resident_(std::min(max_resident, num_pages))
        , sync_primitive_(block_mode)
    {
        if (num_pages == 0) {
            throw ValueError("PrefetchScheduler: num_pages cannot be 0");
        }
        if (max_resident == 0) {
            throw ValueError("PrefetchScheduler: max_resident cannot be 0");
        }
        
        entries_ = std::make_unique<CacheEntry[]>(num_pages);
        policy_ = std::make_unique<LookaheadPolicy>(num_pages);
        
        if (num_workers == 0) {
            std::size_t hw = std::thread::hardware_concurrency();
            num_workers_ = std::max(1u, hw / 4);
        } else {
            num_workers_ = std::min(num_workers, std::size_t{16});
        }
        
        running_.store(true, std::memory_order_release);
        start_workers();
    }
    
    ~PrefetchScheduler() {
        stop();
        
        auto& pool = GlobalPagePool::instance();
        for (std::size_t i = 0; i < num_pages_; ++i) {
            Page* page = entries_[i].page.load(std::memory_order_acquire);
            if (page) {
                writeback_if_dirty(i, page);
                pool.release(page);
            }
        }
    }
    
    PrefetchScheduler(const PrefetchScheduler&) = delete;
    PrefetchScheduler& operator=(const PrefetchScheduler&) = delete;
    
    // =========================================================================
    // Store Registration
    // =========================================================================
    
    void register_store(PageStore* store) {
        if (!store) return;
        
        std::unique_lock lock(stores_lock_);
        stores_.push_back(store);
        stores_by_file_id_[store->file_id()] = store;
    }
    
    // =========================================================================
    // Page Access
    // =========================================================================
    
    PageHandle request(std::size_t page_idx, PageStore* store) {
        if (page_idx >= num_pages_ || !store) {
            return PageHandle();
        }
        
        auto& entry = entries_[page_idx];
        
        entry.pin();
        
        Page* page = entry.page.load(std::memory_order_acquire);
        if (page) {
            entry.mark_accessed();
            stats_.cache_hits.fetch_add(1, std::memory_order_relaxed);
            notify_policy_hit(page_idx);
            return PageHandle(this, page_idx, page, true);
        }
        
        entry.unpin();
        
        stats_.cache_misses.fetch_add(1, std::memory_order_relaxed);
        notify_policy_miss(page_idx);
        return perform_load(page_idx, store);
    }
    
    SCL_NODISCARD bool is_loaded(std::size_t page_idx) const noexcept {
        if (page_idx >= num_pages_) return false;
        return entries_[page_idx].is_loaded();
    }
    
    // =========================================================================
    // ComputeHooks Implementation
    // =========================================================================
    
    void on_computation_begin(std::size_t total_rows) override {
        total_rows_.store(total_rows, std::memory_order_release);
        current_row_.store(0, std::memory_order_release);
        stats_.reset();
    }
    
    void on_computation_end() override {
        sync_barrier();
    }
    
    void on_row_begin(std::size_t row) override {
        current_row_.store(row, std::memory_order_release);
        
        {
            std::shared_lock lock(policy_lock_);
            if (policy_) policy_->on_row_begin(row);
        }
        
        schedule_async();
    }
    
    void on_row_end(std::size_t row) override {
        {
            std::shared_lock lock(policy_lock_);
            if (policy_) policy_->on_row_end(row);
        }
        
        sync_primitive_.notify_all();
    }
    
    void sync_barrier() override {
        sync_primitive_.wait_until([this] {
            return pending_fetches_.load(std::memory_order_acquire) == 0;
        });
    }
    
    bool sync_fence(std::size_t up_to_row, 
                   std::chrono::milliseconds timeout = std::chrono::milliseconds{5000}) override {
        return sync_primitive_.wait_until_for([this, up_to_row] {
            for (std::size_t i = 0; i <= up_to_row && i < num_pages_; ++i) {
                auto& entry = entries_[i];
                if (!entry.is_loaded() && 
                    entry.load_state.load(std::memory_order_acquire) == CacheEntry::STATE_LOADING) {
                    return false;
                }
            }
            return true;
        }, timeout);
    }
    
    void hint_access_pattern(std::span<const std::size_t> rows) override {
        PageStore* store = get_default_store();
        if (!store) return;
        
        std::lock_guard lock(queue_lock_);
        for (auto row : rows) {
            if (row < num_pages_ && !entries_[row].is_loaded()) {
                task_queue_.push(PrefetchTask{row, store, 5, Clock::now()});
            }
        }
        queue_cv_.notify_all();
    }
    
    void hint_priority_boost(std::size_t row) override {
        if (row >= num_pages_) return;
        
        PageStore* store = get_default_store();
        if (!store) return;
        
        std::lock_guard lock(queue_lock_);
        task_queue_.push(PrefetchTask{row, store, 100, Clock::now()});
        queue_cv_.notify_one();
    }
    
    // =========================================================================
    // Policy Management
    // =========================================================================
    
    template <typename Policy, typename... Args>
    void set_policy(Args&&... args) {
        auto new_policy = std::make_unique<Policy>(std::forward<Args>(args)...);
        std::unique_lock lock(policy_lock_);
        policy_ = std::move(new_policy);
    }
    
    void set_policy(std::unique_ptr<SchedulePolicy> policy) {
        std::unique_lock lock(policy_lock_);
        policy_ = std::move(policy);
    }
    
    SCL_NODISCARD const char* policy_name() const {
        std::shared_lock lock(policy_lock_);
        return policy_ ? policy_->name() : "None";
    }
    
    // =========================================================================
    // Configuration
    // =========================================================================
    
    void set_block_mode(BlockMode mode) noexcept {
        sync_primitive_.set_mode(mode);
    }
    
    void set_lookahead(std::size_t lookahead) {
        std::shared_lock lock(policy_lock_);
        if (policy_) policy_->set_lookahead(lookahead);
    }
    
    void set_batch_size(std::size_t batch) {
        std::shared_lock lock(policy_lock_);
        if (policy_) policy_->set_batch_size(batch);
    }
    
    // =========================================================================
    // Statistics
    // =========================================================================
    
    SCL_NODISCARD const SchedulerStats& stats() const noexcept { return stats_; }
    
    SCL_NODISCARD std::size_t resident_count() const noexcept {
        return current_resident_.load(std::memory_order_relaxed);
    }
    
    SCL_NODISCARD std::size_t pending_count() const noexcept {
        return pending_fetches_.load(std::memory_order_relaxed);
    }
    
    // =========================================================================
    // Pin/Unpin
    // =========================================================================
    
    void pin(std::size_t idx) noexcept {
        if (idx < num_pages_) {
            entries_[idx].pin();
        }
    }
    
    void unpin(std::size_t idx) noexcept {
        if (idx < num_pages_) {
            entries_[idx].unpin();
        }
    }
    
    void mark_dirty(std::size_t idx) noexcept {
        if (idx < num_pages_) {
            Page* page = entries_[idx].page.load(std::memory_order_acquire);
            if (page) {
                page->dirty.store(true, std::memory_order_release);
            }
        }
    }

private:
    // =========================================================================
    // Helper
    // =========================================================================
    
    PageStore* get_default_store() noexcept {
        std::shared_lock lock(stores_lock_);
        return stores_.empty() ? nullptr : stores_[0];
    }
    
    // =========================================================================
    // Worker Management
    // =========================================================================
    
    void start_workers() {
        for (std::size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&PrefetchScheduler::worker_loop, this);
        }
    }
    
    void stop() {
        running_.store(false, std::memory_order_release);
        queue_cv_.notify_all();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }
    
    void worker_loop() {
        while (running_.load(std::memory_order_acquire)) {
            PrefetchTask task;
            
            {
                std::unique_lock lock(queue_lock_);
                queue_cv_.wait(lock, [this] {
                    return !task_queue_.empty() || 
                           !running_.load(std::memory_order_relaxed);
                });
                
                if (!running_.load(std::memory_order_relaxed)) break;
                if (task_queue_.empty()) continue;
                
                task = task_queue_.top();
                task_queue_.pop();
            }
            
            execute_prefetch(task);
        }
    }
    
    // =========================================================================
    // Prefetch Execution
    // =========================================================================
    
    void execute_prefetch(const PrefetchTask& task) {
        if (task.page_idx >= num_pages_) return;
        
        auto& entry = entries_[task.page_idx];
        
        if (!entry.try_acquire_load()) {
            return;
        }
        
        if (entry.is_loaded()) {
            entry.abort_load();
            return;
        }
        
        pending_fetches_.fetch_add(1, std::memory_order_release);
        auto start_time = Clock::now();
        
        if (!ensure_capacity()) {
            entry.abort_load();
            pending_fetches_.fetch_sub(1, std::memory_order_release);
            return;
        }
        
        auto& pool = GlobalPagePool::instance();
        Page* page = pool.get_or_create(task.store->file_id(), task.page_idx);
        
        if (!page) {
            entry.abort_load();
            pending_fetches_.fetch_sub(1, std::memory_order_release);
            return;
        }
        
        if (page->get_refcount() == 1) {
            task.store->load(task.page_idx, page);
        }
        
        Page* expected = nullptr;
        if (entry.page.compare_exchange_strong(expected, page,
                std::memory_order_release, std::memory_order_acquire)) {
            current_resident_.fetch_add(1, std::memory_order_relaxed);
        } else {
            pool.release(page);
            page = expected;
        }
        
        entry.finish_load();
        pending_fetches_.fetch_sub(1, std::memory_order_release);
        
        auto latency = Clock::now() - start_time;
        stats_.total_fetches.fetch_add(1, std::memory_order_relaxed);
        stats_.total_fetch_latency_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(latency).count(),
            std::memory_order_relaxed
        );
        
        {
            std::shared_lock lock(policy_lock_);
            if (policy_) policy_->on_fetch_complete(task.page_idx, latency);
        }
        
        sync_primitive_.notify_all();
    }
    
    // =========================================================================
    // Synchronous Load
    // =========================================================================
    
    PageHandle perform_load(std::size_t page_idx, PageStore* store) {
        auto& entry = entries_[page_idx];
        
        for (int retry = 0; retry < kMaxLoadRetries; ++retry) {
            if (entry.try_acquire_load()) {
                return do_load(page_idx, store, entry);
            }
            
            auto result = entry.wait_for_load(kLoadTimeout);
            
            switch (result) {
            case CacheEntry::WaitResult::Loaded:
                entry.pin();
                {
                    Page* page = entry.page.load(std::memory_order_acquire);
                    if (page) {
                        return PageHandle(this, page_idx, page, true);
                    }
                }
                entry.unpin();
                break;
                
            case CacheEntry::WaitResult::Aborted:
                break;
                
            case CacheEntry::WaitResult::Timeout:
                throw RuntimeError("PrefetchScheduler: Load timeout");
            }
        }
        
        throw RuntimeError("PrefetchScheduler: Load failed after retries");
    }
    
    PageHandle do_load(std::size_t page_idx, PageStore* store, CacheEntry& entry) {
        if (!ensure_capacity()) {
            entry.abort_load();
            throw RuntimeError("PrefetchScheduler: Cannot evict - all pages pinned");
        }
        
        auto& pool = GlobalPagePool::instance();
        Page* page = pool.get_or_create(store->file_id(), page_idx);
        
        if (!page) {
            entry.abort_load();
            throw RuntimeError("PrefetchScheduler: Failed to allocate page");
        }
        
        if (page->get_refcount() == 1) {
            store->load(page_idx, page);
        }
        
        Page* expected = nullptr;
        if (entry.page.compare_exchange_strong(expected, page,
                std::memory_order_release, std::memory_order_acquire)) {
            current_resident_.fetch_add(1, std::memory_order_relaxed);
        } else {
            pool.release(page);
            page = expected;
        }
        
        entry.pin();
        entry.finish_load();
        entry.mark_accessed();
        
        stats_.total_fetches.fetch_add(1, std::memory_order_relaxed);
        
        return PageHandle(this, page_idx, page, true);
    }
    
    // =========================================================================
    // Scheduling
    // =========================================================================
    
    void schedule_async() {
        SchedulerState state = build_state();
        
        PolicyDecision decision;
        {
            std::shared_lock lock(policy_lock_);
            if (!policy_) return;
            decision = policy_->decide(state);
        }
        
        if (decision.should_wait || decision.pages_to_fetch.empty()) {
            return;
        }
        
        PageStore* store = get_default_store();
        if (!store) return;
        
        {
            std::lock_guard lock(queue_lock_);
            for (auto page_idx : decision.pages_to_fetch) {
                if (page_idx < num_pages_ && !entries_[page_idx].is_loaded()) {
                    task_queue_.push(PrefetchTask{
                        page_idx, store, decision.priority, Clock::now()
                    });
                }
            }
        }
        
        queue_cv_.notify_all();
    }
    
    SCL_NODISCARD SchedulerState build_state() const noexcept {
        std::size_t resident = current_resident_.load(std::memory_order_relaxed);
        std::size_t pending = pending_fetches_.load(std::memory_order_relaxed);
        
        return SchedulerState{
            .current_row = current_row_.load(std::memory_order_acquire),
            .total_rows = total_rows_.load(std::memory_order_acquire),
            .resident_pages = resident,
            .max_resident = max_resident_,
            .pending_fetches = pending,
            .hit_rate = stats_.hit_rate(),
            .avg_fetch_latency = stats_.avg_fetch_latency(),
            .row_lengths = {}
        };
    }
    
    // =========================================================================
    // Eviction
    // =========================================================================
    
    bool ensure_capacity() {
        std::size_t resident = current_resident_.load(std::memory_order_relaxed);
        std::size_t pending = pending_fetches_.load(std::memory_order_relaxed);
        
        if (resident + pending < max_resident_) {
            return true;
        }
        
        for (std::size_t attempt = 0; attempt < 3; ++attempt) {
            std::size_t victim = select_victim();
            if (victim != SIZE_MAX && evict_page(victim)) {
                return true;
            }
        }
        
        return false;
    }
    
    SCL_NODISCARD std::size_t select_victim() noexcept {
        for (std::size_t iter = 0; iter < num_pages_ * 2; ++iter) {
            std::size_t idx = clock_hand_.fetch_add(1, std::memory_order_relaxed) % num_pages_;
            auto& entry = entries_[idx];
            
            if (!entry.is_loaded()) continue;
            if (entry.is_pinned()) continue;
            
            if (!entry.clear_access()) {
                return idx;
            }
        }
        
        return SIZE_MAX;
    }
    
    bool evict_page(std::size_t page_idx) {
        auto& entry = entries_[page_idx];
        
        entry.pin();
        
        if (entry.pin_count.load(std::memory_order_acquire) > 1) {
            entry.unpin();
            return false;
        }
        
        Page* page = entry.page.exchange(nullptr, std::memory_order_acq_rel);
        if (!page) {
            entry.unpin();
            return false;
        }
        
        entry.reset_load_state();
        entry.unpin();
        
        writeback_if_dirty(page_idx, page);
        
        GlobalPagePool::instance().release(page);
        current_resident_.fetch_sub(1, std::memory_order_relaxed);
        stats_.evictions.fetch_add(1, std::memory_order_relaxed);
        
        {
            std::shared_lock lock(policy_lock_);
            if (policy_) policy_->on_eviction(page_idx);
        }
        
        return true;
    }
    
    void writeback_if_dirty(std::size_t page_idx, Page* page) noexcept {
        if (!page || !page->dirty.load(std::memory_order_acquire)) {
            return;
        }
        
        std::shared_lock lock(stores_lock_);
        auto it = stores_by_file_id_.find(page->file_id);
        if (it != stores_by_file_id_.end() && it->second) {
            it->second->write(page->page_offset, page);
            page->dirty.store(false, std::memory_order_release);
        }
    }
    
    // =========================================================================
    // Policy Notifications
    // =========================================================================
    
    void notify_policy_hit(std::size_t page) noexcept {
        std::shared_lock lock(policy_lock_);
        if (policy_) policy_->on_cache_hit(page);
    }
    
    void notify_policy_miss(std::size_t page) noexcept {
        std::shared_lock lock(policy_lock_);
        if (policy_) policy_->on_cache_miss(page);
    }
};

// =============================================================================
// PageHandle Implementation
// =============================================================================

inline PageHandle::PageHandle(PrefetchScheduler* scheduler, std::size_t idx, Page* page) noexcept
    : scheduler_(scheduler), page_idx_(idx), page_(page), owns_pin_(true)
{
    if (scheduler_ && page_) {
        scheduler_->pin(page_idx_);
    }
}

inline PageHandle::PageHandle(PrefetchScheduler* scheduler, std::size_t idx, Page* page, 
                              bool already_pinned) noexcept
    : scheduler_(scheduler), page_idx_(idx), page_(page), owns_pin_(!already_pinned)
{
    if (scheduler_ && page_ && owns_pin_) {
        scheduler_->pin(page_idx_);
    }
}

inline PageHandle::~PageHandle() {
    if (scheduler_ && page_) {
        scheduler_->unpin(page_idx_);
    }
}

inline PageHandle& PageHandle::operator=(PageHandle&& other) noexcept {
    if (this != &other) {
        if (scheduler_ && page_) {
            scheduler_->unpin(page_idx_);
        }
        scheduler_ = other.scheduler_;
        page_idx_ = other.page_idx_;
        page_ = other.page_;
        owns_pin_ = other.owns_pin_;
        other.scheduler_ = nullptr;
        other.page_ = nullptr;
        other.owns_pin_ = false;
    }
    return *this;
}

inline void PageHandle::mark_dirty() noexcept {
    if (scheduler_) {
        scheduler_->mark_dirty(page_idx_);
    }
}

// =============================================================================
// Factory
// =============================================================================

inline std::shared_ptr<PrefetchScheduler> make_scheduler(
    std::size_t num_pages,
    std::size_t max_resident = 64,
    std::size_t num_workers = 0,
    BlockMode block_mode = BlockMode::Hybrid
) {
    return std::make_shared<PrefetchScheduler>(
        num_pages, max_resident, num_workers, block_mode
    );
}

} // namespace scl::mmap

