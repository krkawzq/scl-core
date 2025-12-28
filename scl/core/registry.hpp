#pragma once

#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/error.hpp"
#include "scl/core/algo.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// =============================================================================
// FILE: scl/core/registry.hpp
// BRIEF: Unified high-performance memory registry with three-layer reference counting
//
// ARCHITECTURE:
//   Layer 1: Buffer  - Real memory block with alias_count
//   Layer 2: Alias   - Access pointer with ref_count (can be shared by multiple instances)
//   Layer 3: Instance - Matrix objects that hold aliases
//
// KEY FEATURES:
//   - Sharded reference counting for lock-free concurrent access
//   - Multiple instances can share the same alias (no need for is_view_ flag)
//   - Zero-copy slicing via alias sharing
// =============================================================================

namespace scl {

using BufferID = std::uint64_t;

// =============================================================================
// Forward Declarations
// =============================================================================

class Registry;
Registry& get_registry();

// =============================================================================
// Allocation Types
// =============================================================================

enum class AllocType : std::uint8_t {
    ArrayNew = 0,
    ScalarNew = 1,
    AlignedAlloc = 2,
    Custom = 3
};

// =============================================================================
// Thread Shard Index
// =============================================================================

namespace detail {

// Thread-local shard index for lock-free access
class ThreadShardIndex {
    struct alignas(64) Counter {
        std::atomic<std::size_t> value;
        Counter() : value(0) {}
    };
    
    static inline Counter next_index_;
    
public:
    static std::size_t get() noexcept {
        thread_local std::size_t cached = next_index_.value.fetch_add(1, 
            std::memory_order_relaxed);
        return cached;
    }
};

inline std::size_t get_thread_shard_index() noexcept {
    return ThreadShardIndex::get();
}

} // namespace detail

// =============================================================================
// Sharded Reference Count - Lock-Free Concurrent Reference Counting
// =============================================================================

class ShardedRefCount {
public:
    static constexpr std::size_t MAX_SHARDS = 16;
    static constexpr std::int32_t BORROW_THRESHOLD = 8;
    
private:
    // Cache-line aligned shard to prevent false sharing
    struct alignas(64) Shard {
        std::atomic<std::int32_t> count{0};
    };
    
    std::atomic<std::int32_t> base_;
    Shard shards_[MAX_SHARDS];
    std::size_t num_shards_;
    
public:
    explicit ShardedRefCount(std::size_t num_shards = 4, std::int32_t initial = 1) noexcept
        : base_(initial)
        , num_shards_(num_shards > MAX_SHARDS ? MAX_SHARDS : (num_shards < 1 ? 1 : num_shards))
    {}
    
    ShardedRefCount(const ShardedRefCount&) = delete;
    ShardedRefCount& operator=(const ShardedRefCount&) = delete;
    
    ShardedRefCount(ShardedRefCount&& other) noexcept
        : base_(other.base_.load(std::memory_order_relaxed))
        , num_shards_(other.num_shards_)
    {
        for (std::size_t i = 0; i < num_shards_; ++i) {
            shards_[i].count.store(
                other.shards_[i].count.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
        }
        other.base_.store(0, std::memory_order_relaxed);
        for (std::size_t i = 0; i < other.num_shards_; ++i) {
            other.shards_[i].count.store(0, std::memory_order_relaxed);
        }
    }
    
    // Fast path: increment current thread's shard (lock-free, no contention)
    SCL_FORCE_INLINE void incref() noexcept {
        std::size_t shard_idx = detail::get_thread_shard_index() % num_shards_;
        shards_[shard_idx].count.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Increment by specified amount
    SCL_FORCE_INLINE void incref(std::int32_t amount) noexcept {
        if (SCL_UNLIKELY(amount <= 0)) return;
        std::size_t shard_idx = detail::get_thread_shard_index() % num_shards_;
        shards_[shard_idx].count.fetch_add(amount, std::memory_order_relaxed);
    }
    
    // Fast path: decrement current thread's shard
    // Returns true if total count reached zero (caller should free)
    SCL_FORCE_INLINE bool decref() noexcept {
        std::size_t shard_idx = detail::get_thread_shard_index() % num_shards_;
        auto& shard = shards_[shard_idx];
        
        // Try to decrement local shard
        std::int32_t old_local = shard.count.load(std::memory_order_relaxed);
        
        if (SCL_LIKELY(old_local > 0)) {
            // Fast path: local shard has count
            shard.count.fetch_sub(1, std::memory_order_acq_rel);
            return false;
        }
        
        // Slow path: need to borrow from base
        return decref_slow_path(shard_idx);
    }
    
    // Decrement by specified amount
    // Returns true if total count reached zero
    SCL_FORCE_INLINE bool decref(std::int32_t amount) noexcept {
        if (SCL_UNLIKELY(amount <= 0)) return false;
        
        std::size_t shard_idx = detail::get_thread_shard_index() % num_shards_;
        auto& shard = shards_[shard_idx];
        
        std::int32_t old_local = shard.count.load(std::memory_order_relaxed);
        
        if (SCL_LIKELY(old_local >= amount)) {
            shard.count.fetch_sub(amount, std::memory_order_acq_rel);
            return false;
        }
        
        // Need to handle across base and shards
        return decref_slow_path_amount(amount);
    }
    
    // Get exact total count (may be slow, requires aggregation)
    SCL_NODISCARD std::int32_t get_count() const noexcept {
        std::atomic_thread_fence(std::memory_order_acquire);
        
        std::int32_t total = base_.load(std::memory_order_relaxed);
        for (std::size_t i = 0; i < num_shards_; ++i) {
            total += shards_[i].count.load(std::memory_order_relaxed);
        }
        
        return total;
    }
    
    // Fast heuristic: is this likely the only reference?
    // May return false negative (safe for optimization decisions)
    SCL_NODISCARD SCL_FORCE_INLINE bool is_likely_unique() const noexcept {
        std::int32_t base = base_.load(std::memory_order_relaxed);
        if (SCL_UNLIKELY(base > 1)) return false;
        if (SCL_UNLIKELY(base < 0)) return false;  // Negative means borrowed, likely not unique
        
        std::size_t shard_idx = detail::get_thread_shard_index() % num_shards_;
        std::int32_t local = shards_[shard_idx].count.load(std::memory_order_relaxed);
        
        return SCL_LIKELY(base + local == 1);
    }
    
    // Precise check: exactly one reference?
    SCL_NODISCARD SCL_FORCE_INLINE bool is_unique() const noexcept {
        return SCL_LIKELY(get_count() == 1);
    }
    
    // Consolidate shards into base (for cleanup or precise operations)
    void consolidate() noexcept {
        std::int32_t total = 0;
        for (std::size_t i = 0; i < num_shards_; ++i) {
            total += shards_[i].count.exchange(0, std::memory_order_acq_rel);
        }
        base_.fetch_add(total, std::memory_order_release);
    }
    
private:
    SCL_FORCE_INLINE bool decref_slow_path(std::size_t shard_idx) noexcept {
        // Try to borrow from base
        std::int32_t old_base = base_.load(std::memory_order_acquire);
        
        while (SCL_LIKELY(old_base > 0)) {
            std::int32_t borrow = (old_base > BORROW_THRESHOLD) ? BORROW_THRESHOLD : old_base;
            if (SCL_LIKELY(base_.compare_exchange_weak(old_base, old_base - borrow,
                                            std::memory_order_acq_rel,
                                            std::memory_order_relaxed))) {
                // Successfully borrowed, add remainder to local shard
                if (SCL_LIKELY(borrow > 1)) {
                    shards_[shard_idx].count.fetch_add(borrow - 1, std::memory_order_relaxed);
                }
                return false;
            }
        }
        
        // Base is zero or negative, check total
        return SCL_UNLIKELY(get_count() <= 0);
    }
    
    bool decref_slow_path_amount(std::int32_t amount) noexcept {
        // Consolidate first for accurate operation
        consolidate();
        
        std::int32_t old_base = base_.fetch_sub(amount, std::memory_order_acq_rel);
        return SCL_UNLIKELY(old_base <= amount);
    }
};

// =============================================================================
// Open-Addressing Hash Table
// =============================================================================

namespace detail {

template <typename K, typename V>
class alignas(64) ConcurrentFlatMap {
    static constexpr std::size_t kInitialCapacity = 256;
    static constexpr double kMaxLoadFactor = 0.7;
    static constexpr std::size_t kSlotsPerStripe = 16;
    
    SCL_FORCE_INLINE static constexpr K empty_key() noexcept {
        if constexpr (std::is_pointer_v<K>) {
            return nullptr;
        } else {
            return K{0};
        }
    }
    
    SCL_FORCE_INLINE static K tombstone_key() noexcept {
        if constexpr (std::is_pointer_v<K>) {
            return reinterpret_cast<K>(static_cast<std::uintptr_t>(1));
        } else {
            return K{1};
        }
    }

    struct Slot {
        std::atomic<K> key;
        V value{};
        
        Slot() : key(empty_key()) {}
        
        Slot(Slot&& other) noexcept
            : key(other.key.load(std::memory_order_relaxed))
            , value(std::move(other.value)) {
            other.key.store(empty_key(), std::memory_order_relaxed);
        }
        
        Slot& operator=(Slot&& other) noexcept {
            if (this != &other) {
                key.store(other.key.load(std::memory_order_relaxed), 
                         std::memory_order_relaxed);
                value = std::move(other.value);
                other.key.store(empty_key(), std::memory_order_relaxed);
            }
            return *this;
        }
        
        Slot(const Slot&) = delete;
        Slot& operator=(const Slot&) = delete;
    };

    std::vector<Slot> slots_;
    std::atomic<std::size_t> size_{0};
    std::size_t capacity_;
    mutable std::shared_mutex rehash_mutex_;
    mutable std::vector<std::unique_ptr<std::mutex>> stripe_mutexes_;
    
    SCL_FORCE_INLINE std::size_t hash(K key) const noexcept {
        return std::hash<K>{}(key);
    }
    
    SCL_FORCE_INLINE std::size_t probe(std::size_t h, std::size_t i) const noexcept {
        return (h + i) & (capacity_ - 1);
    }
    
    SCL_FORCE_INLINE std::mutex& get_stripe_mutex(std::size_t idx) const {
        std::size_t stripe_idx = idx / kSlotsPerStripe;
        return *stripe_mutexes_[stripe_idx % stripe_mutexes_.size()];
    }
    
    void init_stripe_mutexes(std::size_t capacity) {
        std::size_t num_stripes = (capacity + kSlotsPerStripe - 1) / kSlotsPerStripe;
        stripe_mutexes_.clear();
        stripe_mutexes_.reserve(num_stripes);
        for (std::size_t i = 0; i < num_stripes; ++i) {
            stripe_mutexes_.push_back(std::make_unique<std::mutex>());
        }
    }
    
    void rehash_internal(std::size_t new_cap) {
        std::vector<Slot> old_slots = std::move(slots_);
        capacity_ = new_cap;
        slots_.clear();
        slots_.resize(new_cap);
        init_stripe_mutexes(new_cap);
        size_.store(0, std::memory_order_relaxed);
        
        for (auto& slot : old_slots) {
            K k = slot.key.load(std::memory_order_relaxed);
            if (k != empty_key() && k != tombstone_key()) {
                insert_internal_unlocked(k, std::move(slot.value));
            }
        }
    }
    
    bool insert_internal_unlocked(K key, V value) {
        std::size_t h = hash(key);
        for (std::size_t i = 0; i < capacity_; ++i) {
            std::size_t idx = probe(h, i);
            K expected = slots_[idx].key.load(std::memory_order_relaxed);
            
            if (expected == empty_key() || expected == tombstone_key()) {
                slots_[idx].value = std::move(value);
                slots_[idx].key.store(key, std::memory_order_release);
                size_.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            if (expected == key) {
                return false;
            }
        }
        return false;
    }

public:
    ConcurrentFlatMap() : capacity_(kInitialCapacity) {
        slots_.resize(capacity_);
        init_stripe_mutexes(capacity_);
    }
    
    bool insert(K key, V value) {
        if (key == empty_key() || key == tombstone_key()) return false;
        
        std::unique_lock lock(rehash_mutex_);
        
        if (size_.load(std::memory_order_relaxed) > capacity_ * kMaxLoadFactor) {
            rehash_internal(capacity_ * 2);
        }
        
        return insert_internal_unlocked(key, std::move(value));
    }
    
    template <typename U = V, 
              typename = std::enable_if_t<std::is_copy_assignable_v<U>>>
    bool find(K key, V& out) const {
        if (key == empty_key() || key == tombstone_key()) return false;
        
        std::shared_lock lock(rehash_mutex_);
        std::size_t h = hash(key);
        
        for (std::size_t i = 0; i < capacity_; ++i) {
            std::size_t idx = probe(h, i);
            K k = slots_[idx].key.load(std::memory_order_acquire);
            
            if (k == key) {
                std::lock_guard stripe_lock(get_stripe_mutex(idx));
                if (slots_[idx].key.load(std::memory_order_acquire) == key) {
                    out = slots_[idx].value;
                    return true;
                }
                return false;
            }
            if (k == empty_key()) {
                return false;
            }
        }
        return false;
    }
    
    bool contains(K key) const {
        if (key == empty_key() || key == tombstone_key()) return false;
        
        std::shared_lock lock(rehash_mutex_);
        std::size_t h = hash(key);
        
        for (std::size_t i = 0; i < capacity_; ++i) {
            std::size_t idx = probe(h, i);
            K k = slots_[idx].key.load(std::memory_order_acquire);
            
            if (k == key) return true;
            if (k == empty_key()) return false;
        }
        return false;
    }
    
    bool erase(K key) {
        if (key == empty_key() || key == tombstone_key()) return false;
        
        std::unique_lock lock(rehash_mutex_);
        std::size_t h = hash(key);
        
        for (std::size_t i = 0; i < capacity_; ++i) {
            std::size_t idx = probe(h, i);
            K k = slots_[idx].key.load(std::memory_order_acquire);
            
            if (k == key) {
                slots_[idx].key.store(tombstone_key(), std::memory_order_release);
                slots_[idx].value = V{};
                size_.fetch_sub(1, std::memory_order_relaxed);
                return true;
            }
            if (k == empty_key()) {
                return false;
            }
        }
        return false;
    }
    
    bool erase_and_get(K key, V& out) {
        if (key == empty_key() || key == tombstone_key()) return false;
        
        std::unique_lock lock(rehash_mutex_);
        std::size_t h = hash(key);
        
        for (std::size_t i = 0; i < capacity_; ++i) {
            std::size_t idx = probe(h, i);
            K k = slots_[idx].key.load(std::memory_order_acquire);
            
            if (k == key) {
                out = std::move(slots_[idx].value);
                slots_[idx].key.store(tombstone_key(), std::memory_order_release);
                slots_[idx].value = V{};
                size_.fetch_sub(1, std::memory_order_relaxed);
                return true;
            }
            if (k == empty_key()) {
                return false;
            }
        }
        return false;
    }
    
    template <typename Func>
    bool access(K key, Func&& func) {
        if (key == empty_key() || key == tombstone_key()) return false;
        
        std::shared_lock lock(rehash_mutex_);
        std::size_t h = hash(key);
        
        for (std::size_t i = 0; i < capacity_; ++i) {
            std::size_t idx = probe(h, i);
            K k = slots_[idx].key.load(std::memory_order_acquire);
            
            if (k == key) {
                std::lock_guard stripe_lock(get_stripe_mutex(idx));
                if (slots_[idx].key.load(std::memory_order_acquire) == key) {
                    func(slots_[idx].value);
                    return true;
                }
                return false;
            }
            if (k == empty_key()) {
                return false;
            }
        }
        return false;
    }
    
    template <typename Func>
    bool access(K key, Func&& func) const {
        if (key == empty_key() || key == tombstone_key()) return false;
        
        std::shared_lock lock(rehash_mutex_);
        std::size_t h = hash(key);
        
        for (std::size_t i = 0; i < capacity_; ++i) {
            std::size_t idx = probe(h, i);
            K k = slots_[idx].key.load(std::memory_order_acquire);
            
            if (k == key) {
                std::lock_guard stripe_lock(get_stripe_mutex(idx));
                if (slots_[idx].key.load(std::memory_order_acquire) == key) {
                    func(std::as_const(slots_[idx].value));
                    return true;
                }
                return false;
            }
            if (k == empty_key()) {
                return false;
            }
        }
        return false;
    }
    
    template <typename Func>
    bool access_and_maybe_erase(K key, Func&& func) {
        if (key == empty_key() || key == tombstone_key()) return false;
        
        std::unique_lock lock(rehash_mutex_);
        std::size_t h = hash(key);
        
        for (std::size_t i = 0; i < capacity_; ++i) {
            std::size_t idx = probe(h, i);
            K k = slots_[idx].key.load(std::memory_order_acquire);
            
            if (k == key) {
                bool should_erase = func(slots_[idx].value);
                if (should_erase) {
                    slots_[idx].key.store(tombstone_key(), std::memory_order_release);
                    slots_[idx].value = V{};
                    size_.fetch_sub(1, std::memory_order_relaxed);
                }
                return true;
            }
            if (k == empty_key()) {
                return false;
            }
        }
        return false;
    }
    
    std::size_t size() const noexcept {
        return size_.load(std::memory_order_relaxed);
    }
    
    void clear() {
        std::unique_lock lock(rehash_mutex_);
        for (auto& slot : slots_) {
            slot.key.store(empty_key(), std::memory_order_relaxed);
            slot.value = V{};
        }
        size_.store(0, std::memory_order_relaxed);
    }
    
    template <typename Func>
    void for_each(Func&& func) const {
        std::shared_lock lock(rehash_mutex_);
        for (std::size_t i = 0; i < slots_.size(); ++i) {
            K k = slots_[i].key.load(std::memory_order_acquire);
            if (k != empty_key() && k != tombstone_key()) {
                std::lock_guard stripe_lock(get_stripe_mutex(i));
                K k2 = slots_[i].key.load(std::memory_order_acquire);
                if (k2 != empty_key() && k2 != tombstone_key()) {
                    func(k2, std::as_const(slots_[i].value));
                }
            }
        }
    }
    
    template <typename Func>
    void for_each_mut(Func&& func) {
        std::unique_lock lock(rehash_mutex_);
        for (auto& slot : slots_) {
            K k = slot.key.load(std::memory_order_acquire);
            if (k != empty_key() && k != tombstone_key()) {
                func(k, slot.value);
            }
        }
    }
};

SCL_FORCE_INLINE std::size_t get_default_shard_count() {
    std::size_t n = std::thread::hardware_concurrency();
    if (n == 0) n = 16;
    if (n < 4) return 4;
    if (n > 64) return 64;
    return (n + 3) & ~3ull;
}

} // namespace detail

// =============================================================================
// Alias Record - Layer 2: Access Pointer with Reference Count
// =============================================================================

struct AliasRecord {
    BufferID buffer_id;                    // Parent buffer
    std::atomic<std::uint32_t> ref_count;  // Number of instances holding this alias
    
    AliasRecord() noexcept : buffer_id(0), ref_count(0) {}
    
    AliasRecord(BufferID bid, std::uint32_t initial_ref = 1) noexcept
        : buffer_id(bid), ref_count(initial_ref) {}
    
    AliasRecord(const AliasRecord& other) noexcept
        : buffer_id(other.buffer_id)
        , ref_count(other.ref_count.load(std::memory_order_relaxed)) {}
    
    AliasRecord& operator=(const AliasRecord& other) noexcept {
        if (this != &other) {
            buffer_id = other.buffer_id;
            ref_count.store(other.ref_count.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
        }
        return *this;
    }
    
    AliasRecord(AliasRecord&& other) noexcept
        : buffer_id(other.buffer_id)
        , ref_count(other.ref_count.load(std::memory_order_relaxed)) {
        other.buffer_id = 0;
        other.ref_count.store(0, std::memory_order_relaxed);
    }
    
    AliasRecord& operator=(AliasRecord&& other) noexcept {
        if (this != &other) {
            buffer_id = other.buffer_id;
            ref_count.store(other.ref_count.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
            other.buffer_id = 0;
            other.ref_count.store(0, std::memory_order_relaxed);
        }
        return *this;
    }
};

// =============================================================================
// Registry: Unified Memory Management with Three-Layer Reference Counting
// =============================================================================

class Registry {
public:
    using Deleter = void (*)(void*);

    struct PtrRecord {
        std::uint64_t byte_size;
        AllocType type;
        Deleter custom_deleter;
    };

    struct BufferInfo {
        void* real_ptr;
        std::uint64_t byte_size;
        AllocType type;
        Deleter custom_deleter;
    };

    struct RefCountedBuffer {
        BufferInfo info;
        std::atomic<std::uint32_t> alias_count{0};  // Number of aliases pointing to this buffer
        
        RefCountedBuffer() = default;
        
        RefCountedBuffer(BufferInfo info_, std::uint32_t initial_alias_count)
            : info(std::move(info_)), alias_count(initial_alias_count) {}
        
        RefCountedBuffer(const RefCountedBuffer&) = delete;
        RefCountedBuffer& operator=(const RefCountedBuffer&) = delete;
        
        RefCountedBuffer(RefCountedBuffer&& other) noexcept
            : info(std::move(other.info))
            , alias_count(other.alias_count.load(std::memory_order_relaxed)) {
            other.alias_count.store(0, std::memory_order_relaxed);
        }
        
        RefCountedBuffer& operator=(RefCountedBuffer&& other) noexcept {
            if (this != &other) {
                info = std::move(other.info);
                alias_count.store(other.alias_count.load(std::memory_order_relaxed), 
                                 std::memory_order_relaxed);
                other.alias_count.store(0, std::memory_order_relaxed);
            }
            return *this;
        }
    };

private:
    static constexpr std::size_t kCacheLineSize = 64;
    
    struct alignas(kCacheLineSize) PtrShard {
        detail::ConcurrentFlatMap<void*, PtrRecord> records;
    };
    
    // NEW: Alias shard now stores AliasRecord with ref_count
    struct alignas(kCacheLineSize) AliasShard {
        detail::ConcurrentFlatMap<void*, AliasRecord> aliases;
    };
    
    struct alignas(kCacheLineSize) BufferShard {
        detail::ConcurrentFlatMap<BufferID, std::unique_ptr<RefCountedBuffer>> buffers;
    };
    
    std::vector<PtrShard> ptr_shards_;
    std::vector<AliasShard> alias_shards_;
    std::vector<BufferShard> buf_shards_;
    std::size_t num_shards_;
    std::atomic<BufferID> next_buffer_id_{2};
    std::atomic<std::size_t> total_ptrs_{0};
    std::atomic<std::size_t> total_ptr_bytes_{0};
    std::atomic<std::size_t> total_buffers_{0};
    std::atomic<std::size_t> total_buffer_bytes_{0};
    std::atomic<std::size_t> total_aliases_{0};
    
    SCL_FORCE_INLINE std::size_t shard_index(const void* ptr) const noexcept {
        return std::hash<const void*>{}(ptr) % num_shards_;
    }
    
    SCL_FORCE_INLINE std::size_t shard_index(BufferID id) const noexcept {
        return id % num_shards_;
    }
    
    // Pre-defined deleters (avoids lambda creation on each call)
    static void deleter_array_new(void* p) { delete[] static_cast<char*>(p); }
    static void deleter_scalar_new(void* p) { delete static_cast<char*>(p); }
    static void deleter_aligned(void* p) { scl::memory::aligned_free(static_cast<char*>(p)); }
    
    static Deleter get_deleter(AllocType type, Deleter custom) noexcept {
        switch (type) {
            case AllocType::ArrayNew:   return deleter_array_new;
            case AllocType::ScalarNew:  return deleter_scalar_new;
            case AllocType::AlignedAlloc: return deleter_aligned;
            case AllocType::Custom:     return custom;
        }
        return nullptr;
    }

public:
    explicit Registry(std::size_t num_shards = detail::get_default_shard_count())
        : ptr_shards_(num_shards)
        , alias_shards_(num_shards)
        , buf_shards_(num_shards)
        , num_shards_(num_shards) {}
    
    // =========================================================================
    // Simple Pointer Management (Layer 1 Alternative - No Sharing)
    // =========================================================================
    
    void register_ptr(void* ptr, std::size_t byte_size, AllocType type, 
                      Deleter custom_deleter = nullptr) {
        if (SCL_UNLIKELY(!ptr || byte_size == 0)) return;
        PtrRecord rec{byte_size, type, custom_deleter};
        auto& shard = ptr_shards_[shard_index(ptr)];
        
        if (SCL_LIKELY(shard.records.insert(ptr, rec))) {
            total_ptrs_.fetch_add(1, std::memory_order_relaxed);
            total_ptr_bytes_.fetch_add(byte_size, std::memory_order_relaxed);
        }
#ifndef NDEBUG
        else {
            std::fprintf(stderr, "WARNING: Duplicate pointer registration: %p\n", ptr);
        }
#endif
    }
    
    bool unregister_ptr(void* ptr) {
        if (SCL_UNLIKELY(!ptr)) return false;
        auto& shard = ptr_shards_[shard_index(ptr)];
        PtrRecord rec;
        
        if (SCL_UNLIKELY(!shard.records.erase_and_get(ptr, rec))) {
            return false;
        }
        
        auto deleter = get_deleter(rec.type, rec.custom_deleter);
        if (SCL_LIKELY(deleter)) {
            deleter(ptr);
        }
        
        total_ptrs_.fetch_sub(1, std::memory_order_relaxed);
        total_ptr_bytes_.fetch_sub(rec.byte_size, std::memory_order_relaxed);
        return true;
    }
    
    void register_batch(std::span<const std::tuple<void*, std::size_t, AllocType, Deleter>> entries) {
        if (entries.empty()) return;
        
        std::vector<std::vector<std::tuple<void*, std::size_t, AllocType, Deleter>>> 
            shard_entries(num_shards_);
        
        for (const auto& entry : entries) {
            void* ptr = std::get<0>(entry);
            if (!ptr) continue;
            shard_entries[shard_index(ptr)].push_back(entry);
        }
        
        std::size_t total_count = 0;
        std::size_t total_bytes = 0;
        for (std::size_t i = 0; i < num_shards_; ++i) {
            if (shard_entries[i].empty()) continue;
            
            auto& shard = ptr_shards_[i];
            for (const auto& [ptr, byte_size, type, deleter] : shard_entries[i]) {
                if (byte_size == 0) continue;
                PtrRecord rec{byte_size, type, deleter};
                if (shard.records.insert(ptr, rec)) {
                    ++total_count;
                    total_bytes += byte_size;
                }
            }
        }
        
        total_ptrs_.fetch_add(total_count, std::memory_order_relaxed);
        total_ptr_bytes_.fetch_add(total_bytes, std::memory_order_relaxed);
    }
    
    void unregister_batch(std::span<void* const> ptrs) {
        if (ptrs.empty()) return;
        
        std::vector<std::vector<void*>> shard_ptrs(num_shards_);
        for (void* ptr : ptrs) {
            if (!ptr) continue;
            shard_ptrs[shard_index(ptr)].push_back(ptr);
        }
        
        std::size_t total_count = 0;
        std::size_t total_bytes = 0;
        std::vector<std::pair<void*, Deleter>> to_delete;
        to_delete.reserve(ptrs.size());
        
        for (std::size_t i = 0; i < num_shards_; ++i) {
            if (shard_ptrs[i].empty()) continue;
            
            auto& shard = ptr_shards_[i];
            for (void* ptr : shard_ptrs[i]) {
                PtrRecord rec;
                if (shard.records.erase_and_get(ptr, rec)) {
                    to_delete.emplace_back(ptr, get_deleter(rec.type, rec.custom_deleter));
                    ++total_count;
                    total_bytes += rec.byte_size;
                }
            }
        }
        
        total_ptrs_.fetch_sub(total_count, std::memory_order_relaxed);
        total_ptr_bytes_.fetch_sub(total_bytes, std::memory_order_relaxed);
        
        for (auto& [ptr, deleter] : to_delete) {
            if (deleter) deleter(ptr);
        }
    }
    
    template <typename T>
    SCL_NODISCARD T* new_array(std::size_t count) {
        if (count == 0) return nullptr;
        
        T* ptr = nullptr;
        try {
            ptr = new T[count]();
        } catch (...) {
            return nullptr;
        }
        
        register_ptr(ptr, count * sizeof(T), AllocType::ArrayNew);
        return ptr;
    }
    
    template <typename T>
    SCL_NODISCARD T* new_aligned(std::size_t count, std::size_t alignment = 64) {
        T* ptr = scl::memory::aligned_alloc<T>(count, alignment);
        if (ptr) {
            register_ptr(ptr, count * sizeof(T), AllocType::AlignedAlloc);
        }
        return ptr;
    }
    
    template <typename T, typename... Args>
    SCL_NODISCARD T* new_object(Args&&... args) {
        T* ptr = nullptr;
        try {
            ptr = new T(std::forward<Args>(args)...);
        } catch (...) {
            return nullptr;
        }
        
        register_ptr(ptr, sizeof(T), AllocType::ScalarNew);
        return ptr;
    }
    
    // =========================================================================
    // Buffer Management (Layer 1)
    // =========================================================================
    
    SCL_NODISCARD BufferID create_buffer(void* real_ptr, std::size_t byte_size,
                                         AllocType type, Deleter custom_deleter = nullptr) {
        if (!real_ptr || byte_size == 0) return 0;
        BufferID id = next_buffer_id_.fetch_add(1, std::memory_order_relaxed);
        
        BufferInfo info{real_ptr, byte_size, type, custom_deleter};
        auto buffer = std::make_unique<RefCountedBuffer>(std::move(info), 0);
        auto& shard = buf_shards_[shard_index(id)];
        shard.buffers.insert(id, std::move(buffer));
        
        total_buffers_.fetch_add(1, std::memory_order_relaxed);
        total_buffer_bytes_.fetch_add(byte_size, std::memory_order_relaxed);
        
        return id;
    }
    
    // Legacy: register_buffer with initial refcount (backward compatible)
    SCL_NODISCARD BufferID register_buffer(void* real_ptr, std::size_t byte_size,
                                           std::uint32_t initial_alias_count,
                                           AllocType type, Deleter custom_deleter = nullptr) {
        if (!real_ptr || byte_size == 0 || initial_alias_count == 0) return 0;
        BufferID id = next_buffer_id_.fetch_add(1, std::memory_order_relaxed);
        
        BufferInfo info{real_ptr, byte_size, type, custom_deleter};
        auto buffer = std::make_unique<RefCountedBuffer>(std::move(info), initial_alias_count);
        auto& shard = buf_shards_[shard_index(id)];
        shard.buffers.insert(id, std::move(buffer));
        
        total_buffers_.fetch_add(1, std::memory_order_relaxed);
        total_buffer_bytes_.fetch_add(byte_size, std::memory_order_relaxed);
        
        return id;
    }
    
    // =========================================================================
    // Alias Management (Layer 2) - NEW API
    // =========================================================================
    
    /// @brief Create a new alias pointing to a buffer
    /// @param alias_ptr The access pointer
    /// @param buffer_id The parent buffer
    /// @param initial_ref Initial reference count (default 1)
    /// @return true if alias was created
    SCL_FORCE_INLINE bool create_alias(void* alias_ptr, BufferID buffer_id, std::uint32_t initial_ref = 1) {
        if (SCL_UNLIKELY(!alias_ptr || buffer_id == 0)) return false;
        
        AliasRecord rec(buffer_id, initial_ref);
        auto& shard = alias_shards_[shard_index(alias_ptr)];
        
        if (SCL_LIKELY(shard.aliases.insert(alias_ptr, rec))) {
            // Increment buffer's alias count
            auto& buf_shard = buf_shards_[shard_index(buffer_id)];
            buf_shard.buffers.access(buffer_id,
                [](std::unique_ptr<RefCountedBuffer>& buf) {
                    if (SCL_LIKELY(buf)) {
                        buf->alias_count.fetch_add(1, std::memory_order_relaxed);
                    }
                });
            
            total_aliases_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        return false;
    }
    
    /// @brief Increment alias reference count (lock-free fast path)
    /// @param alias_ptr The alias pointer
    /// @param increment Amount to increment (default 1)
    /// @return true if alias exists
    SCL_FORCE_INLINE bool alias_incref(void* alias_ptr, std::uint32_t increment = 1) {
        if (SCL_UNLIKELY(!alias_ptr || increment == 0)) return false;
        
        auto& shard = alias_shards_[shard_index(alias_ptr)];
        return shard.aliases.access(alias_ptr,
            [increment](AliasRecord& rec) {
                rec.ref_count.fetch_add(increment, std::memory_order_relaxed);
            });
    }
    
    /// @brief Decrement alias reference count
    /// @param alias_ptr The alias pointer
    /// @return true if alias was removed (ref_count reached 0)
    SCL_FORCE_INLINE bool alias_decref(void* alias_ptr) {
        if (SCL_UNLIKELY(!alias_ptr)) return false;
        
        auto& alias_shard = alias_shards_[shard_index(alias_ptr)];
        
        BufferID buffer_id = 0;
        bool was_removed = false;
        
        // Atomically decrement and conditionally erase to avoid ABA race
        alias_shard.aliases.access_and_maybe_erase(alias_ptr,
            [&](AliasRecord& rec) -> bool {
                std::uint32_t old_ref = rec.ref_count.fetch_sub(1, std::memory_order_acq_rel);
                if (SCL_UNLIKELY(old_ref == 1)) {
                    buffer_id = rec.buffer_id;
                    was_removed = true;
                    return true;  // Erase atomically
                }
                return false;
            });
        
        if (SCL_UNLIKELY(was_removed)) {
            total_aliases_.fetch_sub(1, std::memory_order_relaxed);
            
            // Decrement buffer's alias_count
            if (SCL_LIKELY(buffer_id != 0)) {
                decrement_buffer_alias_count(buffer_id);
            }
        }
        
        return was_removed;
    }
    
    /// @brief Batch increment alias reference counts
    /// Groups by shard for better cache locality
    void alias_incref_batch(std::span<void* const> alias_ptrs, std::uint32_t increment = 1) {
        if (alias_ptrs.empty() || increment == 0) return;
        
        // Group by shard for better cache locality
        std::vector<std::vector<void*>> shard_ptrs(num_shards_);
        for (void* ptr : alias_ptrs) {
            if (ptr) {
                shard_ptrs[shard_index(ptr)].push_back(ptr);
            }
        }
        
        // Process each shard
        for (std::size_t i = 0; i < num_shards_; ++i) {
            if (shard_ptrs[i].empty()) continue;
            
            auto& shard = alias_shards_[i];
            for (void* ptr : shard_ptrs[i]) {
                shard.aliases.access(ptr,
                    [increment](AliasRecord& rec) {
                        rec.ref_count.fetch_add(increment, std::memory_order_relaxed);
                    });
            }
        }
    }
    
    /// @brief Batch decrement alias reference counts
    /// More efficient than calling alias_decref individually
    void alias_decref_batch(std::span<void* const> alias_ptrs) {
        if (alias_ptrs.empty()) return;
        
        // Group by shard for efficiency
        std::vector<std::vector<void*>> shard_ptrs(num_shards_);
        for (void* ptr : alias_ptrs) {
            if (ptr) {
                shard_ptrs[shard_index(ptr)].push_back(ptr);
            }
        }
        
        // Collect buffer IDs that need decrement
        std::unordered_map<BufferID, std::uint32_t> buffer_decrements;
        std::vector<void*> aliases_to_remove;
        aliases_to_remove.reserve(alias_ptrs.size());
        
        for (std::size_t i = 0; i < num_shards_; ++i) {
            if (shard_ptrs[i].empty()) continue;
            
            auto& shard = alias_shards_[i];
            for (void* ptr : shard_ptrs[i]) {
                shard.aliases.access(ptr,
                    [&](AliasRecord& rec) {
                        std::uint32_t old_ref = rec.ref_count.fetch_sub(1, std::memory_order_acq_rel);
                        if (old_ref == 1) {
                            // Last reference, mark for removal
                            aliases_to_remove.push_back(ptr);
                            buffer_decrements[rec.buffer_id]++;
                        }
                    });
            }
        }
        
        // Remove aliases that reached zero
        for (void* ptr : aliases_to_remove) {
            auto& shard = alias_shards_[shard_index(ptr)];
            shard.aliases.erase(ptr);
        }
        
        if (!aliases_to_remove.empty()) {
            total_aliases_.fetch_sub(aliases_to_remove.size(), std::memory_order_relaxed);
        }
        
        // Decrement buffer alias counts
        for (const auto& [buffer_id, count] : buffer_decrements) {
            decrement_buffer_alias_count(buffer_id, count);
        }
    }
    
    /// @brief Get alias reference count
    SCL_NODISCARD std::uint32_t alias_refcount(void* alias_ptr) const {
        if (!alias_ptr) return 0;
        
        std::uint32_t result = 0;
        alias_shards_[shard_index(alias_ptr)].aliases.access(alias_ptr,
            [&](const AliasRecord& rec) {
                result = rec.ref_count.load(std::memory_order_relaxed);
            });
        return result;
    }
    
    /// @brief Check if alias is unique (only one reference)
    SCL_NODISCARD bool alias_is_unique(void* alias_ptr) const {
        return alias_refcount(alias_ptr) == 1;
    }
    
    // =========================================================================
    // Legacy Alias API (Backward Compatible)
    // =========================================================================
    
    bool register_alias(void* alias_ptr, BufferID buffer_id) {
        return create_alias(alias_ptr, buffer_id, 1);
    }
    
    bool register_buffer_with_aliases(void* real_ptr, std::size_t byte_size,
                                      std::span<void* const> alias_ptrs,
                                      AllocType type, Deleter custom_deleter = nullptr) {
        if (!real_ptr || byte_size == 0 || alias_ptrs.empty()) return false;
        
        std::uint32_t alias_count = 0;
        for (void* p : alias_ptrs) {
            if (p) ++alias_count;
        }
        if (alias_count == 0) return false;
        
        // Create buffer with initial alias_count
        BufferID id = next_buffer_id_.fetch_add(1, std::memory_order_relaxed);
        
        BufferInfo info{real_ptr, byte_size, type, custom_deleter};
        auto buffer = std::make_unique<RefCountedBuffer>(std::move(info), alias_count);
        auto& buf_shard = buf_shards_[shard_index(id)];
        buf_shard.buffers.insert(id, std::move(buffer));
        
        total_buffers_.fetch_add(1, std::memory_order_relaxed);
        total_buffer_bytes_.fetch_add(byte_size, std::memory_order_relaxed);
        
        // Register aliases with ref_count = 1
        std::vector<std::vector<void*>> shard_aliases(num_shards_);
        for (void* ptr : alias_ptrs) {
            if (ptr) {
                shard_aliases[shard_index(ptr)].push_back(ptr);
            }
        }
        
        for (std::size_t i = 0; i < num_shards_; ++i) {
            if (shard_aliases[i].empty()) continue;
            
            auto& shard = alias_shards_[i];
            for (void* ptr : shard_aliases[i]) {
                AliasRecord rec(id, 1);
                shard.aliases.insert(ptr, rec);
            }
        }
        
        total_aliases_.fetch_add(alias_count, std::memory_order_relaxed);
        return true;
    }
    
    // Legacy: unregister_alias (decrements ref_count, removes if zero)
    bool unregister_alias(void* alias_ptr) {
        return alias_decref(alias_ptr);
    }
    
    // Legacy: unregister_aliases batch
    void unregister_aliases(std::span<void* const> alias_ptrs) {
        alias_decref_batch(alias_ptrs);
    }
    
    // Legacy: decrement_buffer_refcounts (now operates on alias ref_counts)
    void decrement_buffer_refcounts(std::span<void* const> alias_ptrs) {
        alias_decref_batch(alias_ptrs);
    }
    
    // Legacy: unregister_aliases_as_owner
    void unregister_aliases_as_owner(std::span<void* const> alias_ptrs) {
        alias_decref_batch(alias_ptrs);
    }
    
    // =========================================================================
    // Buffer Helpers
    // =========================================================================
    
private:
    void decrement_buffer_alias_count(BufferID buffer_id, std::uint32_t count = 1) {
        if (buffer_id == 0 || count == 0) return;
        
        auto& buf_shard = buf_shards_[shard_index(buffer_id)];
        
        void* ptr_to_delete = nullptr;
        Deleter deleter_to_use = nullptr;
        std::uint64_t freed_bytes = 0;
        bool should_free = false;
        
        buf_shard.buffers.access_and_maybe_erase(buffer_id,
            [&](std::unique_ptr<RefCountedBuffer>& buffer_ptr) -> bool {
                if (!buffer_ptr) return false;
                
                std::uint32_t old_count = buffer_ptr->alias_count.fetch_sub(
                    count, std::memory_order_acq_rel);
                
                if (old_count == count) {
                    // Last alias, free the buffer
                    ptr_to_delete = buffer_ptr->info.real_ptr;
                    deleter_to_use = get_deleter(buffer_ptr->info.type, 
                                                  buffer_ptr->info.custom_deleter);
                    freed_bytes = buffer_ptr->info.byte_size;
                    should_free = true;
                    return true;  // Remove buffer record
                }
                return false;
            });
        
        if (should_free) {
            if (deleter_to_use) {
                deleter_to_use(ptr_to_delete);
            }
            total_buffers_.fetch_sub(1, std::memory_order_relaxed);
            total_buffer_bytes_.fetch_sub(freed_bytes, std::memory_order_relaxed);
        }
    }
    
public:
    /// @brief Increment buffer reference count directly
    bool increment_buffer_refcount(BufferID buffer_id, std::uint32_t increment = 1) {
        if (buffer_id == 0 || increment == 0) return false;
        
        bool found = false;
        auto& buf_shard = buf_shards_[shard_index(buffer_id)];
        buf_shard.buffers.access(buffer_id,
            [&](std::unique_ptr<RefCountedBuffer>& buffer_ptr) {
                if (buffer_ptr) {
                    buffer_ptr->alias_count.fetch_add(increment, std::memory_order_acq_rel);
                    found = true;
                }
            });
        return found;
    }
    
    void increment_buffer_refcounts(const std::unordered_map<BufferID, std::uint32_t>& increments) {
        for (const auto& [buffer_id, increment] : increments) {
            increment_buffer_refcount(buffer_id, increment);
        }
    }
    
    // =========================================================================
    // Query Functions
    // =========================================================================
    
    SCL_NODISCARD bool contains_ptr(const void* ptr) const {
        if (!ptr) return false;
        return ptr_shards_[shard_index(ptr)].records.contains(const_cast<void*>(ptr));
    }
    
    SCL_NODISCARD bool contains_alias(const void* ptr) const {
        if (!ptr) return false;
        return alias_shards_[shard_index(ptr)].aliases.contains(const_cast<void*>(ptr));
    }
    
    SCL_NODISCARD bool contains(const void* ptr) const {
        return contains_ptr(ptr) || contains_alias(ptr);
    }
    
    SCL_NODISCARD std::size_t size_of(const void* ptr) const {
        if (!ptr) return 0;
        
        PtrRecord rec;
        if (ptr_shards_[shard_index(ptr)].records.find(const_cast<void*>(ptr), rec)) {
            return rec.byte_size;
        }
        return 0;
    }
    
    SCL_NODISCARD BufferID get_buffer_id(const void* alias_ptr) const {
        if (!alias_ptr) return 0;
        
        BufferID id = 0;
        alias_shards_[shard_index(alias_ptr)].aliases.access(const_cast<void*>(alias_ptr),
            [&](const AliasRecord& rec) {
                id = rec.buffer_id;
            });
        return id;
    }
    
    SCL_NODISCARD std::uint32_t get_refcount(BufferID buffer_id) const {
        if (buffer_id == 0) return 0;
        
        std::uint32_t result = 0;
        buf_shards_[shard_index(buffer_id)].buffers.access(buffer_id,
            [&](const std::unique_ptr<RefCountedBuffer>& buffer_ptr) {
                if (buffer_ptr) {
                    result = buffer_ptr->alias_count.load(std::memory_order_relaxed);
                }
            });
        return result;
    }
    
    // =========================================================================
    // Statistics
    // =========================================================================
    
    SCL_NODISCARD std::size_t ptr_count() const noexcept {
        return total_ptrs_.load(std::memory_order_relaxed);
    }
    
    SCL_NODISCARD std::size_t ptr_bytes() const noexcept {
        return total_ptr_bytes_.load(std::memory_order_relaxed);
    }
    
    SCL_NODISCARD std::size_t buffer_count() const noexcept {
        return total_buffers_.load(std::memory_order_relaxed);
    }
    
    SCL_NODISCARD std::size_t buffer_bytes() const noexcept {
        return total_buffer_bytes_.load(std::memory_order_relaxed);
    }
    
    SCL_NODISCARD std::size_t alias_count() const noexcept {
        return total_aliases_.load(std::memory_order_relaxed);
    }
    
    SCL_NODISCARD std::size_t total_count() const noexcept {
        return ptr_count() + buffer_count();
    }
    
    SCL_NODISCARD std::size_t total_bytes() const noexcept {
        return ptr_bytes() + buffer_bytes();
    }
    
    SCL_NODISCARD std::vector<std::pair<void*, std::size_t>> dump_ptrs() const {
        std::vector<std::pair<void*, std::size_t>> result;
        result.reserve(ptr_count());
        
        for (const auto& shard : ptr_shards_) {
            shard.records.for_each([&](void* ptr, const PtrRecord& rec) {
                result.emplace_back(ptr, rec.byte_size);
            });
        }
        return result;
    }
    
    // =========================================================================
    // Cleanup
    // =========================================================================
    
    void clear_all_and_free() {
        // Free simple pointers
        for (auto& shard : ptr_shards_) {
            shard.records.for_each_mut([this](void* ptr, PtrRecord& rec) {
                auto deleter = get_deleter(rec.type, rec.custom_deleter);
                if (deleter) deleter(ptr);
            });
            shard.records.clear();
        }
        
        // Clear aliases
        for (auto& shard : alias_shards_) {
            shard.aliases.clear();
        }
        
        // Free buffers
        for (auto& shard : buf_shards_) {
            shard.buffers.for_each_mut([this](BufferID, std::unique_ptr<RefCountedBuffer>& buffer_ptr) {
                if (buffer_ptr) {
                    auto deleter = get_deleter(buffer_ptr->info.type, buffer_ptr->info.custom_deleter);
                    if (deleter) deleter(buffer_ptr->info.real_ptr);
                }
            });
            shard.buffers.clear();
        }
        
        total_ptrs_.store(0, std::memory_order_relaxed);
        total_ptr_bytes_.store(0, std::memory_order_relaxed);
        total_buffers_.store(0, std::memory_order_relaxed);
        total_buffer_bytes_.store(0, std::memory_order_relaxed);
        total_aliases_.store(0, std::memory_order_relaxed);
    }
    
    ~Registry() {
#ifndef NDEBUG
        std::size_t leaked_ptrs = ptr_count();
        std::size_t leaked_buffers = buffer_count();
        std::size_t leaked_aliases = alias_count();
        std::size_t leaked_bytes = total_bytes();
        
        if (leaked_ptrs > 0 || leaked_buffers > 0) {
            std::fprintf(stderr,
                "WARNING: Registry leaked %zu pointers + %zu buffers + %zu aliases (%zu bytes total)\n",
                leaked_ptrs, leaked_buffers, leaked_aliases, leaked_bytes);
        }
#endif
        clear_all_and_free();
    }
    
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;
    Registry(Registry&&) = delete;
    Registry& operator=(Registry&&) = delete;
};

// =============================================================================
// Global Registry Access
// =============================================================================

inline Registry& get_registry() {
    static Registry registry;
    return registry;
}

// =============================================================================
// Global Convenience Functions
// =============================================================================

inline void register_ptr(void* ptr, std::size_t byte_size, AllocType type,
                         Registry::Deleter deleter = nullptr) {
    get_registry().register_ptr(ptr, byte_size, type, deleter);
}

inline bool unregister_ptr(void* ptr) {
    return get_registry().unregister_ptr(ptr);
}

template <typename T>
SCL_NODISCARD inline T* new_array(std::size_t count) {
    return get_registry().new_array<T>(count);
}

template <typename T>
SCL_NODISCARD inline T* new_aligned(std::size_t count, std::size_t alignment = 64) {
    return get_registry().new_aligned<T>(count, alignment);
}

template <typename T, typename... Args>
SCL_NODISCARD inline T* new_object(Args&&... args) {
    return get_registry().new_object<T>(std::forward<Args>(args)...);
}

inline bool register_shared_buffer(void* real_ptr, std::size_t byte_size,
                                   std::span<void* const> alias_ptrs,
                                   AllocType type,
                                   Registry::Deleter custom_deleter = nullptr) {
    return get_registry().register_buffer_with_aliases(real_ptr, byte_size, alias_ptrs,
                                                        type, custom_deleter);
}

inline bool unregister_alias(void* alias_ptr) {
    return get_registry().alias_decref(alias_ptr);
}

inline void unregister_aliases(std::span<void* const> alias_ptrs) {
    get_registry().alias_decref_batch(alias_ptrs);
}

// NEW: Alias reference counting functions
inline bool alias_incref(void* alias_ptr, std::uint32_t increment = 1) {
    return get_registry().alias_incref(alias_ptr, increment);
}

inline bool alias_decref(void* alias_ptr) {
    return get_registry().alias_decref(alias_ptr);
}

inline void alias_incref_batch(std::span<void* const> alias_ptrs, std::uint32_t increment = 1) {
    get_registry().alias_incref_batch(alias_ptrs, increment);
}

inline void alias_decref_batch(std::span<void* const> alias_ptrs) {
    get_registry().alias_decref_batch(alias_ptrs);
}

inline std::uint32_t alias_refcount(void* alias_ptr) {
    return get_registry().alias_refcount(alias_ptr);
}

// =============================================================================
// RAII Guard
// =============================================================================

class RegistryGuard {
    void* ptr_;
    bool is_alias_;
    bool released_ = false;

public:
    explicit RegistryGuard(void* ptr, bool is_alias = false) noexcept
        : ptr_(ptr), is_alias_(is_alias) {}
    
    ~RegistryGuard() {
        if (!released_ && ptr_) {
            if (is_alias_) {
                get_registry().alias_decref(ptr_);
            } else {
                get_registry().unregister_ptr(ptr_);
            }
        }
    }
    
    void release() noexcept { released_ = true; }
    
    RegistryGuard(const RegistryGuard&) = delete;
    RegistryGuard& operator=(const RegistryGuard&) = delete;
    
    RegistryGuard(RegistryGuard&& other) noexcept
        : ptr_(other.ptr_), is_alias_(other.is_alias_), released_(other.released_) {
        other.released_ = true;
    }
    
    RegistryGuard& operator=(RegistryGuard&& other) noexcept {
        if (this != &other) {
            if (!released_ && ptr_) {
                if (is_alias_) {
                    get_registry().alias_decref(ptr_);
                } else {
                    get_registry().unregister_ptr(ptr_);
                }
            }
            ptr_ = other.ptr_;
            is_alias_ = other.is_alias_;
            released_ = other.released_;
            other.released_ = true;
        }
        return *this;
    }
};

// =============================================================================
// Legacy Compatibility
// =============================================================================

using HandlerRegistry = Registry;

inline Registry& get_handler_registry() {
    return get_registry();
}

inline Registry& get_refcount_registry() {
    return get_registry();
}

} // namespace scl
