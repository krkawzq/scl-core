#pragma once

#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/error.hpp"

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
// BRIEF: Unified high-performance memory registry with reference counting
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
// Registry: Unified Memory Management
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
        std::atomic<std::uint32_t> refcount{0};
        
        RefCountedBuffer() = default;
        
        RefCountedBuffer(BufferInfo info_, std::uint32_t initial_refcount)
            : info(std::move(info_)), refcount(initial_refcount) {}
        
        RefCountedBuffer(const RefCountedBuffer&) = delete;
        RefCountedBuffer& operator=(const RefCountedBuffer&) = delete;
        
        RefCountedBuffer(RefCountedBuffer&& other) noexcept
            : info(std::move(other.info))
            , refcount(other.refcount.load(std::memory_order_relaxed)) {
            other.refcount.store(0, std::memory_order_relaxed);
        }
        
        RefCountedBuffer& operator=(RefCountedBuffer&& other) noexcept {
            if (this != &other) {
                info = std::move(other.info);
                refcount.store(other.refcount.load(std::memory_order_relaxed), 
                              std::memory_order_relaxed);
                other.refcount.store(0, std::memory_order_relaxed);
            }
            return *this;
        }
    };

private:
    static constexpr std::size_t kCacheLineSize = 64;
    
    struct alignas(kCacheLineSize) PtrShard {
        detail::ConcurrentFlatMap<void*, PtrRecord> records;
    };
    
    struct alignas(kCacheLineSize) AliasShard {
        detail::ConcurrentFlatMap<void*, BufferID> aliases;
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
    
    SCL_FORCE_INLINE std::size_t shard_index(const void* ptr) const noexcept {
        return std::hash<const void*>{}(ptr) % num_shards_;
    }
    
    SCL_FORCE_INLINE std::size_t shard_index(BufferID id) const noexcept {
        return id % num_shards_;
    }
    
    static Deleter get_deleter(AllocType type, Deleter custom) noexcept {
        switch (type) {
            case AllocType::ArrayNew:
                return [](void* p) { delete[] static_cast<char*>(p); };
            case AllocType::ScalarNew:
                return [](void* p) { delete static_cast<char*>(p); };
            case AllocType::AlignedAlloc:
                return [](void* p) { scl::memory::aligned_free(static_cast<char*>(p)); };
            case AllocType::Custom:
                return custom;
        }
        return nullptr;
    }

public:
    explicit Registry(std::size_t num_shards = detail::get_default_shard_count())
        : ptr_shards_(num_shards)
        , alias_shards_(num_shards)
        , buf_shards_(num_shards)
        , num_shards_(num_shards) {}
    
    void register_ptr(void* ptr, std::size_t byte_size, AllocType type, 
                      Deleter custom_deleter = nullptr) {
        if (!ptr || byte_size == 0) return;
        PtrRecord rec{byte_size, type, custom_deleter};
        auto& shard = ptr_shards_[shard_index(ptr)];
        
        if (shard.records.insert(ptr, rec)) {
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
        if (!ptr) return false;
        auto& shard = ptr_shards_[shard_index(ptr)];
        PtrRecord rec;
        
        if (!shard.records.erase_and_get(ptr, rec)) {
            return false;
        }
        
        auto deleter = get_deleter(rec.type, rec.custom_deleter);
        if (deleter) {
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
    
    SCL_NODISCARD BufferID register_buffer(void* real_ptr, std::size_t byte_size,
                                           std::uint32_t initial_refcount,
                                           AllocType type, Deleter custom_deleter = nullptr) {
        if (!real_ptr || byte_size == 0 || initial_refcount == 0) return 0;
        BufferID id = next_buffer_id_.fetch_add(1, std::memory_order_relaxed);
        
        BufferInfo info{real_ptr, byte_size, type, custom_deleter};
        auto buffer = std::make_unique<RefCountedBuffer>(std::move(info), initial_refcount);
        auto& shard = buf_shards_[shard_index(id)];
        shard.buffers.insert(id, std::move(buffer));
        
        total_buffers_.fetch_add(1, std::memory_order_relaxed);
        total_buffer_bytes_.fetch_add(byte_size, std::memory_order_relaxed);
        
        return id;
    }
    
    bool register_alias(void* alias_ptr, BufferID buffer_id) {
        if (!alias_ptr || buffer_id == 0) return false;
        auto& shard = alias_shards_[shard_index(alias_ptr)];
        return shard.aliases.insert(alias_ptr, buffer_id);
    }
    
    bool register_buffer_with_aliases(void* real_ptr, std::size_t byte_size,
                                       std::span<void* const> alias_ptrs,
                                       AllocType type, Deleter custom_deleter = nullptr) {
        if (!real_ptr || byte_size == 0 || alias_ptrs.empty()) return false;
        
        std::uint32_t refcount = 0;
        for (void* p : alias_ptrs) {
            if (p) ++refcount;
        }
        if (refcount == 0) return false;
        
        BufferID id = register_buffer(real_ptr, byte_size, refcount, type, custom_deleter);
        if (id == 0) return false;
        
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
                shard.aliases.insert(ptr, id);
            }
        }
        
        return true;
    }
    
    bool unregister_alias(void* alias_ptr) {
        if (!alias_ptr) return false;
        
        auto& alias_shard = alias_shards_[shard_index(alias_ptr)];
        BufferID buffer_id;
        
        if (!alias_shard.aliases.erase_and_get(alias_ptr, buffer_id)) {
            return false;
        }
        
        auto& buf_shard = buf_shards_[shard_index(buffer_id)];
        
        void* ptr_to_delete = nullptr;
        Deleter deleter_to_use = nullptr;
        std::uint64_t freed_bytes = 0;
        bool should_free = false;
        
        buf_shard.buffers.access_and_maybe_erase(buffer_id, 
            [&](std::unique_ptr<RefCountedBuffer>& buffer_ptr) -> bool {
                if (!buffer_ptr) return false;
                
                std::uint32_t old_refcount = buffer_ptr->refcount.fetch_sub(1, std::memory_order_acq_rel);
                
                if (old_refcount == 1) {
                    ptr_to_delete = buffer_ptr->info.real_ptr;
                    deleter_to_use = get_deleter(buffer_ptr->info.type, buffer_ptr->info.custom_deleter);
                    freed_bytes = buffer_ptr->info.byte_size;
                    should_free = true;
                    return true;
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
        
        return true;
    }
    
    void unregister_aliases(std::span<void* const> alias_ptrs) {
        if (alias_ptrs.empty()) return;

        // Step 1: Look up buffer IDs WITHOUT removing aliases yet
        // Group aliases by buffer_id for efficient processing
        std::unordered_map<BufferID, std::vector<void*>> buffer_to_aliases;
        buffer_to_aliases.reserve(alias_ptrs.size());

        for (void* ptr : alias_ptrs) {
            if (!ptr) continue;
            BufferID id = get_buffer_id(ptr);  // Lookup only, don't erase
            if (id != 0) {
                buffer_to_aliases[id].push_back(ptr);
            }
        }

        if (buffer_to_aliases.empty()) return;

        // Step 2: Decrement refcounts and identify buffers to free
        std::vector<std::pair<void*, Deleter>> to_delete;
        std::vector<BufferID> buffers_to_cleanup;
        to_delete.reserve(buffer_to_aliases.size());
        buffers_to_cleanup.reserve(buffer_to_aliases.size());

        std::size_t freed_buffers = 0;
        std::size_t freed_bytes = 0;

        for (const auto& [buffer_id, aliases] : buffer_to_aliases) {
            std::uint32_t decrement = static_cast<std::uint32_t>(aliases.size());
            auto& buf_shard = buf_shards_[shard_index(buffer_id)];

            buf_shard.buffers.access_and_maybe_erase(buffer_id,
                [&](std::unique_ptr<RefCountedBuffer>& buffer_ptr) -> bool {
                    if (!buffer_ptr) return false;

                    std::uint32_t old_refcount = buffer_ptr->refcount.fetch_sub(
                        decrement, std::memory_order_acq_rel);

                    if (old_refcount == decrement) {
                        // Last reference - prepare to delete and mark for alias cleanup
                        to_delete.emplace_back(
                            buffer_ptr->info.real_ptr,
                            get_deleter(buffer_ptr->info.type, buffer_ptr->info.custom_deleter));
                        freed_bytes += buffer_ptr->info.byte_size;
                        ++freed_buffers;
                        buffers_to_cleanup.push_back(buffer_id);
                        return true;  // Remove buffer record
                    }
                    return false;
                });
        }

        // Step 3: Remove alias mappings ONLY for buffers that were freed
        for (BufferID freed_id : buffers_to_cleanup) {
            const auto& aliases = buffer_to_aliases[freed_id];
            for (void* ptr : aliases) {
                auto& alias_shard = alias_shards_[shard_index(ptr)];
                alias_shard.aliases.erase(ptr);
            }
        }

        if (freed_buffers > 0) {
            total_buffers_.fetch_sub(freed_buffers, std::memory_order_relaxed);
            total_buffer_bytes_.fetch_sub(freed_bytes, std::memory_order_relaxed);
        }

        // Step 4: Actually free the memory
        for (auto& [ptr, deleter] : to_delete) {
            if (deleter) deleter(ptr);
        }
    }
    
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
        alias_shards_[shard_index(alias_ptr)].aliases.find(const_cast<void*>(alias_ptr), id);
        return id;
    }
    
    SCL_NODISCARD std::uint32_t get_refcount(BufferID buffer_id) const {
        if (buffer_id == 0) return 0;
        
        std::uint32_t result = 0;
        buf_shards_[shard_index(buffer_id)].buffers.access(buffer_id,
            [&](const std::unique_ptr<RefCountedBuffer>& buffer_ptr) {
                if (buffer_ptr) {
                    result = buffer_ptr->refcount.load(std::memory_order_relaxed);
                }
            });
        return result;
    }
    
    /// @brief Increment reference count for a buffer
    /// @param buffer_id Buffer ID to increment
    /// @param increment Amount to increment (default 1)
    /// @return true if buffer exists and increment succeeded, false otherwise
    bool increment_buffer_refcount(BufferID buffer_id, std::uint32_t increment = 1) {
        if (buffer_id == 0 || increment == 0) return false;
        
        bool found = false;
        auto& buf_shard = buf_shards_[shard_index(buffer_id)];
        buf_shard.buffers.access(buffer_id,
            [&](std::unique_ptr<RefCountedBuffer>& buffer_ptr) {
                if (buffer_ptr) {
                    buffer_ptr->refcount.fetch_add(increment, std::memory_order_acq_rel);
                    found = true;
                }
            });
        return found;
    }
    
    /// @brief Increment reference counts for multiple buffers
    /// @param increments Map of buffer_id -> increment amount
    void increment_buffer_refcounts(const std::unordered_map<BufferID, std::uint32_t>& increments) {
        for (const auto& [buffer_id, increment] : increments) {
            increment_buffer_refcount(buffer_id, increment);
        }
    }

    /// @brief Decrement buffer reference counts without removing alias mappings
    /// @note Used by views that share pointers with original matrices
    /// @note Alias mappings are only removed when buffer is actually freed
    /// @param alias_ptrs Pointers to look up and decrement
    void decrement_buffer_refcounts(std::span<void* const> alias_ptrs) {
        if (alias_ptrs.empty()) return;

        // Step 1: Look up buffer IDs WITHOUT removing aliases yet
        std::unordered_map<BufferID, std::vector<void*>> buffer_to_aliases;
        buffer_to_aliases.reserve(alias_ptrs.size());

        for (void* ptr : alias_ptrs) {
            if (!ptr) continue;
            BufferID id = get_buffer_id(ptr);  // Only lookup, don't erase
            if (id != 0) {
                buffer_to_aliases[id].push_back(ptr);
            }
        }

        if (buffer_to_aliases.empty()) return;

        // Step 2: Decrement reference counts and identify buffers to free
        std::vector<std::pair<void*, Deleter>> to_delete;
        std::vector<BufferID> buffers_to_cleanup;
        to_delete.reserve(buffer_to_aliases.size());
        buffers_to_cleanup.reserve(buffer_to_aliases.size());

        std::size_t freed_buffers = 0;
        std::size_t freed_bytes = 0;

        for (const auto& [buffer_id, aliases] : buffer_to_aliases) {
            std::uint32_t decrement = static_cast<std::uint32_t>(aliases.size());
            auto& buf_shard = buf_shards_[shard_index(buffer_id)];

            buf_shard.buffers.access_and_maybe_erase(buffer_id,
                [&](std::unique_ptr<RefCountedBuffer>& buffer_ptr) -> bool {
                    if (!buffer_ptr) return false;

                    std::uint32_t old_refcount = buffer_ptr->refcount.fetch_sub(
                        decrement, std::memory_order_acq_rel);

                    if (old_refcount == decrement) {
                        // Last reference - prepare to delete
                        to_delete.emplace_back(
                            buffer_ptr->info.real_ptr,
                            get_deleter(buffer_ptr->info.type, buffer_ptr->info.custom_deleter));
                        freed_bytes += buffer_ptr->info.byte_size;
                        ++freed_buffers;
                        buffers_to_cleanup.push_back(buffer_id);
                        return true;  // Remove buffer record
                    }
                    return false;
                });
        }

        // Step 3: Remove alias mappings ONLY for buffers that were freed
        for (BufferID freed_id : buffers_to_cleanup) {
            const auto& aliases = buffer_to_aliases[freed_id];
            for (void* ptr : aliases) {
                auto& alias_shard = alias_shards_[shard_index(ptr)];
                alias_shard.aliases.erase(ptr);
            }
        }

        if (freed_buffers > 0) {
            total_buffers_.fetch_sub(freed_buffers, std::memory_order_relaxed);
            total_buffer_bytes_.fetch_sub(freed_bytes, std::memory_order_relaxed);
        }

        // Step 4: Actually free the memory
        for (auto& [ptr, deleter] : to_delete) {
            if (deleter) deleter(ptr);
        }
    }

    /// @brief Remove alias mappings and decrement buffer refcounts for owner matrices
    /// @note Used when the original owner matrix (not a view) is destroyed
    /// @param alias_ptrs Pointers to remove from alias table and decrement
    void unregister_aliases_as_owner(std::span<void* const> alias_ptrs) {
        // This is the same as unregister_aliases - keeping for semantic clarity
        unregister_aliases(alias_ptrs);
    }
    
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
    
    void clear_all_and_free() {
        for (auto& shard : ptr_shards_) {
            shard.records.for_each_mut([this](void* ptr, PtrRecord& rec) {
                auto deleter = get_deleter(rec.type, rec.custom_deleter);
                if (deleter) deleter(ptr);
            });
            shard.records.clear();
        }
        
        for (auto& shard : alias_shards_) {
            shard.aliases.clear();
        }
        
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
    }
    
    ~Registry() {
#ifndef NDEBUG
        std::size_t leaked_ptrs = ptr_count();
        std::size_t leaked_buffers = buffer_count();
        std::size_t leaked_bytes = total_bytes();
        
        if (leaked_ptrs > 0 || leaked_buffers > 0) {
            std::fprintf(stderr,
                "WARNING: Registry leaked %zu pointers + %zu buffers (%zu bytes total)\n",
                leaked_ptrs, leaked_buffers, leaked_bytes);
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
    return get_registry().unregister_alias(alias_ptr);
}

inline void unregister_aliases(std::span<void* const> alias_ptrs) {
    get_registry().unregister_aliases(alias_ptrs);
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
                get_registry().unregister_alias(ptr_);
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
                    get_registry().unregister_alias(ptr_);
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
