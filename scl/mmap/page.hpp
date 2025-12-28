#pragma once

#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/mmap/configuration.hpp"

#include <cstddef>
#include <cstring>
#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <functional>
#include <thread>

// =============================================================================
// FILE: scl/mmap/page.hpp
// BRIEF: Optimized Page, PagePool with Sharding and Memory Reclamation
// =============================================================================

namespace scl::mmap {

// =============================================================================
// Page: Physical Memory Block with Identity
// =============================================================================

struct alignas(64) Page {
    std::byte data[kPageSize];
    
    std::size_t file_id;
    std::size_t page_offset;
    
    std::atomic<std::uint32_t> refcount{0};
    std::atomic<bool> dirty{false};
    
    Page* next_free{nullptr};
    
    Page() noexcept : file_id(0), page_offset(0) {}
    
    SCL_FORCE_INLINE void clear() noexcept {
        scl::memory::zero(Array<std::byte>(data, kPageSize));
    }
    
    template <typename T>
    SCL_NODISCARD SCL_FORCE_INLINE T* as() noexcept {
        return reinterpret_cast<T*>(data);
    }
    
    template <typename T>
    SCL_NODISCARD SCL_FORCE_INLINE const T* as() const noexcept {
        return reinterpret_cast<const T*>(data);
    }
    
    void addref() noexcept {
        refcount.fetch_add(1, std::memory_order_relaxed);
    }
    
    std::uint32_t release() noexcept {
        return refcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
    }
    
    std::uint32_t get_refcount() const noexcept {
        return refcount.load(std::memory_order_relaxed);
    }
};

static_assert(sizeof(Page) >= kPageSize);
static_assert(alignof(Page) == 64);

// =============================================================================
// PageKey: Deduplication Key with Improved Hashing
// =============================================================================

struct PageKey {
    std::size_t file_id;
    std::size_t page_offset;
    
    bool operator==(const PageKey& other) const noexcept {
        return file_id == other.file_id && page_offset == other.page_offset;
    }
};

struct PageKeyHash {
    SCL_FORCE_INLINE std::size_t operator()(const PageKey& key) const noexcept {
        constexpr std::uint64_t kFNV64Prime = 0x100000001b3ULL;
        constexpr std::uint64_t kFNV64Offset = 0xcbf29ce484222325ULL;
        
        std::uint64_t hash = kFNV64Offset;
        hash ^= key.file_id;
        hash *= kFNV64Prime;
        hash ^= key.page_offset;
        hash *= kFNV64Prime;
        
        return static_cast<std::size_t>(hash);
    }
};

// =============================================================================
// I/O Callbacks
// =============================================================================

using LoadCallback = std::function<void(std::size_t page_idx, std::byte* dest)>;
using WriteCallback = std::function<void(std::size_t page_idx, const std::byte* src)>;

// =============================================================================
// PageStore: I/O Backend with File Identity
// =============================================================================

class PageStore {
private:
    std::size_t file_id_;
    std::size_t num_pages_;
    std::size_t total_bytes_;
    
    LoadCallback load_cb_;
    WriteCallback write_cb_;

public:
    PageStore(std::size_t file_id, std::size_t total_bytes,
              LoadCallback load_cb, WriteCallback write_cb = nullptr)
        : file_id_(file_id)
        , num_pages_(bytes_to_pages(total_bytes))
        , total_bytes_(total_bytes)
        , load_cb_(std::move(load_cb))
        , write_cb_(std::move(write_cb))
    {
        if (file_id == 0) {
            throw ValueError("PageStore: file_id cannot be 0");
        }
        if (!load_cb_) {
            throw ValueError("PageStore: load callback cannot be null");
        }
        if (total_bytes == 0) {
            throw ValueError("PageStore: total_bytes cannot be 0");
        }
        if (num_pages_ == 0) {
            throw ValueError("PageStore: num_pages cannot be 0");
        }
        constexpr std::size_t kMaxPages = (SIZE_MAX / kPageSize);
        if (num_pages_ > kMaxPages) {
            throw ValueError("PageStore: num_pages exceeds maximum");
        }
    }
    
    SCL_NODISCARD std::size_t file_id() const noexcept { return file_id_; }
    SCL_NODISCARD std::size_t num_pages() const noexcept { return num_pages_; }
    SCL_NODISCARD std::size_t total_bytes() const noexcept { return total_bytes_; }
    
    void load(std::size_t page_idx, Page* dest) {
        if (page_idx >= num_pages_) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: PageStore::load page_idx=%zu >= num_pages_=%zu\n",
                        page_idx, num_pages_);
#endif
            return;
        }
        if (!dest) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: PageStore::load null dest pointer\n");
#endif
            return;
        }
        // Copy callback to local variable for thread safety
        auto cb = load_cb_;
        if (!cb) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: PageStore::load null callback\n");
#endif
            return;
        }
        cb(page_idx, dest->data);
    }

    void write(std::size_t page_idx, const Page* src) {
        if (page_idx >= num_pages_) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: PageStore::write page_idx=%zu >= num_pages_=%zu\n",
                        page_idx, num_pages_);
#endif
            return;
        }
        if (!src) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: PageStore::write null src pointer\n");
#endif
            return;
        }
        // Copy callback to local variable for thread safety
        auto cb = write_cb_;
        if (cb) {
            cb(page_idx, src->data);
        }
    }
    
    PageStore(PageStore&&) noexcept = default;
    PageStore& operator=(PageStore&&) noexcept = default;
};

inline std::size_t generate_file_id() noexcept {
    static std::atomic<std::size_t> counter{1};
    // Mix counter with thread ID hash to improve uniqueness and reduce collision risk
    std::size_t id = counter.fetch_add(1, std::memory_order_relaxed);
    std::size_t thread_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return id ^ (thread_hash << 16) ^ (thread_hash >> 48);
}

// =============================================================================
// Sharded Page Map for Concurrent Access
// =============================================================================

namespace detail {

inline std::size_t get_page_shard_count() noexcept {
    std::size_t n = std::thread::hardware_concurrency();
    if (n == 0) n = 16;
    if (n < 8) return 8;
    if (n > 128) return 128;
    return (n + 7) & ~7ull;
}

template <typename K>
struct KeyTraits;

template <>
struct KeyTraits<PageKey> {
    static PageKey empty() noexcept { return PageKey{0, SIZE_MAX}; }
    static PageKey tombstone() noexcept { return PageKey{SIZE_MAX, SIZE_MAX}; }
    
    static bool is_empty(const PageKey& k) noexcept {
        return k.file_id == 0 && k.page_offset == SIZE_MAX;
    }
    
    static bool is_tombstone(const PageKey& k) noexcept {
        return k.file_id == SIZE_MAX && k.page_offset == SIZE_MAX;
    }
};

template <typename K, typename V, typename Hash = std::hash<K>>
class alignas(64) ConcurrentPageMap {
    static constexpr std::size_t kInitialCapacity = 256;
    static constexpr double kMaxLoadFactor = 0.75;
    
    struct Slot {
        std::atomic<K> key;
        std::atomic<V> value;
        
        Slot() noexcept : key(KeyTraits<K>::empty()), value(nullptr) {}
    };
    
    std::vector<Slot> slots_;
    std::atomic<std::size_t> size_{0};
    std::size_t capacity_;
    mutable std::shared_mutex rehash_mutex_;
    Hash hasher_;

    SCL_FORCE_INLINE std::size_t hash(const K& key) const noexcept {
        return hasher_(key);
    }
    
    SCL_FORCE_INLINE std::size_t probe(std::size_t h, std::size_t i) const noexcept {
        return (h + i * i) & (capacity_ - 1);
    }
    
    void rehash_internal(std::size_t new_cap) {
        std::vector<Slot> old_slots = std::move(slots_);
        capacity_ = new_cap;
        slots_.resize(new_cap);
        size_.store(0, std::memory_order_relaxed);
        
        for (auto& slot : old_slots) {
            K k = slot.key.load(std::memory_order_relaxed);
            V v = slot.value.load(std::memory_order_relaxed);
            
            if (!KeyTraits<K>::is_empty(k) && !KeyTraits<K>::is_tombstone(k) && v) {
                insert_internal(k, v);
            }
        }
    }
    
    bool insert_internal(const K& key, V value) {
        if (!value) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: ConcurrentPageMap::insert_internal null value\n");
#endif
            return false;
        }
        
        std::size_t h = hash(key);
        for (std::size_t i = 0; i < capacity_; ++i) {
            std::size_t idx = probe(h, i);
            if (idx >= capacity_) {
#ifndef NDEBUG
                std::fprintf(stderr, "ERROR: probe() returned idx=%zu >= capacity=%zu\n", 
                            idx, capacity_);
#endif
                return false;
            }
            
            K expected = slots_[idx].key.load(std::memory_order_acquire);
            
            if (KeyTraits<K>::is_empty(expected) || KeyTraits<K>::is_tombstone(expected)) {
                slots_[idx].value.store(value, std::memory_order_release);
                
                K old_key = expected;
                if (slots_[idx].key.compare_exchange_strong(old_key, key, 
                        std::memory_order_release, std::memory_order_acquire)) {
                    size_.fetch_add(1, std::memory_order_relaxed);
                    return true;
                }
                
                V our_value = value;
                slots_[idx].value.compare_exchange_strong(our_value, nullptr,
                    std::memory_order_relaxed, std::memory_order_relaxed);
                expected = slots_[idx].key.load(std::memory_order_acquire);
            }
            
            if (expected == key) {
                slots_[idx].value.store(value, std::memory_order_release);
                return true;
            }
        }
        return false;
    }

public:
    ConcurrentPageMap() : capacity_(kInitialCapacity) {
        slots_.resize(capacity_);
    }
    
    bool insert(const K& key, V value) {
        if (KeyTraits<K>::is_empty(key) || KeyTraits<K>::is_tombstone(key)) return false;
        
        std::shared_lock read_lock(rehash_mutex_);
        
        if (size_.load(std::memory_order_relaxed) > capacity_ * kMaxLoadFactor) {
            read_lock.unlock();
            std::unique_lock write_lock(rehash_mutex_);
            if (size_.load(std::memory_order_relaxed) > capacity_ * kMaxLoadFactor) {
                rehash_internal(capacity_ * 2);
            }
            return insert_internal(key, value);
        }
        
        return insert_internal(key, value);
    }
    
    bool find(const K& key, V& out) const {
        if (KeyTraits<K>::is_empty(key) || KeyTraits<K>::is_tombstone(key)) return false;
        
        std::shared_lock lock(rehash_mutex_);
        std::size_t h = hash(key);
        
        for (std::size_t i = 0; i < capacity_; ++i) {
            std::size_t idx = probe(h, i);
            K k = slots_[idx].key.load(std::memory_order_acquire);
            
            if (k == key) {
                out = slots_[idx].value.load(std::memory_order_acquire);
                return out != nullptr;
            }
            if (KeyTraits<K>::is_empty(k)) {
                return false;
            }
        }
        return false;
    }
    
    bool erase(const K& key) {
        if (KeyTraits<K>::is_empty(key) || KeyTraits<K>::is_tombstone(key)) return false;
        
        std::unique_lock lock(rehash_mutex_);
        std::size_t h = hash(key);
        
        for (std::size_t i = 0; i < capacity_; ++i) {
            std::size_t idx = probe(h, i);
            K k = slots_[idx].key.load(std::memory_order_acquire);
            
            if (k == key) {
                slots_[idx].value.store(nullptr, std::memory_order_release);
                slots_[idx].key.store(KeyTraits<K>::tombstone(), std::memory_order_release);
                size_.fetch_sub(1, std::memory_order_relaxed);
                return true;
            }
            if (KeyTraits<K>::is_empty(k)) {
                return false;
            }
        }
        return false;
    }
    
    std::size_t size() const noexcept {
        return size_.load(std::memory_order_relaxed);
    }
    
    // Note: clear() does NOT release/free the V pointers.
    // Ownership of pointed-to objects must be managed externally.
    // In GlobalPagePool, Pages are owned by page_chunks_, not this map.
    void clear() {
        std::unique_lock lock(rehash_mutex_);
        K empty = KeyTraits<K>::empty();
        for (auto& slot : slots_) {
            slot.value.store(nullptr, std::memory_order_release);
            slot.key.store(empty, std::memory_order_release);
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        size_.store(0, std::memory_order_relaxed);
    }
};

} // namespace detail

// =============================================================================
// GlobalPagePool: Sharded Pool with Memory Reclamation
// =============================================================================

class GlobalPagePool {
private:
    static constexpr std::size_t kCacheLineSize = 64;
    static constexpr std::size_t kPageChunkSize = 64;
    static constexpr std::size_t kMaxPages = 1024 * 1024;
    
    struct alignas(kCacheLineSize) PageShard {
        detail::ConcurrentPageMap<PageKey, Page*> page_map;
        mutable std::mutex mutex;
    };
    
    std::vector<PageShard> shards_;
    std::size_t num_shards_;
    
    std::vector<std::unique_ptr<Page[]>> page_chunks_;
    std::mutex chunks_lock_;
    
    Page* free_list_head_{nullptr};
    std::mutex free_list_lock_;
    std::atomic<std::size_t> free_count_{0};
    
    std::atomic<std::size_t> total_allocated_{0};
    std::atomic<std::size_t> total_active_{0};
    
    GlobalPagePool() 
        : num_shards_(detail::get_page_shard_count())
        , shards_(num_shards_) 
    {
        if (!allocate_chunk()) {
            throw RuntimeError("GlobalPagePool: Failed to allocate initial chunk");
        }
    }

public:
    ~GlobalPagePool() = default;
    
    GlobalPagePool(const GlobalPagePool&) = delete;
    GlobalPagePool& operator=(const GlobalPagePool&) = delete;
    
    static GlobalPagePool& instance() {
        static GlobalPagePool pool;
        return pool;
    }
    
    Page* get_or_create(std::size_t file_id, std::size_t page_offset) {
        if (file_id == 0) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: Invalid file_id=0 in get_or_create\n");
#endif
            return nullptr;
        }
        
        PageKey key{file_id, page_offset};
        auto& shard = shard_for(key);
        
        std::lock_guard<std::mutex> lock(shard.mutex);
        
        Page* page = nullptr;
        if (shard.page_map.find(key, page) && page) {
            std::uint32_t old_count = page->refcount.load(std::memory_order_acquire);
            if (old_count > 0) {
                page->addref();
                return page;
            }
            shard.page_map.erase(key);
            total_active_.fetch_sub(1, std::memory_order_relaxed);
            free_page(page);
        }
        
        page = allocate_page();
        if (!page) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: Failed to allocate page\n");
#endif
            return nullptr;
        }
        
        page->file_id = file_id;
        page->page_offset = page_offset;
        page->refcount.store(1, std::memory_order_relaxed);
        
        shard.page_map.insert(key, page);
        total_active_.fetch_add(1, std::memory_order_relaxed);
        
        return page;
    }
    
    void release(Page* page) {
        if (!page) return;
        
        while (true) {
            std::uint32_t old_count = page->refcount.load(std::memory_order_acquire);
            
            if (old_count == 0) {
#ifndef NDEBUG
                std::fprintf(stderr, "ERROR: Double release detected\n");
#endif
                return;
            }
            
            if (old_count == 1) {
                // Read page identity before acquiring lock
                std::size_t file_id = page->file_id;
                std::size_t page_offset = page->page_offset;

                // Validate page hasn't been freed (file_id=0 indicates freed page)
                if (file_id == 0) {
#ifndef NDEBUG
                    std::fprintf(stderr, "ERROR: Releasing already-freed page\n");
#endif
                    return;
                }

                PageKey key{file_id, page_offset};
                auto& shard = shard_for(key);
                std::lock_guard<std::mutex> lock(shard.mutex);

                // Re-validate page identity after acquiring lock (ABA protection)
                if (page->file_id != file_id || page->page_offset != page_offset) {
                    continue;
                }

                std::uint32_t current = page->refcount.load(std::memory_order_acquire);
                if (current != old_count) {
                    continue;
                }

                if (page->refcount.compare_exchange_strong(old_count, 0,
                        std::memory_order_acq_rel, std::memory_order_acquire)) {
                    shard.page_map.erase(key);
                    total_active_.fetch_sub(1, std::memory_order_relaxed);
                    free_page(page);
                    return;
                }
            } else {
                if (page->refcount.compare_exchange_weak(old_count, old_count - 1,
                        std::memory_order_acq_rel, std::memory_order_acquire)) {
                    return;
                }
            }
        }
    }
    
    SCL_NODISCARD std::size_t total_allocated() const noexcept {
        return total_allocated_.load(std::memory_order_relaxed);
    }
    
    SCL_NODISCARD std::size_t active_pages() const noexcept {
        return total_active_.load(std::memory_order_relaxed);
    }
    
    SCL_NODISCARD std::size_t free_pages() const noexcept {
        return free_count_.load(std::memory_order_relaxed);
    }

private:
    PageShard& shard_for(const PageKey& key) noexcept {
        PageKeyHash hasher;
        return shards_[hasher(key) % num_shards_];
    }
    
    const PageShard& shard_for(const PageKey& key) const noexcept {
        PageKeyHash hasher;
        return shards_[hasher(key) % num_shards_];
    }
    
    Page* allocate_page() {
        {
            std::lock_guard<std::mutex> guard(free_list_lock_);
            if (free_list_head_) {
                Page* page = free_list_head_;
                free_list_head_ = page->next_free;
                page->next_free = nullptr;
                free_count_.fetch_sub(1, std::memory_order_relaxed);
                return page;
            }
        }
        
        return allocate_from_chunk();
    }
    
    void free_page(Page* page) {
        if (page->dirty.load(std::memory_order_acquire)) {
#ifndef NDEBUG
            std::fprintf(stderr, "WARNING: Freeing dirty page (file_id=%zu, offset=%zu)\n",
                        page->file_id, page->page_offset);
#endif
        }
        
        page->clear();
        page->dirty.store(false, std::memory_order_relaxed);
        page->file_id = 0;
        page->page_offset = 0;
        
        std::lock_guard<std::mutex> guard(free_list_lock_);
        page->next_free = free_list_head_;
        free_list_head_ = page;
        free_count_.fetch_add(1, std::memory_order_relaxed);
    }
    
    Page* allocate_from_chunk() {
        std::lock_guard<std::mutex> guard(chunks_lock_);
        
        const std::size_t idx = total_allocated_.load(std::memory_order_relaxed);
        const std::size_t chunk_idx = idx / kPageChunkSize;
        const std::size_t page_in_chunk = idx % kPageChunkSize;
        
        if (chunk_idx >= page_chunks_.size()) {
            if (!allocate_chunk()) {
                return nullptr;
            }
            
            if (chunk_idx >= page_chunks_.size()) {
#ifndef NDEBUG
                std::fprintf(stderr, "ERROR: Chunk allocation succeeded but chunk_idx still invalid\n");
#endif
                return nullptr;
            }
        }
        
        if (page_in_chunk >= kPageChunkSize) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: page_in_chunk=%zu >= kPageChunkSize=%zu\n",
                        page_in_chunk, kPageChunkSize);
#endif
            return nullptr;
        }
        
        total_allocated_.fetch_add(1, std::memory_order_relaxed);
        
        return &page_chunks_[chunk_idx][page_in_chunk];
    }
    
    bool allocate_chunk() {
        if (page_chunks_.size() >= kMaxPages / kPageChunkSize) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: GlobalPagePool reached maximum capacity (%zu pages)\n",
                        kMaxPages);
#endif
            return false;
        }
        
        try {
            auto chunk = std::make_unique<Page[]>(kPageChunkSize);
            page_chunks_.push_back(std::move(chunk));
            return true;
        } catch (...) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: Failed to allocate page chunk\n");
#endif
            return false;
        }
    }
};

} // namespace scl::mmap

namespace std {
    template<>
    struct hash<scl::mmap::PageKey> {
        std::size_t operator()(const scl::mmap::PageKey& key) const noexcept {
            return scl::mmap::PageKeyHash{}(key);
        }
    };
}

