#pragma once

#include "scl/config.hpp"
#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/threading/scheduler.hpp"

#include <vector>
#include <cstring>

// =============================================================================
// FILE: scl/threading/workspace.hpp
// BRIEF: Thread-local workspace management to eliminate per-iteration allocs
// =============================================================================

namespace scl::threading {

// =============================================================================
// Thread-Local Buffer Pool
// =============================================================================

// Pre-allocated buffer that avoids repeated allocations in parallel loops.
// Each thread gets its own buffer indexed by thread rank.
template <typename T>
class WorkspacePool {
public:
    WorkspacePool() = default;

    // Initialize pool for n_threads, each with capacity elements
    void init(size_t n_threads, size_t capacity) {
        n_threads_ = n_threads;
        capacity_ = capacity;

        // Single contiguous allocation for all threads
        total_size_ = n_threads * capacity;
        data_ = scl::memory::aligned_alloc<T>(total_size_, SCL_ALIGNMENT);
    }

    ~WorkspacePool() {
        if (data_) {
            scl::memory::aligned_free(data_, SCL_ALIGNMENT);
        }
    }

    // Non-copyable
    WorkspacePool(const WorkspacePool&) = delete;
    WorkspacePool& operator=(const WorkspacePool&) = delete;

    // Movable
    WorkspacePool(WorkspacePool&& other) noexcept
        : data_(other.data_)
        , n_threads_(other.n_threads_)
        , capacity_(other.capacity_)
        , total_size_(other.total_size_)
    {
        other.data_ = nullptr;
        other.n_threads_ = 0;
        other.capacity_ = 0;
        other.total_size_ = 0;
    }

    WorkspacePool& operator=(WorkspacePool&& other) noexcept {
        if (this != &other) {
            if (data_) scl::memory::aligned_free(data_, SCL_ALIGNMENT);
            data_ = other.data_;
            n_threads_ = other.n_threads_;
            capacity_ = other.capacity_;
            total_size_ = other.total_size_;
            other.data_ = nullptr;
            other.n_threads_ = 0;
            other.capacity_ = 0;
            other.total_size_ = 0;
        }
        return *this;
    }

    // Get buffer for thread with given rank
    SCL_FORCE_INLINE T* get(size_t thread_rank) noexcept {
        return data_ + thread_rank * capacity_;
    }

    SCL_FORCE_INLINE const T* get(size_t thread_rank) const noexcept {
        return data_ + thread_rank * capacity_;
    }

    // Get Array view for thread
    SCL_FORCE_INLINE Array<T> span(size_t thread_rank) noexcept {
        return Array<T>(get(thread_rank), capacity_);
    }

    // Zero-initialize buffer for thread
    SCL_FORCE_INLINE void zero(size_t thread_rank) noexcept {
        std::memset(get(thread_rank), 0, capacity_ * sizeof(T));
    }

    size_t capacity() const noexcept { return capacity_; }
    size_t n_threads() const noexcept { return n_threads_; }

private:
    T* data_ = nullptr;
    size_t n_threads_ = 0;
    size_t capacity_ = 0;
    size_t total_size_ = 0;
};

// =============================================================================
// Dual Buffer Pool (for algorithms needing two buffers per thread)
// =============================================================================

template <typename T>
class DualWorkspacePool {
public:
    DualWorkspacePool() = default;

    void init(size_t n_threads, size_t capacity) {
        pool1_.init(n_threads, capacity);
        pool2_.init(n_threads, capacity);
    }

    SCL_FORCE_INLINE T* get1(size_t thread_rank) noexcept { return pool1_.get(thread_rank); }
    SCL_FORCE_INLINE T* get2(size_t thread_rank) noexcept { return pool2_.get(thread_rank); }

    SCL_FORCE_INLINE Array<T> span1(size_t thread_rank) noexcept { return pool1_.span(thread_rank); }
    SCL_FORCE_INLINE Array<T> span2(size_t thread_rank) noexcept { return pool2_.span(thread_rank); }

    SCL_FORCE_INLINE void zero_both(size_t thread_rank) noexcept {
        pool1_.zero(thread_rank);
        pool2_.zero(thread_rank);
    }

    size_t capacity() const noexcept { return pool1_.capacity(); }
    size_t n_threads() const noexcept { return pool1_.n_threads(); }

private:
    WorkspacePool<T> pool1_;
    WorkspacePool<T> pool2_;
};

// =============================================================================
// Variable-Size Buffer Pool (for variable-length work items)
// =============================================================================

template <typename T>
class DynamicWorkspacePool {
public:
    struct ThreadBuffer {
        T* data = nullptr;
        size_t capacity = 0;
        size_t size = 0;

        SCL_FORCE_INLINE void clear() noexcept { size = 0; }

        SCL_FORCE_INLINE void push_back(T val) noexcept {
            data[size++] = val;
        }

        SCL_FORCE_INLINE T& operator[](size_t i) noexcept { return data[i]; }
        SCL_FORCE_INLINE const T& operator[](size_t i) const noexcept { return data[i]; }

        SCL_FORCE_INLINE T* begin() noexcept { return data; }
        SCL_FORCE_INLINE T* end() noexcept { return data + size; }
        SCL_FORCE_INLINE const T* begin() const noexcept { return data; }
        SCL_FORCE_INLINE const T* end() const noexcept { return data + size; }

        SCL_FORCE_INLINE bool empty() const noexcept { return size == 0; }
    };

    DynamicWorkspacePool() = default;

    void init(size_t n_threads, size_t max_capacity) {
        n_threads_ = n_threads;
        max_capacity_ = max_capacity;

        data_ = scl::memory::aligned_alloc<T>(n_threads * max_capacity, SCL_ALIGNMENT);
        buffers_.resize(n_threads);

        for (size_t i = 0; i < n_threads; ++i) {
            buffers_[i].data = data_ + i * max_capacity;
            buffers_[i].capacity = max_capacity;
            buffers_[i].size = 0;
        }
    }

    ~DynamicWorkspacePool() {
        if (data_) {
            scl::memory::aligned_free(data_, SCL_ALIGNMENT);
        }
    }

    // Non-copyable
    DynamicWorkspacePool(const DynamicWorkspacePool&) = delete;
    DynamicWorkspacePool& operator=(const DynamicWorkspacePool&) = delete;

    // Movable
    DynamicWorkspacePool(DynamicWorkspacePool&& other) noexcept
        : data_(other.data_)
        , buffers_(std::move(other.buffers_))
        , n_threads_(other.n_threads_)
        , max_capacity_(other.max_capacity_)
    {
        other.data_ = nullptr;
        other.n_threads_ = 0;
        other.max_capacity_ = 0;
    }

    DynamicWorkspacePool& operator=(DynamicWorkspacePool&& other) noexcept {
        if (this != &other) {
            if (data_) scl::memory::aligned_free(data_, SCL_ALIGNMENT);
            data_ = other.data_;
            buffers_ = std::move(other.buffers_);
            n_threads_ = other.n_threads_;
            max_capacity_ = other.max_capacity_;
            other.data_ = nullptr;
            other.n_threads_ = 0;
            other.max_capacity_ = 0;
        }
        return *this;
    }

    SCL_FORCE_INLINE ThreadBuffer& get(size_t thread_rank) noexcept {
        return buffers_[thread_rank];
    }

    size_t n_threads() const noexcept { return n_threads_; }
    size_t max_capacity() const noexcept { return max_capacity_; }

private:
    T* data_ = nullptr;
    std::vector<ThreadBuffer> buffers_;
    size_t n_threads_ = 0;
    size_t max_capacity_ = 0;
};

// =============================================================================
// Heap Buffer Pool (for k-heap algorithms)
// =============================================================================

template <typename T>
class HeapWorkspacePool {
public:
    struct HeapBuffer {
        T* data = nullptr;
        size_t k = 0;
        size_t count = 0;

        SCL_FORCE_INLINE void clear() noexcept { count = 0; }

        SCL_FORCE_INLINE T max_val() const noexcept {
            return (count > 0) ? data[0] : std::numeric_limits<T>::max();
        }
    };

    HeapWorkspacePool() = default;

    void init(size_t n_threads, size_t k) {
        n_threads_ = n_threads;
        k_ = k;

        data_ = scl::memory::aligned_alloc<T>(n_threads * k, SCL_ALIGNMENT);
        buffers_.resize(n_threads);

        for (size_t i = 0; i < n_threads; ++i) {
            buffers_[i].data = data_ + i * k;
            buffers_[i].k = k;
            buffers_[i].count = 0;
        }
    }

    ~HeapWorkspacePool() {
        if (data_) {
            scl::memory::aligned_free(data_, SCL_ALIGNMENT);
        }
    }

    // Non-copyable
    HeapWorkspacePool(const HeapWorkspacePool&) = delete;
    HeapWorkspacePool& operator=(const HeapWorkspacePool&) = delete;

    SCL_FORCE_INLINE HeapBuffer& get(size_t thread_rank) noexcept {
        return buffers_[thread_rank];
    }

    size_t n_threads() const noexcept { return n_threads_; }
    size_t k() const noexcept { return k_; }

private:
    T* data_ = nullptr;
    std::vector<HeapBuffer> buffers_;
    size_t n_threads_ = 0;
    size_t k_ = 0;
};

// =============================================================================
// Utility: Get current thread rank in parallel region
// =============================================================================

SCL_FORCE_INLINE size_t get_thread_rank() noexcept {
#if defined(SCL_USE_OPENMP)
    return static_cast<size_t>(omp_get_thread_num());
#else
    // For BS thread pool or serial mode, caller must track rank
    return 0;
#endif
}

SCL_FORCE_INLINE size_t get_num_threads_runtime() noexcept {
#if defined(SCL_USE_OPENMP)
    return static_cast<size_t>(omp_get_num_threads());
#else
    return Scheduler::get_num_threads();
#endif
}

} // namespace scl::threading
