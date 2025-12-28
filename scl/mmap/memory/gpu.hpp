// =============================================================================
// FILE: scl/mmap/memory/gpu.hpp
// BRIEF: GPU memory pool implementation with CUDA/HIP support
// =============================================================================
#pragma once

#include "gpu.h"
#include "../configuration.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"

#include <mutex>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <cstring>
#include <algorithm>

// Conditional GPU headers
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define SCL_HAS_CUDA 1
#else
#define SCL_HAS_CUDA 0
#endif

#if __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#define SCL_HAS_HIP 1
#else
#define SCL_HAS_HIP 0
#endif

namespace scl::mmap::memory {

// =============================================================================
// GPU Error Checking Macros
// =============================================================================

#if SCL_HAS_CUDA
#define SCL_CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw RuntimeError(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

#define SCL_CUDA_CHECK_WARN(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        /* Warning only, do not throw in destructor */ \
    } \
} while(0)
#endif

#if SCL_HAS_HIP
#define SCL_HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        throw RuntimeError(std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)

#define SCL_HIP_CHECK_WARN(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        /* Warning only, do not throw in destructor */ \
    } \
} while(0)
#endif

// =============================================================================
// RAII Device Guard for Exception-Safe Device Switching
// =============================================================================

#if SCL_HAS_CUDA || SCL_HAS_HIP
namespace detail {

class DeviceGuard {
public:
    explicit DeviceGuard(int target_device, GPUBackend backend)
        : backend_(backend), saved_device_(-1) {
#if SCL_HAS_CUDA
        if (backend_ == GPUBackend::CUDA) {
            SCL_CUDA_CHECK(cudaGetDevice(&saved_device_));
            if (target_device != saved_device_) {
                SCL_CUDA_CHECK(cudaSetDevice(target_device));
            } else {
                saved_device_ = -1;  // No restore needed
            }
        }
#endif
#if SCL_HAS_HIP
        if (backend_ == GPUBackend::HIP) {
            SCL_HIP_CHECK(hipGetDevice(&saved_device_));
            if (target_device != saved_device_) {
                SCL_HIP_CHECK(hipSetDevice(target_device));
            } else {
                saved_device_ = -1;  // No restore needed
            }
        }
#endif
    }

    ~DeviceGuard() {
        if (saved_device_ >= 0) {
#if SCL_HAS_CUDA
            if (backend_ == GPUBackend::CUDA) {
                SCL_CUDA_CHECK_WARN(cudaSetDevice(saved_device_));
            }
#endif
#if SCL_HAS_HIP
            if (backend_ == GPUBackend::HIP) {
                SCL_HIP_CHECK_WARN(hipSetDevice(saved_device_));
            }
#endif
        }
    }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    GPUBackend backend_;
    int saved_device_;
};

} // namespace detail
#endif // SCL_HAS_CUDA || SCL_HAS_HIP

// =============================================================================
// GPUPoolConfig Implementation
// =============================================================================

constexpr GPUPoolConfig GPUPoolConfig::default_config() noexcept {
    return GPUPoolConfig{
        .backend = GPUBackend::Auto,
        .device_id = -1,
        .initial_pool_size = 0,
        .max_pool_size = 0,
        .page_size = 2 * 1024 * 1024,
        .enable_async_alloc = true,
        .enable_peer_access = false,
        .reserve_percent = 0.9
    };
}

constexpr GPUPoolConfig GPUPoolConfig::high_throughput() noexcept {
    return GPUPoolConfig{
        .backend = GPUBackend::Auto,
        .device_id = -1,
        .initial_pool_size = 256 * 1024 * 1024,  // 256MB
        .max_pool_size = 0,
        .page_size = 4 * 1024 * 1024,  // 4MB
        .enable_async_alloc = true,
        .enable_peer_access = true,
        .reserve_percent = 0.95
    };
}

constexpr GPUPoolConfig GPUPoolConfig::low_latency() noexcept {
    return GPUPoolConfig{
        .backend = GPUBackend::Auto,
        .device_id = -1,
        .initial_pool_size = 64 * 1024 * 1024,  // 64MB
        .max_pool_size = 512 * 1024 * 1024,
        .page_size = 1 * 1024 * 1024,  // 1MB
        .enable_async_alloc = true,
        .enable_peer_access = false,
        .reserve_percent = 0.8
    };
}

// =============================================================================
// GPUPoolStats Implementation
// =============================================================================

double GPUPoolStats::h2d_bandwidth() const noexcept {
    if (total_h2d_time.count() == 0) return 0.0;
    double seconds = total_h2d_time.count() / 1e9;
    double gb = h2d_bytes / (1024.0 * 1024.0 * 1024.0);
    return gb / seconds;
}

double GPUPoolStats::d2h_bandwidth() const noexcept {
    if (total_d2h_time.count() == 0) return 0.0;
    double seconds = total_d2h_time.count() / 1e9;
    double gb = d2h_bytes / (1024.0 * 1024.0 * 1024.0);
    return gb / seconds;
}

// =============================================================================
// GPUStream Implementation
// =============================================================================

struct GPUStream::Impl {
    void* stream = nullptr;
    bool owns_stream = false;
    GPUBackend backend = GPUBackend::None;

#if SCL_HAS_CUDA
    cudaStream_t cuda_stream() const {
        return static_cast<cudaStream_t>(stream);
    }
#endif

#if SCL_HAS_HIP
    hipStream_t hip_stream() const {
        return static_cast<hipStream_t>(stream);
    }
#endif
};

GPUStream::GPUStream() : impl_(std::make_unique<Impl>()) {
#if SCL_HAS_CUDA
    cudaStream_t s;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess) {
        impl_->stream = s;
        impl_->owns_stream = true;
        impl_->backend = GPUBackend::CUDA;
    }
#elif SCL_HAS_HIP
    hipStream_t s;
    if (hipStreamCreateWithFlags(&s, hipStreamNonBlocking) == hipSuccess) {
        impl_->stream = s;
        impl_->owns_stream = true;
        impl_->backend = GPUBackend::HIP;
    }
#endif
}

GPUStream::GPUStream(std::nullptr_t) : impl_(std::make_unique<Impl>()) {
    impl_->stream = nullptr;
    impl_->owns_stream = false;
#if SCL_HAS_CUDA
    impl_->backend = GPUBackend::CUDA;
#elif SCL_HAS_HIP
    impl_->backend = GPUBackend::HIP;
#else
    impl_->backend = GPUBackend::None;
#endif
}

GPUStream::~GPUStream() {
    if (impl_ && impl_->owns_stream && impl_->stream) {
#if SCL_HAS_CUDA
        if (impl_->backend == GPUBackend::CUDA) {
            cudaStreamDestroy(impl_->cuda_stream());
        }
#endif
#if SCL_HAS_HIP
        if (impl_->backend == GPUBackend::HIP) {
            hipStreamDestroy(impl_->hip_stream());
        }
#endif
    }
}

GPUStream::GPUStream(GPUStream&& other) noexcept = default;
GPUStream& GPUStream::operator=(GPUStream&& other) noexcept = default;

void GPUStream::synchronize() {
    if (!impl_ || !impl_->stream) {
#if SCL_HAS_CUDA
        SCL_CUDA_CHECK(cudaDeviceSynchronize());
#elif SCL_HAS_HIP
        SCL_HIP_CHECK(hipDeviceSynchronize());
#endif
        return;
    }

#if SCL_HAS_CUDA
    if (impl_->backend == GPUBackend::CUDA) {
        SCL_CUDA_CHECK(cudaStreamSynchronize(impl_->cuda_stream()));
    }
#endif
#if SCL_HAS_HIP
    if (impl_->backend == GPUBackend::HIP) {
        SCL_HIP_CHECK(hipStreamSynchronize(impl_->hip_stream()));
    }
#endif
}

bool GPUStream::query() const noexcept {
    if (!impl_) return true;

#if SCL_HAS_CUDA
    if (impl_->backend == GPUBackend::CUDA) {
        return cudaStreamQuery(impl_->cuda_stream()) == cudaSuccess;
    }
#endif
#if SCL_HAS_HIP
    if (impl_->backend == GPUBackend::HIP) {
        return hipStreamQuery(impl_->hip_stream()) == hipSuccess;
    }
#endif
    return true;
}

void* GPUStream::native_handle() const noexcept {
    return impl_ ? impl_->stream : nullptr;
}

bool GPUStream::valid() const noexcept {
    return impl_ && impl_->backend != GPUBackend::None;
}

GPUStream& GPUStream::default_stream() {
    static GPUStream s(nullptr);
    return s;
}

// =============================================================================
// Allocation Tracking
// =============================================================================

struct AllocationInfo {
    std::size_t size;
    GPUMemoryType type;
    std::chrono::steady_clock::time_point alloc_time;
};

// =============================================================================
// GPUMemoryPool::Impl
// =============================================================================

struct GPUMemoryPool::Impl {
    GPUPoolConfig config;
    GPUBackend actual_backend = GPUBackend::None;
    int actual_device_id = 0;
    GPUDeviceInfo device_info_{};

    // Allocation tracking
    std::unordered_map<void*, AllocationInfo> allocations;
    std::mutex alloc_mutex;

    // Free lists (by size class)
    std::unordered_map<std::size_t, std::vector<void*>> device_free_list;
    std::unordered_map<std::size_t, std::vector<void*>> pinned_free_list;
    std::mutex free_list_mutex;

    // Statistics
    std::atomic<std::size_t> total_allocated{0};
    std::atomic<std::size_t> peak_allocated{0};
    std::atomic<std::size_t> total_allocations{0};
    std::atomic<std::size_t> total_frees{0};
    std::atomic<std::size_t> pool_hits{0};
    std::atomic<std::size_t> pool_misses{0};
    std::atomic<std::size_t> h2d_transfers{0};
    std::atomic<std::size_t> d2h_transfers{0};
    std::atomic<std::size_t> h2d_bytes{0};
    std::atomic<std::size_t> d2h_bytes{0};
    std::atomic<std::size_t> total_h2d_time_ns{0};
    std::atomic<std::size_t> total_d2h_time_ns{0};

    explicit Impl(GPUPoolConfig cfg) : config(cfg) {
        detect_backend();
        init_device();
    }

    ~Impl() {
        // Check for active allocations that weren't properly deallocated
        {
            std::lock_guard<std::mutex> lock(alloc_mutex);
            if (!allocations.empty()) {
#ifndef NDEBUG
                std::fprintf(stderr, "WARNING: GPUMemoryPool destructor: %zu allocations "
                             "were not deallocated (total %zu bytes)\n",
                             allocations.size(), total_allocated.load());
#endif
                // Do not free them here - user may have external references
                // Just clear tracking to avoid double-free if user frees later
                allocations.clear();
            }
        }

        // Free all pooled memory
        std::lock_guard<std::mutex> lock(free_list_mutex);

        for (auto& [size, list] : device_free_list) {
            for (void* ptr : list) {
                free_device_memory(ptr);
            }
        }
        device_free_list.clear();

        for (auto& [size, list] : pinned_free_list) {
            for (void* ptr : list) {
                free_pinned_memory(ptr);
            }
        }
        pinned_free_list.clear();
    }

    void detect_backend() {
        if (config.backend == GPUBackend::Auto) {
#if SCL_HAS_CUDA
            int count = 0;
            if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
                actual_backend = GPUBackend::CUDA;
                return;
            }
#endif
#if SCL_HAS_HIP
            int count = 0;
            if (hipGetDeviceCount(&count) == hipSuccess && count > 0) {
                actual_backend = GPUBackend::HIP;
                return;
            }
#endif
            actual_backend = GPUBackend::None;
        } else {
            actual_backend = config.backend;
        }
    }

    void init_device() {
        if (actual_backend == GPUBackend::None) return;

        actual_device_id = config.device_id;
        if (actual_device_id < 0) {
#if SCL_HAS_CUDA
            if (actual_backend == GPUBackend::CUDA) {
                SCL_CUDA_CHECK(cudaGetDevice(&actual_device_id));
            }
#endif
#if SCL_HAS_HIP
            if (actual_backend == GPUBackend::HIP) {
                SCL_HIP_CHECK(hipGetDevice(&actual_device_id));
            }
#endif
        } else {
#if SCL_HAS_CUDA
            if (actual_backend == GPUBackend::CUDA) {
                SCL_CUDA_CHECK(cudaSetDevice(actual_device_id));
            }
#endif
#if SCL_HAS_HIP
            if (actual_backend == GPUBackend::HIP) {
                SCL_HIP_CHECK(hipSetDevice(actual_device_id));
            }
#endif
        }

        query_device_info();
    }

    void query_device_info() {
        device_info_.device_id = actual_device_id;
        std::memset(device_info_.name, 0, sizeof(device_info_.name));

#if SCL_HAS_CUDA
        if (actual_backend == GPUBackend::CUDA) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, actual_device_id) == cudaSuccess) {
                std::strncpy(device_info_.name, prop.name, sizeof(device_info_.name) - 1);
                device_info_.total_memory = prop.totalGlobalMem;
                device_info_.compute_major = prop.major;
                device_info_.compute_minor = prop.minor;
                device_info_.num_sms = prop.multiProcessorCount;
                device_info_.max_threads = prop.maxThreadsPerBlock;
                device_info_.warp_size = prop.warpSize;
                device_info_.supports_managed = prop.managedMemory;
                device_info_.supports_async = prop.asyncEngineCount > 0;
            }

            std::size_t free_mem, total_mem;
            if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
                device_info_.free_memory = free_mem;
            }
        }
#endif

#if SCL_HAS_HIP
        if (actual_backend == GPUBackend::HIP) {
            hipDeviceProp_t prop;
            if (hipGetDeviceProperties(&prop, actual_device_id) == hipSuccess) {
                std::strncpy(device_info_.name, prop.name, sizeof(device_info_.name) - 1);
                device_info_.total_memory = prop.totalGlobalMem;
                device_info_.compute_major = prop.major;
                device_info_.compute_minor = prop.minor;
                device_info_.num_sms = prop.multiProcessorCount;
                device_info_.max_threads = prop.maxThreadsPerBlock;
                device_info_.warp_size = prop.warpSize;
                device_info_.supports_managed = prop.managedMemory;
                device_info_.supports_async = prop.asyncEngineCount > 0;
            }

            std::size_t free_mem, total_mem;
            if (hipMemGetInfo(&free_mem, &total_mem) == hipSuccess) {
                device_info_.free_memory = free_mem;
            }
        }
#endif
    }

    std::size_t round_to_page(std::size_t size) const {
        return (size + config.page_size - 1) & ~(config.page_size - 1);
    }

    void* alloc_device_memory(std::size_t size) {
        void* ptr = nullptr;

#if SCL_HAS_CUDA
        if (actual_backend == GPUBackend::CUDA) {
            if (cudaMalloc(&ptr, size) != cudaSuccess) {
                return nullptr;
            }
        }
#endif

#if SCL_HAS_HIP
        if (actual_backend == GPUBackend::HIP) {
            if (hipMalloc(&ptr, size) != hipSuccess) {
                return nullptr;
            }
        }
#endif

        return ptr;
    }

    void free_device_memory(void* ptr) {
#if SCL_HAS_CUDA
        if (actual_backend == GPUBackend::CUDA) {
            cudaFree(ptr);
        }
#endif
#if SCL_HAS_HIP
        if (actual_backend == GPUBackend::HIP) {
            hipFree(ptr);
        }
#endif
    }

    void* alloc_pinned_memory(std::size_t size) {
        void* ptr = nullptr;

#if SCL_HAS_CUDA
        if (actual_backend == GPUBackend::CUDA) {
            if (cudaMallocHost(&ptr, size) != cudaSuccess) {
                return nullptr;
            }
        }
#endif

#if SCL_HAS_HIP
        if (actual_backend == GPUBackend::HIP) {
            if (hipHostMalloc(&ptr, size, hipHostMallocDefault) != hipSuccess) {
                return nullptr;
            }
        }
#endif

        return ptr;
    }

    void free_pinned_memory(void* ptr) {
#if SCL_HAS_CUDA
        if (actual_backend == GPUBackend::CUDA) {
            cudaFreeHost(ptr);
        }
#endif
#if SCL_HAS_HIP
        if (actual_backend == GPUBackend::HIP) {
            hipHostFree(ptr);
        }
#endif
    }

    void* alloc_managed_memory(std::size_t size) {
        void* ptr = nullptr;

#if SCL_HAS_CUDA
        if (actual_backend == GPUBackend::CUDA) {
            if (cudaMallocManaged(&ptr, size) != cudaSuccess) {
                return nullptr;
            }
        }
#endif

#if SCL_HAS_HIP
        if (actual_backend == GPUBackend::HIP) {
            if (hipMallocManaged(&ptr, size) != hipSuccess) {
                return nullptr;
            }
        }
#endif

        return ptr;
    }

    void track_allocation(void* ptr, std::size_t size, GPUMemoryType type) {
        std::lock_guard<std::mutex> lock(alloc_mutex);
        allocations[ptr] = AllocationInfo{
            size, type, std::chrono::steady_clock::now()
        };

        std::size_t current = total_allocated.fetch_add(size) + size;
        std::size_t peak = peak_allocated.load();
        while (current > peak && !peak_allocated.compare_exchange_weak(peak, current)) {}

        total_allocations.fetch_add(1);
    }

    void untrack_allocation(void* ptr) {
        std::lock_guard<std::mutex> lock(alloc_mutex);
        auto it = allocations.find(ptr);
        if (it != allocations.end()) {
            total_allocated.fetch_sub(it->second.size);
            total_frees.fetch_add(1);
            allocations.erase(it);
        }
    }

    GPUPoolStats get_stats() const {
        return GPUPoolStats{
            .total_allocated = total_allocated.load(),
            .peak_allocated = peak_allocated.load(),
            .total_allocations = total_allocations.load(),
            .total_frees = total_frees.load(),
            .pool_hits = pool_hits.load(),
            .pool_misses = pool_misses.load(),
            .h2d_transfers = h2d_transfers.load(),
            .d2h_transfers = d2h_transfers.load(),
            .h2d_bytes = h2d_bytes.load(),
            .d2h_bytes = d2h_bytes.load(),
            .total_h2d_time = std::chrono::nanoseconds(total_h2d_time_ns.load()),
            .total_d2h_time = std::chrono::nanoseconds(total_d2h_time_ns.load())
        };
    }

    void reset_stats() {
        // Note: Do NOT reset total_allocated and peak_allocated
        // as they track actual memory state, not just statistics
        total_allocations.store(0);
        total_frees.store(0);
        pool_hits.store(0);
        pool_misses.store(0);
        h2d_transfers.store(0);
        d2h_transfers.store(0);
        h2d_bytes.store(0);
        d2h_bytes.store(0);
        total_h2d_time_ns.store(0);
        total_d2h_time_ns.store(0);
    }
};

// =============================================================================
// GPUMemoryPool Implementation
// =============================================================================

GPUMemoryPool::GPUMemoryPool(GPUPoolConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

GPUMemoryPool::~GPUMemoryPool() = default;

GPUMemoryPool::GPUMemoryPool(GPUMemoryPool&& other) noexcept = default;
GPUMemoryPool& GPUMemoryPool::operator=(GPUMemoryPool&& other) noexcept = default;

void* GPUMemoryPool::allocate_device(std::size_t size_bytes) {
    if (size_bytes == 0 || impl_->actual_backend == GPUBackend::None) {
        return nullptr;
    }

    std::size_t rounded_size = impl_->round_to_page(size_bytes);

    // Check free list first
    {
        std::lock_guard<std::mutex> lock(impl_->free_list_mutex);
        auto it = impl_->device_free_list.find(rounded_size);
        if (it != impl_->device_free_list.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            impl_->pool_hits.fetch_add(1);
            impl_->track_allocation(ptr, rounded_size, GPUMemoryType::Device);
            return ptr;
        }
    }

    // Allocate new
    impl_->pool_misses.fetch_add(1);
    void* ptr = impl_->alloc_device_memory(rounded_size);
    if (ptr) {
        impl_->track_allocation(ptr, rounded_size, GPUMemoryType::Device);
    }
    return ptr;
}

void* GPUMemoryPool::allocate_pinned(std::size_t size_bytes) {
    if (size_bytes == 0 || impl_->actual_backend == GPUBackend::None) {
        return nullptr;
    }

    std::size_t rounded_size = impl_->round_to_page(size_bytes);

    // Check free list
    {
        std::lock_guard<std::mutex> lock(impl_->free_list_mutex);
        auto it = impl_->pinned_free_list.find(rounded_size);
        if (it != impl_->pinned_free_list.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            impl_->pool_hits.fetch_add(1);
            impl_->track_allocation(ptr, rounded_size, GPUMemoryType::Pinned);
            return ptr;
        }
    }

    impl_->pool_misses.fetch_add(1);
    void* ptr = impl_->alloc_pinned_memory(rounded_size);
    if (ptr) {
        impl_->track_allocation(ptr, rounded_size, GPUMemoryType::Pinned);
    }
    return ptr;
}

void* GPUMemoryPool::allocate_managed(std::size_t size_bytes) {
    if (size_bytes == 0 || impl_->actual_backend == GPUBackend::None) {
        return nullptr;
    }

    std::size_t rounded_size = impl_->round_to_page(size_bytes);
    void* ptr = impl_->alloc_managed_memory(rounded_size);
    if (ptr) {
        impl_->track_allocation(ptr, rounded_size, GPUMemoryType::Managed);
    }
    return ptr;
}

void GPUMemoryPool::deallocate(void* ptr, std::size_t size_bytes, GPUMemoryType type) {
    if (!ptr || impl_->actual_backend == GPUBackend::None) return;

    std::size_t rounded_size = impl_->round_to_page(size_bytes);
    impl_->untrack_allocation(ptr);

    // Return to free list (except managed)
    if (type == GPUMemoryType::Device) {
        std::lock_guard<std::mutex> lock(impl_->free_list_mutex);
        impl_->device_free_list[rounded_size].push_back(ptr);
    } else if (type == GPUMemoryType::Pinned) {
        std::lock_guard<std::mutex> lock(impl_->free_list_mutex);
        impl_->pinned_free_list[rounded_size].push_back(ptr);
    } else {
        // Managed memory: free immediately
#if SCL_HAS_CUDA
        if (impl_->actual_backend == GPUBackend::CUDA) {
            cudaFree(ptr);
        }
#endif
#if SCL_HAS_HIP
        if (impl_->actual_backend == GPUBackend::HIP) {
            hipFree(ptr);
        }
#endif
    }
}

void GPUMemoryPool::copy_h2d(void* dst, const void* src, std::size_t size_bytes) {
    if (!dst || !src || size_bytes == 0) return;

    auto start = std::chrono::steady_clock::now();

#if SCL_HAS_CUDA
    if (impl_->actual_backend == GPUBackend::CUDA) {
        SCL_CUDA_CHECK(cudaMemcpy(dst, src, size_bytes, cudaMemcpyHostToDevice));
    }
#endif
#if SCL_HAS_HIP
    if (impl_->actual_backend == GPUBackend::HIP) {
        SCL_HIP_CHECK(hipMemcpy(dst, src, size_bytes, hipMemcpyHostToDevice));
    }
#endif

    auto elapsed = std::chrono::steady_clock::now() - start;
    impl_->h2d_transfers.fetch_add(1);
    impl_->h2d_bytes.fetch_add(size_bytes);
    impl_->total_h2d_time_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()
    );
}

void GPUMemoryPool::copy_d2h(void* dst, const void* src, std::size_t size_bytes) {
    if (!dst || !src || size_bytes == 0) return;

    auto start = std::chrono::steady_clock::now();

#if SCL_HAS_CUDA
    if (impl_->actual_backend == GPUBackend::CUDA) {
        SCL_CUDA_CHECK(cudaMemcpy(dst, src, size_bytes, cudaMemcpyDeviceToHost));
    }
#endif
#if SCL_HAS_HIP
    if (impl_->actual_backend == GPUBackend::HIP) {
        SCL_HIP_CHECK(hipMemcpy(dst, src, size_bytes, hipMemcpyDeviceToHost));
    }
#endif

    auto elapsed = std::chrono::steady_clock::now() - start;
    impl_->d2h_transfers.fetch_add(1);
    impl_->d2h_bytes.fetch_add(size_bytes);
    impl_->total_d2h_time_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()
    );
}

void GPUMemoryPool::copy_h2d_async(
    void* dst, const void* src, std::size_t size_bytes, GPUStream& stream
) {
    if (!dst || !src || size_bytes == 0) return;

#if SCL_HAS_CUDA
    if (impl_->actual_backend == GPUBackend::CUDA) {
        SCL_CUDA_CHECK(cudaMemcpyAsync(dst, src, size_bytes, cudaMemcpyHostToDevice,
                        static_cast<cudaStream_t>(stream.native_handle())));
    }
#endif
#if SCL_HAS_HIP
    if (impl_->actual_backend == GPUBackend::HIP) {
        SCL_HIP_CHECK(hipMemcpyAsync(dst, src, size_bytes, hipMemcpyHostToDevice,
                       static_cast<hipStream_t>(stream.native_handle())));
    }
#endif

    // Note: Timing not tracked for async operations as it would require synchronization
    impl_->h2d_transfers.fetch_add(1);
    impl_->h2d_bytes.fetch_add(size_bytes);
}

void GPUMemoryPool::copy_d2h_async(
    void* dst, const void* src, std::size_t size_bytes, GPUStream& stream
) {
    if (!dst || !src || size_bytes == 0) return;

#if SCL_HAS_CUDA
    if (impl_->actual_backend == GPUBackend::CUDA) {
        SCL_CUDA_CHECK(cudaMemcpyAsync(dst, src, size_bytes, cudaMemcpyDeviceToHost,
                        static_cast<cudaStream_t>(stream.native_handle())));
    }
#endif
#if SCL_HAS_HIP
    if (impl_->actual_backend == GPUBackend::HIP) {
        SCL_HIP_CHECK(hipMemcpyAsync(dst, src, size_bytes, hipMemcpyDeviceToHost,
                       static_cast<hipStream_t>(stream.native_handle())));
    }
#endif

    // Note: Timing not tracked for async operations as it would require synchronization
    impl_->d2h_transfers.fetch_add(1);
    impl_->d2h_bytes.fetch_add(size_bytes);
}

void GPUMemoryPool::memset_device(void* ptr, int value, std::size_t size_bytes) {
    if (!ptr || size_bytes == 0) return;

#if SCL_HAS_CUDA
    if (impl_->actual_backend == GPUBackend::CUDA) {
        SCL_CUDA_CHECK(cudaMemset(ptr, value, size_bytes));
    }
#endif
#if SCL_HAS_HIP
    if (impl_->actual_backend == GPUBackend::HIP) {
        SCL_HIP_CHECK(hipMemset(ptr, value, size_bytes));
    }
#endif
}

void GPUMemoryPool::memset_device_async(
    void* ptr, int value, std::size_t size_bytes, GPUStream& stream
) {
    if (!ptr || size_bytes == 0) return;

#if SCL_HAS_CUDA
    if (impl_->actual_backend == GPUBackend::CUDA) {
        SCL_CUDA_CHECK(cudaMemsetAsync(ptr, value, size_bytes,
                        static_cast<cudaStream_t>(stream.native_handle())));
    }
#endif
#if SCL_HAS_HIP
    if (impl_->actual_backend == GPUBackend::HIP) {
        SCL_HIP_CHECK(hipMemsetAsync(ptr, value, size_bytes,
                       static_cast<hipStream_t>(stream.native_handle())));
    }
#endif
}

void GPUMemoryPool::prefetch_to_device(
    void* ptr, std::size_t size_bytes, GPUStream& stream
) {
#if SCL_HAS_CUDA
    if (impl_->actual_backend == GPUBackend::CUDA && impl_->device_info_.supports_managed) {
        cudaMemPrefetchAsync(ptr, size_bytes, impl_->actual_device_id,
                            static_cast<cudaStream_t>(stream.native_handle()));
    }
#endif
#if SCL_HAS_HIP
    if (impl_->actual_backend == GPUBackend::HIP && impl_->device_info_.supports_managed) {
        hipMemPrefetchAsync(ptr, size_bytes, impl_->actual_device_id,
                           static_cast<hipStream_t>(stream.native_handle()));
    }
#endif
    (void)ptr; (void)size_bytes; (void)stream;
}

void GPUMemoryPool::prefetch_to_host(
    void* ptr, std::size_t size_bytes, GPUStream& stream
) {
#if SCL_HAS_CUDA
    if (impl_->actual_backend == GPUBackend::CUDA && impl_->device_info_.supports_managed) {
        cudaMemPrefetchAsync(ptr, size_bytes, cudaCpuDeviceId,
                            static_cast<cudaStream_t>(stream.native_handle()));
    }
#endif
#if SCL_HAS_HIP
    if (impl_->actual_backend == GPUBackend::HIP && impl_->device_info_.supports_managed) {
        hipMemPrefetchAsync(ptr, size_bytes, hipCpuDeviceId,
                           static_cast<hipStream_t>(stream.native_handle()));
    }
#endif
    (void)ptr; (void)size_bytes; (void)stream;
}

int GPUMemoryPool::device_id() const noexcept {
    return impl_->actual_device_id;
}

GPUBackend GPUMemoryPool::backend() const noexcept {
    return impl_->actual_backend;
}

GPUDeviceInfo GPUMemoryPool::device_info() const noexcept {
    return impl_->device_info_;
}

const GPUPoolConfig& GPUMemoryPool::config() const noexcept {
    return impl_->config;
}

GPUPoolStats GPUMemoryPool::stats() const noexcept {
    return impl_->get_stats();
}

void GPUMemoryPool::reset_stats() noexcept {
    impl_->reset_stats();
}

void GPUMemoryPool::trim(std::size_t keep_bytes) {
    std::lock_guard<std::mutex> lock(impl_->free_list_mutex);

    // Calculate current free list size
    std::size_t current_free = 0;
    for (const auto& [size, list] : impl_->device_free_list) {
        current_free += size * list.size();
    }
    for (const auto& [size, list] : impl_->pinned_free_list) {
        current_free += size * list.size();
    }

    // Release memory until we have at most keep_bytes of free memory
    std::size_t to_release = (current_free > keep_bytes) ? (current_free - keep_bytes) : 0;
    std::size_t released = 0;

    // Trim device free list
    for (auto& [size, list] : impl_->device_free_list) {
        while (!list.empty() && released < to_release) {
            impl_->free_device_memory(list.back());
            list.pop_back();
            released += size;
        }
    }

    // Trim pinned free list
    for (auto& [size, list] : impl_->pinned_free_list) {
        while (!list.empty() && released < to_release) {
            impl_->free_pinned_memory(list.back());
            list.pop_back();
            released += size;
        }
    }
}

bool GPUMemoryPool::is_available() noexcept {
    return device_count() > 0;
}

int GPUMemoryPool::device_count() noexcept {
    int count = 0;
#if SCL_HAS_CUDA
    if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
        return count;
    }
#endif
#if SCL_HAS_HIP
    if (hipGetDeviceCount(&count) == hipSuccess && count > 0) {
        return count;
    }
#endif
    return 0;
}

GPUDeviceInfo GPUMemoryPool::get_device_info(int device_id) {
    GPUDeviceInfo info{};
    info.device_id = device_id;

#if SCL_HAS_CUDA
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        std::strncpy(info.name, prop.name, sizeof(info.name) - 1);
        info.total_memory = prop.totalGlobalMem;
        info.compute_major = prop.major;
        info.compute_minor = prop.minor;
        info.num_sms = prop.multiProcessorCount;
        info.max_threads = prop.maxThreadsPerBlock;
        info.warp_size = prop.warpSize;
        info.supports_managed = prop.managedMemory;
        info.supports_async = prop.asyncEngineCount > 0;

        // Use RAII guard for exception-safe device switching
        detail::DeviceGuard guard(device_id, GPUBackend::CUDA);
        std::size_t free_mem, total_mem;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
            info.free_memory = free_mem;
        }
    }
#endif

#if SCL_HAS_HIP
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device_id) == hipSuccess) {
        std::strncpy(info.name, prop.name, sizeof(info.name) - 1);
        info.total_memory = prop.totalGlobalMem;
        info.compute_major = prop.major;
        info.compute_minor = prop.minor;
        info.num_sms = prop.multiProcessorCount;
        info.max_threads = prop.maxThreadsPerBlock;
        info.warp_size = prop.warpSize;
        info.supports_managed = prop.managedMemory;
        info.supports_async = prop.asyncEngineCount > 0;

        // Use RAII guard for exception-safe device switching
        detail::DeviceGuard guard(device_id, GPUBackend::HIP);
        std::size_t free_mem, total_mem;
        if (hipMemGetInfo(&free_mem, &total_mem) == hipSuccess) {
            info.free_memory = free_mem;
        }
    }
#endif

    return info;
}

GPUMemoryPool& GPUMemoryPool::instance() {
    static GPUMemoryPool pool{GPUPoolConfig::default_config()};
    return pool;
}

// =============================================================================
// Free Functions
// =============================================================================

inline const char* gpu_backend_name(GPUBackend backend) noexcept {
    switch (backend) {
        case GPUBackend::None: return "None";
        case GPUBackend::CUDA: return "CUDA";
        case GPUBackend::HIP:  return "HIP";
        case GPUBackend::Auto: return "Auto";
        default:               return "Unknown";
    }
}

inline const char* gpu_memory_type_name(GPUMemoryType type) noexcept {
    switch (type) {
        case GPUMemoryType::Device:     return "Device";
        case GPUMemoryType::Pinned:     return "Pinned";
        case GPUMemoryType::Managed:    return "Managed";
        case GPUMemoryType::HostMapped: return "HostMapped";
        default:                        return "Unknown";
    }
}

} // namespace scl::mmap::memory
