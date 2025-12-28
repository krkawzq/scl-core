// =============================================================================
// FILE: scl/mmap/memory/gpu.h
// BRIEF: API reference for GPU memory pool with CUDA/HIP support
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <functional>
#include <memory>
#include <span>

namespace scl::mmap::memory {

/* =============================================================================
 * ENUM: GPUBackend
 * =============================================================================
 * SUMMARY:
 *     GPU compute backend selection.
 *
 * VALUES:
 *     None   - No GPU support (fallback to host)
 *     CUDA   - NVIDIA CUDA
 *     HIP    - AMD ROCm HIP
 *     Auto   - Detect available backend
 * -------------------------------------------------------------------------- */
enum class GPUBackend : std::uint8_t {
    None,
    CUDA,
    HIP,
    Auto
};

/* =============================================================================
 * ENUM: GPUMemoryType
 * =============================================================================
 * SUMMARY:
 *     GPU memory allocation type.
 *
 * VALUES:
 *     Device      - GPU device memory (fastest for GPU access)
 *     Pinned      - Host memory pinned for DMA (fast H2D/D2H)
 *     Managed     - Unified memory (automatic migration)
 *     HostMapped  - Host memory mapped to GPU address space
 *
 * SELECTION GUIDE:
 *     Device:     Best for GPU-only data
 *     Pinned:     Best for frequent H2D/D2H transfers
 *     Managed:    Best for mixed CPU/GPU access (convenience)
 *     HostMapped: Best for occasional GPU access to large host data
 * -------------------------------------------------------------------------- */
enum class GPUMemoryType : std::uint8_t {
    Device,
    Pinned,
    Managed,
    HostMapped
};

/* =============================================================================
 * STRUCT: GPUDeviceInfo
 * =============================================================================
 * SUMMARY:
 *     Information about a GPU device.
 *
 * FIELDS:
 *     device_id       - Device index (0, 1, ...)
 *     name            - Device name string
 *     total_memory    - Total device memory in bytes
 *     free_memory     - Available device memory in bytes
 *     compute_major   - Compute capability major version
 *     compute_minor   - Compute capability minor version
 *     num_sms         - Number of streaming multiprocessors
 *     max_threads     - Max threads per block
 *     warp_size       - Threads per warp (32 for CUDA, 64 for HIP)
 *     supports_managed - Device supports unified memory
 *     supports_async  - Device supports async memory operations
 * -------------------------------------------------------------------------- */
struct GPUDeviceInfo {
    int device_id;
    char name[256];
    std::size_t total_memory;
    std::size_t free_memory;
    int compute_major;
    int compute_minor;
    int num_sms;
    int max_threads;
    int warp_size;
    bool supports_managed;
    bool supports_async;
};

/* =============================================================================
 * STRUCT: GPUPoolConfig
 * =============================================================================
 * SUMMARY:
 *     Configuration for GPU memory pool.
 *
 * FIELDS:
 *     backend             - GPU backend to use
 *     device_id           - Target device (-1 = current device)
 *     initial_pool_size   - Initial pool size in bytes (0 = on-demand)
 *     max_pool_size       - Maximum pool size (0 = unlimited)
 *     page_size           - Allocation granularity
 *     enable_async_alloc  - Use async memory allocation (CUDA 11.2+)
 *     enable_peer_access  - Enable multi-GPU peer access
 *     reserve_percent     - Percent of device memory to reserve
 * -------------------------------------------------------------------------- */
struct GPUPoolConfig {
    GPUBackend backend = GPUBackend::Auto;
    int device_id = -1;
    std::size_t initial_pool_size = 0;
    std::size_t max_pool_size = 0;
    std::size_t page_size = 2 * 1024 * 1024;  // 2MB default
    bool enable_async_alloc = true;
    bool enable_peer_access = false;
    double reserve_percent = 0.9;

    /* -------------------------------------------------------------------------
     * FACTORY: default_config
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Default configuration with auto-detection.
     * ---------------------------------------------------------------------- */
    static constexpr GPUPoolConfig default_config() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: high_throughput
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration optimized for high transfer throughput.
     * ---------------------------------------------------------------------- */
    static constexpr GPUPoolConfig high_throughput() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: low_latency
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration optimized for low latency access.
     * ---------------------------------------------------------------------- */
    static constexpr GPUPoolConfig low_latency() noexcept;
};

/* =============================================================================
 * STRUCT: GPUPoolStats
 * =============================================================================
 * SUMMARY:
 *     Statistics for GPU memory pool.
 *
 * FIELDS:
 *     total_allocated   - Total bytes currently allocated
 *     peak_allocated    - Peak allocation high watermark
 *     total_allocations - Number of allocations made
 *     total_frees       - Number of deallocations made
 *     pool_hits         - Allocations satisfied from pool
 *     pool_misses       - Allocations requiring new device allocation
 *     h2d_transfers     - Host to device transfer count
 *     d2h_transfers     - Device to host transfer count
 *     h2d_bytes         - Total bytes transferred H2D
 *     d2h_bytes         - Total bytes transferred D2H
 *     total_h2d_time    - Cumulative H2D transfer time
 *     total_d2h_time    - Cumulative D2H transfer time
 * -------------------------------------------------------------------------- */
struct GPUPoolStats {
    std::size_t total_allocated;
    std::size_t peak_allocated;
    std::size_t total_allocations;
    std::size_t total_frees;
    std::size_t pool_hits;
    std::size_t pool_misses;
    std::size_t h2d_transfers;
    std::size_t d2h_transfers;
    std::size_t h2d_bytes;
    std::size_t d2h_bytes;
    std::chrono::nanoseconds total_h2d_time;
    std::chrono::nanoseconds total_d2h_time;

    /* -------------------------------------------------------------------------
     * METHOD: h2d_bandwidth
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Average H2D bandwidth in GB/s.
     * ---------------------------------------------------------------------- */
    double h2d_bandwidth() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: d2h_bandwidth
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Average D2H bandwidth in GB/s.
     * ---------------------------------------------------------------------- */
    double d2h_bandwidth() const noexcept;
};

/* =============================================================================
 * STRUCT: GPUStream
 * =============================================================================
 * SUMMARY:
 *     Wrapper for GPU stream (CUDA stream / HIP stream).
 *
 * DESIGN PURPOSE:
 *     Provides backend-agnostic stream interface:
 *     - Encapsulates cudaStream_t or hipStream_t
 *     - RAII management of stream lifetime
 *     - Synchronization primitives
 * -------------------------------------------------------------------------- */
class GPUStream {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: GPUStream (default)
     * -------------------------------------------------------------------------
     * POSTCONDITIONS:
     *     Creates new non-blocking stream.
     * ---------------------------------------------------------------------- */
    GPUStream();

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: GPUStream (nullptr)
     * -------------------------------------------------------------------------
     * POSTCONDITIONS:
     *     Wraps default stream (stream 0).
     * ---------------------------------------------------------------------- */
    explicit GPUStream(std::nullptr_t);

    ~GPUStream();

    GPUStream(const GPUStream&) = delete;
    GPUStream& operator=(const GPUStream&) = delete;
    GPUStream(GPUStream&&) noexcept;
    GPUStream& operator=(GPUStream&&) noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: synchronize
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Wait for all operations in stream to complete.
     *
     * POSTCONDITIONS:
     *     All pending operations finished.
     * ---------------------------------------------------------------------- */
    void synchronize();

    /* -------------------------------------------------------------------------
     * METHOD: query
     * -------------------------------------------------------------------------
     * RETURNS:
     *     True if all operations in stream have completed.
     * ---------------------------------------------------------------------- */
    bool query() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: native_handle
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Underlying stream handle (cudaStream_t or hipStream_t).
     * ---------------------------------------------------------------------- */
    void* native_handle() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: valid
     * -------------------------------------------------------------------------
     * RETURNS:
     *     True if stream is valid.
     * ---------------------------------------------------------------------- */
    bool valid() const noexcept;

    /* -------------------------------------------------------------------------
     * STATIC METHOD: default_stream
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Reference to default stream (stream 0).
     * ---------------------------------------------------------------------- */
    static GPUStream& default_stream();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* =============================================================================
 * CLASS: GPUMemoryPool
 * =============================================================================
 * SUMMARY:
 *     GPU memory pool with async allocation and transfer support.
 *
 * DESIGN PURPOSE:
 *     Provides efficient GPU memory management:
 *     - Pooled allocations to avoid CUDA malloc overhead
 *     - Async memory transfers with stream support
 *     - Pinned host memory for fast DMA
 *     - Unified memory for convenience
 *
 * ARCHITECTURE:
 *     GPUMemoryPool
 *         ├── Device Pool (cudaMalloc)
 *         │       └── Free list by size class
 *         ├── Pinned Pool (cudaMallocHost)
 *         │       └── Free list by size class
 *         └── Transfer Engine
 *                 └── Async copy operations
 *
 * ALLOCATION STRATEGY:
 *     1. Round size up to page boundary
 *     2. Check free list for matching size
 *     3. If not found, allocate from device
 *     4. Track allocation for statistics
 *
 * THREAD SAFETY:
 *     All methods are thread-safe.
 *     Multiple streams can be used concurrently.
 * -------------------------------------------------------------------------- */
class GPUMemoryPool {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: GPUMemoryPool
     * -------------------------------------------------------------------------
     * PARAMETERS:
     *     config [in] - Pool configuration
     *
     * PRECONDITIONS:
     *     - GPU runtime initialized
     *     - Device exists (if device_id specified)
     *
     * POSTCONDITIONS:
     *     - Pool ready for allocations
     *     - Initial pool allocated if configured
     *
     * THROWS:
     *     GPUError if initialization fails.
     * ---------------------------------------------------------------------- */
    explicit GPUMemoryPool(GPUPoolConfig config = {});

    ~GPUMemoryPool();

    GPUMemoryPool(const GPUMemoryPool&) = delete;
    GPUMemoryPool& operator=(const GPUMemoryPool&) = delete;
    GPUMemoryPool(GPUMemoryPool&&) noexcept;
    GPUMemoryPool& operator=(GPUMemoryPool&&) noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: allocate_device
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Allocate GPU device memory.
     *
     * PARAMETERS:
     *     size_bytes [in] - Allocation size
     *
     * PRECONDITIONS:
     *     size_bytes > 0
     *
     * POSTCONDITIONS:
     *     On success: Returns device pointer.
     *     On failure: Returns nullptr.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void* allocate_device(
        std::size_t size_bytes             // Size to allocate
    );

    /* -------------------------------------------------------------------------
     * METHOD: allocate_pinned
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Allocate pinned (page-locked) host memory.
     *
     * PARAMETERS:
     *     size_bytes [in] - Allocation size
     *
     * POSTCONDITIONS:
     *     On success: Returns host pointer accessible by GPU DMA.
     *     On failure: Returns nullptr.
     *
     * NOTE:
     *     Pinned memory is limited. Use sparingly.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void* allocate_pinned(
        std::size_t size_bytes             // Size to allocate
    );

    /* -------------------------------------------------------------------------
     * METHOD: allocate_managed
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Allocate unified (managed) memory.
     *
     * PARAMETERS:
     *     size_bytes [in] - Allocation size
     *
     * POSTCONDITIONS:
     *     On success: Returns pointer accessible from CPU and GPU.
     *     On failure: Returns nullptr.
     *
     * NOTE:
     *     Managed memory has runtime overhead for page migration.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void* allocate_managed(
        std::size_t size_bytes             // Size to allocate
    );

    /* -------------------------------------------------------------------------
     * METHOD: deallocate
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Free previously allocated GPU memory.
     *
     * PARAMETERS:
     *     ptr        [in] - Pointer from allocate_*()
     *     size_bytes [in] - Original allocation size
     *     type       [in] - Memory type (Device/Pinned/Managed)
     *
     * PRECONDITIONS:
     *     - ptr was allocated by this pool
     *     - size_bytes matches original allocation
     *
     * POSTCONDITIONS:
     *     Memory returned to pool or freed.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void deallocate(
        void* ptr,                         // Pointer to free
        std::size_t size_bytes,            // Original size
        GPUMemoryType type                 // Memory type
    );

    /* -------------------------------------------------------------------------
     * METHOD: copy_h2d
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Copy data from host to device (synchronous).
     *
     * PARAMETERS:
     *     dst        [in] - Device destination pointer
     *     src        [in] - Host source pointer
     *     size_bytes [in] - Number of bytes to copy
     *
     * PRECONDITIONS:
     *     - dst is valid device pointer
     *     - src is valid host pointer
     *     - Both regions have at least size_bytes
     *
     * POSTCONDITIONS:
     *     Data copied to device. Returns on completion.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void copy_h2d(
        void* dst,                         // Device destination
        const void* src,                   // Host source
        std::size_t size_bytes             // Bytes to copy
    );

    /* -------------------------------------------------------------------------
     * METHOD: copy_d2h
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Copy data from device to host (synchronous).
     *
     * PARAMETERS:
     *     dst        [in] - Host destination pointer
     *     src        [in] - Device source pointer
     *     size_bytes [in] - Number of bytes to copy
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void copy_d2h(
        void* dst,                         // Host destination
        const void* src,                   // Device source
        std::size_t size_bytes             // Bytes to copy
    );

    /* -------------------------------------------------------------------------
     * METHOD: copy_h2d_async
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Copy data from host to device (asynchronous).
     *
     * PARAMETERS:
     *     dst        [in] - Device destination pointer
     *     src        [in] - Host source pointer (must be pinned)
     *     size_bytes [in] - Number of bytes to copy
     *     stream     [in] - GPU stream for async operation
     *
     * PRECONDITIONS:
     *     - src should be pinned memory for best performance
     *
     * POSTCONDITIONS:
     *     Copy queued to stream. Use stream.synchronize() to wait.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void copy_h2d_async(
        void* dst,                         // Device destination
        const void* src,                   // Host source (pinned)
        std::size_t size_bytes,            // Bytes to copy
        GPUStream& stream                  // Stream for async
    );

    /* -------------------------------------------------------------------------
     * METHOD: copy_d2h_async
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Copy data from device to host (asynchronous).
     *
     * PARAMETERS:
     *     dst        [in] - Host destination pointer (must be pinned)
     *     src        [in] - Device source pointer
     *     size_bytes [in] - Number of bytes to copy
     *     stream     [in] - GPU stream for async operation
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void copy_d2h_async(
        void* dst,                         // Host destination (pinned)
        const void* src,                   // Device source
        std::size_t size_bytes,            // Bytes to copy
        GPUStream& stream                  // Stream for async
    );

    /* -------------------------------------------------------------------------
     * METHOD: memset_device
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Set device memory to a value.
     *
     * PARAMETERS:
     *     ptr        [in] - Device pointer
     *     value      [in] - Byte value to set
     *     size_bytes [in] - Number of bytes
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void memset_device(
        void* ptr,                         // Device pointer
        int value,                         // Value (0-255)
        std::size_t size_bytes             // Bytes to set
    );

    /* -------------------------------------------------------------------------
     * METHOD: memset_device_async
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Set device memory to a value (asynchronous).
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void memset_device_async(
        void* ptr,                         // Device pointer
        int value,                         // Value
        std::size_t size_bytes,            // Bytes to set
        GPUStream& stream                  // Stream for async
    );

    /* -------------------------------------------------------------------------
     * METHOD: prefetch_to_device
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Prefetch managed memory to device.
     *
     * PARAMETERS:
     *     ptr        [in] - Managed memory pointer
     *     size_bytes [in] - Size to prefetch
     *     stream     [in] - Stream for async prefetch
     *
     * NOTE:
     *     Only works with managed memory. No-op for other types.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void prefetch_to_device(
        void* ptr,                         // Managed pointer
        std::size_t size_bytes,            // Size to prefetch
        GPUStream& stream                  // Stream for async
    );

    /* -------------------------------------------------------------------------
     * METHOD: prefetch_to_host
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Prefetch managed memory to host.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void prefetch_to_host(
        void* ptr,                         // Managed pointer
        std::size_t size_bytes,            // Size to prefetch
        GPUStream& stream                  // Stream for async
    );

    /* -------------------------------------------------------------------------
     * METHOD: device_id
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Device ID this pool is associated with.
     * ---------------------------------------------------------------------- */
    int device_id() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: backend
     * -------------------------------------------------------------------------
     * RETURNS:
     *     GPU backend in use.
     * ---------------------------------------------------------------------- */
    GPUBackend backend() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: device_info
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Information about the associated device.
     * ---------------------------------------------------------------------- */
    GPUDeviceInfo device_info() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: config
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Reference to pool configuration.
     * ---------------------------------------------------------------------- */
    const GPUPoolConfig& config() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: stats
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Current pool statistics.
     * ---------------------------------------------------------------------- */
    GPUPoolStats stats() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: reset_stats
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Reset statistics to zero.
     * ---------------------------------------------------------------------- */
    void reset_stats() noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: trim
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Release unused memory back to the system.
     *
     * PARAMETERS:
     *     keep_bytes [in] - Minimum pool size to keep
     *
     * POSTCONDITIONS:
     *     Pool size reduced to at most keep_bytes of free memory.
     * ---------------------------------------------------------------------- */
    void trim(
        std::size_t keep_bytes = 0         // Minimum to keep
    );

    /* -------------------------------------------------------------------------
     * STATIC METHOD: is_available
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Check if GPU support is available.
     *
     * RETURNS:
     *     True if at least one GPU is detected.
     * ---------------------------------------------------------------------- */
    static bool is_available() noexcept;

    /* -------------------------------------------------------------------------
     * STATIC METHOD: device_count
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Number of available GPU devices.
     * ---------------------------------------------------------------------- */
    static int device_count() noexcept;

    /* -------------------------------------------------------------------------
     * STATIC METHOD: get_device_info
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get information about a specific device.
     *
     * PARAMETERS:
     *     device_id [in] - Device index
     *
     * RETURNS:
     *     Device information structure.
     * ---------------------------------------------------------------------- */
    static GPUDeviceInfo get_device_info(
        int device_id                      // Device index
    );

    /* -------------------------------------------------------------------------
     * STATIC METHOD: instance
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get global singleton pool for default device.
     *
     * RETURNS:
     *     Reference to default pool.
     *
     * THREAD SAFETY:
     *     Thread-safe (static initialization).
     * ---------------------------------------------------------------------- */
    static GPUMemoryPool& instance();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* =============================================================================
 * FUNCTION: gpu_backend_name
 * =============================================================================
 * SUMMARY:
 *     Convert GPUBackend enum to human-readable string.
 *
 * PARAMETERS:
 *     backend [in] - GPU backend enum value
 *
 * RETURNS:
 *     Null-terminated C string (never nullptr).
 * -------------------------------------------------------------------------- */
const char* gpu_backend_name(GPUBackend backend) noexcept;

/* =============================================================================
 * FUNCTION: gpu_memory_type_name
 * =============================================================================
 * SUMMARY:
 *     Convert GPUMemoryType enum to human-readable string.
 *
 * PARAMETERS:
 *     type [in] - Memory type enum value
 *
 * RETURNS:
 *     Null-terminated C string (never nullptr).
 * -------------------------------------------------------------------------- */
const char* gpu_memory_type_name(GPUMemoryType type) noexcept;

} // namespace scl::mmap::memory
