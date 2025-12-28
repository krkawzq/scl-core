// =============================================================================
// FILE: scl/mmap/backend/backend.h
// BRIEF: API reference for storage backend abstraction layer
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <functional>
#include <span>

namespace scl::mmap::backend {

/* =============================================================================
 * ENUM: BackendType
 * =============================================================================
 * SUMMARY:
 *     Identifies the type of storage backend for runtime dispatch decisions.
 *
 * VALUES:
 *     LocalFile   - Local filesystem (mmap or pread)
 *     Compressed  - Compressed storage (zstd/lz4)
 *     Network     - Network storage (S3/HTTP)
 *     Memory      - In-memory buffer
 *     Custom      - User-defined backend
 * -------------------------------------------------------------------------- */
enum class BackendType : std::uint8_t {
    LocalFile,
    Compressed,
    Network,
    Memory,
    Custom
};

/* =============================================================================
 * STRUCT: BackendCapabilities
 * =============================================================================
 * SUMMARY:
 *     Describes capabilities of a storage backend for optimization decisions.
 *
 * DESIGN PURPOSE:
 *     Allows scheduler and cache to adapt behavior based on backend traits:
 *     - Random access support affects prefetch strategy
 *     - Concurrent I/O capability affects worker thread count
 *     - Compression flag affects CPU budget allocation
 *     - Typical latency guides prefetch depth
 *
 * THREAD SAFETY:
 *     Immutable after construction.
 * -------------------------------------------------------------------------- */
struct BackendCapabilities {
    bool supports_random_access;    // Can seek to arbitrary positions
    bool supports_concurrent_io;    // Thread-safe read/write
    bool supports_prefetch_hint;    // Backend can use prefetch hints
    bool supports_writeback;        // Can write modified pages
    bool is_compressed;             // Data is compressed on storage
    bool requires_decompression;    // Needs CPU for decompression

    std::size_t optimal_io_size;    // Optimal I/O request size (bytes)
    std::size_t max_concurrent_ops; // Maximum concurrent operations
    std::chrono::nanoseconds typical_latency; // Expected per-page latency

    /* -------------------------------------------------------------------------
     * FACTORY: local_file
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Default capabilities for local file backend.
     * ---------------------------------------------------------------------- */
    static constexpr BackendCapabilities local_file() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: compressed
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Default capabilities for compressed backend.
     * ---------------------------------------------------------------------- */
    static constexpr BackendCapabilities compressed() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: network
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Default capabilities for network backend.
     * ---------------------------------------------------------------------- */
    static constexpr BackendCapabilities network() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: memory
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Default capabilities for in-memory backend.
     * ---------------------------------------------------------------------- */
    static constexpr BackendCapabilities memory() noexcept;
};

/* =============================================================================
 * STRUCT: IORequest
 * =============================================================================
 * SUMMARY:
 *     Describes a single I/O operation for batch processing.
 *
 * USAGE:
 *     Used by load_pages_async() for batched asynchronous I/O.
 *     Higher priority values indicate more urgent requests.
 * -------------------------------------------------------------------------- */
struct IORequest {
    std::size_t page_idx;           // Page index to load
    std::byte* dest;                // Destination buffer (kPageSize bytes)
    int priority;                   // Request priority (higher = more urgent)
    void* user_data;                // User context for callback

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: IORequest
     * -------------------------------------------------------------------------
     * PARAMETERS:
     *     page_idx  [in] - Page index to load
     *     dest      [in] - Destination buffer
     *     priority  [in] - Request priority (default: 0)
     *     user_data [in] - User context (default: nullptr)
     * ---------------------------------------------------------------------- */
    constexpr IORequest(
        std::size_t page_idx,
        std::byte* dest,
        int priority = 0,
        void* user_data = nullptr
    ) noexcept;
};

/* =============================================================================
 * STRUCT: IOResult
 * =============================================================================
 * SUMMARY:
 *     Result of an I/O operation, passed to completion callback.
 *
 * ERROR HANDLING:
 *     error == 0 indicates success.
 *     error != 0 contains backend-specific error code.
 * -------------------------------------------------------------------------- */
struct IOResult {
    std::size_t page_idx;           // Page index that was loaded
    std::size_t bytes_transferred;  // Actual bytes transferred
    std::chrono::nanoseconds latency; // Operation latency
    int error;                      // 0 = success, non-zero = error code
    void* user_data;                // User context from IORequest

    /* -------------------------------------------------------------------------
     * METHOD: success
     * -------------------------------------------------------------------------
     * RETURNS:
     *     True if operation succeeded.
     * ---------------------------------------------------------------------- */
    constexpr bool success() const noexcept;
};

/* =============================================================================
 * TYPE ALIAS: IOCallback
 * =============================================================================
 * SUMMARY:
 *     Callback function type for asynchronous I/O completion.
 *
 * THREAD SAFETY:
 *     Callback may be invoked from any worker thread.
 *     Implementation must be thread-safe.
 * -------------------------------------------------------------------------- */
using IOCallback = std::function<void(IOResult)>;

/* =============================================================================
 * CLASS: StorageBackend<Derived>
 * =============================================================================
 * SUMMARY:
 *     CRTP base class for storage backends with zero-overhead polymorphism.
 *
 * DESIGN PURPOSE:
 *     Provides unified interface for different storage backends:
 *     - LocalFileBackend: Local filesystem (mmap or pread)
 *     - CompressedBackend: Transparent compression layer
 *     - NetworkBackend: Remote storage (S3/HTTP)
 *     - MemoryBackend: In-memory buffer
 *
 * CRTP PATTERN:
 *     Uses Curiously Recurring Template Pattern for static polymorphism.
 *     All method calls are resolved at compile time (no vtable overhead).
 *
 * TEMPLATE PARAMETERS:
 *     Derived - Concrete backend implementation class
 *
 * DERIVED CLASS REQUIREMENTS:
 *     Derived must implement:
 *     - BackendCapabilities capabilities_impl() const noexcept
 *     - std::size_t load_page_impl(std::size_t page_idx, std::byte* dest)
 *     - std::size_t write_page_impl(std::size_t page_idx, const std::byte* src)
 *     - std::size_t load_pages_async_impl(std::span<const IORequest>, IOCallback)
 *     - void prefetch_hint_impl(std::span<const std::size_t> pages)
 *     - std::size_t num_pages_impl() const noexcept
 *     - std::size_t total_bytes_impl() const noexcept
 *     - std::size_t file_id_impl() const noexcept
 *
 * THREAD SAFETY:
 *     Thread safety depends on derived implementation.
 *     Check capabilities().supports_concurrent_io.
 * -------------------------------------------------------------------------- */
template <typename Derived>
class StorageBackend {
public:
    /* -------------------------------------------------------------------------
     * METHOD: capabilities
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Query backend capabilities for optimization decisions.
     *
     * RETURNS:
     *     BackendCapabilities describing this backend's traits.
     *
     * THREAD SAFETY:
     *     Thread-safe (immutable after construction).
     *
     * COMPLEXITY:
     *     O(1)
     * ---------------------------------------------------------------------- */
    BackendCapabilities capabilities() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: load_page
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Synchronous single page load.
     *
     * PARAMETERS:
     *     page_idx [in]  - Page index to load (0-based)
     *     dest     [out] - Destination buffer (must be kPageSize bytes)
     *
     * PRECONDITIONS:
     *     - page_idx < num_pages()
     *     - dest != nullptr
     *     - dest buffer has at least kPageSize bytes
     *
     * POSTCONDITIONS:
     *     On success: dest contains page data, returns kPageSize.
     *     On error: returns 0, dest contents undefined.
     *
     * RETURNS:
     *     Number of bytes loaded (kPageSize on success, 0 on error).
     *
     * THREAD SAFETY:
     *     Depends on backend - check capabilities().supports_concurrent_io.
     *
     * COMPLEXITY:
     *     O(kPageSize) + I/O latency
     * ---------------------------------------------------------------------- */
    std::size_t load_page(std::size_t page_idx, std::byte* dest);

    /* -------------------------------------------------------------------------
     * METHOD: write_page
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Synchronous single page write.
     *
     * PARAMETERS:
     *     page_idx [in] - Page index to write (0-based)
     *     src      [in] - Source buffer (kPageSize bytes)
     *
     * PRECONDITIONS:
     *     - page_idx < num_pages()
     *     - src != nullptr
     *     - capabilities().supports_writeback == true
     *
     * POSTCONDITIONS:
     *     On success: page data persisted, returns kPageSize.
     *     On error: returns 0, storage state undefined.
     *
     * RETURNS:
     *     Number of bytes written (kPageSize on success, 0 on error).
     *
     * THREAD SAFETY:
     *     Depends on backend - check capabilities().supports_concurrent_io.
     *
     * COMPLEXITY:
     *     O(kPageSize) + I/O latency
     * ---------------------------------------------------------------------- */
    std::size_t write_page(std::size_t page_idx, const std::byte* src);

    /* -------------------------------------------------------------------------
     * METHOD: load_pages_async
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Asynchronous batch page load.
     *
     * PARAMETERS:
     *     requests [in] - Array of I/O requests
     *     callback [in] - Completion callback (called once per request)
     *
     * PRECONDITIONS:
     *     - All page_idx in requests < num_pages()
     *     - All dest pointers != nullptr
     *     - callback != nullptr
     *
     * POSTCONDITIONS:
     *     - Returns number of requests successfully submitted
     *     - callback invoked for each submitted request (success or failure)
     *     - Callback may be invoked from any thread
     *
     * RETURNS:
     *     Number of requests submitted successfully.
     *
     * THREAD SAFETY:
     *     Thread-safe. Callback must be thread-safe.
     *
     * COMPLEXITY:
     *     O(requests.size()) for submission
     *
     * NOTE:
     *     Callbacks may be invoked before this method returns.
     * ---------------------------------------------------------------------- */
    std::size_t load_pages_async(
        std::span<const IORequest> requests,
        IOCallback callback
    );

    /* -------------------------------------------------------------------------
     * METHOD: prefetch_hint
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Hint backend about upcoming page accesses.
     *
     * PARAMETERS:
     *     pages [in] - Page indices that will be accessed soon
     *
     * PRECONDITIONS:
     *     All page indices < num_pages()
     *
     * POSTCONDITIONS:
     *     Backend may pre-stage data in internal buffers.
     *     No guarantee that data will be ready.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     *
     * COMPLEXITY:
     *     O(pages.size())
     *
     * NOTE:
     *     This is a hint only. Backend may ignore if not supported.
     *     Check capabilities().supports_prefetch_hint.
     * ---------------------------------------------------------------------- */
    void prefetch_hint(std::span<const std::size_t> pages);

    /* -------------------------------------------------------------------------
     * METHOD: num_pages
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get total number of pages in this backend.
     *
     * RETURNS:
     *     Total page count.
     *
     * THREAD SAFETY:
     *     Thread-safe (immutable after construction).
     *
     * COMPLEXITY:
     *     O(1)
     * ---------------------------------------------------------------------- */
    std::size_t num_pages() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: total_bytes
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get total data size in bytes.
     *
     * RETURNS:
     *     Total byte count (may be less than num_pages * kPageSize for
     *     last partial page).
     *
     * THREAD SAFETY:
     *     Thread-safe (immutable after construction).
     *
     * COMPLEXITY:
     *     O(1)
     * ---------------------------------------------------------------------- */
    std::size_t total_bytes() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: file_id
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get unique identifier for this backend instance.
     *
     * DESIGN PURPOSE:
     *     Used by GlobalPagePool for page deduplication.
     *     Two backends with same file_id share pages.
     *
     * RETURNS:
     *     Unique identifier (never 0, never repeats).
     *
     * THREAD SAFETY:
     *     Thread-safe (immutable after construction).
     *
     * COMPLEXITY:
     *     O(1)
     * ---------------------------------------------------------------------- */
    std::size_t file_id() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: type
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get backend type identifier.
     *
     * RETURNS:
     *     BackendType enum value.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     *
     * COMPLEXITY:
     *     O(1)
     * ---------------------------------------------------------------------- */
    BackendType type() const noexcept;

protected:
    /* -------------------------------------------------------------------------
     * METHODS: CRTP dispatch helpers
     * -------------------------------------------------------------------------
     * DESIGN PURPOSE:
     *     Provide access to derived class for CRTP dispatch.
     * ---------------------------------------------------------------------- */
    Derived& derived() noexcept;
    const Derived& derived() const noexcept;
};

/* =============================================================================
 * FUNCTION: generate_file_id
 * =============================================================================
 * SUMMARY:
 *     Generate unique file identifier for new backend.
 *
 * ALGORITHM:
 *     Monotonically increasing atomic counter starting from 1.
 *     Counter never wraps (64-bit).
 *
 * POSTCONDITIONS:
 *     Returns unique ID (never 0, never repeats within process lifetime).
 *
 * THREAD SAFETY:
 *     Thread-safe (atomic increment).
 *
 * COMPLEXITY:
 *     O(1)
 * -------------------------------------------------------------------------- */
std::size_t generate_file_id() noexcept;

/* =============================================================================
 * FUNCTION: backend_type_name
 * =============================================================================
 * SUMMARY:
 *     Convert BackendType enum to human-readable string.
 *
 * PARAMETERS:
 *     type [in] - Backend type enum value
 *
 * RETURNS:
 *     Null-terminated C string (never nullptr).
 *
 * THREAD SAFETY:
 *     Thread-safe.
 *
 * COMPLEXITY:
 *     O(1)
 * -------------------------------------------------------------------------- */
const char* backend_type_name(BackendType type) noexcept;

} // namespace scl::mmap::backend
