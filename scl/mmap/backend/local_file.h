// =============================================================================
// FILE: scl/mmap/backend/local_file.h
// BRIEF: API reference for local filesystem storage backend
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "backend.h"

#include <filesystem>

namespace scl::mmap::backend {

/* =============================================================================
 * ENUM: LocalFileMode
 * =============================================================================
 * SUMMARY:
 *     I/O mode selection for local file backend.
 *
 * VALUES:
 *     MMap   - Use memory-mapped I/O
 *     PRead  - Use pread() system call
 *     Auto   - Automatically select based on access pattern
 *
 * SELECTION GUIDE:
 *     MMap:  Best for sequential access, large files, multiple reads of same region
 *            - OS manages page cache automatically
 *            - Zero-copy access possible
 *            - May cause page faults on random access
 *
 *     PRead: Best for random access, small reads, concurrent access
 *            - Explicit control over buffering
 *            - No page fault overhead
 *            - Better for highly concurrent workloads
 *
 *     Auto:  Selects MMap for files > 1GB with sequential hints,
 *            PRead otherwise. Can be overridden by access pattern hints.
 * -------------------------------------------------------------------------- */
enum class LocalFileMode : std::uint8_t {
    MMap,
    PRead,
    Auto
};

/* =============================================================================
 * STRUCT: LocalFileConfig
 * =============================================================================
 * SUMMARY:
 *     Configuration options for local file backend.
 *
 * FIELDS:
 *     mode            - I/O mode selection
 *     use_direct_io   - Bypass OS page cache (O_DIRECT)
 *     use_huge_pages  - Use huge pages for mmap (MAP_HUGETLB)
 *     prefetch_window - Pages to prefetch ahead in mmap mode
 *     read_buffer_size - Buffer size for pread mode (bytes)
 *
 * DEFAULTS:
 *     mode = Auto, direct_io = false, huge_pages = false,
 *     prefetch_window = 4, read_buffer_size = 0 (unbuffered)
 * -------------------------------------------------------------------------- */
struct LocalFileConfig {
    LocalFileMode mode = LocalFileMode::Auto;
    bool use_direct_io = false;     // Bypass OS page cache
    bool use_huge_pages = false;    // Use huge pages (mmap only)
    std::size_t prefetch_window = 4;       // Pages to prefetch ahead
    std::size_t read_buffer_size = 0;      // PRead buffer size (0 = unbuffered)

    /* -------------------------------------------------------------------------
     * FACTORY: sequential
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration optimized for sequential scan.
     * ---------------------------------------------------------------------- */
    static constexpr LocalFileConfig sequential() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: random_access
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration optimized for random access.
     * ---------------------------------------------------------------------- */
    static constexpr LocalFileConfig random_access() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: streaming
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration optimized for streaming large files.
     * ---------------------------------------------------------------------- */
    static constexpr LocalFileConfig streaming() noexcept;
};

/* =============================================================================
 * CLASS: LocalFileBackend
 * =============================================================================
 * SUMMARY:
 *     Storage backend for local filesystem using mmap or pread.
 *
 * DESIGN PURPOSE:
 *     Provides efficient access to local files with automatic mode selection:
 *     - Memory-mapped I/O for sequential workloads
 *     - pread() for random access workloads
 *     - Direct I/O bypass for large files
 *     - Automatic mode selection based on file size and access hints
 *
 * FEATURES:
 *     - Memory-mapped I/O with optional huge pages
 *     - pread() with optional read-ahead buffer
 *     - Direct I/O bypass (O_DIRECT) for large files
 *     - POSIX fadvise() hints for sequential/random patterns
 *     - Thread-safe concurrent access
 *
 * MEMORY USAGE:
 *     MMap mode:  Virtual address space = file size (physical = on demand)
 *     PRead mode: Buffer size (configurable, default = kPageSize)
 *
 * THREAD SAFETY:
 *     Fully thread-safe. Multiple threads can issue concurrent reads.
 *     Write operations are serialized per-page.
 * -------------------------------------------------------------------------- */
class LocalFileBackend : public StorageBackend<LocalFileBackend> {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: LocalFileBackend(path, read_only, config)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Open local file for memory-mapped or pread access.
     *
     * PARAMETERS:
     *     path      [in] - File path (must exist for read-only)
     *     read_only [in] - Open in read-only mode (default: true)
     *     config    [in] - Configuration options
     *
     * PRECONDITIONS:
     *     - If read_only: file must exist and be readable
     *     - If !read_only: file will be created if not exists
     *     - Path must be a regular file (not directory/device)
     *
     * POSTCONDITIONS:
     *     - File opened with appropriate mode
     *     - If MMap: memory mapping established
     *     - file_id() returns unique identifier
     *
     * THROWS:
     *     IOError if file cannot be opened or mapped.
     *
     * COMPLEXITY:
     *     O(1) for PRead mode
     *     O(file_size) virtual address space for MMap mode
     * ---------------------------------------------------------------------- */
    explicit LocalFileBackend(
        const std::filesystem::path& path,
        bool read_only = true,
        LocalFileConfig config = {}
    );

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: LocalFileBackend(fd, total_bytes, read_only, config)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create backend from existing file descriptor.
     *
     * PARAMETERS:
     *     fd          [in] - Open file descriptor (ownership transferred)
     *     total_bytes [in] - Total file size in bytes
     *     read_only   [in] - File opened in read-only mode
     *     config      [in] - Configuration options
     *
     * PRECONDITIONS:
     *     - fd is valid open file descriptor
     *     - fd has appropriate read/write permissions
     *     - total_bytes matches actual file size
     *
     * POSTCONDITIONS:
     *     - Backend takes ownership of fd
     *     - fd will be closed in destructor
     *
     * THROWS:
     *     IOError if mmap fails (for MMap mode).
     * ---------------------------------------------------------------------- */
    explicit LocalFileBackend(
        int fd,
        std::size_t total_bytes,
        bool read_only = true,
        LocalFileConfig config = {}
    );

    /* -------------------------------------------------------------------------
     * DESTRUCTOR: ~LocalFileBackend
     * -------------------------------------------------------------------------
     * POSTCONDITIONS:
     *     - Memory mapping released (if MMap mode)
     *     - File descriptor closed
     *     - All resources freed
     * ---------------------------------------------------------------------- */
    ~LocalFileBackend();

    LocalFileBackend(const LocalFileBackend&) = delete;
    LocalFileBackend& operator=(const LocalFileBackend&) = delete;

    LocalFileBackend(LocalFileBackend&& other) noexcept;
    LocalFileBackend& operator=(LocalFileBackend&& other) noexcept;

    /* -------------------------------------------------------------------------
     * CRTP IMPLEMENTATION METHODS
     * -------------------------------------------------------------------------
     * These methods are called via CRTP dispatch from StorageBackend<>.
     * ---------------------------------------------------------------------- */

    /* -------------------------------------------------------------------------
     * METHOD: capabilities_impl
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Backend capabilities for local file access.
     * ---------------------------------------------------------------------- */
    BackendCapabilities capabilities_impl() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: load_page_impl
     * -------------------------------------------------------------------------
     * ALGORITHM:
     *     MMap mode:
     *         1. Compute page offset in mapped region
     *         2. memcpy from mapped address to dest
     *         3. Issue madvise(MADV_SEQUENTIAL) for next pages
     *
     *     PRead mode:
     *         1. Compute file offset = page_idx * kPageSize
     *         2. pread(fd, dest, kPageSize, offset)
     *         3. Handle partial read (retry or error)
     *
     * RETURNS:
     *     kPageSize on success, 0 on error.
     * ---------------------------------------------------------------------- */
    std::size_t load_page_impl(std::size_t page_idx, std::byte* dest);

    /* -------------------------------------------------------------------------
     * METHOD: write_page_impl
     * -------------------------------------------------------------------------
     * ALGORITHM:
     *     MMap mode:
     *         1. memcpy from src to mapped address
     *         2. msync() if synchronous write requested
     *
     *     PRead mode:
     *         1. pwrite(fd, src, kPageSize, offset)
     *         2. Handle partial write (retry or error)
     *
     * RETURNS:
     *     kPageSize on success, 0 on error.
     * ---------------------------------------------------------------------- */
    std::size_t write_page_impl(std::size_t page_idx, const std::byte* src);

    /* -------------------------------------------------------------------------
     * METHOD: load_pages_async_impl
     * -------------------------------------------------------------------------
     * ALGORITHM:
     *     1. For each request in requests:
     *        a. Submit to thread pool or I/O queue
     *     2. Callback invoked as each operation completes
     *
     * NOTE:
     *     Uses system thread pool for async operations.
     *     True async I/O (io_uring/AIO) planned for future.
     * ---------------------------------------------------------------------- */
    std::size_t load_pages_async_impl(
        std::span<const IORequest> requests,
        IOCallback callback
    );

    /* -------------------------------------------------------------------------
     * METHOD: prefetch_hint_impl
     * -------------------------------------------------------------------------
     * ALGORITHM:
     *     MMap mode:
     *         madvise(MADV_WILLNEED) for specified pages
     *
     *     PRead mode:
     *         posix_fadvise(POSIX_FADV_WILLNEED) for specified regions
     * ---------------------------------------------------------------------- */
    void prefetch_hint_impl(std::span<const std::size_t> pages);

    /* -------------------------------------------------------------------------
     * METHOD: Accessors
     * ---------------------------------------------------------------------- */
    std::size_t num_pages_impl() const noexcept;
    std::size_t total_bytes_impl() const noexcept;
    std::size_t file_id_impl() const noexcept;
    BackendType type_impl() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: path
     * -------------------------------------------------------------------------
     * RETURNS:
     *     File path (empty if created from fd).
     * ---------------------------------------------------------------------- */
    const std::filesystem::path& path() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: is_mmap_mode
     * -------------------------------------------------------------------------
     * RETURNS:
     *     True if using memory-mapped I/O.
     * ---------------------------------------------------------------------- */
    bool is_mmap_mode() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: mmap_base
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get base address of memory mapping (mmap mode only).
     *
     * PRECONDITIONS:
     *     is_mmap_mode() must be true.
     *
     * RETURNS:
     *     Base address of mapped region, nullptr if not mmap mode.
     * ---------------------------------------------------------------------- */
    const std::byte* mmap_base() const noexcept;
    std::byte* mmap_base() noexcept;

private:
    int fd_;                        // File descriptor (-1 if closed)
    void* mmap_base_;               // Memory-mapped base (nullptr if pread mode)
    std::size_t file_size_;         // Total file size in bytes
    std::size_t file_id_;           // Unique identifier
    std::size_t num_pages_;         // Total pages
    std::filesystem::path path_;    // File path (may be empty)
    LocalFileConfig config_;        // Configuration
    bool read_only_;                // Read-only mode
    bool use_mmap_;                 // Using mmap or pread

    void setup_mmap();
    void teardown_mmap();
    void apply_fadvise(int advice);
};

/* =============================================================================
 * FUNCTION: local_file_mode_name
 * =============================================================================
 * SUMMARY:
 *     Convert LocalFileMode enum to human-readable string.
 *
 * PARAMETERS:
 *     mode [in] - File mode enum value
 *
 * RETURNS:
 *     Null-terminated C string (never nullptr).
 *
 * THREAD SAFETY:
 *     Thread-safe.
 * -------------------------------------------------------------------------- */
const char* local_file_mode_name(LocalFileMode mode) noexcept;

} // namespace scl::mmap::backend
