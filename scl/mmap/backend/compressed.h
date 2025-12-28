// =============================================================================
// FILE: scl/mmap/backend/compressed.h
// BRIEF: API reference for compressed storage backend with zstd/lz4 support
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "backend.h"

#include <memory>
#include <vector>

namespace scl::mmap::backend {

/* =============================================================================
 * ENUM: CompressionCodec
 * =============================================================================
 * SUMMARY:
 *     Compression algorithm selection.
 *
 * VALUES:
 *     None    - No compression (passthrough)
 *     Zstd    - Zstandard compression (best ratio, good speed)
 *     Lz4     - LZ4 compression (fastest, moderate ratio)
 *     Lz4HC   - LZ4 High Compression (slower compress, better ratio)
 *
 * SELECTION GUIDE:
 *     Zstd:   Best for storage-bound workloads, ~3:1 ratio typical
 *     Lz4:    Best for CPU-bound workloads, ~2:1 ratio typical
 *     Lz4HC:  Best for write-once read-many, ~2.5:1 ratio
 *     None:   For already compressed data or debugging
 * -------------------------------------------------------------------------- */
enum class CompressionCodec : std::uint8_t {
    None,
    Zstd,
    Lz4,
    Lz4HC
};

/* =============================================================================
 * STRUCT: CompressionConfig
 * =============================================================================
 * SUMMARY:
 *     Configuration options for compressed backend.
 *
 * FIELDS:
 *     codec               - Compression algorithm
 *     compression_level   - Codec-specific compression level
 *     enable_dictionary   - Use trained dictionary for better ratio
 *     dictionary_size     - Dictionary size in bytes (0 = default)
 *     verify_checksum     - Verify checksum on decompression
 *     num_decomp_threads  - Decompression threads (0 = auto)
 *
 * COMPRESSION LEVELS:
 *     Zstd:  1-22 (default: 3, higher = better ratio, slower)
 *     Lz4:   1-12 (default: 1, affects acceleration)
 *     Lz4HC: 1-12 (default: 9, higher = better ratio)
 * -------------------------------------------------------------------------- */
struct CompressionConfig {
    CompressionCodec codec = CompressionCodec::Zstd;
    int compression_level = 3;
    bool enable_dictionary = false;
    std::size_t dictionary_size = 0;
    bool verify_checksum = true;
    std::size_t num_decomp_threads = 0;

    /* -------------------------------------------------------------------------
     * FACTORY: fast
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration optimized for decompression speed.
     * ---------------------------------------------------------------------- */
    static constexpr CompressionConfig fast() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: balanced
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration balancing ratio and speed.
     * ---------------------------------------------------------------------- */
    static constexpr CompressionConfig balanced() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: high_ratio
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration optimized for compression ratio.
     * ---------------------------------------------------------------------- */
    static constexpr CompressionConfig high_ratio() noexcept;
};

/* =============================================================================
 * STRUCT: CompressedPageIndex
 * =============================================================================
 * SUMMARY:
 *     Index structure for random access to compressed pages.
 *
 * DESIGN PURPOSE:
 *     Enables O(1) lookup of compressed page location without scanning.
 *     Stored at beginning or end of compressed file.
 *
 * PAGE ENTRY FORMAT:
 *     offset (8 bytes): Byte offset of compressed page in file
 *     size (4 bytes):   Compressed size in bytes
 *     checksum (4 bytes): Optional CRC32 checksum
 *
 * FILE FORMAT:
 *     [Header: 32 bytes]
 *         magic (8): "SCLCMPG\0"
 *         version (4): Format version
 *         num_pages (8): Total page count
 *         codec (4): CompressionCodec enum
 *         flags (4): Feature flags
 *         original_size (8): Uncompressed total size
 *     [Page Index: num_pages * 16 bytes]
 *         [offset, size, checksum] per page
 *     [Compressed Data]
 *         Compressed pages concatenated
 *     [Optional Dictionary]
 *         Training dictionary if enabled
 * -------------------------------------------------------------------------- */
struct CompressedPageIndex {
    struct PageEntry {
        std::uint64_t offset;       // Offset in compressed file
        std::uint32_t size;         // Compressed size
        std::uint32_t checksum;     // CRC32 (0 if disabled)
    };

    static constexpr std::uint64_t kMagic = 0x0047504D434C4353ULL; // "SCLCMPG\0"
    static constexpr std::uint32_t kVersion = 1;

    std::uint64_t num_pages;
    CompressionCodec codec;
    std::uint32_t flags;
    std::uint64_t original_size;
    std::vector<PageEntry> entries;

    /* -------------------------------------------------------------------------
     * METHOD: header_size
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Size of header in bytes (32).
     * ---------------------------------------------------------------------- */
    static constexpr std::size_t header_size() noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: index_size
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Size of page index in bytes.
     * ---------------------------------------------------------------------- */
    std::size_t index_size() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: data_offset
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Byte offset where compressed data begins.
     * ---------------------------------------------------------------------- */
    std::size_t data_offset() const noexcept;
};

/* =============================================================================
 * CLASS: CompressedBackend
 * =============================================================================
 * SUMMARY:
 *     Storage backend with transparent compression/decompression.
 *
 * DESIGN PURPOSE:
 *     Wraps underlying backend to provide transparent compression:
 *     - Pages compressed independently for random access
 *     - Page index enables O(1) lookup
 *     - Thread-local decompression buffers for concurrency
 *     - Optional dictionary for better ratio on similar data
 *
 * ARCHITECTURE:
 *     CompressedBackend
 *         ↓ wraps
 *     UnderlyingBackend (LocalFileBackend, etc.)
 *         ↓ stores
 *     [Header][Index][Compressed Pages][Dictionary]
 *
 * COMPRESSION STRATEGY:
 *     - Each page compressed independently
 *     - Allows random access without decompressing entire file
 *     - Trade-off: Slightly worse ratio than stream compression
 *
 * DECOMPRESSION BUFFER POOL:
 *     - Thread-local buffers avoid allocation per read
 *     - Sized to hold one compressed page (may exceed kPageSize)
 *     - Automatically grown if needed
 *
 * THREAD SAFETY:
 *     Fully thread-safe. Uses thread-local decompression contexts.
 * -------------------------------------------------------------------------- */
class CompressedBackend : public StorageBackend<CompressedBackend> {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: CompressedBackend(path, config)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Open existing compressed file.
     *
     * PARAMETERS:
     *     path   [in] - Path to compressed file
     *     config [in] - Decompression configuration
     *
     * PRECONDITIONS:
     *     - File must exist and be valid compressed format
     *     - File header must match expected magic/version
     *
     * POSTCONDITIONS:
     *     - Page index loaded into memory
     *     - Ready for random access reads
     *
     * THROWS:
     *     IOError if file cannot be opened or is invalid format.
     * ---------------------------------------------------------------------- */
    explicit CompressedBackend(
        const std::filesystem::path& path,
        CompressionConfig config = {}
    );

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: CompressedBackend(underlying, config)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Wrap existing backend with compression layer.
     *
     * PARAMETERS:
     *     underlying [in] - Backend to wrap (ownership transferred)
     *     config     [in] - Compression configuration
     *
     * PRECONDITIONS:
     *     - underlying != nullptr
     *     - underlying contains valid compressed data
     *
     * POSTCONDITIONS:
     *     - Takes ownership of underlying backend
     *     - Page index loaded from underlying
     * ---------------------------------------------------------------------- */
    explicit CompressedBackend(
        std::unique_ptr<StorageBackend<LocalFileBackend>> underlying,
        CompressionConfig config = {}
    );

    ~CompressedBackend();

    CompressedBackend(const CompressedBackend&) = delete;
    CompressedBackend& operator=(const CompressedBackend&) = delete;
    CompressedBackend(CompressedBackend&&) noexcept;
    CompressedBackend& operator=(CompressedBackend&&) noexcept;

    /* -------------------------------------------------------------------------
     * CRTP IMPLEMENTATION METHODS
     * ---------------------------------------------------------------------- */

    /* -------------------------------------------------------------------------
     * METHOD: capabilities_impl
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Backend capabilities for compressed access.
     *
     * NOTE:
     *     - supports_writeback = false (read-only)
     *     - requires_decompression = true
     *     - typical_latency includes decompression time
     * ---------------------------------------------------------------------- */
    BackendCapabilities capabilities_impl() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: load_page_impl
     * -------------------------------------------------------------------------
     * ALGORITHM:
     *     1. Lookup page entry in index
     *     2. Read compressed data from underlying backend
     *     3. Decompress to destination buffer
     *     4. Verify checksum if enabled
     *
     * RETURNS:
     *     kPageSize on success, 0 on error.
     *
     * ERROR HANDLING:
     *     Returns 0 and zeros dest on:
     *     - Page index out of bounds
     *     - Decompression failure
     *     - Checksum mismatch
     * ---------------------------------------------------------------------- */
    std::size_t load_page_impl(std::size_t page_idx, std::byte* dest);

    /* -------------------------------------------------------------------------
     * METHOD: write_page_impl
     * -------------------------------------------------------------------------
     * NOTE:
     *     CompressedBackend is read-only. Always returns 0.
     * ---------------------------------------------------------------------- */
    std::size_t write_page_impl(std::size_t page_idx, const std::byte* src);

    /* -------------------------------------------------------------------------
     * METHOD: load_pages_async_impl
     * -------------------------------------------------------------------------
     * ALGORITHM:
     *     1. Sort requests by compressed file offset (sequential I/O)
     *     2. Batch read compressed data
     *     3. Decompress in parallel using thread pool
     *     4. Invoke callbacks as pages complete
     * ---------------------------------------------------------------------- */
    std::size_t load_pages_async_impl(
        std::span<const IORequest> requests,
        IOCallback callback
    );

    /* -------------------------------------------------------------------------
     * METHOD: prefetch_hint_impl
     * -------------------------------------------------------------------------
     * ALGORITHM:
     *     Translate page indices to compressed file offsets,
     *     then hint underlying backend.
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
     * METHOD: compression_ratio
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get overall compression ratio.
     *
     * RETURNS:
     *     Ratio = original_size / compressed_size (> 1.0 means compression)
     * ---------------------------------------------------------------------- */
    double compression_ratio() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: codec
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Compression codec used.
     * ---------------------------------------------------------------------- */
    CompressionCodec codec() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: index
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Reference to page index (for inspection).
     * ---------------------------------------------------------------------- */
    const CompressedPageIndex& index() const noexcept;

    /* -------------------------------------------------------------------------
     * STATIC METHOD: create_compressed_file
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create new compressed file from uncompressed source.
     *
     * PARAMETERS:
     *     source     [in] - Source backend with uncompressed data
     *     dest_path  [in] - Destination file path
     *     config     [in] - Compression configuration
     *     progress   [in] - Optional progress callback (0.0 - 1.0)
     *
     * ALGORITHM:
     *     1. Write header with placeholder values
     *     2. For each page in source:
     *        a. Read uncompressed page
     *        b. Compress with configured codec
     *        c. Write to dest, record offset/size in index
     *     3. Write page index
     *     4. Update header with final values
     *
     * RETURNS:
     *     True on success, false on error.
     *
     * THREAD SAFETY:
     *     Not thread-safe. Single-threaded compression.
     * ---------------------------------------------------------------------- */
    template <typename SourceBackend>
    static bool create_compressed_file(
        SourceBackend& source,
        const std::filesystem::path& dest_path,
        CompressionConfig config = {},
        std::function<void(double)> progress = nullptr
    );

private:
    std::unique_ptr<LocalFileBackend> underlying_;
    CompressedPageIndex index_;
    CompressionConfig config_;
    std::size_t file_id_;
    std::size_t compressed_size_;

    // Thread-local decompression state
    struct DecompressContext;
    static thread_local std::unique_ptr<DecompressContext> decomp_ctx_;

    void load_index();
    DecompressContext& get_decomp_context();
    std::size_t decompress_page(
        const std::byte* src,
        std::size_t src_size,
        std::byte* dest
    );
    std::uint32_t compute_checksum(const std::byte* data, std::size_t size) const;
};

/* =============================================================================
 * FUNCTION: compression_codec_name
 * =============================================================================
 * SUMMARY:
 *     Convert CompressionCodec enum to human-readable string.
 *
 * PARAMETERS:
 *     codec [in] - Compression codec enum value
 *
 * RETURNS:
 *     Null-terminated C string (never nullptr).
 * -------------------------------------------------------------------------- */
const char* compression_codec_name(CompressionCodec codec) noexcept;

/* =============================================================================
 * FUNCTION: estimate_compressed_size
 * =============================================================================
 * SUMMARY:
 *     Estimate compressed size for given uncompressed size and codec.
 *
 * PARAMETERS:
 *     uncompressed_size [in] - Original data size in bytes
 *     codec             [in] - Compression codec
 *
 * RETURNS:
 *     Estimated compressed size (conservative, may be larger than actual).
 *
 * USAGE:
 *     Use for pre-allocating buffers or estimating storage requirements.
 * -------------------------------------------------------------------------- */
std::size_t estimate_compressed_size(
    std::size_t uncompressed_size,
    CompressionCodec codec
) noexcept;

} // namespace scl::mmap::backend
