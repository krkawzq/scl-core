// =============================================================================
// FILE: scl/mmap/backend/network.h
// BRIEF: API reference for network storage backend (S3/HTTP)
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "backend.h"

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <optional>

namespace scl::mmap::backend {

/* =============================================================================
 * ENUM: NetworkProtocol
 * =============================================================================
 * SUMMARY:
 *     Network protocol selection.
 *
 * VALUES:
 *     HTTP    - Plain HTTP (not recommended for production)
 *     HTTPS   - Secure HTTPS
 *     S3      - Amazon S3 protocol
 *     GCS     - Google Cloud Storage
 *     Azure   - Azure Blob Storage
 * -------------------------------------------------------------------------- */
enum class NetworkProtocol : std::uint8_t {
    HTTP,
    HTTPS,
    S3,
    GCS,
    Azure
};

/* =============================================================================
 * STRUCT: S3Credentials
 * =============================================================================
 * SUMMARY:
 *     AWS S3 credentials for authentication.
 *
 * FIELDS:
 *     access_key_id     - AWS access key ID
 *     secret_access_key - AWS secret access key
 *     session_token     - Optional session token for temporary credentials
 *     region            - AWS region (e.g., "us-east-1")
 *
 * NOTE:
 *     Leave empty to use environment variables or IAM role.
 * -------------------------------------------------------------------------- */
struct S3Credentials {
    std::string access_key_id;
    std::string secret_access_key;
    std::string session_token;
    std::string region = "us-east-1";

    /* -------------------------------------------------------------------------
     * FACTORY: from_environment
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Load credentials from environment variables.
     *
     * ENVIRONMENT:
     *     AWS_ACCESS_KEY_ID
     *     AWS_SECRET_ACCESS_KEY
     *     AWS_SESSION_TOKEN (optional)
     *     AWS_REGION or AWS_DEFAULT_REGION
     *
     * RETURNS:
     *     Credentials from environment, or empty if not set.
     * ---------------------------------------------------------------------- */
    static S3Credentials from_environment();

    /* -------------------------------------------------------------------------
     * METHOD: is_valid
     * -------------------------------------------------------------------------
     * RETURNS:
     *     True if credentials appear valid (non-empty keys).
     * ---------------------------------------------------------------------- */
    bool is_valid() const noexcept;
};

/* =============================================================================
 * STRUCT: NetworkConfig
 * =============================================================================
 * SUMMARY:
 *     Configuration for network backend.
 *
 * FIELDS:
 *     protocol           - Network protocol to use
 *     endpoint           - Custom endpoint URL (empty = default)
 *     s3_credentials     - S3 authentication credentials
 *     connect_timeout    - Connection timeout
 *     read_timeout       - Read operation timeout
 *     max_retries        - Maximum retry attempts
 *     retry_delay        - Initial retry delay (exponential backoff)
 *     max_connections    - Maximum concurrent connections
 *     enable_compression - Request compressed responses
 *     verify_ssl         - Verify SSL certificates
 *     user_agent         - Custom user agent string
 * -------------------------------------------------------------------------- */
struct NetworkConfig {
    NetworkProtocol protocol = NetworkProtocol::HTTPS;
    std::string endpoint;
    S3Credentials s3_credentials;
    std::chrono::milliseconds connect_timeout{5000};
    std::chrono::milliseconds read_timeout{30000};
    std::size_t max_retries = 3;
    std::chrono::milliseconds retry_delay{100};
    std::size_t max_connections = 8;
    bool enable_compression = true;
    bool verify_ssl = true;
    std::string user_agent = "scl-mmap/1.0";

    /* -------------------------------------------------------------------------
     * FACTORY: s3_default
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration for S3 with default settings.
     * ---------------------------------------------------------------------- */
    static NetworkConfig s3_default();

    /* -------------------------------------------------------------------------
     * FACTORY: http_default
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration for HTTP/HTTPS with default settings.
     * ---------------------------------------------------------------------- */
    static NetworkConfig http_default();

    /* -------------------------------------------------------------------------
     * FACTORY: low_latency
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration optimized for low latency.
     * ---------------------------------------------------------------------- */
    static NetworkConfig low_latency();

    /* -------------------------------------------------------------------------
     * FACTORY: high_throughput
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration optimized for high throughput.
     * ---------------------------------------------------------------------- */
    static NetworkConfig high_throughput();
};

/* =============================================================================
 * STRUCT: NetworkStats
 * =============================================================================
 * SUMMARY:
 *     Statistics for network backend operations.
 *
 * FIELDS:
 *     requests_sent      - Total HTTP requests made
 *     bytes_downloaded   - Total bytes received
 *     bytes_uploaded     - Total bytes sent
 *     cache_hits         - Requests served from local cache
 *     cache_misses       - Requests requiring network fetch
 *     retries            - Total retry attempts
 *     errors             - Total failed requests
 *     total_latency      - Cumulative request latency
 * -------------------------------------------------------------------------- */
struct NetworkStats {
    std::size_t requests_sent;
    std::size_t bytes_downloaded;
    std::size_t bytes_uploaded;
    std::size_t cache_hits;
    std::size_t cache_misses;
    std::size_t retries;
    std::size_t errors;
    std::chrono::nanoseconds total_latency;

    /* -------------------------------------------------------------------------
     * METHOD: avg_latency
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Average request latency.
     * ---------------------------------------------------------------------- */
    std::chrono::nanoseconds avg_latency() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: throughput
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Average throughput in MB/s.
     * ---------------------------------------------------------------------- */
    double throughput() const noexcept;
};

/* =============================================================================
 * STRUCT: ObjectInfo
 * =============================================================================
 * SUMMARY:
 *     Information about a remote object.
 *
 * FIELDS:
 *     size           - Object size in bytes
 *     last_modified  - Last modification timestamp
 *     etag           - Entity tag for caching
 *     content_type   - MIME content type
 *     exists         - True if object exists
 * -------------------------------------------------------------------------- */
struct ObjectInfo {
    std::size_t size;
    std::chrono::system_clock::time_point last_modified;
    std::string etag;
    std::string content_type;
    bool exists;
};

/* =============================================================================
 * CLASS: NetworkBackend
 * =============================================================================
 * SUMMARY:
 *     Storage backend for remote data access via HTTP/S3.
 *
 * DESIGN PURPOSE:
 *     Provides transparent access to remote data:
 *     - Range requests for page-aligned reads
 *     - Connection pooling for efficiency
 *     - Automatic retry with exponential backoff
 *     - Local caching of frequently accessed pages
 *
 * ARCHITECTURE:
 *     NetworkBackend
 *         ├── Connection Pool
 *         │       └── Persistent HTTP connections
 *         ├── Request Queue
 *         │       └── Async request handling
 *         └── Local Cache
 *                 └── LRU cache of fetched pages
 *
 * S3 URL FORMAT:
 *     s3://bucket-name/path/to/object
 *
 * HTTP URL FORMAT:
 *     https://host.com/path/to/file
 *
 * PAGE ACCESS:
 *     Uses HTTP Range requests: "Range: bytes=start-end"
 *     Each page is fetched independently for random access.
 *
 * THREAD SAFETY:
 *     Fully thread-safe. Connection pool handles concurrency.
 * -------------------------------------------------------------------------- */
class NetworkBackend : public StorageBackend<NetworkBackend> {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: NetworkBackend(url, config)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Open remote file for reading.
     *
     * PARAMETERS:
     *     url    [in] - Remote URL (s3://... or https://...)
     *     config [in] - Network configuration
     *
     * PRECONDITIONS:
     *     - URL is well-formed
     *     - For S3: credentials available
     *
     * POSTCONDITIONS:
     *     - Connection pool initialized
     *     - Object metadata fetched (HEAD request)
     *     - Ready for page reads
     *
     * THROWS:
     *     NetworkError if connection fails or object not found.
     * ---------------------------------------------------------------------- */
    explicit NetworkBackend(
        std::string_view url,              // Remote URL
        NetworkConfig config = {}          // Network configuration
    );

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: NetworkBackend(bucket, key, config)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Open S3 object for reading.
     *
     * PARAMETERS:
     *     bucket [in] - S3 bucket name
     *     key    [in] - Object key (path within bucket)
     *     config [in] - Network configuration
     *
     * POSTCONDITIONS:
     *     Same as URL constructor.
     * ---------------------------------------------------------------------- */
    NetworkBackend(
        std::string_view bucket,           // S3 bucket name
        std::string_view key,              // Object key
        NetworkConfig config = {}          // Network configuration
    );

    ~NetworkBackend();

    NetworkBackend(const NetworkBackend&) = delete;
    NetworkBackend& operator=(const NetworkBackend&) = delete;
    NetworkBackend(NetworkBackend&&) noexcept;
    NetworkBackend& operator=(NetworkBackend&&) noexcept;

    /* -------------------------------------------------------------------------
     * CRTP IMPLEMENTATION METHODS
     * ---------------------------------------------------------------------- */

    /* -------------------------------------------------------------------------
     * METHOD: capabilities_impl
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Backend capabilities for network access.
     *
     * NOTE:
     *     - supports_writeback = false (read-only)
     *     - typical_latency = network RTT + transfer time
     * ---------------------------------------------------------------------- */
    BackendCapabilities capabilities_impl() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: load_page_impl
     * -------------------------------------------------------------------------
     * ALGORITHM:
     *     1. Check local cache
     *     2. If miss: issue HTTP GET with Range header
     *     3. Retry on transient errors
     *     4. Update cache on success
     *
     * RETURNS:
     *     kPageSize on success, 0 on error.
     *
     * ERROR HANDLING:
     *     - Retries on 5xx errors and timeouts
     *     - Returns 0 on 4xx errors (no retry)
     * ---------------------------------------------------------------------- */
    std::size_t load_page_impl(std::size_t page_idx, std::byte* dest);

    /* -------------------------------------------------------------------------
     * METHOD: write_page_impl
     * -------------------------------------------------------------------------
     * NOTE:
     *     NetworkBackend is read-only. Always returns 0.
     * ---------------------------------------------------------------------- */
    std::size_t write_page_impl(std::size_t page_idx, const std::byte* src);

    /* -------------------------------------------------------------------------
     * METHOD: load_pages_async_impl
     * -------------------------------------------------------------------------
     * ALGORITHM:
     *     1. Queue all requests to connection pool
     *     2. Execute in parallel (up to max_connections)
     *     3. Invoke callbacks as pages complete
     *
     * NOTE:
     *     True async I/O using non-blocking sockets.
     * ---------------------------------------------------------------------- */
    std::size_t load_pages_async_impl(
        std::span<const IORequest> requests,
        IOCallback callback
    );

    /* -------------------------------------------------------------------------
     * METHOD: prefetch_hint_impl
     * -------------------------------------------------------------------------
     * ALGORITHM:
     *     Issue background requests for hinted pages.
     *     Results cached for later access.
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
     * METHOD: url
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Original URL of the remote object.
     * ---------------------------------------------------------------------- */
    const std::string& url() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: object_info
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Information about the remote object.
     * ---------------------------------------------------------------------- */
    const ObjectInfo& object_info() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: config
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Reference to network configuration.
     * ---------------------------------------------------------------------- */
    const NetworkConfig& config() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: stats
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Current network statistics.
     * ---------------------------------------------------------------------- */
    NetworkStats stats() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: reset_stats
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Reset network statistics to zero.
     * ---------------------------------------------------------------------- */
    void reset_stats() noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: refresh_metadata
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Re-fetch object metadata (HEAD request).
     *
     * POSTCONDITIONS:
     *     - object_info() updated
     *     - Returns true if object still exists
     *
     * USE CASE:
     *     Check for remote file changes.
     * ---------------------------------------------------------------------- */
    bool refresh_metadata();

    /* -------------------------------------------------------------------------
     * METHOD: clear_cache
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Clear local page cache.
     *
     * POSTCONDITIONS:
     *     All cached pages discarded.
     * ---------------------------------------------------------------------- */
    void clear_cache();

    /* -------------------------------------------------------------------------
     * METHOD: set_cache_size
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Set maximum local cache size.
     *
     * PARAMETERS:
     *     num_pages [in] - Maximum pages to cache (0 = disable)
     *
     * POSTCONDITIONS:
     *     Excess pages evicted if current > new limit.
     * ---------------------------------------------------------------------- */
    void set_cache_size(std::size_t num_pages);

    /* -------------------------------------------------------------------------
     * STATIC METHOD: parse_s3_url
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Parse S3 URL into bucket and key.
     *
     * PARAMETERS:
     *     url    [in]  - S3 URL (s3://bucket/key)
     *     bucket [out] - Extracted bucket name
     *     key    [out] - Extracted object key
     *
     * RETURNS:
     *     True if URL is valid S3 format.
     * ---------------------------------------------------------------------- */
    static bool parse_s3_url(
        std::string_view url,              // S3 URL to parse
        std::string& bucket,               // Output bucket name
        std::string& key                   // Output object key
    );

    /* -------------------------------------------------------------------------
     * STATIC METHOD: is_s3_url
     * -------------------------------------------------------------------------
     * RETURNS:
     *     True if URL starts with "s3://".
     * ---------------------------------------------------------------------- */
    static bool is_s3_url(std::string_view url) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* =============================================================================
 * FUNCTION: network_protocol_name
 * =============================================================================
 * SUMMARY:
 *     Convert NetworkProtocol enum to human-readable string.
 *
 * PARAMETERS:
 *     protocol [in] - Network protocol enum value
 *
 * RETURNS:
 *     Null-terminated C string (never nullptr).
 * -------------------------------------------------------------------------- */
const char* network_protocol_name(NetworkProtocol protocol) noexcept;

} // namespace scl::mmap::backend
