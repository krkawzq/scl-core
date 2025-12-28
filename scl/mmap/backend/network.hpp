// =============================================================================
// FILE: scl/mmap/backend/network.hpp
// BRIEF: Network storage backend implementation (S3/HTTP)
// =============================================================================
#pragma once

#include "network.h"
#include "backend.hpp"
#include "../configuration.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"

#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <unordered_map>
#include <queue>
#include <list>
#include <atomic>
#include <thread>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <cstdlib>

// Optional curl support
#if __has_include(<curl/curl.h>)
#include <curl/curl.h>
#define SCL_HAS_CURL 1
#else
#define SCL_HAS_CURL 0
#endif

namespace scl::mmap::backend {

// =============================================================================
// Curl Global Initialization Manager (Thread-Safe Singleton)
// =============================================================================

#if SCL_HAS_CURL
namespace detail {

// Thread-safe curl global init/cleanup using reference counting
class CurlGlobalManager {
public:
    static void acquire() {
        std::call_once(init_flag_, []() {
            curl_global_init(CURL_GLOBAL_DEFAULT);
        });
        ref_count_.fetch_add(1, std::memory_order_relaxed);
    }

    static void release() {
        if (ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // Last user - cleanup
            std::lock_guard<std::mutex> lock(cleanup_mutex_);
            // Double-check after acquiring lock
            if (ref_count_.load(std::memory_order_acquire) == 0) {
                curl_global_cleanup();
            }
        }
    }

private:
    static std::once_flag init_flag_;
    static std::atomic<int> ref_count_;
    static std::mutex cleanup_mutex_;
};

// Static member definitions
inline std::once_flag CurlGlobalManager::init_flag_;
inline std::atomic<int> CurlGlobalManager::ref_count_{0};
inline std::mutex CurlGlobalManager::cleanup_mutex_;

} // namespace detail
#endif // SCL_HAS_CURL

// =============================================================================
// S3Credentials Implementation
// =============================================================================

S3Credentials S3Credentials::from_environment() {
    S3Credentials creds;

    if (const char* key = std::getenv("AWS_ACCESS_KEY_ID")) {
        creds.access_key_id = key;
    }
    if (const char* secret = std::getenv("AWS_SECRET_ACCESS_KEY")) {
        creds.secret_access_key = secret;
    }
    if (const char* token = std::getenv("AWS_SESSION_TOKEN")) {
        creds.session_token = token;
    }
    if (const char* region = std::getenv("AWS_REGION")) {
        creds.region = region;
    } else if (const char* default_region = std::getenv("AWS_DEFAULT_REGION")) {
        creds.region = default_region;
    }

    return creds;
}

bool S3Credentials::is_valid() const noexcept {
    return !access_key_id.empty() && !secret_access_key.empty();
}

// =============================================================================
// NetworkConfig Implementation
// =============================================================================

NetworkConfig NetworkConfig::s3_default() {
    return NetworkConfig{
        .protocol = NetworkProtocol::S3,
        .endpoint = "",
        .s3_credentials = S3Credentials::from_environment(),
        .connect_timeout = std::chrono::milliseconds(5000),
        .read_timeout = std::chrono::milliseconds(30000),
        .max_retries = 3,
        .retry_delay = std::chrono::milliseconds(100),
        .max_connections = 8,
        .enable_compression = true,
        .verify_ssl = true,
        .user_agent = "scl-mmap/1.0"
    };
}

NetworkConfig NetworkConfig::http_default() {
    return NetworkConfig{
        .protocol = NetworkProtocol::HTTPS,
        .endpoint = "",
        .s3_credentials = {},
        .connect_timeout = std::chrono::milliseconds(5000),
        .read_timeout = std::chrono::milliseconds(30000),
        .max_retries = 3,
        .retry_delay = std::chrono::milliseconds(100),
        .max_connections = 4,
        .enable_compression = true,
        .verify_ssl = true,
        .user_agent = "scl-mmap/1.0"
    };
}

NetworkConfig NetworkConfig::low_latency() {
    return NetworkConfig{
        .protocol = NetworkProtocol::HTTPS,
        .endpoint = "",
        .s3_credentials = S3Credentials::from_environment(),
        .connect_timeout = std::chrono::milliseconds(2000),
        .read_timeout = std::chrono::milliseconds(10000),
        .max_retries = 1,
        .retry_delay = std::chrono::milliseconds(50),
        .max_connections = 16,
        .enable_compression = false,
        .verify_ssl = true,
        .user_agent = "scl-mmap/1.0"
    };
}

NetworkConfig NetworkConfig::high_throughput() {
    return NetworkConfig{
        .protocol = NetworkProtocol::HTTPS,
        .endpoint = "",
        .s3_credentials = S3Credentials::from_environment(),
        .connect_timeout = std::chrono::milliseconds(10000),
        .read_timeout = std::chrono::milliseconds(60000),
        .max_retries = 5,
        .retry_delay = std::chrono::milliseconds(200),
        .max_connections = 32,
        .enable_compression = true,
        .verify_ssl = true,
        .user_agent = "scl-mmap/1.0"
    };
}

// =============================================================================
// NetworkStats Implementation
// =============================================================================

std::chrono::nanoseconds NetworkStats::avg_latency() const noexcept {
    if (requests_sent == 0) return std::chrono::nanoseconds(0);
    return total_latency / requests_sent;
}

double NetworkStats::throughput() const noexcept {
    if (total_latency.count() == 0) return 0.0;
    double seconds = total_latency.count() / 1e9;
    double mb = bytes_downloaded / (1024.0 * 1024.0);
    return mb / seconds;
}

// =============================================================================
// Local Cache Entry
// =============================================================================

struct CacheEntry {
    std::vector<std::byte> data;
    std::chrono::steady_clock::time_point last_access;
    std::size_t access_count;
    std::list<std::size_t>::iterator lru_it;  // Iterator into LRU list for O(1) removal
};

// =============================================================================
// NetworkBackend::Impl
// =============================================================================

struct NetworkBackend::Impl {
    std::string url;
    std::string bucket;
    std::string key;
    NetworkConfig config;
    ObjectInfo object_info_{};
    std::size_t file_id;
    std::size_t num_pages_;

    // Local cache with O(1) LRU eviction
    std::unordered_map<std::size_t, CacheEntry> cache;
    std::list<std::size_t> lru_order;  // Front = most recently used, Back = least recently used
    std::size_t max_cache_pages = 64;
    mutable std::shared_mutex cache_mutex;

    // Statistics
    std::atomic<std::size_t> requests_sent{0};
    std::atomic<std::size_t> bytes_downloaded{0};
    std::atomic<std::size_t> bytes_uploaded{0};
    std::atomic<std::size_t> cache_hits{0};
    std::atomic<std::size_t> cache_misses{0};
    std::atomic<std::size_t> retries{0};
    std::atomic<std::size_t> errors{0};
    std::atomic<std::size_t> total_latency_ns{0};

#if SCL_HAS_CURL
    // Connection pool
    CURLM* multi_handle = nullptr;
    std::vector<CURL*> idle_handles;
    std::mutex pool_mutex;
#endif

    Impl(std::string_view url_str, NetworkConfig cfg)
        : url(url_str)
        , config(cfg)
        , file_id(generate_file_id())
    {
        parse_url();
        init_curl();
        fetch_metadata();
    }

    ~Impl() {
        cleanup_curl();
    }

    void parse_url() {
        if (NetworkBackend::is_s3_url(url)) {
            NetworkBackend::parse_s3_url(url, bucket, key);
            if (config.protocol != NetworkProtocol::S3) {
                config.protocol = NetworkProtocol::S3;
            }
        }
    }

    void init_curl() {
#if SCL_HAS_CURL
        detail::CurlGlobalManager::acquire();
        multi_handle = curl_multi_init();

        // Pre-create some handles
        for (std::size_t i = 0; i < config.max_connections; ++i) {
            CURL* handle = curl_easy_init();
            if (handle) {
                configure_handle(handle);
                idle_handles.push_back(handle);
            }
        }
#endif
    }

    void cleanup_curl() {
#if SCL_HAS_CURL
        for (CURL* handle : idle_handles) {
            curl_easy_cleanup(handle);
        }
        idle_handles.clear();

        if (multi_handle) {
            curl_multi_cleanup(multi_handle);
            multi_handle = nullptr;
        }

        detail::CurlGlobalManager::release();
#endif
    }

#if SCL_HAS_CURL
    void configure_handle(CURL* handle) {
        curl_easy_setopt(handle, CURLOPT_USERAGENT, config.user_agent.c_str());
        curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS,
                        static_cast<long>(config.connect_timeout.count()));
        curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS,
                        static_cast<long>(config.read_timeout.count()));
        curl_easy_setopt(handle, CURLOPT_SSL_VERIFYPEER,
                        config.verify_ssl ? 1L : 0L);
        curl_easy_setopt(handle, CURLOPT_SSL_VERIFYHOST,
                        config.verify_ssl ? 2L : 0L);
        curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(handle, CURLOPT_MAXREDIRS, 5L);

        if (config.enable_compression) {
            curl_easy_setopt(handle, CURLOPT_ACCEPT_ENCODING, "gzip, deflate");
        }
    }

    CURL* acquire_handle() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        if (!idle_handles.empty()) {
            CURL* handle = idle_handles.back();
            idle_handles.pop_back();
            return handle;
        }
        CURL* handle = curl_easy_init();
        if (handle) {
            configure_handle(handle);
        }
        return handle;
    }

    void release_handle(CURL* handle) {
        if (!handle) return;
        curl_easy_reset(handle);
        configure_handle(handle);

        std::lock_guard<std::mutex> lock(pool_mutex);
        if (idle_handles.size() < config.max_connections) {
            idle_handles.push_back(handle);
        } else {
            curl_easy_cleanup(handle);
        }
    }
#endif

    std::string build_request_url() const {
        if (config.protocol == NetworkProtocol::S3) {
            // Build S3 URL
            std::string endpoint = config.endpoint;
            if (endpoint.empty()) {
                endpoint = "https://s3." + config.s3_credentials.region +
                          ".amazonaws.com";
            }
            return endpoint + "/" + bucket + "/" + key;
        }
        return url;
    }

    void fetch_metadata() {
#if SCL_HAS_CURL
        CURL* handle = acquire_handle();
        if (!handle) {
            SCL_CHECK_IO(false, "Failed to acquire curl handle");
        }

        std::string request_url = build_request_url();
        curl_easy_setopt(handle, CURLOPT_URL, request_url.c_str());
        curl_easy_setopt(handle, CURLOPT_NOBODY, 1L);  // HEAD request

        struct HeaderData {
            std::size_t content_length = 0;
            std::string etag;
            std::string content_type;
            std::string last_modified;
        } headers;

        curl_easy_setopt(handle, CURLOPT_HEADERFUNCTION,
            +[](char* buffer, std::size_t size, std::size_t nmemb, void* userdata) -> std::size_t {
                std::size_t total = size * nmemb;
                auto* h = static_cast<HeaderData*>(userdata);
                std::string_view line(buffer, total);

                // Helper to extract header value after colon, handling optional whitespace
                auto extract_value = [](std::string_view line, std::size_t header_len) -> std::string_view {
                    if (line.size() <= header_len) return {};
                    std::string_view value = line.substr(header_len);
                    // Skip leading whitespace
                    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
                        value.remove_prefix(1);
                    }
                    // Trim trailing CRLF
                    while (!value.empty() && (value.back() == '\r' || value.back() == '\n')) {
                        value.remove_suffix(1);
                    }
                    return value;
                };

                // Case-insensitive header matching with proper length handling
                // "Content-Length:" = 15 chars, "ETag:" = 5 chars, "Content-Type:" = 13 chars
                if (line.size() > 15 &&
                    (line.substr(0, 15) == "Content-Length:" || line.substr(0, 15) == "content-length:")) {
                    auto value = extract_value(line, 15);
                    if (!value.empty()) {
                        try {
                            h->content_length = std::stoull(std::string(value));
                        } catch (...) {
                            // Malformed Content-Length, ignore
                        }
                    }
                } else if (line.size() > 5 &&
                           (line.substr(0, 5) == "ETag:" || line.substr(0, 5) == "etag:")) {
                    auto value = extract_value(line, 5);
                    h->etag = std::string(value);
                    // Trim surrounding quotes
                    while (!h->etag.empty() && h->etag.front() == '"') {
                        h->etag.erase(0, 1);
                    }
                    while (!h->etag.empty() && h->etag.back() == '"') {
                        h->etag.pop_back();
                    }
                } else if (line.size() > 13 &&
                           (line.substr(0, 13) == "Content-Type:" || line.substr(0, 13) == "content-type:")) {
                    auto value = extract_value(line, 13);
                    h->content_type = std::string(value);
                }
                return total;
            });
        curl_easy_setopt(handle, CURLOPT_HEADERDATA, &headers);

        CURLcode res = curl_easy_perform(handle);

        long http_code = 0;
        curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &http_code);

        release_handle(handle);

        if (res != CURLE_OK || http_code >= 400) {
            object_info_.exists = false;
            SCL_CHECK_IO(false, "Failed to fetch object metadata: HTTP " +
                        std::to_string(http_code));
        }

        object_info_.size = headers.content_length;
        object_info_.etag = headers.etag;
        object_info_.content_type = headers.content_type;
        object_info_.exists = true;
        object_info_.last_modified = std::chrono::system_clock::now();

        num_pages_ = bytes_to_pages(object_info_.size);
#else
        // Without curl, use mock data
        object_info_.size = 0;
        object_info_.exists = false;
        num_pages_ = 0;
#endif
    }

    bool check_cache(std::size_t page_idx, std::byte* dest) {
        std::unique_lock lock(cache_mutex);  // Need unique_lock for LRU update
        auto it = cache.find(page_idx);
        if (it != cache.end()) {
            std::memcpy(dest, it->second.data.data(), kPageSize);
            // Move to front of LRU (most recently used)
            lru_order.erase(it->second.lru_it);
            lru_order.push_front(page_idx);
            it->second.lru_it = lru_order.begin();
            it->second.last_access = std::chrono::steady_clock::now();
            it->second.access_count++;
            cache_hits.fetch_add(1);
            return true;
        }
        cache_misses.fetch_add(1);
        return false;
    }

    void update_cache(std::size_t page_idx, const std::byte* data) {
        std::unique_lock lock(cache_mutex);

        // Check if already cached (update in place)
        auto existing = cache.find(page_idx);
        if (existing != cache.end()) {
            std::memcpy(existing->second.data.data(), data, kPageSize);
            // Move to front of LRU
            lru_order.erase(existing->second.lru_it);
            lru_order.push_front(page_idx);
            existing->second.lru_it = lru_order.begin();
            existing->second.last_access = std::chrono::steady_clock::now();
            return;
        }

        // Evict if at capacity - O(1) eviction from back of LRU list
        while (cache.size() >= max_cache_pages && !lru_order.empty()) {
            std::size_t evict_page = lru_order.back();
            lru_order.pop_back();
            cache.erase(evict_page);
        }

        // Insert new entry
        CacheEntry entry;
        entry.data.resize(kPageSize);
        std::memcpy(entry.data.data(), data, kPageSize);
        entry.last_access = std::chrono::steady_clock::now();
        entry.access_count = 1;

        // Add to front of LRU (most recently used)
        lru_order.push_front(page_idx);
        entry.lru_it = lru_order.begin();

        cache[page_idx] = std::move(entry);
    }

    std::size_t fetch_page(std::size_t page_idx, std::byte* dest) {
#if SCL_HAS_CURL
        CURL* handle = acquire_handle();
        if (!handle) return 0;

        std::string request_url = build_request_url();
        curl_easy_setopt(handle, CURLOPT_URL, request_url.c_str());
        curl_easy_setopt(handle, CURLOPT_NOBODY, 0L);
        curl_easy_setopt(handle, CURLOPT_HTTPGET, 1L);

        // Set range header
        std::size_t start = page_to_byte_offset(page_idx);
        std::size_t end = std::min(start + kPageSize - 1, object_info_.size - 1);
        std::string range = std::to_string(start) + "-" + std::to_string(end);
        curl_easy_setopt(handle, CURLOPT_RANGE, range.c_str());

        // Response buffer
        struct WriteData {
            std::byte* dest;
            std::size_t written;
            std::size_t max_size;
        } write_data{dest, 0, kPageSize};

        curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION,
            +[](char* ptr, std::size_t size, std::size_t nmemb, void* userdata) -> std::size_t {
                auto* wd = static_cast<WriteData*>(userdata);
                std::size_t total = size * nmemb;
                std::size_t to_copy = std::min(total, wd->max_size - wd->written);
                std::memcpy(wd->dest + wd->written, ptr, to_copy);
                wd->written += to_copy;
                return total;
            });
        curl_easy_setopt(handle, CURLOPT_WRITEDATA, &write_data);

        auto start_time = std::chrono::steady_clock::now();

        CURLcode res = CURLE_OK;
        std::size_t retry_count = 0;
        std::chrono::milliseconds delay = config.retry_delay;

        while (retry_count <= config.max_retries) {
            write_data.written = 0;
            res = curl_easy_perform(handle);

            long http_code = 0;
            curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &http_code);

            if (res == CURLE_OK && http_code >= 200 && http_code < 300) {
                break;  // Success
            }

            if (http_code >= 400 && http_code < 500) {
                // Client error, don't retry
                break;
            }

            // Transient error, retry
            ++retry_count;
            retries.fetch_add(1);
            if (retry_count <= config.max_retries) {
                std::this_thread::sleep_for(delay);
                delay *= 2;  // Exponential backoff
            }
        }

        auto elapsed = std::chrono::steady_clock::now() - start_time;
        total_latency_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()
        );
        requests_sent.fetch_add(1);

        release_handle(handle);

        if (res != CURLE_OK) {
            errors.fetch_add(1);
            std::memset(dest, 0, kPageSize);
            return 0;
        }

        bytes_downloaded.fetch_add(write_data.written);

        // Zero remaining bytes if partial page
        if (write_data.written < kPageSize) {
            std::memset(dest + write_data.written, 0, kPageSize - write_data.written);
        }

        // Update cache
        update_cache(page_idx, dest);

        return kPageSize;
#else
        // Without curl, return zeros
        std::memset(dest, 0, kPageSize);
        return 0;
#endif
    }

    NetworkStats get_stats() const {
        return NetworkStats{
            .requests_sent = requests_sent.load(),
            .bytes_downloaded = bytes_downloaded.load(),
            .bytes_uploaded = bytes_uploaded.load(),
            .cache_hits = cache_hits.load(),
            .cache_misses = cache_misses.load(),
            .retries = retries.load(),
            .errors = errors.load(),
            .total_latency = std::chrono::nanoseconds(total_latency_ns.load())
        };
    }

    void reset_stats() {
        requests_sent.store(0);
        bytes_downloaded.store(0);
        bytes_uploaded.store(0);
        cache_hits.store(0);
        cache_misses.store(0);
        retries.store(0);
        errors.store(0);
        total_latency_ns.store(0);
    }
};

// =============================================================================
// NetworkBackend Implementation
// =============================================================================

NetworkBackend::NetworkBackend(std::string_view url, NetworkConfig config)
    : impl_(std::make_unique<Impl>(url, config)) {}

NetworkBackend::NetworkBackend(
    std::string_view bucket,
    std::string_view key,
    NetworkConfig config
) : impl_(std::make_unique<Impl>(
        std::string("s3://") + std::string(bucket) + "/" + std::string(key),
        config
    ))
{
    impl_->bucket = bucket;
    impl_->key = key;
}

NetworkBackend::~NetworkBackend() = default;

NetworkBackend::NetworkBackend(NetworkBackend&& other) noexcept = default;
NetworkBackend& NetworkBackend::operator=(NetworkBackend&& other) noexcept = default;

BackendCapabilities NetworkBackend::capabilities_impl() const noexcept {
    auto caps = BackendCapabilities::network();
    caps.typical_latency = std::chrono::milliseconds(50);  // Network RTT
    return caps;
}

std::size_t NetworkBackend::load_page_impl(std::size_t page_idx, std::byte* dest) {
    if (page_idx >= impl_->num_pages_) {
        std::memset(dest, 0, kPageSize);
        return 0;
    }

    // Check cache first
    if (impl_->check_cache(page_idx, dest)) {
        return kPageSize;
    }

    // Fetch from network
    return impl_->fetch_page(page_idx, dest);
}

std::size_t NetworkBackend::write_page_impl(
    std::size_t /*page_idx*/,
    const std::byte* /*src*/
) {
    // Network backend is read-only
    return 0;
}

std::size_t NetworkBackend::load_pages_async_impl(
    std::span<const IORequest> requests,
    IOCallback callback
) {
    // For now, implement synchronously
    // TODO: Use curl_multi for true async

    std::size_t submitted = 0;

    for (const auto& req : requests) {
        auto start = std::chrono::steady_clock::now();

        std::size_t bytes = load_page_impl(req.page_idx, req.dest);

        auto end = std::chrono::steady_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        IOResult result{
            .page_idx = req.page_idx,
            .bytes_transferred = bytes,
            .latency = latency,
            .error = (bytes > 0) ? 0 : -1,
            .user_data = req.user_data
        };

        callback(result);
        ++submitted;
    }

    return submitted;
}

void NetworkBackend::prefetch_hint_impl(std::span<const std::size_t> pages) {
    // Prefetch pages into local cache
    for (std::size_t page_idx : pages) {
        if (page_idx >= impl_->num_pages_) continue;

        // Check if already cached
        {
            std::shared_lock lock(impl_->cache_mutex);
            if (impl_->cache.find(page_idx) != impl_->cache.end()) {
                continue;
            }
        }

        // Fetch and cache
        std::vector<std::byte> buffer(kPageSize);
        impl_->fetch_page(page_idx, buffer.data());
    }
}

std::size_t NetworkBackend::num_pages_impl() const noexcept {
    return impl_->num_pages_;
}

std::size_t NetworkBackend::total_bytes_impl() const noexcept {
    return impl_->object_info_.size;
}

std::size_t NetworkBackend::file_id_impl() const noexcept {
    return impl_->file_id;
}

BackendType NetworkBackend::type_impl() const noexcept {
    return BackendType::Network;
}

const std::string& NetworkBackend::url() const noexcept {
    return impl_->url;
}

const ObjectInfo& NetworkBackend::object_info() const noexcept {
    return impl_->object_info_;
}

const NetworkConfig& NetworkBackend::config() const noexcept {
    return impl_->config;
}

NetworkStats NetworkBackend::stats() const noexcept {
    return impl_->get_stats();
}

void NetworkBackend::reset_stats() noexcept {
    impl_->reset_stats();
}

bool NetworkBackend::refresh_metadata() {
    impl_->fetch_metadata();
    return impl_->object_info_.exists;
}

void NetworkBackend::clear_cache() {
    std::unique_lock lock(impl_->cache_mutex);
    impl_->cache.clear();
    impl_->lru_order.clear();
}

void NetworkBackend::set_cache_size(std::size_t num_pages) {
    impl_->max_cache_pages = num_pages;

    // Evict excess using LRU order (O(1) per eviction)
    std::unique_lock lock(impl_->cache_mutex);
    while (impl_->cache.size() > num_pages && !impl_->lru_order.empty()) {
        std::size_t evict_page = impl_->lru_order.back();
        impl_->lru_order.pop_back();
        impl_->cache.erase(evict_page);
    }
}

bool NetworkBackend::parse_s3_url(
    std::string_view url,
    std::string& bucket,
    std::string& key
) {
    if (!is_s3_url(url)) return false;

    // Skip "s3://"
    std::string_view remainder = url.substr(5);

    auto slash_pos = remainder.find('/');
    if (slash_pos == std::string_view::npos) {
        bucket = std::string(remainder);
        key = "";
    } else {
        bucket = std::string(remainder.substr(0, slash_pos));
        key = std::string(remainder.substr(slash_pos + 1));
    }

    return !bucket.empty();
}

bool NetworkBackend::is_s3_url(std::string_view url) noexcept {
    return url.size() > 5 && url.substr(0, 5) == "s3://";
}

// =============================================================================
// Free Functions
// =============================================================================

inline const char* network_protocol_name(NetworkProtocol protocol) noexcept {
    switch (protocol) {
        case NetworkProtocol::HTTP:  return "HTTP";
        case NetworkProtocol::HTTPS: return "HTTPS";
        case NetworkProtocol::S3:    return "S3";
        case NetworkProtocol::GCS:   return "GCS";
        case NetworkProtocol::Azure: return "Azure";
        default:                     return "Unknown";
    }
}

} // namespace scl::mmap::backend
