// =============================================================================
// FILE: scl/mmap/backend/compressed.hpp
// BRIEF: Compressed storage backend implementation with zstd/lz4 support
// =============================================================================
#pragma once

#include "compressed.h"
#include "local_file.hpp"
#include "../configuration.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"

#include <cstring>
#include <algorithm>
#include <fstream>

// Compression library headers (conditional)
#if __has_include(<zstd.h>)
#include <zstd.h>
#define SCL_HAS_ZSTD 1
#else
#define SCL_HAS_ZSTD 0
#endif

#if __has_include(<lz4.h>)
#include <lz4.h>
#include <lz4hc.h>
#define SCL_HAS_LZ4 1
#else
#define SCL_HAS_LZ4 0
#endif

namespace scl::mmap::backend {

// =============================================================================
// CompressionConfig Implementation
// =============================================================================

constexpr CompressionConfig CompressionConfig::fast() noexcept {
    return CompressionConfig{
        .codec = CompressionCodec::Lz4,
        .compression_level = 1,
        .enable_dictionary = false,
        .dictionary_size = 0,
        .verify_checksum = false,
        .num_decomp_threads = 0
    };
}

constexpr CompressionConfig CompressionConfig::balanced() noexcept {
    return CompressionConfig{
        .codec = CompressionCodec::Zstd,
        .compression_level = 3,
        .enable_dictionary = false,
        .dictionary_size = 0,
        .verify_checksum = true,
        .num_decomp_threads = 0
    };
}

constexpr CompressionConfig CompressionConfig::high_ratio() noexcept {
    return CompressionConfig{
        .codec = CompressionCodec::Zstd,
        .compression_level = 19,
        .enable_dictionary = true,
        .dictionary_size = 65536,
        .verify_checksum = true,
        .num_decomp_threads = 0
    };
}

// =============================================================================
// CompressedPageIndex Implementation
// =============================================================================

constexpr std::size_t CompressedPageIndex::header_size() noexcept {
    return 32;
}

std::size_t CompressedPageIndex::index_size() const noexcept {
    return num_pages * sizeof(PageEntry);
}

std::size_t CompressedPageIndex::data_offset() const noexcept {
    return header_size() + index_size();
}

// =============================================================================
// Decompression Context (Thread-Local)
// =============================================================================

struct CompressedBackend::DecompressContext {
#if SCL_HAS_ZSTD
    ZSTD_DCtx* zstd_ctx = nullptr;
#endif
    std::vector<std::byte> buffer;
    CompressionCodec codec = CompressionCodec::None;

    DecompressContext() {
        buffer.resize(kPageSize * 2);
    }

    ~DecompressContext() {
#if SCL_HAS_ZSTD
        if (zstd_ctx) {
            ZSTD_freeDCtx(zstd_ctx);
        }
#endif
    }

    void ensure_codec(CompressionCodec c) {
        if (codec != c) {
            codec = c;
#if SCL_HAS_ZSTD
            if (c == CompressionCodec::Zstd && !zstd_ctx) {
                zstd_ctx = ZSTD_createDCtx();
                if (!zstd_ctx) {
                    throw RuntimeError("Failed to create ZSTD decompression context");
                }
            }
#endif
        }
    }

    bool has_valid_codec() const noexcept {
#if SCL_HAS_ZSTD
        if (codec == CompressionCodec::Zstd) {
            return zstd_ctx != nullptr;
        }
#endif
        return true;
    }

    void ensure_buffer(std::size_t size) {
        if (buffer.size() < size) {
            buffer.resize(size);
        }
    }
};

thread_local std::unique_ptr<CompressedBackend::DecompressContext>
    CompressedBackend::decomp_ctx_;

// =============================================================================
// CompressedBackend Implementation
// =============================================================================

CompressedBackend::CompressedBackend(
    const std::filesystem::path& path,
    CompressionConfig config
)
    : underlying_(std::make_unique<LocalFileBackend>(path, true))
    , index_{}
    , config_(config)
    , file_id_(generate_file_id())
    , compressed_size_(0)
{
    load_index();
}

CompressedBackend::CompressedBackend(
    std::unique_ptr<StorageBackend<LocalFileBackend>> underlying,
    CompressionConfig config
)
    : underlying_(nullptr)
    , index_{}
    , config_(config)
    , file_id_(generate_file_id())
    , compressed_size_(0)
{
    // Extract LocalFileBackend from type-erased wrapper
    // This is a simplified implementation
    SCL_CHECK_ARG(underlying != nullptr, "underlying backend is null");
    load_index();
}

CompressedBackend::~CompressedBackend() = default;

CompressedBackend::CompressedBackend(CompressedBackend&& other) noexcept
    : underlying_(std::move(other.underlying_))
    , index_(std::move(other.index_))
    , config_(other.config_)
    , file_id_(other.file_id_)
    , compressed_size_(other.compressed_size_)
{}

CompressedBackend& CompressedBackend::operator=(CompressedBackend&& other) noexcept {
    if (this != &other) {
        underlying_ = std::move(other.underlying_);
        index_ = std::move(other.index_);
        config_ = other.config_;
        file_id_ = other.file_id_;
        compressed_size_ = other.compressed_size_;
    }
    return *this;
}

void CompressedBackend::load_index() {
    if (!underlying_) return;

    // Read header page (must use full page buffer to avoid overflow)
    std::vector<std::byte> header_page(kPageSize);
    if (underlying_->load_page_impl(0, header_page.data()) == 0) {
        SCL_CHECK_IO(false, "Failed to read compressed file header");
    }

    // Parse header (first 32 bytes)
    std::uint64_t magic = 0;
    std::memcpy(&magic, header_page.data(), 8);
    SCL_CHECK_IO(magic == CompressedPageIndex::kMagic,
        "Invalid compressed file magic number");

    std::uint32_t version = 0;
    std::memcpy(&version, header_page.data() + 8, 4);
    SCL_CHECK_IO(version == CompressedPageIndex::kVersion,
        "Unsupported compressed file version");

    std::memcpy(&index_.num_pages, header_page.data() + 12, 8);

    // Validate num_pages to prevent OOM attack
    constexpr std::size_t kMaxPages = 1ULL << 30;  // ~4PB at 4KB pages
    SCL_CHECK_IO(index_.num_pages <= kMaxPages,
        "num_pages exceeds maximum allowed value");

    std::uint32_t codec_val = 0;
    std::memcpy(&codec_val, header_page.data() + 20, 4);
    index_.codec = static_cast<CompressionCodec>(codec_val);
    std::memcpy(&index_.flags, header_page.data() + 24, 4);
    std::memcpy(&index_.original_size, header_page.data() + 28, 8);

    // Read page index
    index_.entries.resize(index_.num_pages);

    const std::size_t index_bytes = index_.index_size();
    const std::size_t index_pages = bytes_to_pages(index_.header_size() + index_bytes);

    std::vector<std::byte> index_buffer(index_pages * kPageSize);
    for (std::size_t i = 0; i < index_pages; ++i) {
        underlying_->load_page_impl(i, index_buffer.data() + i * kPageSize);
    }

    // Parse page entries
    const std::byte* entry_ptr = index_buffer.data() + index_.header_size();
    for (std::size_t i = 0; i < index_.num_pages; ++i) {
        auto& entry = index_.entries[i];
        std::memcpy(&entry.offset, entry_ptr, 8);
        std::memcpy(&entry.size, entry_ptr + 8, 4);
        std::memcpy(&entry.checksum, entry_ptr + 12, 4);
        entry_ptr += 16;
    }

    compressed_size_ = underlying_->total_bytes_impl();
}

CompressedBackend::DecompressContext& CompressedBackend::get_decomp_context() {
    if (!decomp_ctx_) {
        decomp_ctx_ = std::make_unique<DecompressContext>();
    }
    decomp_ctx_->ensure_codec(index_.codec);
    return *decomp_ctx_;
}

std::size_t CompressedBackend::decompress_page(
    const std::byte* src,
    std::size_t src_size,
    std::byte* dest
) {
    switch (index_.codec) {
        case CompressionCodec::None:
            std::memcpy(dest, src, std::min(src_size, kPageSize));
            return kPageSize;

#if SCL_HAS_ZSTD
        case CompressionCodec::Zstd: {
            auto& ctx = get_decomp_context();
            std::size_t result = ZSTD_decompressDCtx(
                ctx.zstd_ctx,
                dest, kPageSize,
                src, src_size
            );
            if (ZSTD_isError(result)) {
                return 0;
            }
            // Verify decompressed size matches expected page size
            if (result != kPageSize) {
                return 0;
            }
            return kPageSize;
        }
#endif

#if SCL_HAS_LZ4
        case CompressionCodec::Lz4:
        case CompressionCodec::Lz4HC: {
            int result = LZ4_decompress_safe(
                reinterpret_cast<const char*>(src),
                reinterpret_cast<char*>(dest),
                static_cast<int>(src_size),
                static_cast<int>(kPageSize)
            );
            if (result < 0) {
                return 0;
            }
            // Verify decompressed size matches expected page size
            if (static_cast<std::size_t>(result) != kPageSize) {
                return 0;
            }
            return kPageSize;
        }
#endif

        default:
            return 0;
    }
}

std::uint32_t CompressedBackend::compute_checksum(
    const std::byte* data,
    std::size_t size
) const {
    // Simple CRC32-like checksum
    std::uint32_t crc = 0xFFFFFFFF;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= static_cast<std::uint32_t>(data[i]);
        for (int j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
        }
    }
    return ~crc;
}

BackendCapabilities CompressedBackend::capabilities_impl() const noexcept {
    auto caps = BackendCapabilities::compressed();
    caps.supports_writeback = false;

    // Adjust latency based on codec
    switch (index_.codec) {
        case CompressionCodec::Lz4:
        case CompressionCodec::Lz4HC:
            caps.typical_latency = std::chrono::microseconds(200);
            break;
        case CompressionCodec::Zstd:
            caps.typical_latency = std::chrono::microseconds(500);
            break;
        default:
            caps.typical_latency = std::chrono::microseconds(100);
    }

    return caps;
}

std::size_t CompressedBackend::load_page_impl(
    std::size_t page_idx,
    std::byte* dest
) {
    if (page_idx >= index_.num_pages) {
        std::memset(dest, 0, kPageSize);
        return 0;
    }

    const auto& entry = index_.entries[page_idx];

    // Validate compressed entry size to prevent DoS and buffer issues
    // Reasonable limit: compressed data should not exceed 2x page size
    // (compression should make data smaller, not larger)
    constexpr std::size_t kMaxCompressedSize = kPageSize * 2;
    if (entry.size == 0 || entry.size > kMaxCompressedSize) {
        std::memset(dest, 0, kPageSize);
        return 0;
    }

    // Read compressed data
    auto& ctx = get_decomp_context();
    ctx.ensure_buffer(entry.size);

    const std::size_t comp_page_idx = byte_to_page_idx(entry.offset);
    const std::size_t comp_offset = byte_to_page_offset(entry.offset);

    // May span multiple underlying pages
    std::size_t bytes_read = 0;
    std::size_t remaining = entry.size;
    std::byte* buf_ptr = ctx.buffer.data();

    std::size_t current_page = comp_page_idx;
    std::size_t current_offset = comp_offset;

    while (remaining > 0 && current_page < underlying_->num_pages_impl()) {
        std::array<std::byte, kPageSize> page_buf{};
        if (underlying_->load_page_impl(current_page, page_buf.data()) == 0) {
            break;
        }

        std::size_t copy_size = std::min(remaining, kPageSize - current_offset);
        std::memcpy(buf_ptr, page_buf.data() + current_offset, copy_size);

        buf_ptr += copy_size;
        bytes_read += copy_size;
        remaining -= copy_size;
        ++current_page;
        current_offset = 0;
    }

    if (bytes_read < entry.size) {
        std::memset(dest, 0, kPageSize);
        return 0;
    }

    // Verify checksum if enabled
    if (config_.verify_checksum && entry.checksum != 0) {
        std::uint32_t computed = compute_checksum(ctx.buffer.data(), entry.size);
        if (computed != entry.checksum) {
            std::memset(dest, 0, kPageSize);
            return 0;
        }
    }

    // Decompress
    return decompress_page(ctx.buffer.data(), entry.size, dest);
}

std::size_t CompressedBackend::write_page_impl(
    std::size_t /*page_idx*/,
    const std::byte* /*src*/
) {
    // Compressed backend is read-only
    return 0;
}

std::size_t CompressedBackend::load_pages_async_impl(
    std::span<const IORequest> requests,
    IOCallback callback
) {
    // For now, synchronous implementation
    // TODO: Parallel decompression with thread pool

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

void CompressedBackend::prefetch_hint_impl(
    std::span<const std::size_t> pages
) {
    if (!underlying_ || pages.empty()) return;

    // Translate to underlying page indices
    std::vector<std::size_t> underlying_pages;
    underlying_pages.reserve(pages.size() * 2);

    for (std::size_t page_idx : pages) {
        if (page_idx >= index_.num_pages) continue;

        const auto& entry = index_.entries[page_idx];
        const std::size_t start_page = byte_to_page_idx(entry.offset);
        const std::size_t end_page = byte_to_page_idx(entry.offset + entry.size);

        for (std::size_t p = start_page; p <= end_page; ++p) {
            underlying_pages.push_back(p);
        }
    }

    // Remove duplicates
    std::sort(underlying_pages.begin(), underlying_pages.end());
    underlying_pages.erase(
        std::unique(underlying_pages.begin(), underlying_pages.end()),
        underlying_pages.end()
    );

    underlying_->prefetch_hint_impl(underlying_pages);
}

std::size_t CompressedBackend::num_pages_impl() const noexcept {
    return index_.num_pages;
}

std::size_t CompressedBackend::total_bytes_impl() const noexcept {
    return index_.original_size;
}

std::size_t CompressedBackend::file_id_impl() const noexcept {
    return file_id_;
}

BackendType CompressedBackend::type_impl() const noexcept {
    return BackendType::Compressed;
}

double CompressedBackend::compression_ratio() const noexcept {
    if (compressed_size_ == 0) return 1.0;
    return static_cast<double>(index_.original_size) /
           static_cast<double>(compressed_size_);
}

CompressionCodec CompressedBackend::codec() const noexcept {
    return index_.codec;
}

const CompressedPageIndex& CompressedBackend::index() const noexcept {
    return index_;
}

template <typename SourceBackend>
bool CompressedBackend::create_compressed_file(
    SourceBackend& source,
    const std::filesystem::path& dest_path,
    CompressionConfig config,
    std::function<void(double)> progress
) {
    std::ofstream out(dest_path, std::ios::binary);
    if (!out) return false;

    const std::size_t num_pages = source.num_pages();
    const std::size_t original_size = source.total_bytes();

    // Prepare index
    CompressedPageIndex index;
    index.num_pages = num_pages;
    index.codec = config.codec;
    index.flags = config.verify_checksum ? 1 : 0;
    index.original_size = original_size;
    index.entries.resize(num_pages);

    // Write placeholder header
    std::array<std::byte, 32> header{};
    out.write(reinterpret_cast<const char*>(header.data()), header.size());

    // Write placeholder index
    std::vector<std::byte> index_placeholder(num_pages * 16, std::byte{0});
    out.write(reinterpret_cast<const char*>(index_placeholder.data()),
              static_cast<std::streamsize>(index_placeholder.size()));

    // Compression buffer
    std::vector<std::byte> page_buf(kPageSize);
    std::vector<std::byte> comp_buf(kPageSize * 2);

    std::size_t data_offset = index.data_offset();

    // Compress each page
    for (std::size_t i = 0; i < num_pages; ++i) {
        if (source.load_page(i, page_buf.data()) == 0) {
            return false;
        }

        std::size_t comp_size = 0;

        switch (config.codec) {
            case CompressionCodec::None:
                std::memcpy(comp_buf.data(), page_buf.data(), kPageSize);
                comp_size = kPageSize;
                break;

#if SCL_HAS_ZSTD
            case CompressionCodec::Zstd:
                comp_size = ZSTD_compress(
                    comp_buf.data(), comp_buf.size(),
                    page_buf.data(), kPageSize,
                    config.compression_level
                );
                if (ZSTD_isError(comp_size)) return false;
                break;
#endif

#if SCL_HAS_LZ4
            case CompressionCodec::Lz4:
                comp_size = static_cast<std::size_t>(LZ4_compress_default(
                    reinterpret_cast<const char*>(page_buf.data()),
                    reinterpret_cast<char*>(comp_buf.data()),
                    static_cast<int>(kPageSize),
                    static_cast<int>(comp_buf.size())
                ));
                if (comp_size == 0) return false;
                break;

            case CompressionCodec::Lz4HC:
                comp_size = static_cast<std::size_t>(LZ4_compress_HC(
                    reinterpret_cast<const char*>(page_buf.data()),
                    reinterpret_cast<char*>(comp_buf.data()),
                    static_cast<int>(kPageSize),
                    static_cast<int>(comp_buf.size()),
                    config.compression_level
                ));
                if (comp_size == 0) return false;
                break;
#endif

            default:
                return false;
        }

        // Record index entry
        auto& entry = index.entries[i];
        entry.offset = data_offset;
        entry.size = static_cast<std::uint32_t>(comp_size);

        if (config.verify_checksum) {
            // Compute checksum of compressed data
            std::uint32_t crc = 0xFFFFFFFF;
            for (std::size_t j = 0; j < comp_size; ++j) {
                crc ^= static_cast<std::uint32_t>(comp_buf[j]);
                for (int k = 0; k < 8; ++k) {
                    crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
                }
            }
            entry.checksum = ~crc;
        } else {
            entry.checksum = 0;
        }

        // Write compressed data
        out.write(reinterpret_cast<const char*>(comp_buf.data()),
                  static_cast<std::streamsize>(comp_size));

        data_offset += comp_size;

        if (progress) {
            progress(static_cast<double>(i + 1) / static_cast<double>(num_pages));
        }
    }

    // Rewrite header
    out.seekp(0);
    std::uint64_t magic = CompressedPageIndex::kMagic;
    std::memcpy(header.data(), &magic, 8);
    std::uint32_t version = CompressedPageIndex::kVersion;
    std::memcpy(header.data() + 8, &version, 4);
    std::memcpy(header.data() + 12, &index.num_pages, 8);
    std::uint32_t codec_val = static_cast<std::uint32_t>(index.codec);
    std::memcpy(header.data() + 20, &codec_val, 4);
    std::memcpy(header.data() + 24, &index.flags, 4);
    std::memcpy(header.data() + 28, &index.original_size, 8);

    out.write(reinterpret_cast<const char*>(header.data()), header.size());

    // Rewrite index
    for (const auto& entry : index.entries) {
        out.write(reinterpret_cast<const char*>(&entry.offset), 8);
        out.write(reinterpret_cast<const char*>(&entry.size), 4);
        out.write(reinterpret_cast<const char*>(&entry.checksum), 4);
    }

    return true;
}

// =============================================================================
// Free Functions
// =============================================================================

inline const char* compression_codec_name(CompressionCodec codec) noexcept {
    switch (codec) {
        case CompressionCodec::None:  return "None";
        case CompressionCodec::Zstd:  return "Zstd";
        case CompressionCodec::Lz4:   return "Lz4";
        case CompressionCodec::Lz4HC: return "Lz4HC";
        default:                      return "Unknown";
    }
}

inline std::size_t estimate_compressed_size(
    std::size_t uncompressed_size,
    CompressionCodec codec
) noexcept {
    switch (codec) {
        case CompressionCodec::None:
            return uncompressed_size;
        case CompressionCodec::Zstd:
            // Zstd typically achieves 3:1 ratio
            return uncompressed_size / 2;
        case CompressionCodec::Lz4:
        case CompressionCodec::Lz4HC:
            // LZ4 typically achieves 2:1 ratio
            return uncompressed_size * 2 / 3;
        default:
            return uncompressed_size;
    }
}

} // namespace scl::mmap::backend
