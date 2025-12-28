// =============================================================================
// FILE: scl/mmap/backend/local_file.hpp
// BRIEF: Local filesystem storage backend implementation
// =============================================================================
#pragma once

#include "local_file.h"
#include "backend.hpp"
#include "../configuration.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>
#include <algorithm>
#include <utility>

#ifdef __linux__
#include <linux/mman.h>
#endif

namespace scl::mmap::backend {

// =============================================================================
// LocalFileConfig Implementation
// =============================================================================

constexpr LocalFileConfig LocalFileConfig::sequential() noexcept {
    return LocalFileConfig{
        .mode = LocalFileMode::MMap,
        .use_direct_io = false,
        .use_huge_pages = false,
        .prefetch_window = 8,
        .read_buffer_size = 0
    };
}

constexpr LocalFileConfig LocalFileConfig::random_access() noexcept {
    return LocalFileConfig{
        .mode = LocalFileMode::PRead,
        .use_direct_io = false,
        .use_huge_pages = false,
        .prefetch_window = 1,
        .read_buffer_size = kPageSize
    };
}

constexpr LocalFileConfig LocalFileConfig::streaming() noexcept {
    return LocalFileConfig{
        .mode = LocalFileMode::MMap,
        .use_direct_io = true,
        .use_huge_pages = true,
        .prefetch_window = 4,
        .read_buffer_size = 0
    };
}

// =============================================================================
// LocalFileBackend Implementation
// =============================================================================

LocalFileBackend::LocalFileBackend(
    const std::filesystem::path& path,
    bool read_only,
    LocalFileConfig config
)
    : fd_(-1)
    , mmap_base_(nullptr)
    , file_size_(0)
    , file_id_(generate_file_id())
    , num_pages_(0)
    , path_(path)
    , config_(config)
    , read_only_(read_only)
    , use_mmap_(false)
{
    int flags = read_only ? O_RDONLY : O_RDWR;

#ifdef O_DIRECT
    if (config_.use_direct_io) {
        flags |= O_DIRECT;
    }
#endif

    fd_ = ::open(path.c_str(), flags);
    SCL_CHECK_IO(fd_ >= 0, "Failed to open file: " + path.string());

    struct stat st;
    SCL_CHECK_IO(::fstat(fd_, &st) == 0, "Failed to stat file");

    file_size_ = static_cast<std::size_t>(st.st_size);
    num_pages_ = bytes_to_pages(file_size_);

    // Determine I/O mode
    if (config_.mode == LocalFileMode::Auto) {
        // Use mmap for large files with sequential access
        use_mmap_ = (file_size_ > 1024 * 1024 * 1024);  // > 1GB
    } else {
        use_mmap_ = (config_.mode == LocalFileMode::MMap);
    }

    if (use_mmap_) {
        setup_mmap();
    }

    // Apply initial access hints
    apply_fadvise(POSIX_FADV_SEQUENTIAL);
}

LocalFileBackend::LocalFileBackend(
    int fd,
    std::size_t total_bytes,
    bool read_only,
    LocalFileConfig config
)
    : fd_(fd)
    , mmap_base_(nullptr)
    , file_size_(total_bytes)
    , file_id_(generate_file_id())
    , num_pages_(bytes_to_pages(total_bytes))
    , path_()
    , config_(config)
    , read_only_(read_only)
    , use_mmap_(false)
{
    SCL_CHECK_ARG(fd >= 0, "Invalid file descriptor");

    if (config_.mode == LocalFileMode::Auto) {
        use_mmap_ = (file_size_ > 1024 * 1024 * 1024);
    } else {
        use_mmap_ = (config_.mode == LocalFileMode::MMap);
    }

    if (use_mmap_) {
        setup_mmap();
    }
}

LocalFileBackend::~LocalFileBackend() {
    if (mmap_base_ != nullptr && mmap_base_ != MAP_FAILED) {
        teardown_mmap();
    }

    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

LocalFileBackend::LocalFileBackend(LocalFileBackend&& other) noexcept
    : fd_(std::exchange(other.fd_, -1))
    , mmap_base_(std::exchange(other.mmap_base_, nullptr))
    , file_size_(std::exchange(other.file_size_, 0))
    , file_id_(other.file_id_)
    , num_pages_(std::exchange(other.num_pages_, 0))
    , path_(std::move(other.path_))
    , config_(other.config_)
    , read_only_(other.read_only_)
    , use_mmap_(other.use_mmap_)
{}

LocalFileBackend& LocalFileBackend::operator=(LocalFileBackend&& other) noexcept {
    if (this != &other) {
        if (mmap_base_ != nullptr && mmap_base_ != MAP_FAILED) {
            teardown_mmap();
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }

        fd_ = std::exchange(other.fd_, -1);
        mmap_base_ = std::exchange(other.mmap_base_, nullptr);
        file_size_ = std::exchange(other.file_size_, 0);
        file_id_ = other.file_id_;
        num_pages_ = std::exchange(other.num_pages_, 0);
        path_ = std::move(other.path_);
        config_ = other.config_;
        read_only_ = other.read_only_;
        use_mmap_ = other.use_mmap_;
    }
    return *this;
}

void LocalFileBackend::setup_mmap() {
    int prot = read_only_ ? PROT_READ : (PROT_READ | PROT_WRITE);
    int flags = MAP_SHARED;

#ifdef MAP_POPULATE
    // Pre-fault pages for faster initial access
    if (config_.prefetch_window > 0) {
        flags |= MAP_POPULATE;
    }
#endif

#ifdef MAP_HUGETLB
    if (config_.use_huge_pages) {
        flags |= MAP_HUGETLB;
    }
#endif

    mmap_base_ = ::mmap(nullptr, file_size_, prot, flags, fd_, 0);

    if (mmap_base_ == MAP_FAILED) {
        // Retry without huge pages if that was the issue
        if (config_.use_huge_pages) {
            flags &= ~MAP_HUGETLB;
            mmap_base_ = ::mmap(nullptr, file_size_, prot, flags, fd_, 0);
        }
    }

    SCL_CHECK_IO(mmap_base_ != MAP_FAILED, "mmap failed");

#ifdef MADV_SEQUENTIAL
    // Hint sequential access pattern
    ::madvise(mmap_base_, file_size_, MADV_SEQUENTIAL);
#endif
}

void LocalFileBackend::teardown_mmap() {
    if (mmap_base_ != nullptr && mmap_base_ != MAP_FAILED) {
        ::munmap(mmap_base_, file_size_);
        mmap_base_ = nullptr;
    }
}

void LocalFileBackend::apply_fadvise(int advice) {
#ifdef POSIX_FADV_SEQUENTIAL
    if (fd_ >= 0) {
        ::posix_fadvise(fd_, 0, static_cast<off_t>(file_size_), advice);
    }
#else
    (void)advice;
#endif
}

BackendCapabilities LocalFileBackend::capabilities_impl() const noexcept {
    auto caps = BackendCapabilities::local_file();
    caps.supports_writeback = !read_only_;

    if (use_mmap_) {
        caps.typical_latency = std::chrono::nanoseconds(1000);  // ~1us for page fault
    } else {
        caps.typical_latency = std::chrono::microseconds(100);  // ~100us for pread
    }

    return caps;
}

std::size_t LocalFileBackend::load_page_impl(
    std::size_t page_idx,
    std::byte* dest
) {
    if (page_idx >= num_pages_) {
        return 0;
    }

    const std::size_t offset = page_to_byte_offset(page_idx);
    const std::size_t bytes_to_read = std::min(kPageSize, file_size_ - offset);

    if (use_mmap_) {
        // MMap mode: memcpy from mapped region
        const std::byte* src = static_cast<const std::byte*>(mmap_base_) + offset;
        std::memcpy(dest, src, bytes_to_read);

        // Zero remaining bytes if partial page
        if (bytes_to_read < kPageSize) {
            std::memset(dest + bytes_to_read, 0, kPageSize - bytes_to_read);
        }

#ifdef MADV_WILLNEED
        // Hint next pages for prefetch
        if (config_.prefetch_window > 0 && page_idx + 1 < num_pages_) {
            const std::size_t prefetch_start = page_to_byte_offset(page_idx + 1);
            const std::size_t prefetch_end = std::min(
                prefetch_start + config_.prefetch_window * kPageSize,
                file_size_
            );
            ::madvise(
                static_cast<char*>(mmap_base_) + prefetch_start,
                prefetch_end - prefetch_start,
                MADV_WILLNEED
            );
        }
#endif

        return kPageSize;
    } else {
        // PRead mode: explicit read
        ssize_t total_read = 0;
        std::byte* buf = dest;
        std::size_t remaining = bytes_to_read;
        off_t file_offset = static_cast<off_t>(offset);

        while (remaining > 0) {
            ssize_t n = ::pread(fd_, buf, remaining, file_offset);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) {
                    continue;
                }
                break;
            }
            buf += n;
            remaining -= static_cast<std::size_t>(n);
            file_offset += n;
            total_read += n;
        }

        // Zero remaining bytes
        if (static_cast<std::size_t>(total_read) < kPageSize) {
            std::memset(dest + total_read, 0, kPageSize - total_read);
        }

        return (total_read > 0) ? kPageSize : 0;
    }
}

std::size_t LocalFileBackend::write_page_impl(
    std::size_t page_idx,
    const std::byte* src
) {
    if (read_only_ || page_idx >= num_pages_) {
        return 0;
    }

    const std::size_t offset = page_to_byte_offset(page_idx);
    const std::size_t bytes_to_write = std::min(kPageSize, file_size_ - offset);

    if (use_mmap_) {
        // MMap mode: memcpy to mapped region
        std::byte* dest = static_cast<std::byte*>(mmap_base_) + offset;
        std::memcpy(dest, src, bytes_to_write);

        // msync for durability (optional, can be batched)
        // ::msync(dest, bytes_to_write, MS_ASYNC);

        return kPageSize;
    } else {
        // PWrite mode
        ssize_t total_written = 0;
        const std::byte* buf = src;
        std::size_t remaining = bytes_to_write;
        off_t file_offset = static_cast<off_t>(offset);

        while (remaining > 0) {
            ssize_t n = ::pwrite(fd_, buf, remaining, file_offset);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) {
                    continue;
                }
                break;
            }
            buf += n;
            remaining -= static_cast<std::size_t>(n);
            file_offset += n;
            total_written += n;
        }

        return (total_written > 0) ? kPageSize : 0;
    }
}

std::size_t LocalFileBackend::load_pages_async_impl(
    std::span<const IORequest> requests,
    IOCallback callback
) {
    // For now, implement synchronously
    // TODO: Use io_uring on Linux for true async I/O

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

void LocalFileBackend::prefetch_hint_impl(
    std::span<const std::size_t> pages
) {
    if (pages.empty()) return;

    if (use_mmap_) {
#ifdef MADV_WILLNEED
        for (std::size_t page_idx : pages) {
            if (page_idx >= num_pages_) continue;

            const std::size_t offset = page_to_byte_offset(page_idx);
            const std::size_t len = std::min(kPageSize, file_size_ - offset);

            ::madvise(
                static_cast<char*>(mmap_base_) + offset,
                len,
                MADV_WILLNEED
            );
        }
#endif
    } else {
#ifdef POSIX_FADV_WILLNEED
        for (std::size_t page_idx : pages) {
            if (page_idx >= num_pages_) continue;

            const std::size_t offset = page_to_byte_offset(page_idx);
            const std::size_t len = std::min(kPageSize, file_size_ - offset);

            ::posix_fadvise(
                fd_,
                static_cast<off_t>(offset),
                static_cast<off_t>(len),
                POSIX_FADV_WILLNEED
            );
        }
#endif
    }
}

std::size_t LocalFileBackend::num_pages_impl() const noexcept {
    return num_pages_;
}

std::size_t LocalFileBackend::total_bytes_impl() const noexcept {
    return file_size_;
}

std::size_t LocalFileBackend::file_id_impl() const noexcept {
    return file_id_;
}

BackendType LocalFileBackend::type_impl() const noexcept {
    return BackendType::LocalFile;
}

const std::filesystem::path& LocalFileBackend::path() const noexcept {
    return path_;
}

bool LocalFileBackend::is_mmap_mode() const noexcept {
    return use_mmap_;
}

const std::byte* LocalFileBackend::mmap_base() const noexcept {
    return static_cast<const std::byte*>(mmap_base_);
}

std::byte* LocalFileBackend::mmap_base() noexcept {
    return static_cast<std::byte*>(mmap_base_);
}

// =============================================================================
// Free Functions
// =============================================================================

inline const char* local_file_mode_name(LocalFileMode mode) noexcept {
    switch (mode) {
        case LocalFileMode::MMap:  return "MMap";
        case LocalFileMode::PRead: return "PRead";
        case LocalFileMode::Auto:  return "Auto";
        default:                   return "Unknown";
    }
}

} // namespace scl::mmap::backend
