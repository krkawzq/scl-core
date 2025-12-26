#pragma once

#include "scl/core/macros.hpp"
#include "scl/mmap/configuration.hpp"
#include "scl/mmap/table.hpp"
#include "scl/mmap/scheduler.hpp"

#include <cstddef>
#include <cstring>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <stdexcept>

namespace scl::mmap {

// =============================================================================
// VirtualArray<T>: 自动分页管理的虚拟数组
// =============================================================================

template <typename T>
class VirtualArray {
    static_assert(SCL_MMAP_PAGE_SIZE % sizeof(T) == 0,
        "Page size must be a multiple of element size");
    static_assert(std::is_trivially_copyable_v<T>,
        "VirtualArray element type must be trivially copyable");

    std::unique_ptr<PageTable> table_;
    std::unique_ptr<PagePool> pool_;
    std::unique_ptr<Scheduler> scheduler_;
    std::size_t num_elements_;
    std::size_t num_bytes_;

public:
    using value_type = T;
    using size_type = std::size_t;

    template <typename SchedulerT = StandardScheduler>
    VirtualArray(std::size_t num_elements, LoadFunc load_func,
                 const MmapConfig& config = MmapConfig{})
        : num_elements_(num_elements), num_bytes_(num_elements * sizeof(T))
    {
        table_ = std::make_unique<PageTable>(num_bytes_);
        pool_ = std::make_unique<PagePool>(config.max_resident_pages);

        if constexpr (std::is_same_v<SchedulerT, SequentialScheduler>) {
            scheduler_ = std::make_unique<SchedulerT>(
                table_.get(), pool_.get(), std::move(load_func), config.prefetch_depth);
        } else {
            scheduler_ = std::make_unique<SchedulerT>(
                table_.get(), pool_.get(), std::move(load_func));
        }
    }

    ~VirtualArray() = default;
    VirtualArray(const VirtualArray&) = delete;
    VirtualArray& operator=(const VirtualArray&) = delete;
    VirtualArray(VirtualArray&&) noexcept = default;
    VirtualArray& operator=(VirtualArray&&) noexcept = default;

    std::size_t size() const { return num_elements_; }
    std::size_t size_bytes() const { return num_bytes_; }

    T operator[](std::size_t i) const {
        std::size_t byte_offset = i * sizeof(T);
        std::size_t page_idx = table_->to_page_idx(byte_offset);
        std::size_t page_off = table_->to_page_offset(byte_offset);
        PageHandle handle = scheduler_->request(page_idx);
        if (SCL_UNLIKELY(!handle)) return T{};
        return *reinterpret_cast<const T*>(handle.data() + page_off);
    }

    void read_range(std::size_t start, std::size_t count, T* out) const {
        if (count == 0) return;
        std::size_t current_idx = start;
        std::size_t remaining = count;
        T* dest = out;

        while (remaining > 0) {
            std::size_t byte_offset = current_idx * sizeof(T);
            std::size_t page_idx = table_->to_page_idx(byte_offset);
            std::size_t page_off = table_->to_page_offset(byte_offset);
            std::size_t bytes_in_page = SCL_MMAP_PAGE_SIZE - page_off;
            std::size_t elements_in_page = bytes_in_page / sizeof(T);
            std::size_t copy_count = std::min(remaining, elements_in_page);

            PageHandle handle = scheduler_->request(page_idx);
            if (SCL_LIKELY(handle)) {
                std::memcpy(dest, handle.data() + page_off, copy_count * sizeof(T));
            } else {
                std::memset(dest, 0, copy_count * sizeof(T));
            }
            remaining -= copy_count;
            current_idx += copy_count;
            dest += copy_count;
        }
    }

    void write_range(std::size_t start, std::size_t count, const T* in) {
        if (count == 0) return;
        std::size_t current_idx = start;
        std::size_t remaining = count;
        const T* src = in;

        while (remaining > 0) {
            std::size_t byte_offset = current_idx * sizeof(T);
            std::size_t page_idx = table_->to_page_idx(byte_offset);
            std::size_t page_off = table_->to_page_offset(byte_offset);
            std::size_t bytes_in_page = SCL_MMAP_PAGE_SIZE - page_off;
            std::size_t elements_in_page = bytes_in_page / sizeof(T);
            std::size_t copy_count = std::min(remaining, elements_in_page);

            PageHandle handle = scheduler_->request(page_idx);
            if (SCL_LIKELY(handle)) {
                std::memcpy(handle.data() + page_off, src, copy_count * sizeof(T));
            }
            remaining -= copy_count;
            current_idx += copy_count;
            src += copy_count;
        }
    }

    void prefetch(std::size_t start, std::size_t count) {
        if (count == 0) return;
        std::size_t start_page = table_->to_page_idx(start * sizeof(T));
        std::size_t end_page = table_->to_page_idx((start + count - 1) * sizeof(T));
        scheduler_->prefetch(start_page, end_page - start_page + 1);
    }

    std::size_t resident_pages() const { return pool_->capacity() - pool_->free_count(); }
    std::size_t total_pages() const { return table_->num_pages(); }
};

} // namespace scl::mmap
