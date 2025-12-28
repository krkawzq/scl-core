#pragma once

#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/mmap/scheduler.hpp"
#include "scl/mmap/configuration.hpp"

#include <memory>
#include <vector>
#include <algorithm>

// =============================================================================
// MmapArray with offset support for non-aligned page boundaries
// =============================================================================

namespace scl::mmap {

// =============================================================================
// MmapArray: Virtual Array View with Prefetch Scheduling
// =============================================================================

template <typename T>
class MmapArray {
    static_assert(std::is_trivially_copyable_v<T>,
        "MmapArray: Element type must be trivially copyable");

public:
    using value_type = T;
    using ValueType = T;

private:
    std::size_t num_elements_;
    std::size_t start_offset_;  // Byte offset within first page
    std::shared_ptr<PrefetchScheduler> scheduler_;
    std::shared_ptr<PageStore> store_;

    static constexpr std::size_t kElementsPerPage = kPageSize / sizeof(T);
    static constexpr std::size_t kMaxHintPages = 256;

public:
    // =========================================================================
    // Construction
    // =========================================================================

    explicit MmapArray(std::shared_ptr<PrefetchScheduler> scheduler,
                      std::size_t num_elements,
                      LoadCallback loader,
                      WriteCallback writer = nullptr,
                      std::size_t start_offset = 0)
        : num_elements_(num_elements)
        , start_offset_(start_offset)
        , scheduler_(std::move(scheduler))
    {
        if (!scheduler_) {
            throw ValueError("MmapArray: scheduler cannot be null");
        }

        if (num_elements == 0) {
            throw ValueError("MmapArray: num_elements cannot be 0");
        }

        if (start_offset >= kPageSize) {
            throw ValueError("MmapArray: start_offset must be less than page size");
        }

        const std::size_t total_bytes = start_offset + num_elements * sizeof(T);

        store_ = std::make_shared<PageStore>(
            generate_file_id(),
            total_bytes,
            std::move(loader),
            std::move(writer));

        scheduler_->register_store(store_.get());
    }

    explicit MmapArray(std::size_t num_elements,
                      LoadCallback loader,
                      WriteCallback writer = nullptr,
                      std::size_t max_resident = 64,
                      std::size_t num_workers = 0,
                      std::size_t start_offset = 0)
        : num_elements_(num_elements)
        , start_offset_(start_offset)
    {
        if (num_elements == 0) {
            throw ValueError("MmapArray: num_elements cannot be 0");
        }

        if (start_offset >= kPageSize) {
            throw ValueError("MmapArray: start_offset must be less than page size");
        }

        const std::size_t total_bytes = start_offset + num_elements * sizeof(T);
        const std::size_t total_pages = (total_bytes + kPageSize - 1) / kPageSize;

        scheduler_ = make_scheduler(
            total_pages,
            std::min(max_resident, total_pages),
            num_workers
        );

        store_ = std::make_shared<PageStore>(
            generate_file_id(),
            total_bytes,
            std::move(loader),
            std::move(writer));

        scheduler_->register_store(store_.get());
    }

    ~MmapArray() = default;

    MmapArray(const MmapArray&) = delete;
    MmapArray& operator=(const MmapArray&) = delete;
    MmapArray(MmapArray&&) noexcept = default;
    MmapArray& operator=(MmapArray&&) noexcept = default;

    // =========================================================================
    // ArrayLike Interface
    // =========================================================================

    SCL_NODISCARD std::size_t size() const noexcept {
        return num_elements_;
    }

    SCL_NODISCARD bool empty() const noexcept {
        return num_elements_ == 0;
    }

    SCL_NODISCARD std::size_t start_offset() const noexcept {
        return start_offset_;
    }

    SCL_NODISCARD T operator[](std::size_t i) const {
        SCL_ASSERT(i < num_elements_,
                   "MmapArray::operator[] index out of bounds");
        if (i >= num_elements_) {
            return T{};
        }

        constexpr std::size_t kMaxByteOffset = SIZE_MAX / sizeof(T);
        SCL_ASSERT(i <= kMaxByteOffset,
                   "MmapArray::operator[] byte offset would overflow");
        if (i > kMaxByteOffset) {
            return T{};
        }

        const std::size_t byte_offset = start_offset_ + i * sizeof(T);
        const std::size_t page_idx = byte_to_page_idx(byte_offset);
        const std::size_t page_off = byte_to_page_offset(byte_offset);

        PageHandle handle = scheduler_->request(page_idx, store_.get());
        if (!handle) return T{};

        const std::size_t elem_in_page = page_off / sizeof(T);
        SCL_ASSERT(elem_in_page < kElementsPerPage,
                   "MmapArray::operator[] elem_in_page out of bounds");
        if (elem_in_page >= kElementsPerPage) {
            return T{};
        }

        return handle.as<T>()[elem_in_page];
    }

    SCL_NODISCARD T at(std::size_t i) const {
        SCL_CHECK_DIM(i < num_elements_, "MmapArray::at: Index out of bounds");
        return (*this)[i];
    }

    // =========================================================================
    // Optimized Batch Operations
    // =========================================================================

    void read_range(std::size_t start, std::size_t count, T* SCL_RESTRICT out) const {
        if (count == 0) return;

        if (!out) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: MmapArray::read_range null output pointer\n");
#endif
            return;
        }

        if (start >= num_elements_) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: MmapArray::read_range start=%zu >= size=%zu\n",
                        start, num_elements_);
#endif
            return;
        }

        if (count > num_elements_ - start) {
#ifndef NDEBUG
            std::fprintf(stderr, "WARNING: MmapArray::read_range count=%zu truncated to %zu\n",
                        count, num_elements_ - start);
#endif
            count = num_elements_ - start;
        }

        hint_range(start, count);

        T* dest = out;
        std::size_t current = start;
        std::size_t remaining = count;

        // Pre-loop validation (checked once, not per iteration)
        constexpr std::size_t kMaxByteOffset = SIZE_MAX / sizeof(T);
        SCL_ASSERT(start <= kMaxByteOffset,
                   "MmapArray::read_range start would cause byte offset overflow");

        while (remaining > 0) {
            const std::size_t byte_off = start_offset_ + current * sizeof(T);
            const std::size_t page_idx = byte_to_page_idx(byte_off);
            const std::size_t page_off = byte_to_page_offset(byte_off);

            // byte_to_page_offset returns byte_off % kPageSize, always < kPageSize
            SCL_ASSERT(page_off < kPageSize,
                       "MmapArray::read_range page_off invariant violated");

            const std::size_t bytes_in_page = kPageSize - page_off;
            const std::size_t elems_in_page = bytes_in_page / sizeof(T);
            const std::size_t copy_count = std::min(remaining, elems_in_page);

            // elems_in_page >= 1 when kPageSize >= sizeof(T), and remaining > 0
            SCL_ASSERT(copy_count > 0,
                       "MmapArray::read_range copy_count invariant violated");

            PageHandle handle = scheduler_->request(page_idx, store_.get());
            if (handle) {
                const std::size_t elem_offset = page_off / sizeof(T);
                const T* src = handle.as<T>() + elem_offset;
                scl::memory::copy_fast(
                    Array<const T>(src, copy_count),
                    Array<T>(dest, copy_count)
                );
            } else {
                std::memset(dest, 0, copy_count * sizeof(T));
            }

            remaining -= copy_count;
            current += copy_count;
            dest += copy_count;
        }
    }

    void write_range(std::size_t start, std::size_t count, const T* SCL_RESTRICT in) {
        if (count == 0) return;

        if (!in) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: MmapArray::write_range null input pointer\n");
#endif
            return;
        }

        if (start >= num_elements_) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: MmapArray::write_range start=%zu >= size=%zu\n",
                        start, num_elements_);
#endif
            return;
        }

        if (count > num_elements_ - start) {
#ifndef NDEBUG
            std::fprintf(stderr, "WARNING: MmapArray::write_range count=%zu truncated to %zu\n",
                        count, num_elements_ - start);
#endif
            count = num_elements_ - start;
        }

        const T* src = in;
        std::size_t current = start;
        std::size_t remaining = count;

        // Pre-loop validation
        constexpr std::size_t kMaxByteOffset = SIZE_MAX / sizeof(T);
        SCL_ASSERT(start <= kMaxByteOffset,
                   "MmapArray::write_range start would cause byte offset overflow");

        while (remaining > 0) {
            const std::size_t byte_off = start_offset_ + current * sizeof(T);
            const std::size_t page_idx = byte_to_page_idx(byte_off);
            const std::size_t page_off = byte_to_page_offset(byte_off);

            SCL_ASSERT(page_off < kPageSize,
                       "MmapArray::write_range page_off invariant violated");

            const std::size_t bytes_in_page = kPageSize - page_off;
            const std::size_t elems_in_page = bytes_in_page / sizeof(T);
            const std::size_t copy_count = std::min(remaining, elems_in_page);

            SCL_ASSERT(copy_count > 0,
                       "MmapArray::write_range copy_count invariant violated");

            PageHandle handle = scheduler_->request(page_idx, store_.get());
            if (handle) {
                const std::size_t elem_offset = page_off / sizeof(T);
                T* dest = handle.as<T>() + elem_offset;
                scl::memory::copy_fast(
                    Array<const T>(src, copy_count),
                    Array<T>(dest, copy_count)
                );
                handle.mark_dirty();
            }

            remaining -= copy_count;
            current += copy_count;
            src += copy_count;
        }
    }

    void fill_range(std::size_t start, std::size_t count, T value) {
        if (count == 0) return;

        if (start >= num_elements_) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: MmapArray::fill_range start=%zu >= size=%zu\n",
                        start, num_elements_);
#endif
            return;
        }

        if (count > num_elements_ - start) {
#ifndef NDEBUG
            std::fprintf(stderr, "WARNING: MmapArray::fill_range count=%zu truncated to %zu\n",
                        count, num_elements_ - start);
#endif
            count = num_elements_ - start;
        }

        std::size_t current = start;
        std::size_t remaining = count;

        // Pre-loop validation
        constexpr std::size_t kMaxByteOffset = SIZE_MAX / sizeof(T);
        SCL_ASSERT(start <= kMaxByteOffset,
                   "MmapArray::fill_range start would cause byte offset overflow");

        while (remaining > 0) {
            const std::size_t byte_off = start_offset_ + current * sizeof(T);
            const std::size_t page_idx = byte_to_page_idx(byte_off);
            const std::size_t page_off = byte_to_page_offset(byte_off);

            SCL_ASSERT(page_off < kPageSize,
                       "MmapArray::fill_range page_off invariant violated");

            const std::size_t bytes_in_page = kPageSize - page_off;
            const std::size_t elems_in_page = bytes_in_page / sizeof(T);
            const std::size_t fill_count = std::min(remaining, elems_in_page);

            SCL_ASSERT(fill_count > 0,
                       "MmapArray::fill_range fill_count invariant violated");

            PageHandle handle = scheduler_->request(page_idx, store_.get());
            if (handle) {
                const std::size_t elem_offset = page_off / sizeof(T);
                T* dest = handle.as<T>() + elem_offset;
                scl::memory::fill(Array<T>(dest, fill_count), value);
                handle.mark_dirty();
            }

            remaining -= fill_count;
            current += fill_count;
        }
    }

    void copy_to(std::size_t src_start, std::size_t count,
                 MmapArray<T>& dest, std::size_t dest_start) const {
        if (count == 0) return;

        if (src_start >= num_elements_) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: MmapArray::copy_to src_start=%zu >= src_size=%zu\n",
                        src_start, num_elements_);
#endif
            return;
        }

        if (dest_start >= dest.num_elements_) {
#ifndef NDEBUG
            std::fprintf(stderr, "ERROR: MmapArray::copy_to dest_start=%zu >= dest_size=%zu\n",
                        dest_start, dest.num_elements_);
#endif
            return;
        }

        count = std::min({count, num_elements_ - src_start, dest.num_elements_ - dest_start});

        constexpr std::size_t kBufferBytes = 4096;
        constexpr std::size_t kBufferElems = std::max(kBufferBytes / sizeof(T), std::size_t{1});

        auto buffer = std::make_unique<T[]>(kBufferElems);

        std::size_t remaining = count;
        std::size_t src_pos = src_start;
        std::size_t dest_pos = dest_start;

        while (remaining > 0) {
            const std::size_t chunk = std::min(remaining, kBufferElems);
            if (chunk == 0) break;

            read_range(src_pos, chunk, buffer.get());
            dest.write_range(dest_pos, chunk, buffer.get());

            remaining -= chunk;
            src_pos += chunk;
            dest_pos += chunk;
        }
    }

    // =========================================================================
    // Prefetch Control
    // =========================================================================

    void hint_range(std::size_t start, std::size_t count) const {
        if (count == 0 || start >= num_elements_) return;

        count = std::min(count, num_elements_ - start);

        const std::size_t start_page = element_to_page(start);
        const std::size_t end_page = element_to_page(start + count - 1);
        const std::size_t max_page = store_->num_pages();

        if (start_page >= max_page) return;

        const std::size_t total_pages = std::min(end_page - start_page + 1, max_page - start_page);
        const std::size_t hint_pages = std::min(total_pages, kMaxHintPages);

        std::vector<std::size_t> pages;
        pages.reserve(hint_pages);
        for (std::size_t p = start_page; p < start_page + hint_pages; ++p) {
            pages.push_back(p);
        }

        scheduler_->hint_access_pattern(pages);
    }

    void hint_priority(std::size_t elem_idx) const {
        if (elem_idx >= num_elements_) return;

        const std::size_t page_idx = element_to_page(elem_idx);
        if (page_idx < store_->num_pages()) {
            scheduler_->hint_priority_boost(page_idx);
        }
    }

    void prefetch_sequential(std::size_t start, std::size_t window_size) {
        if (start >= num_elements_) return;

        const std::size_t count = std::min(window_size, num_elements_ - start);
        hint_range(start, count);
    }

    void prefetch_strided(std::size_t start, std::size_t count, std::size_t stride) {
        if (count == 0 || start >= num_elements_ || stride == 0) return;

        const std::size_t max_page = store_->num_pages();

        std::vector<std::size_t> pages;
        pages.reserve(std::min(count, kMaxHintPages));

        for (std::size_t i = 0; i < count && pages.size() < kMaxHintPages; ++i) {
            if (i > SIZE_MAX / stride) break;
            const std::size_t offset = i * stride;
            if (offset > SIZE_MAX - start) break;

            const std::size_t elem_idx = start + offset;
            if (elem_idx >= num_elements_) break;

            const std::size_t page = element_to_page(elem_idx);
            if (page >= max_page) break;

            pages.push_back(page);
        }

        if (pages.empty()) return;

        std::sort(pages.begin(), pages.end());
        pages.erase(std::unique(pages.begin(), pages.end()), pages.end());

        scheduler_->hint_access_pattern(pages);
    }

    // =========================================================================
    // Compute Hooks Integration
    // =========================================================================

    void begin_computation(std::size_t total_rows) {
        scheduler_->on_computation_begin(total_rows);
    }

    void end_computation() {
        scheduler_->on_computation_end();
    }

    void begin_row(std::size_t row) {
        scheduler_->on_row_begin(row);
    }

    void end_row(std::size_t row) {
        scheduler_->on_row_end(row);
    }

    void sync_barrier() {
        scheduler_->sync_barrier();
    }

    bool sync_fence(std::size_t up_to_row,
                   std::chrono::milliseconds timeout = std::chrono::milliseconds{5000}) {
        return scheduler_->sync_fence(up_to_row, timeout);
    }

    void flush() {
        scheduler_->sync_barrier();
    }

    // =========================================================================
    // Policy Control
    // =========================================================================

    template <typename Policy, typename... Args>
    void set_policy(Args&&... args) {
        scheduler_->template set_policy<Policy>(std::forward<Args>(args)...);
    }

    void set_lookahead(std::size_t lookahead) {
        scheduler_->set_lookahead(lookahead);
    }

    void set_batch_size(std::size_t batch) {
        scheduler_->set_batch_size(batch);
    }

    void set_block_mode(BlockMode mode) {
        scheduler_->set_block_mode(mode);
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    SCL_NODISCARD std::size_t resident_pages() const noexcept {
        return scheduler_->resident_count();
    }

    SCL_NODISCARD std::size_t pending_pages() const noexcept {
        return scheduler_->pending_count();
    }

    SCL_NODISCARD std::size_t total_pages() const noexcept {
        return store_->num_pages();
    }

    SCL_NODISCARD const SchedulerStats& stats() const noexcept {
        return scheduler_->stats();
    }

    SCL_NODISCARD double hit_rate() const noexcept {
        return scheduler_->stats().hit_rate();
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    SCL_NODISCARD PrefetchScheduler* scheduler() noexcept {
        return scheduler_.get();
    }

    SCL_NODISCARD const PrefetchScheduler* scheduler() const noexcept {
        return scheduler_.get();
    }

    SCL_NODISCARD PageStore* store() noexcept {
        return store_.get();
    }

    SCL_NODISCARD const PageStore* store() const noexcept {
        return store_.get();
    }

private:
    // =========================================================================
    // Internal Helpers
    // =========================================================================

    std::size_t element_to_page(std::size_t elem_idx) const noexcept {
        const std::size_t byte_offset = start_offset_ + elem_idx * sizeof(T);
        return byte_to_page_idx(byte_offset);
    }

    std::size_t element_page_offset(std::size_t elem_idx) const noexcept {
        const std::size_t byte_offset = start_offset_ + elem_idx * sizeof(T);
        return byte_to_page_offset(byte_offset);
    }
};

// =============================================================================
// Scoped Computation Guard
// =============================================================================

template <typename T>
class ScopedComputation {
    MmapArray<T>* array_;

public:
    explicit ScopedComputation(MmapArray<T>& array, std::size_t total_rows)
        : array_(&array)
    {
        array_->begin_computation(total_rows);
    }

    ~ScopedComputation() {
        if (array_) {
            array_->end_computation();
        }
    }

    ScopedComputation(const ScopedComputation&) = delete;
    ScopedComputation& operator=(const ScopedComputation&) = delete;

    ScopedComputation(ScopedComputation&& other) noexcept
        : array_(other.array_)
    {
        other.array_ = nullptr;
    }

    ScopedComputation& operator=(ScopedComputation&& other) noexcept {
        if (this != &other) {
            if (array_) array_->end_computation();
            array_ = other.array_;
            other.array_ = nullptr;
        }
        return *this;
    }
};

// =============================================================================
// Scoped Row Guard
// =============================================================================

template <typename T>
class ScopedRow {
    MmapArray<T>* array_;
    std::size_t row_;

public:
    explicit ScopedRow(MmapArray<T>& array, std::size_t row)
        : array_(&array), row_(row)
    {
        array_->begin_row(row);
    }

    ~ScopedRow() {
        if (array_) {
            array_->end_row(row_);
        }
    }

    ScopedRow(const ScopedRow&) = delete;
    ScopedRow& operator=(const ScopedRow&) = delete;

    ScopedRow(ScopedRow&& other) noexcept
        : array_(other.array_), row_(other.row_)
    {
        other.array_ = nullptr;
    }

    ScopedRow& operator=(ScopedRow&& other) noexcept {
        if (this != &other) {
            if (array_) array_->end_row(row_);
            array_ = other.array_;
            row_ = other.row_;
            other.array_ = nullptr;
        }
        return *this;
    }

    SCL_NODISCARD std::size_t row() const noexcept { return row_; }
};

// =============================================================================
// Type Aliases
// =============================================================================

using MmapArrayReal = MmapArray<Real>;
using MmapArrayF32 = MmapArray<float>;
using MmapArrayF64 = MmapArray<double>;
using MmapArrayIndex = MmapArray<Index>;
using MmapArrayI32 = MmapArray<std::int32_t>;
using MmapArrayI64 = MmapArray<std::int64_t>;

// =============================================================================
// Factory Functions
// =============================================================================

template <typename T>
SCL_NODISCARD std::shared_ptr<MmapArray<T>> make_mmap_array(
    std::size_t num_elements,
    LoadCallback loader,
    WriteCallback writer = nullptr,
    std::size_t max_resident = 64,
    std::size_t num_workers = 0,
    std::size_t start_offset = 0
) {
    return std::make_shared<MmapArray<T>>(
        num_elements, std::move(loader), std::move(writer),
        max_resident, num_workers, start_offset
    );
}

template <typename T>
SCL_NODISCARD std::shared_ptr<MmapArray<T>> make_mmap_array(
    std::shared_ptr<PrefetchScheduler> scheduler,
    std::size_t num_elements,
    LoadCallback loader,
    WriteCallback writer = nullptr,
    std::size_t start_offset = 0
) {
    return std::make_shared<MmapArray<T>>(
        std::move(scheduler), num_elements,
        std::move(loader), std::move(writer),
        start_offset
    );
}

} // namespace scl::mmap
