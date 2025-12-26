#pragma once

// =============================================================================
/// @file array.hpp
/// @brief Virtual Array with Automatic Page Management
///
/// VirtualArray provides a contiguous array abstraction over paged storage.
/// Data is loaded on-demand and cached in memory according to the configured
/// scheduling policy.
///
/// Performance Optimizations:
///
/// 1. Batch Operations: read_range/write_range process multiple pages efficiently
/// 2. Prefetch Support: Hardware prefetch hints for sequential access
/// 3. Zero-Copy: Direct pointer access within pages when possible
/// 4. Minimal Overhead: Inline hot paths, constexpr computations
///
/// Memory Model:
///
/// Virtual Address Space:
/// +--------+--------+--------+--------+
/// | Page 0 | Page 1 | Page 2 | Page 3 | ...
/// +--------+--------+--------+--------+
/// ^        ^        ^        ^
/// |        |        |        |
/// Elements are mapped across page boundaries transparently
// =============================================================================

#include "scl/core/macros.hpp"
#include "scl/mmap/configuration.hpp"
#include "scl/mmap/table.hpp"
#include "scl/mmap/scheduler.hpp"

#include <cstddef>
#include <cstring>
#include <algorithm>
#include <memory>
#include <type_traits>

namespace scl::mmap {

// =============================================================================
// VirtualArray<T>: Paged Virtual Array
// =============================================================================

/// @brief Virtual array with automatic page management
///
/// Features:
/// - Transparent paging (data loaded on access)
/// - Configurable scheduling (Standard or Sequential)
/// - Support for trivially copyable types
/// - Efficient batch read/write operations
///
/// Template Requirements:
/// - T must be trivially copyable (for memcpy safety)
/// - sizeof(T) must divide kPageSize evenly
///
/// Usage:
///
///   // Create array with custom loader
///   VirtualArray<float> arr(1000000, [](size_t page, std::byte* dest) {
///       // Load page data from storage
///   });
///
///   // Random access (may trigger page load)
///   float val = arr[500000];
///
///   // Batch read (efficient for large ranges)
///   std::vector<float> buffer(1000);
///   arr.read_range(0, 1000, buffer.data());
template <typename T>
class VirtualArray {
    static_assert(std::is_trivially_copyable_v<T>,
        "VirtualArray element type must be trivially copyable");
    static_assert(kPageSize % sizeof(T) == 0,
        "Page size must be a multiple of element size");

public:
    using value_type = T;
    using size_type = std::size_t;

private:
    // Core components
    std::unique_ptr<PageTable> table_;
    std::unique_ptr<PagePool> pool_;
    std::unique_ptr<Scheduler> scheduler_;
    
    // Dimensions
    std::size_t num_elements_;
    std::size_t num_bytes_;

    // Compile-time constants
    static constexpr std::size_t kElementsPerPage = kPageSize / sizeof(T);
    static constexpr std::size_t kElementShift = detail::constexpr_log2(sizeof(T));

public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /// @brief Construct virtual array with specified scheduler type
    /// @tparam SchedulerT Scheduler type (StandardScheduler or SequentialScheduler)
    /// @param num_elements Number of elements in array
    /// @param load_func Page load callback
    /// @param config Memory pool configuration
    template <typename SchedulerT = StandardScheduler>
    VirtualArray(std::size_t num_elements, LoadFunc load_func,
                 const MmapConfig& config = MmapConfig{})
        : num_elements_(num_elements)
        , num_bytes_(num_elements * sizeof(T))
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
    
    // Non-copyable
    VirtualArray(const VirtualArray&) = delete;
    VirtualArray& operator=(const VirtualArray&) = delete;
    
    // Movable
    VirtualArray(VirtualArray&&) noexcept = default;
    VirtualArray& operator=(VirtualArray&&) noexcept = default;

    // -------------------------------------------------------------------------
    // Properties
    // -------------------------------------------------------------------------

    SCL_NODISCARD SCL_FORCE_INLINE std::size_t size() const noexcept { 
        return num_elements_; 
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE std::size_t size_bytes() const noexcept { 
        return num_bytes_; 
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE std::size_t resident_pages() const noexcept { 
        return pool_->capacity() - pool_->free_count(); 
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE std::size_t total_pages() const noexcept { 
        return table_->num_pages(); 
    }

    // -------------------------------------------------------------------------
    // Element Access
    // -------------------------------------------------------------------------

    /// @brief Access single element (may trigger page load)
    SCL_NODISCARD T operator[](std::size_t i) const {
        const std::size_t byte_offset = i * sizeof(T);
        const std::size_t page_idx = byte_to_page_idx(byte_offset);
        const std::size_t page_off = byte_to_page_offset(byte_offset);
        
        PageHandle handle = scheduler_->request(page_idx);
        if (SCL_UNLIKELY(!handle)) {
            return T{};
        }
        
        return *reinterpret_cast<const T*>(handle.data() + page_off);
    }

    // -------------------------------------------------------------------------
    // Batch Read (Optimized)
    // -------------------------------------------------------------------------

    /// @brief Read contiguous range of elements
    ///
    /// Optimization Strategy:
    /// 1. Process full pages with single memcpy
    /// 2. Use hardware prefetch for upcoming pages
    /// 3. Minimize page handle overhead
    ///
    /// @param start Starting element index
    /// @param count Number of elements to read
    /// @param out Destination buffer (must have space for count elements)
    void read_range(std::size_t start, std::size_t count, T* SCL_RESTRICT out) const {
        if (SCL_UNLIKELY(count == 0)) return;

        const T* dest = out;
        std::size_t current_idx = start;
        std::size_t remaining = count;

        // Process pages
        while (remaining > 0) {
            const std::size_t byte_offset = current_idx * sizeof(T);
            const std::size_t page_idx = byte_to_page_idx(byte_offset);
            const std::size_t page_off = byte_to_page_offset(byte_offset);
            
            // Elements available in this page
            const std::size_t bytes_in_page = kPageSize - page_off;
            const std::size_t elements_in_page = bytes_in_page / sizeof(T);
            const std::size_t copy_count = std::min(remaining, elements_in_page);

            // Prefetch next page (if exists)
            if (remaining > elements_in_page) {
                const std::size_t next_page = page_idx + 1;
                if (next_page < table_->num_pages()) {
                    SCL_PREFETCH_READ(table_->get(next_page), 1);
                }
            }

            // Load and copy
            PageHandle handle = scheduler_->request(page_idx);
            if (SCL_LIKELY(handle)) {
                std::memcpy(const_cast<T*>(dest), 
                           handle.data() + page_off, 
                           copy_count * sizeof(T));
            } else {
                std::memset(const_cast<T*>(dest), 0, copy_count * sizeof(T));
            }

            remaining -= copy_count;
            current_idx += copy_count;
            dest += copy_count;
        }
    }

    // -------------------------------------------------------------------------
    // Batch Write (Optimized)
    // -------------------------------------------------------------------------

    /// @brief Write contiguous range of elements
    /// @param start Starting element index
    /// @param count Number of elements to write
    /// @param in Source buffer
    void write_range(std::size_t start, std::size_t count, const T* SCL_RESTRICT in) {
        if (SCL_UNLIKELY(count == 0)) return;

        const T* src = in;
        std::size_t current_idx = start;
        std::size_t remaining = count;

        while (remaining > 0) {
            const std::size_t byte_offset = current_idx * sizeof(T);
            const std::size_t page_idx = byte_to_page_idx(byte_offset);
            const std::size_t page_off = byte_to_page_offset(byte_offset);
            
            const std::size_t bytes_in_page = kPageSize - page_off;
            const std::size_t elements_in_page = bytes_in_page / sizeof(T);
            const std::size_t copy_count = std::min(remaining, elements_in_page);

            PageHandle handle = scheduler_->request(page_idx);
            if (SCL_LIKELY(handle)) {
                std::memcpy(handle.data() + page_off, src, copy_count * sizeof(T));
                // Mark page as dirty
                table_->entry(page_idx).set_dirty();
            }

            remaining -= copy_count;
            current_idx += copy_count;
            src += copy_count;
        }
    }

    // -------------------------------------------------------------------------
    // Prefetch Hints
    // -------------------------------------------------------------------------

    /// @brief Prefetch element range (hint for upcoming access)
    /// @param start Starting element index
    /// @param count Number of elements
    void prefetch(std::size_t start, std::size_t count) {
        if (SCL_UNLIKELY(count == 0)) return;
        
        const std::size_t start_page = byte_to_page_idx(start * sizeof(T));
        const std::size_t end_byte = (start + count - 1) * sizeof(T);
        const std::size_t end_page = byte_to_page_idx(end_byte);
        
        scheduler_->prefetch(start_page, end_page - start_page + 1);
    }

    // -------------------------------------------------------------------------
    // Direct Page Access (Advanced)
    // -------------------------------------------------------------------------

    /// @brief Get direct pointer to element within page
    ///
    /// Warning: Returned pointer is only valid while PageHandle is alive.
    /// Use for tight loops where operator[] overhead is unacceptable.
    ///
    /// @param i Element index
    /// @param handle Output page handle (keeps page pinned)
    /// @return Pointer to element (nullptr if page load fails)
    const T* get_ptr(std::size_t i, PageHandle& handle) const {
        const std::size_t byte_offset = i * sizeof(T);
        const std::size_t page_idx = byte_to_page_idx(byte_offset);
        const std::size_t page_off = byte_to_page_offset(byte_offset);
        
        handle = scheduler_->request(page_idx);
        if (SCL_UNLIKELY(!handle)) {
            return nullptr;
        }
        
        return reinterpret_cast<const T*>(handle.data() + page_off);
    }

    /// @brief Get mutable pointer to element within page
    T* get_ptr_mut(std::size_t i, PageHandle& handle) {
        const std::size_t byte_offset = i * sizeof(T);
        const std::size_t page_idx = byte_to_page_idx(byte_offset);
        const std::size_t page_off = byte_to_page_offset(byte_offset);
        
        handle = scheduler_->request(page_idx);
        if (SCL_UNLIKELY(!handle)) {
            return nullptr;
        }
        
        table_->entry(page_idx).set_dirty();
        return reinterpret_cast<T*>(handle.data() + page_off);
    }
};

} // namespace scl::mmap
