#pragma once

#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"
#include "scl/mmap/page.hpp"
#include "scl/mmap/cache.hpp"
#include "scl/mmap/configuration.hpp"

#include <memory>
#include <vector>
#include <cstring>
#include <atomic>
#include <optional>

// =============================================================================
// FILE: scl/mmap/sparse.hpp
// BRIEF: Memory-mapped sparse matrix with zero-copy views
// =============================================================================

namespace scl::mmap {

// =============================================================================
// Forward Declarations
// =============================================================================

template <typename T, bool IsCSR>
class MmapSparse;

template <typename T>
class MmapSparseSlice;

template <typename T>
class MmapSparseIterator;

// =============================================================================
// PagedBuffer: Multi-page view with lifetime management
// =============================================================================

template <typename T>
class PagedBuffer {
    static_assert(std::is_trivially_copyable_v<T>);
    
public:
    static constexpr std::size_t kMaxHandles = 8;
    
private:
    std::array<PageHandle, kMaxHandles> handles_;
    std::size_t num_handles_{0};
    
    const T* base_ptr_{nullptr};
    std::size_t length_{0};
    std::size_t first_page_offset_{0};
    
    // For cross-page iteration
    struct PageInfo {
        const T* data;
        std::size_t count;
    };
    std::array<PageInfo, kMaxHandles> pages_;

public:
    PagedBuffer() noexcept = default;
    
    PagedBuffer(PageHandle handle, const T* ptr, std::size_t len) noexcept
        : num_handles_(1)
        , base_ptr_(ptr)
        , length_(len)
    {
        handles_[0] = std::move(handle);
        pages_[0] = {ptr, len};
    }
    
    void add_page(PageHandle handle, const T* ptr, std::size_t count) {
        if (num_handles_ >= kMaxHandles) {
            throw RuntimeError("PagedBuffer: too many pages");
        }
        handles_[num_handles_] = std::move(handle);
        pages_[num_handles_] = {ptr, count};
        length_ += count;
        ++num_handles_;
    }
    
    SCL_NODISCARD bool valid() const noexcept { 
        return num_handles_ > 0 && base_ptr_ != nullptr; 
    }
    SCL_NODISCARD explicit operator bool() const noexcept { return valid(); }
    
    SCL_NODISCARD std::size_t size() const noexcept { return length_; }
    SCL_NODISCARD bool empty() const noexcept { return length_ == 0; }
    SCL_NODISCARD std::size_t num_pages() const noexcept { return num_handles_; }
    
    // Single-page fast access (only valid if fits in one page)
    SCL_NODISCARD bool is_contiguous() const noexcept { 
        return num_handles_ == 1; 
    }
    
    SCL_NODISCARD const T* contiguous_data() const noexcept {
        return is_contiguous() ? base_ptr_ : nullptr;
    }
    
    // Element access (handles cross-page)
    SCL_NODISCARD T operator[](std::size_t i) const noexcept {
        SCL_ASSERT(i < length_, "PagedBuffer: index out of bounds");
        
        if (is_contiguous()) {
            return base_ptr_[i];
        }
        
        std::size_t offset = 0;
        for (std::size_t p = 0; p < num_handles_; ++p) {
            if (i < offset + pages_[p].count) {
                return pages_[p].data[i - offset];
            }
            offset += pages_[p].count;
        }
        
        return T{};
    }
    
    // Copy to contiguous buffer
    void copy_to(T* dest, std::size_t count) const {
        if (!dest || count == 0) return;
        count = std::min(count, length_);
        
        if (is_contiguous()) {
            std::memcpy(dest, base_ptr_, count * sizeof(T));
            return;
        }
        
        std::size_t copied = 0;
        for (std::size_t p = 0; p < num_handles_ && copied < count; ++p) {
            std::size_t to_copy = std::min(pages_[p].count, count - copied);
            std::memcpy(dest + copied, pages_[p].data, to_copy * sizeof(T));
            copied += to_copy;
        }
    }
    
    // Iterator for cross-page traversal
    class Iterator {
        const PagedBuffer* buf_;
        std::size_t page_idx_;
        std::size_t elem_idx_;
        std::size_t global_idx_;
        
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;
        
        Iterator(const PagedBuffer* buf, std::size_t global) noexcept
            : buf_(buf), global_idx_(global)
        {
            if (buf_ && global < buf_->length_) {
                locate(global);
            } else {
                page_idx_ = buf_ ? buf_->num_handles_ : 0;
                elem_idx_ = 0;
            }
        }
        
        reference operator*() const noexcept {
            return buf_->pages_[page_idx_].data[elem_idx_];
        }
        
        pointer operator->() const noexcept {
            return &buf_->pages_[page_idx_].data[elem_idx_];
        }
        
        Iterator& operator++() noexcept {
            ++global_idx_;
            ++elem_idx_;
            if (elem_idx_ >= buf_->pages_[page_idx_].count) {
                ++page_idx_;
                elem_idx_ = 0;
            }
            return *this;
        }
        
        bool operator==(const Iterator& other) const noexcept {
            return global_idx_ == other.global_idx_;
        }
        
        bool operator!=(const Iterator& other) const noexcept {
            return global_idx_ != other.global_idx_;
        }
        
    private:
        void locate(std::size_t global) {
            std::size_t offset = 0;
            for (page_idx_ = 0; page_idx_ < buf_->num_handles_; ++page_idx_) {
                if (global < offset + buf_->pages_[page_idx_].count) {
                    elem_idx_ = global - offset;
                    return;
                }
                offset += buf_->pages_[page_idx_].count;
            }
            elem_idx_ = 0;
        }
    };
    
    Iterator begin() const noexcept { return Iterator(this, 0); }
    Iterator end() const noexcept { return Iterator(this, length_); }
};

// =============================================================================
// MmapSparseSlice: Zero-copy view of a single row/column
// =============================================================================

template <typename T>
class MmapSparseSlice {
public:
    using value_type = T;
    using ValueType = T;
    
private:
    PagedBuffer<T> values_;
    PagedBuffer<Index> indices_;
    Index row_or_col_idx_;
    
public:
    MmapSparseSlice() noexcept : row_or_col_idx_(-1) {}
    
    MmapSparseSlice(PagedBuffer<T> vals, PagedBuffer<Index> idx, Index rc_idx) noexcept
        : values_(std::move(vals))
        , indices_(std::move(idx))
        , row_or_col_idx_(rc_idx)
    {}
    
    SCL_NODISCARD bool valid() const noexcept { 
        return values_.valid() && indices_.valid(); 
    }
    SCL_NODISCARD explicit operator bool() const noexcept { return valid(); }
    
    SCL_NODISCARD std::size_t size() const noexcept { return values_.size(); }
    SCL_NODISCARD std::size_t nnz() const noexcept { return values_.size(); }
    SCL_NODISCARD bool empty() const noexcept { return values_.empty(); }
    SCL_NODISCARD Index index() const noexcept { return row_or_col_idx_; }
    
    // =========================================================================
    // Fast contiguous access (when data fits in single page)
    // =========================================================================
    
    SCL_NODISCARD bool is_contiguous() const noexcept {
        return values_.is_contiguous() && indices_.is_contiguous();
    }
    
    SCL_NODISCARD const T* values_data() const noexcept {
        return values_.contiguous_data();
    }
    
    SCL_NODISCARD const Index* indices_data() const noexcept {
        return indices_.contiguous_data();
    }
    
    // =========================================================================
    // Element access (handles cross-page transparently)
    // =========================================================================
    
    SCL_NODISCARD T value(std::size_t i) const noexcept {
        return values_[i];
    }
    
    SCL_NODISCARD Index col_index(std::size_t i) const noexcept {
        return indices_[i];
    }
    
    SCL_NODISCARD std::pair<Index, T> operator[](std::size_t i) const noexcept {
        return {indices_[i], values_[i]};
    }
    
    // =========================================================================
    // Bulk copy (for algorithms needing contiguous data)
    // =========================================================================
    
    void copy_values_to(T* dest, std::size_t count) const {
        values_.copy_to(dest, count);
    }
    
    void copy_indices_to(Index* dest, std::size_t count) const {
        indices_.copy_to(dest, count);
    }
    
    // =========================================================================
    // Iteration
    // =========================================================================
    
    class Iterator {
        typename PagedBuffer<T>::Iterator val_it_;
        typename PagedBuffer<Index>::Iterator idx_it_;
        
    public:
        Iterator(typename PagedBuffer<T>::Iterator v, 
                typename PagedBuffer<Index>::Iterator i) noexcept
            : val_it_(v), idx_it_(i) {}
        
        std::pair<Index, T> operator*() const noexcept {
            return {*idx_it_, *val_it_};
        }
        
        Iterator& operator++() noexcept {
            ++val_it_;
            ++idx_it_;
            return *this;
        }
        
        bool operator!=(const Iterator& other) const noexcept {
            return val_it_ != other.val_it_;
        }
    };
    
    Iterator begin() const noexcept { 
        return Iterator(values_.begin(), indices_.begin()); 
    }
    
    Iterator end() const noexcept { 
        return Iterator(values_.end(), indices_.end()); 
    }
    
    // =========================================================================
    // Raw buffer access (for advanced use)
    // =========================================================================
    
    SCL_NODISCARD const PagedBuffer<T>& values_buffer() const noexcept {
        return values_;
    }
    
    SCL_NODISCARD const PagedBuffer<Index>& indices_buffer() const noexcept {
        return indices_;
    }
};

// =============================================================================
// IndptrCache: Thread-safe indptr management
// =============================================================================

class IndptrCache {
private:
    mutable std::vector<Index> data_;
    mutable std::atomic<bool> loaded_{false};
    mutable std::mutex load_mutex_;
    
    std::shared_ptr<PageStore> store_;
    CacheManager* cache_;
    Index primary_dim_;
    
    static constexpr std::size_t kIndptrPerPage = kPageSize / sizeof(Index);

public:
    IndptrCache(std::shared_ptr<PageStore> store, CacheManager* cache, Index dim)
        : store_(std::move(store))
        , cache_(cache)
        , primary_dim_(dim)
    {}
    
    SCL_NODISCARD Index operator[](Index i) const {
        ensure_loaded();
        if (i < 0 || static_cast<std::size_t>(i) >= data_.size()) {
            return 0;
        }
        return data_[i];
    }
    
    SCL_NODISCARD std::pair<Index, Index> range(Index i) const {
        ensure_loaded();
        if (i < 0 || static_cast<std::size_t>(i) >= data_.size() - 1) {
            return {0, 0};
        }
        return {data_[i], data_[i + 1]};
    }
    
    SCL_NODISCARD Index length(Index i) const {
        auto [start, end] = range(i);
        return (end > start) ? (end - start) : 0;
    }
    
    SCL_NODISCARD const Index* data() const {
        ensure_loaded();
        return data_.data();
    }
    
    SCL_NODISCARD std::size_t size() const {
        ensure_loaded();
        return data_.size();
    }
    
    SCL_NODISCARD bool is_loaded() const noexcept {
        return loaded_.load(std::memory_order_acquire);
    }

private:
    void ensure_loaded() const {
        if (loaded_.load(std::memory_order_acquire)) {
            return;
        }
        
        std::lock_guard<std::mutex> guard(load_mutex_);
        if (loaded_.load(std::memory_order_relaxed)) {
            return;
        }
        
        const std::size_t indptr_size = static_cast<std::size_t>(primary_dim_ + 1);
        data_.resize(indptr_size);
        
        const std::size_t num_pages = store_->num_pages();
        std::size_t loaded = 0;
        
        for (std::size_t page_idx = 0; page_idx < num_pages && loaded < indptr_size; ++page_idx) {
            PageHandle handle = cache_->request(page_idx, store_.get());
            if (!handle) {
                throw RuntimeError("IndptrCache: failed to load page");
            }
            
            const Index* page_data = handle.as<Index>();
            const std::size_t elems = std::min(kIndptrPerPage, indptr_size - loaded);
            
            std::memcpy(data_.data() + loaded, page_data, elems * sizeof(Index));
            loaded += elems;
        }
        
        if (loaded != indptr_size) {
            throw RuntimeError("IndptrCache: incomplete load");
        }
        
        loaded_.store(true, std::memory_order_release);
    }
};

// =============================================================================
// MmapSparse: Paged Sparse Matrix
// =============================================================================

template <typename T, bool IsCSR>
class MmapSparse {
    static_assert(std::is_trivially_copyable_v<T>,
        "MmapSparse: T must be trivially copyable");

public:
    using ValueType = T;
    using value_type = T;
    using SliceType = MmapSparseSlice<T>;
    using Tag = TagSparse<IsCSR>;
    
    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;

private:
    std::shared_ptr<CacheManager> cache_;
    std::shared_ptr<PageStore> data_store_;
    std::shared_ptr<PageStore> indices_store_;
    std::unique_ptr<IndptrCache> indptr_;
    
    Index rows_;
    Index cols_;
    Index nnz_;
    
    static constexpr std::size_t kValuesPerPage = kPageSize / sizeof(T);
    static constexpr std::size_t kIndicesPerPage = kPageSize / sizeof(Index);

public:
    // =========================================================================
    // Construction
    // =========================================================================
    
    MmapSparse(std::shared_ptr<CacheManager> cache,
               Index rows, Index cols, Index nnz,
               LoadCallback data_loader,
               LoadCallback indices_loader,
               LoadCallback indptr_loader)
        : cache_(std::move(cache))
        , rows_(rows)
        , cols_(cols)
        , nnz_(nnz)
    {
        validate_dimensions(rows, cols, nnz);
        
        const Index primary_dim = IsCSR ? rows : cols;
        
        // Handle empty matrix
        if (nnz == 0) {
            // Create minimal stores with dummy data
            data_store_ = std::make_shared<PageStore>(
                generate_file_id(), sizeof(T),
                [](std::size_t, std::byte* dest) { std::memset(dest, 0, kPageSize); });
            
            indices_store_ = std::make_shared<PageStore>(
                generate_file_id(), sizeof(Index),
                [](std::size_t, std::byte* dest) { std::memset(dest, 0, kPageSize); });
        } else {
            data_store_ = std::make_shared<PageStore>(
                generate_file_id(), 
                static_cast<std::size_t>(nnz) * sizeof(T),
                std::move(data_loader));
            
            indices_store_ = std::make_shared<PageStore>(
                generate_file_id(),
                static_cast<std::size_t>(nnz) * sizeof(Index),
                std::move(indices_loader));
        }
        
        auto indptr_store = std::make_shared<PageStore>(
            generate_file_id(),
            static_cast<std::size_t>(primary_dim + 1) * sizeof(Index),
            std::move(indptr_loader));
        
        cache_->register_store(data_store_.get());
        cache_->register_store(indices_store_.get());
        cache_->register_store(indptr_store.get());
        
        indptr_ = std::make_unique<IndptrCache>(
            std::move(indptr_store), cache_.get(), primary_dim);
    }
    
    ~MmapSparse() = default;
    
    MmapSparse(const MmapSparse&) = delete;
    MmapSparse& operator=(const MmapSparse&) = delete;
    MmapSparse(MmapSparse&&) noexcept = default;
    MmapSparse& operator=(MmapSparse&&) noexcept = default;
    
    // =========================================================================
    // Dimensions
    // =========================================================================
    
    SCL_NODISCARD Index rows() const noexcept { return rows_; }
    SCL_NODISCARD Index cols() const noexcept { return cols_; }
    SCL_NODISCARD Index nnz() const noexcept { return nnz_; }
    SCL_NODISCARD bool empty() const noexcept { return nnz_ == 0; }
    
    SCL_NODISCARD Index primary_dim() const noexcept {
        return IsCSR ? rows_ : cols_;
    }
    
    SCL_NODISCARD Index secondary_dim() const noexcept {
        return IsCSR ? cols_ : rows_;
    }
    
    // =========================================================================
    // Primary Slice Access (row for CSR, column for CSC)
    // =========================================================================
    
    SCL_NODISCARD SliceType slice(Index i) const {
        if (i < 0 || i >= primary_dim()) {
            return SliceType();
        }
        
        auto [start, end] = indptr_->range(i);
        if (end <= start || start < 0 || end > nnz_) {
            return SliceType(PagedBuffer<T>(), PagedBuffer<Index>(), i);
        }
        
        const Index length = end - start;
        
        auto values = load_paged_buffer<T>(
            data_store_.get(), start, length, kValuesPerPage);
        auto indices = load_paged_buffer<Index>(
            indices_store_.get(), start, length, kIndicesPerPage);
        
        if (!values || !indices) {
            return SliceType();
        }
        
        return SliceType(std::move(*values), std::move(*indices), i);
    }
    
    // CSR-specific aliases
    SCL_NODISCARD SliceType row(Index i) const requires (IsCSR) {
        return slice(i);
    }
    
    SCL_NODISCARD Index row_nnz(Index i) const noexcept requires (IsCSR) {
        return slice_nnz(i);
    }
    
    // CSC-specific aliases  
    SCL_NODISCARD SliceType col(Index j) const requires (!IsCSR) {
        return slice(j);
    }
    
    SCL_NODISCARD Index col_nnz(Index j) const noexcept requires (!IsCSR) {
        return slice_nnz(j);
    }
    
    // Generic
    SCL_NODISCARD Index slice_nnz(Index i) const noexcept {
        if (i < 0 || i >= primary_dim()) return 0;
        return indptr_->length(i);
    }
    
    // =========================================================================
    // Batch Slice Access (for algorithms processing multiple rows/cols)
    // =========================================================================
    
    struct BatchSlice {
        Index start_idx;
        Index end_idx;
        Index data_start;
        Index data_end;
        PagedBuffer<T> values;
        PagedBuffer<Index> indices;
        
        SCL_NODISCARD bool valid() const noexcept { return values.valid(); }
        SCL_NODISCARD Index num_slices() const noexcept { return end_idx - start_idx; }
        SCL_NODISCARD std::size_t total_nnz() const noexcept { return values.size(); }
    };
    
    SCL_NODISCARD BatchSlice batch_slices(Index start, Index count) const {
        BatchSlice result{};
        
        if (start < 0 || start >= primary_dim() || count <= 0) {
            return result;
        }
        
        const Index end = std::min(start + count, primary_dim());
        result.start_idx = start;
        result.end_idx = end;
        
        result.data_start = (*indptr_)[start];
        result.data_end = (*indptr_)[end];
        
        if (result.data_end <= result.data_start) {
            return result;
        }
        
        const Index length = result.data_end - result.data_start;
        
        auto values = load_paged_buffer<T>(
            data_store_.get(), result.data_start, length, kValuesPerPage);
        auto indices = load_paged_buffer<Index>(
            indices_store_.get(), result.data_start, length, kIndicesPerPage);
        
        if (values && indices) {
            result.values = std::move(*values);
            result.indices = std::move(*indices);
        }
        
        return result;
    }
    
    // =========================================================================
    // Raw indptr access (for advanced algorithms)
    // =========================================================================
    
    SCL_NODISCARD const Index* indptr_data() const {
        return indptr_->data();
    }
    
    SCL_NODISCARD std::pair<Index, Index> indptr_range(Index i) const {
        return indptr_->range(i);
    }
    
    // =========================================================================
    // Direct element access (for random access patterns)
    // =========================================================================
    
    struct Element {
        Index row;
        Index col;
        T value;
    };
    
    SCL_NODISCARD std::optional<Element> element_at(Index slice_idx, Index local_idx) const {
        auto s = slice(slice_idx);
        if (!s.valid() || static_cast<std::size_t>(local_idx) >= s.size()) {
            return std::nullopt;
        }
        
        auto [col_or_row, val] = s[local_idx];
        
        if constexpr (IsCSR) {
            return Element{slice_idx, col_or_row, val};
        } else {
            return Element{col_or_row, slice_idx, val};
        }
    }
    
    // =========================================================================
    // Prefetch Control
    // =========================================================================
    
    void prefetch_slices(Index start, Index count) const {
        if (start < 0 || start >= primary_dim() || count <= 0) {
            return;
        }
        
        const Index end = std::min(start + count, primary_dim());
        const Index data_start = (*indptr_)[start];
        const Index data_end = (*indptr_)[end];
        
        if (data_end <= data_start) return;
        
        prefetch_range(data_store_.get(), data_start, data_end - data_start, sizeof(T));
        prefetch_range(indices_store_.get(), data_start, data_end - data_start, sizeof(Index));
    }
    
    void prefetch_rows(Index start, Index count) const requires (IsCSR) {
        prefetch_slices(start, count);
    }
    
    void prefetch_cols(Index start, Index count) const requires (!IsCSR) {
        prefetch_slices(start, count);
    }
    
    // =========================================================================
    // Statistics
    // =========================================================================
    
    SCL_NODISCARD std::size_t resident_pages() const noexcept {
        return cache_->resident_count();
    }
    
    SCL_NODISCARD std::size_t total_pages() const noexcept {
        return data_store_->num_pages() + 
               indices_store_->num_pages();
    }
    
    SCL_NODISCARD double avg_slice_nnz() const noexcept {
        Index dim = primary_dim();
        return (dim > 0) ? static_cast<double>(nnz_) / dim : 0.0;
    }
    
    // =========================================================================
    // Accessors
    // =========================================================================
    
    SCL_NODISCARD CacheManager* cache() noexcept { return cache_.get(); }
    SCL_NODISCARD const CacheManager* cache() const noexcept { return cache_.get(); }

private:
    // =========================================================================
    // Validation
    // =========================================================================
    
    static void validate_dimensions(Index rows, Index cols, Index nnz) {
        if (rows < 0) {
            throw ValueError("MmapSparse: rows must be non-negative");
        }
        if (cols < 0) {
            throw ValueError("MmapSparse: cols must be non-negative");
        }
        if (nnz < 0) {
            throw ValueError("MmapSparse: nnz must be non-negative");
        }
        
        // Check for overflow
        const auto max_nnz = static_cast<Index>(SIZE_MAX / std::max(sizeof(T), sizeof(Index)));
        if (nnz > max_nnz) {
            throw ValueError("MmapSparse: nnz too large");
        }
    }
    
    // =========================================================================
    // Page Loading
    // =========================================================================
    
    template <typename U>
    std::optional<PagedBuffer<U>> load_paged_buffer(
        PageStore* store,
        Index start_elem,
        Index count,
        std::size_t elems_per_page) const
    {
        if (count <= 0 || !store) {
            return std::nullopt;
        }
        
        const std::size_t byte_offset = static_cast<std::size_t>(start_elem) * sizeof(U);
        const std::size_t first_page_idx = byte_offset / kPageSize;
        const std::size_t first_page_off = byte_offset % kPageSize;
        
        // Load first page
        PageHandle first_handle = cache_->request(first_page_idx, store);
        if (!first_handle) {
            return std::nullopt;
        }
        
        const U* first_ptr = reinterpret_cast<const U*>(first_handle.data() + first_page_off);
        const std::size_t first_page_elems = (kPageSize - first_page_off) / sizeof(U);
        const std::size_t first_count = std::min(first_page_elems, static_cast<std::size_t>(count));
        
        PagedBuffer<U> buffer(std::move(first_handle), first_ptr, first_count);
        
        // Load additional pages if needed
        std::size_t remaining = static_cast<std::size_t>(count) - first_count;
        std::size_t page_idx = first_page_idx + 1;
        
        while (remaining > 0 && buffer.num_pages() < PagedBuffer<U>::kMaxHandles) {
            PageHandle handle = cache_->request(page_idx, store);
            if (!handle) {
                // Partial load - return what we have
                break;
            }
            
            const U* ptr = handle.as<U>();
            const std::size_t page_elems = std::min(elems_per_page, remaining);
            
            buffer.add_page(std::move(handle), ptr, page_elems);
            remaining -= page_elems;
            ++page_idx;
        }
        
        return buffer;
    }
    
    void prefetch_range(PageStore* store, Index start_elem, Index count, std::size_t elem_size) const {
        if (count <= 0 || !store) return;
        
        const std::size_t start_byte = static_cast<std::size_t>(start_elem) * elem_size;
        const std::size_t end_byte = start_byte + static_cast<std::size_t>(count) * elem_size;
        
        const std::size_t start_page = start_byte / kPageSize;
        const std::size_t end_page = (end_byte + kPageSize - 1) / kPageSize;
        
        for (std::size_t p = start_page; p < end_page && p < store->num_pages(); ++p) {
            cache_->prefetch(p, store);
        }
    }
};

// =============================================================================
// Type Aliases
// =============================================================================

using MmapCSR = MmapSparse<Real, true>;
using MmapCSC = MmapSparse<Real, false>;

template <typename T>
using MmapCSROf = MmapSparse<T, true>;

template <typename T>
using MmapCSCOf = MmapSparse<T, false>;

using MmapCSRF32 = MmapSparse<float, true>;
using MmapCSRF64 = MmapSparse<double, true>;
using MmapCSCF32 = MmapSparse<float, false>;
using MmapCSCF64 = MmapSparse<double, false>;

// =============================================================================
// Factory Functions
// =============================================================================

template <typename T, bool IsCSR = true>
SCL_NODISCARD std::shared_ptr<MmapSparse<T, IsCSR>> make_mmap_sparse(
    std::shared_ptr<CacheManager> cache,
    Index rows, Index cols, Index nnz,
    LoadCallback data_loader,
    LoadCallback indices_loader,
    LoadCallback indptr_loader)
{
    return std::make_shared<MmapSparse<T, IsCSR>>(
        std::move(cache), rows, cols, nnz,
        std::move(data_loader),
        std::move(indices_loader),
        std::move(indptr_loader));
}

template <typename T>
SCL_NODISCARD std::shared_ptr<MmapCSROf<T>> make_mmap_csr(
    std::shared_ptr<CacheManager> cache,
    Index rows, Index cols, Index nnz,
    LoadCallback data_loader,
    LoadCallback indices_loader,
    LoadCallback indptr_loader)
{
    return make_mmap_sparse<T, true>(
        std::move(cache), rows, cols, nnz,
        std::move(data_loader),
        std::move(indices_loader),
        std::move(indptr_loader));
}

template <typename T>
SCL_NODISCARD std::shared_ptr<MmapCSCOf<T>> make_mmap_csc(
    std::shared_ptr<CacheManager> cache,
    Index rows, Index cols, Index nnz,
    LoadCallback data_loader,
    LoadCallback indices_loader,
    LoadCallback indptr_loader)
{
    return make_mmap_sparse<T, false>(
        std::move(cache), rows, cols, nnz,
        std::move(data_loader),
        std::move(indices_loader),
        std::move(indptr_loader));
}

// =============================================================================
// Concept Validation
// =============================================================================

static_assert(CSRLike<MmapCSR>);
static_assert(CSCLike<MmapCSC>);
static_assert(SparseLike<MmapCSR>);
static_assert(SparseLike<MmapCSC>);

} // namespace scl::mmap
