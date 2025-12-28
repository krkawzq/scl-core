// =============================================================================
// FILE: scl/mmap/integration/view.h
// BRIEF: API reference for zero-copy view into mmap arrays
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "../cache/tiered.h"
#include "../configuration.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <memory>
#include <type_traits>

namespace scl::mmap::integration {

/* =============================================================================
 * STRUCT: ViewConfig
 * =============================================================================
 * SUMMARY:
 *     Configuration for zero-copy view behavior.
 *
 * FIELDS:
 *     pin_pages         - Keep pages pinned while view is active
 *     prefetch_adjacent - Prefetch adjacent pages on access
 *     prefetch_count    - Number of pages to prefetch
 *     allow_partial     - Allow views that cross page boundaries
 * -------------------------------------------------------------------------- */
struct ViewConfig {
    bool pin_pages = true;
    bool prefetch_adjacent = true;
    std::size_t prefetch_count = 4;
    bool allow_partial = true;

    /* -------------------------------------------------------------------------
     * FACTORY: pinned
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration that keeps pages pinned.
     * ---------------------------------------------------------------------- */
    static constexpr ViewConfig pinned() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: streaming
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration for sequential access patterns.
     * ---------------------------------------------------------------------- */
    static constexpr ViewConfig streaming() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: random_access
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration for random access patterns.
     * ---------------------------------------------------------------------- */
    static constexpr ViewConfig random_access() noexcept;
};

/* =============================================================================
 * CLASS: ZeroCopyView
 * =============================================================================
 * SUMMARY:
 *     Zero-copy view into memory-mapped array data.
 *
 * DESIGN PURPOSE:
 *     Provides efficient, zero-copy access to contiguous array slices:
 *     - No data copying when view is within single page
 *     - Automatic page pinning prevents eviction
 *     - Type-safe access with bounds checking
 *     - RAII management of page lifetimes
 *
 * ARCHITECTURE:
 *     ZeroCopyView<T>
 *         ├── PageHandle (one per covered page)
 *         └── Metadata (offset, length, stride)
 *
 * MEMORY LAYOUT:
 *     For single-page views: Direct pointer into page buffer
 *     For multi-page views: Contiguous only if pages are adjacent
 *
 * THREAD SAFETY:
 *     Views are NOT thread-safe. Create separate views per thread.
 *     The underlying cache is thread-safe.
 *
 * LIFETIME:
 *     View is valid as long as:
 *     - View object exists AND
 *     - Underlying MmapArray exists AND
 *     - File has not been remapped
 * -------------------------------------------------------------------------- */
template <typename T>
class ZeroCopyView {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using iterator = T*;
    using const_iterator = const T*;

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: ZeroCopyView (default)
     * -------------------------------------------------------------------------
     * POSTCONDITIONS:
     *     Creates empty view (data() == nullptr, size() == 0).
     * ---------------------------------------------------------------------- */
    ZeroCopyView() noexcept;

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: ZeroCopyView (copy)
     * -------------------------------------------------------------------------
     * NOTE:
     *     Shallow copy - shares underlying page handles.
     *     Both views keep pages pinned.
     * ---------------------------------------------------------------------- */
    ZeroCopyView(const ZeroCopyView& other);
    ZeroCopyView& operator=(const ZeroCopyView& other);

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: ZeroCopyView (move)
     * ---------------------------------------------------------------------- */
    ZeroCopyView(ZeroCopyView&& other) noexcept;
    ZeroCopyView& operator=(ZeroCopyView&& other) noexcept;

    /* -------------------------------------------------------------------------
     * DESTRUCTOR
     * -------------------------------------------------------------------------
     * POSTCONDITIONS:
     *     - Page handles released
     *     - Pages unpinned (may be evicted)
     * ---------------------------------------------------------------------- */
    ~ZeroCopyView();

    /* -------------------------------------------------------------------------
     * METHOD: data
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Pointer to first element, or nullptr if empty.
     *
     * PRECONDITIONS:
     *     View must be valid (valid() == true).
     *
     * LIFETIME:
     *     Valid until view is destroyed or moved.
     * ---------------------------------------------------------------------- */
    const T* data() const noexcept;
    T* data() noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: size
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Number of elements in view.
     * ---------------------------------------------------------------------- */
    size_type size() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: size_bytes
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Size of view in bytes.
     * ---------------------------------------------------------------------- */
    size_type size_bytes() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: empty
     * -------------------------------------------------------------------------
     * RETURNS:
     *     True if size() == 0.
     * ---------------------------------------------------------------------- */
    bool empty() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: valid
     * -------------------------------------------------------------------------
     * RETURNS:
     *     True if view points to valid data.
     * ---------------------------------------------------------------------- */
    bool valid() const noexcept;
    explicit operator bool() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: is_contiguous
     * -------------------------------------------------------------------------
     * RETURNS:
     *     True if view data is contiguous in memory.
     *
     * NOTE:
     *     Always true for single-page views.
     *     May be false for multi-page views if pages are non-adjacent.
     * ---------------------------------------------------------------------- */
    bool is_contiguous() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: operator[]
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Access element by index (no bounds check).
     *
     * PRECONDITIONS:
     *     - idx < size()
     *     - View is contiguous
     * ---------------------------------------------------------------------- */
    const_reference operator[](size_type idx) const noexcept;
    reference operator[](size_type idx) noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: at
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Access element by index with bounds check.
     *
     * THROWS:
     *     std::out_of_range if idx >= size().
     * ---------------------------------------------------------------------- */
    const_reference at(size_type idx) const;
    reference at(size_type idx);

    /* -------------------------------------------------------------------------
     * METHOD: front / back
     * -------------------------------------------------------------------------
     * PRECONDITIONS:
     *     !empty()
     * ---------------------------------------------------------------------- */
    const_reference front() const noexcept;
    reference front() noexcept;
    const_reference back() const noexcept;
    reference back() noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: begin / end
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Iterators for range-based for loops.
     *
     * PRECONDITIONS:
     *     is_contiguous() == true
     * ---------------------------------------------------------------------- */
    const_iterator begin() const noexcept;
    iterator begin() noexcept;
    const_iterator end() const noexcept;
    iterator end() noexcept;
    const_iterator cbegin() const noexcept;
    const_iterator cend() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: as_span
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Convert to std::span for STL compatibility.
     *
     * PRECONDITIONS:
     *     is_contiguous() == true
     *
     * RETURNS:
     *     Span covering the view data.
     * ---------------------------------------------------------------------- */
    std::span<const T> as_span() const noexcept;
    std::span<T> as_span() noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: subview
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create a sub-view of this view.
     *
     * PARAMETERS:
     *     offset [in] - Starting element index
     *     count  [in] - Number of elements (npos = rest)
     *
     * PRECONDITIONS:
     *     - offset <= size()
     *     - offset + count <= size()
     *
     * RETURNS:
     *     New view sharing underlying pages.
     * ---------------------------------------------------------------------- */
    ZeroCopyView subview(
        size_type offset,                  // Start index
        size_type count = npos             // Element count
    ) const;

    /* -------------------------------------------------------------------------
     * METHOD: copy_to
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Copy view data to destination buffer.
     *
     * PARAMETERS:
     *     dest [out] - Destination buffer (must have size() capacity)
     *
     * POSTCONDITIONS:
     *     - dest contains copy of view data
     *     - Works for non-contiguous views
     *
     * RETURNS:
     *     Number of elements copied.
     * ---------------------------------------------------------------------- */
    size_type copy_to(T* dest) const;
    size_type copy_to(std::span<T> dest) const;

    /* -------------------------------------------------------------------------
     * METHOD: page_count
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Number of pages this view spans.
     * ---------------------------------------------------------------------- */
    size_type page_count() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: mark_dirty
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Mark all pages as modified.
     *
     * POSTCONDITIONS:
     *     Pages will be written back on eviction.
     * ---------------------------------------------------------------------- */
    void mark_dirty() noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: release
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Release pages early (without destroying view object).
     *
     * POSTCONDITIONS:
     *     - View becomes invalid
     *     - Pages unpinned
     * ---------------------------------------------------------------------- */
    void release() noexcept;

    /* -------------------------------------------------------------------------
     * CONSTANT: npos
     * -------------------------------------------------------------------------
     * Maximum value for size_type (used as "rest of view" marker).
     * ---------------------------------------------------------------------- */
    static constexpr size_type npos = static_cast<size_type>(-1);

private:
    friend class ViewBuilder;

    struct Impl;
    std::shared_ptr<Impl> impl_;

    // Private constructor for ViewBuilder
    explicit ZeroCopyView(std::shared_ptr<Impl> impl);
};

/* =============================================================================
 * CLASS: ViewBuilder
 * =============================================================================
 * SUMMARY:
 *     Factory for creating ZeroCopyView instances.
 *
 * DESIGN PURPOSE:
 *     Separates view construction from view usage:
 *     - Handles page loading and pinning
 *     - Manages multi-page views
 *     - Provides type erasure for different backends
 *
 * USAGE:
 *     ViewBuilder builder(cache);
 *     auto view = builder.build<float>(mmap_array, 0, 1000);
 * -------------------------------------------------------------------------- */
class ViewBuilder {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: ViewBuilder
     * -------------------------------------------------------------------------
     * PARAMETERS:
     *     cache  [in] - Tiered cache for page access
     *     config [in] - View configuration
     * ---------------------------------------------------------------------- */
    explicit ViewBuilder(
        cache::TieredCache& cache,         // Cache for page access
        ViewConfig config = {}             // View configuration
    );

    ~ViewBuilder();

    ViewBuilder(const ViewBuilder&) = delete;
    ViewBuilder& operator=(const ViewBuilder&) = delete;

    /* -------------------------------------------------------------------------
     * METHOD: build
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create a view into array data.
     *
     * TEMPLATE PARAMETERS:
     *     T       - Element type
     *     ArrayT  - MmapArray type
     *
     * PARAMETERS:
     *     array   [in] - Source MmapArray
     *     offset  [in] - Starting element index
     *     count   [in] - Number of elements
     *
     * PRECONDITIONS:
     *     - offset + count <= array.size()
     *
     * POSTCONDITIONS:
     *     - Pages loaded and pinned
     *     - View valid for duration of its lifetime
     *
     * RETURNS:
     *     Valid view, or empty view on failure.
     *
     * THREAD SAFETY:
     *     Thread-safe (multiple builders can work concurrently).
     * ---------------------------------------------------------------------- */
    template <typename T, typename ArrayT>
    ZeroCopyView<T> build(
        ArrayT& array,                     // Source array
        std::size_t offset,                // Start index
        std::size_t count                  // Element count
    );

    /* -------------------------------------------------------------------------
     * METHOD: build_all
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create a view of entire array.
     *
     * RETURNS:
     *     View covering all elements.
     * ---------------------------------------------------------------------- */
    template <typename T, typename ArrayT>
    ZeroCopyView<T> build_all(
        ArrayT& array                      // Source array
    );

    /* -------------------------------------------------------------------------
     * METHOD: config
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Reference to view configuration.
     * ---------------------------------------------------------------------- */
    const ViewConfig& config() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* =============================================================================
 * TYPE TRAITS
 * =============================================================================
 * Helper traits for ZeroCopyView.
 * -------------------------------------------------------------------------- */

template <typename T>
struct is_zero_copy_view : std::false_type {};

template <typename T>
struct is_zero_copy_view<ZeroCopyView<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_zero_copy_view_v = is_zero_copy_view<T>::value;

/* =============================================================================
 * FUNCTION: make_view
 * =============================================================================
 * SUMMARY:
 *     Convenience function to create a view.
 *
 * PARAMETERS:
 *     cache   [in] - Tiered cache
 *     array   [in] - Source array
 *     offset  [in] - Start index
 *     count   [in] - Element count
 *
 * RETURNS:
 *     ZeroCopyView into the specified range.
 * -------------------------------------------------------------------------- */
template <typename T, typename ArrayT>
ZeroCopyView<T> make_view(
    cache::TieredCache& cache,
    ArrayT& array,
    std::size_t offset,
    std::size_t count
);

/* =============================================================================
 * FUNCTION: make_view_all
 * =============================================================================
 * SUMMARY:
 *     Create view of entire array.
 * -------------------------------------------------------------------------- */
template <typename T, typename ArrayT>
ZeroCopyView<T> make_view_all(
    cache::TieredCache& cache,
    ArrayT& array
);

} // namespace scl::mmap::integration
