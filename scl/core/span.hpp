#pragma once

/// @file scl/core/span.hpp
/// @brief Shared Buffer and Span with Reference-Counted Ownership
///
/// This header provides:
///   - SharedBuffer: Reference-counted raw memory block
///   - SharedSpan<T>: Typed array view with three ownership modes
///
/// ## Ownership Modes (Compact Layout)
///
///   | Mode   | buffer_           | Description                          |
///   |--------|-------------------|--------------------------------------|
///   | Owned  | &OWNED_SENTINEL   | Exclusive ownership, delete[] data_  |
///   | Shared | valid SharedBuffer| Shared ownership via SharedBuffer    |
///   | View   | nullptr           | Non-owning view                      |
///
/// ## Memory Layout
///
///   SharedSpan<T> uses only 24 bytes (3 pointers):
///     - buffer_: SharedBuffer* or sentinel pointer
///     - data_: T* data pointer
///     - size_: Size element count
///
/// ## Thread Safety
///
///   - SharedBuffer: Thread-safe reference counting (atomic operations)
///   - SharedSpan: Thread-safe for read-only access when sharing
///   - Concurrent writes to the same data require external synchronization
///
/// @note Uses scl::memory module for allocations

#include "scl/core/macro.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

namespace scl {

// =============================================================================
// SECTION 1: Forward Declarations and Tags
// =============================================================================

class SharedBuffer;

template<typename T>
class SharedSpan;

/// @brief Tag for adopting ownership without incrementing reference count
struct AdoptTag { explicit AdoptTag() = default; };

/// @brief Global adopt tag instance
inline constexpr AdoptTag adopt{};

// =============================================================================
// SECTION 2: Deleter Types
// =============================================================================

/// @brief Deleter function signature
using Deleter = void(*)(void*) noexcept;

/// @brief Built-in deleter functions
namespace deleter {

/// @brief Free memory with std::free (for posix_memalign)
inline
auto free_raw(void* ptr) noexcept -> void {
    std::free(ptr);
}

/// @brief Delete array allocated with new char[]
inline
auto delete_char_array(void* ptr) noexcept -> void {
    delete[] static_cast<char*>(ptr);
}

/// @brief No-op deleter for non-owned memory
inline
auto noop(void*) noexcept -> void {}

#if SCL_PLATFORM_WINDOWS
/// @brief Windows aligned free
inline
auto aligned_free_raw(void* ptr) noexcept -> void {
    _aligned_free(ptr);
}
#else
/// @brief POSIX aligned free (same as std::free)
inline
auto aligned_free_raw(void* ptr) noexcept -> void {
    std::free(ptr);
}
#endif

}  // namespace deleter

// =============================================================================
// SECTION 3: SharedBuffer
// =============================================================================

/// @brief Reference-counted shared memory buffer
///
/// SharedBuffer manages a raw memory block with automatic cleanup when
/// the reference count reaches zero.
///
/// ## Thread Safety
///
/// Reference counting operations are thread-safe:
///   - incref()/decref(): Atomic operations with appropriate memory ordering
///   - Data access: Safe for concurrent reads; writes require synchronization
///
/// ## Lifecycle
///
///   1. Factory method creates buffer with ref_count = 1
///   2. SharedSpan::adopt() takes ownership without incref
///   3. SharedSpan copy/share operations call incref()
///   4. Destruction calls decref()
///   5. When ref_count reaches 0, buffer self-destructs
///
/// @note Not copyable/movable - use incref/decref for ownership transfer
/// @note Self-destructs when reference count reaches zero
class SharedBuffer {
public:
    // -------------------------------------------------------------------------
    // Factory Methods
    // -------------------------------------------------------------------------

    /// @brief Create buffer with unaligned allocation
    /// @param[in] size Size in bytes
    /// @return Pointer to SharedBuffer (ref_count=1), or nullptr on failure
    /// @note Uses new char[] for allocation
    [[nodiscard]]
    static
    auto create(std::size_t size) noexcept -> SharedBuffer* {
        if (size == 0) [[unlikely]] return nullptr;

        auto* data = new (std::nothrow) char[size]();
        if (!data) [[unlikely]] return nullptr;

        auto* buffer = new (std::nothrow) SharedBuffer{
            data, size, deleter::delete_char_array, 1};
        if (!buffer) [[unlikely]] {
            delete[] data;
            return nullptr;
        }

        return buffer;
    }

    /// @brief Create buffer with aligned allocation
    /// @param[in] size Size in bytes
    /// @param[in] alignment Alignment requirement (must be power of 2, >= sizeof(void*))
    /// @return Pointer to SharedBuffer (ref_count=1), or nullptr on failure
    [[nodiscard]]
    static
    auto create_aligned(
        std::size_t size,
        std::size_t alignment = memory::DEFAULT_ALIGNMENT
    ) noexcept -> SharedBuffer* {
        if (size == 0) [[unlikely]] return nullptr;

        // Validate alignment
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) [[unlikely]] {
            return nullptr;  // Not power of 2
        }
        if (alignment < sizeof(void*)) {
            alignment = sizeof(void*);
        }

        void* data = nullptr;

#if SCL_PLATFORM_WINDOWS
        data = _aligned_malloc(size, alignment);
#else
        if (::posix_memalign(&data, alignment, size) != 0) {
            data = nullptr;
        }
#endif

        if (!data) [[unlikely]] return nullptr;

        // Zero-initialize
        std::memset(data, 0, size);

        auto* buffer = new (std::nothrow) SharedBuffer{
            data, size, deleter::aligned_free_raw, alignment};
        if (!buffer) [[unlikely]] {
            deleter::aligned_free_raw(data);
            return nullptr;
        }

        return buffer;
    }

    /// @brief Wrap external memory
    /// @param[in] data Pointer to external data
    /// @param[in] size Size in bytes
    /// @param[in] del Deleter function (use deleter::noop for non-owning)
    /// @return Pointer to SharedBuffer (ref_count=1), or nullptr on failure
    [[nodiscard]]
    static
    auto from_external(
        void* data,
        std::size_t size,
        Deleter del = deleter::noop
    ) noexcept -> SharedBuffer* {
        if (!data || size == 0) [[unlikely]] return nullptr;
        return new (std::nothrow) SharedBuffer{data, size, del, 1};
    }

    // -------------------------------------------------------------------------
    // Reference Counting
    // -------------------------------------------------------------------------

    /// @brief Increment reference count
    /// @param[in] n Number of references to add (default 1)
    /// @return New reference count
    /// @note Memory order: relaxed (only atomicity required)
    SCL_FORCE_INLINE
    auto incref(std::uint32_t n = 1) noexcept -> std::uint32_t {
        return refcount_.fetch_add(n, std::memory_order_relaxed) + n;
    }

    /// @brief Decrement reference count, releasing buffer if count reaches zero
    /// @param[in] n Number of references to remove (default 1)
    /// @return true if buffer was released (do not access after this!)
    /// @note Memory order: acq_rel (synchronizes with prior incref/writes)
    SCL_FORCE_INLINE
    auto decref(std::uint32_t n = 1) noexcept -> bool {
        const auto old = refcount_.fetch_sub(n, std::memory_order_acq_rel);
        SCL_DEBUG_ASSERT(old >= n);  // Underflow check
        if (old == n) {
            release_impl();
            return true;
        }
        return false;
    }

    /// @brief Get current reference count
    /// @return Approximate reference count (may be stale in concurrent scenarios)
    /// @warning Do not use for synchronization; use is_unique() for ownership checks
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto use_count() const noexcept -> std::uint32_t {
        return refcount_.load(std::memory_order_relaxed);
    }

    /// @brief Check if this is the only reference
    /// @return true if ref_count == 1
    /// @note Safe for ownership checks before modification
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto is_unique() const noexcept -> bool {
        return refcount_.load(std::memory_order_acquire) == 1;
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    [[nodiscard]] SCL_FORCE_INLINE auto data() noexcept -> void* { return data_; }
    [[nodiscard]] SCL_FORCE_INLINE auto data() const noexcept -> const void* { return data_; }

    template<typename T>
    [[nodiscard]] SCL_FORCE_INLINE auto data_as() noexcept -> T* { return static_cast<T*>(data_); }
    template<typename T>
    [[nodiscard]] SCL_FORCE_INLINE auto data_as() const noexcept -> const T* { return static_cast<const T*>(data_); }

    [[nodiscard]] SCL_FORCE_INLINE auto size() const noexcept -> std::size_t { return size_; }
    [[nodiscard]] SCL_FORCE_INLINE auto alignment() const noexcept -> std::size_t { return alignment_; }

    template<typename T>
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto count() const noexcept -> Size {
        return static_cast<Size>(size_ / sizeof(T));
    }

    template<typename T>
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto is_aligned_for() const noexcept -> bool {
        return alignment_ >= alignof(T) &&
               (reinterpret_cast<std::uintptr_t>(data_) % alignof(T)) == 0;
    }

    [[nodiscard]] SCL_FORCE_INLINE explicit operator bool() const noexcept { return data_ != nullptr; }

    // -------------------------------------------------------------------------
    // Non-copyable, Non-movable
    // -------------------------------------------------------------------------

    SharedBuffer(const SharedBuffer&) = delete;
    auto operator=(const SharedBuffer&) -> SharedBuffer& = delete;
    SharedBuffer(SharedBuffer&&) = delete;
    auto operator=(SharedBuffer&&) -> SharedBuffer& = delete;

private:
    SharedBuffer(void* data, std::size_t size, Deleter del, std::size_t alignment) noexcept
        : data_(data), size_(size), deleter_(del), alignment_(alignment), refcount_(1) {}

    ~SharedBuffer() = default;

    auto release_impl() noexcept -> void {
        if (deleter_ && data_) {
            deleter_(data_);
        }
        data_ = nullptr;
        delete this;
    }

    void* data_;
    std::size_t size_;
    Deleter deleter_;
    std::size_t alignment_;
    std::atomic<std::uint32_t> refcount_;
};

// =============================================================================
// SECTION 4: Ownership Sentinel
// =============================================================================

namespace detail {

/// @brief Sentinel object to mark owned spans
///
/// When buffer_ points to this sentinel, the span is in owned mode
/// and data_ should be freed with delete[].
///
/// This allows SharedSpan to use only 3 members (24 bytes) instead of 4 (40 bytes).
struct OwnedSentinel {
    // Storage with same alignment as SharedBuffer, used only for its address
    alignas(SharedBuffer) std::array<std::byte, sizeof(SharedBuffer)> storage{};
};

/// @brief Global sentinel instance for owned spans
/// @note Only the address is used; the object is never dereferenced
inline OwnedSentinel OWNED_SENTINEL_INSTANCE;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

/// @brief Get pointer to owned sentinel (as SharedBuffer*)
/// @note The sentinel is never dereferenced, only compared by address
[[nodiscard]]
SCL_FORCE_INLINE
auto owned_sentinel() noexcept -> SharedBuffer* {
    return reinterpret_cast<SharedBuffer*>(&OWNED_SENTINEL_INSTANCE);
}

/// @brief Check if a buffer pointer is the owned sentinel
[[nodiscard]]
SCL_FORCE_INLINE
auto is_owned_sentinel(const SharedBuffer* ptr) noexcept -> bool {
    return ptr == reinterpret_cast<const SharedBuffer*>(&OWNED_SENTINEL_INSTANCE);
}

}  // namespace detail

// =============================================================================
// SECTION 5: SharedSpan (Compact 24-byte Layout)
// =============================================================================

/// @brief Typed array view with shared or exclusive ownership
///
/// SharedSpan provides three ownership modes with a compact 24-byte layout:
///
///   | Mode   | buffer_           | data_  | Cleanup               |
///   |--------|-------------------|--------|-----------------------|
///   | Owned  | &OWNED_SENTINEL   | valid  | delete[] data_        |
///   | Shared | valid SharedBuffer| valid  | buffer_->decref()     |
///   | View   | nullptr           | valid  | nothing               |
///
/// ## Memory Layout (24 bytes)
///
///   - buffer_: 8 bytes (SharedBuffer* or sentinel)
///   - data_: 8 bytes (T*)
///   - size_: 8 bytes (Size)
///
/// ## Subspan Safety
///
/// **Warning**: subspan() on an Owned span creates a View that will become
/// dangling if the original span is destroyed. Use to_shared() first if
/// you need safe subspan operations.
///
/// @tparam T Element type
template<typename T>
class SharedSpan {
public:
    // -------------------------------------------------------------------------
    // Type Aliases (STL-compatible)
    // -------------------------------------------------------------------------

    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = Size;
    using difference_type = std::ptrdiff_t;
    using iterator = T*;
    using const_iterator = const T*;

    /// @brief Sentinel value for "rest of span"
    static constexpr Size npos = static_cast<Size>(-1);

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    /// @brief Default: empty span (view mode)
    constexpr
    SharedSpan() noexcept = default;

    /// @brief Construct from SharedBuffer with sharing semantics (incref)
    /// @param[in] buffer Shared buffer (reference count incremented)
    /// @param[in] offset Byte offset from buffer start
    /// @param[in] count Number of elements
    SharedSpan(SharedBuffer* buffer, std::ptrdiff_t offset, Size count) noexcept
        : buffer_(buffer)
        , data_(buffer ? reinterpret_cast<T*>(
              static_cast<char*>(buffer->data()) + offset) : nullptr)
        , size_(count) {
        if (buffer_ && !detail::is_owned_sentinel(buffer_)) {
            buffer_->incref();
        }
    }

    /// @brief Construct from SharedBuffer at offset 0 with sharing semantics
    SharedSpan(SharedBuffer* buffer, Size count) noexcept
        : SharedSpan(buffer, 0, count) {}

    /// @brief Adopt ownership of SharedBuffer without incref
    /// @param[in] buffer Shared buffer (ownership transferred, no incref)
    /// @param[in] offset Byte offset from buffer start
    /// @param[in] count Number of elements
    /// @param[in] tag AdoptTag to select this constructor
    SharedSpan(SharedBuffer* buffer, std::ptrdiff_t offset, Size count, AdoptTag) noexcept
        : buffer_(buffer)
        , data_(buffer ? reinterpret_cast<T*>(
              static_cast<char*>(buffer->data()) + offset) : nullptr)
        , size_(count) {
        // No incref - adopting existing reference
    }

    // -------------------------------------------------------------------------
    // Factory Methods
    // -------------------------------------------------------------------------

    /// @brief Create owned span with new[] allocation
    /// @param[in] count Number of elements
    /// @return SharedSpan in owned mode, or empty on failure
    [[nodiscard]]
    static
    auto create_owned(Size count) noexcept -> SharedSpan {
        if (count <= 0) [[unlikely]] return {};

        auto* data = new (std::nothrow) T[static_cast<std::size_t>(count)]();
        if (!data) [[unlikely]] return {};

        SharedSpan span;
        span.buffer_ = detail::owned_sentinel();  // Mark as owned
        span.data_ = data;
        span.size_ = count;
        return span;
    }

    /// @brief Create span with aligned allocation via SharedBuffer
    /// @param[in] count Number of elements
    /// @param[in] alignment Byte alignment (must be power of 2)
    /// @return SharedSpan in shared mode, or empty on failure
    [[nodiscard]]
    static
    auto create_aligned(
        Size count,
        std::size_t alignment = memory::DEFAULT_ALIGNMENT
    ) noexcept -> SharedSpan {
        if (count <= 0) [[unlikely]] return {};

        auto* buffer = SharedBuffer::create_aligned(
            static_cast<std::size_t>(count) * sizeof(T), alignment);
        if (!buffer) [[unlikely]] return {};

        // Adopt the buffer (ref_count is already 1)
        return SharedSpan{buffer, 0, count, adopt};
    }

    /// @brief Create shared span from buffer (incref)
    [[nodiscard]]
    static
    auto create_shared(SharedBuffer* buffer, std::ptrdiff_t offset, Size count) noexcept
        -> SharedSpan {
        return SharedSpan{buffer, offset, count};
    }

    /// @brief Create non-owning view of raw pointer
    /// @param[in] data Raw pointer
    /// @param[in] count Number of elements
    /// @return SharedSpan that does NOT own data
    /// @warning Caller must ensure data outlives the span
    [[nodiscard]]
    static
    auto view(T* data, Size count) noexcept -> SharedSpan {
        SharedSpan span;
        span.data_ = data;
        span.size_ = count;
        // buffer_ = nullptr means view mode
        return span;
    }

    /// @brief Create view from std::span
    template<std::size_t Extent>
    [[nodiscard]]
    static
    auto view(std::span<T, Extent> s) noexcept -> SharedSpan {
        return view(s.data(), static_cast<Size>(s.size()));
    }

    // -------------------------------------------------------------------------
    // Copy/Move Semantics
    // -------------------------------------------------------------------------

    /// @brief Copy constructor
    /// @note Shared mode: incref buffer
    /// @note Owned mode: deep copy (may fail silently → empty span)
    /// @note View mode: shallow copy
    SharedSpan(const SharedSpan& other)
        : buffer_(other.buffer_)
        , size_(other.size_) {
        if (detail::is_owned_sentinel(buffer_)) {
            // Owned mode: deep copy
            if (other.data_ && other.size_ > 0) {
                auto* new_data = new (std::nothrow) T[static_cast<std::size_t>(size_)];
                if (new_data) {
                    std::copy(other.data_, other.data_ + size_, new_data);
                    data_ = new_data;
                } else {
                    // Allocation failed - become empty view
                    buffer_ = nullptr;
                    size_ = 0;
                }
            }
        } else if (buffer_) {
            // Shared mode: share the buffer
            buffer_->incref();
            data_ = other.data_;
        } else {
            // View mode: just copy pointer
            data_ = other.data_;
        }
    }

    /// @brief Copy assignment
    auto operator=(const SharedSpan& other) -> SharedSpan& {
        if (this != &other) {
            SharedSpan tmp{other};
            swap(tmp);
        }
        return *this;
    }

    /// @brief Move constructor
    SharedSpan(SharedSpan&& other) noexcept
        : buffer_(std::exchange(other.buffer_, nullptr))
        , data_(std::exchange(other.data_, nullptr))
        , size_(std::exchange(other.size_, 0)) {}

    /// @brief Move assignment
    auto operator=(SharedSpan&& other) noexcept -> SharedSpan& {
        if (this != &other) {
            release_resources();
            buffer_ = std::exchange(other.buffer_, nullptr);
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    /// @brief Destructor
    ~SharedSpan() {
        release_resources();
    }

    // -------------------------------------------------------------------------
    // Element Access (STL-compatible)
    // -------------------------------------------------------------------------

    /// @brief Access element with bounds checking
    /// @throws IndexError if index out of bounds
    [[nodiscard]]
    auto at(Size idx) -> reference {
        SCL_CHECK_INDEX(idx, size_);
        return data_[idx];
    }

    /// @brief Access element with bounds checking (const)
    [[nodiscard]]
    auto at(Size idx) const -> const_reference {
        SCL_CHECK_INDEX(idx, size_);
        return data_[idx];
    }

    /// @brief Access element without bounds checking
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto operator[](Size idx) noexcept -> reference {
        return data_[idx];
    }

    /// @brief Access element without bounds checking (const)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto operator[](Size idx) const noexcept -> const_reference {
        return data_[idx];
    }

    [[nodiscard]] SCL_FORCE_INLINE auto front() noexcept -> reference { return data_[0]; }
    [[nodiscard]] SCL_FORCE_INLINE auto front() const noexcept -> const_reference { return data_[0]; }
    [[nodiscard]] SCL_FORCE_INLINE auto back() noexcept -> reference { return data_[size_ - 1]; }
    [[nodiscard]] SCL_FORCE_INLINE auto back() const noexcept -> const_reference { return data_[size_ - 1]; }
    [[nodiscard]] SCL_FORCE_INLINE auto data() noexcept -> pointer { return data_; }
    [[nodiscard]] SCL_FORCE_INLINE auto data() const noexcept -> const_pointer { return data_; }

    // -------------------------------------------------------------------------
    // Capacity (STL-compatible)
    // -------------------------------------------------------------------------

    [[nodiscard]] SCL_FORCE_INLINE auto size() const noexcept -> size_type { return size_; }
    [[nodiscard]] SCL_FORCE_INLINE auto size_bytes() const noexcept -> std::size_t {
        return static_cast<std::size_t>(size_) * sizeof(T);
    }
    [[nodiscard]] SCL_FORCE_INLINE auto empty() const noexcept -> bool { return size_ == 0 || data_ == nullptr; }
    [[nodiscard]] SCL_FORCE_INLINE explicit operator bool() const noexcept { return data_ != nullptr && size_ > 0; }

    // -------------------------------------------------------------------------
    // Iterators (STL-compatible)
    // -------------------------------------------------------------------------

    [[nodiscard]] SCL_FORCE_INLINE auto begin() noexcept -> iterator { return data_; }
    [[nodiscard]] SCL_FORCE_INLINE auto end() noexcept -> iterator { return data_ + size_; }
    [[nodiscard]] SCL_FORCE_INLINE auto begin() const noexcept -> const_iterator { return data_; }
    [[nodiscard]] SCL_FORCE_INLINE auto end() const noexcept -> const_iterator { return data_ + size_; }
    [[nodiscard]] SCL_FORCE_INLINE auto cbegin() const noexcept -> const_iterator { return data_; }
    [[nodiscard]] SCL_FORCE_INLINE auto cend() const noexcept -> const_iterator { return data_ + size_; }

    // -------------------------------------------------------------------------
    // Ownership Queries
    // -------------------------------------------------------------------------

    /// @brief Check if span owns its data exclusively (owned mode)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto is_owned() const noexcept -> bool {
        return detail::is_owned_sentinel(buffer_);
    }

    /// @brief Check if span shares data via buffer (shared mode)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto is_shared() const noexcept -> bool {
        return buffer_ != nullptr && !detail::is_owned_sentinel(buffer_);
    }

    /// @brief Check if span is a non-owning view
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto is_view() const noexcept -> bool {
        return buffer_ == nullptr;
    }

    /// @brief Check if this is the only reference
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto is_unique() const noexcept -> bool {
        if (is_shared()) return buffer_->is_unique();
        return is_owned();  // Owned is always unique
    }

    /// @brief Get reference count
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto use_count() const noexcept -> std::uint32_t {
        if (is_shared()) return buffer_->use_count();
        return is_owned() ? 1 : 0;
    }

    /// @brief Get underlying buffer (nullptr if not shared, owned sentinel if owned)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto buffer() const noexcept -> SharedBuffer* {
        return is_shared() ? buffer_ : nullptr;
    }

    // -------------------------------------------------------------------------
    // Ownership Conversion
    // -------------------------------------------------------------------------

    /// @brief Convert owned span to shared mode
    /// @return New SharedSpan in shared mode, original becomes empty
    /// @note Only valid for owned mode; shared/view modes return moved-from self
    [[nodiscard]]
    auto to_shared() && -> SharedSpan {
        if (!is_owned()) {
            // Already shared or view - just move
            return std::move(*this);
        }

        // Create SharedBuffer wrapping our data
        auto* new_buffer = SharedBuffer::from_external(
            data_, size_bytes(),
            [](void* p) noexcept { delete[] static_cast<T*>(p); }
        );

        if (!new_buffer) [[unlikely]] {
            // Failed to create buffer - clean up and return empty
            delete[] data_;
            buffer_ = nullptr;
            data_ = nullptr;
            size_ = 0;
            return {};
        }

        SharedSpan result;
        result.buffer_ = new_buffer;  // Adopt (ref_count = 1)
        result.data_ = data_;
        result.size_ = size_;

        // Clear this span (now shared mode, don't delete data_)
        buffer_ = nullptr;
        data_ = nullptr;
        size_ = 0;

        return result;
    }

    // -------------------------------------------------------------------------
    // Subspan Operations
    // -------------------------------------------------------------------------

    /// @brief Create subspan
    /// @param[in] offset Element offset from start
    /// @param[in] count Number of elements (npos = rest of span)
    /// @return New SharedSpan
    ///
    /// ## Safety
    ///
    ///   | Original Mode | Result Mode | Safety          |
    ///   |---------------|-------------|-----------------|
    ///   | Shared        | Shared      | ✓ Safe          |
    ///   | Owned         | View        | ⚠ Dangling risk |
    ///   | View          | View        | Same as original |
    ///
    /// @warning For Owned spans, use std::move(*this).to_shared().subspan(...)
    [[nodiscard]]
    auto subspan(Size offset, Size count = npos) const noexcept -> SharedSpan {
        if (offset > size_) [[unlikely]] {
            return {};
        }

        const Size actual = (count == npos) ? (size_ - offset) : count;
        if (offset + actual > size_) [[unlikely]] {
            return {};
        }

        SharedSpan result;
        result.data_ = data_ + offset;
        result.size_ = actual;

        if (is_shared()) {
            result.buffer_ = buffer_;
            buffer_->incref();
        }
        // Owned/View → View (buffer_ = nullptr)

        return result;
    }

    /// @brief Get first N elements
    [[nodiscard]]
    auto first(Size count) const noexcept -> SharedSpan {
        return subspan(0, count);
    }

    /// @brief Get last N elements
    [[nodiscard]]
    auto last(Size count) const noexcept -> SharedSpan {
        if (count > size_) [[unlikely]] return {};
        return subspan(size_ - count, count);
    }

    // -------------------------------------------------------------------------
    // Byte Offset (for shared mode)
    // -------------------------------------------------------------------------

    /// @brief Get byte offset from buffer start
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto offset_bytes() const noexcept -> std::ptrdiff_t {
        if (!is_shared() || !data_) return 0;
        return reinterpret_cast<const char*>(data_) -
               static_cast<const char*>(buffer_->data());
    }

    /// @brief Get element offset from buffer start
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto offset() const noexcept -> Size {
        return static_cast<Size>(offset_bytes() / static_cast<std::ptrdiff_t>(sizeof(T)));
    }

    // -------------------------------------------------------------------------
    // Utilities
    // -------------------------------------------------------------------------

    /// @brief Swap with another span
    auto swap(SharedSpan& other) noexcept -> void {
        std::swap(buffer_, other.buffer_);
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

    /// @brief Reset to empty state
    auto reset() noexcept -> void {
        release_resources();
        buffer_ = nullptr;
        data_ = nullptr;
        size_ = 0;
    }

    /// @brief Release ownership and return pointer (owned mode only)
    /// @return Data pointer (caller takes ownership), or nullptr if not owned
    [[nodiscard]]
    auto release_ownership() noexcept -> pointer {
        if (!is_owned()) return nullptr;

        auto* ptr = data_;
        buffer_ = nullptr;
        data_ = nullptr;
        size_ = 0;
        return ptr;
    }

    /// @brief Create deep copy in owned mode
    [[nodiscard]]
    auto clone() const -> SharedSpan {
        if (!data_ || size_ <= 0) return {};

        auto result = create_owned(size_);
        if (result) {
            std::copy(data_, data_ + size_, result.data_);
        }
        return result;
    }

    /// @brief Fill with value
    auto fill(const T& value) -> void {
        std::fill(begin(), end(), value);
    }

    // -------------------------------------------------------------------------
    // Comparison
    // -------------------------------------------------------------------------

    [[nodiscard]] auto same_data(const SharedSpan& other) const noexcept -> bool {
        return data_ == other.data_;
    }

    [[nodiscard]] auto same_buffer(const SharedSpan& other) const noexcept -> bool {
        return is_shared() && other.is_shared() && buffer_ == other.buffer_;
    }

    // -------------------------------------------------------------------------
    // STL Compatibility
    // -------------------------------------------------------------------------

    /// @brief Convert to std::span
    [[nodiscard]]
    auto to_std_span() noexcept -> std::span<T> {
        return std::span<T>{data_, static_cast<std::size_t>(size_)};
    }

    /// @brief Convert to const std::span
    [[nodiscard]]
    auto to_std_span() const noexcept -> std::span<const T> {
        return std::span<const T>{data_, static_cast<std::size_t>(size_)};
    }

private:
    auto release_resources() noexcept -> void {
        if (detail::is_owned_sentinel(buffer_)) {
            // Owned mode: delete the data
            delete[] data_;
        } else if (buffer_) {
            // Shared mode: decref the buffer
            buffer_->decref();
        }
        // View mode: do nothing
    }

    SharedBuffer* buffer_ = nullptr;  ///< nullptr=view, sentinel=owned, else=shared
    T* data_ = nullptr;               ///< Pointer to data
    Size size_ = 0;                   ///< Element count
};

// =============================================================================
// SECTION 6: Static Assertions for Memory Layout
// =============================================================================

static_assert(sizeof(SharedSpan<Real>) == 24,
              "SharedSpan should be exactly 24 bytes (3 pointers)");
static_assert(std::is_nothrow_move_constructible_v<SharedSpan<Real>>,
              "SharedSpan must be nothrow move constructible");
static_assert(std::is_nothrow_move_assignable_v<SharedSpan<Real>>,
              "SharedSpan must be nothrow move assignable");

// =============================================================================
// SECTION 7: Multi-Span Buffer Utilities
// =============================================================================

/// @brief Create multiple spans sharing a single buffer
///
/// @tparam T Element type
/// @tparam N Number of spans
/// @param[in] sizes Array of sizes for each span
/// @param[in] alignment Buffer alignment
/// @return Tuple of (SharedBuffer*, array of SharedSpan<T>)
///
/// ## Example
///
/// ```cpp
/// auto [buffer, spans] = make_shared_spans<Real, 3>({100, 200, 150});
/// auto& col0 = spans[0];  // 100 elements at offset 0
/// auto& col1 = spans[1];  // 200 elements at offset 100*sizeof(Real)
/// auto& col2 = spans[2];  // 150 elements at offset 300*sizeof(Real)
/// ```
template<typename T, std::size_t N>
[[nodiscard]]
auto make_shared_spans(
    const std::array<Size, N>& sizes,
    std::size_t alignment = memory::DEFAULT_ALIGNMENT
) -> std::pair<SharedBuffer*, std::array<SharedSpan<T>, N>> {
    // Calculate total size
    Size total = 0;
    for (auto s : sizes) {
        total += s;
    }

    if (total == 0) [[unlikely]] {
        return {nullptr, {}};
    }

    // Create shared buffer
    auto* buffer = SharedBuffer::create_aligned(
        static_cast<std::size_t>(total) * sizeof(T), alignment);
    if (!buffer) [[unlikely]] {
        return {nullptr, {}};
    }

    // Create spans with offsets
    std::array<SharedSpan<T>, N> spans;
    std::ptrdiff_t offset = 0;

    for (std::size_t i = 0; i < N; ++i) {
        if (i == 0) {
            // First span adopts the buffer (ref_count stays 1)
            spans[i] = SharedSpan<T>{buffer, offset, sizes[i], adopt};
        } else {
            // Subsequent spans share (incref)
            spans[i] = SharedSpan<T>{buffer, offset, sizes[i]};
        }
        offset += static_cast<std::ptrdiff_t>(sizes[i]) * sizeof(T);
    }

    return {buffer, std::move(spans)};
}

/// @brief Create multiple spans sharing a single buffer (variadic)
///
/// @tparam T Element type
/// @tparam Sizes Variadic size parameters
/// @param[in] sizes Size of each span
/// @return Tuple of SharedSpan<T>...
///
/// ## Example
///
/// ```cpp
/// auto [col0, col1, col2] = make_shared_spans<Real>(100, 200, 150);
/// ```
template<typename T, typename... Sizes>
requires (std::is_convertible_v<Sizes, Size> && ...)
[[nodiscard]]
auto make_shared_spans(Sizes... sizes)
    -> std::tuple<SharedSpan<T>, decltype((void(sizes), SharedSpan<T>{}))...> {
    constexpr std::size_t N = sizeof...(Sizes);
    std::array<Size, N> size_array = {static_cast<Size>(sizes)...};

    auto result = make_shared_spans<T, N>(size_array);
    auto& spans = result.second;

    return [&spans]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::make_tuple(std::move(spans[Is])...);
    }(std::make_index_sequence<N>{});
}

// =============================================================================
// SECTION 8: Type Aliases
// =============================================================================

using RealSpan = SharedSpan<Real>;
using IndexSpan = SharedSpan<Index>;
using FloatSpan = SharedSpan<float>;
using DoubleSpan = SharedSpan<double>;
using Int32Span = SharedSpan<std::int32_t>;
using Int64Span = SharedSpan<std::int64_t>;
using ByteSpan = SharedSpan<std::byte>;
using CharSpan = SharedSpan<char>;

// =============================================================================
// SECTION 9: Factory Functions
// =============================================================================

/// @brief Create owned span
template<typename T>
[[nodiscard]]
inline
auto make_span(Size count) -> SharedSpan<T> {
    return SharedSpan<T>::create_owned(count);
}

/// @brief Create aligned span (via SharedBuffer)
template<typename T>
[[nodiscard]]
inline
auto make_span_aligned(Size count, std::size_t alignment = memory::DEFAULT_ALIGNMENT)
    -> SharedSpan<T> {
    return SharedSpan<T>::create_aligned(count, alignment);
}

/// @brief Create shared span from buffer (incref)
template<typename T>
[[nodiscard]]
inline
auto make_span_shared(SharedBuffer* buffer, std::ptrdiff_t offset, Size count)
    -> SharedSpan<T> {
    return SharedSpan<T>::create_shared(buffer, offset, count);
}

/// @brief Create view span (non-owning)
template<typename T>
[[nodiscard]]
inline
auto make_span_view(T* data, Size count) -> SharedSpan<T> {
    return SharedSpan<T>::view(data, count);
}

/// @brief Create SharedBuffer with specified size
[[nodiscard]]
inline
auto make_buffer(std::size_t size) -> SharedBuffer* {
    return SharedBuffer::create(size);
}

/// @brief Create aligned SharedBuffer
[[nodiscard]]
inline
auto make_buffer_aligned(
    std::size_t size,
    std::size_t alignment = memory::DEFAULT_ALIGNMENT
) -> SharedBuffer* {
    return SharedBuffer::create_aligned(size, alignment);
}

}  // namespace scl
