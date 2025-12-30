#pragma once

/// @file scl/core/buffer.hpp
/// @brief Shared Buffer and Span with Reference-Counted Ownership
///
/// This header provides:
///   - SharedBuffer: Reference-counted raw memory block
///   - SharedSpan<T>: Typed array view with three ownership modes
///
/// Ownership modes for SharedSpan:
///   - **Owned**: buffer_==nullptr, owns_data_==true  -> Exclusive ownership
///   - **Shared**: buffer_!=nullptr                    -> Shared via SharedBuffer
///   - **View**: buffer_==nullptr, owns_data_==false  -> Non-owning view
///
/// @note Thread-safe reference counting using atomic operations
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
#include <type_traits>
#include <utility>

namespace scl {

// =============================================================================
// SECTION 1: Deleter Types
// =============================================================================

/// @brief Deleter function signature
using Deleter = void(*)(void*) noexcept;

/// @brief Built-in deleter functions
namespace deleter {

/// @brief Free memory with std::free (for posix_memalign)
inline
auto free_deleter(void* ptr) noexcept -> void {
    std::free(ptr);
}

/// @brief Delete array allocated with new[]
inline
auto array_deleter(void* ptr) noexcept -> void {
    delete[] static_cast<char*>(ptr);
}

/// @brief No-op deleter for non-owned memory
inline
auto noop(void*) noexcept -> void {}

#if SCL_PLATFORM_WINDOWS
/// @brief Windows aligned free
inline
auto aligned_deleter(void* ptr) noexcept -> void {
    _aligned_free(ptr);
}
#else
/// @brief POSIX aligned free (same as std::free)
inline
auto aligned_deleter(void* ptr) noexcept -> void {
    std::free(ptr);
}
#endif

}  // namespace deleter

// =============================================================================
// SECTION 2: SharedBuffer
// =============================================================================

/// @brief Reference-counted shared memory buffer
///
/// SharedBuffer manages a raw memory block with automatic cleanup when
/// the reference count reaches zero. Thread-safe via atomic operations.
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
    /// @return Pointer to SharedBuffer, or nullptr on failure
    [[nodiscard]]
    static
    auto create(std::size_t size) noexcept -> SharedBuffer* {
        if (size == 0) [[unlikely]] return nullptr;

        auto* data = new (std::nothrow) char[size]();
        if (!data) [[unlikely]] return nullptr;

        auto* buffer = new (std::nothrow) SharedBuffer{data, size, deleter::array_deleter};
        if (!buffer) [[unlikely]] {
            delete[] data;
            return nullptr;
        }

        return buffer;
    }

    /// @brief Create buffer with aligned allocation
    /// @param[in] size Size in bytes
    /// @param[in] alignment Alignment requirement (default: 64 bytes)
    /// @return Pointer to SharedBuffer, or nullptr on failure
    [[nodiscard]]
    static
    auto create_aligned(
        std::size_t size,
        std::size_t alignment = memory::DEFAULT_ALIGNMENT
    ) noexcept -> SharedBuffer* {
        if (size == 0) [[unlikely]] return nullptr;

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

        auto* buffer = new (std::nothrow) SharedBuffer{data, size, deleter::aligned_deleter};
        if (!buffer) [[unlikely]] {
            deleter::aligned_deleter(data);
            return nullptr;
        }

        return buffer;
    }

    /// @brief Wrap external memory
    /// @param[in] data Pointer to external data
    /// @param[in] size Size in bytes
    /// @param[in] del Deleter function (use deleter::noop for non-owning)
    /// @return Pointer to SharedBuffer, or nullptr on failure
    [[nodiscard]]
    static
    auto from_external(
        void* data,
        std::size_t size,
        Deleter del = deleter::noop
    ) noexcept -> SharedBuffer* {
        if (!data || size == 0) [[unlikely]] return nullptr;
        return new (std::nothrow) SharedBuffer{data, size, del};
    }

    // -------------------------------------------------------------------------
    // Reference Counting
    // -------------------------------------------------------------------------

    /// @brief Increment reference count
    /// @param[in] n Number of references to add
    /// @return New reference count
    SCL_FORCE_INLINE
    auto incref(std::uint32_t n = 1) noexcept -> std::uint32_t {
        return refcount_.fetch_add(n, std::memory_order_relaxed) + n;
    }

    /// @brief Decrement reference count
    /// @param[in] n Number of references to remove
    /// @return true if buffer was released
    /// @warning Do not access buffer after this returns true
    SCL_FORCE_INLINE
    auto decref(std::uint32_t n = 1) noexcept -> bool {
        const auto old = refcount_.fetch_sub(n, std::memory_order_acq_rel);
        SCL_DEBUG_ASSERT(old >= n);
        if (old == n) {
            release();
            return true;
        }
        return false;
    }

    /// @brief Get current reference count
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto use_count() const noexcept -> std::uint32_t {
        return refcount_.load(std::memory_order_relaxed);
    }

    /// @brief Check if this is the only reference
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto is_unique() const noexcept -> bool {
        return use_count() == 1;
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    /// @brief Get raw pointer to data
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto data() noexcept -> void* { return data_; }

    /// @brief Get const raw pointer to data
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto data() const noexcept -> const void* { return data_; }

    /// @brief Get typed pointer to data
    template<typename T>
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto data_as() noexcept -> T* { return static_cast<T*>(data_); }

    /// @brief Get typed const pointer to data
    template<typename T>
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto data_as() const noexcept -> const T* { return static_cast<const T*>(data_); }

    /// @brief Get buffer size in bytes
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto size() const noexcept -> std::size_t { return size_; }

    /// @brief Get element count for type T
    template<typename T>
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto count() const noexcept -> Size {
        return static_cast<Size>(size_ / sizeof(T));
    }

    /// @brief Check if buffer is valid
    [[nodiscard]]
    SCL_FORCE_INLINE
    explicit operator bool() const noexcept { return data_ != nullptr; }

    // -------------------------------------------------------------------------
    // Non-copyable, Non-movable
    // -------------------------------------------------------------------------

    SharedBuffer(const SharedBuffer&) = delete;
    auto operator=(const SharedBuffer&) -> SharedBuffer& = delete;
    SharedBuffer(SharedBuffer&&) = delete;
    auto operator=(SharedBuffer&&) -> SharedBuffer& = delete;

private:
    SharedBuffer(void* data, std::size_t size, Deleter del) noexcept
        : data_(data), size_(size), deleter_(del), refcount_(1) {}

    ~SharedBuffer() = default;

    auto release() noexcept -> void {
        if (deleter_ && data_) {
            deleter_(data_);
        }
        data_ = nullptr;
        delete this;
    }

    void* data_;
    std::size_t size_;
    Deleter deleter_;
    std::atomic<std::uint32_t> refcount_;
};

// =============================================================================
// SECTION 3: SharedSpan
// =============================================================================

/// @brief Typed array view with shared or exclusive ownership
///
/// SharedSpan provides three ownership modes:
///   - **Owned**: Exclusive ownership, self-managed memory
///   - **Shared**: Shared ownership via SharedBuffer
///   - **View**: Non-owning view of external memory
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

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    /// @brief Default: empty span
    constexpr
    SharedSpan() noexcept = default;

    /// @brief Construct from SharedBuffer (shared mode)
    /// @param[in] buffer Shared buffer (reference count incremented)
    /// @param[in] offset Byte offset from buffer start
    /// @param[in] count Number of elements
    SharedSpan(SharedBuffer* buffer, std::ptrdiff_t offset, Size count) noexcept
        : buffer_(buffer)
        , data_(buffer ? reinterpret_cast<T*>(
              static_cast<char*>(buffer->data()) + offset) : nullptr)
        , size_(count) {
        if (buffer_) {
            buffer_->incref();
        }
    }

    /// @brief Construct from SharedBuffer at offset 0
    SharedSpan(SharedBuffer* buffer, Size count) noexcept
        : SharedSpan(buffer, 0, count) {}

    // -------------------------------------------------------------------------
    // Factory Methods
    // -------------------------------------------------------------------------

    /// @brief Create owned span with new allocation
    /// @param[in] count Number of elements
    /// @return SharedSpan in owned mode, or empty on failure
    [[nodiscard]]
    static
    auto create_owned(Size count) noexcept -> SharedSpan {
        if (count <= 0) [[unlikely]] return {};

        auto* data = new (std::nothrow) T[static_cast<std::size_t>(count)]();
        if (!data) [[unlikely]] return {};

        SharedSpan span;
        span.data_ = data;
        span.size_ = count;
        span.owns_data_ = true;
        return span;
    }

    /// @brief Create owned span with aligned allocation
    /// @param[in] count Number of elements
    /// @param[in] alignment Byte alignment
    /// @return SharedSpan in owned mode, or empty on failure
    [[nodiscard]]
    static
    auto create_aligned(
        Size count,
        std::size_t alignment = memory::DEFAULT_ALIGNMENT
    ) noexcept -> SharedSpan {
        if (count <= 0) [[unlikely]] return {};

        // Use SharedBuffer for aligned allocations to handle cleanup
        auto* buffer = SharedBuffer::create_aligned(
            static_cast<std::size_t>(count) * sizeof(T), alignment);
        if (!buffer) [[unlikely]] return {};

        SharedSpan span;
        span.buffer_ = buffer;
        span.data_ = buffer->data_as<T>();
        span.size_ = count;
        span.owns_data_ = false;  // SharedBuffer handles cleanup
        return span;
    }

    /// @brief Create shared span from buffer
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
        span.owns_data_ = false;
        return span;
    }

    // -------------------------------------------------------------------------
    // Copy/Move Semantics
    // -------------------------------------------------------------------------

    /// @brief Copy constructor
    SharedSpan(const SharedSpan& other)
        : buffer_(other.buffer_)
        , size_(other.size_) {
        if (buffer_) {
            // Shared mode: share the buffer
            buffer_->incref();
            data_ = other.data_;
        } else if (other.owns_data_ && other.data_ && other.size_ > 0) {
            // Owned mode: deep copy
            data_ = new (std::nothrow) T[static_cast<std::size_t>(size_)];
            if (data_) {
                std::copy(other.data_, other.data_ + size_, data_);
                owns_data_ = true;
            } else {
                size_ = 0;
            }
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
        , size_(std::exchange(other.size_, 0))
        , owns_data_(std::exchange(other.owns_data_, false)) {}

    /// @brief Move assignment
    auto operator=(SharedSpan&& other) noexcept -> SharedSpan& {
        if (this != &other) {
            release();
            buffer_ = std::exchange(other.buffer_, nullptr);
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
            owns_data_ = std::exchange(other.owns_data_, false);
        }
        return *this;
    }

    /// @brief Destructor
    ~SharedSpan() {
        release();
    }

    // -------------------------------------------------------------------------
    // Element Access (STL-compatible)
    // -------------------------------------------------------------------------

    /// @brief Access element with bounds checking
    /// @param[in] idx Element index
    /// @return Reference to element
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
    /// @param[in] idx Element index
    /// @return Reference to element
    /// @warning No bounds checking - undefined behavior if out of bounds
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

    /// @brief Access first element
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto front() noexcept -> reference { return data_[0]; }

    /// @brief Access first element (const)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto front() const noexcept -> const_reference { return data_[0]; }

    /// @brief Access last element
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto back() noexcept -> reference { return data_[size_ - 1]; }

    /// @brief Access last element (const)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto back() const noexcept -> const_reference { return data_[size_ - 1]; }

    /// @brief Get pointer to data
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto data() noexcept -> pointer { return data_; }

    /// @brief Get const pointer to data
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto data() const noexcept -> const_pointer { return data_; }

    // -------------------------------------------------------------------------
    // Capacity (STL-compatible)
    // -------------------------------------------------------------------------

    /// @brief Get number of elements
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto size() const noexcept -> size_type { return size_; }

    /// @brief Get size in bytes
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto size_bytes() const noexcept -> std::size_t {
        return static_cast<std::size_t>(size_) * sizeof(T);
    }

    /// @brief Check if span is empty
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto empty() const noexcept -> bool { return size_ == 0 || data_ == nullptr; }

    /// @brief Check if span is valid
    [[nodiscard]]
    SCL_FORCE_INLINE
    explicit operator bool() const noexcept { return data_ != nullptr && size_ > 0; }

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

    /// @brief Check if span owns its data (owned mode)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto owns_data() const noexcept -> bool {
        return owns_data_ && buffer_ == nullptr;
    }

    /// @brief Check if span shares data via buffer (shared mode)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto is_shared() const noexcept -> bool {
        return buffer_ != nullptr;
    }

    /// @brief Check if span is a non-owning view
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto is_view() const noexcept -> bool {
        return buffer_ == nullptr && !owns_data_;
    }

    /// @brief Check if this is the only reference
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto is_unique() const noexcept -> bool {
        if (buffer_) return buffer_->is_unique();
        return owns_data_;
    }

    /// @brief Get reference count
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto use_count() const noexcept -> std::uint32_t {
        if (buffer_) return buffer_->use_count();
        return owns_data_ ? 1 : 0;
    }

    /// @brief Get underlying buffer (nullptr if not shared)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto buffer() const noexcept -> SharedBuffer* { return buffer_; }

    // -------------------------------------------------------------------------
    // Subspan Operations
    // -------------------------------------------------------------------------

    /// @brief Sentinel value for "rest of span"
    static constexpr Size npos = static_cast<Size>(-1);

    /// @brief Create subspan
    /// @param[in] offset Element offset
    /// @param[in] count Number of elements (npos for rest)
    /// @return New SharedSpan (shares buffer if shared mode, otherwise view)
    [[nodiscard]]
    auto subspan(Size offset, Size count = npos) const noexcept -> SharedSpan {
        SCL_DEBUG_ASSERT(offset <= size_);

        const Size actual = (count == npos) ? (size_ - offset) : count;
        SCL_DEBUG_ASSERT(offset + actual <= size_);

        SharedSpan result;
        result.data_ = data_ + offset;
        result.size_ = actual;

        if (buffer_) {
            result.buffer_ = buffer_;
            buffer_->incref();
        }
        // Subspan of owned/view becomes view

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
        return subspan(size_ - count, count);
    }

    // -------------------------------------------------------------------------
    // Byte Offset (for shared mode)
    // -------------------------------------------------------------------------

    /// @brief Get byte offset from buffer start
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto offset_bytes() const noexcept -> std::ptrdiff_t {
        if (!buffer_ || !data_) return 0;
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
        std::swap(owns_data_, other.owns_data_);
    }

    /// @brief Reset to empty state
    auto reset() noexcept -> void {
        release();
        buffer_ = nullptr;
        data_ = nullptr;
        size_ = 0;
        owns_data_ = false;
    }

    /// @brief Release ownership and return pointer
    /// @return Data pointer (caller takes ownership if owned mode)
    /// @warning Only valid for owned mode
    [[nodiscard]]
    auto release_ownership() noexcept -> pointer {
        if (!owns_data_) return nullptr;

        auto* ptr = data_;
        data_ = nullptr;
        size_ = 0;
        owns_data_ = false;
        return ptr;
    }

    /// @brief Create deep copy
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

    /// @brief Check if pointing to same data
    [[nodiscard]]
    auto same_data(const SharedSpan& other) const noexcept -> bool {
        return data_ == other.data_;
    }

    /// @brief Check if sharing same buffer
    [[nodiscard]]
    auto same_buffer(const SharedSpan& other) const noexcept -> bool {
        return buffer_ != nullptr && buffer_ == other.buffer_;
    }

private:
    auto release() noexcept -> void {
        if (buffer_) {
            buffer_->decref();
        } else if (owns_data_ && data_) {
            delete[] data_;
        }
    }

    SharedBuffer* buffer_ = nullptr;
    T* data_ = nullptr;
    Size size_ = 0;
    bool owns_data_ = false;
};

// =============================================================================
// SECTION 4: Type Aliases
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
// SECTION 5: Factory Functions
// =============================================================================

/// @brief Create owned span
/// @tparam T Element type
/// @param[in] count Number of elements
/// @return SharedSpan in owned mode
template<typename T>
[[nodiscard]]
inline
auto make_span(Size count) -> SharedSpan<T> {
    return SharedSpan<T>::create_owned(count);
}

/// @brief Create aligned owned span
/// @tparam T Element type
/// @param[in] count Number of elements
/// @param[in] alignment Byte alignment
/// @return SharedSpan in owned mode
template<typename T>
[[nodiscard]]
inline
auto make_span_aligned(Size count, std::size_t alignment = memory::DEFAULT_ALIGNMENT)
    -> SharedSpan<T> {
    return SharedSpan<T>::create_aligned(count, alignment);
}

/// @brief Create shared span from buffer
/// @tparam T Element type
/// @param[in] buffer SharedBuffer pointer
/// @param[in] offset Byte offset
/// @param[in] count Number of elements
/// @return SharedSpan in shared mode
template<typename T>
[[nodiscard]]
inline
auto make_span_shared(SharedBuffer* buffer, std::ptrdiff_t offset, Size count)
    -> SharedSpan<T> {
    return SharedSpan<T>::create_shared(buffer, offset, count);
}

/// @brief Create view span (non-owning)
/// @tparam T Element type
/// @param[in] data Data pointer
/// @param[in] count Number of elements
/// @return SharedSpan in view mode
template<typename T>
[[nodiscard]]
inline
auto make_span_view(T* data, Size count) -> SharedSpan<T> {
    return SharedSpan<T>::view(data, count);
}

/// @brief Create SharedBuffer with specified size
/// @param[in] size Size in bytes
/// @return Pointer to SharedBuffer
[[nodiscard]]
inline
auto make_buffer(std::size_t size) -> SharedBuffer* {
    return SharedBuffer::create(size);
}

/// @brief Create aligned SharedBuffer
/// @param[in] size Size in bytes
/// @param[in] alignment Byte alignment
/// @return Pointer to SharedBuffer
[[nodiscard]]
inline
auto make_buffer_aligned(
    std::size_t size,
    std::size_t alignment = memory::DEFAULT_ALIGNMENT
) -> SharedBuffer* {
    return SharedBuffer::create_aligned(size, alignment);
}

// =============================================================================
// SECTION 6: Static Assertions
// =============================================================================

static_assert(std::is_nothrow_move_constructible_v<SharedSpan<Real>>,
              "SharedSpan must be nothrow move constructible");
static_assert(std::is_nothrow_move_assignable_v<SharedSpan<Real>>,
              "SharedSpan must be nothrow move assignable");

}  // namespace scl
