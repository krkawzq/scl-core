#pragma once

/**
 * @file scl/core/span.hpp
 * @brief Shared Buffer and Span with Reference-Counted Ownership
 *
 * This header provides:
 *   - Storage: Reference-counted raw memory block
 *   - Span<T>: Typed array view with three ownership modes
 *
 * ## Ownership Modes (Compact Layout)
 *
 *   | Mode   | storage_          | Description                          |
 *   |--------|-------------------|--------------------------------------|
 *   | Owned  | &OWNED_SENTINEL   | Exclusive ownership, delete[] data_  |
 *   | Shared | valid Storage     | Shared ownership via Storage         |
 *   | View   | nullptr           | Non-owning view                      |
 *
 * ## Memory Layout
 *
 *   Span<T> uses only 24 bytes (3 pointers):
 *     - storage_: Storage* or sentinel pointer
 *     - data_: T* data pointer
 *     - size_: Size element count
 *
 * ## Thread Safety
 *
 *   - Storage: Thread-safe reference counting (atomic operations)
 *   - Span: Thread-safe for read-only access when sharing
 *   - Concurrent writes to the same data require external synchronization
 *
 * @note Uses scl::memory module for allocations
 */

#include "scl/config.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macro.hpp"
#include "scl/core/type.hpp"
#include "scl/core/vectorize.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <numeric>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace scl {

// =============================================================================
// Configuration Constants
// =============================================================================

namespace detail {
/// @brief Default memory alignment (64 bytes for AVX-512)
inline constexpr std::size_t kDefaultAlignment = 64;

/// @brief Safe empty sentinel buffer size (1KB)
inline constexpr std::size_t kEmptySentinelSize = 1024;
}  // namespace detail

// =============================================================================
// SECTION 1: Forward Declarations and Tags
// =============================================================================

class Storage;

template <typename T>
class Span;

/// @brief Tag for adopting ownership without incrementing reference count
struct AdoptTag {
  explicit AdoptTag() = default;
};

/// @brief Global adopt tag instance
inline constexpr AdoptTag adopt{};

// =============================================================================
// SECTION 2: Allocator System
// =============================================================================

/// @brief Deallocator function signature (type-erased)
/// @param ptr Pointer to free
/// @param size Size in bytes (needed for mmap/munmap)
/// @param context Optional context (e.g., pool pointer)
using Deallocator = void (*)(void* ptr, std::size_t size, void* context) noexcept;

/// @brief Allocator concept
template<typename A>
concept StorageAllocator = requires(std::size_t size, std::size_t align) {
    { A::allocate(size, align) } -> std::same_as<void*>;
    { A::deallocate(static_cast<void*>(nullptr), size) } -> std::same_as<void>;
};

/// @brief Allocator type tags
struct AllocatorTag {
    enum class Type : std::uint8_t {
        Default,      ///< new char[]
        Aligned,      ///< posix_memalign / _aligned_malloc
        Virtual,      ///< mmap / VirtualAlloc
        HugePage,     ///< mmap + MAP_HUGETLB
        External,     ///< User-provided memory
    };
};

/// @brief Built-in allocators
namespace allocator {

/// @brief Default allocator using new/delete
struct Default {
    [[nodiscard]]
    static auto allocate(std::size_t size, std::size_t /*align*/ = 0) noexcept -> void* {
        return new (std::nothrow) char[size]();
    }
    
    static auto deallocate(void* ptr, std::size_t /*size*/) noexcept -> void {
        delete[] static_cast<char*>(ptr);
    }
};

/// @brief Aligned allocator
struct Aligned {
    [[nodiscard]]
    static auto allocate(std::size_t size, 
                         std::size_t alignment = detail::kDefaultAlignment) noexcept -> void* {
        if (alignment < sizeof(void*)) {
            alignment = sizeof(void*);
        }
        void* ptr = nullptr;
#if SCL_CONFIG_PLATFORM_WINDOWS
        ptr = _aligned_malloc(size, alignment);
#else
        if (::posix_memalign(&ptr, alignment, size) != 0) {
            ptr = nullptr;
        }
#endif
        if (ptr != nullptr) {
            std::memset(ptr, 0, size);
        }
        return ptr;
    }
    
    static auto deallocate(void* ptr, std::size_t /*size*/) noexcept -> void {
#if SCL_CONFIG_PLATFORM_WINDOWS
        _aligned_free(ptr);
#else
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
        std::free(ptr);
#endif
    }
};

/// @brief No-op allocator for external memory
struct External {
    [[nodiscard]]
    static auto allocate(std::size_t /*size*/, std::size_t /*align*/ = 0) noexcept -> void* {
        return nullptr;  // External doesn't allocate
    }
    
    static auto deallocate(void* /*ptr*/, std::size_t /*size*/) noexcept -> void {
        // No-op: external memory is not owned
    }
};

}  // namespace allocator

// =============================================================================
// SECTION 3: Storage (Type-Erased Allocator)
// =============================================================================

/// @brief Reference-counted shared memory buffer with type-erased deallocation
///
/// Storage provides:
///   - Automatic memory management via reference counting
///   - Type-erased deallocation (supports any allocator)
///   - Thread-safe reference counting
///   - Self-destruction when ref_count reaches zero
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
///   2. Span::adopt() takes ownership without incref
///   3. Span copy/share operations call incref()
///   4. Destruction calls decref()
///   5. When ref_count reaches 0, buffer self-destructs
///
/// @note Not copyable/movable - use incref/decref for ownership transfer
/// @note Self-destructs when reference count reaches zero
class Storage {
 public:
  // -------------------------------------------------------------------------
  // Factory Methods (Template for static dispatch, type-erased storage)
  // -------------------------------------------------------------------------

  /// @brief Create storage with specified allocator
  /// @tparam Alloc Allocator type satisfying StorageAllocator concept
  /// @param[in] size Size in bytes
  /// @param[in] alignment Alignment requirement
  /// @return Pointer to Storage (ref_count=1), or nullptr on failure
  template<StorageAllocator Alloc = allocator::Aligned>
  [[nodiscard]]
  static auto create(std::size_t size, 
                     std::size_t alignment = detail::kDefaultAlignment) noexcept 
      -> Storage* 
  {
    if (size == 0) [[unlikely]] {
      return nullptr;
    }
    
    void* data = Alloc::allocate(size, alignment);
    if (data == nullptr) [[unlikely]] {
      return nullptr;
    }
    
    auto* storage = new (std::nothrow) Storage{
        data, size, alignment,
        &Storage::deallocate_impl<Alloc>,  // Type-erased deallocator
        nullptr,                            // No context needed
        alloc_type_for<Alloc>()
    };
    
    if (storage == nullptr) [[unlikely]] {
      Alloc::deallocate(data, size);
      return nullptr;
    }
    
    return storage;
  }

  /// @brief Create buffer with unaligned allocation
  /// @param[in] size Size in bytes
  /// @return Pointer to Storage (ref_count=1), or nullptr on failure
  /// @note Uses new char[] for allocation
  [[nodiscard]]
  static auto create_default(std::size_t size) noexcept -> Storage* {
    return create<allocator::Default>(size, 1);
  }

  /// @brief Create buffer with aligned allocation
  /// @param[in] size Size in bytes
  /// @param[in] alignment Alignment requirement (must be power of 2, >=
  /// sizeof(void*))
  /// @return Pointer to Storage (ref_count=1), or nullptr on failure
  [[nodiscard]]
  static auto create_aligned(std::size_t size,
                             std::size_t alignment = detail::kDefaultAlignment) noexcept
      -> Storage* 
  {
    // Validate alignment
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) [[unlikely]] {
      return nullptr;  // Not power of 2
    }
    return create<allocator::Aligned>(size, alignment);
  }

  /// @brief Wrap external memory with custom deallocator
  /// @param[in] data Pointer to external data
  /// @param[in] size Size in bytes
  /// @param[in] deallocator Custom deallocator (nullptr for non-owning)
  /// @param[in] context Optional context passed to deallocator
  /// @return Pointer to Storage (ref_count=1), or nullptr on failure
  [[nodiscard]]
  static auto from_external(void* data, std::size_t size,
                            Deallocator deallocator = nullptr,
                            void* context = nullptr) noexcept -> Storage* 
  {
    if (data == nullptr || size == 0) [[unlikely]] {
      return nullptr;
    }
    
    return new (std::nothrow) Storage{
        data, size, 1,
        deallocator != nullptr ? deallocator : &noop_deallocator,
        context,
        AllocatorTag::Type::External
    };
  }
  
  /// @brief Wrap external array with typed deleter (convenience)
  /// @tparam T Element type
  /// @param[in] data Array pointer
  /// @param[in] count Number of elements
  /// @return Pointer to Storage (ref_count=1), or nullptr on failure
  template<typename T>
  [[nodiscard]]
  static auto from_array(T* data, std::size_t count) noexcept -> Storage* {
    return from_external(
        data, count * sizeof(T),
        [](void* p, std::size_t, void*) noexcept { 
          delete[] static_cast<T*>(p); 
        },
        nullptr
    );
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
    // Debug check for underflow
#ifndef NDEBUG
    if (old < n) {
      detail::debug_assert_fail("Storage refcount underflow",
                                source_location::current());
    }
#endif
    if (old == n) {
      release_impl();
      return true;
    }
    return false;
  }

  /// @brief Get current reference count
  /// @return Approximate reference count (may be stale in concurrent scenarios)
  /// @warning Do not use for synchronization; use is_unique() for ownership
  /// checks
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

  [[nodiscard]] SCL_FORCE_INLINE auto data() noexcept -> void* {
    return data_;
  }
  [[nodiscard]] SCL_FORCE_INLINE auto data() const noexcept -> const void* {
    return data_;
  }

  template <typename T>
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto data_as() noexcept -> T* {
    return static_cast<T*>(data_);
  }

  template <typename T>
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto data_as() const noexcept -> const T* {
    return static_cast<const T*>(data_);
  }

  [[nodiscard]] SCL_FORCE_INLINE auto size() const noexcept -> std::size_t {
    return size_;
  }
  [[nodiscard]] SCL_FORCE_INLINE auto alignment() const noexcept
      -> std::size_t {
    return alignment_;
  }
  [[nodiscard]] SCL_FORCE_INLINE auto alloc_type() const noexcept 
      -> AllocatorTag::Type {
    return alloc_type_;
  }

  template <typename T>
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto count() const noexcept -> Size {
    return static_cast<Size>(size_ / sizeof(T));
  }

  template <typename T>
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto is_aligned_for() const noexcept -> bool {
    return alignment_ >= alignof(T) &&
           (reinterpret_cast<std::uintptr_t>(data_) % alignof(T)) == 0;
  }

  [[nodiscard]] SCL_FORCE_INLINE explicit operator bool() const noexcept {
    return data_ != nullptr;
  }
  
  /// @brief Check if storage is externally managed
  [[nodiscard]] auto is_external() const noexcept -> bool {
    return alloc_type_ == AllocatorTag::Type::External;
  }

  // -------------------------------------------------------------------------
  // Non-copyable, Non-movable
  // -------------------------------------------------------------------------

  Storage(const Storage&) = delete;
  auto operator=(const Storage&) -> Storage& = delete;
  Storage(Storage&&) = delete;
  auto operator=(Storage&&) -> Storage& = delete;

 private:
  Storage(void* data, std::size_t size, std::size_t alignment,
          Deallocator deallocator, void* context, 
          AllocatorTag::Type alloc_type) noexcept
      : data_(data), 
        size_(size), 
        alignment_(alignment),
        deallocator_(deallocator), 
        context_(context),
        alloc_type_(alloc_type), 
        refcount_(1) {}

  ~Storage() = default;

  auto release_impl() noexcept -> void {
    if (deallocator_ != nullptr && data_ != nullptr) {
      deallocator_(data_, size_, context_);
    }
    data_ = nullptr;
    delete this;
  }
  
  /// @brief Type-erased deallocator wrapper
  template<StorageAllocator Alloc>
  static auto deallocate_impl(void* ptr, std::size_t size, void* /*ctx*/) noexcept -> void {
    Alloc::deallocate(ptr, size);
  }
  
  static auto noop_deallocator(void* /*ptr*/, std::size_t /*size*/, void* /*ctx*/) noexcept -> void {}
  
  template<StorageAllocator Alloc>
  static constexpr auto alloc_type_for() noexcept -> AllocatorTag::Type {
    if constexpr (std::is_same_v<Alloc, allocator::Default>) {
      return AllocatorTag::Type::Default;
    } else if constexpr (std::is_same_v<Alloc, allocator::Aligned>) {
      return AllocatorTag::Type::Aligned;
    } else {
      return AllocatorTag::Type::External;
    }
  }

  void* data_;
  std::size_t size_;
  std::size_t alignment_;
  Deallocator deallocator_;
  void* context_;
  AllocatorTag::Type alloc_type_;
  std::atomic<std::uint32_t> refcount_;
};

// =============================================================================
// SECTION 4: Ownership Sentinel
// =============================================================================

namespace detail {

/// @brief Sentinel object to mark owned spans
///
/// When storage_ points to this sentinel, the span is in owned mode
/// and data_ should be freed with delete[].
///
/// This allows Span to use only 3 members (24 bytes) instead of 4 (40 bytes).
struct OwnedSentinel {
  // Storage with same alignment as Storage, used only for its address
  alignas(Storage) std::array<std::byte, sizeof(Storage)> storage{};
};

/// @brief Global sentinel instance for owned spans
/// @note Only the address is used; the object is never dereferenced
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
inline OwnedSentinel OWNED_SENTINEL_INSTANCE;

/// @brief Get pointer to owned sentinel (as Storage*)
/// @note The sentinel is never dereferenced, only compared by address
[[nodiscard]]
SCL_FORCE_INLINE
auto owned_sentinel() noexcept -> Storage* {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return reinterpret_cast<Storage*>(&OWNED_SENTINEL_INSTANCE);
}

/// @brief Check if a buffer pointer is the owned sentinel
[[nodiscard]]
SCL_FORCE_INLINE
auto is_owned_sentinel(const Storage* ptr) noexcept -> bool {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return ptr == reinterpret_cast<const Storage*>(&OWNED_SENTINEL_INSTANCE);
}

// -----------------------------------------------------------------------------
// Empty Sentinel (Safe Pointer for Empty Spans)
// -----------------------------------------------------------------------------

/// @brief Safe empty sentinel for empty spans (zero-initialized 1KB buffer)
/// @note This prevents undefined behavior from nullptr dereference
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
alignas(kDefaultAlignment) inline const char EMPTY_SENTINEL_BUFFER[kEmptySentinelSize] = {};

/// @brief Get pointer to empty sentinel
/// @tparam T Element type
/// @return Pointer to safe empty sentinel (typed)
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
auto empty_sentinel() noexcept -> T* {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-type-const-cast)
  return const_cast<T*>(reinterpret_cast<const T*>(EMPTY_SENTINEL_BUFFER));
}

/// @brief Check if pointer points to empty sentinel
/// @tparam T Element type
/// @param[in] ptr Pointer to check
/// @return true if pointer is within empty sentinel buffer
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
auto is_empty_sentinel(const T* ptr) noexcept -> bool {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
  const char* sentinel_start = EMPTY_SENTINEL_BUFFER;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-array-to-pointer-decay)
  const char* sentinel_end = EMPTY_SENTINEL_BUFFER + kEmptySentinelSize;
  const void* ptr_void = static_cast<const void*>(ptr);
  return ptr_void >= sentinel_start && ptr_void < sentinel_end;
}

}  // namespace detail

// =============================================================================
// SECTION 5: Span (Compact 24-byte Layout)
// =============================================================================

/// @brief Typed array view with shared or exclusive ownership
///
/// Span provides three ownership modes with a compact 24-byte layout:
///
///   | Mode   | storage_          | data_  | Cleanup               |
///   |--------|-------------------|--------|-----------------------|
///   | Owned  | &OWNED_SENTINEL   | valid  | delete[] data_        |
///   | Shared | valid Storage     | valid  | storage_->decref()    |
///   | View   | nullptr           | valid  | nothing               |
///
/// ## Memory Layout (24 bytes)
///
///   - storage_: 8 bytes (Storage* or sentinel)
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
template <typename T>
class Span {
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
  /// @brief Default constructor creates empty span (points to safe sentinel)
  /// @note data_ points to safe sentinel buffer, not nullptr
  constexpr Span() noexcept 
      : data_(detail::empty_sentinel<T>()) {}

  /// @brief Construct from Storage with sharing semantics (incref)
  /// @param[in] storage Shared storage (reference count incremented)
  /// @param[in] offset Byte offset from storage start
  /// @param[in] count Number of elements
  Span(Storage* storage, std::ptrdiff_t offset, Size count) noexcept
      : storage_(storage),
        data_(storage != nullptr
                  ? reinterpret_cast<T*>(static_cast<char*>(storage->data()) +
                                         offset)
                  : nullptr),
        size_(count) {
    if (storage_ != nullptr && !detail::is_owned_sentinel(storage_)) {
      storage_->incref();
    }
  }

  /// @brief Construct from Storage at offset 0 with sharing semantics
  Span(Storage* storage, Size count) noexcept : Span(storage, 0, count) {}

  /// @brief Adopt ownership of Storage without incref
  /// @param[in] storage Shared storage (ownership transferred, no incref)
  /// @param[in] offset Byte offset from storage start
  /// @param[in] count Number of elements
  /// @param[in] tag AdoptTag to select this constructor
  Span(Storage* storage, std::ptrdiff_t offset, Size count, AdoptTag tag) noexcept  // NOLINT(bugprone-easily-swappable-parameters)
      : storage_(storage),
        data_(storage != nullptr
                  ? reinterpret_cast<T*>(static_cast<char*>(storage->data()) +
                                         offset)
                  : nullptr),
        size_(count) {
    (void)tag;  // Tag is only used for constructor overload resolution
    // No incref - adopting existing reference
  }

  // -------------------------------------------------------------------------
  // Factory Methods
  // -------------------------------------------------------------------------

  /// @brief Create owned span with new[] allocation
  /// @param[in] count Number of elements
  /// @return Span in owned mode, or empty on failure
  [[nodiscard]]
  static
  auto create_owned(Size count) noexcept -> Span {
    if (count <= 0) [[unlikely]] {
      return {};
    }

    auto* data = new (std::nothrow) T[static_cast<std::size_t>(count)]();
    if (data == nullptr) [[unlikely]] {
      return {};
    }

    Span span;
    span.storage_ = detail::owned_sentinel();  // Mark as owned
    span.data_ = data;
    span.size_ = count;
    return span;
  }

  /// @brief Create span with aligned allocation via Storage
  /// @param[in] count Number of elements
  /// @param[in] alignment Byte alignment (must be power of 2)
  /// @return Span in shared mode, or empty on failure
  [[nodiscard]]
  static
  auto create_aligned(Size count,
                      std::size_t alignment = detail::kDefaultAlignment) noexcept
      -> Span {
    (void)alignment;  // May be unused in some platform branches
    if (count <= 0) [[unlikely]] {
      return {};
    }

    auto* storage = Storage::create_aligned(
        static_cast<std::size_t>(count) * sizeof(T), alignment);
    if (storage == nullptr) [[unlikely]] {
      return {};
    }

    // Adopt the storage (ref_count is already 1)
    return Span{storage, 0, count, adopt};
  }

  /// @brief Create shared span from storage (incref)
  [[nodiscard]]
  static
  auto create_shared(Storage* storage, std::ptrdiff_t offset,
                     Size count) noexcept -> Span {
    return Span{storage, offset, count};
  }

  /// @brief Create non-owning view of raw pointer
  /// @param[in] data Raw pointer
  /// @param[in] count Number of elements
  /// @return Span that does NOT own data
  /// @warning Caller must ensure data outlives the span
  [[nodiscard]]
  static
  auto view(T* data, Size count) noexcept -> Span {
    Span span;
    span.data_ = data;
    span.size_ = count;
    // storage_ = nullptr means view mode
    return span;
  }

  /// @brief Create view from std::span
  template <std::size_t Extent>
  [[nodiscard]]
  static
  auto view(std::span<T, Extent> s) noexcept -> Span {
    return view(s.data(), static_cast<Size>(s.size()));
  }
  
  /// @brief Create from initializer list
  [[nodiscard]]
  static auto from_list(std::initializer_list<T> init) -> Span {
    auto span = create_owned(static_cast<Size>(init.size()));
    if (span) {
      std::copy(init.begin(), init.end(), span.begin());
    }
    return span;
  }
  
  /// @brief Create from range
  template<typename R>
    requires std::ranges::input_range<R> && 
             std::convertible_to<std::ranges::range_value_t<R>, T>
  [[nodiscard]]
  static auto from_range(R&& range) -> Span {
    auto size = static_cast<Size>(std::ranges::distance(range));
    auto span = create_owned(size);
    if (span) {
      std::ranges::copy(range, span.begin());
    }
    return span;
  }
  
  // -------------------------------------------------------------------------
  // Error-Checked Factory Methods (throws on failure)
  // -------------------------------------------------------------------------
  
  /// @brief Create owned span with error checking
  /// @param[in] count Number of elements
  /// @return Span in owned mode
  /// @throws std::bad_alloc if allocation fails
  /// @throws ValueError if count <= 0
  [[nodiscard]]
  static auto try_create_owned(Size count) -> Span {
    if (count <= 0) {
      throw ValueError("count must be positive", 
                      source_location::current());
    }
    auto span = create_owned(count);
    if (!span) {
      throw std::bad_alloc();
    }
    return span;
  }
  
  /// @brief Create aligned span with error checking
  /// @param[in] count Number of elements
  /// @param[in] alignment Byte alignment
  /// @return Span in shared mode
  /// @throws std::bad_alloc if allocation fails
  /// @throws ValueError if count <= 0 or invalid alignment
  [[nodiscard]]
  static auto try_create_aligned(Size count, 
                                 std::size_t alignment = detail::kDefaultAlignment) -> Span {
    if (count <= 0) {
      throw ValueError("count must be positive", 
                      source_location::current());
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
      throw ValueError("alignment must be power of 2", 
                      source_location::current());
    }
    auto span = create_aligned(count, alignment);
    if (!span) {
      throw std::bad_alloc();
    }
    return span;
  }

  // -------------------------------------------------------------------------
  // Copy/Move Semantics
  // -------------------------------------------------------------------------

  /// @brief Copy constructor
  /// @note Shared mode: incref storage
  /// @note Owned mode: deep copy (may fail silently → empty span)
  /// @note View mode: shallow copy
  Span(const Span& other) : storage_(other.storage_), size_(other.size_) {
    if (detail::is_owned_sentinel(storage_)) {
      // Owned mode: deep copy
      if (other.data_ != nullptr && other.size_ > 0) {
        auto* new_data = new (std::nothrow) T[static_cast<std::size_t>(size_)];
        if (new_data != nullptr) {
          std::copy(other.data_, other.data_ + size_, new_data);
          data_ = new_data;
        } else {
          // Allocation failed - become empty view
          storage_ = nullptr;
          size_ = 0;
        }
      }
    } else if (storage_ != nullptr) {
      // Shared mode: share the storage
      storage_->incref();
      data_ = other.data_;
    } else {
      // View mode: just copy pointer
      data_ = other.data_;
    }
  }

  /// @brief Copy assignment
  auto operator=(const Span& other) -> Span& {
    if (this != &other) {
      Span tmp{other};
      swap(tmp);
    }
    return *this;
  }

  /// @brief Move constructor
  Span(Span&& other) noexcept
      : storage_(std::exchange(other.storage_, nullptr)),
        data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}

  /// @brief Move assignment
  auto operator=(Span&& other) noexcept -> Span& {
    if (this != &other) {
      release_resources();
      storage_ = std::exchange(other.storage_, nullptr);
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  /// @brief Destructor
  ~Span() { release_resources(); }

  // -------------------------------------------------------------------------
  // Advanced Mutators (with safety checks)
  // -------------------------------------------------------------------------

  /// @brief Set new size (with bounds checking)
  /// @param[in] new_size New element count
  /// @throws ValueError if new_size > current capacity or invalid for storage
  auto set_size(Size new_size) -> void {
    // Check if new size is valid
    if (storage_ != nullptr && !detail::is_owned_sentinel(storage_)) {
      // Shared mode: check against storage bounds
      const std::size_t byte_offset = reinterpret_cast<const char*>(data_) -
                                      static_cast<const char*>(storage_->data());
      const std::size_t new_end_offset = byte_offset + new_size * sizeof(T);
      
      error::check_arg(new_end_offset <= storage_->size(),
                       "new size exceeds storage bounds");
    } else if (detail::is_owned_sentinel(storage_)) {
      // Owned mode: cannot grow, only shrink
      error::check_arg(new_size <= size_, "cannot grow owned span");
    }
    
    size_ = new_size;
  }

  /// @brief Adjust data pointer (with bounds checking)
  /// @param[in] offset Offset in elements (can be negative)
  /// @throws ValueError if new pointer is out of storage bounds
  auto adjust_data(std::ptrdiff_t offset) -> void {
    if (offset == 0) {
      return;
    }
    
    T* new_data = data_ + offset;
    
    // Check bounds for Shared mode
    if (storage_ != nullptr && !detail::is_owned_sentinel(storage_)) {
      const auto* storage_start = static_cast<const char*>(storage_->data());
      const auto* storage_end = storage_start + storage_->size();
      const auto* new_ptr = reinterpret_cast<const char*>(new_data);
      const auto* new_end = new_ptr + size_ * sizeof(T);
      
      error::check_arg(new_ptr >= storage_start && new_end <= storage_end,
                       "adjusted pointer out of storage bounds");
    } else if (detail::is_owned_sentinel(storage_)) {
      // Owned mode: check against original allocation (approximate)
      error::check_arg(offset >= 0, "cannot adjust owned span pointer backward");
    }
    
    data_ = new_data;
  }

  /// @brief Advanced: directly set data pointer and size (unsafe, for internal use)
  /// @param[in] new_data New data pointer
  /// @param[in] new_size New size
  /// @throws ValueError if pointer/size invalid for storage
  auto reset_unchecked(T* new_data, Size new_size) -> void {
    // Only allow for Shared/View modes
    if (storage_ != nullptr && !detail::is_owned_sentinel(storage_)) {
      const auto* storage_start = static_cast<const char*>(storage_->data());
      const auto* storage_end = storage_start + storage_->size();
      const auto* new_ptr = reinterpret_cast<const char*>(new_data);
      const auto* new_end = new_ptr + new_size * sizeof(T);
      
      error::check_arg(new_ptr >= storage_start && new_end <= storage_end,
                       "new range out of storage bounds");
    }
    
    data_ = new_data;
    size_ = new_size;
  }

  // -------------------------------------------------------------------------
  // Clone (Explicit Deep Copy)
  // -------------------------------------------------------------------------

  /// @brief Explicit deep copy (always allocates new memory)
  /// @return New owned span with copied data
  /// @note Always creates an Owned span regardless of source mode
  [[nodiscard]]
  auto clone() const -> Span {
    if (empty()) {
      return Span{};
    }

    auto* new_data = new (std::nothrow) T[static_cast<std::size_t>(size_)];
    if (new_data == nullptr) {
      return Span{};  // Allocation failed
    }

    // Deep copy data
    std::copy(data_, data_ + size_, new_data);

    // Create owned span
    Span result;
    result.storage_ = detail::owned_sentinel();
    result.data_ = new_data;
    result.size_ = size_;
    
    return result;
  }

  // -------------------------------------------------------------------------
  // Element Access (STL-compatible)
  // -------------------------------------------------------------------------

  /// @brief Access element without bounds checking (unchecked version)
  /// @param[in] idx Element index
  /// @return Reference to element at index
  /// @warning No bounds checking - caller must ensure idx < size()
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto get_unchecked(Size idx) noexcept -> reference {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data_[idx];
  }

  /// @brief Access element without bounds checking (const, unchecked version)
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto get_unchecked(Size idx) const noexcept -> const_reference {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data_[idx];
  }

  /// @brief Access element without bounds checking
  /// @note Delegates to unchecked version for maximum performance
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto operator[](Size idx) noexcept -> reference {
    return get_unchecked(idx);
  }

  /// @brief Access element without bounds checking (const)
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto operator[](Size idx) const noexcept -> const_reference {
    return get_unchecked(idx);
  }

  /// @brief Access element with bounds checking
  /// @throws IndexError if index out of bounds
  [[nodiscard]]
  auto at(Size idx) -> reference {
    error::check_arg(!empty(), "cannot access element of empty span");
    error::check_index(idx, size_);
    return get_unchecked(idx);
  }

  /// @brief Access element with bounds checking (const)
  [[nodiscard]]
  auto at(Size idx) const -> const_reference {
    error::check_arg(!empty(), "cannot access element of empty span");
    error::check_index(idx, size_);
    return get_unchecked(idx);
  }

  /// @brief Get first element (unchecked)
  /// @warning No bounds checking - caller must ensure !empty()
  [[nodiscard]] 
  SCL_FORCE_INLINE 
  auto front_unchecked() noexcept -> reference {
    return data_[0];
  }
  
  [[nodiscard]] 
  SCL_FORCE_INLINE 
  auto front_unchecked() const noexcept -> const_reference {
    return data_[0];
  }

  /// @brief Get first element
  /// @throws ValueError if span is empty
  [[nodiscard]] 
  SCL_FORCE_INLINE 
  auto front() -> reference {
    error::check_arg(!empty(), "front() called on empty span");
    return front_unchecked();
  }
  
  [[nodiscard]] 
  SCL_FORCE_INLINE 
  auto front() const -> const_reference {
    error::check_arg(!empty(), "front() called on empty span");
    return front_unchecked();
  }

  /// @brief Get last element (unchecked)
  /// @warning No bounds checking - caller must ensure !empty()
  [[nodiscard]] 
  SCL_FORCE_INLINE 
  auto back_unchecked() noexcept -> reference {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data_[size_ - 1];
  }
  
  [[nodiscard]] 
  SCL_FORCE_INLINE 
  auto back_unchecked() const noexcept -> const_reference {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data_[size_ - 1];
  }

  /// @brief Get last element
  /// @throws ValueError if span is empty
  [[nodiscard]] 
  SCL_FORCE_INLINE 
  auto back() -> reference {
    error::check_arg(!empty(), "back() called on empty span");
    return back_unchecked();
  }
  
  [[nodiscard]] 
  SCL_FORCE_INLINE 
  auto back() const -> const_reference {
    error::check_arg(!empty(), "back() called on empty span");
    return back_unchecked();
  }

  [[nodiscard]] SCL_FORCE_INLINE auto data() noexcept -> pointer {
    return data_;
  }
  [[nodiscard]] SCL_FORCE_INLINE auto data() const noexcept -> const_pointer {
    return data_;
  }

  // -------------------------------------------------------------------------
  // Capacity (STL-compatible)
  // -------------------------------------------------------------------------

  [[nodiscard]] SCL_FORCE_INLINE auto size() const noexcept -> size_type {
    return size_;
  }
  
  /// @brief Check if span is empty (no data)
  /// @return true if size is zero or data points to empty sentinel
  [[nodiscard]]
  constexpr auto empty() const noexcept -> bool {
    return size_ == 0 || detail::is_empty_sentinel(data_);
  }
  
  [[nodiscard]] SCL_FORCE_INLINE auto size_bytes() const noexcept
      -> std::size_t {
    return static_cast<std::size_t>(size_) * sizeof(T);
  }
  
  [[nodiscard]] SCL_FORCE_INLINE explicit operator bool() const noexcept {
    return data_ != nullptr && size_ > 0;
  }

  // -------------------------------------------------------------------------
  // Iterators (STL-compatible)
  // -------------------------------------------------------------------------

  [[nodiscard]] SCL_FORCE_INLINE auto begin() noexcept -> iterator {
    return data_;
  }
  [[nodiscard]] SCL_FORCE_INLINE auto end() noexcept -> iterator {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data_ + size_;
  }
  [[nodiscard]] SCL_FORCE_INLINE auto begin() const noexcept
      -> const_iterator {
    return data_;
  }
  [[nodiscard]] SCL_FORCE_INLINE auto end() const noexcept -> const_iterator {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data_ + size_;
  }
  [[nodiscard]] SCL_FORCE_INLINE auto cbegin() const noexcept
      -> const_iterator {
    return data_;
  }
  [[nodiscard]] SCL_FORCE_INLINE auto cend() const noexcept -> const_iterator {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data_ + size_;
  }

  // -------------------------------------------------------------------------
  // Ownership Queries
  // -------------------------------------------------------------------------

  /// @brief Check if span owns its data exclusively (owned mode)
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto is_owned() const noexcept -> bool {
    return detail::is_owned_sentinel(storage_);
  }

  /// @brief Check if span shares data via storage (shared mode)
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto is_shared() const noexcept -> bool {
    return storage_ != nullptr && !detail::is_owned_sentinel(storage_);
  }

  /// @brief Check if span is a non-owning view
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto is_view() const noexcept -> bool {
    return storage_ == nullptr;
  }

  /// @brief Check if this is the only reference
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto is_unique() const noexcept -> bool {
    if (is_shared()) {
      return storage_->is_unique();
    }
    return is_owned();  // Owned is always unique
  }

  /// @brief Get reference count
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto use_count() const noexcept -> std::uint32_t {
    if (is_shared()) {
      return storage_->use_count();
    }
    return is_owned() ? 1 : 0;
  }

  /// @brief Get underlying storage (nullptr if not shared, owned sentinel if
  /// owned)
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto storage() const noexcept -> Storage* {
    return is_shared() ? storage_ : nullptr;
  }

  // -------------------------------------------------------------------------
  // Ownership Conversion
  // -------------------------------------------------------------------------

  /// @brief Convert owned span to shared mode
  /// @return New Span in shared mode, original becomes empty
  /// @note Only valid for owned mode; shared/view modes return moved-from self
  [[nodiscard]]
  auto to_shared() && -> Span {
    if (!is_owned()) {
      // Already shared or view - just move
      return std::move(*this);
    }

    // Create Storage wrapping our data
    auto* new_storage = Storage::from_external(
        data_, size_bytes(), [](void* p) noexcept { delete[] static_cast<T*>(p); });

    if (new_storage == nullptr) [[unlikely]] {
      // Failed to create storage - clean up and return empty
      delete[] data_;
      storage_ = nullptr;
      data_ = nullptr;
      size_ = 0;
      return {};
    }

    Span result;
    result.storage_ = new_storage;  // Adopt (ref_count = 1)
    result.data_ = data_;
    result.size_ = size_;

    // Clear this span (now shared mode, don't delete data_)
    storage_ = nullptr;
    data_ = nullptr;
    size_ = 0;

    return result;
  }

  // -------------------------------------------------------------------------
  // Subspan Operations
  // -------------------------------------------------------------------------

  /// @brief Create subspan without bounds checking (unchecked version)
  /// @param[in] offset Element offset from start
  /// @param[in] count Number of elements (npos = rest of span)
  /// @return New Span
  /// @warning No bounds checking - caller must ensure valid range
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto subspan_unchecked(Size offset, Size count = npos) const noexcept -> Span {
    const Size actual = (count == npos) ? (size_ - offset) : count;
    
    Span result;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    result.data_ = data_ + offset;
    result.size_ = actual;

    if (is_shared()) {
      result.storage_ = storage_;
      storage_->incref();
    }
    // Owned/View → View (storage_ = nullptr)

    return result;
  }

  /// @brief Create subspan with bounds checking
  /// @param[in] offset Element offset from start
  /// @param[in] count Number of elements (npos = rest of span)
  /// @return New Span
  ///
  /// ## Safety
  ///
  ///   | Original Mode | Result Mode | Safety          |
  ///   |---------------|-------------|-----------------|
  ///   | Shared        | Shared      | ✓ Safe          |
  ///   | Owned         | View        | ⚠ Dangling risk |
  ///   | View          | View        | Same as original |
  ///
  /// @warning For Owned spans, result becomes View and may dangle!
  ///          Use std::move(span).safe_subspan(...) for safety
  [[nodiscard]]
  auto subspan(Size offset, Size count = npos) const noexcept -> Span {
    if (empty() || offset > size_) [[unlikely]] {
      return {};
    }

    const Size actual = (count == npos) ? (size_ - offset) : count;
    if (offset + actual > size_) [[unlikely]] {
      return {};
    }

    return subspan_unchecked(offset, actual);
  }

  /// @brief Create safe subspan (converts Owned to Shared first)
  /// @param[in] offset Element offset from start
  /// @param[in] count Number of elements (npos = rest of span)
  /// @return New Span
  /// @note This is the recommended way to create subspans - always safe
  /// @note Rvalue overload ensures original span is moved when Owned
  [[nodiscard]]
  auto safe_subspan(Size offset, Size count = npos) && -> Span {
    // Rvalue: convert Owned to Shared first, then subspan
    if (is_owned()) {
      return std::move(*this).to_shared().subspan(offset, count);
    }
    return subspan(offset, count);
  }

  /// @brief Get first N elements (unchecked)
  /// @warning No bounds checking - caller must ensure count <= size()
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto first_unchecked(Size count) const noexcept -> Span {
    return subspan_unchecked(0, count);
  }

  /// @brief Get first N elements
  [[nodiscard]]
  auto first(Size count) const noexcept -> Span {
    if (empty() || count > size_) [[unlikely]] {
      return {};
    }
    return first_unchecked(count);
  }

  /// @brief Get last N elements (unchecked)
  /// @warning No bounds checking - caller must ensure count <= size()
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto last_unchecked(Size count) const noexcept -> Span {
    return subspan_unchecked(size_ - count, count);
  }

  /// @brief Get last N elements
  [[nodiscard]]
  auto last(Size count) const noexcept -> Span {
    if (empty() || count > size_) [[unlikely]] {
      return {};
    }
    return last_unchecked(count);
  }

  // -------------------------------------------------------------------------
  // Byte Offset (for shared mode)
  // -------------------------------------------------------------------------

  /// @brief Get byte offset from storage start
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto offset_bytes() const noexcept -> std::ptrdiff_t {
    if (!is_shared() || data_ == nullptr) {
      return 0;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<const char*>(data_) -
           static_cast<const char*>(storage_->data());
  }

  /// @brief Get element offset from storage start
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto offset() const noexcept -> Size {
    return static_cast<Size>(offset_bytes() /
                             static_cast<std::ptrdiff_t>(sizeof(T)));
  }

  // -------------------------------------------------------------------------
  // Utilities
  // -------------------------------------------------------------------------

  /// @brief Swap with another span
  auto swap(Span& other) noexcept -> void {
    std::swap(storage_, other.storage_);
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }

  /// @brief Reset to empty state
  auto reset() noexcept -> void {
    release_resources();
    storage_ = nullptr;
    data_ = nullptr;
    size_ = 0;
  }

  /// @brief Release ownership and return pointer (owned mode only)
  /// @return Data pointer (caller takes ownership), or nullptr if not owned
  [[nodiscard]]
  auto release_ownership() noexcept -> pointer {
    if (!is_owned()) {
      return nullptr;
    }

    auto* ptr = data_;
    storage_ = nullptr;
    data_ = nullptr;
    size_ = 0;
    return ptr;
  }

  /// @brief Try to clone with error handling (deprecated - use clone())
  [[nodiscard]]
  auto try_clone() const -> Span {
    if (data_ == nullptr || size_ <= 0) {
      return {};
    }

    auto result = create_owned(size_);
    if (result) {
      std::copy(data_, data_ + size_, result.data_);
    }
    return result;
  }
  

  /// @brief Fill with value (SIMD-optimized for arithmetic types)
  auto fill(const T& value) -> void {
    if constexpr (std::is_arithmetic_v<T>) {
      // Use SIMD-optimized fill for arithmetic types
      vectorize::fill(to_std_span(), value);
    } else {
      std::fill(begin(), end(), value);
    }
  }
  
  /// @brief Copy from another span
  /// @param[in] other Source span
  /// @return true on success, false if size mismatch
  auto copy_from(const Span& other) -> bool {
    if (size_ != other.size_) {
      return false;
    }
    if (data_ == other.data_) {
      return true;
    }
    std::copy(other.begin(), other.end(), begin());
    return true;
  }
  
  /// @brief Copy from std::span
  /// @param[in] other Source std::span
  /// @return true on success, false if size mismatch
  auto copy_from(std::span<const T> other) -> bool {
    if (size_ != static_cast<Size>(other.size())) {
      return false;
    }
    std::copy(other.begin(), other.end(), begin());
    return true;
  }
  
  /// @brief Transform elements in-place
  /// @tparam UnaryOp Transformation operation
  /// @param[in] op Transformation function
  /// @return Reference to this span for chaining
  template<typename UnaryOp>
  auto transform(UnaryOp op) -> Span& {
    std::transform(begin(), end(), begin(), op);
    return *this;
  }
  
  /// @brief Apply function to each element
  /// @tparam UnaryFunc Function to apply
  /// @param[in] func Function to call on each element
  /// @return Reference to this span for chaining
  template<typename UnaryFunc>
  auto for_each(UnaryFunc func) -> Span& {
    std::for_each(begin(), end(), func);
    return *this;
  }
  
  /// @brief Reduce elements with binary operation
  /// @tparam BinaryOp Reduction operation
  /// @param[in] init Initial value
  /// @param[in] op Binary operation
  /// @return Reduced value
  template<typename BinaryOp>
  [[nodiscard]]
  auto reduce(T init, BinaryOp op) const -> T {
    return std::accumulate(begin(), end(), init, op);
  }
  
  /// @brief Sum all elements (SIMD-optimized)
  /// @return Sum of all elements
  /// @note Only available for arithmetic types
  [[nodiscard]]
  auto sum() const -> T requires std::is_arithmetic_v<T> {
    return vectorize::sum(to_std_span());
  }
  
  /// @brief Dot product with another span (SIMD-optimized)
  /// @param[in] other Other span
  /// @return Dot product
  /// @note Only available for arithmetic types
  /// @pre size() == other.size()
  [[nodiscard]]
  auto dot(const Span& other) const -> T requires std::is_arithmetic_v<T> {
    return vectorize::dot(to_std_span(), other.to_std_span());
  }
  
  /// @brief Find first occurrence of value (SIMD-optimized)
  /// @param[in] value Value to find
  /// @return Index of first occurrence, or size() if not found
  [[nodiscard]]
  auto find_value(const T& value) const -> Size {
    if constexpr (std::is_arithmetic_v<T>) {
      return vectorize::find(to_std_span(), value);
    } else {
      auto it = std::find(begin(), end(), value);
      return (it != end()) ? static_cast<Size>(it - begin()) : size_;
    }
  }
  
  /// @brief Check if value exists (SIMD-optimized)
  /// @param[in] value Value to find
  /// @return true if found
  [[nodiscard]]
  auto contains(const T& value) const -> bool {
    return find_value(value) < size_;
  }
  
  /// @brief Find minimum element (SIMD-optimized)
  /// @return Minimum value
  /// @note Only available for totally ordered types
  /// @pre !empty()
  [[nodiscard]]
  auto min() const -> T requires std::totally_ordered<T> {
    if constexpr (std::is_arithmetic_v<T>) {
      return vectorize::min_value(to_std_span());
    } else {
      error::check_arg(!empty(), "min: empty span");
      return *std::min_element(begin(), end());
    }
  }
  
  /// @brief Find maximum element (SIMD-optimized)
  /// @return Maximum value
  /// @note Only available for totally ordered types
  /// @pre !empty()
  [[nodiscard]]
  auto max() const -> T requires std::totally_ordered<T> {
    if constexpr (std::is_arithmetic_v<T>) {
      return vectorize::max_value(to_std_span());
    } else {
      error::check_arg(!empty(), "max: empty span");
      return *std::max_element(begin(), end());
    }
  }
  
  /// @brief Scale elements in-place (SIMD-optimized)
  /// @param[in] factor Scale factor
  /// @note Only available for arithmetic types
  auto scale(T factor) -> Span& requires std::is_arithmetic_v<T> {
    vectorize::scale(to_std_span(), to_std_span(), factor);
    return *this;
  }
  
  /// @brief Element-wise addition (SIMD-optimized)
  /// @param[in] other Source span to add
  /// @note Only available for arithmetic types
  /// @pre size() == other.size()
  auto add(const Span& other) -> Span& requires std::is_arithmetic_v<T> {
    vectorize::add(to_std_span(), other.to_std_span(), to_std_span());
    return *this;
  }
  
  /// @brief Element-wise multiplication (SIMD-optimized)
  /// @param[in] other Source span to multiply
  /// @note Only available for arithmetic types
  /// @pre size() == other.size()
  auto multiply(const Span& other) -> Span& requires std::is_arithmetic_v<T> {
    vectorize::mul(to_std_span(), other.to_std_span(), to_std_span());
    return *this;
  }

  // -------------------------------------------------------------------------
  // Comparison
  // -------------------------------------------------------------------------

  [[nodiscard]]
  auto same_data(const Span& other) const noexcept -> bool {
    return data_ == other.data_;
  }

  [[nodiscard]]
  auto same_storage(const Span& other) const noexcept -> bool {
    return is_shared() && other.is_shared() && storage_ == other.storage_;
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
    if (detail::is_owned_sentinel(storage_)) {
      // Owned mode: delete the data
      delete[] data_;
    } else if (storage_ != nullptr) {
      // Shared mode: decref the storage
      storage_->decref();
    }
    // View mode: do nothing
  }

  Storage* storage_ = nullptr;  ///< nullptr=view, sentinel=owned, else=shared
  T* data_ = nullptr;           ///< Pointer to data
  Size size_ = 0;               ///< Element count
};

// =============================================================================
// SECTION 6: Static Assertions for Memory Layout
// =============================================================================

namespace detail {
inline constexpr std::size_t kSpanExpectedSize = 24;  // 3 pointers on 64-bit
}

static_assert(sizeof(Span<Real>) == detail::kSpanExpectedSize,
              "Span should be exactly 24 bytes (3 pointers)");
static_assert(std::is_nothrow_move_constructible_v<Span<Real>>,
              "Span must be nothrow move constructible");
static_assert(std::is_nothrow_move_assignable_v<Span<Real>>,
              "Span must be nothrow move assignable");

// =============================================================================
// SECTION 7: Multi-Span Storage Utilities
// =============================================================================

/// @brief Create multiple spans sharing a single storage
///
/// @tparam T Element type
/// @tparam N Number of spans
/// @param[in] sizes Array of sizes for each span
/// @param[in] alignment Buffer alignment
/// @return Pair of (Storage*, array of Span<T>)
///
/// ## Example
///
/// ```cpp
/// auto [storage, spans] = make_shared_spans<Real, 3>({100, 200, 150});
/// auto& col0 = spans[0];  // 100 elements at offset 0
/// auto& col1 = spans[1];  // 200 elements at offset 100*sizeof(Real)
/// auto& col2 = spans[2];  // 150 elements at offset 300*sizeof(Real)
/// ```
template <typename T, std::size_t N>
[[nodiscard]]
auto make_shared_spans(const std::array<Size, N>& sizes,
                       std::size_t alignment = detail::kDefaultAlignment)
    -> std::pair<Storage*, std::array<Span<T>, N>> {
  // Calculate total size
  Size total = 0;
  for (auto s : sizes) {
    total += s;
  }

  if (total == 0) [[unlikely]] {
    return {nullptr, {}};
  }

  // Create shared storage
  auto* storage = Storage::create_aligned(
      static_cast<std::size_t>(total) * sizeof(T), alignment);
  if (storage == nullptr) [[unlikely]] {
    return {nullptr, {}};
  }

  // Create spans with offsets
  std::array<Span<T>, N> spans;
  std::ptrdiff_t offset = 0;

  for (std::size_t i = 0; i < N; ++i) {
    if (i == 0) {
      // First span adopts the storage (ref_count stays 1)
      spans[i] = Span<T>{storage, offset, sizes[i], adopt};
    } else {
      // Subsequent spans share (incref)
      spans[i] = Span<T>{storage, offset, sizes[i]};
    }
    offset += static_cast<std::ptrdiff_t>(sizes[i]) * sizeof(T);
  }

  return {storage, std::move(spans)};
}

/// @brief Create multiple spans sharing a single storage (variadic)
///
/// @tparam T Element type
/// @tparam Sizes Variadic size parameters
/// @param[in] sizes Size of each span
/// @return Tuple of Span<T>...
///
/// ## Example
///
/// ```cpp
/// auto [col0, col1, col2] = make_shared_spans<Real>(100, 200, 150);
/// ```
template <typename T, typename... Sizes>
  requires(std::is_convertible_v<Sizes, Size> && ...)
[[nodiscard]]
auto make_shared_spans(Sizes... sizes)
    -> std::tuple<Span<T>, decltype((void(sizes), Span<T>{}))...> {
  constexpr std::size_t N = sizeof...(Sizes);
  std::array<Size, N> size_array = {static_cast<Size>(sizes)...};

  auto result = make_shared_spans<T, N>(size_array);
  auto& spans = result.second;

  return [&spans]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::make_tuple(std::move(spans[Is])...);
  }(std::make_index_sequence<N>{});
}

// =============================================================================
// SECTION 8: Batch Allocation Utilities
// =============================================================================

/// @brief Result of batch allocation
/// @tparam T Element type
template<typename T>
struct BatchAllocation {
  Storage* storage;           ///< Shared storage (nullptr on failure)
  std::vector<Span<T>> spans; ///< Vector of spans sharing storage
  
  /// @brief Check if allocation succeeded
  [[nodiscard]] explicit operator bool() const noexcept { 
    return storage != nullptr; 
  }
  
  /// @brief Get total element count
  [[nodiscard]] auto total_size() const noexcept -> Size {
    Size total = 0;
    for (const auto& s : spans) {
      total += s.size();
    }
    return total;
  }
  
  /// @brief Get span by index (unchecked)
  /// @warning No bounds checking - caller must ensure i < count()
  [[nodiscard]] 
  SCL_FORCE_INLINE
  auto get_unchecked(std::size_t i) noexcept -> Span<T>& { 
    return spans[i]; 
  }
  
  [[nodiscard]] 
  SCL_FORCE_INLINE
  auto get_unchecked(std::size_t i) const noexcept -> const Span<T>& { 
    return spans[i]; 
  }

  /// @brief Get span by index
  /// @note In debug builds, asserts i < count()
  [[nodiscard]] auto operator[](std::size_t i) -> Span<T>& { 
#ifndef NDEBUG
    if (i >= spans.size()) {
      detail::debug_assert_fail("batch index out of bounds", 
                               source_location::current());
    }
#endif
    return get_unchecked(i);
  }
  
  /// @note In debug builds, asserts i < count()
  [[nodiscard]] auto operator[](std::size_t i) const -> const Span<T>& { 
#ifndef NDEBUG
    if (i >= spans.size()) {
      detail::debug_assert_fail("batch index out of bounds", 
                               source_location::current());
    }
#endif
    return get_unchecked(i);
  }
  
  /// @brief Get number of spans
  [[nodiscard]] auto count() const noexcept -> std::size_t { 
    return spans.size(); 
  }
  
  /// @brief Check if empty
  [[nodiscard]] auto empty() const noexcept -> bool {
    return spans.empty();
  }
};

/// @brief Create multiple spans from a dynamic size array
/// @tparam T Element type
/// @param[in] sizes Span of sizes for each allocation
/// @param[in] alignment Memory alignment
/// @return BatchAllocation containing shared storage and vector of spans
///
/// ## Example
/// ```cpp
/// std::vector<Size> sizes = {100, 200, 150, 300};
/// auto batch = make_batch_spans<Real>(sizes);
/// 
/// for (auto& span : batch.spans) {
///     span.fill(0.0);
/// }
/// ```
template<typename T>
[[nodiscard]]
auto make_batch_spans(std::span<const Size> sizes,
                      std::size_t alignment = detail::kDefaultAlignment)
    -> BatchAllocation<T> 
{
  BatchAllocation<T> result{nullptr, {}};
  
  if (sizes.empty()) {
    return result;
  }
  
  // Calculate total size with alignment padding
  std::size_t total_bytes = 0;
  std::vector<std::size_t> offsets;
  offsets.reserve(sizes.size());
  
  for (auto size : sizes) {
    // Align each span start to element alignment
    total_bytes = (total_bytes + alignof(T) - 1) & ~(alignof(T) - 1);
    offsets.push_back(total_bytes);
    total_bytes += static_cast<std::size_t>(size) * sizeof(T);
  }
  
  if (total_bytes == 0) {
    return result;
  }
  
  // Allocate single storage block
  auto* storage = Storage::create_aligned(total_bytes, alignment);
  if (storage == nullptr) {
    return result;
  }
  
  result.storage = storage;
  result.spans.reserve(sizes.size());
  
  // Create spans with offsets
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    if (i == 0) {
      // First span adopts storage
      result.spans.emplace_back(storage, 
                                static_cast<std::ptrdiff_t>(offsets[i]), 
                                sizes[i], adopt);
    } else {
      // Subsequent spans share (incref)
      result.spans.emplace_back(storage, 
                                static_cast<std::ptrdiff_t>(offsets[i]), 
                                sizes[i]);
    }
  }
  
  return result;
}

/// @brief Create batch spans from vector of sizes
/// @tparam T Element type
/// @param[in] sizes Vector of sizes
/// @param[in] alignment Memory alignment
/// @return BatchAllocation
template<typename T>
[[nodiscard]]
auto make_batch_spans(const std::vector<Size>& sizes,
                      std::size_t alignment = detail::kDefaultAlignment)
    -> BatchAllocation<T> 
{
  return make_batch_spans<T>(std::span{sizes}, alignment);
}

/// @brief Create batch spans from initializer list
/// @tparam T Element type
/// @param[in] sizes Initializer list of sizes
/// @param[in] alignment Memory alignment
/// @return BatchAllocation
template<typename T>
[[nodiscard]]
auto make_batch_spans(std::initializer_list<Size> sizes,
                      std::size_t alignment = detail::kDefaultAlignment)
    -> BatchAllocation<T> 
{
  std::vector<Size> v(sizes);
  return make_batch_spans<T>(std::span{v}, alignment);
}

/// @brief Create uniform batch (all spans same size)
/// @tparam T Element type
/// @param[in] count Number of spans
/// @param[in] each_size Size of each span
/// @param[in] alignment Memory alignment
/// @return BatchAllocation
template<typename T>
[[nodiscard]]
auto make_uniform_batch(std::size_t count, Size each_size,
                        std::size_t alignment = detail::kDefaultAlignment)
    -> BatchAllocation<T> 
{
  std::vector<Size> sizes(count, each_size);
  return make_batch_spans<T>(std::span{sizes}, alignment);
}

/// @brief Create 2D matrix as batch of row spans
/// @tparam T Element type
/// @param[in] rows Number of rows
/// @param[in] cols Number of columns
/// @param[in] alignment Memory alignment
/// @return BatchAllocation where each span is a row
template<typename T>
[[nodiscard]]
auto make_matrix_spans(Size rows, Size cols,
                       std::size_t alignment = detail::kDefaultAlignment)
    -> BatchAllocation<T> 
{
  std::vector<Size> sizes(static_cast<std::size_t>(rows), cols);
  return make_batch_spans<T>(std::span{sizes}, alignment);
}

/// @brief Result of strided matrix allocation
/// @tparam T Element type
///
/// StridedMatrix provides efficient row-major matrix storage with aligned rows.
/// Each row starts at an aligned address for optimal SIMD performance.
template<typename T>
struct StridedMatrix {
  Storage* storage;  ///< Underlying storage
  T* data;           ///< Pointer to matrix data
  Size rows;         ///< Number of rows
  Size cols;         ///< Number of columns
  Size stride;       ///< Elements per row (>= cols for alignment)
  
  /// @brief Check if allocation succeeded
  [[nodiscard]] explicit operator bool() const noexcept { 
    return storage != nullptr && data != nullptr; 
  }
  
  /// @brief Get row as std::span (unchecked)
  /// @param[in] i Row index
  /// @return Span view of row i
  /// @warning No bounds checking - caller must ensure i < rows
  [[nodiscard]] 
  SCL_FORCE_INLINE
  auto row_unchecked(Size i) noexcept -> std::span<T> {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return {data + i * stride, static_cast<std::size_t>(cols)};
  }
  
  [[nodiscard]] 
  SCL_FORCE_INLINE
  auto row_unchecked(Size i) const noexcept -> std::span<const T> {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return {data + i * stride, static_cast<std::size_t>(cols)};
  }

  /// @brief Get row as std::span
  /// @param[in] i Row index
  /// @return Span view of row i
  /// @note In debug builds, asserts i < rows
  [[nodiscard]] auto row(Size i) noexcept -> std::span<T> {
#ifndef NDEBUG
    if (i >= rows) {
      detail::debug_assert_fail("row index out of bounds", 
                               source_location::current());
    }
#endif
    return row_unchecked(i);
  }
  
  /// @brief Get row as std::span (const)
  /// @param[in] i Row index
  /// @return Const span view of row i
  /// @note In debug builds, asserts i < rows
  [[nodiscard]] auto row(Size i) const noexcept -> std::span<const T> {
#ifndef NDEBUG
    if (i >= rows) {
      detail::debug_assert_fail("row index out of bounds", 
                               source_location::current());
    }
#endif
    return row_unchecked(i);
  }
  
  /// @brief Element access (unchecked)
  /// @param[in] i Row index
  /// @param[in] j Column index
  /// @return Reference to element (i, j)
  /// @warning No bounds checking - caller must ensure valid indices
  [[nodiscard]] 
  SCL_FORCE_INLINE
  auto at_unchecked(Size i, Size j) noexcept -> T& {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data[i * stride + j];
  }
  
  [[nodiscard]] 
  SCL_FORCE_INLINE
  auto at_unchecked(Size i, Size j) const noexcept -> const T& {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data[i * stride + j];
  }
  
  /// @brief Element access
  /// @param[in] i Row index
  /// @param[in] j Column index
  /// @return Reference to element (i, j)
  /// @note In debug builds, asserts valid indices
  [[nodiscard]] auto operator()(Size i, Size j) noexcept -> T& {
#ifndef NDEBUG
    if (i >= rows || j >= cols) {
      detail::debug_assert_fail("matrix indices out of bounds", 
                               source_location::current());
    }
#endif
    return at_unchecked(i, j);
  }
  
  /// @brief Element access (const)
  /// @note In debug builds, asserts valid indices
  [[nodiscard]] auto operator()(Size i, Size j) const noexcept -> const T& {
#ifndef NDEBUG
    if (i >= rows || j >= cols) {
      detail::debug_assert_fail("matrix indices out of bounds", 
                               source_location::current());
    }
#endif
    return at_unchecked(i, j);
  }
  
  /// @brief Get total number of elements (including padding)
  [[nodiscard]] auto total_elements() const noexcept -> Size {
    return rows * stride;
  }
  
  /// @brief Get total bytes
  [[nodiscard]] auto total_bytes() const noexcept -> std::size_t {
    return static_cast<std::size_t>(rows * stride) * sizeof(T);
  }
  
  /// @brief Get number of active elements (excluding padding)
  [[nodiscard]] auto active_elements() const noexcept -> Size {
    return rows * cols;
  }
  
  /// @brief Fill entire matrix with value
  auto fill(const T& value) noexcept -> void {
    for (Size i = 0; i < rows; ++i) {
      auto r = row(i);
      std::fill(r.begin(), r.end(), value);
    }
  }
};

/// @brief Create matrix with aligned row stride
/// @tparam T Element type
/// @param[in] rows Number of rows
/// @param[in] cols Number of columns
/// @param[in] alignment Byte alignment for each row
/// @return StridedMatrix with aligned rows
///
/// ## Example
/// ```cpp
/// auto matrix = make_strided_matrix<Real>(100, 64, 64);  // 64-byte aligned rows
/// for (Size i = 0; i < matrix.rows; ++i) {
///     auto row = matrix.row(i);
///     std::fill(row.begin(), row.end(), static_cast<Real>(i));
/// }
/// ```
template<typename T>
[[nodiscard]]
auto make_strided_matrix(Size rows, Size cols,
                         std::size_t alignment = detail::kDefaultAlignment)
    -> StridedMatrix<T> 
{
  if (rows <= 0 || cols <= 0) [[unlikely]] {
    return {nullptr, nullptr, 0, 0, 0};
  }
  
  // Calculate stride to align each row
  const Size min_stride = cols;
  const Size align_elems = static_cast<Size>(alignment / sizeof(T));
  const Size stride = (align_elems > 0) 
      ? ((min_stride + align_elems - 1) / align_elems) * align_elems
      : min_stride;
  
  const std::size_t total_bytes = static_cast<std::size_t>(rows * stride) * sizeof(T);
  
  auto* storage = Storage::create_aligned(total_bytes, alignment);
  if (storage == nullptr) [[unlikely]] {
    return {nullptr, nullptr, 0, 0, 0};
  }
  
  return {
      storage,
      storage->data_as<T>(),
      rows,
      cols,
      stride
  };
}

// =============================================================================
// SECTION 9: Type Aliases
// =============================================================================

using RealSpan = Span<Real>;
using IndexSpan = Span<Index>;
using FloatSpan = Span<float>;
using DoubleSpan = Span<double>;
using Int32Span = Span<std::int32_t>;
using Int64Span = Span<std::int64_t>;
using ByteSpan = Span<std::byte>;
using CharSpan = Span<char>;

// =============================================================================
// SECTION 10: Advanced Batch Operations
// =============================================================================

/// @brief Split a span into equal-sized chunks
/// @tparam T Element type
/// @param[in] source Source span (will be moved and converted to shared)
/// @param[in] chunk_size Size of each chunk
/// @return Vector of span chunks
template<typename T>
[[nodiscard]]
auto split_span(Span<T>&& source, Size chunk_size) -> std::vector<Span<T>> {
  std::vector<Span<T>> chunks;
  
  if (chunk_size <= 0 || source.empty()) {
    return chunks;
  }
  
  // Convert to shared first for safe subspans
  auto shared = std::move(source).to_shared();
  
  const Size num_chunks = (shared.size() + chunk_size - 1) / chunk_size;
  chunks.reserve(static_cast<std::size_t>(num_chunks));
  
  for (Size i = 0; i < shared.size(); i += chunk_size) {
    Size actual = std::min(chunk_size, shared.size() - i);
    chunks.push_back(shared.subspan(i, actual));
  }
  
  return chunks;
}

/// @brief Concatenate multiple spans into one
/// @tparam T Element type
/// @param[in] sources Source spans
/// @return New span containing concatenated data
template<typename T>
[[nodiscard]]
auto concat_spans(std::span<const Span<T>> sources) -> Span<T> {
  Size total = 0;
  for (const auto& s : sources) {
    total += s.size();
  }
  
  if (total == 0) {
    return {};
  }
  
  auto result = Span<T>::create_owned(total);
  if (!result) {
    return {};
  }
  
  T* dst = result.data();
  for (const auto& src : sources) {
    std::copy(src.begin(), src.end(), dst);
    dst += src.size();
  }
  
  return result;
}

/// @brief Interleave multiple spans into one
/// @tparam T Element type
/// @param[in] sources Source spans (must all have same size)
/// @return New span with interleaved elements
template<typename T>
[[nodiscard]]
auto interleave_spans(std::span<const Span<T>> sources) -> Span<T> {
  if (sources.empty()) {
    return {};
  }
  
  // All sources must have same size
  const Size size = sources[0].size();
  for (const auto& s : sources) {
    if (s.size() != size) {
      return {};
    }
  }
  
  const Size total = size * static_cast<Size>(sources.size());
  auto result = Span<T>::create_owned(total);
  if (!result) {
    return {};
  }
  
  T* dst = result.data();
  for (Size i = 0; i < size; ++i) {
    for (const auto& src : sources) {
      *dst++ = src[i];
    }
  }
  
  return result;
}

// =============================================================================
// SECTION 11: Factory Functions
// =============================================================================

/// @brief Create owned span
template <typename T>
[[nodiscard]]
inline
auto make_span(Size count) -> Span<T> {
  return Span<T>::create_owned(count);
}

/// @brief Create aligned span (via Storage)
template <typename T>
[[nodiscard]]
inline
auto make_span_aligned(Size count,
                       std::size_t alignment = detail::kDefaultAlignment)
    -> Span<T> {
  return Span<T>::create_aligned(count, alignment);
}

/// @brief Create shared span from storage (incref)
template <typename T>
[[nodiscard]]
inline
auto make_span_shared(Storage* storage, std::ptrdiff_t offset, Size count)
    -> Span<T> {
  return Span<T>::create_shared(storage, offset, count);
}

/// @brief Create view span (non-owning)
template <typename T>
[[nodiscard]]
inline
auto make_span_view(T* data, Size count) -> Span<T> {
  return Span<T>::view(data, count);
}

/// @brief Create Storage with specified size
[[nodiscard]]
inline
auto make_storage(std::size_t size) -> Storage* {
  return Storage::create_default(size);
}

/// @brief Create aligned Storage
[[nodiscard]]
inline
auto make_storage_aligned(std::size_t size,
                          std::size_t alignment = detail::kDefaultAlignment)
    -> Storage* {
  return Storage::create_aligned(size, alignment);
}

}  // namespace scl

// =============================================================================
// Integration with scl::memory (optional, for convenience)
// =============================================================================

namespace scl::memory {

/// @brief Fill span using memory module
/// @tparam T Element type
/// @param[in] span Span to fill
/// @param[in] value Fill value
template<typename T>
SCL_FORCE_INLINE
void fill(Span<T>& span, const T& value) {
  fill(span.to_std_span(), value);
}

/// @brief Zero span using memory module
/// @tparam T Element type
/// @param[in] span Span to zero
template<typename T>
SCL_FORCE_INLINE
void zero(Span<T>& span) {
  zero(span.to_std_span());
}

/// @brief Copy between spans using memory module
/// @tparam T Element type
/// @param[in] src Source span
/// @param[in] dst Destination span
template<typename T>
SCL_FORCE_INLINE
void copy(const Span<T>& src, Span<T>& dst) {
  copy(src.to_std_span(), dst.to_std_span());
}

/// @brief Fast copy without overlap check
/// @tparam T Element type
/// @param[in] src Source span
/// @param[in] dst Destination span
template<typename T>
SCL_FORCE_INLINE
void copy_fast(const Span<T>& src, Span<T>& dst) {
  copy_fast(src.to_std_span(), dst.to_std_span());
}

/// @brief Stream copy for large spans (non-temporal stores)
/// @tparam T Element type
/// @param[in] src Source span
/// @param[in] dst Destination span
template<typename T>
SCL_FORCE_INLINE
void stream_copy(const Span<T>& src, Span<T>& dst) {
  stream_copy(src.to_std_span(), dst.to_std_span());
}

/// @brief Prefetch span for reading
/// @tparam Locality Cache locality (0-3)
/// @tparam T Element type
/// @param[in] span Span to prefetch
template<int Locality = 3, typename T>
SCL_FORCE_INLINE
void prefetch_read(const Span<T>& span) {
  prefetch_read<Locality>(span.to_std_span());
}

/// @brief Prefetch span for writing
/// @tparam Locality Cache locality (0-3)
/// @tparam T Element type
/// @param[in] span Span to prefetch
template<int Locality = 3, typename T>
SCL_FORCE_INLINE
void prefetch_write(Span<T>& span) {
  prefetch_write<Locality>(span.to_std_span());
}

/// @brief Compare two spans for equality
/// @tparam T Element type
/// @param[in] a First span
/// @param[in] b Second span
/// @return true if equal
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
auto equal(const Span<T>& a, const Span<T>& b) -> bool {
  return equal(a.to_std_span(), b.to_std_span());
}

/// @brief Reverse span in-place
/// @tparam T Element type
/// @param[in] span Span to reverse
template<typename T>
void reverse(Span<T>& span) {
  reverse(span.to_std_span());
}

}  // namespace scl::memory
