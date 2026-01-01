#pragma once

/// @file scl/core/memory.hpp
/// @brief SCL Memory Module - High-Performance Memory Primitives
///
/// This header provides:
///   - Memory configuration constants (alignment, cache line, thresholds)
///   - Aligned memory allocation (RAII and raw)
///   - Data movement operations (copy, fill, zero, stream copy)
///   - Prefetch utilities
///   - Memory comparison and swap operations
///   - SIMD-optimized reverse operations
///   - Large memory allocation (mmap/VirtualAlloc)
///   - Huge pages support
///
/// @note Uses std::span for non-owning views (C++20)
/// @note Delegates to optimized libc/compiler when beneficial
/// @note Platform-specific optimizations via SCL_PLATFORM_* macros

#include "scl/core/type.hpp"
#include "scl/core/macro.hpp"
#include "scl/core/error.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

// Platform-specific headers for memory operations
#if SCL_PLATFORM_WINDOWS
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <malloc.h>
#elif SCL_PLATFORM_POSIX
    #include <sys/mman.h>
    #include <unistd.h>
#endif

// SIMD headers for streaming operations
#if SCL_ARCH_X86 && (SCL_SIMD_SSE2 || SCL_SIMD_AVX || SCL_SIMD_AVX512)
    #include <immintrin.h>
#elif SCL_ARCH_ARM && SCL_SIMD_NEON
    #include <arm_neon.h>
#endif

namespace scl::memory {

// =============================================================================
// SECTION 1: Memory Configuration Constants
// =============================================================================

/// @brief Default memory alignment (64 bytes for AVX-512)
inline constexpr Size DEFAULT_ALIGNMENT = 64;

/// @brief Cache line size (64 bytes for modern x86/ARM)
inline constexpr Size CACHE_LINE_SIZE = 64;

/// @brief Threshold for using non-temporal (streaming) stores (256 KB)
inline constexpr Size STREAM_THRESHOLD = Size{256} * Size{1024};

/// @brief Alignment required for streaming stores
inline constexpr Size STREAM_ALIGNMENT = 64;

/// @brief Default prefetch distance (in elements)
inline constexpr Size DEFAULT_PREFETCH_DISTANCE = 8;

/// @brief Maximum prefetch operations per call
inline constexpr Size DEFAULT_MAX_PREFETCHES = 16;

/// @brief Standard page size (4 KB)
inline constexpr Size PAGE_SIZE = 4096;

/// @brief Huge page size (2 MB on x86_64, varies on ARM)
#if SCL_ARCH_X86_64
    inline constexpr Size HUGE_PAGE_SIZE = Size{2} * Size{1024} * Size{1024};
#elif SCL_ARCH_ARM64
    // ARM64 supports various huge page sizes, 2MB is common
    inline constexpr Size HUGE_PAGE_SIZE = Size{2} * Size{1024} * Size{1024};
#else
    inline constexpr Size HUGE_PAGE_SIZE = Size{2} * Size{1024} * Size{1024};
#endif

/// @brief Threshold for using large page allocation (1 MB)
inline constexpr Size LARGE_ALLOC_THRESHOLD = Size{1024} * Size{1024};

/// @brief Threshold for using mmap/VirtualAlloc (64 KB)
inline constexpr Size MMAP_THRESHOLD = Size{64} * Size{1024};

// =============================================================================
// SECTION 2: Alignment Utility Functions (Forward)
// =============================================================================

/// @brief Calculate aligned size (round up)
/// @param[in] size Original size
/// @param[in] alignment Alignment requirement
/// @return Aligned size >= original size
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto align_up(Size size, Size alignment) noexcept -> Size {
    return (size + alignment - 1) & ~(alignment - 1);
}

/// @brief Calculate aligned size (round down)
/// @param[in] size Original size
/// @param[in] alignment Alignment requirement
/// @return Aligned size <= original size
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto align_down(Size size, Size alignment) noexcept -> Size {
    return size & ~(alignment - 1);
}

// =============================================================================
// SECTION 3: Aligned Memory Allocation
// =============================================================================

/// @brief Custom deleter for aligned memory
/// @tparam T Element type
/// @note Uses platform-specific deallocation:
///       - Windows: _aligned_free()
///       - POSIX: std::free() (posix_memalign compatible)
///       - C++17: operator delete with align_val_t for arithmetic types
template<typename T>
struct AlignedDeleter {
    Size alignment_;

    /// @brief Construct with alignment
    /// @param[in] alignment Memory alignment in bytes
    explicit
    constexpr
    AlignedDeleter(Size alignment = DEFAULT_ALIGNMENT) noexcept
        : alignment_(alignment) {}

    /// @brief Delete aligned memory
    /// @param[in] ptr Pointer to free
    void operator()(T* ptr) const noexcept {
        if (!ptr) [[unlikely]] { return; };

        if constexpr (std::is_arithmetic_v<T>) {
            // Use C++17 aligned delete for arithmetic types
            ::operator delete[](ptr, std::align_val_t(alignment_));
        } else {
            // Use platform-specific deallocation for complex types
#if SCL_PLATFORM_WINDOWS
            _aligned_free(ptr);
#elif SCL_PLATFORM_POSIX
            // posix_memalign returns memory compatible with std::free
            std::free(ptr);
#else
            // Fallback: assume standard free is compatible
            std::free(ptr);
#endif
        }
    }
};

/// @brief Unique pointer type for aligned arrays
/// @tparam T Element type
/// @note unique_ptr<T[]> is the standard way to manage dynamic arrays
template<typename T>
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
using AlignedPtr = std::unique_ptr<T[], AlignedDeleter<T>>;

/// @brief Allocate aligned memory with automatic cleanup
/// @tparam T Element type (must be trivially constructible)
/// @param[in] count Number of elements
/// @param[in] alignment Memory alignment in bytes
/// @return Unique pointer to aligned memory (nullptr on failure)
/// @note Memory is zero-initialized for arithmetic types
/// @note Platform-specific allocation:
///       - Windows: _aligned_malloc()
///       - POSIX: posix_memalign()
///       - C++17: aligned new for arithmetic types
template<typename T>
[[nodiscard]]
auto aligned_alloc(Size count, Size alignment = DEFAULT_ALIGNMENT) -> AlignedPtr<T> {
    static_assert(std::is_trivially_constructible_v<T>,
                  "aligned_alloc: Type must be trivially constructible");

    if (count == 0) [[unlikely]] {
        return AlignedPtr<T>(nullptr, AlignedDeleter<T>(alignment));
    }

    const auto byte_size = static_cast<std::size_t>(count) * sizeof(T);
    T* raw_ptr = nullptr;

    if constexpr (std::is_arithmetic_v<T>) {
        // Use C++17 aligned allocation with value initialization
        try {
            raw_ptr = new (std::align_val_t(alignment)) T[count]();
        } catch (...) {
            return AlignedPtr<T>(nullptr, AlignedDeleter<T>(alignment));
        }
    } else {
        // Use platform-specific aligned allocation for complex types
        void* ptr = nullptr;

#if SCL_PLATFORM_WINDOWS
        // Windows: _aligned_malloc (requires _aligned_free)
        ptr = _aligned_malloc(byte_size, alignment);
#elif SCL_PLATFORM_POSIX
        // POSIX: posix_memalign (requires std::free)
        // Note: alignment must be power of 2 and >= sizeof(void*)
        const auto adjusted_alignment = std::max(alignment, static_cast<Size>(sizeof(void*)));
        if (::posix_memalign(&ptr, adjusted_alignment, byte_size) != 0) {
            ptr = nullptr;
        }
#else
        // Fallback: use C11 aligned_alloc if available
        #if __STDC_VERSION__ >= 201112L
            ptr = std::aligned_alloc(alignment, byte_size);
        #else
            // Last resort: over-allocate and manually align
            const auto total_size = byte_size + alignment + sizeof(void*);
            void* base = std::malloc(total_size);
            if (base) {
                auto aligned = reinterpret_cast<void*>(
                    (reinterpret_cast<std::uintptr_t>(base) + sizeof(void*) + alignment - 1) 
                    & ~(alignment - 1));
                // Store original pointer before aligned pointer
                reinterpret_cast<void**>(aligned)[-1] = base;
                ptr = aligned;
            }
        #endif
#endif

        if (ptr) [[likely]] {
            raw_ptr = static_cast<T*>(ptr);
            // Placement new for non-arithmetic types
            for (Size i = 0; i < count; ++i) {
                new (raw_ptr + i) T();
            }
        } else {
            return AlignedPtr<T>(nullptr, AlignedDeleter<T>(alignment));
        }
    }

    return AlignedPtr<T>(raw_ptr, AlignedDeleter<T>(alignment));
}

/// @brief Free aligned memory (legacy compatibility)
/// @tparam T Element type
/// @param[in] ptr Pointer to free
/// @param[in] alignment Alignment used during allocation
template<typename T>
SCL_FORCE_INLINE
void aligned_free(T* ptr, Size alignment = DEFAULT_ALIGNMENT) noexcept {
    AlignedDeleter<T>{alignment}(ptr);
}

// =============================================================================
// SECTION 4: Aligned Buffer (RAII Wrapper)
// =============================================================================

/// @brief RAII wrapper for aligned memory with span interface
/// @tparam T Element type
template<typename T>
class AlignedBuffer {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = Size;
    using span_type = std::span<T>;
    using const_span_type = std::span<const T>;

    /// @brief Construct buffer with specified size and alignment
    /// @param[in] count Number of elements
    /// @param[in] alignment Memory alignment in bytes
    explicit AlignedBuffer(Size count, Size alignment = DEFAULT_ALIGNMENT)
        : ptr_(aligned_alloc<T>(count, alignment)), count_(count) {}

    ~AlignedBuffer() = default;

    // Non-copyable
    AlignedBuffer(const AlignedBuffer&) = delete;
    auto operator=(const AlignedBuffer&) -> AlignedBuffer& = delete;

    // Movable
    AlignedBuffer(AlignedBuffer&&) noexcept = default;
    auto operator=(AlignedBuffer&&) noexcept -> AlignedBuffer& = default;

    // -------------------------------------------------------------------------
    // Span Interface
    // -------------------------------------------------------------------------

    /// @brief Get span view of entire buffer
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto span() noexcept -> span_type {
        return span_type(ptr_.get(), count_);
    }

    /// @brief Get const span view of entire buffer
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto span() const noexcept -> const_span_type {
        return const_span_type(ptr_.get(), count_);
    }

    /// @brief Implicit conversion to span
    [[nodiscard]]
    explicit operator span_type() noexcept { return span(); }

    /// @brief Implicit conversion to const span
    [[nodiscard]]
    explicit operator const_span_type() const noexcept { return span(); }

    // -------------------------------------------------------------------------
    // Element Access
    // -------------------------------------------------------------------------

    /// @brief Get raw pointer
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto data() noexcept -> pointer { return ptr_.get(); }

    /// @brief Get const raw pointer
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto data() const noexcept -> const_pointer { return ptr_.get(); }

    /// @brief Get buffer size
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto size() const noexcept -> size_type { return count_; }

    /// @brief Check if buffer is empty
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto empty() const noexcept -> bool { return count_ == 0; }

    /// @brief Check if buffer is valid
    [[nodiscard]]
    SCL_FORCE_INLINE
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    /// @brief Element access
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto operator[](Size i) noexcept -> reference { return ptr_[i]; }

    /// @brief Const element access
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto operator[](Size i) const noexcept -> const_reference { return ptr_[i]; }

    // -------------------------------------------------------------------------
    // Iterators
    // -------------------------------------------------------------------------

    [[nodiscard]] SCL_FORCE_INLINE auto begin() noexcept -> pointer { return ptr_.get(); }
    [[nodiscard]] SCL_FORCE_INLINE auto end() noexcept -> pointer { return ptr_.get() + count_; }
    [[nodiscard]] SCL_FORCE_INLINE auto begin() const noexcept -> const_pointer { return ptr_.get(); }
    [[nodiscard]] SCL_FORCE_INLINE auto end() const noexcept -> const_pointer { return ptr_.get() + count_; }
    [[nodiscard]] SCL_FORCE_INLINE auto cbegin() const noexcept -> const_pointer { return ptr_.get(); }
    [[nodiscard]] SCL_FORCE_INLINE auto cend() const noexcept -> const_pointer { return ptr_.get() + count_; }

private:
    AlignedPtr<T> ptr_;
    Size count_;
};

// =============================================================================
// SECTION 5: Large Memory Allocation (mmap/VirtualAlloc)
// =============================================================================

/// @brief Allocation flags for large memory operations
enum class AllocFlags : std::uint32_t {
    None        = 0,
    HugePages   = 1 << 0,   ///< Request huge pages (2MB on x86_64)
    Executable  = 1 << 1,   ///< Allow code execution
    ReadOnly    = 1 << 2,   ///< Read-only mapping
    NoReserve   = 1 << 3    ///< Don't reserve swap space (Linux)
};

/// @brief Bitwise OR for AllocFlags
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto operator|(AllocFlags a, AllocFlags b) noexcept -> AllocFlags {
    return static_cast<AllocFlags>(
        static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

/// @brief Bitwise AND for AllocFlags
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto operator&(AllocFlags a, AllocFlags b) noexcept -> AllocFlags {
    return static_cast<AllocFlags>(
        static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b));
}

/// @brief Check if flag is set
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto has_flag(AllocFlags flags, AllocFlags test) noexcept -> bool {
    return (static_cast<std::uint32_t>(flags) & static_cast<std::uint32_t>(test)) != 0;
}

/// @brief Allocate large memory block using OS virtual memory
/// @param[in] byte_size Size in bytes (will be rounded up to page boundary)
/// @param[in] flags Allocation flags
/// @return Pointer to allocated memory, nullptr on failure
/// @note Uses mmap on POSIX, VirtualAlloc on Windows
/// @note Memory is zero-initialized by the OS
[[nodiscard]]
inline
auto virtual_alloc(Size byte_size, AllocFlags flags = AllocFlags::None) noexcept -> void* {
    if (byte_size == 0) [[unlikely]] { return nullptr; };

#if SCL_PLATFORM_WINDOWS
    // Windows: VirtualAlloc
    DWORD alloc_type = MEM_COMMIT | MEM_RESERVE;
    DWORD protect = PAGE_READWRITE;
    
    if (has_flag(flags, AllocFlags::HugePages)) {
        alloc_type |= MEM_LARGE_PAGES;
        // Huge pages require SeLockMemoryPrivilege and aligned size
        byte_size = align_up(byte_size, HUGE_PAGE_SIZE);
    }
    
    if (has_flag(flags, AllocFlags::Executable)) {
        protect = PAGE_EXECUTE_READWRITE;
    } else if (has_flag(flags, AllocFlags::ReadOnly)) {
        protect = PAGE_READONLY;
    }
    
    return VirtualAlloc(nullptr, byte_size, alloc_type, protect);

#elif SCL_PLATFORM_POSIX
    // POSIX: mmap
    int prot = PROT_READ | PROT_WRITE;
    int map_flags = MAP_PRIVATE | MAP_ANONYMOUS;
    
    if (has_flag(flags, AllocFlags::Executable)) {
        prot |= PROT_EXEC;
    }
    if (has_flag(flags, AllocFlags::ReadOnly)) {
        prot = PROT_READ;
    }
    
#if SCL_PLATFORM_LINUX
    if (has_flag(flags, AllocFlags::HugePages)) {
        map_flags |= MAP_HUGETLB;
        byte_size = align_up(byte_size, HUGE_PAGE_SIZE);
    }
    if (has_flag(flags, AllocFlags::NoReserve)) {
        map_flags |= MAP_NORESERVE;
    }
#elif SCL_PLATFORM_MACOS
    // macOS uses VM_FLAGS_SUPERPAGE_SIZE_2MB via mach_vm_allocate
    // For simplicity, we skip huge pages on macOS in mmap
    SCL_UNUSED(HUGE_PAGE_SIZE);
#endif
    
    void* ptr = mmap(nullptr, byte_size, prot, map_flags, -1, 0);
    // MAP_FAILED is defined as ((void*)-1) in system headers - suppress warning
    if (ptr == MAP_FAILED) [[unlikely]] {  // NOLINT(performance-no-int-to-ptr,cppcoreguidelines-pro-type-cstyle-cast)
        return nullptr;
    }
    
    // Advise kernel about expected access pattern
#if SCL_PLATFORM_LINUX
    madvise(ptr, byte_size, MADV_WILLNEED);
#endif
    
    return ptr;

#else
    // Fallback: use aligned_alloc
    return aligned_alloc<std::byte>(byte_size, PAGE_SIZE).release();
#endif
}

/// @brief Free memory allocated with virtual_alloc
/// @param[in] ptr Pointer to memory
/// @param[in] byte_size Size in bytes (required for munmap on POSIX)
inline
void virtual_free(void* ptr, Size byte_size) noexcept {
    if (ptr == nullptr) [[unlikely]] { return; };

#if SCL_PLATFORM_WINDOWS
    SCL_UNUSED(byte_size);
    VirtualFree(ptr, 0, MEM_RELEASE);
#elif SCL_PLATFORM_POSIX
    munmap(ptr, byte_size);
#else
    SCL_UNUSED(byte_size);
    std::free(ptr);
#endif
}

/// @brief RAII wrapper for virtual memory allocation
/// @tparam T Element type
template<typename T>
class VirtualBuffer {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = Size;
    using span_type = std::span<T>;
    using const_span_type = std::span<const T>;

    /// @brief Construct buffer with specified size
    /// @param[in] count Number of elements
    /// @param[in] flags Allocation flags
    explicit VirtualBuffer(Size count, AllocFlags flags = AllocFlags::None)
        : count_(count) {
        if (count == 0) [[unlikely]] { return; };
        byte_size_ = align_up(count * sizeof(T), PAGE_SIZE);
        ptr_ = static_cast<T*>(virtual_alloc(byte_size_, flags));
    }

    ~VirtualBuffer() {
        if (ptr_) {
            virtual_free(ptr_, byte_size_);
        }
    }

    // Non-copyable
    VirtualBuffer(const VirtualBuffer&) = delete;
    auto operator=(const VirtualBuffer&) -> VirtualBuffer& = delete;

    // Movable
    VirtualBuffer(VirtualBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_), byte_size_(other.byte_size_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
        other.byte_size_ = 0;
    }

    auto operator=(VirtualBuffer&& other) noexcept -> VirtualBuffer& {
        if (this != &other) {
            if (ptr_) { virtual_free(ptr_, byte_size_); };
            ptr_ = other.ptr_;
            count_ = other.count_;
            byte_size_ = other.byte_size_;
            other.ptr_ = nullptr;
            other.count_ = 0;
            other.byte_size_ = 0;
        }
        return *this;
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    [[nodiscard]] SCL_FORCE_INLINE auto data() noexcept -> pointer { return ptr_; }
    [[nodiscard]] SCL_FORCE_INLINE auto data() const noexcept -> const_pointer { return ptr_; }
    [[nodiscard]] SCL_FORCE_INLINE auto size() const noexcept -> size_type { return count_; }
    [[nodiscard]] SCL_FORCE_INLINE auto byte_size() const noexcept -> size_type { return byte_size_; }
    [[nodiscard]] SCL_FORCE_INLINE auto empty() const noexcept -> bool { return count_ == 0; }
    [[nodiscard]] SCL_FORCE_INLINE explicit operator bool() const noexcept { return ptr_ != nullptr; }

    [[nodiscard]] SCL_FORCE_INLINE auto span() noexcept -> span_type { return {ptr_, count_}; }
    [[nodiscard]] SCL_FORCE_INLINE auto span() const noexcept -> const_span_type { return {ptr_, count_}; }

    [[nodiscard]] SCL_FORCE_INLINE auto operator[](Size i) noexcept -> T& { return ptr_[i]; }
    [[nodiscard]] SCL_FORCE_INLINE auto operator[](Size i) const noexcept -> const T& { return ptr_[i]; }

    [[nodiscard]] SCL_FORCE_INLINE auto begin() noexcept -> pointer { return ptr_; }
    [[nodiscard]] SCL_FORCE_INLINE auto end() noexcept -> pointer { return ptr_ + count_; }
    [[nodiscard]] SCL_FORCE_INLINE auto begin() const noexcept -> const_pointer { return ptr_; }
    [[nodiscard]] SCL_FORCE_INLINE auto end() const noexcept -> const_pointer { return ptr_ + count_; }

private:
    T* ptr_ = nullptr;
    Size count_ = 0;
    Size byte_size_ = 0;
};

// =============================================================================
// SECTION 6: Fill and Zero Operations
// =============================================================================

/// @brief Fill span with a value
/// @tparam T Element type
/// @param[out] dest Destination span
/// @param[in] value Value to fill with
/// @note Uses optimized paths for trivially copyable types
template<typename T>
SCL_FORCE_INLINE
void fill(std::span<T> dest, T value) {
    if (dest.empty()) [[unlikely]] { return; };

    if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 1) {
        // Single-byte: memset is optimal (uses AVX-512 + NT stores)
        std::memset(dest.data(), static_cast<unsigned char>(value), dest.size());
    } else if constexpr (std::is_trivially_copyable_v<T>) {
        // Multi-byte: let compiler auto-vectorize
        std::fill(dest.begin(), dest.end(), value);
    } else {
        // Non-trivially copyable: manual loop
        for (auto& elem : dest) {
            elem = value;
        }
    }
}

/// @brief Zero-initialize span
/// @tparam T Element type
/// @param[out] dest Destination span
template<typename T>
SCL_FORCE_INLINE
void zero(std::span<T> dest) {
    if constexpr (std::is_trivial_v<T>) {
        std::memset(dest.data(), 0, dest.size_bytes());
    } else {
        fill(dest, T{});
    }
}

// =============================================================================
// SECTION 7: Copy Operations
// =============================================================================

/// @brief Fast copy (no overlap allowed)
/// @tparam T Element type
/// @param[in] src Source span
/// @param[out] dest Destination span
/// @pre src.size() == dest.size()
/// @pre No overlap between src and dest
template<typename T>
SCL_FORCE_INLINE
void copy_fast(std::span<const T> src, std::span<T> dest) {
    SCL_DEBUG_ASSERT_MSG(src.size() == dest.size(), "copy_fast: size mismatch");
    SCL_DEBUG_ASSERT_MSG(
        src.data() + src.size() <= dest.data() || dest.data() + dest.size() <= src.data(),
        "copy_fast: overlap detected, use copy() instead"
    );

    if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(dest.data(), src.data(), src.size_bytes());
    } else {
        std::copy(src.begin(), src.end(), dest.begin());
    }
}

/// @brief Safe copy (handles overlap)
/// @tparam T Element type
/// @param[in] src Source span
/// @param[out] dest Destination span
/// @pre src.size() == dest.size()
template<typename T>
SCL_FORCE_INLINE
void copy(std::span<const T> src, std::span<T> dest) {
    SCL_DEBUG_ASSERT_MSG(src.size() == dest.size(), "copy: size mismatch");

    if (src.data() == dest.data()) [[unlikely]] { return; };

    if constexpr (std::is_trivially_copyable_v<T>) {
        std::memmove(dest.data(), src.data(), src.size_bytes());
    } else {
        if (dest.data() < src.data()) {
            std::copy(src.begin(), src.end(), dest.begin());
        } else {
            std::copy_backward(src.begin(), src.end(), dest.end());
        }
    }
}

/// @brief Stream copy using non-temporal stores (bypasses cache)
/// @tparam T Element type
/// @param[in] src Source span (must be 64-byte aligned)
/// @param[out] dest Destination span (must be 64-byte aligned)
/// @pre src.size() == dest.size()
/// @note Falls back to copy_fast for small arrays or unaligned data
/// @note Platform-specific SIMD streaming:
///       - x86 AVX-512: _mm512_stream_si512
///       - x86 AVX: _mm256_stream_si256
///       - x86 SSE2: _mm_stream_si128
///       - ARM NEON: vst1q (no true NT store, uses regular store)
template<typename T>
void stream_copy(std::span<const T> src, std::span<T> dest) {
    SCL_DEBUG_ASSERT_MSG(src.size() == dest.size(), "stream_copy: size mismatch");

    const auto byte_size = src.size_bytes();

    // Small arrays: use regular copy (NT stores have overhead)
    if (byte_size < STREAM_THRESHOLD) [[likely]] {
        copy_fast(src, dest);
        return;
    }

    // Check alignment
    const auto src_align = reinterpret_cast<std::uintptr_t>(src.data()) % STREAM_ALIGNMENT;
    const auto dest_align = reinterpret_cast<std::uintptr_t>(dest.data()) % STREAM_ALIGNMENT;

    if (src_align != 0 || dest_align != 0) [[unlikely]] {
        copy_fast(src, dest);
        return;
    }

    const auto* src_ptr = reinterpret_cast<const char*>(src.data());
    auto* dest_ptr = reinterpret_cast<char*>(dest.data());

#if SCL_ARCH_X86 && SCL_SIMD_AVX512
    // AVX-512: 64-byte streaming stores
    constexpr Size VECTOR_SIZE = 64;
    const Size vector_count = byte_size / VECTOR_SIZE;
    const Size remainder = byte_size % VECTOR_SIZE;
    
    for (Size i = 0; i < vector_count; ++i) {
        __m512i data = _mm512_load_si512(
            reinterpret_cast<const __m512i*>(src_ptr + i * VECTOR_SIZE));
        _mm512_stream_si512(
            reinterpret_cast<__m512i*>(dest_ptr + i * VECTOR_SIZE), data);
    }
    
    // Handle remainder
    if (remainder > 0) {
        std::memcpy(dest_ptr + vector_count * VECTOR_SIZE,
                    src_ptr + vector_count * VECTOR_SIZE, remainder);
    }

#elif SCL_ARCH_X86 && SCL_SIMD_AVX
    // AVX: 32-byte streaming stores
    constexpr Size VECTOR_SIZE = 32;
    const Size vector_count = byte_size / VECTOR_SIZE;
    const Size remainder = byte_size % VECTOR_SIZE;
    
    for (Size i = 0; i < vector_count; ++i) {
        __m256i data = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(src_ptr + i * VECTOR_SIZE));
        _mm256_stream_si256(
            reinterpret_cast<__m256i*>(dest_ptr + i * VECTOR_SIZE), data);
    }
    
    // Handle remainder
    if (remainder > 0) {
        std::memcpy(dest_ptr + vector_count * VECTOR_SIZE,
                    src_ptr + vector_count * VECTOR_SIZE, remainder);
    }

#elif SCL_ARCH_X86 && SCL_SIMD_SSE2
    // SSE2: 16-byte streaming stores
    constexpr Size VECTOR_SIZE = 16;
    const Size vector_count = byte_size / VECTOR_SIZE;
    const Size remainder = byte_size % VECTOR_SIZE;
    
    for (Size i = 0; i < vector_count; ++i) {
        __m128i data = _mm_load_si128(
            reinterpret_cast<const __m128i*>(src_ptr + i * VECTOR_SIZE));
        _mm_stream_si128(
            reinterpret_cast<__m128i*>(dest_ptr + i * VECTOR_SIZE), data);
    }
    
    // Handle remainder
    if (remainder > 0) {
        std::memcpy(dest_ptr + vector_count * VECTOR_SIZE,
                    src_ptr + vector_count * VECTOR_SIZE, remainder);
    }

#elif SCL_ARCH_ARM && SCL_SIMD_NEON
    // ARM NEON: No true non-temporal stores, but we can use regular NEON stores
    // with cache-line prefetch hints
    constexpr Size VECTOR_SIZE = 16;
    const Size vector_count = byte_size / VECTOR_SIZE;
    const Size remainder = byte_size % VECTOR_SIZE;
    
    for (Size i = 0; i < vector_count; ++i) {
        // Prefetch next cache line
        if (i + 4 < vector_count) {
            SCL_PREFETCH_NTA(src_ptr + (i + 4) * VECTOR_SIZE);
        }
        
        uint8x16_t data = vld1q_u8(
            reinterpret_cast<const uint8_t*>(src_ptr + i * VECTOR_SIZE));
        vst1q_u8(
            reinterpret_cast<uint8_t*>(dest_ptr + i * VECTOR_SIZE), data);
    }
    
    // Handle remainder
    if (remainder > 0) {
        std::memcpy(dest_ptr + vector_count * VECTOR_SIZE,
                    src_ptr + vector_count * VECTOR_SIZE, remainder);
    }

#else
    // Fallback: delegate to memcpy (may use NT stores internally on modern libc)
    std::memcpy(dest_ptr, src_ptr, byte_size);
#endif

    // Memory fence to ensure all streaming stores are globally visible
    // This is required after non-temporal stores
#if SCL_ARCH_X86
    _mm_sfence();
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

// =============================================================================
// SECTION 8: Prefetch Utilities
// =============================================================================

/// @brief Prefetch span for reading
/// @tparam Locality Cache locality hint (0=NTA, 1=L3, 2=L2, 3=L1)
/// @tparam T Element type
/// @param[in] src Span to prefetch
/// @param[in] max_prefetches Maximum prefetch operations
template<int Locality = 3, typename T>
SCL_FORCE_INLINE
void prefetch_read(
    std::span<const T> src,
    Size max_prefetches = DEFAULT_MAX_PREFETCHES
) {
    static_assert(Locality >= 0 && Locality <= 3, "Locality must be 0-3");

    const auto* p = reinterpret_cast<const char*>(src.data());
    const auto* end = p + src.size_bytes();

    Size count = 0;
    for (; p < end && count < max_prefetches; p += CACHE_LINE_SIZE, ++count) {
        SCL_PREFETCH_READ(p, Locality);
    }
}

/// @brief Prefetch span for writing
/// @tparam Locality Cache locality hint (0=NTA, 1=L3, 2=L2, 3=L1)
/// @tparam T Element type
/// @param[in] dest Span to prefetch
/// @param[in] max_prefetches Maximum prefetch operations
template<int Locality = 3, typename T>
SCL_FORCE_INLINE
void prefetch_write(
    std::span<T> dest,
    Size max_prefetches = DEFAULT_MAX_PREFETCHES
) {
    static_assert(Locality >= 0 && Locality <= 3, "Locality must be 0-3");

    auto* p = reinterpret_cast<char*>(dest.data());
    const auto* end = p + dest.size_bytes();

    Size count = 0;
    for (; p < end && count < max_prefetches; p += CACHE_LINE_SIZE, ++count) {
        SCL_PREFETCH_WRITE(p, Locality);
    }
}

/// @brief Prefetch ahead during iteration
/// @tparam T Element type
/// @tparam Distance Prefetch distance in elements
/// @param[in] src Source span
/// @param[in] current_idx Current iteration index
template<typename T, Size Distance = DEFAULT_PREFETCH_DISTANCE>
SCL_FORCE_INLINE
void prefetch_ahead(std::span<const T> src, Size current_idx) {
    const Size ahead_idx = current_idx + Distance;
    if (ahead_idx < src.size()) [[likely]] {
        SCL_PREFETCH_READ(src.data() + ahead_idx, 0);
    }
}

// =============================================================================
// SECTION 9: Memory Comparison
// =============================================================================

/// @brief Compare two spans for equality
/// @tparam T Element type
/// @param[in] a First span
/// @param[in] b Second span
/// @return true if spans have same content
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
auto equal(std::span<const T> a, std::span<const T> b) -> bool {
    if (a.size() != b.size()) [[unlikely]] { return false; };
    if (a.data() == b.data()) [[unlikely]] { return true; };
    if (a.empty()) [[unlikely]] { return true; };

    if constexpr (std::is_trivially_copyable_v<T>) {
        return std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
    } else {
        return std::equal(a.begin(), a.end(), b.begin());
    }
}

/// @brief Lexicographic comparison of two spans
/// @tparam T Element type
/// @param[in] a First span
/// @param[in] b Second span
/// @return -1 if a < b, 0 if a == b, 1 if a > b
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
auto compare(std::span<const T> a, std::span<const T> b) -> int {
    if constexpr (std::is_trivially_copyable_v<T> && std::is_arithmetic_v<T>) {
        const Size min_len = std::min(a.size(), b.size());
        if (min_len > 0) {
            const int cmp = std::memcmp(a.data(), b.data(), min_len * sizeof(T));
            if (cmp != 0) { return (cmp < 0) ? -1 : 1; };
        }
    } else {
        const Size min_len = std::min(a.size(), b.size());
        for (Size i = 0; i < min_len; ++i) {
            if (a[i] < b[i]) { return -1; };
            if (a[i] > b[i]) { return 1; };
        }
    }

    if (a.size() < b.size()) { return -1; };
    if (a.size() > b.size()) { return 1; };
    return 0;
}

// =============================================================================
// SECTION 10: Swap Operations
// =============================================================================

/// @brief Swap two values
/// @tparam T Element type
/// @param[in,out] a First value
/// @param[in,out] b Second value
template<typename T>
SCL_FORCE_INLINE
void swap(T& a, T& b) noexcept {
    T tmp = static_cast<T&&>(a);
    a = static_cast<T&&>(b);
    b = static_cast<T&&>(tmp);
}

/// @brief Swap contents of two spans
/// @tparam T Element type
/// @param[in,out] a First span
/// @param[in,out] b Second span
/// @pre a.size() == b.size()
/// @pre No overlap between a and b
template<typename T>
void swap_ranges(std::span<T> a, std::span<T> b) {
    SCL_DEBUG_ASSERT_MSG(a.size() == b.size(), "swap_ranges: size mismatch");
    if (a.data() == b.data()) [[unlikely]] { return; }
    SCL_DEBUG_ASSERT_MSG(
        a.data() + a.size() <= b.data() || b.data() + b.size() <= a.data(),
        "swap_ranges: overlap detected"
    );

    std::swap_ranges(a.begin(), a.end(), b.begin());
}

// =============================================================================
// SECTION 11: Reverse Operations
// =============================================================================

/// @brief Reverse span in-place
/// @tparam T Element type
/// @param[in,out] data Span to reverse
template<typename T>
void reverse(std::span<T> data) {
    if (data.size() <= 1) [[unlikely]] { return; };
    std::reverse(data.begin(), data.end());
}

/// @brief Copy reversed span
/// @tparam T Element type
/// @param[in] src Source span
/// @param[out] dest Destination span
/// @pre src.size() == dest.size()
template<typename T>
SCL_FORCE_INLINE
void reverse_copy(std::span<const T> src, std::span<T> dest) {
    SCL_DEBUG_ASSERT_MSG(src.size() == dest.size(), "reverse_copy: size mismatch");
    std::reverse_copy(src.begin(), src.end(), dest.begin());
}

// =============================================================================
// SECTION 12: Utility Functions
// =============================================================================

/// @brief Check if pointer is aligned
/// @param[in] ptr Pointer to check
/// @param[in] alignment Required alignment
/// @return true if aligned
[[nodiscard]]
SCL_FORCE_INLINE
auto is_aligned(const void* ptr, Size alignment) noexcept -> bool {
    return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

// Note: align_up and align_down are defined in SECTION 2 (before large memory allocation)

/// @brief Calculate number of cache lines covered
/// @param[in] bytes Number of bytes
/// @return Number of cache lines
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto cache_lines(Size bytes) noexcept -> Size {
    return align_up(bytes, CACHE_LINE_SIZE) / CACHE_LINE_SIZE;
}

/// @brief Calculate number of pages covered
/// @param[in] bytes Number of bytes
/// @return Number of pages
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto pages(Size bytes) noexcept -> Size {
    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    return align_up(bytes, PAGE_SIZE) / PAGE_SIZE;
}

// =============================================================================
// SECTION 13: Memory Advice and Locking
// =============================================================================

/// @brief Memory advice hints
enum class MemoryAdvice : std::uint32_t {
    Normal,         ///< Default access pattern
    Sequential,     ///< Sequential access pattern (prefetch ahead)
    Random,         ///< Random access pattern (disable prefetch)
    WillNeed,       ///< Will access soon (prefetch into cache)
    DontNeed,       ///< Won't access soon (can be paged out)
    Free            ///< Memory can be freed (Linux MADV_FREE)
};

/// @brief Advise kernel about expected memory access pattern
/// @param[in] ptr Pointer to memory region
/// @param[in] byte_size Size of region in bytes
/// @param[in] advice Memory advice hint
/// @return true on success
/// @note Uses madvise on POSIX, VirtualAlloc hints on Windows
[[nodiscard]]
inline
auto memory_advise(void* ptr, Size byte_size, MemoryAdvice advice) noexcept -> bool {
    if (ptr == nullptr || byte_size == 0) [[unlikely]] {
      return false;
    }

#if SCL_PLATFORM_LINUX
    int linux_advice = MADV_NORMAL;
    switch (advice) {
        case MemoryAdvice::Normal:     linux_advice = MADV_NORMAL; break;
        case MemoryAdvice::Sequential: linux_advice = MADV_SEQUENTIAL; break;
        case MemoryAdvice::Random:     linux_advice = MADV_RANDOM; break;
        case MemoryAdvice::WillNeed:   linux_advice = MADV_WILLNEED; break;
        case MemoryAdvice::DontNeed:   linux_advice = MADV_DONTNEED; break;
        case MemoryAdvice::Free:
#ifdef MADV_FREE
            linux_advice = MADV_FREE;
#else
            linux_advice = MADV_DONTNEED;
#endif
            break;
    }
    return madvise(ptr, byte_size, linux_advice) == 0;

#elif SCL_PLATFORM_MACOS
    int macos_advice = MADV_NORMAL;
    switch (advice) {
        case MemoryAdvice::Normal:     macos_advice = MADV_NORMAL; break;
        case MemoryAdvice::Sequential: macos_advice = MADV_SEQUENTIAL; break;
        case MemoryAdvice::Random:     macos_advice = MADV_RANDOM; break;
        case MemoryAdvice::WillNeed:   macos_advice = MADV_WILLNEED; break;
        case MemoryAdvice::DontNeed:   macos_advice = MADV_DONTNEED; break;
        case MemoryAdvice::Free:       macos_advice = MADV_FREE; break;
    }
    return madvise(ptr, byte_size, macos_advice) == 0;

#elif SCL_PLATFORM_WINDOWS
    // Windows doesn't have direct madvise equivalent
    // Use VirtualAlloc with MEM_RESET for DontNeed
    if (advice == MemoryAdvice::DontNeed || advice == MemoryAdvice::Free) {
        return VirtualAlloc(ptr, byte_size, MEM_RESET, PAGE_READWRITE) != nullptr;
    }
    // Other hints are no-op on Windows
    return true;

#else
    SCL_UNUSED(ptr);
    SCL_UNUSED(byte_size);
    SCL_UNUSED(advice);
    return true;
#endif
}

/// @brief Lock memory pages into RAM (prevent paging)
/// @param[in] ptr Pointer to memory region
/// @param[in] byte_size Size of region in bytes
/// @return true on success
/// @note Uses mlock on POSIX, VirtualLock on Windows
/// @warning Requires appropriate privileges
[[nodiscard]]
inline
auto memory_lock(void* ptr, Size byte_size) noexcept -> bool {
    if (ptr == nullptr || byte_size == 0) [[unlikely]] {
      return false;
    }

#if SCL_PLATFORM_WINDOWS
    return VirtualLock(ptr, byte_size) != 0;
#elif SCL_PLATFORM_POSIX
    return mlock(ptr, byte_size) == 0;
#else
    SCL_UNUSED(ptr);
    SCL_UNUSED(byte_size);
    return false;
#endif
}

/// @brief Unlock memory pages (allow paging)
/// @param[in] ptr Pointer to memory region
/// @param[in] byte_size Size of region in bytes
/// @return true on success
inline
auto memory_unlock(void* ptr, Size byte_size) noexcept -> bool {
    if (ptr == nullptr || byte_size == 0) [[unlikely]] {
      return false;
    }

#if SCL_PLATFORM_WINDOWS
    return VirtualUnlock(ptr, byte_size) != 0;
#elif SCL_PLATFORM_POSIX
    return munlock(ptr, byte_size) == 0;
#else
    SCL_UNUSED(ptr);
    SCL_UNUSED(byte_size);
    return false;
#endif
}

// =============================================================================
// SECTION 14: Memory Fence Operations
// =============================================================================

/// @brief Store fence - ensure all prior stores are visible
SCL_FORCE_INLINE
void store_fence() noexcept {
#if SCL_ARCH_X86
    _mm_sfence();
#elif SCL_ARCH_ARM
    #if SCL_COMPILER_GCC_LIKE
        __asm__ __volatile__("dmb ishst" ::: "memory");
    #else
        std::atomic_thread_fence(std::memory_order_release);
    #endif
#else
    std::atomic_thread_fence(std::memory_order_release);
#endif
}

/// @brief Load fence - ensure all prior loads are complete
SCL_FORCE_INLINE
void load_fence() noexcept {
#if SCL_ARCH_X86
    _mm_lfence();
#elif SCL_ARCH_ARM
    #if SCL_COMPILER_GCC_LIKE
        __asm__ __volatile__("dmb ishld" ::: "memory");
    #else
        std::atomic_thread_fence(std::memory_order_acquire);
    #endif
#else
    std::atomic_thread_fence(std::memory_order_acquire);
#endif
}

/// @brief Full memory fence - ensure ordering of all memory operations
SCL_FORCE_INLINE
void memory_fence() noexcept {
#if SCL_ARCH_X86
    _mm_mfence();
#elif SCL_ARCH_ARM
    #if SCL_COMPILER_GCC_LIKE
        __asm__ __volatile__("dmb ish" ::: "memory");
    #else
        std::atomic_thread_fence(std::memory_order_seq_cst);
    #endif
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

// =============================================================================
// SECTION 15: Page Size Query
// =============================================================================

/// @brief Get system page size at runtime
/// @return Page size in bytes
[[nodiscard]]
SCL_FORCE_INLINE
auto get_page_size() noexcept -> Size {
#if SCL_PLATFORM_WINDOWS
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<Size>(si.dwPageSize);
#elif SCL_PLATFORM_POSIX
    const auto page_size = sysconf(_SC_PAGESIZE);
    return (page_size > 0) ? static_cast<Size>(page_size) : PAGE_SIZE;
#else
    return PAGE_SIZE;
#endif
}

/// @brief Get huge page size at runtime
/// @return Huge page size in bytes, or 0 if not supported
[[nodiscard]]
SCL_FORCE_INLINE
auto get_huge_page_size() noexcept -> Size {
#if SCL_PLATFORM_WINDOWS
    return GetLargePageMinimum();
#elif SCL_PLATFORM_LINUX
    // Read from /proc/meminfo or use sysconf
    #ifdef _SC_LARGE_PAGESIZE
        const auto huge_size = sysconf(_SC_LARGE_PAGESIZE);
        return (huge_size > 0) ? static_cast<Size>(huge_size) : HUGE_PAGE_SIZE;
    #else
        return HUGE_PAGE_SIZE;
    #endif
#else
    return HUGE_PAGE_SIZE;
#endif
}

}  // namespace scl::memory

