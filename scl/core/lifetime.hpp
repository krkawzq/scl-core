#pragma once

#include "scl/config.hpp"
#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"

#include <cstdlib>  // malloc, free, aligned_alloc
#include <cstring>  // memcpy, memset
#include <utility>  // std::exchange
#include <new>      // std::bad_alloc

// =============================================================================
/// @file lifetime.hpp
/// @brief Memory Lifetime Management for C/Python Interop
///
/// Provides RAII-safe memory handles for cross-language boundary operations.
///
/// Core Concepts:
///
/// 1. View vs Owner:
///    - View: Borrowed memory (no-op deleter)
///    - Owner: Owned memory (will be freed on destruction)
///
/// 2. Handoff Protocol:
///    - release(): Transfer ownership to external code
///    - Critical for C++ -> Python memory passing
///
/// 3. Zero-Copy:
///    - Views enable zero-copy data sharing
///    - No intermediate buffers needed
///
/// Use Cases:
///
/// Python -> C++ (View):
/// void scl_process(float* py_ptr, size_t size) {
///     auto view = scl::core::mem::view(py_ptr, size * sizeof(float));
///     // ... use view ...
///     // Destructor does nothing (borrowed memory)
/// }
///
/// C++ -> Python (Handoff):
/// float* scl_create_array(size_t n) {
///     auto owned = scl::core::mem::alloc(n * sizeof(float));
///     // ... fill data ...
///     return owned.release<float>();  // Python now owns this!
/// }
///
/// Internal C++ (RAII):
/// void internal_function() {
///     auto temp = scl::core::mem::alloc_aligned(1024, 64);
///     // ... use temp ...
///     // Automatic cleanup on scope exit
/// }
// =============================================================================

namespace scl::core {

// =============================================================================
// Deleter Function Types
// =============================================================================

/// @brief Function pointer for memory cleanup.
///
/// Signature: void deleter(void* ptr)
/// - Standard: std::free
/// - Aligned: _aligned_free (MSVC) or std::free (POSIX)
/// - No-op: nullptr or no_op_deleter (for Views)
using DeleterFn = void(*)(void*);

/// @brief Standard C free deleter.
inline void standard_deleter(void* ptr) {
    std::free(ptr);
}

/// @brief No-op deleter for borrowed memory (Views).
inline void no_op_deleter(void* /* ptr */) {
    // Intentionally empty - borrowed memory is not freed
}

#if defined(_MSC_VER)
/// @brief MSVC aligned memory deleter.
inline void aligned_deleter_msvc(void* ptr) {
    _aligned_free(ptr);
}
#endif

// =============================================================================
// Memory Handle (RAII-Safe Memory Wrapper)
// =============================================================================

/// @brief RAII handle for memory blocks with explicit lifetime control.
///
/// Features:
/// - Move-only (unique ownership)
/// - Automatic cleanup (unless released)
/// - Type-safe accessors
/// - Support for aligned allocations
/// - Integration with SCL error system
///
/// Memory Model:
/// [ptr] -> [size bytes of memory]
///          |
///    [deleter(ptr)] called on destruction
struct MemHandle {
    void* ptr;              ///< Pointer to memory block
    Size size;              ///< Size in bytes
    DeleterFn deleter;      ///< Cleanup function (nullptr = View)

    // -------------------------------------------------------------------------
    // Constructors & Destructor
    // -------------------------------------------------------------------------

    /// @brief Default constructor (null handle).
    constexpr MemHandle() noexcept 
        : ptr(nullptr), size(0), deleter(nullptr) {}

    /// @brief Construct from raw pointer with explicit deleter.
    ///
    /// @param p Memory pointer
    /// @param s Size in bytes
    /// @param d Deleter function (nullptr = View/No cleanup)
    constexpr MemHandle(void* p, Size s, DeleterFn d = nullptr) noexcept
        : ptr(p), size(s), deleter(d) {}

    /// @brief Destructor - Invokes deleter if owning.
    ~MemHandle() {
        reset();
    }

    // -------------------------------------------------------------------------
    // Move Semantics (Ownership Transfer)
    // -------------------------------------------------------------------------

    /// @brief Move constructor.
    MemHandle(MemHandle&& other) noexcept 
        : ptr(std::exchange(other.ptr, nullptr)),
          size(std::exchange(other.size, 0)),
          deleter(std::exchange(other.deleter, nullptr)) {}

    /// @brief Move assignment.
    MemHandle& operator=(MemHandle&& other) noexcept {
        if (this != &other) {
            reset();  // Clean up current memory
            ptr = std::exchange(other.ptr, nullptr);
            size = std::exchange(other.size, 0);
            deleter = std::exchange(other.deleter, nullptr);
        }
        return *this;
    }

    // Disable Copy (Prevent double-free)
    MemHandle(const MemHandle&) = delete;
    MemHandle& operator=(const MemHandle&) = delete;

    // -------------------------------------------------------------------------
    // Lifetime Control
    // -------------------------------------------------------------------------

    /// @brief Free the memory immediately (if owned).
    ///
    /// After reset(), the handle becomes null.
    /// Safe to call multiple times (idempotent).
    void reset() noexcept {
        if (ptr && deleter) {
            deleter(ptr);
        }
        ptr = nullptr;
        size = 0;
        deleter = nullptr;
    }

    /// @brief Release ownership without freeing memory.
    ///
    /// CRITICAL FOR HANDOFF: Use this when transferring ownership
    /// to external code (Python, C, another library).
    ///
    /// After release(), the handle becomes null and will NOT free memory
    /// on destruction. The caller is now responsible for cleanup!
    ///
    /// @return The raw pointer (caller must free it)
    [[nodiscard]] void* release() noexcept {
        void* p = ptr;
        ptr = nullptr;
        size = 0;
        deleter = nullptr;
        return p;
    }

    /// @brief Typed release for convenience.
    ///
    /// @tparam T Target type
    /// @return Typed pointer
    template <typename T>
    [[nodiscard]] T* release() noexcept {
        return static_cast<T*>(release());
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    /// @brief Check if handle is null or empty.
    [[nodiscard]] bool empty() const noexcept { 
        return ptr == nullptr || size == 0; 
    }

    /// @brief Check if handle owns the memory (will free on destruction).
    [[nodiscard]] bool is_owning() const noexcept { 
        return deleter != nullptr && deleter != no_op_deleter; 
    }

    /// @brief Get raw pointer.
    [[nodiscard]] void* data() const noexcept { 
        return ptr; 
    }

    /// @brief Get size in bytes.
    [[nodiscard]] Size bytes() const noexcept { 
        return size; 
    }

    /// @brief Typed accessor with bounds check.
    ///
    /// @tparam T Element type
    /// @return Typed pointer (or nullptr if empty)
    template <typename T>
    [[nodiscard]] T* as() const noexcept {
#if !defined(NDEBUG)
        // Debug check: Verify alignment
        if (ptr && reinterpret_cast<std::uintptr_t>(ptr) % alignof(T) != 0) {
            SCL_ASSERT(false, "MemHandle: Pointer not aligned for type T");
        }
        // Debug check: Verify size is multiple of T
        if (size > 0 && size % sizeof(T) != 0) {
            SCL_ASSERT(false, "MemHandle: Size not multiple of sizeof(T)");
        }
#endif
        return static_cast<T*>(ptr);
    }

    /// @brief Get number of elements of type T that fit in this memory.
    ///
    /// @tparam T Element type
    /// @return Number of T elements
    template <typename T>
    [[nodiscard]] Size count() const noexcept {
        return size / sizeof(T);
    }

    /// @brief Create a Span view of the memory.
    ///
    /// @tparam T Element type
    /// @return Span<T> wrapping the memory
    template <typename T>
    [[nodiscard]] Span<T> as_span() const noexcept {
        return Span<T>(as<T>(), count<T>());
    }

    /// @brief Create a mutable Span view.
    template <typename T>
    [[nodiscard]] MutableSpan<T> as_mutable_span() const noexcept {
        return MutableSpan<T>(as<T>(), count<T>());
    }
};

// =============================================================================
// Memory Allocation Factories
// =============================================================================

namespace mem {

// -----------------------------------------------------------------------------
// View Creation (Zero-Copy Borrowing)
// -----------------------------------------------------------------------------

/// @brief Create a non-owning view of existing memory.
///
/// The handle will NOT free the memory on destruction.
/// Use for wrapping Python/NumPy arrays or stack memory.
///
/// @param ptr Pointer to borrowed memory
/// @param bytes Size in bytes
/// @return View handle (non-owning)
inline MemHandle view(void* ptr, Size bytes) {
    return MemHandle(ptr, bytes, no_op_deleter);
}

/// @brief Create a typed view.
template <typename T>
inline MemHandle view(T* ptr, Size count) {
    return MemHandle(ptr, count * sizeof(T), no_op_deleter);
}

// -----------------------------------------------------------------------------
// Owned Memory Allocation
// -----------------------------------------------------------------------------

/// @brief Allocate uninitialized memory (standard malloc).
///
/// @param bytes Size in bytes
/// @return Owned handle (will be freed on destruction)
/// @throws scl::RuntimeError if allocation fails
inline MemHandle alloc(Size bytes) {
    if (bytes == 0) return MemHandle();
    
    void* p = std::malloc(bytes);
    if (SCL_UNLIKELY(!p)) {
        throw RuntimeError("MemHandle: malloc failed for " + 
                          std::to_string(bytes) + " bytes");
    }
    
    return MemHandle(p, bytes, standard_deleter);
}

/// @brief Allocate zero-initialized memory (calloc).
///
/// @param bytes Size in bytes
/// @return Owned handle with zeroed memory
/// @throws scl::RuntimeError if allocation fails
inline MemHandle alloc_zero(Size bytes) {
    if (bytes == 0) return MemHandle();
    
    void* p = std::calloc(1, bytes);
    if (SCL_UNLIKELY(!p)) {
        throw RuntimeError("MemHandle: calloc failed for " + 
                          std::to_string(bytes) + " bytes");
    }
    
    return MemHandle(p, bytes, standard_deleter);
}

/// @brief Allocate aligned memory for SIMD operations.
///
/// Alignment must be a power of 2 (typically 16, 32, 64, 128).
///
/// Platform Behavior:
/// - POSIX: Uses aligned_alloc (size must be multiple of alignment)
/// - MSVC: Uses _aligned_malloc (no size restriction)
///
/// @param bytes Size in bytes
/// @param alignment Alignment requirement (default 64 for AVX-512)
/// @return Owned handle with aligned memory
/// @throws scl::RuntimeError if allocation fails
inline MemHandle alloc_aligned(Size bytes, Size alignment = 64) {
    if (bytes == 0) return MemHandle();
    
    // Validate alignment (must be power of 2)
#if !defined(NDEBUG)
    SCL_ASSERT((alignment & (alignment - 1)) == 0, 
               "MemHandle: Alignment must be power of 2");
#endif
    
    void* p = nullptr;
    
#if defined(_MSC_VER)
    // MSVC: _aligned_malloc allows any size
    p = _aligned_malloc(bytes, alignment);
    if (SCL_UNLIKELY(!p)) {
        throw RuntimeError("MemHandle: _aligned_malloc failed for " + 
                          std::to_string(bytes) + " bytes");
    }
    return MemHandle(p, bytes, aligned_deleter_msvc);
    
#else
    // POSIX: aligned_alloc requires size to be multiple of alignment
    Size padded_size = ((bytes + alignment - 1) / alignment) * alignment;
    
    p = std::aligned_alloc(alignment, padded_size);
    if (SCL_UNLIKELY(!p)) {
        throw RuntimeError("MemHandle: aligned_alloc failed for " + 
                          std::to_string(bytes) + " bytes");
    }
    return MemHandle(p, bytes, standard_deleter);
#endif
}

/// @brief Allocate zero-initialized aligned memory.
///
/// @param bytes Size in bytes
/// @param alignment Alignment requirement
/// @return Owned handle with zeroed aligned memory
inline MemHandle alloc_aligned_zero(Size bytes, Size alignment = 64) {
    auto handle = alloc_aligned(bytes, alignment);
    if (!handle.empty()) {
        std::memset(handle.data(), 0, bytes);
    }
    return handle;
}

// -----------------------------------------------------------------------------
// Typed Allocation (Convenience Wrappers)
// -----------------------------------------------------------------------------

/// @brief Allocate array of N elements of type T.
///
/// @tparam T Element type
/// @param count Number of elements
/// @return Owned handle
template <typename T>
inline MemHandle alloc_array(Size count) {
    return alloc(count * sizeof(T));
}

/// @brief Allocate zero-initialized array of N elements of type T.
///
/// @tparam T Element type
/// @param count Number of elements
/// @return Owned handle with zeroed memory
template <typename T>
inline MemHandle alloc_array_zero(Size count) {
    return alloc_zero(count * sizeof(T));
}

/// @brief Allocate aligned array for SIMD operations.
///
/// @tparam T Element type
/// @param count Number of elements
/// @param alignment Alignment requirement
/// @return Owned handle with aligned memory
template <typename T>
inline MemHandle alloc_array_aligned(Size count, Size alignment = 64) {
    return alloc_aligned(count * sizeof(T), alignment);
}

// -----------------------------------------------------------------------------
// Memory Operations
// -----------------------------------------------------------------------------

/// @brief Copy data from source to destination handle.
///
/// @param src Source handle
/// @param dst Destination handle
/// @throws scl::DimensionError if sizes don't match
inline void copy(const MemHandle& src, MemHandle& dst) {
    SCL_CHECK_DIM(src.bytes() == dst.bytes(), 
                  "MemHandle: Cannot copy - size mismatch");
    SCL_CHECK_ARG(!src.empty() && !dst.empty(), 
                  "MemHandle: Cannot copy - null handle");
    
    std::memcpy(dst.data(), src.data(), src.bytes());
}

/// @brief Clone a memory handle (deep copy).
///
/// Creates a new owned handle with copied data.
///
/// @param src Source handle to clone
/// @return New owned handle with copied data
inline MemHandle clone(const MemHandle& src) {
    if (src.empty()) return MemHandle();
    
    auto dst = alloc(src.bytes());
    std::memcpy(dst.data(), src.data(), src.bytes());
    return dst;
}

/// @brief Clone with aligned memory.
///
/// @param src Source handle
/// @param alignment Alignment requirement
/// @return New aligned owned handle
inline MemHandle clone_aligned(const MemHandle& src, Size alignment = 64) {
    if (src.empty()) return MemHandle();
    
    auto dst = alloc_aligned(src.bytes(), alignment);
    std::memcpy(dst.data(), src.data(), src.bytes());
    return dst;
}

/// @brief Zero out memory.
///
/// @param handle Handle to zero
inline void zero(MemHandle& handle) {
    if (!handle.empty()) {
        std::memset(handle.data(), 0, handle.bytes());
    }
}

} // namespace mem

// =============================================================================
// Scoped Handle (Stack-Based RAII)
// =============================================================================

/// @brief Convenience alias for clearer intent in local scopes.
///
/// Use ScopedMem when you want to emphasize RAII behavior:
/// void function() {
///     ScopedMem temp = mem::alloc(1024);
///     // ... use temp ...
///     // Automatic cleanup here
/// }
using ScopedMem = MemHandle;

// =============================================================================
// C API Helpers (For Python Binding)
// =============================================================================

namespace c_api {

/// @brief Package C++ allocated memory for Python handoff.
///
/// Returns a POD struct that can be safely passed through C ABI.
///
/// CRITICAL: The returned pointer is now orphaned!
/// Python MUST call the corresponding free function, or leak occurs.
struct CMemBlock {
    void* ptr;
    Size size;
    int deleter_type;  // 0=standard, 1=aligned_msvc, 2=no_op
};

/// @brief Create CMemBlock from MemHandle (releases ownership).
///
/// @param handle Handle to package (will be released!)
/// @return C-compatible memory block descriptor
inline CMemBlock package_for_python(MemHandle& handle) {
    int deleter_type = 0;  // Standard by default
    
    if (handle.deleter == no_op_deleter || handle.deleter == nullptr) {
        deleter_type = 2;  // No-op (view)
    }
#if defined(_MSC_VER)
    else if (handle.deleter == aligned_deleter_msvc) {
        deleter_type = 1;  // MSVC aligned
    }
#endif
    
    Size s = handle.bytes();
    void* p = handle.release();  // Use the returned pointer directly
    
    return CMemBlock{p, s, deleter_type};
}

/// @brief Free memory from CMemBlock (for Python cleanup).
///
/// Python should call this via C API when done with the memory.
///
/// @param block Block to free
inline void free_cmemblock(const CMemBlock& block) {
    if (!block.ptr) return;
    
    switch (block.deleter_type) {
        case 0:  // Standard
            std::free(block.ptr);
            break;
#if defined(_MSC_VER)
        case 1:  // MSVC aligned
            _aligned_free(block.ptr);
            break;
#endif
        case 2:  // No-op (view)
            // Do nothing
            break;
        default:
            // Unknown deleter type - log warning but don't crash
            break;
    }
}

} // namespace c_api

} // namespace scl::core

// =============================================================================
// Example Usage Patterns
// =============================================================================
//
// 1. Python Input (Zero-Copy View):
// extern "C" int scl_process_array(float* data, size_t n) {
//     auto view = scl::core::mem::view(data, n * sizeof(float));
//     auto span = view.as_span<float>();
//     // ... process span ...
//     return 0;  // view destructor does nothing
// }
//
// 2. C++ Output to Python (Handoff):
// extern "C" float* scl_create_array(size_t n, int* out_size) {
//     auto owned = scl::core::mem::alloc_array<float>(n);
//     float* arr = owned.as<float>();
//     // ... fill arr ...
//     *out_size = n;
//     return owned.release<float>();  // Python now owns!
// }
//
// extern "C" void scl_free_array(float* ptr) {
//     std::free(ptr);  // Python calls this when done
// }
//
// 3. Internal Temporary Buffer (RAII):
// void internal_computation() {
//     auto temp = scl::core::mem::alloc_aligned_zero(4096, 64);
//     auto workspace = temp.as_span<double>();
//     // ... use workspace ...
//     // Automatic cleanup on return/exception
// }
//
// 4. Workspace with Manual Control:
// scl::core::MemHandle workspace;
// 
// void initialize() {
//     workspace = scl::core::mem::alloc_aligned(1024 * 1024, 64);
// }
//
// void process() {
//     if (workspace.empty()) initialize();
//     auto span = workspace.as_span<float>();
//     // ... use span ...
// }
//
// void cleanup() {
//     workspace.reset();  // Explicit cleanup
// }
// =============================================================================

