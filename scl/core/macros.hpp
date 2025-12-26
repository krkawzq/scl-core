#pragma once

#include "scl/config.hpp"
#include <cstdint>
#include <cstdlib>

// =============================================================================
/// @file macros.hpp
/// @brief SCL Core Compiler Abstractions & Optimization Hints
///
/// This header provides cross-platform macros for compiler-specific features
/// like branch prediction, alignment, inlining, and symbol visibility.
///
/// Naming Convention:
///
/// All macros are prefixed with SCL_ to avoid global namespace pollution.
///
// =============================================================================

// =============================================================================
// SECTION 0: Platform Detection
// =============================================================================

/// @defgroup PlatformDetection Platform Detection Macros
/// @{

/// @brief Platform detection for OS-specific code paths.
///
/// These macros enable compile-time selection of platform-specific implementations
/// for file I/O, memory mapping, and system calls.
///
/// Detection Strategy:
/// - Windows: Defined when _WIN32 or _WIN64 is present
/// - macOS: Defined when __APPLE__ or __MACH__ is present
/// - Linux: Defined when __linux__ or __linux is present
/// - Unix: Defined for any POSIX-compliant system (Linux/macOS/BSD)

#if defined(_WIN32) || defined(_WIN64)
    /// @brief Windows platform identifier (Win32/Win64)
    #define SCL_PLATFORM_WINDOWS 1
    #define SCL_PLATFORM_POSIX 0
    #define SCL_PLATFORM_UNIX 0
#elif defined(__APPLE__) || defined(__MACH__)
    /// @brief macOS platform identifier
    #define SCL_PLATFORM_WINDOWS 0
    #define SCL_PLATFORM_POSIX 1
    #define SCL_PLATFORM_UNIX 1
    #define SCL_PLATFORM_MACOS 1
#elif defined(__linux__) || defined(__linux)
    /// @brief Linux platform identifier
    #define SCL_PLATFORM_WINDOWS 0
    #define SCL_PLATFORM_POSIX 1
    #define SCL_PLATFORM_UNIX 1
    #define SCL_PLATFORM_LINUX 1
#elif defined(__unix__) || defined(__unix)
    /// @brief Generic Unix platform identifier
    #define SCL_PLATFORM_WINDOWS 0
    #define SCL_PLATFORM_POSIX 1
    #define SCL_PLATFORM_UNIX 1
#else
    /// @brief Unknown platform (fallback)
    #define SCL_PLATFORM_WINDOWS 0
    #define SCL_PLATFORM_POSIX 0
    #define SCL_PLATFORM_UNIX 0
    #define SCL_PLATFORM_UNKNOWN 1
#endif

/// @}

// =============================================================================
// SECTION 1: Branch Prediction Hints
// =============================================================================

/// @defgroup BranchPrediction Branch Prediction Hints
/// @{

#if defined(__clang__) || defined(__GNUC__)
    /// @brief Hint that the condition is likely to be true.
    /// Use to optimize hot paths where the branch is taken >90% of the time.
    #define SCL_LIKELY(x)   (__builtin_expect(!!(x), 1))

    /// @brief Hint that the condition is unlikely to be true.
    /// Use for error checks or rare edge cases.
    #define SCL_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
    #define SCL_LIKELY(x)   (x)
    #define SCL_UNLIKELY(x) (x)
#endif

/// @}

// =============================================================================
// SECTION 2: Compiler Warnings & Attributes
// =============================================================================

/// @defgroup Attributes Compiler Attributes
/// @{

#if defined(__has_cpp_attribute)
    #if __has_cpp_attribute(nodiscard) >= 201603L
        /// @brief Mark function return value must be used (C++17).
        #define SCL_NODISCARD [[nodiscard]]
    #else
        #define SCL_NODISCARD
    #endif
#else
    #define SCL_NODISCARD
#endif

/// @}

// =============================================================================
// SECTION 3: Function Inlining & Visibility
// =============================================================================

/// @defgroup Inlining Inlining Control
/// @{

#if defined(_MSC_VER)
    /// @brief Force the compiler to inline a function (ignores cost model).
    #define SCL_FORCE_INLINE __forceinline
    
    /// @brief Hint that a pointer is not aliased (optimizes memory access).
    #define SCL_RESTRICT __restrict
    
    /// @brief Export symbol for DLL/Shared Library.
    #define SCL_EXPORT __declspec(dllexport)
#else
    /// @brief Force the compiler to inline a function.
    #define SCL_FORCE_INLINE inline __attribute__((always_inline))
    
    /// @brief Hint that a pointer is not aliased.
    #define SCL_RESTRICT __restrict__
    
    /// @brief Export symbol for Shared Library.
    #define SCL_EXPORT __attribute__((visibility("default")))
#endif

/// @}

// =============================================================================
// SECTION 4: Memory Alignment & Prefetching
// =============================================================================

/// @defgroup MemoryTools Memory Optimization Tools
/// @{

/// @brief Default alignment for SIMD operations (64 bytes for AVX-512 compatibility)
#define SCL_ALIGNMENT 64

// --- Alignment Specification ---
#if defined(_MSC_VER)
    #define SCL_ALIGN_AS(N) __declspec(align(N))
#else
    #define SCL_ALIGN_AS(N) __attribute__((aligned(N)))
#endif

// --- Pointer Alignment Hint ---
// Tells the compiler "I promise this pointer is aligned to N bytes"
#if defined(__clang__) || defined(__GNUC__)
    #define SCL_ASSUME_ALIGNED(ptr, N) reinterpret_cast<decltype(ptr)>(__builtin_assume_aligned((ptr), (N)))
#else
    #define SCL_ASSUME_ALIGNED(ptr, N) (ptr)
#endif

// --- Data Prefetching ---
// Locality: 0 (none), 1 (low), 2 (moderate), 3 (high - keep in L1)
#if defined(__clang__) || defined(__GNUC__)
    /// @brief Prefetch memory into cache for reading.
    #define SCL_PREFETCH_READ(ptr, locality) __builtin_prefetch((ptr), 0, (locality))
    
    /// @brief Prefetch memory into cache for writing.
    #define SCL_PREFETCH_WRITE(ptr, locality) __builtin_prefetch((ptr), 1, (locality))
#elif defined(_MSC_VER)
    #include <xmmintrin.h>
    // MSVC uses _mm_prefetch with hint constants
    // locality 0-3 maps to: _MM_HINT_NTA, _MM_HINT_T2, _MM_HINT_T1, _MM_HINT_T0
    #define SCL_PREFETCH_READ(ptr, locality) \
        _mm_prefetch(reinterpret_cast<const char*>(ptr), \
                     (locality) == 0 ? _MM_HINT_NTA : \
                     (locality) == 1 ? _MM_HINT_T2  : \
                     (locality) == 2 ? _MM_HINT_T1  : _MM_HINT_T0)
    #define SCL_PREFETCH_WRITE(ptr, locality) SCL_PREFETCH_READ(ptr, locality)
#else
    #define SCL_PREFETCH_READ(ptr, locality) ((void)0)
    #define SCL_PREFETCH_WRITE(ptr, locality) ((void)0)
#endif

/// @}

// =============================================================================
// SECTION 5: Aligned Memory Management (Low Level)
// =============================================================================

/// @defgroup MemoryAlloc Aligned Allocation Macros
/// @{
/// @warning Use these only when raw memory management is strictly required.
/// Consider using wrappers or containers in `scl::core` for safety.

#if defined(_MSC_VER)
    #include <malloc.h>
    
    /// @brief Allocate aligned memory (Raw).
    /// @param ptr  Pointer variable to store the result.
    /// @param size Size in bytes.
    #define SCL_MALLOC_ALIGNED(ptr, size) \
        do { (ptr) = static_cast<void*>(_aligned_malloc((size), SCL_ALIGNMENT)); } while(0)
    
    /// @brief Free aligned memory (Raw).
    #define SCL_FREE_ALIGNED(ptr) \
        do { _aligned_free((ptr)); } while(0)

#else
    /// @brief Allocate aligned memory (Raw).
    /// On POSIX, posix_memalign returns 0 on success.
    #define SCL_MALLOC_ALIGNED(ptr, size) \
        do { \
            void* _tmp_ptr = nullptr; \
            if (posix_memalign(&_tmp_ptr, SCL_ALIGNMENT, (size)) == 0) { \
                (ptr) = _tmp_ptr; \
            } else { \
                (ptr) = nullptr; \
            } \
        } while(0)

    /// @brief Free aligned memory (Raw).
    #define SCL_FREE_ALIGNED(ptr) \
        do { free((ptr)); } while(0)

#endif

/// @}

// =============================================================================
// SECTION 6: Memory-Mapped File Abstractions
// =============================================================================

/// @defgroup MemoryMapping Memory-Mapped File Operations
/// @{

/// @brief Platform-agnostic file handle types.
#if SCL_PLATFORM_WINDOWS
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    
    using SCL_FileHandle = HANDLE;
    using SCL_MapHandle = HANDLE;
    constexpr SCL_FileHandle SCL_INVALID_FILE_HANDLE = INVALID_HANDLE_VALUE;
    constexpr SCL_MapHandle SCL_INVALID_MAP_HANDLE = NULL;
    
#elif SCL_PLATFORM_POSIX
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
    
    using SCL_FileHandle = int;
    using SCL_MapHandle = void*;
    constexpr SCL_FileHandle SCL_INVALID_FILE_HANDLE = -1;
    constexpr SCL_MapHandle SCL_INVALID_MAP_HANDLE = nullptr;
#endif

/// @brief Open file for memory mapping (read-only).
#if SCL_PLATFORM_WINDOWS
    #define SCL_MMAP_OPEN_FILE(path, handle, size) \
        do { \
            (handle) = ::CreateFileA( \
                (path), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, \
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL \
            ); \
            if ((handle) != INVALID_HANDLE_VALUE) { \
                LARGE_INTEGER _fs; \
                if (::GetFileSizeEx((handle), &_fs)) { \
                    (size) = static_cast<size_t>(_fs.QuadPart); \
                } else { \
                    ::CloseHandle((handle)); \
                    (handle) = INVALID_HANDLE_VALUE; \
                    (size) = 0; \
                } \
            } else { \
                (size) = 0; \
            } \
        } while(0)
#elif SCL_PLATFORM_POSIX
    #define SCL_MMAP_OPEN_FILE(path, handle, size) \
        do { \
            (handle) = ::open((path), O_RDONLY); \
            if ((handle) != -1) { \
                struct stat _sb; \
                if (::fstat((handle), &_sb) == 0) { \
                    (size) = static_cast<size_t>(_sb.st_size); \
                } else { \
                    ::close((handle)); \
                    (handle) = -1; \
                    (size) = 0; \
                } \
            } else { \
                (size) = 0; \
            } \
        } while(0)
#endif

/// @brief Create memory mapping from file handle (read-only).
#if SCL_PLATFORM_WINDOWS
    #define SCL_MMAP_CREATE(file_handle, size, map_handle, ptr) \
        do { \
            (map_handle) = ::CreateFileMappingA( \
                (file_handle), NULL, PAGE_READONLY, 0, 0, NULL \
            ); \
            if ((map_handle) != NULL) { \
                (ptr) = ::MapViewOfFile((map_handle), FILE_MAP_READ, 0, 0, 0); \
                if ((ptr) == NULL) { \
                    ::CloseHandle((map_handle)); \
                    (map_handle) = NULL; \
                } \
            } else { \
                (ptr) = nullptr; \
            } \
        } while(0)
#elif SCL_PLATFORM_POSIX
    #define SCL_MMAP_CREATE(file_handle, size, map_handle, ptr) \
        do { \
            (void)(map_handle); \
            (ptr) = ::mmap(nullptr, (size), PROT_READ, MAP_SHARED, (file_handle), 0); \
            if ((ptr) == MAP_FAILED) { \
                (ptr) = nullptr; \
            } \
        } while(0)
#endif

/// @brief Create writable memory mapping from file handle.
#if SCL_PLATFORM_WINDOWS
    #define SCL_MMAP_CREATE_WRITABLE(file_handle, size, map_handle, ptr) \
        do { \
            (map_handle) = ::CreateFileMappingA( \
                (file_handle), NULL, PAGE_READWRITE, 0, 0, NULL \
            ); \
            if ((map_handle) != NULL) { \
                (ptr) = ::MapViewOfFile((map_handle), FILE_MAP_ALL_ACCESS, 0, 0, 0); \
                if ((ptr) == NULL) { \
                    ::CloseHandle((map_handle)); \
                    (map_handle) = NULL; \
                } \
            } else { \
                (ptr) = nullptr; \
            } \
        } while(0)
#elif SCL_PLATFORM_POSIX
    #define SCL_MMAP_CREATE_WRITABLE(file_handle, size, map_handle, ptr) \
        do { \
            (void)(map_handle); \
            (ptr) = ::mmap(nullptr, (size), PROT_READ | PROT_WRITE, MAP_SHARED, (file_handle), 0); \
            if ((ptr) == MAP_FAILED) { \
                (ptr) = nullptr; \
            } \
        } while(0)
#endif

/// @brief Sync memory-mapped changes to disk.
#if SCL_PLATFORM_WINDOWS
    #define SCL_MMAP_SYNC(ptr, size) \
        do { if ((ptr)) ::FlushViewOfFile((ptr), (size)); } while(0)
#elif SCL_PLATFORM_POSIX
    #define SCL_MMAP_SYNC(ptr, size) \
        do { if ((ptr)) ::msync((ptr), (size), MS_SYNC); } while(0)
#endif

/// @brief Async sync (non-blocking).
#if SCL_PLATFORM_WINDOWS
    #define SCL_MMAP_SYNC_ASYNC(ptr, size) \
        do { if ((ptr)) ::FlushViewOfFile((ptr), (size)); } while(0)
#elif SCL_PLATFORM_POSIX
    #define SCL_MMAP_SYNC_ASYNC(ptr, size) \
        do { if ((ptr)) ::msync((ptr), (size), MS_ASYNC); } while(0)
#endif

/// @brief Unmap memory and close handles.
#if SCL_PLATFORM_WINDOWS
    #define SCL_MMAP_CLOSE(ptr, size, map_handle, file_handle) \
        do { \
            if ((ptr)) ::UnmapViewOfFile((ptr)); \
            if ((map_handle)) ::CloseHandle((map_handle)); \
            if ((file_handle) != INVALID_HANDLE_VALUE) ::CloseHandle((file_handle)); \
        } while(0)
#elif SCL_PLATFORM_POSIX
    #define SCL_MMAP_CLOSE(ptr, size, map_handle, file_handle) \
        do { \
            (void)(map_handle); \
            if ((ptr) && (ptr) != MAP_FAILED) ::munmap((ptr), (size)); \
            if ((file_handle) != -1) ::close((file_handle)); \
        } while(0)
#endif

/// @brief Memory access pattern hints.
#if SCL_PLATFORM_POSIX
    #define SCL_MMAP_ADVISE_SEQUENTIAL(ptr, size) \
        do { if ((ptr)) ::madvise((ptr), (size), MADV_SEQUENTIAL); } while(0)
    
    #define SCL_MMAP_ADVISE_RANDOM(ptr, size) \
        do { if ((ptr)) ::madvise((ptr), (size), MADV_RANDOM); } while(0)
    
    #define SCL_MMAP_ADVISE_WILLNEED(ptr, size) \
        do { if ((ptr)) ::madvise((ptr), (size), MADV_WILLNEED); } while(0)
    
    #define SCL_MMAP_ADVISE_DONTNEED(ptr, size) \
        do { if ((ptr)) ::madvise((ptr), (size), MADV_DONTNEED); } while(0)
    
    #define SCL_MMAP_ADVISE_HUGEPAGE(ptr, size) \
        do { \
            if ((ptr) && (size) > 100 * 1024 * 1024) { \
                ::madvise((ptr), (size), MADV_HUGEPAGE); \
            } \
        } while(0)
#else
    #define SCL_MMAP_ADVISE_SEQUENTIAL(ptr, size) ((void)0)
    #define SCL_MMAP_ADVISE_RANDOM(ptr, size) ((void)0)
    #define SCL_MMAP_ADVISE_WILLNEED(ptr, size) ((void)0)
    #define SCL_MMAP_ADVISE_DONTNEED(ptr, size) ((void)0)
    #define SCL_MMAP_ADVISE_HUGEPAGE(ptr, size) ((void)0)
#endif

/// @}
