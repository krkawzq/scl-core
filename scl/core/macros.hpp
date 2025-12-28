#pragma once

#include "scl/config.hpp"
#include <cstdint>
#include <cstdlib>

// =============================================================================
// FILE: scl/core/macros.hpp
// BRIEF: Cross-platform compiler abstractions and optimization hints
// =============================================================================

// =============================================================================
// SECTION 0: Platform Detection
// =============================================================================

#if defined(_WIN32) || defined(_WIN64)
    #define SCL_PLATFORM_WINDOWS 1
    #define SCL_PLATFORM_POSIX 0
    #define SCL_PLATFORM_UNIX 0
#elif defined(__APPLE__) || defined(__MACH__)
    #define SCL_PLATFORM_WINDOWS 0
    #define SCL_PLATFORM_POSIX 1
    #define SCL_PLATFORM_UNIX 1
    #define SCL_PLATFORM_MACOS 1
#elif defined(__linux__) || defined(__linux)
    #define SCL_PLATFORM_WINDOWS 0
    #define SCL_PLATFORM_POSIX 1
    #define SCL_PLATFORM_UNIX 1
    #define SCL_PLATFORM_LINUX 1
#elif defined(__unix__) || defined(__unix)
    #define SCL_PLATFORM_WINDOWS 0
    #define SCL_PLATFORM_POSIX 1
    #define SCL_PLATFORM_UNIX 1
#else
    #define SCL_PLATFORM_WINDOWS 0
    #define SCL_PLATFORM_POSIX 0
    #define SCL_PLATFORM_UNIX 0
    #define SCL_PLATFORM_UNKNOWN 1
#endif

// =============================================================================
// SECTION 1: Branch Prediction Hints
// =============================================================================

#if defined(__clang__) || defined(__GNUC__)
    #define SCL_LIKELY(x)   (__builtin_expect(!!(x), 1))
    #define SCL_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
    #define SCL_LIKELY(x)   (x)
    #define SCL_UNLIKELY(x) (x)
#endif

// =============================================================================
// SECTION 2: Compiler Attributes
// =============================================================================

#if defined(__has_cpp_attribute)
    #if __has_cpp_attribute(nodiscard) >= 201603L
        #define SCL_NODISCARD [[nodiscard]]
    #else
        #define SCL_NODISCARD
    #endif
#else
    #define SCL_NODISCARD
#endif

// =============================================================================
// SECTION 3: Function Inlining & Visibility
// =============================================================================

#if defined(_MSC_VER)
    #define SCL_FORCE_INLINE __forceinline
    #define SCL_RESTRICT __restrict
    #define SCL_EXPORT __declspec(dllexport)
#else
    #define SCL_FORCE_INLINE inline __attribute__((always_inline))
    #define SCL_RESTRICT __restrict__
    #define SCL_EXPORT __attribute__((visibility("default")))
#endif

// =============================================================================
// SECTION 4: Memory Alignment & Prefetching
// =============================================================================

#define SCL_ALIGNMENT 64  // 64-byte alignment for AVX-512

#if defined(_MSC_VER)
    #define SCL_ALIGN_AS(N) __declspec(align(N))
#else
    #define SCL_ALIGN_AS(N) __attribute__((aligned(N)))
#endif

#if defined(__clang__) || defined(__GNUC__)
    #define SCL_ASSUME_ALIGNED(ptr, N) \
        reinterpret_cast<decltype(ptr)>(__builtin_assume_aligned((ptr), (N)))
#else
    #define SCL_ASSUME_ALIGNED(ptr, N) (ptr)
#endif

#if defined(__clang__) || defined(__GNUC__)
    #define SCL_PREFETCH_READ(ptr, locality) __builtin_prefetch((ptr), 0, (locality))
    #define SCL_PREFETCH_WRITE(ptr, locality) __builtin_prefetch((ptr), 1, (locality))
#elif defined(_MSC_VER)
    #include <xmmintrin.h>
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

// =============================================================================
// SECTION 5: Aligned Memory Allocation
// =============================================================================

#if defined(_MSC_VER)
    #include <malloc.h>
    
    #define SCL_MALLOC_ALIGNED(ptr, size) \
        do { (ptr) = static_cast<void*>(_aligned_malloc((size), SCL_ALIGNMENT)); } while(0)
    
    #define SCL_FREE_ALIGNED(ptr) \
        do { _aligned_free((ptr)); } while(0)

#else
    #define SCL_MALLOC_ALIGNED(ptr, size) \
        do { \
            void* _tmp_ptr = nullptr; \
            if (SCL_LIKELY(posix_memalign(&_tmp_ptr, SCL_ALIGNMENT, (size)) == 0)) { \
                (ptr) = _tmp_ptr; \
            } else { \
                (ptr) = nullptr; \
            } \
        } while(0)

    #define SCL_FREE_ALIGNED(ptr) \
        do { free((ptr)); } while(0)

#endif

// =============================================================================
// SECTION 6: Memory-Mapped Files
// =============================================================================

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

#if SCL_PLATFORM_WINDOWS
    #define SCL_MMAP_OPEN_FILE(path, handle, size) \
        do { \
            (handle) = ::CreateFileA( \
                (path), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, \
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL \
            ); \
            if (SCL_LIKELY((handle) != INVALID_HANDLE_VALUE)) { \
                LARGE_INTEGER _fs; \
                if (SCL_LIKELY(::GetFileSizeEx((handle), &_fs))) { \
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
            if (SCL_LIKELY((handle) != -1)) { \
                struct stat _sb; \
                if (SCL_LIKELY(::fstat((handle), &_sb) == 0)) { \
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

#if SCL_PLATFORM_WINDOWS
    #define SCL_MMAP_CREATE(file_handle, size, map_handle, ptr) \
        do { \
            (map_handle) = ::CreateFileMappingA( \
                (file_handle), NULL, PAGE_READONLY, 0, 0, NULL \
            ); \
            if (SCL_LIKELY((map_handle) != NULL)) { \
                (ptr) = ::MapViewOfFile((map_handle), FILE_MAP_READ, 0, 0, 0); \
                if (SCL_UNLIKELY((ptr) == NULL)) { \
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
            if (SCL_UNLIKELY((ptr) == MAP_FAILED)) { \
                (ptr) = nullptr; \
            } \
        } while(0)
#endif

#if SCL_PLATFORM_WINDOWS
    #define SCL_MMAP_CREATE_WRITABLE(file_handle, size, map_handle, ptr) \
        do { \
            (map_handle) = ::CreateFileMappingA( \
                (file_handle), NULL, PAGE_READWRITE, 0, 0, NULL \
            ); \
            if (SCL_LIKELY((map_handle) != NULL)) { \
                (ptr) = ::MapViewOfFile((map_handle), FILE_MAP_ALL_ACCESS, 0, 0, 0); \
                if (SCL_UNLIKELY((ptr) == NULL)) { \
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
            if (SCL_UNLIKELY((ptr) == MAP_FAILED)) { \
                (ptr) = nullptr; \
            } \
        } while(0)
#endif

#if SCL_PLATFORM_WINDOWS
    #define SCL_MMAP_SYNC(ptr, size) \
        do { if ((ptr)) ::FlushViewOfFile((ptr), (size)); } while(0)
#elif SCL_PLATFORM_POSIX
    #define SCL_MMAP_SYNC(ptr, size) \
        do { if ((ptr)) ::msync((ptr), (size), MS_SYNC); } while(0)
#endif

#if SCL_PLATFORM_WINDOWS
    #define SCL_MMAP_SYNC_ASYNC(ptr, size) \
        do { if ((ptr)) ::FlushViewOfFile((ptr), (size)); } while(0)
#elif SCL_PLATFORM_POSIX
    #define SCL_MMAP_SYNC_ASYNC(ptr, size) \
        do { if ((ptr)) ::msync((ptr), (size), MS_ASYNC); } while(0)
#endif

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
            if (SCL_LIKELY((ptr) && (size) > 100 * 1024 * 1024)) { \
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

// =============================================================================
// SECTION 7: Stack Array and Register Optimization
// =============================================================================

// Stack-allocated array with alignment (VLA-like but portable)
// Use for small, fixed-size buffers to avoid heap allocation
#if defined(__clang__) || defined(__GNUC__)
    #define SCL_STACK_ARRAY(T, name, size) \
        T name[(size)] __attribute__((aligned(SCL_ALIGNMENT)))
#elif defined(_MSC_VER)
    #define SCL_STACK_ARRAY(T, name, size) \
        __declspec(align(64)) T name[(size)]
#else
    #define SCL_STACK_ARRAY(T, name, size) T name[(size)]
#endif

// Register hint: suggest compiler to keep variable in register
#if defined(__clang__) || defined(__GNUC__)
    #define SCL_REGISTER register
#else
    #define SCL_REGISTER
#endif

// Hot path: mark function as frequently called, optimize for speed
#if defined(__clang__) || defined(__GNUC__)
    #define SCL_HOT __attribute__((hot))
#else
    #define SCL_HOT
#endif

// Cold path: mark function as rarely called, optimize for size
#if defined(__clang__) || defined(__GNUC__)
    #define SCL_COLD __attribute__((cold))
#else
    #define SCL_COLD
#endif

// Loop unrolling hints
#if defined(__clang__)
    #define SCL_UNROLL(n) _Pragma("clang loop unroll_count(" #n ")")
    #define SCL_UNROLL_FULL _Pragma("clang loop unroll(full)")
    #define SCL_VECTORIZE _Pragma("clang loop vectorize(enable)")
    #define SCL_NO_VECTORIZE _Pragma("clang loop vectorize(disable)")
#elif defined(__GNUC__) && __GNUC__ >= 8
    #define SCL_UNROLL(n) _Pragma(SCL_STRINGIFY_VALUE(GCC unroll n))
    #define SCL_UNROLL_FULL _Pragma("GCC unroll 16")
    #define SCL_VECTORIZE
    #define SCL_NO_VECTORIZE
#elif defined(_MSC_VER)
    #define SCL_UNROLL(n)
    #define SCL_UNROLL_FULL
    #define SCL_VECTORIZE
    #define SCL_NO_VECTORIZE
#else
    #define SCL_UNROLL(n)
    #define SCL_UNROLL_FULL
    #define SCL_VECTORIZE
    #define SCL_NO_VECTORIZE
#endif

// Flatten: inline all calls within this function
#if defined(__clang__) || defined(__GNUC__)
    #define SCL_FLATTEN __attribute__((flatten))
#else
    #define SCL_FLATTEN
#endif

// Pure function: no side effects, result depends only on arguments
#if defined(__clang__) || defined(__GNUC__)
    #define SCL_PURE __attribute__((pure))
    #define SCL_CONST __attribute__((const))
#else
    #define SCL_PURE
    #define SCL_CONST
#endif

// No-throw guarantee for optimizer
#if defined(__clang__) || defined(__GNUC__)
    #define SCL_NOTHROW __attribute__((nothrow))
#elif defined(_MSC_VER)
    #define SCL_NOTHROW __declspec(nothrow)
#else
    #define SCL_NOTHROW
#endif

// Assume: tell compiler to assume condition is true (undefined behavior if false)
#if defined(__clang__)
    #define SCL_ASSUME(cond) __builtin_assume(cond)
#elif defined(__GNUC__) && __GNUC__ >= 13
    #define SCL_ASSUME(cond) __attribute__((assume(cond)))
#elif defined(_MSC_VER)
    #define SCL_ASSUME(cond) __assume(cond)
#else
    #define SCL_ASSUME(cond) ((void)0)
#endif

// Unreachable: mark code path as unreachable for optimizer
#if defined(__clang__) || defined(__GNUC__)
    #define SCL_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
    #define SCL_UNREACHABLE() __assume(0)
#else
    #define SCL_UNREACHABLE() ((void)0)
#endif

// =============================================================================
// SECTION 8: SIMD Lane Count Helpers
// =============================================================================

// Maximum SIMD vector width in bytes (AVX-512 = 64, AVX2 = 32, SSE = 16)
#if defined(__AVX512F__)
    #define SCL_SIMD_WIDTH 64
    #define SCL_SIMD_LANES_F64 8
    #define SCL_SIMD_LANES_F32 16
#elif defined(__AVX2__) || defined(__AVX__)
    #define SCL_SIMD_WIDTH 32
    #define SCL_SIMD_LANES_F64 4
    #define SCL_SIMD_LANES_F32 8
#elif defined(__SSE2__)
    #define SCL_SIMD_WIDTH 16
    #define SCL_SIMD_LANES_F64 2
    #define SCL_SIMD_LANES_F32 4
#else
    #define SCL_SIMD_WIDTH 8
    #define SCL_SIMD_LANES_F64 1
    #define SCL_SIMD_LANES_F32 2
#endif

// =============================================================================
// SECTION 9: Cache Line and Memory Layout
// =============================================================================

#define SCL_CACHE_LINE_SIZE 64

// Pad structure to cache line boundary to avoid false sharing
#if defined(__clang__) || defined(__GNUC__)
    #define SCL_CACHE_ALIGNED __attribute__((aligned(SCL_CACHE_LINE_SIZE)))
#elif defined(_MSC_VER)
    #define SCL_CACHE_ALIGNED __declspec(align(64))
#else
    #define SCL_CACHE_ALIGNED
#endif

// Padding bytes to avoid false sharing between struct members
#define SCL_PADDING(n) char _padding_##n[SCL_CACHE_LINE_SIZE]

// =============================================================================
// SECTION 10: Compile-Time Utilities
// =============================================================================

// Static assertion with message
#define SCL_STATIC_ASSERT(cond, msg) static_assert(cond, msg)

// Compile-time array size
#define SCL_ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

// Stringify macro argument
#define SCL_STRINGIFY(x) #x
#define SCL_STRINGIFY_VALUE(x) SCL_STRINGIFY(x)
