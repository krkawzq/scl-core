// =============================================================================
// FILE: scl/core/macros.h
// BRIEF: API reference for compiler abstractions and optimization hints
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

// =============================================================================
// SECTION 0: Platform Detection
// =============================================================================

/* -----------------------------------------------------------------------------
 * MACRO: SCL_PLATFORM_WINDOWS
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Indicates Windows platform (Win32 or Win64).
 *
 * VALUES:
 *     1 - Running on Windows
 *     0 - Not running on Windows
 *
 * DETECTION:
 *     Set to 1 when _WIN32 or _WIN64 is defined by the compiler.
 *
 * USAGE:
 *     Use for conditional compilation of Windows-specific code paths.
 * -------------------------------------------------------------------------- */
#define SCL_PLATFORM_WINDOWS

/* -----------------------------------------------------------------------------
 * MACRO: SCL_PLATFORM_POSIX
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Indicates POSIX-compliant platform (Linux, macOS, BSD, etc.).
 *
 * VALUES:
 *     1 - Running on POSIX system
 *     0 - Not running on POSIX system
 *
 * DETECTION:
 *     Set to 1 for any Unix-like system (Linux, macOS, BSD).
 *
 * USAGE:
 *     Use for POSIX API calls (open, mmap, pthread, etc.).
 * -------------------------------------------------------------------------- */
#define SCL_PLATFORM_POSIX

/* -----------------------------------------------------------------------------
 * MACRO: SCL_PLATFORM_UNIX
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Indicates Unix-like operating system.
 *
 * VALUES:
 *     1 - Running on Unix-like system
 *     0 - Not running on Unix-like system
 *
 * DETECTION:
 *     Set to 1 for Linux, macOS, BSD, or generic Unix.
 *
 * USAGE:
 *     Use for Unix-specific features (signals, file descriptors, etc.).
 * -------------------------------------------------------------------------- */
#define SCL_PLATFORM_UNIX

/* -----------------------------------------------------------------------------
 * MACRO: SCL_PLATFORM_LINUX
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Indicates Linux operating system.
 *
 * VALUES:
 *     1 - Running on Linux
 *     0 - Not running on Linux
 *
 * DETECTION:
 *     Set to 1 when __linux__ or __linux is defined.
 *
 * USAGE:
 *     Use for Linux-specific features (epoll, inotify, etc.).
 * -------------------------------------------------------------------------- */
#define SCL_PLATFORM_LINUX

/* -----------------------------------------------------------------------------
 * MACRO: SCL_PLATFORM_MACOS
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Indicates macOS operating system.
 *
 * VALUES:
 *     1 - Running on macOS
 *     0 - Not running on macOS
 *
 * DETECTION:
 *     Set to 1 when __APPLE__ or __MACH__ is defined.
 *
 * USAGE:
 *     Use for macOS-specific features (kqueue, Accelerate framework, etc.).
 * -------------------------------------------------------------------------- */
#define SCL_PLATFORM_MACOS

/* -----------------------------------------------------------------------------
 * MACRO: SCL_PLATFORM_UNKNOWN
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Indicates unknown or unsupported platform.
 *
 * VALUES:
 *     1 - Platform not recognized
 *     0 - Platform recognized
 *
 * DETECTION:
 *     Set to 1 when no known platform macro is detected.
 *
 * USAGE:
 *     Check this to provide fallback behavior or error messages.
 * -------------------------------------------------------------------------- */
#define SCL_PLATFORM_UNKNOWN

// =============================================================================
// SECTION 1: Branch Prediction Hints
// =============================================================================

/* -----------------------------------------------------------------------------
 * MACRO: SCL_LIKELY(x)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Hint to compiler that condition x is likely to be true.
 *
 * PARAMETERS:
 *     x [in] - Condition expression to evaluate
 *
 * SEMANTICS:
 *     Informs the compiler to optimize for the case where x evaluates to true.
 *     The compiler may reorder code to minimize branch mispredictions.
 *
 * USAGE GUIDELINES:
 *     - Use when the condition is true in >90% of cases
 *     - Ideal for hot paths and common cases
 *     - Do NOT use for conditions with ~50% probability
 *
 * PERFORMANCE:
 *     Reduces branch misprediction penalties on modern CPUs by improving
 *     instruction cache layout and prefetching.
 *
 * COMPILER SUPPORT:
 *     - GCC/Clang: Uses __builtin_expect
 *     - MSVC: No-op (returns x unchanged)
 *     - Other: No-op (returns x unchanged)
 *
 * THREAD SAFETY:
 *     Safe - pure compile-time hint
 * -------------------------------------------------------------------------- */
#define SCL_LIKELY(x)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_UNLIKELY(x)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Hint to compiler that condition x is unlikely to be true.
 *
 * PARAMETERS:
 *     x [in] - Condition expression to evaluate
 *
 * SEMANTICS:
 *     Informs the compiler to optimize for the case where x evaluates to false.
 *     The compiler may reorder code to place the unlikely path out-of-line.
 *
 * USAGE GUIDELINES:
 *     - Use for error conditions and edge cases
 *     - Ideal for validation checks and assertion-like conditions
 *     - Typically used with early returns or exception handling
 *
 * PERFORMANCE:
 *     Optimizes hot path by moving unlikely code out of instruction cache,
 *     improving cache hit rate for common paths.
 *
 * COMPILER SUPPORT:
 *     - GCC/Clang: Uses __builtin_expect
 *     - MSVC: No-op (returns x unchanged)
 *     - Other: No-op (returns x unchanged)
 *
 * THREAD SAFETY:
 *     Safe - pure compile-time hint
 * -------------------------------------------------------------------------- */
#define SCL_UNLIKELY(x)

// =============================================================================
// SECTION 2: Compiler Attributes
// =============================================================================

/* -----------------------------------------------------------------------------
 * MACRO: SCL_NODISCARD
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Mark function return value must be used (C++17 attribute).
 *
 * SEMANTICS:
 *     Compiler will emit a warning if the return value is discarded.
 *     Prevents silent bugs from ignoring error codes or important results.
 *
 * USAGE GUIDELINES:
 *     - Use for functions returning error codes
 *     - Use for factory functions returning allocated resources
 *     - Use for pure functions where ignoring result is meaningless
 *
 * COMPILER SUPPORT:
 *     - C++17 and later: [[nodiscard]]
 *     - Pre-C++17: No-op
 *
 * THREAD SAFETY:
 *     Safe - compile-time attribute only
 * -------------------------------------------------------------------------- */
#define SCL_NODISCARD

// =============================================================================
// SECTION 3: Function Inlining & Visibility
// =============================================================================

/* -----------------------------------------------------------------------------
 * MACRO: SCL_FORCE_INLINE
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Force compiler to inline function, ignoring cost heuristics.
 *
 * SEMANTICS:
 *     Instructs compiler to inline the function even if it would normally
 *     decide against it. Overrides compiler cost model.
 *
 * USAGE GUIDELINES:
 *     - Use for tiny hot-path functions (1-3 lines)
 *     - Use for wrapper functions with zero overhead requirement
 *     - Do NOT use for large functions (increases code size)
 *     - Do NOT use for recursive functions
 *
 * PERFORMANCE:
 *     Eliminates function call overhead but increases code size.
 *     Best for functions called in tight loops.
 *
 * COMPILER SUPPORT:
 *     - MSVC: __forceinline
 *     - GCC/Clang: inline __attribute__((always_inline))
 *
 * THREAD SAFETY:
 *     Safe - compile-time attribute only
 * -------------------------------------------------------------------------- */
#define SCL_FORCE_INLINE

/* -----------------------------------------------------------------------------
 * MACRO: SCL_RESTRICT
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Hint that pointer does not alias with any other pointer.
 *
 * SEMANTICS:
 *     Promises to compiler that the pointer is the only way to access the
 *     pointed-to memory. Enables aggressive optimization of memory accesses.
 *
 * PRECONDITIONS:
 *     - Pointer must not alias with any other pointer in the function
 *     - Violating this promise results in undefined behavior
 *
 * USAGE GUIDELINES:
 *     - Use for function parameters known to be non-aliased
 *     - Common in numerical kernels with separate input/output buffers
 *     - Do NOT use if unsure about aliasing
 *
 * PERFORMANCE:
 *     Enables vectorization, reordering, and other memory optimizations.
 *     Can provide 2-3x speedup in tight computational loops.
 *
 * COMPILER SUPPORT:
 *     - MSVC: __restrict
 *     - GCC/Clang: __restrict__
 *
 * THREAD SAFETY:
 *     Safe - compile-time hint only
 *
 * WARNING:
 *     Incorrect use leads to undefined behavior. Only use when aliasing
 *     is impossible by design.
 * -------------------------------------------------------------------------- */
#define SCL_RESTRICT

/* -----------------------------------------------------------------------------
 * MACRO: SCL_EXPORT
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Mark symbol for export in shared library (DLL/SO).
 *
 * SEMANTICS:
 *     Makes symbol visible outside the shared library for dynamic linking.
 *     Required for C-ABI interface functions.
 *
 * USAGE GUIDELINES:
 *     - Use for all public C-ABI functions
 *     - Use for exported classes (when necessary)
 *     - Do NOT use for internal implementation details
 *
 * COMPILER SUPPORT:
 *     - MSVC: __declspec(dllexport)
 *     - GCC/Clang: __attribute__((visibility("default")))
 *
 * THREAD SAFETY:
 *     Safe - compile-time attribute only
 * -------------------------------------------------------------------------- */
#define SCL_EXPORT

// =============================================================================
// SECTION 4: Memory Alignment & Prefetching
// =============================================================================

/* -----------------------------------------------------------------------------
 * MACRO: SCL_ALIGNMENT
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Default alignment for SIMD operations (64 bytes).
 *
 * RATIONALE:
 *     64-byte alignment ensures compatibility with:
 *     - AVX-512 (64-byte vectors)
 *     - Cache line alignment (typically 64 bytes)
 *     - Huge page alignment hints
 *
 * USAGE:
 *     Use as alignment parameter for aligned allocations and data structures.
 * -------------------------------------------------------------------------- */
#define SCL_ALIGNMENT 64

/* -----------------------------------------------------------------------------
 * MACRO: SCL_ALIGN_AS(N)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Specify alignment requirement for type or variable.
 *
 * PARAMETERS:
 *     N [in] - Alignment in bytes (must be power of 2)
 *
 * PRECONDITIONS:
 *     - N must be a power of 2
 *     - N should typically be 16, 32, or 64 for SIMD types
 *
 * USAGE GUIDELINES:
 *     - Apply to struct/class definitions for SIMD data
 *     - Apply to stack variables needing specific alignment
 *
 * COMPILER SUPPORT:
 *     - MSVC: __declspec(align(N))
 *     - GCC/Clang: __attribute__((aligned(N)))
 *
 * THREAD SAFETY:
 *     Safe - compile-time attribute only
 * -------------------------------------------------------------------------- */
#define SCL_ALIGN_AS(N)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_ASSUME_ALIGNED(ptr, N)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Inform compiler that pointer is aligned to N bytes.
 *
 * PARAMETERS:
 *     ptr [in] - Pointer to check/cast
 *     N   [in] - Alignment in bytes (must be power of 2)
 *
 * PRECONDITIONS:
 *     - ptr must actually be aligned to N bytes
 *     - Violating this assumption results in undefined behavior
 *
 * POSTCONDITIONS:
 *     - Returns ptr (possibly with metadata for compiler)
 *
 * USAGE GUIDELINES:
 *     - Use after allocating with SCL_MALLOC_ALIGNED
 *     - Use for pointers known to be aligned by construction
 *     - Do NOT use speculatively
 *
 * PERFORMANCE:
 *     Enables vectorization and eliminates runtime alignment checks.
 *
 * COMPILER SUPPORT:
 *     - GCC/Clang: __builtin_assume_aligned
 *     - MSVC: No-op (returns ptr unchanged)
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 *
 * WARNING:
 *     Incorrect alignment assumption leads to undefined behavior.
 * -------------------------------------------------------------------------- */
#define SCL_ASSUME_ALIGNED(ptr, N)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_PREFETCH_READ(ptr, locality)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Prefetch memory into cache for reading.
 *
 * PARAMETERS:
 *     ptr      [in] - Pointer to memory address to prefetch
 *     locality [in] - Cache locality hint (0-3)
 *
 * LOCALITY VALUES:
 *     0 - No temporal locality (NTA - non-temporal access)
 *     1 - Low temporal locality (T2 cache)
 *     2 - Moderate temporal locality (T1 cache)
 *     3 - High temporal locality (T0 cache, keep in L1)
 *
 * SEMANTICS:
 *     Hints to CPU to load memory into cache before it is accessed.
 *     Non-blocking operation that improves latency hiding.
 *
 * USAGE GUIDELINES:
 *     - Prefetch 1-2 cache lines ahead in sequential access
 *     - Use locality 3 for data accessed multiple times
 *     - Use locality 0 for streaming data accessed once
 *
 * PERFORMANCE:
 *     Reduces cache miss latency by overlapping computation and memory access.
 *     Effective for predictable access patterns.
 *
 * COMPILER SUPPORT:
 *     - GCC/Clang: __builtin_prefetch
 *     - MSVC: _mm_prefetch
 *     - Other: No-op
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
#define SCL_PREFETCH_READ(ptr, locality)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_PREFETCH_WRITE(ptr, locality)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Prefetch memory into cache for writing.
 *
 * PARAMETERS:
 *     ptr      [in] - Pointer to memory address to prefetch
 *     locality [in] - Cache locality hint (0-3)
 *
 * LOCALITY VALUES:
 *     Same as SCL_PREFETCH_READ
 *
 * SEMANTICS:
 *     Hints to CPU to load memory into cache with exclusive ownership
 *     for upcoming write operations.
 *
 * USAGE GUIDELINES:
 *     - Use when memory will be written (not just read)
 *     - Prefetch destination buffers before writing
 *     - Use locality 3 for reduction targets
 *
 * PERFORMANCE:
 *     Acquires cache line in exclusive state, avoiding later upgrade latency.
 *
 * COMPILER SUPPORT:
 *     - GCC/Clang: __builtin_prefetch with write intent
 *     - MSVC: Same as read prefetch
 *     - Other: No-op
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
#define SCL_PREFETCH_WRITE(ptr, locality)

// =============================================================================
// SECTION 5: Aligned Memory Allocation
// =============================================================================

/* -----------------------------------------------------------------------------
 * MACRO: SCL_MALLOC_ALIGNED(ptr, size)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Allocate memory aligned to SCL_ALIGNMENT boundary.
 *
 * PARAMETERS:
 *     ptr  [out] - Pointer variable to receive allocated memory
 *     size [in]  - Size in bytes to allocate
 *
 * PRECONDITIONS:
 *     - size > 0
 *
 * POSTCONDITIONS:
 *     - ptr is set to allocated memory (aligned to SCL_ALIGNMENT)
 *     - ptr is set to nullptr on allocation failure
 *
 * MUTABILITY:
 *     ALLOCATES - allocates new memory
 *
 * USAGE GUIDELINES:
 *     - Always check ptr for nullptr after allocation
 *     - Must free with SCL_FREE_ALIGNED (not standard free)
 *     - Prefer RAII wrappers over raw usage
 *
 * COMPLEXITY:
 *     Time:  O(1) typical, O(log n) worst case
 *     Space: O(size) plus alignment overhead
 *
 * THREAD SAFETY:
 *     Safe - uses thread-safe allocators
 *
 * PLATFORM DIFFERENCES:
 *     - Windows: Uses _aligned_malloc
 *     - POSIX: Uses posix_memalign
 *
 * WARNING:
 *     Raw memory management. Prefer using scl::core memory abstractions.
 * -------------------------------------------------------------------------- */
#define SCL_MALLOC_ALIGNED(ptr, size)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_FREE_ALIGNED(ptr)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Free memory allocated with SCL_MALLOC_ALIGNED.
 *
 * PARAMETERS:
 *     ptr [in,out] - Pointer to aligned memory to free
 *
 * PRECONDITIONS:
 *     - ptr must be allocated with SCL_MALLOC_ALIGNED or be nullptr
 *
 * POSTCONDITIONS:
 *     - Memory is freed
 *     - ptr value is undefined (should be set to nullptr by caller)
 *
 * MUTABILITY:
 *     FREES - deallocates memory
 *
 * USAGE GUIDELINES:
 *     - Safe to call with nullptr
 *     - Set ptr to nullptr after freeing to avoid double-free
 *
 * COMPLEXITY:
 *     Time:  O(1) typical
 *     Space: N/A
 *
 * THREAD SAFETY:
 *     Safe - uses thread-safe deallocators
 *
 * PLATFORM DIFFERENCES:
 *     - Windows: Uses _aligned_free
 *     - POSIX: Uses standard free
 * -------------------------------------------------------------------------- */
#define SCL_FREE_ALIGNED(ptr)

// =============================================================================
// SECTION 6: Memory-Mapped Files
// =============================================================================

/* -----------------------------------------------------------------------------
 * TYPE: SCL_FileHandle
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Platform-agnostic file handle type.
 *
 * PLATFORM TYPES:
 *     - Windows: HANDLE
 *     - POSIX: int (file descriptor)
 *
 * INVALID VALUE:
 *     SCL_INVALID_FILE_HANDLE
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * TYPE: SCL_MapHandle
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Platform-agnostic memory mapping handle type.
 *
 * PLATFORM TYPES:
 *     - Windows: HANDLE (file mapping object)
 *     - POSIX: void* (not used, mapping is just the pointer)
 *
 * INVALID VALUE:
 *     SCL_INVALID_MAP_HANDLE
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * MACRO: SCL_MMAP_OPEN_FILE(path, handle, size)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Open file for memory mapping in read-only mode.
 *
 * PARAMETERS:
 *     path   [in]  - File path (const char*)
 *     handle [out] - File handle variable
 *     size   [out] - File size variable (size_t)
 *
 * PRECONDITIONS:
 *     - path must point to valid file path string
 *     - File must exist and be readable
 *
 * POSTCONDITIONS:
 *     - handle is set to valid file handle on success
 *     - handle is set to SCL_INVALID_FILE_HANDLE on failure
 *     - size is set to file size in bytes on success
 *     - size is set to 0 on failure
 *
 * MUTABILITY:
 *     ALLOCATES - opens system resource
 *
 * USAGE GUIDELINES:
 *     - Check handle against SCL_INVALID_FILE_HANDLE
 *     - Close with SCL_MMAP_CLOSE when done
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 *
 * PLATFORM DIFFERENCES:
 *     - Windows: Uses CreateFileA with sequential scan hint
 *     - POSIX: Uses open with O_RDONLY flag
 * -------------------------------------------------------------------------- */
#define SCL_MMAP_OPEN_FILE(path, handle, size)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_MMAP_CREATE(file_handle, size, map_handle, ptr)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Create read-only memory mapping from file handle.
 *
 * PARAMETERS:
 *     file_handle [in]  - Valid file handle from SCL_MMAP_OPEN_FILE
 *     size        [in]  - Size to map (use size from SCL_MMAP_OPEN_FILE)
 *     map_handle  [out] - Mapping handle variable (Windows only)
 *     ptr         [out] - Pointer to mapped memory
 *
 * PRECONDITIONS:
 *     - file_handle must be valid
 *     - size must be > 0
 *
 * POSTCONDITIONS:
 *     - ptr points to mapped memory on success
 *     - ptr is nullptr on failure
 *     - map_handle is set (Windows) or unused (POSIX)
 *
 * MUTABILITY:
 *     ALLOCATES - creates memory mapping
 *
 * USAGE GUIDELINES:
 *     - Check ptr for nullptr
 *     - Memory is read-only
 *     - Unmap with SCL_MMAP_CLOSE when done
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(size) virtual address space
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 *
 * PLATFORM DIFFERENCES:
 *     - Windows: Uses CreateFileMapping + MapViewOfFile
 *     - POSIX: Uses mmap with PROT_READ
 * -------------------------------------------------------------------------- */
#define SCL_MMAP_CREATE(file_handle, size, map_handle, ptr)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_MMAP_CREATE_WRITABLE(file_handle, size, map_handle, ptr)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Create writable memory mapping from file handle.
 *
 * PARAMETERS:
 *     file_handle [in]  - Valid file handle
 *     size        [in]  - Size to map
 *     map_handle  [out] - Mapping handle variable (Windows only)
 *     ptr         [out] - Pointer to mapped memory
 *
 * PRECONDITIONS:
 *     - file_handle must be valid and opened with write access
 *     - size must be > 0
 *
 * POSTCONDITIONS:
 *     - ptr points to mapped memory on success (read-write access)
 *     - ptr is nullptr on failure
 *     - map_handle is set (Windows) or unused (POSIX)
 *
 * MUTABILITY:
 *     ALLOCATES - creates writable memory mapping
 *
 * USAGE GUIDELINES:
 *     - Modifications are written back to file
 *     - Use SCL_MMAP_SYNC to control synchronization
 *     - Unmap with SCL_MMAP_CLOSE when done
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(size) virtual address space
 *
 * THREAD SAFETY:
 *     Safe for mapping creation
 *     Unsafe for concurrent access to mapped memory without synchronization
 *
 * PLATFORM DIFFERENCES:
 *     - Windows: Uses CreateFileMapping (PAGE_READWRITE) + MapViewOfFile
 *     - POSIX: Uses mmap with PROT_READ | PROT_WRITE
 * -------------------------------------------------------------------------- */
#define SCL_MMAP_CREATE_WRITABLE(file_handle, size, map_handle, ptr)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_MMAP_SYNC(ptr, size)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Synchronously flush memory-mapped changes to disk (blocking).
 *
 * PARAMETERS:
 *     ptr  [in] - Pointer to mapped memory
 *     size [in] - Size of region to sync
 *
 * PRECONDITIONS:
 *     - ptr must be valid writable mapping
 *     - size must be > 0
 *
 * POSTCONDITIONS:
 *     - All changes in [ptr, ptr+size) are written to disk
 *     - Function returns after sync completes
 *
 * MUTABILITY:
 *     INPLACE - flushes pending writes
 *
 * USAGE GUIDELINES:
 *     - Call before unmapping to ensure durability
 *     - Blocks until I/O completes
 *     - Use SCL_MMAP_SYNC_ASYNC for non-blocking variant
 *
 * COMPLEXITY:
 *     Time:  O(dirty pages) - depends on I/O subsystem
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 *
 * PLATFORM DIFFERENCES:
 *     - Windows: Uses FlushViewOfFile (synchronous)
 *     - POSIX: Uses msync with MS_SYNC flag
 * -------------------------------------------------------------------------- */
#define SCL_MMAP_SYNC(ptr, size)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_MMAP_SYNC_ASYNC(ptr, size)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Asynchronously flush memory-mapped changes to disk (non-blocking).
 *
 * PARAMETERS:
 *     ptr  [in] - Pointer to mapped memory
 *     size [in] - Size of region to sync
 *
 * PRECONDITIONS:
 *     - ptr must be valid writable mapping
 *     - size must be > 0
 *
 * POSTCONDITIONS:
 *     - Sync operation is scheduled
 *     - Function returns immediately
 *     - Changes will eventually reach disk
 *
 * MUTABILITY:
 *     INPLACE - schedules pending writes
 *
 * USAGE GUIDELINES:
 *     - Use for background checkpointing
 *     - Does not guarantee immediate durability
 *     - Call SCL_MMAP_SYNC before critical operations
 *
 * COMPLEXITY:
 *     Time:  O(1) to schedule, O(dirty pages) background
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 *
 * PLATFORM DIFFERENCES:
 *     - Windows: Uses FlushViewOfFile (may be synchronous)
 *     - POSIX: Uses msync with MS_ASYNC flag
 * -------------------------------------------------------------------------- */
#define SCL_MMAP_SYNC_ASYNC(ptr, size)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_MMAP_CLOSE(ptr, size, map_handle, file_handle)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Unmap memory and close all associated handles.
 *
 * PARAMETERS:
 *     ptr         [in] - Pointer to mapped memory
 *     size        [in] - Size of mapping
 *     map_handle  [in] - Mapping handle (Windows only)
 *     file_handle [in] - File handle
 *
 * PRECONDITIONS:
 *     - Handles must be valid or set to invalid values
 *
 * POSTCONDITIONS:
 *     - Memory is unmapped
 *     - All handles are closed
 *     - Pointers/handles become invalid
 *
 * MUTABILITY:
 *     FREES - releases system resources
 *
 * USAGE GUIDELINES:
 *     - Call even if mapping failed (safe with invalid handles)
 *     - Set pointers to nullptr after closing
 *     - Pending writes may be lost if not synced first
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 *
 * PLATFORM DIFFERENCES:
 *     - Windows: Closes both mapping handle and file handle
 *     - POSIX: Unmaps memory and closes file descriptor
 * -------------------------------------------------------------------------- */
#define SCL_MMAP_CLOSE(ptr, size, map_handle, file_handle)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_MMAP_ADVISE_SEQUENTIAL(ptr, size)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Hint that memory will be accessed sequentially.
 *
 * PARAMETERS:
 *     ptr  [in] - Pointer to mapped memory
 *     size [in] - Size of region
 *
 * SEMANTICS:
 *     Advises kernel to optimize for sequential access (readahead).
 *
 * USAGE GUIDELINES:
 *     - Use for streaming file processing
 *     - Apply before starting sequential read
 *
 * PERFORMANCE:
 *     Increases readahead window, reducing page faults.
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 *
 * PLATFORM DIFFERENCES:
 *     - POSIX: Uses madvise with MADV_SEQUENTIAL
 *     - Windows: No-op
 * -------------------------------------------------------------------------- */
#define SCL_MMAP_ADVISE_SEQUENTIAL(ptr, size)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_MMAP_ADVISE_RANDOM(ptr, size)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Hint that memory will be accessed randomly.
 *
 * PARAMETERS:
 *     ptr  [in] - Pointer to mapped memory
 *     size [in] - Size of region
 *
 * SEMANTICS:
 *     Advises kernel to optimize for random access (disable readahead).
 *
 * USAGE GUIDELINES:
 *     - Use for database-like access patterns
 *     - Use for index structures
 *
 * PERFORMANCE:
 *     Reduces unnecessary readahead, conserving cache.
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 *
 * PLATFORM DIFFERENCES:
 *     - POSIX: Uses madvise with MADV_RANDOM
 *     - Windows: No-op
 * -------------------------------------------------------------------------- */
#define SCL_MMAP_ADVISE_RANDOM(ptr, size)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_MMAP_ADVISE_WILLNEED(ptr, size)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Hint that memory will be needed soon (prefetch into cache).
 *
 * PARAMETERS:
 *     ptr  [in] - Pointer to mapped memory
 *     size [in] - Size of region
 *
 * SEMANTICS:
 *     Initiates readahead to bring pages into memory before access.
 *
 * USAGE GUIDELINES:
 *     - Use before processing a known region
 *     - Overlaps I/O with computation
 *
 * PERFORMANCE:
 *     Reduces page fault latency by prefaulting pages.
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 *
 * PLATFORM DIFFERENCES:
 *     - POSIX: Uses madvise with MADV_WILLNEED
 *     - Windows: No-op
 * -------------------------------------------------------------------------- */
#define SCL_MMAP_ADVISE_WILLNEED(ptr, size)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_MMAP_ADVISE_DONTNEED(ptr, size)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Hint that memory is no longer needed (can be evicted from cache).
 *
 * PARAMETERS:
 *     ptr  [in] - Pointer to mapped memory
 *     size [in] - Size of region
 *
 * SEMANTICS:
 *     Allows kernel to free physical pages backing the mapping.
 *     Pages will be refaulted on next access.
 *
 * USAGE GUIDELINES:
 *     - Use after processing large regions once
 *     - Use to control memory footprint
 *
 * PERFORMANCE:
 *     Frees physical memory for other uses, but causes refault overhead.
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 *
 * PLATFORM DIFFERENCES:
 *     - POSIX: Uses madvise with MADV_DONTNEED
 *     - Windows: No-op
 * -------------------------------------------------------------------------- */
#define SCL_MMAP_ADVISE_DONTNEED(ptr, size)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_MMAP_ADVISE_HUGEPAGE(ptr, size)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Hint to use huge pages for large mappings (>100MB).
 *
 * PARAMETERS:
 *     ptr  [in] - Pointer to mapped memory
 *     size [in] - Size of region
 *
 * SEMANTICS:
 *     Requests kernel to back mapping with huge pages (2MB or 1GB).
 *     Only applied if size > 100MB.
 *
 * USAGE GUIDELINES:
 *     - Use for very large matrix data
 *     - Requires huge pages configured on system
 *
 * PERFORMANCE:
 *     Reduces TLB misses for large memory accesses.
 *     Can provide 5-10% speedup for memory-intensive workloads.
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 *
 * PLATFORM DIFFERENCES:
 *     - POSIX: Uses madvise with MADV_HUGEPAGE (Linux)
 *     - Windows: No-op
 *
 * NUMERICAL NOTES:
 *     Threshold of 100MB is chosen to avoid huge page fragmentation
 *     for smaller allocations.
 * -------------------------------------------------------------------------- */
#define SCL_MMAP_ADVISE_HUGEPAGE(ptr, size)

