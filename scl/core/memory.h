// =============================================================================
// FILE: scl/core/memory.h
// BRIEF: API reference for SCL Low-Level Memory Primitives
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdlib>
#include "scl/core/type.hpp"

namespace scl::memory {

// =============================================================================
// MODULE OVERVIEW
// =============================================================================

/* -----------------------------------------------------------------------------
 * MODULE: Memory Primitives
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Low-level memory operations optimized for high-performance computing.
 *
 * PURPOSE:
 *     Provides SIMD-accelerated memory operations, aligned allocation,
 *     and cache-aware utilities for numerical kernels.
 *
 * MODULE SECTIONS:
 *     1. Aligned Memory Allocation
 *     2. Initialization (Fill/Zero)
 *     3. Data Movement (Copy operations)
 *     4. Prefetch Utilities
 *     5. Memory Comparison
 *     6. Swap Operations
 *     7. Reverse Operations
 *
 * SAFETY LEVELS:
 *     Safe      - Handles overlaps (memmove), checks bounds in Debug
 *     Fast      - Assumes NO overlap (memcpy), UB if violated
 *     Stream    - Bypasses cache (non-temporal), assumes NO overlap
 *
 * ALIGNMENT REQUIREMENTS:
 *     For optimal SIMD performance, allocate buffers with 64-byte alignment
 *     (matches AVX-512 cache line size).
 *
 * THREAD SAFETY:
 *     All operations are safe for concurrent use on non-overlapping memory.
 *     Operations on the same memory region from multiple threads are unsafe.
 * -------------------------------------------------------------------------- */

// =============================================================================
// ALIGNED MEMORY ALLOCATION
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: aligned_alloc
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Allocate aligned memory for primitive types.
 *
 * SIGNATURE:
 *     template <typename T>
 *     T* aligned_alloc(size_t count, size_t alignment = 64)
 *
 * PARAMETERS:
 *     T         [template] - Element type (must be trivially constructible)
 *     count     [in]       - Number of elements to allocate
 *     alignment [in]       - Alignment in bytes (default: 64)
 *
 * PRECONDITIONS:
 *     - T must be trivially constructible
 *     - alignment must be a power of 2
 *     - alignment >= sizeof(void*)
 *     - count * sizeof(T) must not overflow
 *
 * POSTCONDITIONS:
 *     If count == 0:
 *         - Returns nullptr
 *     If allocation succeeds:
 *         - Returns aligned pointer to zero-initialized memory
 *         - Pointer is aligned to specified boundary
 *         - All elements are zero-initialized
 *     If allocation fails:
 *         - Returns nullptr
 *
 * RETURN VALUE:
 *     Aligned pointer to allocated memory, or nullptr on failure
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(count * sizeof(T))
 *
 * THREAD SAFETY:
 *     Safe - each call allocates independent memory
 *
 * PLATFORM SUPPORT:
 *     - C++17+: Uses aligned operator new
 *     - POSIX: Uses posix_memalign
 *     - Windows: Uses _aligned_malloc
 *
 * IMPORTANT:
 *     Memory MUST be freed with aligned_free, NOT free() or delete[]
 *
 * USE CASES:
 *     - SIMD workspaces (require 16/32/64-byte alignment)
 *     - Cache-line optimization (64-byte alignment)
 *     - Large temporary buffers in kernels
 * -------------------------------------------------------------------------- */
template <typename T>
T* aligned_alloc(
    size_t count,              // Number of elements
    size_t alignment = 64      // Alignment in bytes
);

/* -----------------------------------------------------------------------------
 * FUNCTION: aligned_free
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Free memory allocated with aligned_alloc.
 *
 * SIGNATURE:
 *     template <typename T>
 *     void aligned_free(T* ptr, size_t alignment = 64)
 *
 * PARAMETERS:
 *     T         [template] - Element type
 *     ptr       [in]       - Pointer to free (nullptr is safe)
 *     alignment [in]       - Alignment value used in allocation (default: 64)
 *
 * PRECONDITIONS:
 *     - ptr must be nullptr or allocated with aligned_alloc
 *     - alignment must match the alignment used in aligned_alloc
 *
 * POSTCONDITIONS:
 *     - Memory is deallocated
 *     - ptr is invalid (dangling pointer)
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - if ptr is not accessed concurrently
 *
 * WARNING:
 *     Using regular free() or delete[] on aligned memory may crash
 *     Alignment value must match the value used in aligned_alloc
 * -------------------------------------------------------------------------- */
template <typename T>
void aligned_free(
    T* ptr,                    // Pointer to free
    size_t alignment = 64      // Alignment value from allocation
);

/* -----------------------------------------------------------------------------
 * CLASS: AlignedBuffer
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     RAII wrapper for aligned memory.
 *
 * PURPOSE:
 *     Provides automatic cleanup of aligned memory allocations, ensuring
 *     exception safety and preventing memory leaks.
 *
 * MEMBERS:
 *     ptr_       [private] - Raw aligned pointer
 *     count_     [private] - Number of elements
 *     alignment_ [private] - Alignment value used in allocation
 *
 * CONSTRUCTORS:
 *     AlignedBuffer(size_t count, size_t alignment = 64)
 *         Allocates aligned buffer with specified element count and alignment.
 *
 * DESTRUCTOR:
 *     ~AlignedBuffer()
 *         Automatically frees allocated memory using correct alignment value.
 *
 * METHODS:
 *     T* get() noexcept
 *         Returns raw pointer to buffer.
 *
 *     size_t size() const noexcept
 *         Returns number of elements in buffer.
 *
 *     T& operator[](size_t i) noexcept
 *         Access element at index i (no bounds checking).
 *
 *     Array<T> span() noexcept
 *         Returns Array view of buffer.
 *
 *     explicit operator bool() const noexcept
 *         Checks if buffer allocation succeeded.
 *
 * MOVABLE:
 *     Yes - supports move construction and assignment
 *
 * COPYABLE:
 *     No - copy operations are deleted
 *
 * THREAD SAFETY:
 *     Safe - if not accessed concurrently
 *
 * USE CASE:
 *     Automatic cleanup of temporary buffers in exception-safe code.
 * -------------------------------------------------------------------------- */
template <typename T>
class AlignedBuffer {
public:
    AlignedBuffer(size_t count, size_t alignment = 64);
    ~AlignedBuffer();

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept;
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;

    T* get() noexcept;
    const T* get() const noexcept;

    size_t size() const noexcept;

    T& operator[](size_t i) noexcept;
    const T& operator[](size_t i) const noexcept;

    Array<T> span() noexcept;
    Array<const T> span() const noexcept;

    explicit operator bool() const noexcept;

private:
    T* ptr_;
    size_t count_;
};

// =============================================================================
// INITIALIZATION
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: fill
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Fill memory with a value using SIMD acceleration.
 *
 * SIGNATURE:
 *     template <typename T>
 *     void fill(Array<T> span, T value)
 *
 * PARAMETERS:
 *     T     [template] - Element type
 *     span  [in,out]   - Memory range to fill
 *     value [in]       - Value to fill with
 *
 * PRECONDITIONS:
 *     - span.ptr must be valid or nullptr (if span.len == 0)
 *
 * POSTCONDITIONS:
 *     - All elements in span are set to value
 *
 * MUTABILITY:
 *     INPLACE - modifies span contents
 *
 * ALGORITHM:
 *     1. Create SIMD vector with broadcasted value
 *     2. 4-way unrolled SIMD loop for bulk of data
 *     3. Remainder SIMD loop
 *     4. Scalar loop for tail elements
 *
 * COMPLEXITY:
 *     Time:  O(n / lanes) where lanes is SIMD width
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - if span is not accessed concurrently
 *
 * PERFORMANCE:
 *     - Automatically uses best SIMD instructions (AVX2/AVX-512/NEON)
 *     - 4-way unrolling for better instruction throughput
 *     - Optimal for arrays larger than ~64 elements
 * -------------------------------------------------------------------------- */
template <typename T>
void fill(
    Array<T> span,             // Memory range to fill
    T value                    // Fill value
);

/* -----------------------------------------------------------------------------
 * FUNCTION: zero
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Zero out memory efficiently.
 *
 * SIGNATURE:
 *     template <typename T>
 *     void zero(Array<T> span)
 *
 * PARAMETERS:
 *     T    [template] - Element type
 *     span [in,out]   - Memory range to zero
 *
 * PRECONDITIONS:
 *     - span.ptr must be valid or nullptr (if span.len == 0)
 *
 * POSTCONDITIONS:
 *     - All bytes in span are set to zero
 *
 * MUTABILITY:
 *     INPLACE - modifies span contents
 *
 * ALGORITHM:
 *     For trivial types: uses memset
 *     For non-trivial types: uses fill(span, T(0))
 *
 * COMPLEXITY:
 *     Time:  O(n * sizeof(T))
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - if span is not accessed concurrently
 *
 * PERFORMANCE:
 *     - Trivial types use optimized memset
 *     - Non-trivial types use SIMD fill
 * -------------------------------------------------------------------------- */
template <typename T>
void zero(
    Array<T> span              // Memory range to zero
);

// =============================================================================
// DATA MOVEMENT
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: copy_fast
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Fast copy assuming NO overlap (uses memcpy semantics).
 *
 * SIGNATURE:
 *     template <typename T>
 *     void copy_fast(Array<const T> src, Array<T> dst)
 *
 * PARAMETERS:
 *     T   [template] - Element type
 *     src [in]       - Source memory range
 *     dst [out]      - Destination memory range
 *
 * PRECONDITIONS:
 *     - src.len == dst.len
 *     - src and dst must NOT overlap
 *
 * POSTCONDITIONS:
 *     - dst contains copy of src elements
 *
 * MUTABILITY:
 *     CONST on src, INPLACE on dst
 *
 * COMPLEXITY:
 *     Time:  O(n * sizeof(T))
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - if src and dst are not accessed concurrently
 *
 * WARNING:
 *     Undefined Behavior if src and dst overlap
 *     Use copy() for potentially overlapping ranges
 *
 * PERFORMANCE:
 *     - Uses memcpy for trivially copyable types
 *     - Compiler can optimize with __restrict__ semantics
 *     - Fastest copy option when overlap is impossible
 *
 * WHEN TO USE:
 *     - Copying between distinct buffers
 *     - Absolutely certain no overlap exists
 *     - Maximum performance is critical
 * -------------------------------------------------------------------------- */
template <typename T>
void copy_fast(
    Array<const T> src,        // Source range
    Array<T> dst               // Destination range
);

/* -----------------------------------------------------------------------------
 * FUNCTION: copy
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Safe copy handling overlap correctly (uses memmove semantics).
 *
 * SIGNATURE:
 *     template <typename T>
 *     void copy(Array<const T> src, Array<T> dst)
 *
 * PARAMETERS:
 *     T   [template] - Element type
 *     src [in]       - Source memory range
 *     dst [out]      - Destination memory range
 *
 * PRECONDITIONS:
 *     - src.len == dst.len
 *
 * POSTCONDITIONS:
 *     - dst contains copy of src elements
 *     - Correct even if src and dst overlap
 *     - No-op if src.ptr == dst.ptr
 *
 * MUTABILITY:
 *     CONST on src, INPLACE on dst
 *
 * ALGORITHM:
 *     1. Early return if src.ptr == dst.ptr (no-op)
 *     2. For trivial types: uses memmove
 *     3. For non-trivial types:
 *        If dst < src: copy forward
 *        If dst > src: copy backward
 *
 * COMPLEXITY:
 *     Time:  O(n * sizeof(T))
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - if src and dst are not accessed concurrently
 *
 * PERFORMANCE:
 *     - Slightly slower than copy_fast due to overlap handling
 *     - Still highly optimized by standard library
 *     - Handles same-pointer case efficiently (early return)
 *
 * WHEN TO USE:
 *     - Potentially overlapping ranges (e.g., sliding windows)
 *     - Safety is more important than maximum speed
 *     - Default choice when overlap status is unknown
 * -------------------------------------------------------------------------- */
template <typename T>
void copy(
    Array<const T> src,        // Source range
    Array<T> dst               // Destination range
);

/* -----------------------------------------------------------------------------
 * FUNCTION: stream_copy
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Non-temporal copy bypassing CPU cache.
 *
 * SIGNATURE:
 *     template <typename T>
 *     void stream_copy(Array<const T> src, Array<T> dst)
 *
 * PARAMETERS:
 *     T   [template] - Element type
 *     src [in]       - Source memory range
 *     dst [out]      - Destination memory range
 *
 * PRECONDITIONS:
 *     - src.len == dst.len
 *     - src and dst must NOT overlap
 *     - src.ptr and dst.ptr should be 64-byte aligned for optimal performance
 *
 * POSTCONDITIONS:
 *     - dst contains copy of src elements
 *     - Data bypasses CPU cache (non-temporal stores) if aligned
 *     - Memory fence ensures stores are visible
 *
 * MUTABILITY:
 *     CONST on src, INPLACE on dst
 *
 * ALGORITHM:
 *     1. Check if pointers are 64-byte aligned
 *     2. If not aligned: fallback to copy_fast
 *     3. If aligned: 2-way unrolled SIMD loop with non-temporal stores
 *     4. Remainder SIMD loop with non-temporal stores
 *     5. Scalar tail with regular stores
 *     6. Memory fence to ensure visibility
 *
 * COMPLEXITY:
 *     Time:  O(n / lanes) where lanes is SIMD width
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - if src and dst are not accessed concurrently
 *
 * WARNING:
 *     Assumes NO overlap - UB if violated
 *     Falls back to regular copy if memory is not 64-byte aligned
 *
 * PERFORMANCE NOTES:
 *     - Writes directly to RAM, bypassing L1/L2 caches (when aligned)
 *     - Best for large buffers (> 1MB) not read immediately
 *     - Avoid for small buffers (cache bypass overhead too high)
 *     - Can reduce cache pollution in streaming scenarios
 *     - Requires 64-byte alignment for non-temporal stores
 *
 * WHEN TO USE:
 *     - Large data transfers to memory not accessed soon
 *     - Write-only buffers (e.g., output to disk)
 *     - Preventing cache pollution in streaming workloads
 *     - Buffers allocated with 64-byte alignment
 *
 * WHEN NOT TO USE:
 *     - Small buffers (< 256KB)
 *     - Data that will be read back immediately
 *     - Overlapping memory regions
 *     - Unaligned memory (will fallback to regular copy)
 * -------------------------------------------------------------------------- */
template <typename T>
void stream_copy(
    Array<const T> src,        // Source range
    Array<T> dst               // Destination range
);

// =============================================================================
// PREFETCH UTILITIES
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: prefetch_range_read
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Prefetch contiguous memory range for reading.
 *
 * SIGNATURE:
 *     template <int Locality = 3, typename T>
 *     void prefetch_range_read(const T* ptr, size_t bytes, size_t max_prefetches = 16)
 *
 * PARAMETERS:
 *     Locality      [template] - Cache locality hint (0=none, 3=keep in L1)
 *     T             [template] - Element type
 *     ptr           [in]       - Start of memory range
 *     bytes         [in]       - Number of bytes to prefetch
 *     max_prefetches [in]      - Maximum number of prefetch instructions (default: 16)
 *
 * PRECONDITIONS:
 *     - Locality must be in range [0, 3]
 *     - ptr must be valid or nullptr (if bytes == 0)
 *
 * POSTCONDITIONS:
 *     - Memory range is loaded into CPU cache (up to max_prefetches cache lines)
 *
 * ALGORITHM:
 *     Issues prefetch instructions at 64-byte (cache line) granularity
 *     Limited to max_prefetches to prevent instruction cache overflow
 *
 * COMPLEXITY:
 *     Time:  O(min(bytes / 64, max_prefetches))
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe
 *
 * LOCALITY LEVELS:
 *     0 - No temporal locality (evict soon)
 *     1 - Low temporal locality
 *     2 - Moderate temporal locality
 *     3 - High temporal locality (keep in L1)
 *
 * PERFORMANCE NOTES:
 *     - Limited prefetches to prevent instruction cache pollution
 *     - Default limit of 16 prefetches is typically sufficient
 *     - Adjust max_prefetches based on workload characteristics
 *
 * WHEN TO USE:
 *     - Before sequential read loop
 *     - Random access patterns known in advance
 *     - Latency hiding in long-running operations
 * -------------------------------------------------------------------------- */
template <int Locality = 3, typename T>
void prefetch_range_read(
    const T* ptr,              // Start of range
    size_t bytes,              // Number of bytes
    size_t max_prefetches = 16 // Maximum prefetch instructions
);

/* -----------------------------------------------------------------------------
 * FUNCTION: prefetch_range_write
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Prefetch contiguous memory range for writing.
 *
 * SIGNATURE:
 *     template <int Locality = 3, typename T>
 *     void prefetch_range_write(T* ptr, size_t bytes)
 *
 * PARAMETERS:
 *     Locality      [template] - Cache locality hint (0=none, 3=keep in L1)
 *     T             [template] - Element type
 *     ptr           [in]       - Start of memory range
 *     bytes         [in]       - Number of bytes to prefetch
 *     max_prefetches [in]      - Maximum number of prefetch instructions (default: 16)
 *
 * PRECONDITIONS:
 *     - Locality must be in range [0, 3]
 *     - ptr must be valid or nullptr (if bytes == 0)
 *
 * POSTCONDITIONS:
 *     - Memory range is loaded into CPU cache in exclusive state (up to max_prefetches cache lines)
 *
 * ALGORITHM:
 *     Issues write prefetch instructions at cache-line granularity
 *     Limited to max_prefetches to prevent instruction cache overflow
 *
 * COMPLEXITY:
 *     Time:  O(min(bytes / 64, max_prefetches))
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe
 *
 * PERFORMANCE NOTES:
 *     - Limited prefetches to prevent instruction cache pollution
 *     - Default limit of 16 prefetches is typically sufficient
 *
 * WHEN TO USE:
 *     - Before sequential write loop
 *     - Initializing large buffers
 *     - Write-intensive operations
 * -------------------------------------------------------------------------- */
template <int Locality = 3, typename T>
void prefetch_range_write(
    T* ptr,                    // Start of range
    size_t bytes,              // Number of bytes
    size_t max_prefetches = 16 // Maximum prefetch instructions
);

/* -----------------------------------------------------------------------------
 * FUNCTION: prefetch_ahead
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Software prefetch helper for loop iteration.
 *
 * SIGNATURE:
 *     template <typename T, size_t PREFETCH_DISTANCE = 8>
 *     void prefetch_ahead(const T* data, size_t current_idx, size_t total_size)
 *
 * PARAMETERS:
 *     T                 [template] - Element type
 *     PREFETCH_DISTANCE [template] - Elements to prefetch ahead (default: 8)
 *     data              [in]       - Array being iterated
 *     current_idx       [in]       - Current loop index
 *     total_size        [in]       - Total array size
 *
 * PRECONDITIONS:
 *     - data must be valid or nullptr (if total_size == 0)
 *     - current_idx < total_size
 *
 * POSTCONDITIONS:
 *     - data[current_idx + PREFETCH_DISTANCE] is prefetched (if in bounds)
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 *
 * USAGE PATTERN:
 *     for (size_t i = 0; i < N; ++i) {
 *         prefetch_ahead(data, i, N);
 *         // ... process data[i] ...
 *     }
 *
 * TUNING:
 *     - Increase PREFETCH_DISTANCE for high-latency memory access
 *     - Decrease for tight loops with low latency
 *     - Typical range: 4-16 elements ahead
 * -------------------------------------------------------------------------- */
template <typename T, size_t PREFETCH_DISTANCE = 8>
void prefetch_ahead(
    const T* data,             // Array being iterated
    size_t current_idx,        // Current index
    size_t total_size          // Total size
);

// =============================================================================
// MEMORY COMPARISON
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: equal
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Check if two memory ranges are equal.
 *
 * SIGNATURE:
 *     template <typename T>
 *     bool equal(Array<const T> a, Array<const T> b)
 *
 * PARAMETERS:
 *     T [template] - Element type
 *     a [in]       - First memory range
 *     b [in]       - Second memory range
 *
 * PRECONDITIONS:
 *     - a.ptr and b.ptr must be valid or nullptr (if respective len == 0)
 *
 * POSTCONDITIONS:
 *     Returns true if:
 *         - a.len == b.len AND
 *         - All corresponding elements are equal
 *     Returns false otherwise
 *
 * RETURN VALUE:
 *     true if ranges are equal, false otherwise
 *
 * COMPLEXITY:
 *     Time:  O(min(n, first_diff)) - early exit on mismatch
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - if a and b are not modified concurrently
 *
 * PERFORMANCE:
 *     - Uses memcmp for trivially copyable types
 *     - Early exit on first mismatch
 *     - Optimized for equality (same pointer check)
 *
 * NUMERICAL NOTES:
 *     - For floating-point: uses exact bitwise comparison
 *     - NaN != NaN (IEEE 754 semantics)
 *     - Consider approximate comparison for numerical tolerance
 * -------------------------------------------------------------------------- */
template <typename T>
bool equal(
    Array<const T> a,          // First range
    Array<const T> b           // Second range
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compare
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Lexicographic comparison of two memory ranges.
 *
 * SIGNATURE:
 *     template <typename T>
 *     int compare(Array<const T> a, Array<const T> b)
 *
 * PARAMETERS:
 *     T [template] - Element type
 *     a [in]       - First memory range
 *     b [in]       - Second memory range
 *
 * PRECONDITIONS:
 *     - a.ptr and b.ptr must be valid or nullptr (if respective len == 0)
 *     - T must support < and > operators
 *
 * POSTCONDITIONS:
 *     Returns:
 *         -1 if a < b lexicographically
 *          0 if a == b
 *          1 if a > b lexicographically
 *
 * RETURN VALUE:
 *     -1, 0, or 1 indicating comparison result
 *
 * ALGORITHM:
 *     For trivial arithmetic types: uses memcmp for speed
 *     For other types: compares elements pairwise until mismatch
 *     If all common elements equal, shorter range is less
 *
 * COMPLEXITY:
 *     Time:  O(min(a.len, b.len)) - early exit on first difference
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - if a and b are not modified concurrently
 *
 * PERFORMANCE:
 *     - Uses memcmp for trivial arithmetic types (faster)
 *     - Scalar comparison for non-trivial types
 *     - Early exit on first difference
 *
 * SEMANTICS:
 *     - Dictionary ordering (like strcmp for strings)
 *     - Shorter range is less if all common elements equal
 *     - Equivalent to std::lexicographical_compare
 * -------------------------------------------------------------------------- */
template <typename T>
int compare(
    Array<const T> a,          // First range
    Array<const T> b           // Second range
);

// =============================================================================
// SWAP OPERATIONS
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: swap
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Swap two elements in place.
 *
 * SIGNATURE:
 *     template <typename T>
 *     void swap(T& a, T& b) noexcept
 *
 * PARAMETERS:
 *     T [template] - Element type
 *     a [in,out]   - First element
 *     b [in,out]   - Second element
 *
 * PRECONDITIONS:
 *     None
 *
 * POSTCONDITIONS:
 *     - a contains original value of b
 *     - b contains original value of a
 *
 * MUTABILITY:
 *     INPLACE - modifies both parameters
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(sizeof(T)) for temporary
 *
 * THREAD SAFETY:
 *     Unsafe - concurrent access to a or b causes race condition
 *
 * PERFORMANCE:
 *     - Uses move semantics
 *     - Compiles to register swaps for small types
 * -------------------------------------------------------------------------- */
template <typename T>
void swap(
    T& a,                      // First element
    T& b                       // Second element
) noexcept;

/* -----------------------------------------------------------------------------
 * FUNCTION: swap_ranges
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     SIMD-optimized swap of two memory ranges.
 *
 * SIGNATURE:
 *     template <typename T>
 *     void swap_ranges(Array<T> a, Array<T> b)
 *
 * PARAMETERS:
 *     T [template] - Element type
 *     a [in,out]   - First memory range
 *     b [in,out]   - Second memory range
 *
 * PRECONDITIONS:
 *     - a.len == b.len
 *     - a and b must NOT overlap (unless a.ptr == b.ptr)
 *
 * POSTCONDITIONS:
 *     - a contains original values from b
 *     - b contains original values from a
 *     - If a.ptr == b.ptr, no changes are made
 *
 * MUTABILITY:
 *     INPLACE - modifies both ranges (no-op if same memory)
 *
 * ALGORITHM:
 *     1. Early return if a.ptr == b.ptr (no-op)
 *     2. SIMD loop: load from both, store swapped
 *     3. Scalar tail for remaining elements
 *
 * COMPLEXITY:
 *     Time:  O(n / lanes) where lanes is SIMD width
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - concurrent access causes race condition
 *
 * WARNING:
 *     Undefined Behavior if ranges overlap (except when a.ptr == b.ptr)
 *
 * PERFORMANCE:
 *     - Uses SIMD for bulk swapping
 *     - More efficient than element-wise swap for large ranges
 *     - Handles same-memory case efficiently (early return)
 * -------------------------------------------------------------------------- */
template <typename T>
void swap_ranges(
    Array<T> a,                // First range
    Array<T> b                 // Second range
);

// =============================================================================
// REVERSE OPERATIONS
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: reverse
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Reverse elements in place.
 *
 * SIGNATURE:
 *     template <typename T>
 *     void reverse(Array<T> span)
 *
 * PARAMETERS:
 *     T    [template] - Element type
 *     span [in,out]   - Memory range to reverse
 *
 * PRECONDITIONS:
 *     - span.ptr must be valid or nullptr (if span.len == 0)
 *
 * POSTCONDITIONS:
 *     - Elements in span are reversed
 *     - span[i] contains original span[n-1-i]
 *
 * MUTABILITY:
 *     INPLACE - modifies span
 *
 * ALGORITHM:
 *     Two-pointer approach swapping from both ends toward center
 *
 * COMPLEXITY:
 *     Time:  O(n / 2)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - concurrent access causes race condition
 *
 * PERFORMANCE:
 *     - Uses move-based swap
 *     - Minimal memory traffic (n/2 swaps)
 * -------------------------------------------------------------------------- */
template <typename T>
void reverse(
    Array<T> span              // Range to reverse
);

/* -----------------------------------------------------------------------------
 * FUNCTION: reverse_copy
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Copy elements in reverse order to destination.
 *
 * SIGNATURE:
 *     template <typename T>
 *     void reverse_copy(Array<const T> src, Array<T> dst)
 *
 * PARAMETERS:
 *     T   [template] - Element type
 *     src [in]       - Source memory range
 *     dst [out]      - Destination memory range
 *
 * PRECONDITIONS:
 *     - src.len == dst.len
 *     - src and dst must NOT overlap
 *
 * POSTCONDITIONS:
 *     - dst[i] contains src[n-1-i]
 *
 * MUTABILITY:
 *     CONST on src, INPLACE on dst
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - if src and dst are not accessed concurrently
 *
 * WARNING:
 *     Undefined Behavior if ranges overlap
 * -------------------------------------------------------------------------- */
template <typename T>
void reverse_copy(
    Array<const T> src,        // Source range
    Array<T> dst               // Destination range (reversed)
);

} // namespace scl::memory
