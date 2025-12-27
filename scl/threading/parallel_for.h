// =============================================================================
// FILE: scl/threading/parallel_for.h
// BRIEF: API reference for Unified Parallel Loop Interface
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>

namespace scl::threading {

// =============================================================================
// MODULE OVERVIEW
// =============================================================================

/* -----------------------------------------------------------------------------
 * MODULE: Parallel For Loop
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Backend-agnostic parallel loop abstraction.
 *
 * PURPOSE:
 *     Provides unified interface for parallel iteration across multiple
 *     threading backends with consistent semantics:
 *     - OpenMP (HPC standard, widespread support)
 *     - Intel TBB (work-stealing, advanced load balancing)
 *     - BS::thread_pool (portable, header-only)
 *     - Serial (debug/single-threaded fallback)
 *
 * DESIGN GOALS:
 *     1. Minimal abstraction overhead (inline, not force-inline)
 *     2. Consistent behavior across backends
 *     3. Simple, intuitive API
 *     4. Compile-time backend selection
 *
 * BACKEND SELECTION:
 *     Controlled by preprocessor defines from scl/config.hpp:
 *     - SCL_USE_OPENMP: Uses #pragma omp parallel for
 *     - SCL_USE_TBB:    Uses tbb::parallel_for
 *     - SCL_USE_BS:     Uses BS::thread_pool::submit() with futures
 *     - SCL_USE_SERIAL: Uses standard for loop
 *
 * USAGE PATTERN:
 *     Basic parallel loop:
 *         scl::threading::parallel_for(0, n, [&](size_t i) {
 *             data[i] = compute(i);
 *         });
 *
 *     With captured variables:
 *         std::vector<double> input(n), output(n);
 *         scl::threading::parallel_for(0, n, [&](size_t i) {
 *             output[i] = expensive_function(input[i]);
 *         });
 *
 * THREAD SAFETY:
 *     - Loop body must be thread-safe
 *     - Avoid race conditions on shared mutable state
 *     - Use atomic operations or per-thread accumulators
 *
 * PERFORMANCE CHARACTERISTICS:
 *     OpenMP:
 *         - Static scheduling (default)
 *         - Best for uniform workloads
 *         - Minimal overhead
 *
 *     TBB:
 *         - Dynamic work-stealing
 *         - Best for irregular workloads
 *         - Automatic load balancing
 *
 *     BS::thread_pool:
 *         - Task-based parallelism
 *         - Good for moderate parallelism
 *         - Portable, no external dependencies
 *
 *     Serial:
 *         - Zero overhead
 *         - For debugging and single-threaded execution
 *
 * OVERHEAD ANALYSIS:
 *     - Function uses inline (not force-inline) to avoid code bloat
 *     - Empty range check: Single branch (likely predicted)
 *     - BS backend: Allocates futures vector (O(num_threads))
 *     - No dynamic dispatch
 *
 * LIMITATIONS:
 *     - Nested parallelism: OpenMP serializes, other backends support it
 *     - Exception propagation: Serial/TBB/BS propagate, OpenMP terminates
 *     - No cancellation support
 *     - Integer index only (size_t)
 * -------------------------------------------------------------------------- */

// =============================================================================
// PARALLEL FOR
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: parallel_for
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Execute loop iterations in parallel across worker threads.
 *
 * SIGNATURE:
 *     template <typename Func>
 *     void parallel_for(size_t start, size_t end, Func&& func)
 *
 * PARAMETERS:
 *     Func  [template] - Callable with signature void(size_t)
 *     start [in]       - Start index (inclusive)
 *     end   [in]       - End index (exclusive)
 *     func  [in]       - Loop body function
 *
 * PRECONDITIONS:
 *     - Func must be callable as func(size_t)
 *     - Func must be thread-safe (no data races)
 *     - start <= end (undefined if start > end, but handled gracefully)
 *
 * POSTCONDITIONS:
 *     - func(i) called for all i in [start, end)
 *     - All iterations complete before return (barrier synchronization)
 *     - No ordering guarantees between iterations
 *
 * MUTABILITY:
 *     CONST on [start, end) range
 *     func may mutate captured state (user responsibility for thread safety)
 *
 * ALGORITHM:
 *     OpenMP:
 *         1. Create parallel region with N threads
 *         2. Static partition: [start, end) divided into N chunks
 *         3. Each thread executes its chunk sequentially
 *         4. Implicit barrier at end
 *
 *     TBB:
 *         1. Create blocked_range(start, end)
 *         2. Work-stealing scheduler dynamically distributes blocks
 *         3. Each thread processes blocks until none remain
 *         4. Implicit barrier at end
 *
 *     BS::thread_pool:
 *         1. Manually chunk range based on thread count
 *         2. Submit each chunk as task returning future
 *         3. Wait for all futures and propagate exceptions
 *
 *     Serial:
 *         1. Sequential for loop: for (i = start; i < end; ++i)
 *
 * COMPLEXITY:
 *     Time:  O((end - start) * T(func) / N) expected
 *            where N = number of threads, T(func) = func time
 *     Space: O(N) thread stack space
 *            O(N) auxiliary for BS backend (futures vector)
 *            O(1) auxiliary for other backends
 *
 * THREAD SAFETY:
 *     Unsafe - func must ensure thread safety for shared state
 *
 * PERFORMANCE NOTES:
 *     - Effective for (end - start) >> num_threads
 *     - Overhead becomes significant when range < 1000 elements
 *     - Best when func execution time >> scheduling overhead
 *     - No benefit if func is too fast (< ~1 microsecond per call)
 *
 * OPTIMIZATION TIPS:
 *     1. Minimize false sharing:
 *        - Use thread-local accumulators
 *        - Align data to cache lines
 *
 *     2. Minimize synchronization:
 *        - Avoid locks in hot paths
 *        - Use atomic operations sparingly
 *
 *     3. Balance workload:
 *        - TBB handles imbalance automatically
 *        - OpenMP static scheduling assumes uniform work
 *
 *     4. Reduce overhead:
 *        - Don't parallelize tiny loops (< 1000 iterations)
 *        - Ensure func does meaningful work (> 1us)
 *
 * WHEN TO USE:
 *     - Independent iterations (no loop-carried dependencies)
 *     - Computationally expensive loop body
 *     - Large iteration count (> 1000)
 *     - Read-heavy or embarrassingly parallel workloads
 *
 * WHEN NOT TO USE:
 *     - Small iteration counts (< 100)
 *     - Fast loop body (< 1us per iteration)
 *     - Heavy synchronization requirements
 *     - Sequential dependencies between iterations
 *
 * BACKEND-SPECIFIC NOTES:
 *     OpenMP:
 *         - schedule(static): Each thread gets contiguous chunk
 *         - Uses unsigned loop variables (OpenMP 4.5+)
 *         - Nested parallel_for calls execute serially to prevent thread explosion
 *
 *     TBB:
 *         - Blocked range automatically partitioned
 *         - Work-stealing provides automatic load balancing
 *         - Best for irregular workloads
 *
 *     BS::thread_pool:
 *         - Manually chunks range and uses submit() for futures
 *         - Exceptions propagate via future.get()
 *         - Barrier synchronization via future.wait()
 *
 * EXAMPLES:
 *     Element-wise array operation:
 *         parallel_for(0, n, [&](size_t i) {
 *             output[i] = input[i] * 2.0;
 *         });
 *
 *     Reduction with thread-local accumulator:
 *         std::vector<double> local_sums(num_threads, 0.0);
 *         parallel_for(0, n, [&](size_t i) {
 *             size_t tid = get_thread_id();
 *             local_sums[tid] += data[i];
 *         });
 *         double total = std::accumulate(local_sums.begin(), local_sums.end(), 0.0);
 *
 *     Matrix row processing:
 *         parallel_for(0, num_rows, [&](size_t row) {
 *             process_row(matrix, row);
 *         });
 *
 * ERROR HANDLING:
 *     - Empty range (start >= end): Returns immediately, no iterations
 *     - Exceptions in func: Backend-dependent behavior
 *         - Serial: Exception propagates normally
 *         - OpenMP: Implementation-defined (often terminates) - nested calls execute serially
 *         - TBB: Exception propagates to caller
 *         - BS: Exception propagates to caller (uses futures)
 * -------------------------------------------------------------------------- */
template <typename Func>
void parallel_for(
    size_t start,          // Start index (inclusive)
    size_t end,            // End index (exclusive)
    Func&& func            // Loop body: void(size_t index)
);

} // namespace scl::threading
