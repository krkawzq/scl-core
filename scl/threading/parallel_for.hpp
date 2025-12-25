#pragma once

#include "scl/config.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/scheduler.hpp" // Imports detail::get_global_pool and headers

#include <cstddef>
#include <utility>
#include <type_traits>

// =============================================================================
// Backend Specific Headers (Execution Logic)
// =============================================================================
// Note: Scheduler.hpp handles global state headers (like BS_thread_pool.hpp or omp.h).
// We only explicitly include headers required for the *loop constructs* here.

#if defined(SCL_USE_TBB)
    // TBB parallel_for template definitions are needed here
    #include <tbb/parallel_for.h>
    #include <tbb/blocked_range.h>
#endif

namespace scl::threading {

// =============================================================================
// Public Parallel Interface
// =============================================================================

/// @brief Unified Parallel Loop Interface (Backend Agnostic)
///
/// Executes a loop in parallel using the backend selected at compile time via `scl/config.hpp`.
/// The thread count and pool lifecycle are managed by `scl::threading::Scheduler`.
///
/// Backends:
/// - **OpenMP**: Uses `#pragma omp parallel for`.
/// - **TBB**: Uses `tbb::parallel_for` (Work-stealing).
/// - **BS::thread_pool**: Uses `detach_loop` on the global singleton pool.
/// - **Serial**: Fallback to simple `for` loop.
///
/// Usage:
/// ```cpp
/// scl::threading::parallel_for(0, size, [&](size_t i) {
///     data[i] = do_work(i);
/// });
/// ```
///
/// @tparam Func Function object type, signature: `void(size_t index)`
/// @param start Start index (inclusive)
/// @param end   End index (exclusive)
/// @param func  The kernel function to execute
/// 
/// [Owner: Human] - Infrastructure code.
template <typename Func>
SCL_FORCE_INLINE void parallel_for(size_t start, size_t end, Func&& func) {
    // Fast path: Branch prediction hint optimized for non-empty ranges
    if (SCL_UNLIKELY(start >= end)) {
        return;
    }

#if defined(SCL_USE_SERIAL)
    // --- Serial Backend (Debug / Single Thread) ---
    // [Optimization] No overhead, direct execution for debugging
    for (size_t i = start; i < end; ++i) {
        func(i);
    }

#elif defined(SCL_USE_OPENMP)
    // --- OpenMP Backend (HPC Standard) ---
    // Using signed integer for OpenMP 2.0 compatibility and safer optimization
    #pragma omp parallel for schedule(static)
    for (long long i = static_cast<long long>(start); i < static_cast<long long>(end); ++i) {
        func(static_cast<size_t>(i));
    }

#elif defined(SCL_USE_TBB)
    // --- Intel TBB Backend (Work Stealing) ---
    // TBB handles load balancing automatically via work-stealing
    tbb::parallel_for(tbb::blocked_range<size_t>(start, end), 
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                func(i);
            }
        });

#elif defined(SCL_USE_BS)
    // --- BS::thread_pool Backend (Portable) ---
    // Retrieve the singleton pool managed by Scheduler
    // detail::get_global_pool is visible here because it's in the same namespace (scl::threading)
    auto& pool = detail::get_global_pool();
    
    // detach_loop automatically chunks the range and distributes to threads
    pool.detach_loop(start, end, std::forward<Func>(func));
    pool.wait(); // Barrier synchronization to mimic parallel_for semantics

#else
    // --- Fallback (Safety Net) ---
    // Should be caught by config.hpp checks, but just in case:
    #warning "No threading backend defined in parallel_for.hpp, falling back to serial."
    for (size_t i = start; i < end; ++i) {
        func(i);
    }
#endif
}

} // namespace scl::threading
