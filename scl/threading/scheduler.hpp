#pragma once

#include "scl/config.hpp"
#include "scl/core/macros.hpp"
#include <thread>
#include <algorithm>
#include <cstddef>

// =============================================================================
// Backend Headers Inclusion
// =============================================================================

#if defined(SCL_USE_BS)
    #include "BS_thread_pool.hpp"
#elif defined(SCL_USE_OPENMP)
    #include <omp.h>
#elif defined(SCL_USE_TBB)
    #include <tbb/global_control.h>
#endif

namespace scl::threading {

// =============================================================================
// Internal State (Hidden)
// =============================================================================

namespace detail {
#if defined(SCL_USE_BS)
    /// @brief Global singleton for BS::thread_pool.
    ///
    /// The Scheduler manages the lifecycle of this pool.
    /// parallel_for uses this pool to submit tasks.
    SCL_FORCE_INLINE BS::thread_pool& get_global_pool() {
        static BS::thread_pool pool;
        return pool;
    }
#endif

#if defined(SCL_USE_TBB)
    // TBB global control usually requires an object instance to stay alive.
    // Managing this in a header-only static class is tricky. 
    // We defer TBB thread management to TBB's own environment variables 
    // or external initialization to avoid static destruction order issues.
#endif
}

// =============================================================================
// Public Scheduler API
// =============================================================================

/// @brief Global Thread Scheduler and Configuration.
///
/// Controls the concurrency level of the SCL library.
/// This class is entirely static and acts as a facade over the underlying
/// threading backend (OpenMP, TBB, or BS::thread_pool).
class Scheduler {
public:
    /// @brief Get the number of hardware threads available.
    /// @return Number of concurrent threads supported by implementation.
    SCL_FORCE_INLINE static size_t hardware_concurrency() {
        return std::thread::hardware_concurrency();
    }

    /// @brief Set the number of threads for parallel execution.
    ///
    /// @param n Number of threads. 
    ///          If n=0, resets to hardware default.
    ///          If n=1, effectively forces serial execution.
    static void set_num_threads(size_t n) {
        if (n == 0) {
            n = hardware_concurrency();
        }

#if defined(SCL_USE_SERIAL)
        // No-op in serial mode, but strictly we can't spawn threads.
        (void)n; 

#elif defined(SCL_USE_OPENMP)
        omp_set_num_threads(static_cast<int>(n));

#elif defined(SCL_USE_BS)
        // Resetting the pool recreates the threads.
        // This is an expensive operation, do not call inside hot loops.
        detail::get_global_pool().reset(n);

#elif defined(SCL_USE_TBB)
        // TBB control is complex in a library context. 
        // We generally recommend users control TBB via `TBB_NUM_THREADS` env var
        // or tbb::global_control in their main().
        // SCL intentionally skips TBB programmatic control to avoid ABI/Global state issues.
        (void)n;
#endif
    }

    /// @brief Get the currently configured number of threads.
    ///
    /// @return Current thread count limit.
    static size_t get_num_threads() {
#if defined(SCL_USE_SERIAL)
        return 1;

#elif defined(SCL_USE_OPENMP)
        return static_cast<size_t>(omp_get_max_threads());

#elif defined(SCL_USE_BS)
        return detail::get_global_pool().get_thread_count();

#elif defined(SCL_USE_TBB)
        // TBB doesn't expose a simple "current limit" API easily 
        // without creating a scheduler observer. 
        // Return hardware concurrency as a best guess for "default".
        return tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);

#else
        return 1;
#endif
    }

    /// @brief Initialize the library's concurrency settings.
    /// 
    /// Can be called at startup. If not called, defaults to all available cores.
    /// @param n Number of threads (0 = auto).
    static void init(size_t n = 0) {
        set_num_threads(n);
    }
};

} // namespace scl::threading
