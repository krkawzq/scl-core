#pragma once

#include <memory>      // Must be first to ensure std namespace is available
#include <thread>
#include <cstddef>

#include "scl/config.hpp"
#include "scl/core/macros.hpp"

// =============================================================================
// FILE: scl/threading/scheduler.hpp
// BRIEF: Unified thread pool scheduler interface across multiple backends
// =============================================================================

// =============================================================================
// Backend Headers
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
// Internal State
// =============================================================================

namespace detail {
#if defined(SCL_USE_BS)
    // Global singleton for BS::thread_pool (never destroyed)
    // Using function-local static with reference return to avoid memory leak
    inline BS::thread_pool& get_global_pool() {
        static BS::thread_pool pool;
        return pool;
    }
#endif

#if defined(SCL_USE_TBB)
    // TBB thread management deferred to environment variables or external init
    // Global control is managed via static variable in set_num_threads()
#endif
}

// =============================================================================
// Public Scheduler API
// =============================================================================

class Scheduler {
public:
    // Get number of hardware threads available
    // Returns std::thread::hardware_concurrency(), or 1 if detection fails
    SCL_FORCE_INLINE static size_t hardware_concurrency() noexcept {
        size_t hw = std::thread::hardware_concurrency();
        // Return at least 1 if hardware_concurrency() returns 0 (detection failure)
        return (hw > 0) ? hw : 1;
    }

    // Set number of worker threads for parallel execution
    // If n == 0, uses hardware_concurrency()
    // Note: For BS backend, this is expensive (recreates thread pool)
    static void set_num_threads(size_t n) {
        if (n == 0) {
            n = hardware_concurrency();
        }

        // Clamp to reasonable maximum to avoid overflow issues
        constexpr size_t MAX_THREADS = 1024;
        if (n > MAX_THREADS) {
            n = MAX_THREADS;
        }

#if defined(SCL_USE_SERIAL)
        (void)n;  // Serial mode ignores thread count

#elif defined(SCL_USE_OPENMP)
        omp_set_num_threads(static_cast<int>(n));

#elif defined(SCL_USE_BS)
        // Expensive operation - do not call in hot loops
        // Resets the entire thread pool with new thread count
        detail::get_global_pool().reset(n);

#elif defined(SCL_USE_TBB)
        // TBB uses global_control with scope lifetime
        // Store in static to keep it alive (global scope, not thread_local)
        static ::std::unique_ptr<tbb::global_control> gc;
        gc = ::std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, static_cast<int>(n));

#else
        // Fallback: should not reach here due to config validation
        (void)n;
#endif
    }

    // Get current number of threads configured for parallel execution
    // Returns at least 1 (guaranteed)
    static size_t get_num_threads() noexcept {
#if defined(SCL_USE_SERIAL)
        return 1;

#elif defined(SCL_USE_OPENMP)
        int n = omp_get_max_threads();
        return (n > 0) ? static_cast<size_t>(n) : 1;

#elif defined(SCL_USE_BS)
        size_t n = detail::get_global_pool().get_thread_count();
        return (n > 0) ? n : 1;

#elif defined(SCL_USE_TBB)
        int n = tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
        return (n > 0) ? static_cast<size_t>(n) : 1;

#else
        // Fallback: should not reach here due to config validation
        return 1;
#endif
    }

    // Initialize scheduler with specified thread count
    // If n == 0, uses hardware_concurrency()
    static void init(size_t n = 0) {
        set_num_threads(n);
    }
};

} // namespace scl::threading
