#pragma once

#include "scl/config.hpp"
#include "scl/core/macros.hpp"
#include <thread>
#include <algorithm>
#include <cstddef>

// =============================================================================
// Backend Headers
// =============================================================================

#if defined(SCL_USE_BS)
    #include "BS_thread_pool.hpp"
#elif defined(SCL_USE_OPENMP)
    #include <omp.h>
#elif defined(SCL_USE_TBB)
    #include <tbb/global_control.h>
    #include <memory>
#endif

namespace scl::threading {

// =============================================================================
// Internal State
// =============================================================================

namespace detail {
#if defined(SCL_USE_BS)
    // Global singleton for BS::thread_pool (never destroyed)
    inline BS::thread_pool& get_global_pool() {
        static BS::thread_pool* pool = new BS::thread_pool();
        return *pool;
    }
#endif

#if defined(SCL_USE_TBB)
    // TBB thread management deferred to environment variables or external init
#endif
}

// =============================================================================
// Public Scheduler API
// =============================================================================

class Scheduler {
public:
    SCL_FORCE_INLINE static size_t hardware_concurrency() {
        return std::thread::hardware_concurrency();
    }

    static void set_num_threads(size_t n) {
        if (n == 0) {
            n = hardware_concurrency();
        }

#if defined(SCL_USE_SERIAL)
        (void)n;

#elif defined(SCL_USE_OPENMP)
        omp_set_num_threads(static_cast<int>(n));

#elif defined(SCL_USE_BS)
        // Expensive operation - do not call in hot loops
        detail::get_global_pool().reset(n);

#elif defined(SCL_USE_TBB)
        // TBB uses global_control with scope lifetime
        // Store in static to keep it alive (global scope, not thread_local)
        static std::unique_ptr<tbb::global_control> gc;
        gc = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, n);
#endif
    }

    static size_t get_num_threads() {
#if defined(SCL_USE_SERIAL)
        return 1;

#elif defined(SCL_USE_OPENMP)
        return static_cast<size_t>(omp_get_max_threads());

#elif defined(SCL_USE_BS)
        return detail::get_global_pool().get_thread_count();

#elif defined(SCL_USE_TBB)
        return tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);

#else
        return 1;
#endif
    }

    static void init(size_t n = 0) {
        set_num_threads(n);
    }
};

} // namespace scl::threading
