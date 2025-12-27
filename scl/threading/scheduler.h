// =============================================================================
// FILE: scl/threading/scheduler.h
// BRIEF: API reference for Global Thread Scheduler
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>

namespace scl::threading {

// =============================================================================
// MODULE OVERVIEW
// =============================================================================

/* -----------------------------------------------------------------------------
 * MODULE: Thread Scheduler
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Global thread pool and concurrency configuration for SCL.
 *
 * PURPOSE:
 *     Provides unified interface for controlling parallelism across
 *     multiple threading backends:
 *     - OpenMP (HPC standard)
 *     - Intel TBB (work-stealing scheduler)
 *     - BS::thread_pool (portable, header-only)
 *     - Serial (debug/single-threaded fallback)
 *
 * DESIGN:
 *     - Static class (no instances)
 *     - Facade over backend-specific APIs
 *     - Backend selected at compile time via scl/config.hpp
 *     - Singleton pattern for BS::thread_pool (heap-allocated, never destroyed)
 *
 * BACKEND SELECTION:
 *     Controlled by preprocessor defines in scl/config.hpp:
 *     - SCL_BACKEND_OPENMP -> SCL_USE_OPENMP
 *     - SCL_BACKEND_TBB    -> SCL_USE_TBB
 *     - SCL_BACKEND_BS     -> SCL_USE_BS
 *     - SCL_BACKEND_SERIAL -> SCL_USE_SERIAL
 *
 * THREAD SAFETY:
 *     Safe - all methods are thread-safe, but set_num_threads() should
 *     be called during initialization, not in hot paths
 *
 * PERFORMANCE NOTES:
 *     - set_num_threads() may be expensive (recreates thread pool)
 *     - Call once during initialization
 *     - Avoid calling in performance-critical sections
 * -------------------------------------------------------------------------- */

// =============================================================================
// SCHEDULER CLASS
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: Scheduler
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Static class for global thread pool management.
 *
 * PURPOSE:
 *     Controls concurrency level for all SCL parallel operations:
 *     - Set number of worker threads
 *     - Query hardware capabilities
 *     - Initialize threading subsystem
 *
 * USAGE PATTERN:
 *     Typical initialization:
 *         scl::threading::Scheduler::init();  // Use all cores
 *         scl::threading::Scheduler::init(4); // Use 4 threads
 *
 *     Query capabilities:
 *         size_t cores = Scheduler::hardware_concurrency();
 *         size_t threads = Scheduler::get_num_threads();
 *
 *     Runtime adjustment (not recommended in hot paths):
 *         Scheduler::set_num_threads(8);
 *
 * BACKEND BEHAVIOR:
 *     OpenMP:
 *         - Calls omp_set_num_threads()
 *         - Affects all #pragma omp parallel regions
 *
 *     TBB:
 *         - Uses tbb::global_control internally
 *         - Stored in static variable for lifetime management
 *         - Can also use TBB_NUM_THREADS environment variable
 *
 *     BS::thread_pool:
 *         - Recreates thread pool with new size
 *         - Expensive operation (joins old threads, spawns new)
 *
 *     Serial:
 *         - No-op for set_num_threads()
 *         - Always returns 1 for get_num_threads()
 * -------------------------------------------------------------------------- */
class Scheduler {
public:
    /* -------------------------------------------------------------------------
     * FUNCTION: hardware_concurrency
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get number of hardware threads available.
     *
     * SIGNATURE:
     *     static size_t hardware_concurrency()
     *
     * RETURN VALUE:
     *     Number of concurrent threads supported by hardware
     *
     * POSTCONDITIONS:
     *     Returns std::thread::hardware_concurrency()
     *
     * THREAD SAFETY:
     *     Safe
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     *
     * NOTES:
     *     - Typically equals number of logical CPU cores
     *     - May return 0 if unable to detect
     * ---------------------------------------------------------------------- */
    static size_t hardware_concurrency();

    /* -------------------------------------------------------------------------
     * FUNCTION: set_num_threads
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Set number of worker threads for parallel execution.
     *
     * SIGNATURE:
     *     static void set_num_threads(size_t n)
     *
     * PARAMETERS:
     *     n [in] - Number of threads to use (0 = auto-detect)
     *
     * PRECONDITIONS:
     *     None
     *
     * POSTCONDITIONS:
     *     Thread pool resized to n threads (or hardware_concurrency if n=0)
     *
     * MUTABILITY:
     *     INPLACE - modifies global thread pool state
     *
 * COMPLEXITY:
 *     Time:  O(n) for BS backend (thread creation)
 *            O(1) for OpenMP and TBB
 *     Space: O(n) thread stack space for BS
 *            O(1) for OpenMP and TBB
     *
     * THREAD SAFETY:
     *     Safe but not reentrant - do not call concurrently
     *
     * PERFORMANCE WARNING:
 *     BS backend: Expensive operation (joins and recreates threads)
 *     Do NOT call in hot loops or performance-critical sections
 *
 * BACKEND BEHAVIOR:
 *     OpenMP:  omp_set_num_threads(n) - affects all parallel regions
 *     TBB:     Creates tbb::global_control stored in static variable
 *              (lifetime managed automatically, replaces previous control)
 *     BS:      pool.reset(n) - recreates thread pool (expensive)
 *     Serial:  No-op
     *
     * WHEN TO USE:
     *     - During application initialization
     *     - Before parallel workload execution
     *     - When workload characteristics change significantly
     *
     * WHEN NOT TO USE:
     *     - Inside parallel regions
     *     - In performance-critical loops
     *     - For minor thread count adjustments
     * ---------------------------------------------------------------------- */
    static void set_num_threads(
        size_t n           // Number of threads (0 = auto)
    );

    /* -------------------------------------------------------------------------
     * FUNCTION: get_num_threads
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get currently configured number of worker threads.
     *
     * SIGNATURE:
     *     static size_t get_num_threads()
     *
     * RETURN VALUE:
     *     Current thread pool size
     *
     * POSTCONDITIONS:
     *     Returns active thread count
     *
     * THREAD SAFETY:
     *     Safe
     *
     * COMPLEXITY:
     *     Time:  O(1)
     *     Space: O(1)
     *
     * BACKEND BEHAVIOR:
     *     OpenMP:  omp_get_max_threads()
     *     TBB:     tbb::global_control::active_value()
     *     BS:      pool.get_thread_count()
     *     Serial:  Always returns 1
     *
     * NOTES:
     *     - May differ from hardware_concurrency()
     *     - Reflects last set_num_threads() call
     *     - TBB: Returns system-wide parallelism limit
     * ---------------------------------------------------------------------- */
    static size_t get_num_threads();

    /* -------------------------------------------------------------------------
     * FUNCTION: init
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Initialize threading subsystem.
     *
     * SIGNATURE:
     *     static void init(size_t n = 0)
     *
     * PARAMETERS:
     *     n [in] - Number of threads (0 = auto-detect all cores)
     *
     * POSTCONDITIONS:
     *     Thread pool initialized with n threads
     *
     * MUTABILITY:
     *     INPLACE - initializes global state
     *
     * THREAD SAFETY:
     *     Safe but should be called once during startup
     *
     * COMPLEXITY:
     *     Same as set_num_threads()
     *
     * WHEN TO USE:
     *     - Application startup (main() or library initialization)
     *     - Before any parallel operations
     *
     * NOTES:
     *     - Optional: SCL initializes with all cores by default
     *     - Equivalent to set_num_threads(n)
     *     - Provided for clarity in initialization code
     * ---------------------------------------------------------------------- */
    static void init(
        size_t n = 0       // Number of threads (0 = auto)
    );
};

} // namespace scl::threading
