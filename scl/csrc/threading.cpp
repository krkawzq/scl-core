#include "scl/threading.hpp"

// =============================================================================
// Backend-specific implementations
// =============================================================================

#if defined(SCL_USE_OPENMP)
    #include <omp.h>
    
    namespace scl {
    namespace threading {
    
    void parallel_for(size_t start, size_t end, const std::function<void(size_t)>& func) {
        #pragma omp parallel for
        for (size_t i = start; i < end; ++i) {
            func(i);
        }
    }
    
    size_t num_threads() {
        return static_cast<size_t>(omp_get_max_threads());
    }
    
    void set_num_threads(size_t n) {
        omp_set_num_threads(static_cast<int>(n));
    }
    
    } // namespace threading
    } // namespace scl

#elif defined(SCL_USE_TBB)
    #include <tbb/parallel_for.h>
    #include <tbb/blocked_range.h>
    
    namespace scl {
    namespace threading {
    
    void parallel_for(size_t start, size_t end, const std::function<void(size_t)>& func) {
        tbb::parallel_for(start, end, func);
    }
    
    size_t num_threads() {
        return static_cast<size_t>(tbb::task_scheduler_init::default_num_threads());
    }
    
    void set_num_threads(size_t n) {
        // TBB manages threads automatically, but we can set a limit
        // Note: This requires proper TBB initialization elsewhere
    }
    
    } // namespace threading
    } // namespace scl

#elif defined(SCL_USE_BS)
    #include "BS_thread_pool.hpp"
    #include <thread>
    #include <vector>
    
    namespace scl {
    namespace threading {
    
    // Global thread pool instance (initialized on first use)
    static BS::thread_pool* get_pool() {
        static BS::thread_pool pool(std::thread::hardware_concurrency());
        return &pool;
    }
    
    void parallel_for(size_t start, size_t end, const std::function<void(size_t)>& func) {
        auto* pool = get_pool();
        const size_t chunk_size = (end - start) / pool->get_thread_count();
        
        if (chunk_size == 0) {
            // Small range: execute sequentially
            for (size_t i = start; i < end; ++i) {
                func(i);
            }
            return;
        }
        
        // Submit tasks to thread pool
        for (size_t chunk_start = start; chunk_start < end; chunk_start += chunk_size) {
            size_t chunk_end = (chunk_start + chunk_size < end) ? chunk_start + chunk_size : end;
            pool->push_task([chunk_start, chunk_end, &func]() {
                for (size_t i = chunk_start; i < chunk_end; ++i) {
                    func(i);
                }
            });
        }
        
        pool->wait_for_tasks();
    }
    
    size_t num_threads() {
        return get_pool()->get_thread_count();
    }
    
    void set_num_threads(size_t n) {
        // BS::thread_pool doesn't support dynamic resizing
        // The pool is initialized with hardware_concurrency() threads
    }
    
    } // namespace threading
    } // namespace scl

#elif defined(SCL_USE_SERIAL)
    namespace scl {
    namespace threading {
    
    void parallel_for(size_t start, size_t end, const std::function<void(size_t)>& func) {
        // Serial execution: simple sequential loop
        for (size_t i = start; i < end; ++i) {
            func(i);
        }
    }
    
    size_t num_threads() {
        return 1;
    }
    
    void set_num_threads(size_t n) {
        // No-op for serial backend
    }
    
    } // namespace threading
    } // namespace scl

#else
    #error "No threading backend defined! Check scl/config.hpp"
#endif

