#pragma once

#include "scl/config.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/scheduler.hpp"

#include <cstddef>
#include <utility>
#include <type_traits>
#include <vector>
#include <future>

// =============================================================================
// Backend Specific Headers
// =============================================================================

#if defined(SCL_USE_TBB)
    #include <tbb/parallel_for.h>
    #include <tbb/blocked_range.h>
    #include <tbb/task_arena.h>
#elif defined(SCL_USE_OPENMP)
    #include <omp.h>
#endif

namespace scl::threading {

// =============================================================================
// Parallel Loop Interface
// =============================================================================

// Unified parallel loop supporting both single-arg and dual-arg (with thread rank) lambdas
// Usage:
//   parallel_for(0, n, [&](size_t i) { ... });                    // Single arg
//   parallel_for(0, n, [&](size_t i, size_t thread_rank) { ... }); // Dual arg
template <typename Func>
inline void parallel_for(size_t start, size_t end, Func&& func) {
    if (SCL_UNLIKELY(start >= end)) {
        return;
    }

    // Detect if func accepts two arguments (index, thread_rank)
    constexpr bool has_rank_arg = std::is_invocable_v<Func, size_t, size_t>;

#if defined(SCL_USE_SERIAL)
    for (size_t i = start; i < end; ++i) {
        if constexpr (has_rank_arg) {
            func(i, 0);
        } else {
            func(i);
        }
    }

#elif defined(SCL_USE_OPENMP)
    if (omp_in_parallel()) {
        for (size_t i = start; i < end; ++i) {
            if constexpr (has_rank_arg) {
                func(i, static_cast<size_t>(omp_get_thread_num()));
            } else {
                func(i);
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t i = start; i < end; ++i) {
            if constexpr (has_rank_arg) {
                func(i, static_cast<size_t>(omp_get_thread_num()));
            } else {
                func(i);
            }
        }
    }

#elif defined(SCL_USE_TBB)
    tbb::parallel_for(tbb::blocked_range<size_t>(start, end),
        [&](const tbb::blocked_range<size_t>& r) {
            size_t thread_rank = tbb::this_task_arena::current_thread_index();
            for (size_t i = r.begin(); i != r.end(); ++i) {
                if constexpr (has_rank_arg) {
                    func(i, thread_rank);
                } else {
                    func(i);
                }
            }
        });

#elif defined(SCL_USE_BS)
    auto& pool = detail::get_global_pool();
    const size_t num_threads = pool.get_thread_count();
    const size_t range_size = end - start;
    const size_t chunk_size = (range_size + num_threads - 1) / num_threads;

    if (chunk_size == 0) return;

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    size_t thread_rank = 0;
    for (size_t chunk_start = start; chunk_start < end; chunk_start += chunk_size) {
        const size_t chunk_end = (chunk_start + chunk_size < end) ? (chunk_start + chunk_size) : end;
        const size_t rank = thread_rank++;

        futures.push_back(pool.submit([&func, chunk_start, chunk_end, rank]() {
            for (size_t i = chunk_start; i < chunk_end; ++i) {
                if constexpr (has_rank_arg) {
                    func(i, rank);
                } else {
                    func(i);
                }
            }
        }));
    }

    for (auto& future : futures) {
        future.get();
    }

#else
    #if defined(_MSC_VER)
        #pragma message("SCL: No threading backend defined, falling back to serial")
    #else
        #warning "No threading backend defined, falling back to serial"
    #endif
    for (size_t i = start; i < end; ++i) {
        if constexpr (has_rank_arg) {
            func(i, 0);
        } else {
            func(i);
        }
    }
#endif
}

} // namespace scl::threading
