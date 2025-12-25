#pragma once

#include "scl/config.hpp"
#include <cstddef>
#include <functional>

// =============================================================================
/// @file threading.hpp
/// @brief Unified Threading Backend Abstraction Layer
///
/// This header provides a backend-agnostic interface for parallel execution.
/// All kernel code must use this interface instead of directly calling OpenMP,
/// TBB, or pthread APIs.
///
/// @section Architecture
///
/// The threading layer abstracts away the underlying parallelization backend,
/// allowing the build system to switch between:
/// - Serial execution (single-threaded)
/// - OpenMP (industry standard)
/// - Intel TBB (high-performance)
/// - BS::thread_pool (header-only, zero dependency)
///
/// @section Usage
///
/// @code{.cpp}
/// scl::threading::parallel_for(0, size, [](size_t i) {
///     // Your computation here
/// });
/// @endcode
///
/// =============================================================================

namespace scl {
namespace threading {

/// @brief Execute a parallel loop over a range [start, end)
///
/// @param start Starting index (inclusive)
/// @param end Ending index (exclusive)
/// @param func Function to execute for each index
///
/// @note The function object must be thread-safe and not allocate memory.
/// @note This is a zero-overhead abstraction - no virtual calls or indirection.
void parallel_for(size_t start, size_t end, const std::function<void(size_t)>& func);

/// @brief Get the number of available threads
///
/// @return Number of threads available for parallel execution
size_t num_threads();

/// @brief Set the number of threads (if supported by backend)
///
/// @param n Number of threads to use
void set_num_threads(size_t n);

} // namespace threading
} // namespace scl

