#pragma once

/**
 * @file scl/core/threading.hpp
 * @brief Unified threading abstraction for SCL.
 *
 * This header provides a unified interface for parallel execution that
 * abstracts away the differences between threading backends:
 *   - OpenMP (SCL_USE_OPENMP)
 *   - Intel TBB (SCL_USE_TBB)
 *   - BS::thread_pool (SCL_USE_BS)
 *   - Serial execution (default)
 *
 * Key Features:
 *   - parallel_for: Parallel loop execution
 *   - parallel_reduce: Parallel reduction operations
 *   - get_num_threads / set_num_threads: Thread count management
 *   - get_backend_name: Runtime backend identification
 *   - ThreadLocal<T>: Thread-local storage wrapper
 *   - ThreadWorkArea<T>: Reusable per-thread work buffers
 *
 * @note All functions are designed for zero-overhead abstraction where possible
 */

#include "scl/config.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

// Backend-specific includes
#if defined(SCL_USE_OPENMP)
  #include <omp.h>
#elif defined(SCL_USE_TBB)
  #include <tbb/blocked_range.h>
  #include <tbb/global_control.h>
  #include <tbb/parallel_for.h>
  #include <tbb/parallel_reduce.h>
  #include <tbb/task_arena.h>
#elif defined(SCL_USE_BS)
  #include "BS_thread_pool.hpp"
#endif

namespace scl::threading {

// =============================================================================
// SECTION 1: Type Aliases and Constants
// =============================================================================

/// @brief Index type for parallel loops
using Index = std::ptrdiff_t;

/// @brief Size type
using Size = std::size_t;

/// @brief Default grain size for parallel partitioning
inline constexpr Size DEFAULT_GRAIN_SIZE = 1024;

/// @brief Minimum elements per thread for parallel execution
inline constexpr Size MIN_PARALLEL_SIZE = 4096;

/// @brief Default minimum iterations per thread for grain size calculation
inline constexpr Size DEFAULT_MIN_PER_THREAD = 64;

// =============================================================================
// SECTION 2: Backend Information
// =============================================================================

/// @brief Threading backend enumeration
enum class Backend : std::uint8_t {
  Serial,  ///< No parallelization
  OpenMP,  ///< OpenMP backend
  TBB,     ///< Intel TBB backend
  BS       ///< BS::thread_pool backend
};

/// @brief Get the current threading backend
/// @return Backend enum value
[[nodiscard]]
constexpr
auto get_backend() noexcept -> Backend {
#if defined(SCL_USE_OPENMP)
  return Backend::OpenMP;
#elif defined(SCL_USE_TBB)
  return Backend::TBB;
#elif defined(SCL_USE_BS)
  return Backend::BS;
#else
  return Backend::Serial;
#endif
}

/// @brief Get the threading backend name as string
/// @return Backend name string
[[nodiscard]]
constexpr
auto get_backend_name() noexcept -> const char* {
#if defined(SCL_USE_OPENMP)
  return "OpenMP";
#elif defined(SCL_USE_TBB)
  return "TBB";
#elif defined(SCL_USE_BS)
  return "BS::thread_pool";
#else
  return "Serial";
#endif
}

/// @brief Check if parallel execution is available
/// @return true if backend supports parallelism
[[nodiscard]]
constexpr
auto is_parallel_available() noexcept -> bool {
  return get_backend() != Backend::Serial;
}

// =============================================================================
// SECTION 3: Thread Count Management
// =============================================================================

namespace detail {

#if defined(SCL_USE_BS)
/// @brief Global thread pool instance for BS backend
inline auto& get_thread_pool() {
  static BS::thread_pool pool;
  return pool;
}
#endif

}  // namespace detail

/// @brief Get the number of available threads
/// @return Number of threads available for parallel execution
[[nodiscard]]
inline
auto get_num_threads() noexcept -> Size {
#if defined(SCL_USE_OPENMP)
  return static_cast<Size>(omp_get_max_threads());
#elif defined(SCL_USE_TBB)
  return static_cast<Size>(tbb::this_task_arena::max_concurrency());
#elif defined(SCL_USE_BS)
  return detail::get_thread_pool().get_thread_count();
#else
  return 1;
#endif
}

/// @brief Set the number of threads for parallel execution
/// @param[in] num_threads Desired number of threads (0 = use hardware default)
/// @note Some backends may not support dynamic thread count changes
inline
auto set_num_threads(Size num_threads) noexcept -> void {
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
  }

#if defined(SCL_USE_OPENMP)
  omp_set_num_threads(static_cast<int>(num_threads));
#elif defined(SCL_USE_TBB)
  // TBB uses global_control for thread limits
  static std::unique_ptr<tbb::global_control> gc;
  gc = std::make_unique<tbb::global_control>(
      tbb::global_control::max_allowed_parallelism, num_threads);
#elif defined(SCL_USE_BS)
  // BS::thread_pool doesn't support resizing after construction
  (void)num_threads;
#else
  (void)num_threads;
#endif
}

/// @brief Get the current thread index within a parallel region
/// @return Thread index (0 to num_threads-1), or 0 if not in parallel region
[[nodiscard]]
inline
auto get_thread_id() noexcept -> Size {
#if defined(SCL_USE_OPENMP)
  return static_cast<Size>(omp_get_thread_num());
#elif defined(SCL_USE_TBB)
  return static_cast<Size>(tbb::this_task_arena::current_thread_index());
#else
  return 0;
#endif
}

/// @brief Get hardware concurrency (number of logical CPU cores)
/// @return Number of logical CPU cores
[[nodiscard]]
inline
auto hardware_concurrency() noexcept -> Size {
  return std::thread::hardware_concurrency();
}

// =============================================================================
// SECTION 4: Parallel For
// =============================================================================

/// @brief Execute a function in parallel over a range
/// @tparam Func Callable type with signature void(Index)
/// @param[in] begin Start index (inclusive)
/// @param[in] end End index (exclusive)
/// @param[in] func Function to execute for each index
/// @param[in] grain_size Minimum iterations per thread (hint)
///
/// @note The function is called with indices in [begin, end)
/// @note Order of execution is not guaranteed
template <typename Func>
inline
auto parallel_for(Index begin, Index end, Func&& func,
                  Size grain_size = DEFAULT_GRAIN_SIZE) -> void {
  if (begin >= end) {
    return;
  }

  const auto count = static_cast<Size>(end - begin);

  // Fall back to serial for small ranges
  if (count < MIN_PARALLEL_SIZE || !is_parallel_available()) {
    for (Index i = begin; i < end; ++i) {
      func(i);
    }
    return;
  }

#if defined(SCL_USE_OPENMP)
  (void)grain_size;
  #pragma omp parallel for schedule(static)
  for (Index i = begin; i < end; ++i) {
    func(i);
  }

#elif defined(SCL_USE_TBB)
  tbb::parallel_for(tbb::blocked_range<Index>(begin, end, grain_size),
                    [&func](const tbb::blocked_range<Index>& range) {
                      for (Index i = range.begin(); i < range.end(); ++i) {
                        func(i);
                      }
                    });

#elif defined(SCL_USE_BS)
  detail::get_thread_pool().detach_blocks(
      begin, end,
      [&func](Index block_begin, Index block_end) {
        for (Index i = block_begin; i < block_end; ++i) {
          func(i);
        }
      },
      static_cast<std::size_t>(get_num_threads()));
  detail::get_thread_pool().wait();

#else
  // Serial fallback
  (void)grain_size;
  for (Index i = begin; i < end; ++i) {
    func(i);
  }
#endif
}

/// @brief Convenience overload for range starting at 0
/// @tparam Func Callable type with signature void(Index)
/// @param[in] count Number of iterations
/// @param[in] func Function to execute for each index
template <typename Func>
inline
auto parallel_for(Size count, Func&& func) -> void {
  parallel_for(Index{0}, static_cast<Index>(count), std::forward<Func>(func));
}

// =============================================================================
// SECTION 5: Parallel Reduce
// =============================================================================

/// @brief Execute a parallel reduction over a range
/// @tparam T Result type
/// @tparam MapFunc Map function type: T(Index)
/// @tparam ReduceFunc Reduce function type: T(T, T)
/// @param[in] begin Start index (inclusive)
/// @param[in] end End index (exclusive)
/// @param[in] identity Identity element for reduction
/// @param[in] map Function to compute value at each index
/// @param[in] reduce Function to combine two values
/// @return Final reduced value
template <typename T, typename MapFunc, typename ReduceFunc>
[[nodiscard]]
auto parallel_reduce(Index begin, Index end, T identity,
                     MapFunc&& map, ReduceFunc&& reduce) -> T {
  if (begin >= end) {
    return identity;
  }

  const auto count = static_cast<Size>(end - begin);

  // Fall back to serial for small ranges
  if (count < MIN_PARALLEL_SIZE || !is_parallel_available()) {
    T result = identity;
    for (Index i = begin; i < end; ++i) {
      result = reduce(result, map(i));
    }
    return result;
  }

#if defined(SCL_USE_OPENMP)
  T result = identity;
  #pragma omp parallel
  {
    T local_result = identity;
    #pragma omp for nowait schedule(static)
    for (Index i = begin; i < end; ++i) {
      local_result = reduce(local_result, map(i));
    }
    #pragma omp critical
    {
      result = reduce(result, local_result);
    }
  }
  return result;

#elif defined(SCL_USE_TBB)
  return tbb::parallel_reduce(
      tbb::blocked_range<Index>(begin, end), identity,
      [&map, &reduce](const tbb::blocked_range<Index>& range, T init) {
        for (Index i = range.begin(); i < range.end(); ++i) {
          init = reduce(init, map(i));
        }
        return init;
      },
      reduce);

#elif defined(SCL_USE_BS)
  // Use mutex-protected vector to avoid hash collision issues
  std::vector<T> partial_results;
  std::mutex results_mutex;

  detail::get_thread_pool().detach_blocks(
      begin, end,
      [&map, &reduce, identity, &partial_results,
       &results_mutex](Index block_begin, Index block_end) {
        // Compute local result for this block
        T local_result = identity;
        for (Index i = block_begin; i < block_end; ++i) {
          local_result = reduce(local_result, map(i));
        }
        // Store result with mutex protection
        {
          std::lock_guard<std::mutex> lock(results_mutex);
          partial_results.push_back(local_result);
        }
      },
      get_num_threads());
  detail::get_thread_pool().wait();

  // Combine all partial results
  T result = identity;
  for (const auto& partial : partial_results) {
    result = reduce(result, partial);
  }
  return result;

#else
  // Serial fallback
  T result = identity;
  for (Index i = begin; i < end; ++i) {
    result = reduce(result, map(i));
  }
  return result;
#endif
}

/// @brief Parallel sum reduction (convenience function)
/// @tparam T Result type
/// @tparam MapFunc Map function type: T(Index)
/// @param[in] begin Start index (inclusive)
/// @param[in] end End index (exclusive)
/// @param[in] map Function to compute value at each index
/// @return Sum of all mapped values
template <typename T, typename MapFunc>
[[nodiscard]]
auto parallel_sum(Index begin, Index end, MapFunc&& map) -> T {
  return parallel_reduce(begin, end, T{0}, std::forward<MapFunc>(map),
                         std::plus<T>{});
}

// =============================================================================
// SECTION 6: Parallel Sections
// =============================================================================

/// @brief Execute multiple independent tasks in parallel
/// @tparam Funcs Callable types with signature void()
/// @param[in] funcs Functions to execute in parallel
template <typename... Funcs>
inline
auto parallel_invoke(Funcs&&... funcs) -> void {
  if constexpr (sizeof...(Funcs) == 0) {
    return;
  } else if constexpr (sizeof...(Funcs) == 1) {
    // Single function: just execute it
    (funcs(), ...);
  } else {
#if defined(SCL_USE_OPENMP)
    // Use tasks instead of sections for proper variadic expansion
    #pragma omp parallel
    {
      #pragma omp single
      {
        (
            [&] {
              #pragma omp task
              { funcs(); }
            }(),
            ...);
      }
    }

#elif defined(SCL_USE_TBB)
    tbb::parallel_invoke(std::forward<Funcs>(funcs)...);

#elif defined(SCL_USE_BS)
    auto& pool = detail::get_thread_pool();
    (pool.detach_task(funcs), ...);
    pool.wait();

#else
    // Serial fallback
    (funcs(), ...);
#endif
  }
}

// =============================================================================
// SECTION 7: Synchronization Primitives
// =============================================================================

/// @brief Execute function once across all threads (critical section)
/// @tparam Func Callable type: void()
/// @param[in] func Function to execute once
template <typename Func>
inline
auto critical(Func&& func) -> void {
#if defined(SCL_USE_OPENMP)
  #pragma omp critical
  { func(); }
#else
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  func();
#endif
}

/// @brief Memory fence for synchronization
inline
auto memory_fence() noexcept -> void {
  std::atomic_thread_fence(std::memory_order_seq_cst);
}

// =============================================================================
// SECTION 8: Utility Functions
// =============================================================================

/// @brief Check if current execution is within a parallel region
/// @return true if inside a parallel region
[[nodiscard]]
inline
auto in_parallel_region() noexcept -> bool {
#if defined(SCL_USE_OPENMP)
  return omp_in_parallel() != 0;
#else
  return false;  // Cannot reliably detect for other backends
#endif
}

/// @brief Calculate optimal grain size for parallel loop
/// @param[in] total_size Total number of iterations
/// @param[in] min_per_thread Minimum iterations per thread
/// @return Recommended grain size
[[nodiscard]]
inline
auto calculate_grain_size(Size total_size, 
                          Size min_per_thread = DEFAULT_MIN_PER_THREAD) noexcept 
    -> Size {
  const Size num_threads = get_num_threads();
  if (num_threads == 0) {
    return total_size;
  }

  const Size per_thread = total_size / num_threads;
  return std::max(min_per_thread, per_thread);
}

/// @brief Determine if parallel execution is beneficial for given size
/// @param[in] size Number of elements to process
/// @param[in] element_cost Estimated cost per element (higher = more beneficial)
/// @return true if parallel execution is recommended
[[nodiscard]]
inline
auto should_parallelize(Size size, Size element_cost = 1) noexcept -> bool {
  if (!is_parallel_available()) {
    return false;
  }
  return size * element_cost >= MIN_PARALLEL_SIZE;
}

// =============================================================================
// SECTION 9: Thread-Local Storage
// =============================================================================

/// @brief Thread-local value wrapper for parallel regions
/// @tparam T Value type
///
/// Provides a thread-safe way to maintain per-thread state in parallel regions.
///
/// Example:
/// @code
///   scl::threading::ThreadLocal<std::vector<double>> local_buffers(
///       []() { return std::vector<double>(1024); }
///   );
///
///   scl::threading::parallel_for(0, n, [&](Index i) {
///       auto& buffer = local_buffers.get();
///       // use buffer...
///   });
/// @endcode
template <typename T>
class ThreadLocal {
 public:
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using initializer_type = std::function<T()>;

  /// @brief Construct with default initialization
  ThreadLocal() : ThreadLocal([]() { return T{}; }) {}

  /// @brief Construct with custom initializer
  /// @param[in] init Function that creates a new T for each thread
  explicit ThreadLocal(initializer_type init)
      : initializer_(std::move(init)),
        values_(hardware_concurrency()),
        initialized_(hardware_concurrency()) {
    for (auto& flag : initialized_) {
      flag.store(false, std::memory_order_relaxed);
    }
  }

  // Non-copyable, movable
  ThreadLocal(const ThreadLocal&) = delete;
  auto operator=(const ThreadLocal&) -> ThreadLocal& = delete;
  ThreadLocal(ThreadLocal&&) noexcept = default;
  auto operator=(ThreadLocal&&) noexcept -> ThreadLocal& = default;
  ~ThreadLocal() = default;

  /// @brief Get reference to thread-local value
  /// @return Reference to value for current thread
  [[nodiscard]]
  auto get() -> reference {
    const Size idx = get_thread_index();
    ensure_initialized(idx);
    return values_[idx];
  }

  /// @brief Get const reference to thread-local value
  /// @note Will initialize the value if not already initialized
  [[nodiscard]]
  auto get() const -> const_reference {
    const Size idx = get_thread_index();
    ensure_initialized(idx);
    return values_[idx];
  }

  /// @brief Apply function to all initialized values
  /// @tparam Func Callable type: void(T&) or R(T&)
  /// @param[in] func Function to apply
  template <typename Func>
  auto for_each(Func&& func) -> void {
    for (Size i = 0; i < values_.size(); ++i) {
      if (initialized_[i].load(std::memory_order_acquire)) {
        func(values_[i]);
      }
    }
  }

  /// @brief Combine all values using a reduce function
  /// @tparam ReduceFunc Callable type: T(T, T)
  /// @param[in] identity Identity element
  /// @param[in] reduce Reduction function
  /// @return Combined result
  template <typename ReduceFunc>
  [[nodiscard]]
  auto combine(T identity, ReduceFunc&& reduce) const -> T {
    T result = identity;
    for (Size i = 0; i < values_.size(); ++i) {
      if (initialized_[i].load(std::memory_order_acquire)) {
        result = reduce(result, values_[i]);
      }
    }
    return result;
  }

  /// @brief Clear all values and reset initialization flags
  auto clear() -> void {
    for (Size i = 0; i < values_.size(); ++i) {
      if (initialized_[i].load(std::memory_order_acquire)) {
        values_[i] = T{};
        initialized_[i].store(false, std::memory_order_release);
      }
    }
  }

 private:
  /// @brief Ensure value at index is initialized
  auto ensure_initialized(Size idx) const -> void {
    // Double-checked locking with atomic CAS
    if (!initialized_[idx].load(std::memory_order_acquire)) {
      // Create temporary value outside the critical section
      T temp = initializer_();

      // Try to claim initialization responsibility
      bool expected = false;
      if (initialized_[idx].compare_exchange_strong(
              expected, true, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        // We won the race, store our value
        values_[idx] = std::move(temp);
      }
      // If CAS failed, another thread already initialized - discard temp
    }
  }

  [[nodiscard]]
  static auto get_thread_index() -> Size {
#if defined(SCL_USE_OPENMP)
    return static_cast<Size>(omp_get_thread_num());
#elif defined(SCL_USE_TBB)
    auto idx = tbb::this_task_arena::current_thread_index();
    return idx >= 0 ? static_cast<Size>(idx) : 0;
#else
    // Hash thread ID for other backends
    static thread_local Size cached_index =
        std::hash<std::thread::id>{}(std::this_thread::get_id()) %
        std::thread::hardware_concurrency();
    return cached_index;
#endif
  }

  initializer_type initializer_;
  mutable std::vector<T> values_;
  mutable std::vector<std::atomic<bool>> initialized_;
};

// =============================================================================
// SECTION 10: Thread Work Area (Reusable Per-Thread Buffers)
// =============================================================================

/// @brief Thread-independent work area with separated init/reset logic
/// @tparam T Work area type (e.g., std::vector<double>, custom buffer)
///
/// This class provides efficient per-thread work buffers that:
/// - Initialize lazily (only when first accessed by a thread)
/// - Support reset operations (cheaper than re-initialization)
/// - Can be pre-allocated for all threads before parallel execution
/// - Avoid frequent memory allocations in hot loops
///
/// The key difference from ThreadLocal is the separation of:
/// - Initializer: Called ONCE per thread to allocate resources
/// - Resetter: Called before EACH use to reset state (not reallocate)
///
/// Example:
/// @code
///   // Work buffer that allocates once, resets each iteration
///   scl::threading::ThreadWorkArea<std::vector<double>> work_buffers(
///       // Initializer: allocate buffer (called once per thread)
///       [](std::vector<double>& buf) {
///           buf.resize(1024);
///       },
///       // Resetter: clear contents but keep capacity (called each use)
///       [](std::vector<double>& buf) {
///           buf.clear();  // O(1), doesn't deallocate
///       }
///   );
///
///   // Pre-allocate all buffers before parallel work
///   work_buffers.preallocate();
///
///   for (int iter = 0; iter < 100; ++iter) {
///       // Reset all buffers (cheap, no allocation)
///       work_buffers.reset_all();
///
///       scl::threading::parallel_for(0, n, [&](Index i) {
///           auto& buf = work_buffers.get();  // Already initialized & reset
///           // ... use buf ...
///       });
///
///       // Combine results from all threads
///       work_buffers.for_each([&](auto& buf) {
///           // process each thread's buffer
///       });
///   }
/// @endcode
template <typename T>
class ThreadWorkArea {
 public:
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;

  /// @brief Initializer function type: void(T&) - sets up initial state
  using initializer_type = std::function<void(T&)>;

  /// @brief Resetter function type: void(T&) - resets for reuse
  using resetter_type = std::function<void(T&)>;

  /// @brief Default constructor with trivial init/reset
  ThreadWorkArea()
      : initializer_([](T&) {}),
        resetter_([](T&) {}),
        num_slots_(hardware_concurrency()),
        values_(num_slots_),
        state_(num_slots_) {
    init_state();
  }

  /// @brief Construct with initializer only (no reset logic)
  /// @param[in] init Function to initialize work area (called once per thread)
  explicit ThreadWorkArea(initializer_type init)
      : initializer_(std::move(init)),
        resetter_(nullptr),
        num_slots_(hardware_concurrency()),
        values_(num_slots_),
        state_(num_slots_) {
    init_state();
  }

  /// @brief Construct with initializer and resetter
  /// @param[in] init Function to initialize work area (called once per thread)
  /// @param[in] reset Function to reset work area (called before each use)
  ThreadWorkArea(initializer_type init, resetter_type reset)
      : initializer_(std::move(init)),
        resetter_(std::move(reset)),
        num_slots_(hardware_concurrency()),
        values_(num_slots_),
        state_(num_slots_) {
    init_state();
  }

  // Non-copyable, movable
  ThreadWorkArea(const ThreadWorkArea&) = delete;
  auto operator=(const ThreadWorkArea&) -> ThreadWorkArea& = delete;
  ThreadWorkArea(ThreadWorkArea&&) noexcept = default;
  auto operator=(ThreadWorkArea&&) noexcept -> ThreadWorkArea& = default;
  ~ThreadWorkArea() = default;

  /// @brief Get reference to current thread's work area
  /// @return Reference to work area (initialized and ready to use)
  /// @note Initializes on first access, resets if marked for reset
  [[nodiscard]]
  auto get() -> reference {
    const Size idx = get_thread_index();
    ensure_ready(idx);
    return values_[idx];
  }

  /// @brief Get const reference to current thread's work area
  [[nodiscard]]
  auto get() const -> const_reference {
    const Size idx = get_thread_index();
    ensure_ready(idx);
    return values_[idx];
  }

  /// @brief Pre-allocate and initialize all thread work areas
  /// @note Call this before entering parallel region for best performance
  /// @note Thread-safe, can be called from any thread
  auto preallocate() -> void {
    for (Size i = 0; i < num_slots_; ++i) {
      ensure_initialized(i);
    }
  }

  /// @brief Mark all work areas for reset (will reset on next get())
  /// @note Does NOT immediately reset - lazy reset on next access
  /// @note Very cheap O(n) where n = number of threads
  auto mark_reset_all() noexcept -> void {
    for (Size i = 0; i < num_slots_; ++i) {
      auto expected = State::Ready;
      state_[i].compare_exchange_strong(expected, State::NeedsReset,
                                        std::memory_order_release,
                                        std::memory_order_relaxed);
    }
  }

  /// @brief Immediately reset all initialized work areas
  /// @note More expensive than mark_reset_all(), but guarantees reset state
  auto reset_all() -> void {
    if (!resetter_) {
      return;
    }

    for (Size i = 0; i < num_slots_; ++i) {
      State s = state_[i].load(std::memory_order_acquire);
      if (s == State::Ready || s == State::NeedsReset) {
        resetter_(values_[i]);
        state_[i].store(State::Ready, std::memory_order_release);
      }
    }
  }

  /// @brief Apply function to all initialized work areas
  /// @tparam Func Callable type: void(T&)
  /// @param[in] func Function to apply to each work area
  template <typename Func>
  auto for_each(Func&& func) -> void {
    for (Size i = 0; i < num_slots_; ++i) {
      State s = state_[i].load(std::memory_order_acquire);
      if (s != State::Uninitialized) {
        func(values_[i]);
      }
    }
  }

  /// @brief Apply function to all initialized work areas (const version)
  template <typename Func>
  auto for_each(Func&& func) const -> void {
    for (Size i = 0; i < num_slots_; ++i) {
      State s = state_[i].load(std::memory_order_acquire);
      if (s != State::Uninitialized) {
        func(values_[i]);
      }
    }
  }

  /// @brief Combine all work areas using a reduce function
  /// @tparam R Result type
  /// @tparam ReduceFunc Callable type: R(R, const T&)
  /// @param[in] identity Identity element for reduction
  /// @param[in] reduce Reduction function
  /// @return Combined result from all work areas
  template <typename R, typename ReduceFunc>
  [[nodiscard]]
  auto combine(R identity, ReduceFunc&& reduce) const -> R {
    R result = std::move(identity);
    for (Size i = 0; i < num_slots_; ++i) {
      State s = state_[i].load(std::memory_order_acquire);
      if (s != State::Uninitialized) {
        result = reduce(std::move(result), values_[i]);
      }
    }
    return result;
  }

  /// @brief Get number of currently initialized work areas
  [[nodiscard]]
  auto num_initialized() const noexcept -> Size {
    Size count = 0;
    for (Size i = 0; i < num_slots_; ++i) {
      if (state_[i].load(std::memory_order_acquire) != State::Uninitialized) {
        ++count;
      }
    }
    return count;
  }

  /// @brief Get total number of slots (== hardware_concurrency())
  [[nodiscard]]
  auto num_slots() const noexcept -> Size {
    return num_slots_;
  }

  /// @brief Check if all slots are initialized
  [[nodiscard]]
  auto all_initialized() const noexcept -> bool {
    return num_initialized() == num_slots_;
  }

  /// @brief Clear all work areas (deallocate and reset to uninitialized state)
  auto clear() -> void {
    for (Size i = 0; i < num_slots_; ++i) {
      if (state_[i].load(std::memory_order_acquire) != State::Uninitialized) {
        values_[i] = T{};
        state_[i].store(State::Uninitialized, std::memory_order_release);
      }
    }
  }

 private:
  /// @brief Work area state
  enum class State : std::uint8_t {
    Uninitialized = 0,  ///< Not yet initialized
    NeedsReset = 1,     ///< Initialized but needs reset before use
    Ready = 2           ///< Ready for use
  };

  /// @brief Initialize state vector (called from constructors)
  auto init_state() -> void {
    for (auto& s : state_) {
      s.store(State::Uninitialized, std::memory_order_relaxed);
    }
  }

  /// @brief Ensure work area at index is initialized
  auto ensure_initialized(Size idx) const -> void {
    State expected = State::Uninitialized;
    if (state_[idx].compare_exchange_strong(
            expected, State::Ready, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      // We won the race, initialize
      if (initializer_) {
        initializer_(values_[idx]);
      }
    }
  }

  /// @brief Ensure work area at index is ready for use
  auto ensure_ready(Size idx) const -> void {
    State s = state_[idx].load(std::memory_order_acquire);

    if (s == State::Uninitialized) {
      // Need to initialize
      ensure_initialized(idx);
    } else if (s == State::NeedsReset) {
      // Need to reset - use CAS to ensure only one thread resets
      State expected = State::NeedsReset;
      if (state_[idx].compare_exchange_strong(
              expected, State::Ready, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        // We won the race, perform reset
        if (resetter_) {
          resetter_(values_[idx]);
        }
      }
      // If CAS failed, another thread is resetting or already reset
    }
    // State::Ready - already good to go
  }

  [[nodiscard]]
  static auto get_thread_index() -> Size {
#if defined(SCL_USE_OPENMP)
    const int tid = omp_get_thread_num();
    return static_cast<Size>(tid >= 0 ? tid : 0);
#elif defined(SCL_USE_TBB)
    const auto idx = tbb::this_task_arena::current_thread_index();
    return idx >= 0 ? static_cast<Size>(idx) : 0;
#else
    // Hash thread ID for other backends
    static thread_local Size cached_index =
        std::hash<std::thread::id>{}(std::this_thread::get_id()) %
        std::thread::hardware_concurrency();
    return cached_index;
#endif
  }

  initializer_type initializer_;
  resetter_type resetter_;
  Size num_slots_;
  mutable std::vector<T> values_;
  mutable std::vector<std::atomic<State>> state_;
};

// =============================================================================
// SECTION 11: Work Area Factory Functions
// =============================================================================

/// @brief Convenience factory for creating ThreadWorkArea with vector buffers
/// @tparam T Element type of the buffer
/// @param[in] buffer_size Size of each thread's buffer
/// @return ThreadWorkArea with pre-sized vectors that clear on reset
///
/// Example:
/// @code
///   auto work = scl::threading::make_work_buffers<double>(1024);
///   work.preallocate();
///
///   parallel_for(0, n, [&](Index i) {
///       auto& buf = work.get();  // std::vector<double> with capacity 1024
///       buf.push_back(data[i]);
///   });
/// @endcode
template <typename T>
[[nodiscard]]
auto make_work_buffers(Size buffer_size) -> ThreadWorkArea<std::vector<T>> {
  return ThreadWorkArea<std::vector<T>>(
      // Initializer: allocate buffer
      [buffer_size](std::vector<T>& buf) { buf.reserve(buffer_size); },
      // Resetter: clear but keep capacity
      [](std::vector<T>& buf) { buf.clear(); });
}

/// @brief Convenience factory for fixed-size array work areas
/// @tparam T Element type
/// @tparam N Array size
/// @return ThreadWorkArea with std::array buffers that zero on reset
template <typename T, std::size_t N>
[[nodiscard]]
auto make_fixed_work_buffers() -> ThreadWorkArea<std::array<T, N>> {
  return ThreadWorkArea<std::array<T, N>>(
      // Initializer: zero-fill
      [](std::array<T, N>& buf) { buf.fill(T{}); },
      // Resetter: zero-fill
      [](std::array<T, N>& buf) { buf.fill(T{}); });
}

}  // namespace scl::threading
