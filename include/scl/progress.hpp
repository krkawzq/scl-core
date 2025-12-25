#pragma once

#include <cstddef>
#include <atomic>
#include <cstdint>

// =============================================================================
/// @file progress.hpp
/// @brief Asynchronous Progress Tracking System
///
/// This header provides a non-intrusive progress tracking mechanism for
/// long-running operations. Progress updates are lock-free and asynchronous
/// to avoid impacting computation performance.
///
/// @section Architecture
///
/// Progress tracking uses a pre-allocated pool of progress slots:
/// 1. Operators request a progress buffer via SCL_GET_PROGRESS macro
/// 2. Progress is updated using lock-free atomic operations
/// 3. Python can poll progress asynchronously without blocking computation
/// 4. Precision is secondary to performance
///
/// @section Usage
///
/// @code{.cpp}
/// SCL_GET_PROGRESS(progress_ptr);
/// for (size_t i = 0; i < size; ++i) {
///     // Computation...
///     if (i % 1000 == 0) {
///         *progress_ptr = (i * 100) / size;  // Update progress (0-100)
///     }
/// }
/// @endcode
///
/// =============================================================================

namespace scl {
namespace progress {

/// @brief Progress value type (0-100, representing percentage)
using ProgressValue = std::uint8_t;

/// @brief Progress slot in the pool
struct ProgressSlot {
    std::atomic<ProgressValue> value{0};
    std::atomic<bool> active{false};
};

/// @brief Progress pool manager
///
/// Manages a pre-allocated pool of progress slots to avoid allocation
/// during computation.
class ProgressPool {
public:
    /// @brief Get the global progress pool instance
    static ProgressPool& instance();
    
    /// @brief Acquire a progress slot from the pool
    ///
    /// @return Pointer to progress slot, or nullptr if pool is exhausted
    ProgressSlot* acquire();
    
    /// @brief Release a progress slot back to the pool
    ///
    /// @param slot Pointer to slot to release
    void release(ProgressSlot* slot);
    
    /// @brief Get current progress value from a slot
    ///
    /// @param slot Pointer to progress slot
    /// @return Current progress value (0-100)
    ProgressValue get_value(ProgressSlot* slot) const;
    
    /// @brief Check if a slot is active
    ///
    /// @param slot Pointer to progress slot
    /// @return True if slot is active
    bool is_active(ProgressSlot* slot) const;

private:
    static constexpr size_t POOL_SIZE = 64;
    ProgressSlot slots_[POOL_SIZE];
    std::atomic<size_t> next_index_{0};
    
    ProgressPool() = default;
    ~ProgressPool() = default;
    ProgressPool(const ProgressPool&) = delete;
    ProgressPool& operator=(const ProgressPool&) = delete;
};

} // namespace progress
} // namespace scl

// =============================================================================
// Convenience Macro
// =============================================================================

/// @brief Get a progress pointer for the current scope
///
/// This macro acquires a progress slot and declares a local variable
/// `progress_ptr` that can be used to update progress.
///
/// @note The slot is automatically released when the scope exits.
#define SCL_GET_PROGRESS(progress_ptr) \
    auto* progress_ptr = scl::progress::ProgressPool::instance().acquire(); \
    if (progress_ptr) { \
        progress_ptr->active.store(true, std::memory_order_relaxed); \
        progress_ptr->value.store(0, std::memory_order_relaxed); \
    }

