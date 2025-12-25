#include "scl/progress.hpp"

namespace scl {
namespace progress {

// =============================================================================
// ProgressPool Implementation
// =============================================================================

ProgressPool& ProgressPool::instance() {
    static ProgressPool pool;
    return pool;
}

ProgressSlot* ProgressPool::acquire() {
    // Simple round-robin allocation (lock-free for common case)
    size_t start_index = next_index_.load(std::memory_order_relaxed);
    
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        size_t index = (start_index + i) % POOL_SIZE;
        ProgressSlot& slot = slots_[index];
        
        // Try to acquire slot (lock-free CAS)
        bool expected = false;
        if (slot.active.compare_exchange_weak(
                expected, true,
                std::memory_order_acquire,
                std::memory_order_relaxed)) {
            // Successfully acquired slot
            slot.value.store(0, std::memory_order_relaxed);
            next_index_.store((index + 1) % POOL_SIZE, std::memory_order_relaxed);
            return &slot;
        }
    }
    
    // Pool exhausted - return nullptr
    // This is acceptable: progress tracking is optional
    return nullptr;
}

void ProgressPool::release(ProgressSlot* slot) {
    if (slot == nullptr) {
        return;
    }
    
    // Validate pointer is within our pool
    const ProgressSlot* pool_start = slots_;
    const ProgressSlot* pool_end = slots_ + POOL_SIZE;
    
    if (slot < pool_start || slot >= pool_end) {
        return;  // Invalid pointer
    }
    
    // Release slot (lock-free)
    slot->active.store(false, std::memory_order_release);
    slot->value.store(0, std::memory_order_relaxed);
}

ProgressValue ProgressPool::get_value(ProgressSlot* slot) const {
    if (slot == nullptr) {
        return 0;
    }
    
    // Lock-free read
    return slot->value.load(std::memory_order_acquire);
}

bool ProgressPool::is_active(ProgressSlot* slot) const {
    if (slot == nullptr) {
        return false;
    }
    
    // Lock-free read
    return slot->active.load(std::memory_order_acquire);
}

} // namespace progress
} // namespace scl

