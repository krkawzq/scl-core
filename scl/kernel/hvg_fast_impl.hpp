#pragma once

#include "scl/core/type.hpp"
#include "scl/core/argsort.hpp"

// =============================================================================
/// @file hvg_fast_impl.hpp
/// @brief Fast Path for HVG Selection
///
/// Note: HVG is dominated by sorting (O(n log n)), not memory access.
/// The generic implementation with VQSort-based argsort is already optimal.
///
/// HVG wraps feature statistics + sorting, both of which have
/// their own fast paths (feature_fast_impl.hpp + VQSort).
// =============================================================================

namespace scl::kernel::hvg::fast {

// HVG fast path is achieved through:
// 1. feature_fast_impl.hpp for statistics computation
// 2. VQSort for sorting (already optimal)
// No additional fast path needed.

} // namespace scl::kernel::hvg::fast

