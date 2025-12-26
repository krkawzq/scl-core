#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

// =============================================================================
/// @file merge_fast_impl.hpp
/// @brief Fast Path for Matrix Merging
///
/// Note: Merge operations are O(rows) index array operations,
/// not data-intensive. The generic implementation is already optimal.
///
/// Merge is fundamentally about building indirection maps,
/// which is already very fast (just integer array copies).
// =============================================================================

namespace scl::kernel::merge::fast {

// Merge fast path not needed:
// - Operation is O(rows) index manipulation
// - No heavy computation or data movement
// - Generic path is already near-optimal

} // namespace scl::kernel::merge::fast

