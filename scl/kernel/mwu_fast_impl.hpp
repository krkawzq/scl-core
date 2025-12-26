#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/sparse.hpp"

// =============================================================================
/// @file mwu_fast_impl.hpp
/// @brief Fast Path for Mann-Whitney U Test
///
/// Note: MWU is dominated by sorting (O(n log n)), not memory access.
/// The generic implementation with VQSort is already near-optimal.
/// 
/// Potential optimizations:
/// - Batch sorting multiple features simultaneously
/// - SIMD-optimized rank computation
///
/// However, these provide <10% improvement, so generic path is sufficient.
// =============================================================================

namespace scl::kernel::mwu::fast {

// MWU fast path would require algorithmic changes (batch sorting)
// rather than just memory access optimization.
// The generic path with VQSort is already very fast.

} // namespace scl::kernel::mwu::fast

