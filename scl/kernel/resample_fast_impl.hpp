#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

// =============================================================================
/// @file resample_fast_impl.hpp
/// @brief Fast Path for Resampling
///
/// Note: Resampling is dominated by RNG calls, not memory access.
/// The bottleneck is random number generation, not data access patterns.
///
/// Potential optimizations:
/// - SIMD RNG (e.g., vectorized Xoshiro)
/// - Batch binomial sampling
///
/// However, these are complex and provide <20% improvement.
/// Generic path is sufficient.
// =============================================================================

namespace scl::kernel::resample::fast {

// Resampling fast path would require:
// 1. Vectorized RNG (complex)
// 2. SIMD binomial sampling (complex)
// 
// The generic path is already fast enough for this operation.
// Focus optimization efforts on memory-bound operations instead.

} // namespace scl::kernel::resample::fast

