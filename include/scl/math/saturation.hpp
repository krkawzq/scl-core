#pragma once

#include "scl/common.hpp"
#include <cstddef>

// =============================================================================
/// @file saturation.hpp
/// @brief Saturation Operations
///
/// This header provides saturation/clamping operations commonly used in
/// biological modeling and signal processing.
///
/// @section Overview
///
/// Saturation operations ensure values remain within valid ranges, which is
/// critical for biological simulations where concentrations, probabilities, or
/// other quantities must stay within physical bounds.
///
/// =============================================================================

namespace scl {
namespace math {

/// @brief Apply saturation (clamping) to an array
///
/// Clamps values to range [min_val, max_val]: $y = \max(\min(x, max), min)$
///
/// @param input Input array
/// @param output Output array
/// @param size Array size
/// @param min_val Minimum value
/// @param max_val Maximum value
///
/// @note Input and output buffers must be pre-allocated
/// @note Supports in-place operation (input == output)
///
/// [Owner: AI]
void saturation(
    const Float* input,
    Float* output,
    Size size,
    Float min_val,
    Float max_val
);

/// @brief Apply smooth saturation (tanh-based)
///
/// Applies smooth saturation using hyperbolic tangent:
/// $y = \frac{max - min}{2} \tanh\left(\frac{2(x - center)}{max - min}\right) + center$
///
/// @param input Input array
/// @param output Output array
/// @param size Array size
/// @param min_val Minimum value
/// @param max_val Maximum value
/// @param center Center value for the tanh function
///
/// @note Provides smooth transitions instead of hard clipping
///
/// [Owner: AI]
void smooth_saturation(
    const Float* input,
    Float* output,
    Size size,
    Float min_val,
    Float max_val,
    Float center
);

} // namespace math
} // namespace scl

