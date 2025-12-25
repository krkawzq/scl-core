#pragma once

#include "scl/common.hpp"
#include <cstddef>

// =============================================================================
/// @file diffusion.hpp
/// @brief Diffusion Operations
///
/// This header provides diffusion-related mathematical operations for
/// biological simulations and image processing.
///
/// @section Overview
///
/// Diffusion operations are fundamental in biological modeling, particularly
/// for simulating concentration gradients, heat transfer, and similar phenomena.
///
/// =============================================================================

namespace scl {
namespace math {

/// @brief Apply diffusion operator to a 2D grid
///
/// Performs diffusion operation: $u_{t+1} = u_t + \alpha \nabla^2 u_t$
///
/// @param input Input grid data (row-major, size: width * height)
/// @param output Output grid data (row-major, size: width * height)
/// @param width Grid width
/// @param height Grid height
/// @param alpha Diffusion coefficient (typically 0 < alpha < 0.25 for stability)
/// @param boundary_type Boundary condition type (0=zero, 1=reflective, 2=periodic)
///
/// @note Input and output buffers must be pre-allocated and non-overlapping
/// @note This function uses the unified threading backend
///
/// [Owner: AI]
void diffusion_2d(
    const Float* input,
    Float* output,
    Size width,
    Size height,
    Float alpha,
    Int boundary_type = 0
);

/// @brief Apply anisotropic diffusion to a 2D grid
///
/// Performs edge-preserving anisotropic diffusion using Perona-Malik model.
///
/// @param input Input grid data (row-major)
/// @param output Output grid data (row-major)
/// @param width Grid width
/// @param height Grid height
/// @param kappa Edge-stopping parameter
/// @param iterations Number of diffusion iterations
///
/// @note Input and output buffers must be pre-allocated
///
/// [Owner: AI]
void anisotropic_diffusion_2d(
    const Float* input,
    Float* output,
    Size width,
    Size height,
    Float kappa,
    Int iterations = 10
);

} // namespace math
} // namespace scl

