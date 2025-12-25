#include "scl/math/diffusion.hpp"
#include "scl/math/saturation.hpp"
#include "scl/error.hpp"
#include "scl/common.hpp"
#include <cstring>
#include <exception>
#include <stdexcept>

// =============================================================================
/// @file bind_math.cpp
/// @brief C-ABI Bindings for Math Operations
///
/// This file provides C-compatible wrappers for mathematical operations,
/// following the C-ABI firewall pattern: exceptions are caught and converted
/// to error instances.
///
/// =============================================================================

extern "C" {

// C-ABI compatible type aliases
using Float = scl::Float;
using Size = scl::Size;
using Int = scl::Int;

// =============================================================================
// Diffusion Operations
// =============================================================================

/// @brief Apply 2D diffusion operation (C-ABI wrapper)
///
/// @param input Input grid data (row-major, size: width * height)
/// @param output Output grid data (row-major, size: width * height)
/// @param width Grid width
/// @param height Grid height
/// @param alpha Diffusion coefficient
/// @param boundary_type Boundary condition type
/// @return nullptr on success, error instance on failure
scl_error_t scl_diffusion_2d(
    const Float* input,
    Float* output,
    Size width,
    Size height,
    Float alpha,
    Int boundary_type
) {
    try {
        // Validate inputs
        if (input == nullptr || output == nullptr) {
            return scl::error::ErrorRegistry::register_error(
                scl::error::ErrorCode::INVALID_ARGUMENT,
                "scl_diffusion_2d: input or output pointer is null"
            );
        }
        
        if (width == 0 || height == 0) {
            return scl::error::ErrorRegistry::register_error(
                scl::error::ErrorCode::INVALID_ARGUMENT,
                "scl_diffusion_2d: width or height is zero"
            );
        }
        
        if (alpha < 0.0f || alpha > 1.0f) {
            return scl::error::ErrorRegistry::register_error(
                scl::error::ErrorCode::INVALID_ARGUMENT,
                "scl_diffusion_2d: alpha must be in [0, 1]"
            );
        }
        
        // Call C++ implementation
        scl::math::diffusion_2d(input, output, width, height, alpha, boundary_type);
        
        return nullptr;  // Success
    } catch (const std::exception& e) {
        return scl::error::exception_to_error(e);
    } catch (...) {
        return scl::error::unknown_exception_to_error();
    }
}

/// @brief Apply anisotropic diffusion (C-ABI wrapper)
///
/// @param input Input grid data
/// @param output Output grid data
/// @param width Grid width
/// @param height Grid height
/// @param kappa Edge-stopping parameter
/// @param iterations Number of iterations
/// @return nullptr on success, error instance on failure
scl_error_t scl_anisotropic_diffusion_2d(
    const Float* input,
    Float* output,
    Size width,
    Size height,
    Float kappa,
    Int iterations
) {
    try {
        // Validate inputs
        if (input == nullptr || output == nullptr) {
            return scl::error::ErrorRegistry::register_error(
                scl::error::ErrorCode::INVALID_ARGUMENT,
                "scl_anisotropic_diffusion_2d: input or output pointer is null"
            );
        }
        
        if (width == 0 || height == 0) {
            return scl::error::ErrorRegistry::register_error(
                scl::error::ErrorCode::INVALID_ARGUMENT,
                "scl_anisotropic_diffusion_2d: width or height is zero"
            );
        }
        
        if (iterations < 0) {
            return scl::error::ErrorRegistry::register_error(
                scl::error::ErrorCode::INVALID_ARGUMENT,
                "scl_anisotropic_diffusion_2d: iterations must be non-negative"
            );
        }
        
        // Call C++ implementation
        scl::math::anisotropic_diffusion_2d(input, output, width, height, kappa, iterations);
        
        return nullptr;  // Success
    } catch (const std::exception& e) {
        return scl::error::exception_to_error(e);
    } catch (...) {
        return scl::error::unknown_exception_to_error();
    }
}

// =============================================================================
// Saturation Operations
// =============================================================================

/// @brief Apply saturation (clamping) operation (C-ABI wrapper)
///
/// @param input Input array
/// @param output Output array
/// @param size Array size
/// @param min_val Minimum value
/// @param max_val Maximum value
/// @return nullptr on success, error instance on failure
scl_error_t scl_saturation(
    const Float* input,
    Float* output,
    Size size,
    Float min_val,
    Float max_val
) {
    try {
        // Validate inputs
        if (input == nullptr || output == nullptr) {
            return scl::error::ErrorRegistry::register_error(
                scl::error::ErrorCode::INVALID_ARGUMENT,
                "scl_saturation: input or output pointer is null"
            );
        }
        
        if (size == 0) {
            return scl::error::ErrorRegistry::register_error(
                scl::error::ErrorCode::INVALID_ARGUMENT,
                "scl_saturation: size is zero"
            );
        }
        
        if (min_val > max_val) {
            return scl::error::ErrorRegistry::register_error(
                scl::error::ErrorCode::INVALID_ARGUMENT,
                "scl_saturation: min_val must be <= max_val"
            );
        }
        
        // Call C++ implementation
        scl::math::saturation(input, output, size, min_val, max_val);
        
        return nullptr;  // Success
    } catch (const std::exception& e) {
        return scl::error::exception_to_error(e);
    } catch (...) {
        return scl::error::unknown_exception_to_error();
    }
}

/// @brief Apply smooth saturation (C-ABI wrapper)
///
/// @param input Input array
/// @param output Output array
/// @param size Array size
/// @param min_val Minimum value
/// @param max_val Maximum value
/// @param center Center value for tanh function
/// @return nullptr on success, error instance on failure
scl_error_t scl_smooth_saturation(
    const Float* input,
    Float* output,
    Size size,
    Float min_val,
    Float max_val,
    Float center
) {
    try {
        // Validate inputs
        if (input == nullptr || output == nullptr) {
            return scl::error::ErrorRegistry::register_error(
                scl::error::ErrorCode::INVALID_ARGUMENT,
                "scl_smooth_saturation: input or output pointer is null"
            );
        }
        
        if (size == 0) {
            return scl::error::ErrorRegistry::register_error(
                scl::error::ErrorCode::INVALID_ARGUMENT,
                "scl_smooth_saturation: size is zero"
            );
        }
        
        if (min_val > max_val) {
            return scl::error::ErrorRegistry::register_error(
                scl::error::ErrorCode::INVALID_ARGUMENT,
                "scl_smooth_saturation: min_val must be <= max_val"
            );
        }
        
        // Call C++ implementation
        scl::math::smooth_saturation(input, output, size, min_val, max_val, center);
        
        return nullptr;  // Success
    } catch (const std::exception& e) {
        return scl::error::exception_to_error(e);
    } catch (...) {
        return scl::error::unknown_exception_to_error();
    }
}

} // extern "C"

