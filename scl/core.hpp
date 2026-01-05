#pragma once

/**
 * @file scl/core.hpp
 * @brief SCL Core Module - Unified Header
 *
 * This header provides a convenient single-include interface to all SCL core
 * modules. Include this header to access all fundamental SCL functionality.
 *
 * Included modules:
 *   - Configuration and platform detection (config.hpp)
 *   - Source location utilities (source_location.hpp)
 *   - Compiler abstractions and macros (macro.hpp)
 *   - SIMD wrapper (Highway integration) (simd.hpp)
 *   - Error handling and exceptions (error.hpp)
 *   - Bit manipulation utilities (bits.hpp)
 *   - Type system and precision configuration (type.hpp)
 *   - Memory operations and allocation (memory.hpp)
 *   - Platform IO types (io.hpp)
 *
 * @note Headers are included in dependency order to ensure proper compilation.
 *
 * Usage:
 * @code
 * #include "scl/core.hpp"
 *
 * void example() {
 *   using namespace scl;
 *   
 *   // Type system
 *   Real x = 3.14;
 *   Index idx = 0;
 *   
 *   // Memory operations
 *   scl::memory::AlignedBuffer<float> buffer(1024);
 *   scl::memory::fill(buffer.span(), 0.0f);
 *   
 *   // Error handling
 *   scl::error::check_arg(idx >= 0, "Index must be non-negative");
 *   
 *   // Bit operations
 *   auto count = scl::bits::popcount(0b1010);
 * }
 * @endcode
 */

// =============================================================================
// Core Module Includes (in dependency order)
// =============================================================================

// 1. Configuration (no dependencies)
#include "scl/config.hpp"

// 2. Source location (no dependencies)
#include "scl/core/source_location.hpp"

// 3. Macros and compiler abstractions (depends on: config, source_location)
#include "scl/core/macro.hpp"

// 4. SIMD wrapper (depends on: config)
#include "scl/core/simd.hpp"

// 5. Error handling (depends on: source_location)
#include "scl/core/error.hpp"

// 6. Bit manipulation (depends on: config)
#include "scl/core/bits.hpp"

// 7. Type system (depends on: config)
#include "scl/core/type.hpp"

// 8. Memory operations (depends on: config, error, simd, macro)
#include "scl/core/memory.hpp"

// 9. IO types (depends on: config)
#include "scl/core/io.hpp"

