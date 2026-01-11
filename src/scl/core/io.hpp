#pragma once

/**
 * @file scl/core/io.hpp
 * @brief Platform-specific IO types and includes
 *
 * Provides:
 * - Platform-specific headers for memory operations
 * - Memory-mapped file handle types
 *
 * @note This file depends only on scl/config.hpp
 */

#include "scl/config.hpp"

// =============================================================================
// SECTION 1: Platform-Specific Includes for Memory Operations
// =============================================================================

#if SCL_CONFIG_PLATFORM_WINDOWS
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  #include <malloc.h>
#elif SCL_CONFIG_PLATFORM_LINUX || SCL_CONFIG_PLATFORM_MACOS
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

// =============================================================================
// SECTION 2: Memory-Mapped File Types
// =============================================================================

namespace scl::io {

#if SCL_CONFIG_PLATFORM_WINDOWS
using FileHandle = HANDLE;
using MapHandle = HANDLE;
inline constexpr FileHandle INVALID_FILE_HANDLE = INVALID_HANDLE_VALUE;
inline constexpr MapHandle INVALID_MAP_HANDLE = nullptr;
#elif SCL_CONFIG_PLATFORM_LINUX || SCL_CONFIG_PLATFORM_MACOS
using FileHandle = int;
using MapHandle = void*;
inline constexpr FileHandle INVALID_FILE_HANDLE = -1;
inline constexpr MapHandle INVALID_MAP_HANDLE = nullptr;
#else
// Fallback for unknown platforms
using FileHandle = int;
using MapHandle = void*;
inline constexpr FileHandle INVALID_FILE_HANDLE = -1;
inline constexpr MapHandle INVALID_MAP_HANDLE = nullptr;
#endif

}  // namespace scl::io

