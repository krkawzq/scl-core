#pragma once

/**
 * @file scl/core/source_location.hpp
 * @brief Source location support (C++20 std::source_location with fallback)
 *
 * Provides a cross-platform source location implementation:
 * - Uses std::source_location when available (C++20)
 * - Falls back to compiler builtins for older compilers
 *
 * @note This file depends only on scl/config.hpp
 */

#include <cstdint>
#include <string_view>

// =============================================================================
// SECTION 1: Feature Detection
// =============================================================================

#if __has_include(<version>)
  #include <version>
#endif

#if __has_include(<source_location>) && \
    defined(__cpp_lib_source_location) && __cpp_lib_source_location >= 201907L
  #include <source_location>
  #define SCL_HAS_SOURCE_LOCATION 1
#else
  #define SCL_HAS_SOURCE_LOCATION 0
#endif

// =============================================================================
// SECTION 2: Source Location Type
// =============================================================================

namespace scl {

#if SCL_HAS_SOURCE_LOCATION

/// @brief Source location information (C++20 std::source_location)
using source_location = std::source_location;

#else

/// @brief Fallback source location for pre-C++20 compilers
/// @note Uses compiler builtins to provide similar functionality
struct source_location {
private:
  const char* file_;
  const char* function_;
  std::uint32_t line_;
  std::uint32_t column_;

public:
  /// @brief Default constructor
  constexpr source_location() noexcept
      : file_(""), function_(""), line_(0), column_(0) {}

  /// @brief Constructor with location information
  constexpr source_location(const char* file, const char* func,
                            std::uint32_t line,
                            std::uint32_t col = 0) noexcept
      : file_(file), function_(func), line_(line), column_(col) {}

  /// @brief Get current source location
  /// @return Source location at call site
  [[nodiscard]] static constexpr auto current(
      const char* file = __builtin_FILE(),
      const char* func = __builtin_FUNCTION(),
      std::uint32_t line = __builtin_LINE()) noexcept -> source_location {
    return source_location{file, func, line, 0};
  }

  /// @brief Get file name
  [[nodiscard]] constexpr auto file_name() const noexcept -> const char* {
    return file_;
  }

  /// @brief Get function name
  [[nodiscard]] constexpr auto function_name() const noexcept -> const char* {
    return function_;
  }

  /// @brief Get line number
  [[nodiscard]] constexpr auto line() const noexcept -> std::uint32_t {
    return line_;
  }

  /// @brief Get column number
  [[nodiscard]] constexpr auto column() const noexcept -> std::uint32_t {
    return column_;
  }
};

#endif

// =============================================================================
// SECTION 3: Utility Functions
// =============================================================================

/// @brief Extract filename from full path
/// @param[in] path Full file path
/// @return Pointer to filename portion
/// @note Constexpr utility for stripping directory path
[[nodiscard]]
constexpr
auto filename_only(const char* path) noexcept -> const char* {
  std::string_view path_view(path);
  if (auto pos = path_view.find_last_of("/\\");
      pos != std::string_view::npos) {
    return path_view.substr(pos + 1).data();
  }
  return path;
}

}  // namespace scl

