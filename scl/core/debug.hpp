#pragma once

/// @file scl/core/debug.hpp
/// @brief Debug utilities and compile-time type inspection
///
/// This header provides:
/// - Compile-time type name extraction with boundary checks
/// - Debug print macros (only active in debug builds)
/// - Compile-time assertions for debugging
/// - Type inspection utilities
/// - Platform-aware debug breakpoints
///
/// @note All debug utilities are intended for development and debugging only.
///       In release builds (NDEBUG defined), most debug utilities become no-ops.
/// @note This header is safe to include in any translation unit.

#include "scl/core/macro.hpp"

#include <cstdio>
#include <cstddef>
#include <type_traits>
#include <string_view>
#include <string>

// Verify compiler detection macros are defined
#if !defined(SCL_COMPILER_CLANG) && !defined(SCL_COMPILER_GCC) && !defined(SCL_COMPILER_MSVC)
    #warning "SCL compiler detection macros not defined. Type names will return 'unknown'."
#endif

// =============================================================================
// Debug Namespace - Compile-Time Type Inspection
// =============================================================================

namespace scl::debug {

/// @brief Extract compile-time type name using compiler intrinsics
/// @tparam T Type to extract name from
/// @return String view containing the type name
/// @note This is a compile-time function (consteval) and has zero runtime cost
/// @note The returned string_view points to compiler-generated string literals
///       which have static storage duration, making it safe to use at runtime.
/// @warning Returns "unknown" on unsupported compilers
///
/// Example:
/// @code{.cpp}
///   constexpr auto name = scl::debug::type_name<int>();
///   static_assert(name == "int");
/// @endcode
template<typename T>
[[nodiscard]]
consteval
auto type_name() -> std::string_view {
#if SCL_COMPILER_CLANG
    constexpr std::string_view prefix = "[T = ";
    constexpr std::string_view suffix = "]";
    constexpr std::string_view function = __PRETTY_FUNCTION__;
#elif SCL_COMPILER_GCC
    constexpr std::string_view prefix = "with T = ";
    constexpr std::string_view suffix = "]";
    constexpr std::string_view function = __PRETTY_FUNCTION__;
#elif SCL_COMPILER_MSVC
    constexpr std::string_view prefix = "type_name<";
    constexpr std::string_view suffix = ">(void)";
    constexpr std::string_view function = __FUNCSIG__;
#else
    return "unknown";
#endif

    // Boundary check: ensure prefix and suffix are found
    constexpr auto prefix_pos = function.find(prefix);
    constexpr auto suffix_pos = function.rfind(suffix);

    static_assert(prefix_pos != std::string_view::npos,
                  "Failed to parse type name: prefix not found in function signature");
    static_assert(suffix_pos != std::string_view::npos,
                  "Failed to parse type name: suffix not found in function signature");
    static_assert(suffix_pos > prefix_pos + prefix.size(),
                  "Failed to parse type name: invalid prefix/suffix positions");

    constexpr auto start = prefix_pos + prefix.size();
    constexpr auto length = suffix_pos - start;

    return function.substr(start, length);
}

}  // namespace scl::debug

// =============================================================================
// Debug Print Macros (Only Active in Debug Builds)
// =============================================================================

#ifdef NDEBUG
    /// @brief Debug print macro (disabled in release builds)
    #define SCL_DEBUG_PRINT(...) ((void)0)

    /// @brief Debug print with location (disabled in release builds)
    #define SCL_DEBUG_PRINT_LOC(...) ((void)0)

    /// @brief Debug print variable name and value (disabled in release builds)
    #define SCL_DEBUG_VAR(var) ((void)0)

#else

namespace scl::debug::detail {

// Helper function templates for type-safe printing
template<typename T>
auto print_var_value(const T& var) -> void {
    if constexpr (std::is_same_v<T, bool>) {
        std::fprintf(stderr, "%s\n", var ? "true" : "false");
    } else if constexpr (std::is_pointer_v<T>) {
        std::fprintf(stderr, "%p\n", static_cast<const void*>(var));
    } else if constexpr (std::is_floating_point_v<T>) {
        std::fprintf(stderr, "%g\n", static_cast<double>(var));
    } else if constexpr (std::is_signed_v<T> && std::is_integral_v<T>) {
        std::fprintf(stderr, "%lld\n", static_cast<long long>(var));
    } else if constexpr (std::is_unsigned_v<T> && std::is_integral_v<T>) {
        std::fprintf(stderr, "%llu\n", static_cast<unsigned long long>(var));
    } else {
        std::fprintf(stderr, "<complex type>\n");
    }
}

}  // namespace scl::debug::detail

    /// @brief Debug print macro (only active in debug builds)
    /// @param ... Format string and arguments (printf-style)
    ///
    /// Example:
    /// @code{.cpp}
    ///   SCL_DEBUG_PRINT("Value: %d", x);
    ///   // Output: [SCL DEBUG] Value: 42
    /// @endcode
    #define SCL_DEBUG_PRINT(...)                                               \
        do {                                                                   \
            std::fprintf(stderr, "[SCL DEBUG] ");                              \
            std::fprintf(stderr, __VA_ARGS__);                                 \
            std::fprintf(stderr, "\n");                                        \
        } while (0)

    /// @brief Debug print with source location (only active in debug builds)
    /// @param ... Format string and arguments (printf-style)
    ///
    /// Example:
    /// @code{.cpp}
    ///   SCL_DEBUG_PRINT_LOC("Checkpoint reached");
    ///   // Output: [SCL DEBUG] file.cpp:42: Checkpoint reached
    /// @endcode
    #define SCL_DEBUG_PRINT_LOC(...)                                           \
        do {                                                                   \
            std::fprintf(stderr, "[SCL DEBUG] %s:%d: ", __FILE__, __LINE__);   \
            std::fprintf(stderr, __VA_ARGS__);                                 \
            std::fprintf(stderr, "\n");                                        \
        } while (0)

    /// @brief Debug print variable name and value (only active in debug builds)
    /// @param var Variable to print (supports integral, floating-point, and pointer types)
    ///
    /// Example:
    /// @code{.cpp}
    ///   int x = 42;
    ///   SCL_DEBUG_VAR(x);
    ///   // Output: [SCL DEBUG] x = 42
    ///
    ///   double pi = 3.14159;
    ///   SCL_DEBUG_VAR(pi);
    ///   // Output: [SCL DEBUG] pi = 3.14159
    /// @endcode
    ///
    /// @note Uses helper template function for type-safe printing
    /// @note For custom types, use SCL_DEBUG_PRINT instead
    #define SCL_DEBUG_VAR(var)                                                 \
        do {                                                                   \
            std::fprintf(stderr, "[SCL DEBUG] %s = ", #var);                   \
            ::scl::debug::detail::print_var_value(var);                        \
        } while (0)

#endif

// =============================================================================
// Debug Markers (Active in All Builds)
// =============================================================================

// Helper macro to generate unique identifier names
#define SCL_DEBUG_CONCAT_IMPL(a, b) a##b
#define SCL_DEBUG_CONCAT(a, b) SCL_DEBUG_CONCAT_IMPL(a, b)
#define SCL_DEBUG_UNIQUE(prefix) SCL_DEBUG_CONCAT(prefix, __LINE__)

/// @brief Mark code section as TODO with a descriptive message
/// @param msg Message describing what needs to be done
///
/// Example:
/// @code{.cpp}
///   void optimize_kernel() {
///       SCL_DEBUG_TODO("Implement SIMD version for AVX-512");
///       // Current scalar implementation...
///   }
/// @endcode
///
/// @note This generates a constexpr variable that can be inspected by tools
/// @note Multiple TODO markers in the same scope are supported (uses __LINE__)
#define SCL_DEBUG_TODO(msg)                                                    \
    [[maybe_unused]] static constexpr const char*                              \
        SCL_DEBUG_UNIQUE(scl_todo_msg_) = "TODO: " msg

/// @brief Mark code section as FIXME with a description of the issue
/// @param msg Message describing what needs to be fixed
///
/// Example:
/// @code{.cpp}
///   void process_data() {
///       SCL_DEBUG_FIXME("Handle edge case for zero-length input");
///       // Temporary implementation without proper edge case handling
///   }
/// @endcode
#define SCL_DEBUG_FIXME(msg)                                                   \
    [[maybe_unused]] static constexpr const char*                              \
        SCL_DEBUG_UNIQUE(scl_fixme_msg_) = "FIXME: " msg

/// @brief Mark code section as HACK with explanation of the workaround
/// @param msg Message describing the temporary workaround
///
/// Example:
/// @code{.cpp}
///   void workaround_compiler_bug() {
///       SCL_DEBUG_HACK("Temporary workaround for GCC 11 constexpr bug");
///       // Hacky implementation...
///   }
/// @endcode
#define SCL_DEBUG_HACK(msg)                                                    \
    [[maybe_unused]] static constexpr const char*                              \
        SCL_DEBUG_UNIQUE(scl_hack_msg_) = "HACK: " msg

// =============================================================================
// Compile-Time Debug Assertions
// =============================================================================

/// @brief Static assertion that always fails (for compile-time debugging)
/// @param T Type parameter to trigger instantiation
/// @param msg Error message
///
/// Example:
///   template<typename T>
///   void foo() {
///       if constexpr (std::is_same_v<T, void>) {
///           SCL_DEBUG_STATIC_FAIL(T, "void type not supported");
///       }
///   }

namespace scl::debug {

template<typename T>
struct always_false : std::false_type {};

}  // namespace scl::debug

#define SCL_DEBUG_STATIC_FAIL(T, msg) \
    static_assert(::scl::debug::always_false<T>::value, msg)

// =============================================================================
// Debug-Only Type Checking
// =============================================================================

namespace scl::debug {

/// @brief Check if type satisfies concept at compile time (for debugging)
/// @tparam T Type to check
/// @return true if type is arithmetic, false otherwise
template<typename T>
[[nodiscard]] consteval 
auto is_arithmetic_type() -> bool {
    return std::is_arithmetic_v<T>;
}

/// @brief Check if type is floating point at compile time
/// @tparam T Type to check
/// @return true if type is floating point, false otherwise
template<typename T>
[[nodiscard]] consteval 
auto is_floating_point_type() -> bool {
    return std::is_floating_point_v<T>;
}

/// @brief Check if type is integral at compile time
/// @tparam T Type to check
/// @return true if type is integral, false otherwise
template<typename T>
[[nodiscard]] consteval 
auto is_integral_type() -> bool {
    return std::is_integral_v<T>;
}

/// @brief Get size of type at compile time
/// @tparam T Type to get size of
/// @return Size of type in bytes
template<typename T>
[[nodiscard]] consteval 
auto type_size() -> std::size_t {
    return sizeof(T);
}

/// @brief Get alignment of type at compile time
/// @tparam T Type to get alignment of
/// @return Alignment of type in bytes
template<typename T>
[[nodiscard]] consteval
auto type_alignment() -> std::size_t {
    return alignof(T);
}

}  // namespace scl::debug

// =============================================================================
// Debug Breakpoint Hints
// =============================================================================

/// @brief Insert debug breakpoint instruction (triggers debugger if attached)
/// @note Only use in debug builds when running under a debugger
/// @note On x86/x64: int3 instruction
/// @note On ARM64: brk instruction
/// @note Falls back to trap on unsupported architectures
/// @warning Program will crash if no debugger is attached on some platforms
///
/// Example:
/// @code{.cpp}
///   if (unexpected_condition) {
///       SCL_DEBUG_PRINT_LOC("Unexpected state reached");
///       SCL_DEBUG_BREAK();  // Pause execution for inspection
///   }
/// @endcode
#if SCL_COMPILER_MSVC
    // MSVC has dedicated debugger intrinsic
    #define SCL_DEBUG_BREAK() __debugbreak()

#elif SCL_COMPILER_GCC_LIKE
    // GCC/Clang: Use inline assembly for real breakpoints
    #if defined(__i386__) || defined(__x86_64__)
        // x86/x64: int3 instruction
        #define SCL_DEBUG_BREAK() __asm__ __volatile__("int3")
    #elif defined(__aarch64__) || defined(__arm64__)
        // ARM64: brk #0 instruction
        #define SCL_DEBUG_BREAK() __asm__ __volatile__(".inst 0xd4200000")
    #elif defined(__arm__)
        // ARM32: bkpt instruction
        #define SCL_DEBUG_BREAK() __asm__ __volatile__(".inst 0xe7f001f0")
    #else
        // Fallback: trap (terminates instead of breaking)
        #define SCL_DEBUG_BREAK() __builtin_trap()
    #endif

#else
    // Unknown compiler: no-op (safe default)
    #define SCL_DEBUG_BREAK() ((void)0)
#endif

// =============================================================================
// Compile-Time Type Name Validation Tests
// =============================================================================

#if defined(SCL_DEBUG_ENABLE_TYPE_TESTS) && (SCL_COMPILER_CLANG || SCL_COMPILER_GCC || SCL_COMPILER_MSVC)
namespace scl::debug::tests {

    // Verify type name extraction works correctly
    static_assert(type_name<int>().find("int") != std::string_view::npos,
                  "type_name<int> should contain 'int'");
    static_assert(type_name<double>().find("double") != std::string_view::npos,
                  "type_name<double> should contain 'double'");
    static_assert(type_name<void*>().size() > 0,
                  "type_name<void*> should not be empty");

    // Verify type inspection utilities
    static_assert(is_arithmetic_type<int>(), "int should be arithmetic");
    static_assert(is_integral_type<int>(), "int should be integral");
    static_assert(!is_floating_point_type<int>(), "int should not be floating point");
    static_assert(is_floating_point_type<double>(), "double should be floating point");

    // Verify size/alignment queries
    static_assert(type_size<int>() == sizeof(int), "type_size should match sizeof");
    static_assert(type_alignment<int>() == alignof(int), "type_alignment should match alignof");

}  // namespace scl::debug::tests
#endif

// =============================================================================
// Additional Debug Utilities
// =============================================================================

namespace scl::debug {

/// @brief Check if type is trivially copyable at compile time
/// @tparam T Type to check
/// @return true if type is trivially copyable, false otherwise
template<typename T>
[[nodiscard]]
consteval
auto is_trivially_copyable_type() -> bool {
    return std::is_trivially_copyable_v<T>;
}

/// @brief Check if type is standard layout at compile time
/// @tparam T Type to check
/// @return true if type is standard layout, false otherwise
template<typename T>
[[nodiscard]]
consteval
auto is_standard_layout_type() -> bool {
    return std::is_standard_layout_v<T>;
}

/// @brief Check if type is POD (Plain Old Data) at compile time
/// @tparam T Type to check
/// @return true if type is POD, false otherwise
/// @note In C++20, prefer is_trivial && is_standard_layout
template<typename T>
[[nodiscard]]
consteval
auto is_pod_type() -> bool {
    return std::is_trivial_v<T> && std::is_standard_layout_v<T>;
}

}  // namespace scl::debug

// =============================================================================
// Debug Performance Profiling (Debug Builds Only)
// =============================================================================

#ifndef NDEBUG

#include <chrono>

namespace scl::debug {

/// @brief Simple RAII timer for performance profiling in debug builds
/// @note Automatically prints elapsed time on destruction
/// @note Zero overhead in release builds (entire class is conditionally compiled)
///
/// Example:
/// @code{.cpp}
///   void expensive_function() {
///       scl::debug::ScopedTimer timer("expensive_function");
///       // ... computation ...
///   }  // Prints: [SCL TIMER] expensive_function: 123.456 ms
/// @endcode
class ScopedTimer {
public:
    explicit ScopedTimer(const char* name)
        : name_(name)
        , start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        double ms = duration.count() / 1000.0;
        std::fprintf(stderr, "[SCL TIMER] %s: %.3f ms\n", name_, ms);
    }

    // Non-copyable, non-movable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

private:
    const char* name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

}  // namespace scl::debug

/// @brief Measure execution time of a code block (debug builds only)
/// @param name Name/description of the timed block
///
/// Example:
/// @code{.cpp}
///   void process() {
///       SCL_DEBUG_TIMER("matrix multiplication");
///       // ... expensive computation ...
///   }  // Prints timing on scope exit
/// @endcode
#define SCL_DEBUG_TIMER(name) \
    ::scl::debug::ScopedTimer SCL_DEBUG_UNIQUE(scl_timer_)(name)

#else
    // Release build: no-op
    #define SCL_DEBUG_TIMER(name) ((void)0)
#endif

// =============================================================================
// Debug Memory Inspection
// =============================================================================

#ifndef NDEBUG

namespace scl::debug {

/// @brief Dump memory contents in hexadecimal format (debug builds only)
/// @param ptr Pointer to memory region
/// @param size Number of bytes to dump
/// @param label Optional label for the dump
///
/// Example output:
/// @code
/// [SCL MEMDUMP] buffer (16 bytes):
/// 0x00: 48 65 6c 6c 6f 20 57 6f 72 6c 64 21 00 00 00 00  Hello World!....
/// @endcode
inline
auto dump_memory(const void* ptr, std::size_t size, const char* label = "memory") -> void {
    std::fprintf(stderr, "[SCL MEMDUMP] %s (%zu bytes):\n", label, size);

    const auto* bytes = static_cast<const unsigned char*>(ptr);
    constexpr std::size_t bytes_per_line = 16;

    for (std::size_t i = 0; i < size; i += bytes_per_line) {
        // Print offset
        std::fprintf(stderr, "0x%04zx: ", i);

        // Print hex values
        for (std::size_t j = 0; j < bytes_per_line; ++j) {
            if (i + j < size) {
                std::fprintf(stderr, "%02x ", bytes[i + j]);
            } else {
                std::fprintf(stderr, "   ");
            }
        }

        std::fprintf(stderr, " ");

        // Print ASCII representation
        for (std::size_t j = 0; j < bytes_per_line && i + j < size; ++j) {
            unsigned char c = bytes[i + j];
            std::fprintf(stderr, "%c", (c >= 32 && c <= 126) ? c : '.');
        }

        std::fprintf(stderr, "\n");
    }
}

/// @brief Print array contents (debug builds only)
/// @tparam T Element type (must be printable)
/// @param data Pointer to array
/// @param size Number of elements
/// @param label Optional label for the array
template<typename T>
auto print_array(const T* data, std::size_t size, const char* label = "array") -> void {
    std::fprintf(stderr, "[SCL DEBUG] %s[%zu] = { ", label, size);

    constexpr std::size_t max_display = 20;
    const std::size_t display_count = (size <= max_display) ? size : 10;

    for (std::size_t i = 0; i < display_count; ++i) {
        if constexpr (std::is_integral_v<T>) {
            if constexpr (std::is_signed_v<T>) {
                std::fprintf(stderr, "%lld", (long long)data[i]);
            } else {
                std::fprintf(stderr, "%llu", (unsigned long long)data[i]);
            }
        } else if constexpr (std::is_floating_point_v<T>) {
            std::fprintf(stderr, "%g", (double)data[i]);
        } else if constexpr (std::is_pointer_v<T>) {
            std::fprintf(stderr, "%p", (void*)data[i]);
        }

        if (i < display_count - 1) {
            std::fprintf(stderr, ", ");
        }
    }

    if (size > max_display) {
        std::fprintf(stderr, ", ... (%zu more elements) ..., ", size - 20);
        for (std::size_t i = size - 10; i < size; ++i) {
            if constexpr (std::is_integral_v<T>) {
                if constexpr (std::is_signed_v<T>) {
                    std::fprintf(stderr, "%lld", (long long)data[i]);
                } else {
                    std::fprintf(stderr, "%llu", (unsigned long long)data[i]);
                }
            } else if constexpr (std::is_floating_point_v<T>) {
                std::fprintf(stderr, "%g", (double)data[i]);
            }
            if (i < size - 1) {
                std::fprintf(stderr, ", ");
            }
        }
    }

    std::fprintf(stderr, " }\n");
}

}  // namespace scl::debug

/// @brief Dump memory region in hex format (debug builds only)
/// @param ptr Pointer to memory
/// @param size Number of bytes
///
/// Example:
/// @code{.cpp}
///   uint8_t buffer[16] = {...};
///   SCL_DEBUG_DUMP_MEMORY(buffer, sizeof(buffer));
/// @endcode
#define SCL_DEBUG_DUMP_MEMORY(ptr, size) \
    ::scl::debug::dump_memory(ptr, size, #ptr)

/// @brief Print array contents (debug builds only)
/// @param arr Array variable name
/// @param size Number of elements
///
/// Example:
/// @code{.cpp}
///   int values[100] = {...};
///   SCL_DEBUG_PRINT_ARRAY(values, 100);
/// @endcode
#define SCL_DEBUG_PRINT_ARRAY(arr, size) \
    ::scl::debug::print_array(arr, size, #arr)

#else
    // Release builds: no-op
    #define SCL_DEBUG_DUMP_MEMORY(ptr, size) ((void)0)
    #define SCL_DEBUG_PRINT_ARRAY(arr, size) ((void)0)
#endif

// =============================================================================
// Debug Assertions with Custom Messages
// =============================================================================

#ifndef NDEBUG

/// @brief Debug-only assertion with custom message and expression printing
/// @param expr Boolean expression to check
/// @param msg Custom error message
///
/// Example:
/// @code{.cpp}
///   SCL_DEBUG_ASSERT_MSG(size > 0, "Size must be positive");
///   // If fails: [SCL ASSERT] file.cpp:42: size > 0 failed: Size must be positive
/// @endcode
///
/// @note Terminates program on failure in debug builds
/// @note Completely removed in release builds
#define SCL_DEBUG_ASSERT_MSG(expr, msg)                                        \
    do {                                                                       \
        if (!(expr)) {                                                         \
            std::fprintf(stderr, "[SCL ASSERT] %s:%d: %s failed: %s\n",        \
                        __FILE__, __LINE__, #expr, msg);                       \
            SCL_DEBUG_BREAK();                                                 \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

/// @brief Debug-only assertion with expression printing
/// @param expr Boolean expression to check
///
/// Example:
/// @code{.cpp}
///   SCL_DEBUG_ASSERT(index < size);
/// @endcode
#define SCL_DEBUG_ASSERT(expr)                                                 \
    SCL_DEBUG_ASSERT_MSG(expr, "assertion failed")

/// @brief Unconditional failure in debug builds (unreachable code marker)
/// @param msg Message describing why this code should be unreachable
///
/// Example:
/// @code{.cpp}
///   switch (type) {
///       case A: return handle_a();
///       case B: return handle_b();
///       default: SCL_DEBUG_UNREACHABLE("Unknown type");
///   }
/// @endcode
#define SCL_DEBUG_UNREACHABLE(msg)                                             \
    do {                                                                       \
        std::fprintf(stderr, "[SCL UNREACHABLE] %s:%d: %s\n",                  \
                    __FILE__, __LINE__, msg);                                  \
        SCL_DEBUG_BREAK();                                                     \
        std::abort();                                                          \
    } while (0)

#else
    // Release builds: minimal overhead
    #define SCL_DEBUG_ASSERT_MSG(expr, msg) ((void)0)
    #define SCL_DEBUG_ASSERT(expr) ((void)0)

    #if SCL_COMPILER_GCC_LIKE
        #define SCL_DEBUG_UNREACHABLE(msg) __builtin_unreachable()
    #elif SCL_COMPILER_MSVC
        #define SCL_DEBUG_UNREACHABLE(msg) __assume(0)
    #else
        #define SCL_DEBUG_UNREACHABLE(msg) ((void)0)
    #endif
#endif

