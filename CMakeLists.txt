# =============================================================================
# SCL-Core CMake Build Configuration
# =============================================================================
#
# Modern CMake (3.21+) configuration for scl-core library
#
# Build Options:
#   -DSCL_BACKEND=OPENMP|TBB|BS|SERIAL  Threading backend (default: auto)
#   -DSCL_BUILD_SHARED=ON|OFF           Build shared library (default: ON)
#   -DSCL_BUILD_STATIC=ON|OFF           Build static library (default: ON)
#   -DSCL_ENABLE_LTO=ON|OFF             Link-time optimization (default: Release)
#   -DSCL_ENABLE_NATIVE=ON|OFF          Native arch optimization (default: ON)
#   -DSCL_ENABLE_ASAN=ON|OFF            Address sanitizer (default: OFF)
#   -DSCL_ENABLE_UBSAN=ON|OFF           UB sanitizer (default: OFF)
#   -DSCL_ENABLE_TSAN=ON|OFF            Thread sanitizer (default: OFF)
#
# Note: Tests, benchmarks, examples, and Python bindings are built separately
# =============================================================================

cmake_minimum_required(VERSION 3.21)

# -----------------------------------------------------------------------------
# Project Definition
# -----------------------------------------------------------------------------

project(scl-core
    VERSION 0.5.0
    DESCRIPTION "High-performance biological operator library"
    LANGUAGES CXX C
)

# Prevent in-source builds
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are not allowed. Use: cmake -B build")
endif()

# -----------------------------------------------------------------------------
# CMake Configuration
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 正确连接 compile_commands.json
# 若不是构建在顶层目录下，则在源目录生成一个软链接，指向实际的 compile_commands.json
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(_compile_commands_src "${CMAKE_BINARY_DIR}/compile_commands.json")
    set(_compile_commands_dst "${CMAKE_SOURCE_DIR}/compile_commands.json")
    if(EXISTS "${_compile_commands_src}" AND NOT "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
        if(NOT EXISTS "${_compile_commands_dst}")
            # Unix like systems: create symlink
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E create_symlink
                "${_compile_commands_src}" "${_compile_commands_dst}"
            )
            message(STATUS "Created symlink: ${_compile_commands_dst} -> ${_compile_commands_src}")
        endif()
    endif()
endif()

# Default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

message(STATUS "")
message(STATUS "========================================")
message(STATUS "  SCL-Core v${PROJECT_VERSION}")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")

# -----------------------------------------------------------------------------
# Build Options
# -----------------------------------------------------------------------------

# Threading backend selection
set(SCL_BACKEND "AUTO" CACHE STRING "Threading backend: AUTO, OPENMP, TBB, BS, SERIAL")
set_property(CACHE SCL_BACKEND PROPERTY STRINGS AUTO OPENMP TBB BS SERIAL)

# Library build options
option(SCL_BUILD_SHARED "Build shared library with C FFI exports" ON)
option(SCL_BUILD_STATIC "Build static library with C FFI exports" ON)

# Optimization options
option(SCL_ENABLE_LTO "Enable link-time optimization" OFF)
option(SCL_ENABLE_NATIVE "Enable native architecture optimization" ON)

# Sanitizer options
option(SCL_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(SCL_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(SCL_ENABLE_TSAN "Enable ThreadSanitizer" OFF)

# Auto-enable LTO for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT DEFINED CACHE{SCL_ENABLE_LTO})
    set(SCL_ENABLE_LTO ON)
endif()

# -----------------------------------------------------------------------------
# Platform Detection
# -----------------------------------------------------------------------------

if(WIN32)
    set(SCL_PLATFORM "WINDOWS")
elseif(APPLE)
    set(SCL_PLATFORM "MACOS")
elseif(UNIX)
    set(SCL_PLATFORM "LINUX")
else()
    set(SCL_PLATFORM "UNKNOWN")
endif()

message(STATUS "Platform: ${SCL_PLATFORM}")

# -----------------------------------------------------------------------------
# Threading Backend Selection
# -----------------------------------------------------------------------------

set(SCL_BACKEND_FOUND FALSE)

if(SCL_BACKEND STREQUAL "AUTO")
    # Auto-detect based on platform
    if(SCL_PLATFORM STREQUAL "MACOS")
        # macOS: Prefer BS::thread_pool to avoid libomp issues
        set(SCL_BACKEND_SELECTED "BS")
        message(STATUS "Threading: Auto-selected BS::thread_pool (macOS default)")
    else()
        # Linux/Windows: Try OpenMP first
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            set(SCL_BACKEND_SELECTED "OPENMP")
            message(STATUS "Threading: Auto-selected OpenMP")
        else()
            set(SCL_BACKEND_SELECTED "BS")
            message(STATUS "Threading: Auto-selected BS::thread_pool (OpenMP not found)")
        endif()
    endif()
else()
    set(SCL_BACKEND_SELECTED ${SCL_BACKEND})
endif()

# Validate and configure selected backend
if(SCL_BACKEND_SELECTED STREQUAL "OPENMP")
    find_package(OpenMP REQUIRED)
    set(SCL_THREADING_LIBS OpenMP::OpenMP_CXX)
    set(SCL_THREADING_DEFINES SCL_BACKEND_OPENMP=1)
    set(SCL_BACKEND_FOUND TRUE)

elseif(SCL_BACKEND_SELECTED STREQUAL "TBB")
    find_package(TBB REQUIRED)
    set(SCL_THREADING_LIBS TBB::tbb)
    set(SCL_THREADING_DEFINES SCL_BACKEND_TBB=1)
    set(SCL_BACKEND_FOUND TRUE)

elseif(SCL_BACKEND_SELECTED STREQUAL "BS")
    # BS::thread_pool is header-only, included in third_party
    set(SCL_THREADING_LIBS "")
    set(SCL_THREADING_DEFINES SCL_BACKEND_BS=1)
    set(SCL_BACKEND_FOUND TRUE)

elseif(SCL_BACKEND_SELECTED STREQUAL "SERIAL")
    set(SCL_THREADING_LIBS "")
    set(SCL_THREADING_DEFINES SCL_BACKEND_SERIAL=1)
    set(SCL_BACKEND_FOUND TRUE)
endif()

if(NOT SCL_BACKEND_FOUND)
    message(FATAL_ERROR "Invalid threading backend: ${SCL_BACKEND_SELECTED}")
endif()

message(STATUS "Threading backend: ${SCL_BACKEND_SELECTED}")

# -----------------------------------------------------------------------------
# Compiler Configuration
# -----------------------------------------------------------------------------

# Common warning flags
set(SCL_WARNING_FLAGS "")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND SCL_WARNING_FLAGS
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wcast-qual
        -Wunused
        -Wformat=2
        -Wnull-dereference
        -Wdouble-promotion
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND SCL_WARNING_FLAGS
            -Wduplicated-cond
            -Wduplicated-branches
            -Wlogical-op
        )
    endif()

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    list(APPEND SCL_WARNING_FLAGS
        /W4
        /permissive-
        /Zc:__cplusplus
        /Zc:preprocessor
    )
endif()

# Optimization flags
set(SCL_OPTIMIZATION_FLAGS "")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Release optimizations
    if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
        list(APPEND SCL_OPTIMIZATION_FLAGS -O3)

        if(SCL_ENABLE_NATIVE)
            list(APPEND SCL_OPTIMIZATION_FLAGS -march=native)
            message(STATUS "Native architecture optimization: ON")
        endif()

        # Fast math (careful with NaN/Inf handling)
        # list(APPEND SCL_OPTIMIZATION_FLAGS -ffast-math)
    endif()

    # LTO
    if(SCL_ENABLE_LTO)
        list(APPEND SCL_OPTIMIZATION_FLAGS -flto)
        message(STATUS "Link-time optimization: ON")
    endif()

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
        list(APPEND SCL_OPTIMIZATION_FLAGS /O2 /Ob3)

        if(SCL_ENABLE_NATIVE)
            # MSVC auto-detects, but we can hint AVX2
            list(APPEND SCL_OPTIMIZATION_FLAGS /arch:AVX2)
        endif()
    endif()

    if(SCL_ENABLE_LTO)
        list(APPEND SCL_OPTIMIZATION_FLAGS /GL)
    endif()
endif()

# Sanitizer flags
set(SCL_SANITIZER_FLAGS "")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(SCL_ENABLE_ASAN)
        list(APPEND SCL_SANITIZER_FLAGS -fsanitize=address -fno-omit-frame-pointer)
        message(STATUS "AddressSanitizer: ON")
    endif()

    if(SCL_ENABLE_UBSAN)
        list(APPEND SCL_SANITIZER_FLAGS -fsanitize=undefined)
        message(STATUS "UndefinedBehaviorSanitizer: ON")
    endif()

    if(SCL_ENABLE_TSAN)
        list(APPEND SCL_SANITIZER_FLAGS -fsanitize=thread)
        message(STATUS "ThreadSanitizer: ON")
    endif()
endif()

# -----------------------------------------------------------------------------
# Source Files
# -----------------------------------------------------------------------------

# Directory structure:
# - scl/             : Header-only C++ library (core implementation)
# - scl/api/         : C API (FFI) sources for Python/C bindings
#   - scl/api/core/  : Core C-API (types, error, sparse handlers)
# - tests/           : Test files
# - examples/        : Example files
# - third_party/     : Third-party dependencies

set(SCL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SCL_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/scl/api)

# -----------------------------------------------------------------------------
# Source Files (Read from .compile file)
# -----------------------------------------------------------------------------

# Read source file list from .compile (one file per line, relative paths allowed)
set(SCL_COMPILE_LIST_FILE "${CMAKE_CURRENT_SOURCE_DIR}/.compile")
set(SCL_ALL_SOURCES "")

if(EXISTS "${SCL_COMPILE_LIST_FILE}")
    file(STRINGS "${SCL_COMPILE_LIST_FILE}" SCL_COMPILE_LIST)
    foreach(src_file ${SCL_COMPILE_LIST})
        # Skip empty lines and comments (lines starting with #)
        string(STRIP "${src_file}" src_file)
        if(src_file AND NOT src_file MATCHES "^#")
            # Convert relative path to absolute
            if(NOT IS_ABSOLUTE "${src_file}")
                set(src_file "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
            endif()
            # Check if file exists
            if(EXISTS "${src_file}")
                list(APPEND SCL_ALL_SOURCES "${src_file}")
            else()
                message(WARNING "Source file not found: ${src_file}")
            endif()
        endif()
    endforeach()
endif()

# Report source files
list(LENGTH SCL_ALL_SOURCES SCL_SOURCE_COUNT)
message(STATUS "Source files from .compile: ${SCL_SOURCE_COUNT} files")

# -----------------------------------------------------------------------------
# Header Files (Auto-discovered)
# -----------------------------------------------------------------------------

# Find all headers dynamically
file(GLOB_RECURSE SCL_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/scl/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scl/*.h
)

# Report configuration
list(LENGTH SCL_HEADERS SCL_HEADER_COUNT)
message(STATUS "")
message(STATUS "Source Configuration:")
message(STATUS "  Sources (from .compile): ${SCL_SOURCE_COUNT} files")
message(STATUS "  Headers (auto-discovered): ${SCL_HEADER_COUNT} files")
message(STATUS "")

# -----------------------------------------------------------------------------
# Library Targets
# -----------------------------------------------------------------------------

# Interface library for common settings (header-only base)
add_library(scl_common INTERFACE)

target_include_directories(scl_common INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
    $<INSTALL_INTERFACE:include>
)

# Version defaults (fallback to 0.0.0 if not defined)
if(NOT DEFINED PROJECT_VERSION_MAJOR)
    set(PROJECT_VERSION_MAJOR 0)
endif()
if(NOT DEFINED PROJECT_VERSION_MINOR)
    set(PROJECT_VERSION_MINOR 0)
endif()
if(NOT DEFINED PROJECT_VERSION_PATCH)
    set(PROJECT_VERSION_PATCH 0)
endif()

target_compile_definitions(scl_common INTERFACE
    ${SCL_THREADING_DEFINES}
    # Version information
    SCL_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    SCL_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    SCL_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    $<$<CONFIG:Debug>:SCL_DEBUG=1>
    $<$<CONFIG:Release>:NDEBUG>
)

target_compile_options(scl_common INTERFACE
    ${SCL_WARNING_FLAGS}
    ${SCL_OPTIMIZATION_FLAGS}
    ${SCL_SANITIZER_FLAGS}
)

target_link_libraries(scl_common INTERFACE
    ${SCL_THREADING_LIBS}
)

if(SCL_SANITIZER_FLAGS)
    target_link_options(scl_common INTERFACE ${SCL_SANITIZER_FLAGS})
endif()

# Shared library
if(SCL_BUILD_SHARED AND SCL_ALL_SOURCES)
    add_library(scl SHARED ${SCL_ALL_SOURCES})
    target_link_libraries(scl PUBLIC scl_common)

    set_target_properties(scl PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        OUTPUT_NAME "scl"
        EXPORT_NAME "scl"
        # C-API visibility
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )

    # Export symbols
    target_compile_definitions(scl PRIVATE
        SCL_BUILDING_LIBRARY=1
        # Allow unsafe access for internal implementation
        SCL_UNSAFE_ACCESS=1
    )

    if(WIN32)
        target_compile_definitions(scl PUBLIC SCL_DLL=1)
    endif()

    message(STATUS "Building shared library: libscl.so")
elseif(SCL_BUILD_SHARED)
    message(STATUS "Skipping shared library (no source files found)")
endif()

# Static library
if(SCL_BUILD_STATIC AND SCL_ALL_SOURCES)
    add_library(scl_static STATIC ${SCL_ALL_SOURCES})
    target_link_libraries(scl_static PUBLIC scl_common)

    set_target_properties(scl_static PROPERTIES
        OUTPUT_NAME "scl_static"
        EXPORT_NAME "scl_static"
    )

    # Export symbols for static builds
    target_compile_definitions(scl_static PRIVATE
        SCL_BUILDING_LIBRARY=1
        SCL_UNSAFE_ACCESS=1
    )

    message(STATUS "Building static library: libscl_static.a")
elseif(SCL_BUILD_STATIC)
    message(STATUS "Skipping static library (no source files found)")
endif()

# Alias for default library (fallback to interface-only if no compiled libraries)
if(TARGET scl)
    add_library(scl::scl ALIAS scl)
elseif(TARGET scl_static)
    add_library(scl::scl ALIAS scl_static)
else()
    # Header-only fallback
    add_library(scl::scl ALIAS scl_common)
    message(STATUS "Using header-only configuration")
endif()

# -----------------------------------------------------------------------------
# Highway SIMD Library (Optional)
# -----------------------------------------------------------------------------

find_package(hwy QUIET)
if(hwy_FOUND)
    message(STATUS "Highway SIMD: Found")
    target_link_libraries(scl_common INTERFACE hwy::hwy)
    target_compile_definitions(scl_common INTERFACE SCL_USE_HIGHWAY=1)
else()
    message(STATUS "Highway SIMD: Not found (using fallback)")
    target_compile_definitions(scl_common INTERFACE SCL_USE_HIGHWAY=0)
endif()

# -----------------------------------------------------------------------------
# Note: This CMakeLists.txt Only Builds the Core Library
# -----------------------------------------------------------------------------
#
# This project ONLY builds the core SCL library with C FFI exports.
#
# Tests, benchmarks, examples, and Python bindings are built in SEPARATE
# projects that use the installed SCL library through its C API interface.
#
# Benefits of this design:
#   1. Clean separation: library vs consumers
#   2. Tests validate the public C API (not internals)
#   3. Library is independently installable
#   4. Easier maintenance and versioning
#   5. External projects can use SCL without test/benchmark dependencies
#
# To test/benchmark SCL, see the separate test and benchmark projects.

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install libraries (only if they exist)
if(TARGET scl)
    install(TARGETS scl scl_common
        EXPORT scl-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

if(TARGET scl_static)
    install(TARGETS scl_static
        EXPORT scl-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

# Install headers
# C++ core headers (*.hpp)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scl/core
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/scl
    FILES_MATCHING PATTERN "*.hpp"
)

# C-API headers (*.h)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scl/api
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/scl
    FILES_MATCHING PATTERN "*.h"
)

# Install CMake config (only if we have targets)
if(TARGET scl OR TARGET scl_static)
    install(EXPORT scl-targets
        FILE scl-targets.cmake
        NAMESPACE scl::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scl
    )
endif()

# TODO: Create cmake/scl-config.cmake.in template file
# configure_package_config_file(
#     ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scl-config.cmake.in
#     ${CMAKE_CURRENT_BINARY_DIR}/scl-config.cmake
#     INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scl
# )
#
# write_basic_package_version_file(
#     ${CMAKE_CURRENT_BINARY_DIR}/scl-config-version.cmake
#     VERSION ${PROJECT_VERSION}
#     COMPATIBILITY SameMajorVersion
# )
#
# install(FILES
#     ${CMAKE_CURRENT_BINARY_DIR}/scl-config.cmake
#     ${CMAKE_CURRENT_BINARY_DIR}/scl-config-version.cmake
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scl
# )

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

message(STATUS "")
message(STATUS "========================================")
message(STATUS "  Configuration Summary")
message(STATUS "========================================")
message(STATUS "Version:          ${PROJECT_VERSION}")
message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform:         ${SCL_PLATFORM}")
message(STATUS "Compiler:         ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Threading:        ${SCL_BACKEND_SELECTED}")
message(STATUS "")
message(STATUS "Libraries:")
message(STATUS "  Shared:         ${SCL_BUILD_SHARED}")
message(STATUS "  Static:         ${SCL_BUILD_STATIC}")
message(STATUS "")
message(STATUS "Sources (from .compile):")
message(STATUS "  Total:          ${SCL_SOURCE_COUNT} files")
message(STATUS "")
message(STATUS "Optimizations:")
message(STATUS "  Native arch:    ${SCL_ENABLE_NATIVE}")
message(STATUS "  LTO:            ${SCL_ENABLE_LTO}")
message(STATUS "")
if(SCL_ENABLE_ASAN OR SCL_ENABLE_UBSAN OR SCL_ENABLE_TSAN)
message(STATUS "Sanitizers:")
message(STATUS "  ASan:           ${SCL_ENABLE_ASAN}")
message(STATUS "  UBSan:          ${SCL_ENABLE_UBSAN}")
message(STATUS "  TSan:           ${SCL_ENABLE_TSAN}")
message(STATUS "")
endif()
message(STATUS "========================================")
message(STATUS "")
