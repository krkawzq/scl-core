# =============================================================================
# SCL Core v0.5 - C API Tests
# =============================================================================
#
# Standalone test build that links against the installed SCL library.
#
# Prerequisites:
#   1. Build and install SCL library first:
#      cd /path/to/scl-core
#      cmake -B build -DCMAKE_BUILD_TYPE=Release
#      cmake --build build
#      sudo cmake --install build (optional, or set CMAKE_PREFIX_PATH)
#
#   2. Build tests:
#      cd test/C
#      cmake -B build
#      cmake --build build
#
#   3. Run tests:
#      cd build
#      ctest
#      # or run individual tests:
#      ./test_error
#      ./test_sparse
#
# =============================================================================

cmake_minimum_required(VERSION 3.21)

project(scl-c-api-tests
    VERSION 0.5.0
    LANGUAGES CXX C
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

message(STATUS "")
message(STATUS "========================================")
message(STATUS "  SCL C-API Tests v${PROJECT_VERSION}")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")

# -----------------------------------------------------------------------------
# Find Dependencies
# -----------------------------------------------------------------------------

# Find Eigen3 (required for numerical verification)
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(WARNING "Eigen3 not found - some tests will be disabled")
    set(EIGEN3_AVAILABLE OFF)
else()
    message(STATUS "Found Eigen3: ${Eigen3_DIR}")
    set(EIGEN3_AVAILABLE ON)
endif()

# Find BLAS (optional for additional reference implementations)
find_package(BLAS QUIET)
if(BLAS_FOUND)
    message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
    set(BLAS_AVAILABLE ON)
else()
    message(STATUS "BLAS not found - BLAS reference tests will be skipped")
    set(BLAS_AVAILABLE OFF)
endif()

# -----------------------------------------------------------------------------
# Find SCL Library
# -----------------------------------------------------------------------------

# Try to find installed SCL library
# If not found, will look in ../build for locally built library
set(SCL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "Path to SCL root")

# 项目根目录（用于导出 compile_commands.json）
set(PROJECT_ROOT_DIR "${SCL_ROOT}")

# Include directories
set(SCL_INCLUDE_DIR "${SCL_ROOT}")
set(TEST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Library directories
set(SCL_LIB_DIR "${SCL_ROOT}/build/cmake")

message(STATUS "SCL root: ${SCL_ROOT}")
message(STATUS "SCL include: ${SCL_INCLUDE_DIR}")
message(STATUS "SCL library: ${SCL_LIB_DIR}")

# Check if library exists
find_library(SCL_LIBRARY 
    NAMES scl
    PATHS ${SCL_LIB_DIR}
    NO_DEFAULT_PATH
)

if(NOT SCL_LIBRARY)
    message(FATAL_ERROR "
    ========================================
    SCL library not found!
    
    Please build SCL first:
      cd ${SCL_ROOT}
      cmake -B build
      cmake --build build
    
    Then re-run this CMake.
    ========================================
    ")
endif()

message(STATUS "Found SCL library: ${SCL_LIBRARY}")
message(STATUS "")

# -----------------------------------------------------------------------------
# Compiler Options
# -----------------------------------------------------------------------------

set(TEST_COMPILE_OPTIONS)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND TEST_COMPILE_OPTIONS
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-unused-variable
    )
    
    # Debug flags
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        list(APPEND TEST_COMPILE_OPTIONS -g -O0)
    endif()
    
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    list(APPEND TEST_COMPILE_OPTIONS
        /W4
        /permissive-
    )
endif()

# -----------------------------------------------------------------------------
# Export compile_commands.json Setup
# -----------------------------------------------------------------------------

# Export compile_commands.json to project root as compile_commands_test.json
# This ensures the file is available after cmake configure and updated after builds
add_custom_command(
    OUTPUT "${PROJECT_ROOT_DIR}/compile_commands_test.json"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${PROJECT_ROOT_DIR}/compile_commands_test.json"
    DEPENDS "${CMAKE_BINARY_DIR}/compile_commands.json"
    COMMENT "Export compile_commands.json to ${PROJECT_ROOT_DIR}/compile_commands_test.json"
    VERBATIM
)

# Create a target that depends on the export command
# This ensures the file is generated during build
add_custom_target(export_compile_commands_test
    DEPENDS "${PROJECT_ROOT_DIR}/compile_commands_test.json"
    COMMENT "Exporting compile_commands_test.json to project root"
)

# -----------------------------------------------------------------------------
# Test Helper Function
# -----------------------------------------------------------------------------

function(add_scl_test test_name source_file)
    set(target_name "test_${test_name}")
    
    add_executable(${target_name} ${source_file})
    
    # Make test target depend on export_compile_commands_test
    # This ensures compile_commands_test.json is updated when tests are built
    add_dependencies(${target_name} export_compile_commands_test)
    
    # Include directories
    target_include_directories(${target_name} PRIVATE
        ${SCL_INCLUDE_DIR}
        ${TEST_INCLUDE_DIR}
    )
    
    # Compile options
    target_compile_options(${target_name} PRIVATE ${TEST_COMPILE_OPTIONS})
    
    # Link to SCL library
    target_link_libraries(${target_name} PRIVATE
        ${SCL_LIBRARY}
        ${CMAKE_DL_LIBS}
    )
    
    # Link Eigen3 if available
    if(EIGEN3_AVAILABLE)
        target_link_libraries(${target_name} PRIVATE Eigen3::Eigen)
        target_compile_definitions(${target_name} PRIVATE SCL_TEST_USE_EIGEN=1)
    endif()
    
    # Link BLAS if available
    if(BLAS_AVAILABLE)
        target_link_libraries(${target_name} PRIVATE ${BLAS_LIBRARIES})
        target_compile_definitions(${target_name} PRIVATE SCL_TEST_USE_BLAS=1)
    endif()
    
    # Set RPATH to find library at runtime
    if(UNIX AND NOT APPLE)
        set_target_properties(${target_name} PROPERTIES
            INSTALL_RPATH "${SCL_LIB_DIR}"
            BUILD_RPATH "${SCL_LIB_DIR}"
            BUILD_WITH_INSTALL_RPATH ON
        )
    elseif(APPLE)
        set_target_properties(${target_name} PROPERTIES
            INSTALL_RPATH "@loader_path/../../build"
            BUILD_RPATH "${SCL_LIB_DIR}"
            BUILD_WITH_INSTALL_RPATH ON
        )
    endif()
    
    # Add to CTest
    add_test(NAME ${test_name} COMMAND ${target_name})
    
    # Set test environment
    if(UNIX)
        set_tests_properties(${test_name} PROPERTIES
            ENVIRONMENT "LD_LIBRARY_PATH=${SCL_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
        )
    endif()
    
    message(STATUS "  ✓ ${test_name}")
endfunction()

# -----------------------------------------------------------------------------
# Enable Testing
# -----------------------------------------------------------------------------

enable_testing()

# -----------------------------------------------------------------------------
# Register Test Files
# -----------------------------------------------------------------------------

message(STATUS "Registering test files:")

# Core tests
add_scl_test(simple             src/test_simple.cpp)
add_scl_test(error              src/test_error.cpp)
add_scl_test(type               src/test_type.cpp)
add_scl_test(sparse             src/test_sparse.cpp)
add_scl_test(sparse_creation    src/test_sparse_creation.cpp)
add_scl_test(sparse_operations  src/test_sparse_operations.cpp)
add_scl_test(summary            src/test_summary.cpp)
add_scl_test(alignment          src/test_alignment.cpp)
add_scl_test(performance        src/test_performance.cpp)
add_scl_test(boundaries         src/test_boundaries.cpp)
add_scl_test(data_patterns      src/test_data_patterns.cpp)
add_scl_test(integer_types      src/test_integer_types.cpp)
add_scl_test(lifecycle          src/test_lifecycle.cpp)
add_scl_test(comprehensive      src/test_comprehensive.cpp)
add_scl_test(final_coverage     src/test_final_coverage.cpp)
add_scl_test(slice              src/test_slice.cpp)
add_scl_test(final_42           src/test_final_42.cpp)
add_scl_test(numerical_validation src/test_numerical_validation.cpp)

# Kernel tests
add_scl_test(log1p              src/test_log1p.cpp)

# Future tests can be added here:
# add_scl_test(dense       src/test_dense.cpp)
# add_scl_test(operations  src/test_operations.cpp)

# -----------------------------------------------------------------------------
# Custom Targets
# -----------------------------------------------------------------------------

# Collect all test targets
set(ALL_TEST_TARGETS
    test_simple
    test_error
    test_type
    test_sparse
    test_sparse_creation
    test_sparse_operations
)

# Run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${ALL_TEST_TARGETS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all C API tests..."
)

# Run tests with verbose output
add_custom_target(run_tests_verbose
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
    DEPENDS ${ALL_TEST_TARGETS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running tests (verbose)..."
)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

message(STATUS "")
message(STATUS "Test configuration complete:")
message(STATUS "  Binaries: ${CMAKE_BINARY_DIR}/test_*")
message(STATUS "  Library:  ${SCL_LIBRARY}")
message(STATUS "")
message(STATUS "Build tests: cmake --build build")
message(STATUS "Run tests:   cd build && ctest")
message(STATUS "            or: cmake --build build --target run_tests")
message(STATUS "")
message(STATUS "========================================")
message(STATUS ""
)
