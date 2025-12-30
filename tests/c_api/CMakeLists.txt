# =============================================================================
# SCL Core - C API Tests (Standalone Build)
# =============================================================================
#
# Tests the C API by dynamically linking to PRE-COMPILED libraries.
# Does NOT trigger recompilation of main project (lazy/incremental).
#
# Prerequisites:
#   1. Run 'make compile-cpp' to build libraries first
#   2. Libraries must exist in python/scl/libs/
#
# =============================================================================

cmake_minimum_required(VERSION 3.15)

# -----------------------------------------------------------------------------
# Standalone Configuration (No Recompilation of Main Project)
# -----------------------------------------------------------------------------

# This prevents CMake from trying to build the main project
# Tests only consume pre-built libraries from python/scl/libs/

message(STATUS "")
message(STATUS "========================================")
message(STATUS "SCL C API Tests (Standalone Build)")
message(STATUS "========================================")

# Test binary output directory
set(TEST_UNITS_DIR "${CMAKE_SOURCE_DIR}/tests/c_api/units")
file(MAKE_DIRECTORY ${TEST_UNITS_DIR})

message(STATUS "  Output directory: ${TEST_UNITS_DIR}")
message(STATUS "  Library path: ${CMAKE_SOURCE_DIR}/python/scl/libs")

# Check if libraries exist
set(LIB_PATH "${CMAKE_SOURCE_DIR}/python/scl/libs/libscl_f64_i64.so")
if(NOT EXISTS ${LIB_PATH})
    message(WARNING "")
    message(WARNING "⚠️  Library not found: ${LIB_PATH}")
    message(WARNING "   Please run: make compile-cpp")
    message(WARNING "")
endif()

# -----------------------------------------------------------------------------
# Helper Function: Add Test Executable
# -----------------------------------------------------------------------------

function(add_scl_test test_name source_file)
    set(target_name "test_${test_name}")
    
    add_executable(${target_name} ${source_file})
    
    # Include directories
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}                      # For scl/ includes
        ${CMAKE_CURRENT_SOURCE_DIR}/include      # For test utilities
    )
    
    # C++20 standard
    target_compile_features(${target_name} PRIVATE cxx_std_20)
    
    # Compiler flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE
            -Wall -Wextra -Wpedantic
            -Wno-unused-parameter
            -Wno-unused-variable
            -Wno-sign-compare
        )
    endif()
    
    # Link to PRE-BUILT library (not a CMake target)
    target_link_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/python/scl/libs
    )
    
    target_link_libraries(${target_name} PRIVATE
        scl_f64_i64  # Links to libscl_f64_i64.so
        ${CMAKE_DL_LIBS}
    )
    
    # Link Eigen3 if available
    if(TARGET Eigen3::Eigen)
        target_link_libraries(${target_name} PRIVATE Eigen3::Eigen)
    endif()
    
    # Set output directory
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_UNITS_DIR}"
    )
    
    # Set RPATH to find libraries at runtime
    if(UNIX AND NOT APPLE)
        set_target_properties(${target_name} PROPERTIES
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/python/scl/libs"
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/python/scl/libs"
            BUILD_WITH_INSTALL_RPATH ON
        )
    elseif(APPLE)
        set_target_properties(${target_name} PROPERTIES
            INSTALL_RPATH "@loader_path/../../../../python/scl/libs"
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/python/scl/libs"
            BUILD_WITH_INSTALL_RPATH ON
        )
    endif()
    
    # Add to CTest
    add_test(NAME ${test_name} COMMAND ${target_name})
    
    # Set test environment
    set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_SOURCE_DIR}/python/scl/libs:$ENV{LD_LIBRARY_PATH}"
    )
    
    message(STATUS "  ✓ ${test_name}")
endfunction()

# -----------------------------------------------------------------------------
# Register Test Files
# -----------------------------------------------------------------------------

message(STATUS "")
message(STATUS "Registering test files:")

# Core infrastructure tests
add_scl_test(guards       src/test_guards.cpp)
add_scl_test(core         src/test_core.cpp)
add_scl_test(sparse       src/test_sparse.cpp)
add_scl_test(dense        src/test_dense.cpp)
add_scl_test(tools        src/test_tools.cpp)
add_scl_test(unsafe       src/test_unsafe.cpp)

# Algebra tests
add_scl_test(algebra      src/test_algebra.cpp)
add_scl_test(algebra_spmv src/test_algebra_spmv.cpp)

# Normalization tests
add_scl_test(normalize    src/test_normalize.cpp)
add_scl_test(log1p        src/test_log1p.cpp)

# Matrix operations
add_scl_test(comparison   src/test_comparison.cpp)
add_scl_test(merge        src/test_merge.cpp)

# Graph algorithms
add_scl_test(components   src/test_components.cpp)
add_scl_test(neighbors    src/test_neighbors.cpp)
add_scl_test(bbknn        src/test_bbknn.cpp)
add_scl_test(louvain      src/test_louvain.cpp)
add_scl_test(leiden       src/test_leiden.cpp)
add_scl_test(centrality   src/test_centrality.cpp)
add_scl_test(communication src/test_communication.cpp)

# Clustering and community detection
add_scl_test(permutation  src/test_permutation.cpp)
add_scl_test(propagation  src/test_propagation.cpp)

# Statistical tests
add_scl_test(mwu          src/test_mwu.cpp)
add_scl_test(multiple_testing src/test_multiple_testing.cpp)
add_scl_test(comp_effect_size src/test_comp_effect_size.cpp)
add_scl_test(correlation  src/test_correlation.cpp)
add_scl_test(association  src/test_association.cpp)

# Feature selection and analysis
add_scl_test(feature      src/test_feature.cpp)
add_scl_test(hvg          src/test_hvg.cpp)
add_scl_test(markers      src/test_markers.cpp)
add_scl_test(hotspot      src/test_hotspot.cpp)

# Quality control
add_scl_test(qc           src/test_qc.cpp)
add_scl_test(outlier      src/test_outlier.cpp)

# Trajectory and pseudotime
add_scl_test(pseudotime   src/test_pseudotime.cpp)
add_scl_test(lineage     src/test_lineage.cpp)

# Dimensionality reduction
add_scl_test(projection   src/test_projection.cpp)

# Imputation and preprocessing
add_scl_test(impute       src/test_impute.cpp)
add_scl_test(alignment    src/test_alignment.cpp)

# Metrics and evaluation
add_scl_test(metrics      src/test_metrics.cpp)
add_scl_test(entropy      src/test_entropy.cpp)
add_scl_test(mmd          src/test_mmd.cpp)

# Niche analysis
add_scl_test(niche        src/test_niche.cpp)

# Kernel operations
add_scl_test(kernel       src/test_kernel.cpp)

# Annotation
add_scl_test(annotation   src/test_annotation.cpp)
add_scl_test(group        src/test_group.cpp)

# -----------------------------------------------------------------------------
# Custom Targets
# -----------------------------------------------------------------------------

# Collect all test target names
set(ALL_TEST_TARGETS
    test_guards test_core test_sparse test_dense test_tools test_unsafe
    test_algebra test_algebra_spmv
    test_normalize test_log1p
    test_comparison test_merge
    test_components test_neighbors test_bbknn test_louvain test_leiden
    test_centrality test_communication
    test_permutation test_propagation
    test_mwu test_multiple_testing test_comp_effect_size test_correlation
    test_association
    test_feature test_hvg test_markers test_hotspot
    test_qc test_outlier
    test_pseudotime test_lineage
    test_projection
    test_impute test_alignment
    test_metrics test_entropy test_mmd
    test_niche
    test_kernel
    test_annotation test_group
)

# Run all tests
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${ALL_TEST_TARGETS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all C API tests..."
)

# Run tests with verbose output
add_custom_target(run_all_tests_verbose
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
message(STATUS "  Binaries: ${TEST_UNITS_DIR}/test_*")
message(STATUS "  Libraries: ${CMAKE_SOURCE_DIR}/python/scl/libs/libscl_*.so")
message(STATUS "  Eigen3: ${Eigen3_FOUND}")
message(STATUS "")
message(STATUS "Build tests: ninja")
message(STATUS "Run tests:   ctest OR ninja run_all_tests")
message(STATUS "")
message(STATUS "========================================")
message(STATUS "")
