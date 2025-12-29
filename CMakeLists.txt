cmake_minimum_required(VERSION 3.15)

# =============================================================================
# SCL Core - Unified CMake Configuration
# =============================================================================
project(scl-core LANGUAGES CXX VERSION 0.1.0)

# -----------------------------------------------------------------------------
# 1. Global Configuration
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler Flags (GCC/Clang)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -DNDEBUG)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0 -fsanitize=address -fsanitize=undefined)
    endif()
endif()

# Compiler Flags (MSVC)
if(MSVC)
    add_compile_options(/W4 /permissive-)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /DNDEBUG)
    endif()
endif()

# -----------------------------------------------------------------------------
# 2. Configuration Options
# -----------------------------------------------------------------------------
# Threading Backend Selection
set(SCL_THREADING_BACKEND "AUTO" CACHE STRING "Select threading backend")
set_property(CACHE SCL_THREADING_BACKEND PROPERTY STRINGS AUTO SERIAL BS OPENMP TBB)

# macOS specific options
option(SCL_MAC_USE_OPENMP "Force enable OpenMP on macOS (Requires libomp)" OFF)

# SIMD Configuration
option(SCL_ONLY_SCALAR "Force scalar execution (disable SIMD optimizations)" OFF)

# Output directory (can be overridden)
set(SCL_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/python/scl/libs" CACHE PATH "Output directory for compiled libraries")

# -----------------------------------------------------------------------------
# 3. Dependencies (FetchContent)
# -----------------------------------------------------------------------------
include(FetchContent)
message(STATUS "Configuring Dependencies...")

# --- 3.1 BS::thread_pool (Header-only) ---
FetchContent_Declare(
    bs_thread_pool
    URL https://github.com/bshoshany/thread-pool/archive/refs/tags/v4.1.0.tar.gz
)
FetchContent_MakeAvailable(bs_thread_pool)
if(NOT TARGET bs_thread_pool)
    set(BS_THREAD_POOL_INCLUDE_DIR ${bs_thread_pool_SOURCE_DIR})
endif()

# --- 3.2 Google Highway (Static for embedding into shared libs) ---
FetchContent_Declare(
    hwy
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG master
)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB ON CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(hwy)

# Ensure Highway is built with PIC
set_target_properties(hwy PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(TARGET hwy_contrib)
    set_target_properties(hwy_contrib PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# --- 3.3 HDF5 (Optional) ---
find_package(HDF5 COMPONENTS C CXX)
if(HDF5_FOUND)
    message(STATUS "HDF5 found: ${HDF5_VERSION}")
else()
    message(WARNING "HDF5 not found. Native h5ad support will be disabled.")
endif()

# --- 3.4 Eigen3 (For Testing Reference Implementation) ---
find_package(Eigen3 3.4 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${Eigen3_VERSION}")
else()
    message(STATUS "Eigen3 not found, will fetch from GitHub")
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen3)
endif()

# --- 3.5 Catch2 (For C API Testing) ---
option(SCL_BUILD_TESTS "Build C API tests" ON)
if(SCL_BUILD_TESTS)
    find_package(Catch2 3 QUIET)
    if(NOT Catch2_FOUND)
        message(STATUS "Catch2 not found, will fetch from GitHub")
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.5.2
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(Catch2)
    endif()
endif()

# -----------------------------------------------------------------------------
# 4. Threading Backend Logic
# -----------------------------------------------------------------------------
message(STATUS "Configuring Threading Backend...")

if(SCL_THREADING_BACKEND STREQUAL "AUTO")
    if(APPLE AND NOT SCL_MAC_USE_OPENMP)
        set(CHOSEN_BACKEND "BS")
    else()
        set(CHOSEN_BACKEND "OPENMP")
    endif()
else()
    set(CHOSEN_BACKEND ${SCL_THREADING_BACKEND})
endif()

message(STATUS "Active Threading Backend: ${CHOSEN_BACKEND}")

set(SCL_BACKEND_DEFINES "")
set(SCL_BACKEND_LIBS "")

if(CHOSEN_BACKEND STREQUAL "SERIAL")
    list(APPEND SCL_BACKEND_DEFINES "SCL_BACKEND_SERIAL")
elseif(CHOSEN_BACKEND STREQUAL "BS")
    list(APPEND SCL_BACKEND_DEFINES "SCL_BACKEND_BS")
elseif(CHOSEN_BACKEND STREQUAL "OPENMP")
    find_package(OpenMP REQUIRED)
    list(APPEND SCL_BACKEND_LIBS OpenMP::OpenMP_CXX)
    list(APPEND SCL_BACKEND_DEFINES "SCL_BACKEND_OPENMP")
    if(APPLE AND SCL_MAC_USE_OPENMP)
        list(APPEND SCL_BACKEND_DEFINES "SCL_MAC_USE_OPENMP")
    endif()
elseif(CHOSEN_BACKEND STREQUAL "TBB")
    find_package(TBB REQUIRED)
    list(APPEND SCL_BACKEND_LIBS TBB::tbb)
    list(APPEND SCL_BACKEND_DEFINES "SCL_BACKEND_TBB")
else()
    message(FATAL_ERROR "Unknown backend: ${CHOSEN_BACKEND}")
endif()

# -----------------------------------------------------------------------------
# 5. Source Collection
# -----------------------------------------------------------------------------
file(GLOB_RECURSE SCL_ALL_SOURCES
    "scl/core/*.cpp"
    "scl/kernel/*.cpp"
    "scl/math/*.cpp"
    "scl/mmap/*.cpp"
    "scl/threading/*.cpp"
    "scl/binding/c_api/*.cpp"
)

list(LENGTH SCL_ALL_SOURCES SCL_SOURCES_COUNT)
message(STATUS "Found ${SCL_SOURCES_COUNT} C++ source files")

# Create output directory
file(MAKE_DIRECTORY ${SCL_OUTPUT_DIR})

# -----------------------------------------------------------------------------
# 6. Multi-Precision Target Generation Macro
# -----------------------------------------------------------------------------
macro(create_scl_lib float_type int_type suffix)
    set(TARGET_NAME "scl_${suffix}")
    
    message(STATUS "Configuring Target: ${TARGET_NAME}")

    # Create shared library
    add_library(${TARGET_NAME} SHARED ${SCL_ALL_SOURCES})

    # Include directories
    target_include_directories(${TARGET_NAME} 
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${BS_THREAD_POOL_INCLUDE_DIR}
            $<TARGET_PROPERTY:hwy,INTERFACE_INCLUDE_DIRECTORIES>
    )

    # Compile definitions (precision + backend)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        ${SCL_BACKEND_DEFINES}
        SCL_FLOAT_TYPE=${float_type}
        SCL_INT_TYPE=${int_type}
    )

    # SIMD Configuration
    if(SCL_ONLY_SCALAR)
        target_compile_definitions(${TARGET_NAME} PRIVATE HWY_COMPILE_ONLY_SCALAR)
    endif()

    # Linking (with whole-archive for Highway on Unix)
    if(UNIX AND NOT APPLE)
        # Linux: Use whole-archive to ensure all Highway symbols are exported
        target_link_options(${TARGET_NAME} PRIVATE
            "LINKER:--whole-archive,$<TARGET_FILE:hwy>,$<TARGET_FILE:hwy_contrib>,--no-whole-archive")
        target_link_libraries(${TARGET_NAME} PRIVATE ${SCL_BACKEND_LIBS})
    elseif(APPLE)
        # macOS: Use -force_load
        target_link_options(${TARGET_NAME} PRIVATE
            "LINKER:-force_load,$<TARGET_FILE:hwy>"
            "LINKER:-force_load,$<TARGET_FILE:hwy_contrib>")
        target_link_libraries(${TARGET_NAME} PRIVATE ${SCL_BACKEND_LIBS})
    else()
        # Windows: Standard linking
        target_link_libraries(${TARGET_NAME} PRIVATE hwy hwy_contrib ${SCL_BACKEND_LIBS})
    endif()

    # Ensure hwy libs are built first
    add_dependencies(${TARGET_NAME} hwy hwy_contrib)

    # HDF5 Support
    if(HDF5_FOUND)
        target_include_directories(${TARGET_NAME} PRIVATE ${HDF5_INCLUDE_DIRS})
        target_link_libraries(${TARGET_NAME} PRIVATE ${HDF5_LIBRARIES})
        target_compile_definitions(${TARGET_NAME} PRIVATE SCL_HAS_HDF5)
    endif()

    # Output configuration: directly to python/scl/libs/
    set_target_properties(${TARGET_NAME} PROPERTIES
        OUTPUT_NAME "scl_${suffix}"
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY "${SCL_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${SCL_OUTPUT_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${SCL_OUTPUT_DIR}"
    )

    # Installation rules
    install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION python/scl/libs
        RUNTIME DESTINATION python/scl/libs
        ARCHIVE DESTINATION python/scl/libs
    )
endmacro()

# -----------------------------------------------------------------------------
# 7. Create Library Variants
# -----------------------------------------------------------------------------
create_scl_lib(0 0 "f32_i32")
create_scl_lib(1 1 "f64_i64")
create_scl_lib(0 1 "f32_i64")
create_scl_lib(1 0 "f64_i32")

# -----------------------------------------------------------------------------
# 8. Install Headers
# -----------------------------------------------------------------------------
install(DIRECTORY scl/
    DESTINATION include/scl
    FILES_MATCHING PATTERN "*.hpp"
)

# -----------------------------------------------------------------------------
# 9. Build Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "==========================================")
message(STATUS "SCL Core Build Configuration")
message(STATUS "==========================================")
message(STATUS "  Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "  Threading:       ${CHOSEN_BACKEND}")
message(STATUS "  SIMD Scalar:     ${SCL_ONLY_SCALAR}")
message(STATUS "  HDF5:            ${HDF5_FOUND}")
message(STATUS "  Source Files:    ${SCL_SOURCES_COUNT}")
message(STATUS "  Output Dir:      ${SCL_OUTPUT_DIR}")
message(STATUS "  Targets:         scl_f32_i32, scl_f64_i64, scl_f32_i64, scl_f64_i32")
message(STATUS "==========================================")
message(STATUS "")

# -----------------------------------------------------------------------------
# 10. C API Tests
# -----------------------------------------------------------------------------
option(SCL_BUILD_TESTS "Build C API tests" ON)

if(SCL_BUILD_TESTS)
    message(STATUS "Configuring C API tests...")
    enable_testing()
    add_subdirectory(tests/c_api)
else()
    message(STATUS "Tests disabled (SCL_BUILD_TESTS=OFF)")
endif()

# -----------------------------------------------------------------------------
# 11. Tooling Support (compile_commands.json)
# -----------------------------------------------------------------------------
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    if(NOT WIN32)
        execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${CMAKE_SOURCE_DIR}/compile_commands.json"
            ERROR_QUIET)
    else()
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${CMAKE_SOURCE_DIR}/compile_commands.json"
            ERROR_QUIET)
    endif()
endif()
