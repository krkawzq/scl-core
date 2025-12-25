cmake_minimum_required(VERSION 3.15)

# -----------------------------------------------------------------------------
# 1. Project Setup
# -----------------------------------------------------------------------------
project(scl-core LANGUAGES CXX VERSION 0.1.0)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position Independent Code (required for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Generate compile_commands.json for clangd and other tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type (default to Release if not specified)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Warning flags
    add_compile_options(-Wall -Wextra -Wpedantic)
    
    # Optimization flags
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -DNDEBUG)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0 -fsanitize=address -fsanitize=undefined)
    endif()
endif()

# MSVC-specific options
if(MSVC)
    add_compile_options(/W4 /permissive-)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /DNDEBUG)
    endif()
endif()

# Output directory for build artifacts
# We output directly to the python package directory for easier development/packaging
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/scl_core/libs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/scl_core/libs)

# -----------------------------------------------------------------------------
# 2. Configuration Options
# -----------------------------------------------------------------------------
# Threading Backend Selection
# Options: AUTO, SERIAL, BS, OPENMP, TBB
set(SCL_THREADING_BACKEND "AUTO" CACHE STRING "Select threading backend")
set_property(CACHE SCL_THREADING_BACKEND PROPERTY STRINGS AUTO SERIAL BS OPENMP TBB)

# macOS specific options
option(SCL_MAC_USE_OPENMP "Force enable OpenMP on macOS (Requires libomp)" OFF)

# SIMD Configuration
# If ON, forces Google Highway to generate scalar-only code (no SIMD)
option(SCL_ONLY_SCALAR "Force scalar execution (disable SIMD optimizations)" OFF)

# -----------------------------------------------------------------------------
# 3. Dependency Management (FetchContent)
# -----------------------------------------------------------------------------
include(FetchContent)
message(STATUS "Configuring Dependencies...")

# --- 3.1 BS::thread_pool (Header-only) ---
FetchContent_Declare(
    bs_thread_pool
    URL https://github.com/bshoshany/thread-pool/archive/refs/tags/v4.1.0.tar.gz
)
FetchContent_MakeAvailable(bs_thread_pool)
# Fix for BS::thread_pool include path if it doesn't expose a target properly
if(NOT TARGET bs_thread_pool)
    set(BS_THREAD_POOL_INCLUDE_DIR ${bs_thread_pool_SOURCE_DIR})
endif()

# --- 3.2 Google Highway (SIMD Framework) ---
# We treat Highway as a core static dependency (Deep Coupling)

# Configure Highway options BEFORE add_subdirectory or FetchContent
# This prevents Highway from downloading googletest and building unnecessary components
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "Build Highway examples" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "Build Highway tests" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "Build Highway contrib" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Build tests" FORCE)

# Strategy: Try to find Highway in libs/ first, then fall back to FetchContent
set(HWY_LOCAL_PATH "${CMAKE_SOURCE_DIR}/libs/highway")

if(EXISTS "${HWY_LOCAL_PATH}/CMakeLists.txt")
    message(STATUS "Found local Highway in libs/highway, using it instead of FetchContent")
    add_subdirectory(${HWY_LOCAL_PATH} ${CMAKE_BINARY_DIR}/highway)
else()
    message(STATUS "Highway not found in libs/, using FetchContent to download")
    FetchContent_Declare(
        hwy
        GIT_REPOSITORY https://github.com/google/highway.git
        GIT_TAG 1.0.7  # Use stable release instead of master
    )
    FetchContent_MakeAvailable(hwy)
endif()

# -----------------------------------------------------------------------------
# 4. Threading Backend Logic (The Arbiter)
# -----------------------------------------------------------------------------
message(STATUS "Configuring Threading Backend...")

# Resolve AUTO mode
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

# Prepare compile definitions and link libraries based on selection
set(SCL_BACKEND_DEFINES "")
set(SCL_BACKEND_LIBS "")

if(CHOSEN_BACKEND STREQUAL "SERIAL")
    list(APPEND SCL_BACKEND_DEFINES "SCL_BACKEND_SERIAL")

elseif(CHOSEN_BACKEND STREQUAL "BS")
    list(APPEND SCL_BACKEND_DEFINES "SCL_BACKEND_BS")
    # BS is header only, include dir added later

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
# Collect all C++ core implementation and binding wrappers
file(GLOB_RECURSE SCL_SOURCES 
    "src/cpp/*.cpp" 
    "src/bindings/*.cpp"
)

# -----------------------------------------------------------------------------
# 6. Multi-Precision Target Generation
# -----------------------------------------------------------------------------
# Helper macro to create a shared library for a specific precision
# precision_val: 0=f32, 1=f64, 2=f16
macro(create_scl_backend target_name output_name precision_val)
    add_library(${target_name} SHARED ${SCL_SOURCES})
    
    # 6.1 Includes
    target_include_directories(${target_name} 
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE 
            ${CMAKE_SOURCE_DIR}/include
            ${BS_THREAD_POOL_INCLUDE_DIR}
    )

    # 6.2 Backend Configuration
    target_compile_definitions(${target_name} PRIVATE 
        ${SCL_BACKEND_DEFINES}
        SCL_PRECISION=${precision_val}
    )

    # 6.3 SIMD Configuration (Proxy for Highway)
    if(SCL_ONLY_SCALAR)
        target_compile_definitions(${target_name} PRIVATE HWY_COMPILE_ONLY_SCALAR)
    endif()
    
    # 6.4 Linking
    target_link_libraries(${target_name} PRIVATE 
        hwy       # Google Highway (Static)
        ${SCL_BACKEND_LIBS} # Threading Libs
    )

    # 6.5 Output Naming (e.g., scl_core_f32.so)
    set_target_properties(${target_name} PROPERTIES 
        OUTPUT_NAME "${output_name}"
        PREFIX "" # Remove 'lib' prefix on Linux/Mac to make Python loading easier
    )
    
    # 6.6 Installation
    # Install libraries directly into the Python package structure
    install(TARGETS ${target_name} 
        LIBRARY DESTINATION src/scl_core/libs
        RUNTIME DESTINATION src/scl_core/libs
    )
endmacro()

# --- Create Targets ---

# 1. Float 32 (Standard)
create_scl_backend(scl_core_f32 "scl_core_f32" 0)

# 2. Float 64 (High Precision)
create_scl_backend(scl_core_f64 "scl_core_f64" 1)

# 3. Float 16 (Optional, commented out by default)
# create_scl_backend(scl_core_f16 "scl_core_f16" 2)

# -----------------------------------------------------------------------------
# 7. Build Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "==========================================")
message(STATUS "SCL Core Build Configuration")
message(STATUS "==========================================")
message(STATUS "  Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:        ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Threading:       ${CHOSEN_BACKEND}")
message(STATUS "  SIMD:            ${SCL_ONLY_SCALAR} (ONLY_SCALAR Mode)")
message(STATUS "  Targets:         scl_core_f32, scl_core_f64")
message(STATUS "  Output Dir:      ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "==========================================")
message(STATUS "")
