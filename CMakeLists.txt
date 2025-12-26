cmake_minimum_required(VERSION 3.15)

# -----------------------------------------------------------------------------
# 1. Project Setup
# -----------------------------------------------------------------------------
project(scl-core LANGUAGES CXX VERSION 0.1.0)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
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
    add_compile_options(-Wall -Wextra -Wpedantic)
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
if(NOT TARGET bs_thread_pool)
    set(BS_THREAD_POOL_INCLUDE_DIR ${bs_thread_pool_SOURCE_DIR})
endif()

# --- 3.2 Google Highway (Deep Coupling / Static) ---
# We force a static build to embed Highway into our shared libraries.
FetchContent_Declare(
    hwy
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG master # Use stable tag in production if needed
)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE) # Force Static
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(hwy)

# --- 3.3 HDF5 (Optional, for native h5ad support) ---
find_package(HDF5 COMPONENTS C CXX)
if(HDF5_FOUND)
    message(STATUS "HDF5 found: ${HDF5_VERSION}")
    message(STATUS "HDF5 includes: ${HDF5_INCLUDE_DIRS}")
    message(STATUS "HDF5 libraries: ${HDF5_LIBRARIES}")
else()
    message(WARNING "HDF5 not found. Native h5ad support will be disabled.")
    message(WARNING "Install HDF5 development libraries to enable native h5ad I/O.")
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
# 5. Source Collection (Updated for Colocation)
# -----------------------------------------------------------------------------
# Recursively find all .cpp files in the scl/ directory
file(GLOB_RECURSE SCL_SOURCES "scl/*.cpp")

# -----------------------------------------------------------------------------
# 6. Multi-Precision Target Generation
# -----------------------------------------------------------------------------
macro(create_scl_backend target_name output_name precision_val)
    add_library(${target_name} SHARED ${SCL_SOURCES})
    
    # 6.1 Includes (Updated)
    target_include_directories(${target_name} 
        PUBLIC
            # Point to root so #include "scl/config.hpp" works
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>
        PRIVATE 
            ${BS_THREAD_POOL_INCLUDE_DIR}
    )

    # 6.2 Backend Configuration
    target_compile_definitions(${target_name} PRIVATE 
        ${SCL_BACKEND_DEFINES}
        SCL_PRECISION=${precision_val}
    )

    # 6.3 SIMD Configuration
    if(SCL_ONLY_SCALAR)
        target_compile_definitions(${target_name} PRIVATE HWY_COMPILE_ONLY_SCALAR)
    endif()
    
    # 6.4 Linking
    target_link_libraries(${target_name} PRIVATE 
        hwy       # Google Highway (Static)
        ${SCL_BACKEND_LIBS} # Threading Libs
    )
    
    # Link HDF5 if available
    if(HDF5_FOUND)
        target_include_directories(${target_name} PRIVATE ${HDF5_INCLUDE_DIRS})
        target_link_libraries(${target_name} PRIVATE ${HDF5_LIBRARIES})
        target_compile_definitions(${target_name} PRIVATE SCL_HAS_HDF5)
    endif()

    # 6.5 Output Naming
    set_target_properties(${target_name} PROPERTIES 
        OUTPUT_NAME "${output_name}"
        PREFIX "" 
    )
    
    # 6.6 Output Directory (for both build and install)
    # Build output: build/lib/ (for development)
    # Install output: src/scl_core/libs/ (for package installation)
    set_target_properties(${target_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
    
    # 6.7 Installation
    # Install libraries to scl/libs/ (inside the Python package)
    install(TARGETS ${target_name} 
        LIBRARY DESTINATION src/scl/libs
        RUNTIME DESTINATION src/scl/libs
    )
endmacro()

# --- Create Targets ---

# 1. Float 32 (Standard)
create_scl_backend(scl_core_f32 "scl_core_f32" 0)

# 2. Float 64 (High Precision)
create_scl_backend(scl_core_f64 "scl_core_f64" 1)

# -----------------------------------------------------------------------------
# 7. Install Headers (Updated)
# -----------------------------------------------------------------------------
# Copy the 'scl' folder structure (headers only) to 'include/scl'
install(DIRECTORY scl/ 
    DESTINATION include/scl
    FILES_MATCHING PATTERN "*.hpp"
)

# -----------------------------------------------------------------------------
# 8. Build Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "==========================================")
message(STATUS "SCL Core Build Configuration")
message(STATUS "==========================================")
message(STATUS "  Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "  Threading:       ${CHOSEN_BACKEND}")
message(STATUS "  SIMD:            ${SCL_ONLY_SCALAR} (ONLY_SCALAR Mode)")
message(STATUS "  HDF5:            ${HDF5_FOUND}")
message(STATUS "  Targets:         scl_core_f32, scl_core_f64")
message(STATUS "  Output Dir:      ${CMAKE_BINARY_DIR}/lib")
message(STATUS "  Install Dir:     src/scl/libs")
message(STATUS "==========================================")
message(STATUS "")

# -----------------------------------------------------------------------------
# 9. Tooling Support (Compile Commands Link)
# -----------------------------------------------------------------------------
# Automatically link compile_commands.json to project root for clangd/LSP
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    message(STATUS "Linking compile_commands.json to source directory")
    if(NOT WIN32)
        # Linux/macOS: Create a symbolic link
        execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${CMAKE_SOURCE_DIR}/compile_commands.json"
            ERROR_QUIET
        )
    else()
        # Windows: Copy the file (Symlinks require admin privileges often)
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${CMAKE_SOURCE_DIR}/compile_commands.json"
            ERROR_QUIET
        )
    endif()
endif()
