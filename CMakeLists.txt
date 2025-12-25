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
    
    # Note: Exceptions are enabled by default for error handling in C-ABI layer
    # If you want to disable exceptions in hot paths, do it selectively per target
endif()

# MSVC-specific options
if(MSVC)
    add_compile_options(/W4 /permissive-)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /DNDEBUG)
    endif()
endif()

# Output directory for build artifacts (convenience for local dev)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/libs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/libs)

# -----------------------------------------------------------------------------
# 2. Configuration Options
# -----------------------------------------------------------------------------
# Threading Backend Selection
# Options: AUTO, SERIAL, BS, OPENMP, TBB
set(SCL_THREADING_BACKEND "AUTO" CACHE STRING "Select threading backend")
set_property(CACHE SCL_THREADING_BACKEND PROPERTY STRINGS AUTO SERIAL BS OPENMP TBB)

# macOS specific options
option(SCL_MAC_USE_OPENMP "Force enable OpenMP on macOS (Requires libomp)" OFF)

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

# --- 3.2 Google Highway (SIMD) ---
FetchContent_Declare(
    hwy
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG master # Use stable tag in production
)
# Disable Highway examples/tests to speed up build
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(hwy)

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
    
    # 6.3 Linking
    target_link_libraries(${target_name} PRIVATE 
        hwy       # Google Highway
        ${SCL_BACKEND_LIBS} # Threading Libs
    )

    # 6.4 Output Naming (e.g., scl_core_f32.so)
    set_target_properties(${target_name} PROPERTIES 
        OUTPUT_NAME "${output_name}"
        PREFIX "" # Remove 'lib' prefix on Linux/Mac to make Python loading easier
    )
    
    # 6.5 Installation
    # Install libraries to libs directory
    install(TARGETS ${target_name} 
        LIBRARY DESTINATION libs
        RUNTIME DESTINATION libs
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
# 7. Install Configuration
# -----------------------------------------------------------------------------
install(DIRECTORY include/ DESTINATION include)

# -----------------------------------------------------------------------------
# 8. Build Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "==========================================")
message(STATUS "SCL Core Build Configuration")
message(STATUS "==========================================")
message(STATUS "  Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:        ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Threading:       ${CHOSEN_BACKEND}")
message(STATUS "  Targets:         scl_core_f32, scl_core_f64")
message(STATUS "  Output Dir:      ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "  Compile Commands: ${CMAKE_BINARY_DIR}/compile_commands.json")
message(STATUS "==========================================")
message(STATUS "")
