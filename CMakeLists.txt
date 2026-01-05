cmake_minimum_required(VERSION 3.21) # 指定CMake的最小版本

project(scl-core
    VERSION 0.6.0            # 工程版本
    LANGUAGES CXX C          # 支持的语言
)

# -----------------------------------------------------------------------------
# C++ Standard & Compile Commands
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)             # 设置C++标准为20
set(CMAKE_CXX_STANDARD_REQUIRED ON)    # 强制使用C++20
set(CMAKE_CXX_EXTENSIONS OFF)          # 禁用编译器扩展
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # 导出compile_commands.json，便于IDE与工具使用

# Symlink compile_commands.json for IDEs/tools (if out of source)
# 将编译指令文件符号链接（如build目录和源码目录不同），有助于各种工具定位
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(_compile_commands_src "${CMAKE_BINARY_DIR}/compile_commands.json")
    set(_compile_commands_dst "${CMAKE_BINARY_DIR}/compile_commands.json")
    if(EXISTS "${_compile_commands_src}" AND NOT "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
        if(NOT EXISTS "${_compile_commands_dst}")
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E create_symlink
                "${_compile_commands_src}" "${_compile_commands_dst}"
            )
            message(STATUS "Created symlink: ${_compile_commands_dst} -> ${_compile_commands_src}")
        endif()
    endif()
endif()

# -----------------------------------------------------------------------------
# Build Type & Platform
# -----------------------------------------------------------------------------
# 默认构建类型为Release，并给CMAKE_BUILD_TYPE添加可选项
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# 平台检测，根据变量设置平台
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
# Options & Backend Selection
# -----------------------------------------------------------------------------
# 后端线程库选择（可选OpenMP、TBB、BS等），以及常用构建配置选项
set(SCL_BACKEND "AUTO" CACHE STRING "Threading backend: AUTO, OPENMP, TBB, BS, SERIAL")
set_property(CACHE SCL_BACKEND PROPERTY STRING AUTO OPENMP TBB BS SERIAL)

option(SCL_BUILD_SHARED "Build shared library with C FFI exports" ON)    # 构建动态库
option(SCL_BUILD_STATIC "Build Static library with C FFI exports" ON)    # 构建静态库

option(SCL_ENABLE_LTO    "Enable link-time optimization" OFF)            # 启用LTO链接时优化
option(SCL_ENABLE_NATIVE "Enable native architecture optimization" ON)   # 针对本机架构优化
option(SCL_ENABLE_ASAN   "Enable AddressSanitizer" OFF)                 # 启用AddressSanitizer
option(SCL_ENABLE_UBSAN  "Enable UndefinedBehaviorSanitizer" OFF)       # 启用Undefined Behavior Sanitizer
option(SCL_ENABLE_TSAN   "Enable ThreadSanitizer" OFF)                  # 启用ThreadSanitizer

# Release默认启用LTO优化
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT DEFINED SCL_ENABLE_LTO)
    set(SCL_ENABLE_LTO ON)
endif()

set(SCL_BACKEND_FOUND FALSE)  # 标记是否找到线程后端
# 自动选择线程后端
if(SCL_BACKEND STREQUAL "AUTO")
    if(SCL_PLATFORM STREQUAL "MACOS")
        set(SCL_BACKEND_SELECTED "BS") # macOS默认BS::thread_pool
        message(STATUS "Threading: Auto-selected BS::thread_pool (macOS default)")
    else()
        find_package(OpenMP QUIET)     # 优先查找OpenMP
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

# 针对不同后端设置相关变量和标志
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
    set(SCL_THREADING_LIBS "")
    set(SCL_THREADING_DEFINES SCL_BACKEND_BS=1)
    set(SCL_BACKEND_FOUND TRUE)
elseif(SCL_BACKEND_SELECTED STREQUAL "SERIAL")
    set(SCL_THREADING_LIBS "")
    set(SCL_THREADING_DEFINES SCL_BACKEND_SERIAL=1)
    set(SCL_BACKEND_FOUND TRUE)
endif()

# 没有合法后端则报错
if(NOT SCL_BACKEND_FOUND)
    message(FATAL_ERROR "Invalid threading backend: ${SCL_BACKEND_SELECTED}")
endif()
message(STATUS "Threading backend: ${SCL_BACKEND_SELECTED}")

# -----------------------------------------------------------------------------
# Compiler Flags & Optimization
# -----------------------------------------------------------------------------
set(SCL_WARNING_FLAGS "") # 编译器警告选项
# GNU/Clang警告增强
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND SCL_WARNING_FLAGS
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wcast-qual -Wunused
        -Wformat=2 -Wnull-dereference -Wdouble-promotion
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND SCL_WARNING_FLAGS -Wduplicated-cond -Wduplicated-branches -Wlogical-op)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    list(APPEND SCL_WARNING_FLAGS /W4 /permissive- /Zc:__cplusplus /Zc:preprocessor)
endif()

set(SCL_OPTIMIZATION_FLAGS "") # 优化选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
        list(APPEND SCL_OPTIMIZATION_FLAGS -O3)
        if(SCL_ENABLE_NATIVE)
            list(APPEND SCL_OPTIMIZATION_FLAGS -march=native)
            message(STATUS "Native architecture optimization: ON")
        endif()
        # list(APPEND SCL_OPTIMIZATION_FLAGS -ffast-math) # 可选极致优化
    endif()
    if(SCL_ENABLE_LTO)
        list(APPEND SCL_OPTIMIZATION_FLAGS -flto)
        message(STATUS "Link-time optimization: ON")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
        list(APPEND SCL_OPTIMIZATION_FLAGS /O2 /Ob3)
        if(SCL_ENABLE_NATIVE)
            list(APPEND SCL_OPTIMIZATION_FLAGS /arch:AVX2)
        endif()
    endif()
    if(SCL_ENABLE_LTO)
        list(APPEND SCL_OPTIMIZATION_FLAGS /GL)
    endif()
endif()

set(SCL_SANITIZER_FLAGS "")  # 各种Sanitizer相关选项
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
# Source Files & Headers
# -----------------------------------------------------------------------------
set(SCL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/c_api) # 头文件目录
set(SCL_API_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/src)       # API头文件目录

set(SCL_COMPILE_LIST_FILE "${CMAKE_CURRENT_SOURCE_DIR}/.compile-files") # 维护的源码列表
set(SCL_ALL_SOURCES "")
if(EXISTS "${SCL_COMPILE_LIST_FILE}")
    file(STRINGS "${SCL_COMPILE_LIST_FILE}" SCL_COMPILE_LIST)
    foreach(src_file ${SCL_COMPILE_LIST})
        string(STRIP "${src_file}" src_file)
        if(src_file AND NOT src_file MATCHES "^#") # 跳过注释行和空行
            if(NOT IS_ABSOLUTE "${src_file}")
                set(src_file "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}") # 转成绝对路径
            endif()
            if(EXISTS "${src_file}")
                list(APPEND SCL_ALL_SOURCES "${src_file}")
            else()
                message(WARNING "Source file not found: ${src_file}")
            endif()
        endif()
    endforeach()
endif()
list(LENGTH SCL_ALL_SOURCES SCL_SOURCE_COUNT) # 获取代码文件数量
message(STATUS "Source files from .compile-files: ${SCL_SOURCE_COUNT} files")

# 自动查找头文件，递归scl目录
file(GLOB_RECURSE SCL_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/scl/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scl/*.h
)
list(LENGTH SCL_HEADERS SCL_HEADER_COUNT)

message(STATUS "\nSource Configuration:")
message(STATUS "  Sources (from .compile): ${SCL_SOURCE_COUNT} files")
message(STATUS "  Headers (auto-discovered): ${SCL_HEADER_COUNT} files\n")

# -----------------------------------------------------------------------------
# Target Definitions
# -----------------------------------------------------------------------------
# scl_common为INTERFACE库，仅用于公共头文件和编译选项
add_library(scl_common INTERFACE)
target_include_directories(scl_common INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
    $<INSTALL_INTERFACE:include>
)

# 版本号保护：当未定义时降级为0
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
    ${SCL_THREADING_DEFINES}                               # 线程后端宏定义
    SCL_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}             # 主版本号
    SCL_VERSION_MINOR=${PROJECT_VERSION_MINOR}             # 次版本号
    SCL_VERSION_PATCH=${PROJECT_VERSION_PATCH}             # 修订号
    $<$<CONFIG:Debug>:SCL_DEBUG=1>                         # Debug构建宏
    $<$<CONFIG:Release>:NDEBUG>                            # Release NDEBUG
)

target_compile_options(scl_common INTERFACE
    ${SCL_WARNING_FLAGS}       # 警告
    ${SCL_OPTIMIZATION_FLAGS}  # 优化
    ${SCL_SANITIZER_FLAGS}     # Sanitizer
)

target_link_libraries(scl_common INTERFACE ${SCL_THREADING_LIBS}) # 链接线程库
if(SCL_SANITIZER_FLAGS)
    target_link_options(scl_common INTERFACE ${SCL_SANITIZER_FLAGS}) # Sanitizer相关链接选项
endif()

# Shared Library
# 动态库目标，包含所有源码
if(SCL_BUILD_SHARED AND SCL_ALL_SOURCES)
    add_library(scl SHARED ${SCL_ALL_SOURCES})
    target_link_libraries(scl PUBLIC scl_common)
    set_target_properties(scl PROPERTIES
        VERSION   ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        OUTPUT_NAME "scl"
        EXPORT_NAME "scl"
        C_VISIBILITY_PRESET hidden           # C接口符号默认隐藏
        CXX_VISIBILITY_PRESET hidden         # C++符号默认隐藏
        VISIBILITY_INLINES_HIDDEN ON
    )
    target_compile_definitions(scl PRIVATE
        SCL_BUILDING_LIBRARY=1               # 库内部标志
        SCL_UNSAFE_ACCESS=1                  # 性能优化相关标志
    )
    if(WIN32)
        target_compile_definitions(scl PUBLIC SCL_DLL=1)  # Windows特定动态库宏
    endif()
    message(STATUS "Building shared library: libscl.so")
elseif(SCL_BUILD_SHARED)
    message(STATUS "Skipping shared library (no source files found)")
endif()

# Static Library
# 静态库目标，包含所有源码
if(SCL_BUILD_STATIC AND SCL_ALL_SOURCES)
    add_library(scl_static STATIC ${SCL_ALL_SOURCES})
    target_link_libraries(scl_static PUBLIC scl_common)
    set_target_properties(scl_static PROPERTIES
        OUTPUT_NAME "scl_static"
        EXPORT_NAME "scl_static"
    )
    target_compile_definitions(scl_static PRIVATE
        SCL_BUILDING_LIBRARY=1     # 静态库标志
        SCL_UNSAFE_ACCESS=1
    )
    message(STATUS "Building static library: libscl_static.a")
elseif(SCL_BUILD_STATIC)
    message(STATUS "Skipping static library (no source files found)")
endif()

# Default Library Alias
# 默认导出scl::scl别名，方便下游项目引用
if(TARGET scl)
    add_library(scl::scl ALIAS scl)
elseif(TARGET scl_static)
    add_library(scl::scl ALIAS scl_static)
else()
    add_library(scl::scl ALIAS scl_common)
    message(STATUS "Using header-only configuration")
endif()

# -----------------------------------------------------------------------------
# Output Summary
# -----------------------------------------------------------------------------
# 显示所有配置和构建选项信息
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
