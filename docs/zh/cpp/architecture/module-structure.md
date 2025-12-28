# 模块结构

本文档描述 SCL-Core 的模块组织、依赖关系和设计模式。架构遵循严格的分层以保持关注点的清晰分离并最小化编译依赖。

## 设计哲学

### 严格分层

依赖关系仅单向流动：从上层到下层。下层永不依赖上层。

```
上层（特定于应用）
    ↓
    ↓ 依赖于
    ↓
下层（基础）
```

这确保：
- 下层可以独立使用
- 对上层的更改不影响下层
- 编译顺序是确定的
- 无循环依赖

### 最小依赖

每个模块只依赖它需要的东西。核心层除了 C++ 标准库和 Google Highway（用于 SIMD）外零内部依赖。

### 尽可能仅头文件

大多数核心组件是仅头文件的，用于：
- 零运行时开销（完美内联）
- 简化的构建系统
- 易于集成到其他项目
- 模板重代码自然工作

## 目录布局

```
scl/
├── config.hpp          # 构建配置（类型、特性）
├── version.hpp         # 版本信息
│
├── core/               # 基础层（无依赖）
│   ├── type.hpp        # 基本类型（Real、Index、Size）
│   ├── macros.hpp      # 编译器宏和属性
│   ├── error.hpp       # 错误处理、断言
│   ├── memory.hpp      # 对齐内存分配
│   ├── registry.hpp    # 内存生命周期跟踪
│   ├── simd.hpp        # SIMD 抽象（Highway）
│   ├── vectorize.hpp   # 向量化操作
│   ├── sparse.hpp      # 稀疏矩阵基础设施
│   ├── sort.hpp        # 排序算法
│   ├── argsort.hpp     # Argsort 实现
│   └── algo.hpp        # 通用算法
│
├── threading/          # 并行处理（依赖: core/）
│   ├── scheduler.hpp   # 工作窃取线程池
│   ├── parallel_for.hpp # 并行循环抽象
│   └── workspace.hpp   # 每线程工作空间池
│
├── kernel/             # 计算算子（依赖: core/、threading/）
│   ├── normalize.hpp   # 行/列归一化
│   ├── neighbors.hpp   # KNN 计算
│   ├── leiden.hpp      # Leiden 聚类
│   ├── ttest.hpp       # 统计检验
│   ├── spatial.hpp     # 空间分析
│   └── ... (80+ 文件)
│
├── math/               # 统计函数（依赖: core/、threading/、kernel/）
│   ├── stats.hpp       # 基本统计
│   ├── regression.hpp  # 线性回归
│   ├── mwu.hpp         # Mann-Whitney U 检验
│   └── approx/         # 快速近似
│       ├── exp.hpp
│       ├── log.hpp
│       └── special.hpp
│
├── mmap/               # 内存映射数组（依赖: core/）
│   ├── backend/        # OS 特定实现
│   │   ├── posix.hpp
│   │   └── windows.hpp
│   ├── cache/          # 页缓存管理
│   ├── memory/         # 页的内存池
│   ├── array.hpp       # 内存映射数组
│   └── sparse.hpp      # 内存映射稀疏矩阵
│
└── io/                 # 文件 I/O 实用程序（依赖: core/）
    ├── matrix.hpp      # 矩阵序列化
    └── format.hpp      # 格式转换
```

## 依赖图

```
┌──────────────────────────────────────┐
│        Python 绑定                   │
│    (scl-py，外部项目)                │
└────────────────┬─────────────────────┘
                 │
                 │ 使用
                 ↓
┌────────────────────────────────────┐
│              math/                 │
│         统计计算                   │
│  (stats, regression, mwu)          │
└────────────────┬───────────────────┘
                 │
                 │ 使用
                 ↓
┌────────────────────────────────────┐
│            kernel/                 │
│         计算算子                   │
│  (normalize, neighbors, leiden)    │
└────────────────┬───────────────────┘
                 │
                 │ 使用
                 ↓
┌────────────────────────────────────┐
│          threading/                │
│         并行处理                   │
│  (parallel_for, scheduler)         │
└────────────────┬───────────────────┘
                 │
                 │ 使用
                 ↓
┌────────────────────────────────────┐
│             core/                  │
│     基础类型和实用程序             │
│  (types, sparse, SIMD, registry)   │
└────────────────────────────────────┘
         │
         │ 依赖于
         ↓
┌────────────────────────────────────┐
│  C++ 标准库 + Highway              │
└────────────────────────────────────┘
```

**依赖规则：**
- core/ → 仅 C++ 标准库、Highway
- threading/ → core/
- kernel/ → core/、threading/
- math/ → core/、threading/、kernel/
- Python 绑定 → SCL-Core 全部

## 层描述

### 核心层 (scl/core/)

**目的：** 所有其他层使用的基础类型、数据结构和实用程序。

**关键文件：**

| 文件 | 目的 | 依赖 |
|------|---------|--------------|
| type.hpp | 基本类型（Real、Index、Size） | C++ 标准库 |
| macros.hpp | 编译器宏（SCL_FORCE_INLINE 等） | 无 |
| error.hpp | 错误处理（SCL_ASSERT、SCL_CHECK_*） | macros.hpp |
| memory.hpp | 对齐内存分配 | type.hpp、error.hpp |
| registry.hpp | 内存生命周期跟踪 | type.hpp、error.hpp |
| simd.hpp | SIMD 抽象（Highway 包装器） | type.hpp、Highway |
| vectorize.hpp | 向量化操作 | simd.hpp |
| sparse.hpp | 稀疏矩阵基础设施 | type.hpp、registry.hpp |
| sort.hpp | 排序算法 | type.hpp |
| argsort.hpp | Argsort 实现 | type.hpp、sort.hpp |
| algo.hpp | 通用算法（min、max 等） | type.hpp |

**关键类型：**

```cpp
namespace scl {
    // 可配置的基本类型（编译时）
    using Real = /* float | double | _Float16 */;
    using Index = /* int16_t | int32_t | int64_t */;
    using Size = size_t;
    
    // 稀疏矩阵（不连续存储）
    template <typename T, bool IsCSR>
    struct Sparse {
        using Pointer = T*;
        Pointer* data_ptrs_;      // 每行/列数据指针
        Pointer* indices_ptrs_;   // 每行/列索引指针
        Index* lengths_;          // 每行/列长度
        Index rows_, cols_, nnz_;
    };
    
    // 内存注册表（单例）
    class Registry {
        template <typename T> T* new_array(size_t count);
        void register_ptr(void* ptr, size_t bytes, AllocType type);
        void unregister_ptr(void* ptr);
        // ...
    };
}
```

**设计注释：**
- 除注册表（单例需要定义）外仅头文件
- 对其他 SCL 模块零依赖
- 最小模板以减少编译时间
- scl 命名空间中的所有公共 API

### 线程层 (scl/threading/)

**目的：** 多核 CPU 的并行处理基础设施。

**关键文件：**

| 文件 | 目的 | 依赖 |
|------|---------|--------------|
| scheduler.hpp | 工作窃取线程池 | core/ |
| parallel_for.hpp | 并行循环抽象 | scheduler.hpp |
| workspace.hpp | 每线程工作空间池 | core/memory.hpp |

**关键 API：**

```cpp
namespace scl::threading {
    // 自动并行化的并行循环
    void parallel_for(Size begin, Size end, 
                      std::function<void(size_t, size_t)> func);
    
    // 线程池
    class Scheduler {
        void execute(Task task);
        size_t num_threads() const;
        void set_num_threads(size_t n);
    };
    
    // 临时存储的工作空间池
    template <typename T>
    class WorkspacePool {
        WorkspacePool(size_t num_threads, size_t workspace_size);
        T* get(size_t thread_rank);
    };
}
```

**设计注释：**
- 负载平衡的工作窃取调度器
- 基于问题大小的自动并行化
- 每线程工作空间避免同步
- 支持嵌套并行

### 内核层 (scl/kernel/)

**目的：** 用于生物数据分析的计算算子（80+ 文件中的 400+ 函数）。

**按功能组织：**

| 类别 | 文件 | 关键函数 |
|----------|-------|---------------|
| 稀疏工具 | sparse.hpp、sparse_opt.hpp | to_csr、to_csc、to_contiguous_arrays |
| 归一化 | normalize.hpp、scale.hpp、log1p.hpp | normalize_rows、scale_rows、log1p_inplace |
| 统计 | ttest.hpp、mwu.hpp、metrics.hpp | ttest_rows、mann_whitney_u、compute_metrics |
| 邻居 | neighbors.hpp、bbknn.hpp | compute_knn、batch_balanced_knn |
| 聚类 | leiden.hpp、louvain.hpp | leiden_clustering、louvain_clustering |
| 空间 | spatial.hpp、hotspot.hpp | spatial_autocorr、hotspot_analysis |
| 富集 | enrichment.hpp、scoring.hpp | gene_set_enrichment、compute_scores |
| 降维 | pca.hpp、svd.hpp | compute_pca、truncated_svd |

**关键模式：**

**函数式 API - 无状态函数：**

```cpp
namespace scl::kernel::normalize {
    // 纯函数 - 无副作用
    void row_norms(
        const Sparse<Real, true>& matrix,
        NormMode mode,
        MutableSpan<Real> output
    );
    
    // 就地修改
    void normalize_rows_inplace(
        Sparse<Real, true>& matrix,
        NormMode mode,
        Real epsilon = 1e-12
    );
}
```

**基于模板的多态：**

```cpp
// 适用于任何 CSR 类型
template <CSRLike MatrixT>
void process_matrix(const MatrixT& matrix) {
    // MatrixT 可以是 Sparse<Real, true> 或任何兼容类型
}

// 概念定义
template <typename T>
concept CSRLike = requires(T matrix) {
    { matrix.rows() } -> std::convertible_to<Index>;
    { matrix.cols() } -> std::convertible_to<Index>;
    { matrix.nnz() } -> std::convertible_to<Index>;
    { matrix.primary_length(0) } -> std::convertible_to<Index>;
    { matrix.primary_values(0) } -> std::convertible_to<Span<Real>>;
    { matrix.primary_indices(0) } -> std::convertible_to<Span<Index>>;
};
```

**设计注释：**
- 无状态函数 - 无隐藏状态
- 通过 parallel_for 显式并行化
- SIMD 优化的热路径（多累加器、融合操作）
- 最小分配 - 使用工作空间池
- 双文件文档（.h 文档、.hpp 实现）

### 数学层 (scl/math/)

**目的：** 统计函数和数学计算。

**关键文件：**

| 文件 | 目的 | 依赖 |
|------|---------|--------------|
| stats.hpp | 基本统计（均值、方差等） | core/、threading/ |
| regression.hpp | 线性回归 | core/、threading/、kernel/ |
| mwu.hpp | Mann-Whitney U 检验 | core/、threading/ |
| approx/ | 快速近似 | core/simd.hpp |

**关键 API：**

```cpp
namespace scl::math {
    // 基本统计
    Real mean(const Real* data, size_t n);
    Real variance(const Real* data, size_t n, Real mean);
    Real std_dev(const Real* data, size_t n, Real mean);
    
    // 回归
    void linear_regression(
        const Real* x, const Real* y, size_t n,
        Real& slope, Real& intercept, Real& r_squared
    );
    
    // 统计检验
    Real mann_whitney_u(
        const Real* x, size_t nx,
        const Real* y, size_t ny
    );
}

namespace scl::math::approx {
    // 快速近似（SIMD 优化）
    Vec exp_fast(Tag d, Vec x);     // 比 std::exp 快约 2-3 倍
    Vec log_fast(Tag d, Vec x);     // 比 std::log 快约 2-3 倍
    Vec erf_fast(Tag d, Vec x);     // 比 std::erf 快约 5-10 倍
}
```

**设计注释：**
- 构建在 kernel/ 之上进行复杂操作
- 使用 SIMD 进行数值计算
- 提供精确和快速近似
- 线程安全，可以从并行循环调用

### 内存映射层 (scl/mmap/)

**目的：** 通过内存映射文件支持不适合 RAM 的大型数据集。

**关键文件：**

| 目录 | 目的 |
|-----------|---------|
| backend/ | OS 特定的 mmap 实现（POSIX、Windows） |
| cache/ | 页缓存管理（LRU 驱逐） |
| memory/ | 页的内存池 |

| 文件 | 目的 |
|------|---------|
| array.hpp | 内存映射密集数组 |
| sparse.hpp | 内存映射稀疏矩阵 |

**关键 API：**

```cpp
namespace scl::mmap {
    // 内存映射密集数组
    template <typename T>
    class Array {
        Array(const std::string& path, size_t size);
        T& operator[](size_t i);        // 延迟加载
        void flush();                    // 写回磁盘
    };
    
    // 内存映射稀疏矩阵
    template <typename T, bool IsCSR>
    class MappedSparse {
        // 与 scl::Sparse 相同的 API
        Index rows() const;
        Index cols() const;
        Index nnz() const;
        Span<T> primary_values(Index i);
        // ...
    };
}
```

**设计注释：**
- 延迟加载 - 仅在访问时加载页
- 用于内存管理的 LRU 缓存驱逐
- OS 特定后端（posix mmap、Windows MapViewOfFile）
- 与内存结构相同的 API 以便于迁移

### I/O 层 (scl/io/)

**目的：** 序列化和格式转换的文件 I/O 实用程序。

**关键文件：**

| 文件 | 目的 |
|------|---------|
| matrix.hpp | 矩阵序列化（二进制格式） |
| format.hpp | 格式转换（CSR ↔ CSC、密集 ↔ 稀疏） |

**关键 API：**

```cpp
namespace scl::io {
    // 保存/加载稀疏矩阵
    void save_matrix(const std::string& path, const Sparse<Real, true>& matrix);
    Sparse<Real, true> load_matrix(const std::string& path);
    
    // 格式转换
    Sparse<Real, false> csr_to_csc(const Sparse<Real, true>& matrix);
    DenseMatrix sparse_to_dense(const Sparse<Real, true>& matrix);
}
```

## 模块间通信

### 注册表模式

所有模块间的共享内存跟踪：

```cpp
// 在 kernel/ - 分配内存
auto& reg = scl::get_registry();
Real* data = reg.new_array<Real>(count);

// 在 math/ - 使用内存
process(data, count);

// 在 Python 绑定 - 传输所有权
py::array_t<Real> numpy_array = wrap_pointer(data, count);
// Python 现在拥有内存，将在销毁时调用 reg.unregister_ptr()
```

### 工作空间池

共享临时存储模式：

```cpp
// 在 threading/ - 创建池
WorkspacePool<Real> pool(num_threads, workspace_size);

// 在 kernel/ - 在并行循环中使用池
parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    auto* workspace = pool.get(thread_rank);
    compute_with_workspace(workspace, workspace_size);
});
```

### 类型系统

core/ 中定义的共享类型：

```cpp
// 所有模块使用相同的 Real、Index、Size 类型
// 在编译时配置一次

#include "scl/core/type.hpp"

void kernel_function(const Real* data, Index n);
void math_function(Real* output, Size count);
```

## 构建配置

### 编译时选项

```cpp
// scl/config.hpp
#pragma once

// 类型配置
#ifdef SCL_USE_FLOAT32
    using Real = float;
#elif defined(SCL_USE_FLOAT64)
    using Real = double;
#elif defined(SCL_USE_FLOAT16)
    using Real = _Float16;
#else
    using Real = float;  // 默认
#endif

#ifdef SCL_USE_INT16
    using Index = int16_t;
#elif defined(SCL_USE_INT32)
    using Index = int32_t;
#elif defined(SCL_USE_INT64)
    using Index = int64_t;
#else
    using Index = int32_t;  // 默认
#endif

// 特性标志
#ifdef SCL_ENABLE_SIMD
    #define SCL_SIMD_ENABLED 1
#else
    #define SCL_SIMD_ENABLED 0
#endif

#ifdef SCL_ENABLE_OPENMP
    #define SCL_OPENMP_ENABLED 1
    #include <omp.h>
#else
    #define SCL_OPENMP_ENABLED 0
#endif
```

### CMake 配置

```cmake
# 配置类型
option(SCL_USE_FLOAT32 "使用 32 位浮点数" ON)
option(SCL_USE_INT32 "使用 32 位整数" ON)

# 配置特性
option(SCL_ENABLE_SIMD "启用 SIMD 优化" ON)
option(SCL_ENABLE_OPENMP "启用 OpenMP" OFF)
option(SCL_BUILD_TESTS "构建测试" ON)

# 核心库（仅头文件接口）
add_library(scl-core INTERFACE)
target_include_directories(scl-core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(scl-core INTERFACE cxx_std_17)

# 内核（编译库）
add_library(scl-kernels STATIC
    scl/kernel/normalize.cpp
    scl/kernel/neighbors.cpp
    scl/kernel/leiden.cpp
    # ... 更多内核
)
target_link_libraries(scl-kernels PUBLIC scl-core)

# 查找 Highway 用于 SIMD
if(SCL_ENABLE_SIMD)
    find_package(hwy REQUIRED)
    target_link_libraries(scl-core INTERFACE hwy::hwy)
    target_compile_definitions(scl-core INTERFACE SCL_ENABLE_SIMD)
endif()

# 编译器特定优化
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(scl-kernels PRIVATE
        -march=native
        -mtune=native
        -ffast-math
        -fno-exceptions  # 可选: 减少二进制大小
    )
endif()
```

## 添加新模块

### 模块放置指南

1. **核心层：** 仅当被多个上层需要且无依赖时
2. **线程层：** 仅并行化实用程序
3. **内核层：** 大多数新算子放在这里
4. **数学层：** 如果它构建在 kernel/ 操作之上
5. **新层：** 仅当它形成多个模块使用的连贯抽象时

### 模块模板

**实现文件 (scl/kernel/mymodule.hpp)：**

```cpp
// =============================================================================
// FILE: scl/kernel/mymodule.hpp
// BRIEF: 我的模块的实现
// =============================================================================
#pragma once

#include "scl/kernel/mymodule.h"
#include "scl/core/type.hpp"
#include "scl/threading/parallel_for.hpp"

namespace scl::kernel::mymodule {

template <typename T>
void my_function(const T* input, T* output, size_t n) {
    // 最少的内联注释
    parallel_for(Size(0), n, [&](size_t i) {
        output[i] = process(input[i]);
    });
}

} // namespace scl::kernel::mymodule
```

**文档文件 (scl/kernel/mymodule.h)：**

```cpp
// =============================================================================
// FILE: scl/kernel/mymodule.h
// BRIEF: 我的模块的 API 参考
// NOTE: 仅文档 - 不要包含在构建中
// =============================================================================
#pragma once

namespace scl::kernel::mymodule {

/* -----------------------------------------------------------------------------
 * FUNCTION: my_function
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     简短的单行描述。
 *
 * PARAMETERS:
 *     input  [in]  输入数组，大小 n
 *     output [out] 输出数组，大小 n（预分配）
 *     n      [in]  数组大小
 *
 * PRECONDITIONS:
 *     - input 和 output 是有效指针
 *     - output 有 n 个元素的空间
 *
 * POSTCONDITIONS:
 *     - output[i] 包含 input[i] 的处理值
 *     - input 不变
 *
 * COMPLEXITY:
 *     时间:  O(n)
 *     空间: O(1) 辅助
 *
 * THREAD SAFETY:
 *     安全 - 内部并行化
 * -------------------------------------------------------------------------- */
template <typename T>
void my_function(
    const T* input,    // 输入数组 [n]
    T* output,         // 输出数组 [n]（预分配）
    size_t n           // 数组大小
);

} // namespace scl::kernel::mymodule
```

**CMakeLists.txt 更新：**

```cmake
# 如果模块需要编译
add_library(scl-kernels STATIC
    # ... 现有文件 ...
    scl/kernel/mymodule.cpp
)
```

### 新模块检查清单

- [ ] 模块放置在适当的层
- [ ] 依赖关系仅向下流动
- [ ] .hpp 实现文件带有最少的注释
- [ ] .h 文档文件带有全面的文档
- [ ] 函数遵循无状态函数模式
- [ ] 模板用于零开销抽象
- [ ] 适当的 SIMD 优化
- [ ] 通过 parallel_for 并行化
- [ ] 通过注册表进行内存管理（如果需要）
- [ ] 添加到 CMakeLists.txt
- [ ] 编写测试
- [ ] 更新文档

---

::: tip 模块独立性
良好的模块设计支持独立开发和测试。每个模块应该以最小的依赖自行可用。这加速开发并简化调试。
:::

