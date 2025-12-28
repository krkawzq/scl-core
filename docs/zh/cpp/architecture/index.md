# 架构概览

SCL-Core 是一个高性能生物数据分析库，建立在三个基础原则之上：零开销抽象、数据导向设计和显式资源管理。本节解释了使 SCL-Core 能够为稀疏矩阵计算提供最优性能的架构决策。

## 核心使命

**构建高性能生物算子库，具有零开销 C++ 内核和稳定的 C-ABI 表面以实现 Python 集成。**

战略焦点：**稀疏 + 非线性**

SCL-Core 专注于稀疏矩阵操作与非线性计算的交汇处——这是传统密集线性代数库（BLAS/LAPACK/Eigen）服务不足的领域。我们避免重新发明成熟的密集线性代数原语，而是专注于利用稀疏性同时执行生物数据分析中常见的复杂非线性变换的算法。

## 设计哲学

### 1. 零开销抽象

每一层抽象都必须编译为最优机器码，不产生任何运行时惩罚。

**原则：**
- 性能关键路径中无虚函数
- 基于模板的编译时多态
- 热点函数的激进内联
- 编译时计算的 constexpr
- 仅在 API 边界进行类型擦除

**示例：**

```cpp
// 高层 API
template <CSRLike MatrixT>
void normalize_rows(MatrixT& matrix, NormMode mode, Real eps);

// 编译为紧凑的 SIMD 循环，无抽象开销
// 与手写内联汇编相同的性能
```

模板系统在编译时解决所有多态，允许优化器为类型和参数的每种组合生成专门的代码路径。

### 2. 数据导向设计

内存访问模式决定现代硬件上的性能。SCL-Core 组织数据以最大化缓存局部性和内存带宽利用率。

**原则：**
- 连续内存布局以启用预取
- 当向量化收益超过局部性时使用结构数组（SoA）
- 批处理以分摊操作开销
- 可预测访问模式的显式预取
- 热循环中最小化指针间接引用

**内存布局策略：**

与传统的连续存储 CSR 不同，SCL-Core 使用基于指针的不连续结构，这使得：
- 块分配以更好地管理内存
- 灵活的行/列所有权模型
- 与 Python 的零拷贝集成
- 高效的部分矩阵操作

### 3. 显式资源管理

无隐藏分配。无隐式成本。每个内存操作都是显式且可跟踪的。

**原则：**
- 为临时存储预分配工作空间池
- 对齐分配的手动内存管理
- 基于注册表的 Python 集成生命周期跟踪
- 热路径中不使用 std::vector - 使用固定大小缓冲区
- 共享缓冲区的引用计数

**注册表模式：**

全局注册表跟踪所有分配，实现：
- 无垃圾收集的确定性清理
- 跨语言边界（C++ ↔ Python）的安全内存传输
- 共享缓冲区的引用计数
- 线程安全的并发访问
- 内存使用分析和泄漏检测

## 模块架构

SCL-Core 遵循严格的分层架构，具有明确的依赖关系：

```
┌─────────────────────────────────────────┐
│         Python 绑定 (scl-py)            │
│      (NumPy/SciPy/AnnData 接口)        │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│              math/                      │
│    (统计、回归、近似)                   │
│         统计计算                        │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│            kernel/                      │
│   计算算子 (400+ 函数)                  │
│  normalize, neighbors, leiden, 等       │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│          threading/                     │
│  工作窃取调度器、parallel_for           │
│      线程池和工作空间                   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│             core/                       │
│  类型、稀疏、SIMD、注册表、内存         │
│           基础层                        │
└─────────────────────────────────────────┘
```

**依赖规则：**
- 低层永不依赖上层
- 每层只能使用下层的 API
- core/ 除 Highway（用于 SIMD）外零内部依赖

## 关键组件

### 核心类型

```cpp
namespace scl {
    // 可配置的基础类型
    using Real = /* float32 | float64 | float16 */;
    using Index = /* int16 | int32 | int64 */;
    using Size = size_t;
}
```

类型配置是编译时的，支持针对特定精度要求的全程序优化。

### 稀疏矩阵基础设施

使用指针数组的不连续存储以实现灵活的内存管理：

```cpp
template <typename T, bool IsCSR>
struct Sparse {
    using Pointer = T*;
    
    Pointer* data_ptrs_;      // 每行/列数据指针
    Pointer* indices_ptrs_;   // 每行/列索引指针
    Index* lengths_;          // 每行/列长度
    
    Index rows_, cols_, nnz_;
};
```

**内存布局：**

```
行 0: data_ptrs_[0] → [v0, v1, v2]    indices_ptrs_[0] → [c0, c1, c2]
行 1: data_ptrs_[1] → [v3, v4]        indices_ptrs_[1] → [c3, c4]
...
```

这种设计允许：
- 可配置块大小的块分配
- 引用计数的缓冲区共享
- 零拷贝视图和切片
- 通过注册表的高效 Python 集成

### 注册表系统

用于生命周期管理的集中式内存跟踪：

```cpp
class Registry {
    // 简单分配跟踪
    template <typename T>
    T* new_array(size_t count);
    void register_ptr(void* ptr, size_t bytes, AllocType type);
    void unregister_ptr(void* ptr);
    
    // 共享内存的引用计数缓冲区
    BufferID register_buffer_with_aliases(
        void* real_ptr, size_t byte_size,
        std::span<void*> alias_ptrs, AllocType type);
    
    // 内省
    bool is_registered(void* ptr) const;
    size_t get_total_bytes() const;
};
```

注册表使用分片设计以最小化并行工作负载中的锁争用。

### SIMD 抽象

通过 Google Highway 的可移植 SIMD：

```cpp
namespace scl::simd {
    using Tag = hn::ScalableTag<Real>;
    
    // 操作编译为本机内联函数
    auto Load(Tag d, const T* ptr);
    auto Add(Vec a, Vec b);
    auto MulAdd(Vec a, Vec b, Vec c);  // FMA
    auto SumOfLanes(Tag d, Vec v);
}
```

Highway 提供基于目标架构的编译时分派到最优 SIMD 指令（AVX2、AVX-512、NEON 等）。

## 性能策略

### 1. SIMD 优化

**多累加器模式：**

现代 CPU 具有多个 FMA 单元，延迟为 4-5 个周期。使用 4 个独立累加器隐藏此延迟并实现接近峰值吞吐量：

```cpp
auto v_sum0 = s::Zero(d), v_sum1 = s::Zero(d);
auto v_sum2 = s::Zero(d), v_sum3 = s::Zero(d);

for (; i + 4*lanes <= n; i += 4*lanes) {
    v_sum0 = s::Add(v_sum0, s::Load(d, data + i + 0*lanes));
    v_sum1 = s::Add(v_sum1, s::Load(d, data + i + 1*lanes));
    v_sum2 = s::Add(v_sum2, s::Load(d, data + i + 2*lanes));
    v_sum3 = s::Add(v_sum3, s::Load(d, data + i + 3*lanes));
}

auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
```

**融合操作：**

组合相关计算以最小化内存流量：

```cpp
// 单次遍历计算均值和方差
auto v_sum = s::Zero(d), v_sumsq = s::Zero(d);
for (size_t i = 0; i < n; i += lanes) {
    auto v = s::Load(d, data + i);
    v_sum = s::Add(v_sum, v);
    v_sumsq = s::MulAdd(v, v, v_sumsq);  // FMA
}
```

详见[设计原则](design-principles.md)获取全面的 SIMD 模式。

### 2. 并行处理

**工作窃取调度器：**

SCL-Core 使用自定义工作窃取线程池，它：
- 根据问题大小自动并行化
- 动态平衡线程间负载
- 最小化同步开销
- 支持嵌套并行

**每线程工作空间：**

通过为每个线程提供自己的工作空间避免同步：

```cpp
WorkspacePool<Real> pool(num_threads, workspace_size);

parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    auto* workspace = pool.get(thread_rank);  // 无锁
    // 使用工作空间进行临时存储
});
```

详见[线程文档](/cpp/threading/)。

### 3. 内存管理

**SIMD 的对齐分配：**

SIMD 代码中使用的所有数组都是 64 字节对齐以获得最优性能：

```cpp
Real* data = scl::memory::aligned_alloc<Real>(count, 64);
```

**块分配策略：**

稀疏矩阵行/列按块分配（4KB-1MB）以平衡：
- 内存重用（更大的块减少开销）
- 部分释放（更小的块允许细粒度释放）
- 并行性（多个块支持并发操作）

详见[内存模型](memory-model.md)获取全面的内存管理模式。

## 文档系统

SCL-Core 使用双文件文档方法，将实现与规范分离：

### 实现文件 (.hpp)

包含实际代码和最少的内联注释：

```cpp
// scl/kernel/normalize.hpp
template <CSRLike MatrixT>
void normalize_rows_inplace(MatrixT& matrix, NormMode mode, Real eps) {
    // 使用 Kahan 求和以提高数值稳定性
    parallel_for(Size(0), matrix.rows(), [&](Index i) {
        // 实现...
    });
}
```

### API 文档文件 (.h)

具有结构化部分的全面文档：

```cpp
// scl/kernel/normalize.h
/* -----------------------------------------------------------------------------
 * FUNCTION: normalize_rows_inplace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     将 CSR 矩阵的每一行就地归一化为单位范数。
 *
 * PARAMETERS:
 *     matrix   [in,out] 可变 CSR 矩阵，就地修改
 *     mode     [in]     归一化的范数类型
 *     epsilon  [in]     防止除零的小常数
 *
 * PRECONDITIONS:
 *     - matrix 必须是有效的 CSR 格式
 *     - matrix 值必须可变
 *
 * POSTCONDITIONS:
 *     - 每行具有单位范数（如果原始范数 > epsilon）
 *     - 矩阵结构不变
 *
 * COMPLEXITY:
 *     时间:  O(nnz)
 *     空间: O(1) 辅助
 *
 * THREAD SAFETY:
 *     安全 - 按行并行化
 * -------------------------------------------------------------------------- */
template <CSRLike MatrixT>
void normalize_rows_inplace(
    MatrixT& matrix,               // CSR 矩阵，就地修改
    NormMode mode,                 // 归一化类型: L1, L2, 或 Max
    Real epsilon = 1e-12           // 零范数阈值 (默认: 1e-12)
);
```

**关键文档部分：**
- SUMMARY: 单行目的
- PARAMETERS: 每个参数带 [in]、[out] 或 [in,out] 标签
- PRECONDITIONS: 调用前的要求
- POSTCONDITIONS: 执行后的保证
- MUTABILITY: INPLACE、CONST 或 ALLOCATES
- ALGORITHM: 逐步描述
- COMPLEXITY: 时间和空间分析
- THREAD SAFETY: 安全、不安全或条件性
- NUMERICAL NOTES: 精度和稳定性考虑

详见[文档标准](documentation-standard.md)获取完整指南。

## 外部依赖

**设计上的最小化：**

- **Google Highway**: SIMD 抽象（仅头文件，强制）
- **C++17 标准库**: 热路径中的最小使用

无其他依赖。这保持编译快速并消除版本冲突。

## 构建系统

基于 CMake 的构建，包括：
- **编译器检测**: 为 GCC/Clang/MSVC 自动选择最优标志
- **SIMD 目标选择**: 基于架构的 AVX2、AVX-512、NEON
- **LTO/IPO**: 跨模块内联的链接时优化
- **统一构建**: 可选，以加快编译

配置：

```cmake
# 配置精度
set(SCL_REAL_TYPE "float32")  # 或 float64, float16
set(SCL_INDEX_TYPE "int32")   # 或 int16, int64

# 启用功能
set(SCL_ENABLE_SIMD ON)
set(SCL_ENABLE_OPENMP ON)
```

## 性能特征

**典型性能指标：**

- **归一化**: 5-10 GB/s 内存带宽（接近硬件峰值）
- **KNN**: 对于稀疏矩阵比 Scanpy/sklearn 快 2-3 倍
- **统计检验**: 对于逐基因操作比 scipy.stats 快 10-100 倍
- **Leiden 聚类**: 与原始 igraph 实现竞争

性能取决于架构。SCL-Core 针对以下优化：
- 具有 AVX2 或 AVX-512 的 x86_64
- 具有 NEON 的 ARM（Apple Silicon、AWS Graviton）

## 开发原则

### 1. 性能优先

每个架构决策都优先考虑性能。当有疑问时，测量并优化热路径。

### 2. 零隐藏成本

如果操作分配内存、抛出异常或执行 I/O，它必须在 API 中明确。

### 3. 通过类型保证正确性

使用类型系统强制正确性：
- 类型级别的 CSR/CSC 区分
- 只读操作的 const 正确性
- 约束模板参数的概念

### 4. 可测试设计

所有内核都是纯函数，易于独立测试。

## 下一步

- [设计原则](design-principles.md) - 深入优化策略
- [文档标准](documentation-standard.md) - 编写 API 文档
- [内存模型](memory-model.md) - 注册表和生命周期管理
- [模块结构](module-structure.md) - 详细的模块依赖

---

::: tip 性能哲学
最好的优化是你不需要的优化。SCL-Core 的架构在问题发生之前就消除了开销，而不是之后。每个抽象都设计为完全编译掉。
:::

