# C++ 开发者指南

欢迎来到 SCL-Core C++ 开发者指南！本文档面向想要为 SCL-Core 做贡献、扩展其功能或将其集成到自己的 C++ 项目中的开发者。

## 概览

SCL-Core 是一个基于现代 C++20 构建的高性能生物算子库。它提供：

- **零开销抽象** - 实现最大性能
- **SIMD 加速** - 计算内核
- **默认并行** - 操作默认并行化
- **内存高效** - 稀疏矩阵基础设施
- **稳定的 C-ABI** - 用于语言绑定

## 快速导航

### 贡献者

- [快速开始](/zh/cpp/getting-started/) - 设置开发环境
- [从源码构建](/zh/cpp/getting-started/building) - 编译和测试库
- [贡献指南](/zh/cpp/getting-started/contributing) - 代码标准和流程

### 库使用者

- [架构概览](/zh/cpp/architecture/) - 理解设计哲学
- [核心模块](/zh/cpp/core/) - 类型、稀疏矩阵、内存管理
- [并行处理](/zh/cpp/threading/) - 并行处理基础设施
- [内核函数](/zh/cpp/kernels/) - 计算算子

### 高级用户

- [API 参考](/zh/cpp/reference/) - 完整函数参考
- [设计原则](/zh/cpp/architecture/design-principles) - 性能优化策略
- [内存模型](/zh/cpp/architecture/memory-model) - Registry 和生命周期管理

## 核心特性

### 零开销性能

所有抽象都编译为最优机器代码：

```cpp
// 高级 API
scl::kernel::normalize::normalize_rows_inplace(matrix, NormMode::L2);

// 编译为紧凑的 SIMD 循环，无开销
```

### SIMD 加速

通过 Highway 库内置 SIMD 支持：

```cpp
namespace s = scl::simd;
const s::Tag d;

auto v_sum = s::Zero(d);
for (size_t i = 0; i < n; i += s::lanes()) {
    auto v = s::Load(d, data + i);
    v_sum = s::Add(v_sum, v);
}
```

### 默认并行

自动并行化，最优工作分配：

```cpp
scl::threading::parallel_for(Size(0), n_rows, [&](size_t i) {
    // 并行处理第 i 行
    process_row(matrix, i);
});
```

### 内存效率

高级稀疏矩阵基础设施：

```cpp
// 非连续存储，灵活的内存管理
scl::Sparse<Real, true> matrix = 
    scl::Sparse<Real, true>::create(rows, cols, nnz_per_row);

// 基于 Registry 的生命周期管理
auto arrays = scl::kernel::sparse::to_contiguous_arrays(matrix);
// 数组自动注册和跟踪
```

## 模块组织

```
scl/
├── core/           # 核心类型、稀疏矩阵、SIMD、内存
├── threading/      # 并行处理基础设施
├── kernel/         # 计算算子 (80+ 文件, 400+ 函数)
├── math/           # 统计函数和回归
├── mmap/           # 大数据集的内存映射数组
└── io/             # I/O 工具
```

## 文档结构

本指南分为几个部分：

### 快速开始
学习如何设置开发环境、构建库和贡献代码。

### 架构设计
理解使 SCL-Core 快速的设计原则、模块结构和内存模型。

### 核心模块
深入了解基础构建块：类型、稀疏矩阵、SIMD 和内存管理。

### 并行处理
学习并行处理基础设施以及如何编写线程安全代码。

### 内核函数
探索 400+ 个按功能组织的计算算子。

### 参考
从源代码文档中提取的完整 API 参考。

## 设计哲学

SCL-Core 遵循以下核心原则：

1. **零开销抽象** - 高级 API 无运行时成本
2. **数据导向设计** - 优化缓存局部性和内存带宽
3. **显式资源管理** - 无隐藏分配或隐式成本
4. **编译时多态** - 模板优于虚函数
5. **文档即代码** - `.h` 文件包含全面的 API 文档

## 获取帮助

- **问题报告**: 在 [GitHub Issues](https://github.com/krkawzq/scl-core/issues) 上报告错误
- **讨论**: 在 [GitHub Discussions](https://github.com/krkawzq/scl-core/discussions) 中提问
- **贡献**: 查看 [贡献指南](/zh/cpp/getting-started/contributing)

## 下一步

- **SCL-Core 新手？** 从 [快速开始](/zh/cpp/getting-started/) 开始
- **想要贡献？** 阅读 [贡献指南](/zh/cpp/getting-started/contributing)
- **需要 API 文档？** 浏览 [核心模块](/zh/cpp/core/) 或 [内核函数](/zh/cpp/kernels/)
- **想了解设计？** 探索 [架构设计](/zh/cpp/architecture/)

---

::: tip 开发状态
SCL-Core 正在积极开发中，欢迎贡献！查看 [GitHub 仓库](https://github.com/krkawzq/scl-core) 获取最新更新。
:::

