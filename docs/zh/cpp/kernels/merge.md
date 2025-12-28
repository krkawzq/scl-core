# 矩阵合并

垂直和水平堆叠稀疏矩阵。

## 概述

合并操作提供：

- **垂直堆叠** - 沿主轴连接矩阵（CSR 为行，CSC 为列）
- **水平堆叠** - 沿次轴连接矩阵（CSR 为列，CSC 为行）
- **高效复制** - 带 SIMD 优化的并行内存操作
- **内存管理** - 灵活的块分配策略

## 垂直堆叠

### vstack

垂直堆叠两个稀疏矩阵（沿主轴连接）。

```cpp
#include "scl/kernel/merge.hpp"
#include "scl/core/sparse.hpp"

Sparse<Real, true> matrix1 = /* ... */;  // 第一个矩阵
Sparse<Real, true> matrix2 = /* ... */;  // 第二个矩阵

auto result = scl::kernel::merge::vstack(matrix1, matrix2);
// result 包含垂直堆叠的矩阵
```

**参数：**
- `matrix1` [in] - 第一个稀疏矩阵
- `matrix2` [in] - 第二个稀疏矩阵
- `strategy` [in] - 结果的块分配策略（默认：自适应）

**前置条件：**
- 对于 CSR：列可以不同（结果使用最大值）
- 对于 CSC：行可以不同（结果使用最大值）

**后置条件：**
- 结果 primary_dim = matrix1.primary_dim + matrix2.primary_dim
- 结果 secondary_dim = max(matrix1.secondary_dim, matrix2.secondary_dim)
- 行 0..n1-1 来自 matrix1，行 n1..n1+n2-1 来自 matrix2
- 索引不变（次维保留）

**返回：**
包含垂直堆叠数据的新稀疏矩阵

**算法：**
1. 计算结果的行长度
2. 使用组合结构分配结果矩阵
3. 并行复制 matrix1 的行到 result[0:n1]
4. 并行复制 matrix2 的行到 result[n1:n1+n2]

**复杂度：**
- 时间: O(nnz1 + nnz2)
- 空间: O(nnz1 + nnz2) 用于结果

**线程安全：**
安全 - 并行复制独立区域

**使用场景：**
- 合并具有相同特征的数据集
- 将新样本追加到现有矩阵
- 合并时间序列数据

## 水平堆叠

### hstack

水平堆叠两个稀疏矩阵（沿次轴连接）。

```cpp
auto result = scl::kernel::merge::hstack(matrix1, matrix2);
// result 包含水平堆叠的矩阵
```

**参数：**
- `matrix1` [in] - 第一个稀疏矩阵
- `matrix2` [in] - 第二个稀疏矩阵
- `strategy` [in] - 结果的块分配策略（默认：自适应）

**前置条件：**
- matrix1.primary_dim == matrix2.primary_dim（必须匹配）

**后置条件：**
- 结果 primary_dim = matrix1.primary_dim（不变）
- 结果 secondary_dim = matrix1.secondary_dim + matrix2.secondary_dim
- 对于每一行：[matrix1 列 | 带偏移的 matrix2 列]
- matrix2 索引偏移 matrix1.secondary_dim

**返回：**
包含水平堆叠数据的新稀疏矩阵

**算法：**
1. 验证主维匹配
2. 计算组合的行长度
3. 分配结果矩阵
4. 并行处理行：
   - 复制 matrix1 的值和索引
   - 复制 matrix2 的值
   - 向 matrix2 索引添加偏移（SIMD 优化）

**复杂度：**
- 时间: O(nnz1 + nnz2)
- 空间: O(nnz1 + nnz2) 用于结果

**线程安全：**
安全 - 在独立行上并行

**抛出：**
`DimensionError` - 如果主维不匹配

**使用场景：**
- 合并特征集
- 连接来自不同批次的基因表达
- 合并具有相同样本的矩阵

## 示例

### 合并数据集

合并具有相同基因的两个表达矩阵：

```cpp
Sparse<Real, true> batch1 = /* ... */;  // 细胞 x 基因
Sparse<Real, true> batch2 = /* ... */;  // 细胞 x 基因

// 垂直堆叠（合并细胞）
auto combined = scl::kernel::merge::vstack(batch1, batch2);
// combined 有 (batch1.rows() + batch2.rows()) 行
```

### 连接特征

合并具有相同细胞但不同特征的矩阵：

```cpp
Sparse<Real, true> rna = /* ... */;    // 细胞 x RNA 基因
Sparse<Real, true> protein = /* ... */; // 细胞 x 蛋白质

// 水平堆叠（合并特征）
auto multiome = scl::kernel::merge::hstack(rna, protein);
// multiome 有 rna.rows() 行和 (rna.cols() + protein.cols()) 列
```

### 内存策略

为大型矩阵选择分配策略：

```cpp
// 使用自适应策略（默认）
auto result1 = scl::kernel::merge::vstack(m1, m2);

// 使用特定策略
auto result2 = scl::kernel::merge::vstack(
    m1, m2,
    BlockStrategy::contiguous()  // 强制连续分配
);
```

## 性能

### 并行化

- 大型块的并行内存复制
- 独立行处理
- 无同步开销

### SIMD 优化

- SIMD 优化的索引偏移加法
- 复制循环中的预取
- 高效的内存访问模式

### 内存效率

- 自适应块分配
- 最小化中间分配
- 高效的稀疏矩阵构建

## 实现细节

### 索引偏移加法

对于水平堆叠，matrix2 的索引通过 SIMD 优化的加法偏移 matrix1.secondary_dim：

```cpp
// offset == 0: 直接 memcpy（提前退出）
// 否则：2 路 SIMD 展开循环
```

### 并行内存复制

大型内存块使用并行复制：
- count < chunk_size: 单次 memcpy
- 否则：并行处理块，带预取
