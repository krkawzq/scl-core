# 矩阵切片

沿主维或次维切片和过滤稀疏矩阵。

## 概述

切片操作提供：

- **主维切片** - 选择特定行（CSR）或列（CSC）
- **次维过滤** - 使用布尔掩码过滤列（CSR）或行（CSC）
- **高效检查** - 在分配前统计非零数
- **内存高效** - 两阶段方法（检查然后物化）

## 主维切片

### slice_primary

创建包含所选主维切片的新稀疏矩阵。

```cpp
#include "scl/kernel/slice.hpp"
#include "scl/core/sparse.hpp"

Sparse<Real, true> matrix = /* ... */;
Array<Index> keep_indices = /* ... */;  // 要保留的行的索引

auto result = scl::kernel::slice::slice_primary(matrix, keep_indices);
// result 仅包含选定的行
```

**参数：**
- `matrix` [in] - 源稀疏矩阵
- `keep_indices` [in] - 要保留的行（CSR）或列（CSC）的索引

**前置条件：**
- 所有索引在范围 [0, primary_dim) 内

**后置条件：**
- 结果仅包含选定的行/列
- 列/行索引不变（次维保留）
- 顺序匹配 keep_indices 顺序

**返回：**
包含选定切片的新稀疏矩阵

**算法：**
1. inspect_slice_primary 统计输出非零数
2. 分配输出数组
3. materialize_slice_primary 复制数据
4. 包装为新 Sparse 矩阵

**复杂度：**
- 时间: O(nnz_output / n_threads + n_keep)
- 空间: O(nnz_output) 用于结果

**线程安全：**
安全 - 使用并行物化

**使用场景：**
- 选择样本/细胞子集
- 按元数据过滤
- 创建训练/测试分割

### inspect_slice_primary

统计所选主维切片中的总非零数。

```cpp
Index nnz_output = scl::kernel::slice::inspect_slice_primary(
    matrix,
    keep_indices
);
// 返回选定切片中的总非零数
```

**参数：**
- `matrix` [in] - 要切片的稀疏矩阵
- `keep_indices` [in] - 要保留的主维元素的索引

**前置条件：**
- keep_indices 中的所有索引在范围 [0, primary_dim) 内

**后置条件：**
- 返回选定索引的行长度之和

**返回：**
选定切片中的总非零数

**算法：**
使用 parallel_reduce_nnz 对 keep_indices 进行并行归约

**复杂度：**
- 时间: O(n_keep / n_threads)
- 空间: O(n_threads) 用于部分和

**线程安全：**
安全 - 只读并行归约

**使用场景：**
- 预分配输出数组
- 估算内存需求

### materialize_slice_primary

将选定的主维切片复制到预分配的输出数组。

```cpp
Array<Real> out_data(nnz_output);
Array<Index> out_indices(nnz_output);
Array<Index> out_indptr(keep_indices.len + 1);

scl::kernel::slice::materialize_slice_primary(
    matrix,
    keep_indices,
    out_data,
    out_indices,
    out_indptr
);
```

**参数：**
- `matrix` [in] - 源稀疏矩阵
- `keep_indices` [in] - 要保留的行/列的索引
- `out_data` [out] - 输出值数组
- `out_indices` [out] - 输出列/行索引数组
- `out_indptr` [out] - 输出行/列指针数组

**前置条件：**
- out_data.len >= inspect_slice_primary 结果
- out_indices.len >= inspect_slice_primary 结果
- out_indptr.len >= keep_indices.len + 1

**后置条件：**
- out_data 包含按顺序复制的值
- out_indices 包含复制的索引（不变）
- out_indptr[i] = 第 i 个选定行的起始位置

**算法：**
1. 顺序扫描构建 out_indptr
2. 使用 fast_copy_with_prefetch 并行复制数据和索引

**复杂度：**
- 时间: O(nnz_output / n_threads + n_keep)
- 空间: O(1) 超出输出

**线程安全：**
安全 - 并行复制到不相交的输出区域

## 次维过滤

### filter_secondary

创建通过次维掩码过滤的新稀疏矩阵。

```cpp
Array<uint8_t> mask(secondary_dim);  // 1 = 保留, 0 = 移除
// ... 填充 mask ...

auto result = scl::kernel::slice::filter_secondary(matrix, mask);
// result 仅包含 mask[index] == 1 的元素
```

**参数：**
- `matrix` [in] - 源稀疏矩阵
- `mask` [in] - 列（CSR）或行（CSC）的布尔掩码

**前置条件：**
- mask.len >= secondary_dim
- mask 值为 0 或 1

**后置条件：**
- 结果 secondary_dim = mask 中 1 的计数
- 仅保留 mask[index] == 1 的元素
- 索引重新映射到紧凑范围 [0, new_secondary_dim)

**返回：**
具有过滤次维的新稀疏矩阵

**算法：**
1. 构建索引映射（旧 -> 新索引）
2. inspect_filter_secondary 统计输出非零数
3. 分配输出数组
4. materialize_filter_secondary 复制并重新映射

**复杂度：**
- 时间: O(nnz / n_threads + secondary_dim)
- 空间: O(nnz_output + secondary_dim)

**线程安全：**
安全 - 使用并行物化

**使用场景：**
- 选择特征/基因子集
- 按表达阈值过滤
- 特征选择

### inspect_filter_secondary

通过次维掩码过滤后统计非零数。

```cpp
Index nnz_output = scl::kernel::slice::inspect_filter_secondary(
    matrix,
    mask
);
// 返回 mask[index] == 1 的元素计数
```

**参数：**
- `matrix` [in] - 要过滤的稀疏矩阵
- `mask` [in] - 次维的布尔掩码（1 = 保留）

**前置条件：**
- mask.len >= secondary_dim
- mask 值为 0 或 1

**后置条件：**
- 返回 mask[index] == 1 的元素计数

**返回：**
过滤后的总非零数

**算法：**
使用 count_masked_fast（8 路展开）进行并行归约

**复杂度：**
- 时间: O(nnz / n_threads)
- 空间: O(n_threads) 用于部分和

**线程安全：**
安全 - 只读并行归约

### materialize_filter_secondary

将通过次维掩码的元素复制到预分配的输出。

```cpp
Array<Index> new_indices = /* 从 mask 构建 */;
Array<Real> out_data(nnz_output);
Array<Index> out_indices(nnz_output);
Array<Index> out_indptr(primary_dim + 1);

scl::kernel::slice::materialize_filter_secondary(
    matrix,
    mask,
    new_indices,
    out_data,
    out_indices,
    out_indptr
);
```

**参数：**
- `matrix` [in] - 源稀疏矩阵
- `mask` [in] - 次维的布尔掩码
- `new_indices` [in] - 从旧到新次维索引的映射
- `out_data` [out] - 输出值
- `out_indices` [out] - 输出索引（重新映射）
- `out_indptr` [out] - 输出行指针

**前置条件：**
- new_indices 通过 build_index_mapping 构建
- 输出数组按 inspect_filter_secondary 调整大小

**后置条件：**
- out_data 包含 mask[old_index] == 1 的值
- out_indices 包含通过 new_indices 重新映射的索引
- out_indptr 包含累积计数

**复杂度：**
- 时间: O(nnz / n_threads + primary_dim)
- 空间: O(1) 超出输出

**线程安全：**
安全 - 在主维上并行

## 示例

### 选择细胞

按索引选择特定细胞：

```cpp
Sparse<Real, true> expression = /* ... */;  // 细胞 x 基因
Array<Index> selected_cells = {0, 5, 10, 15, /* ... */};

auto subset = scl::kernel::slice::slice_primary(expression, selected_cells);
// subset 仅包含选定的细胞
```

### 过滤基因

按表达阈值过滤基因：

```cpp
Sparse<Real, true> expression = /* ... */;  // 细胞 x 基因
Array<uint8_t> gene_mask(expression.cols());

// 构建掩码：保留平均表达 > 阈值的基因
for (Index g = 0; g < expression.cols(); ++g) {
    Real mean_expr = /* 计算均值 */;
    gene_mask[g] = (mean_expr > threshold) ? 1 : 0;
}

auto filtered = scl::kernel::slice::filter_secondary(expression, gene_mask);
// filtered 仅包含高表达基因
```

### 两阶段方法

使用检查然后物化以提高内存效率：

```cpp
// 阶段 1：统计非零数
Index nnz = scl::kernel::slice::inspect_slice_primary(matrix, keep_indices);

// 阶段 2：分配和复制
Array<Real> out_data(nnz);
Array<Index> out_indices(nnz);
Array<Index> out_indptr(keep_indices.len + 1);

scl::kernel::slice::materialize_slice_primary(
    matrix, keep_indices, out_data, out_indices, out_indptr
);
```

## 性能

### 并行化

- 用于检查的并行归约
- 用于物化的并行复制
- 无同步开销

### SIMD 优化

- 8 路展开的掩码计数
- 复制循环中的预取
- 高效的内存访问模式

### 内存效率

- 两阶段方法减少内存使用
- 预分配输出数组
- 最小化中间分配

## 实现细节

### 掩码计数

使用 8 路标量展开来计数掩码元素：
- 间接访问 mask[indices[k]] 阻止 SIMD gather
- 8 路标量展开为此模式提供最佳 ILP

### 索引映射

从布尔掩码构建旧到新索引的映射：
- new_indices[i] = 如果 mask[i] == 1 则为新紧凑索引
- new_indices[i] = 如果 mask[i] == 0 则为 -1
- 返回 mask 中 1 的计数
