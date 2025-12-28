# 矩阵重排序

稀疏矩阵重排序和排列操作。

## 概览

重排序内核提供：

- **行重排序** - 对稀疏矩阵的行进行排列
- **列重排序** - 对稀疏矩阵的列进行排列
- **并行处理** - 针对大型矩阵的高效重排序
- **内存高效** - 针对稀疏结构优化

## 行重排序

### reorder_rows

根据排列对稀疏矩阵的行进行重排序：

```cpp
#include "scl/kernel/reorder.hpp"

Sparse<Real, true> matrix = /* ... */;      // 输入 CSR 矩阵
Array<const Index> permutation = /* ... */;  // 行排列 [n_rows]
Index n_rows = matrix.rows();

Sparse<Real, true> output;
output = Sparse<Real, true>::create(n_rows, matrix.cols(), /* 估算 nnz */);

scl::kernel::reorder::reorder_rows(matrix, permutation, n_rows, output);
```

**参数：**
- `matrix`: 输入稀疏矩阵（CSR 格式）
- `permutation`: 行排列数组，大小 = `n_rows`
- `n_rows`: 行数
- `output`: 输出重排序后的矩阵（必须预分配）

**后置条件：**
- `output[i]` 包含输入矩阵中位置 `permutation[i]` 的行
- 每行的矩阵结构（列、值）保留
- 输入矩阵不变

**算法：**
- 对每行并行：
  1. 从输入矩阵位置 `permutation[i]` 读取行
  2. 将值和索引复制到输出的位置 `i`
  3. 更新 indptr 数组

**复杂度：**
- 时间：O(nnz) - 与非零元素数量线性相关
- 空间：O(nnz) 辅助空间用于输出矩阵

**线程安全：**
- 安全 - 跨行并行化
- 每个线程处理独立的行
- 无共享可变状态

**用例：**
- 聚类结果可视化
- 按元数据排序（细胞类型、批次等）
- 为下游分析组织数据
- 算法的矩阵变换

## 列重排序

### reorder_columns

根据排列对稀疏矩阵的列进行重排序：

```cpp
Sparse<Real, true> matrix = /* ... */;      // 输入矩阵
Array<const Index> permutation = /* ... */;  // 列排列 [n_cols]
Index n_cols = matrix.cols();

Sparse<Real, true> output;
output = Sparse<Real, true>::create(matrix.rows(), n_cols, /* 估算 nnz */);

scl::kernel::reorder::reorder_columns(matrix, permutation, n_cols, output);
```

**参数：**
- `matrix`: 输入稀疏矩阵（CSR 或 CSC 格式）
- `permutation`: 列排列数组，大小 = `n_cols`
- `n_cols`: 列数
- `output`: 输出重排序后的矩阵（必须预分配）

**后置条件：**
- 输出的列按排列顺序
- 输出中的列 `j` 对应于输入中的列 `permutation[j]`
- 行结构保留
- 输入矩阵不变

**算法：**
- 对于 CSR：需要值重映射（列改变索引）
- 对于 CSC：直接对列切片进行行排列
- 跨主维度并行处理

**复杂度：**
- 时间：O(nnz) - 与非零元素数量线性相关
- 空间：O(nnz) 辅助空间用于输出矩阵

**线程安全：**
- 安全 - 并行处理
- 无共享可变状态

**用例：**
- 基因排序（按方差、表达水平）
- 特征选择结果组织
- 列基础算法的矩阵变换
- 数据可视化（排序的热图）

## 示例

### 按聚类分配排序细胞

```cpp
#include "scl/kernel/reorder.hpp"
#include "scl/core/sparse.hpp"

Sparse<Real, true> expression = /* ... */;
Array<Index> cluster_labels = /* ... */;  // 每个细胞的聚类分配

// 创建排列：按聚类标签排序
Index n_cells = expression.rows();
std::vector<Index> cell_indices(n_cells);
std::iota(cell_indices.begin(), cell_indices.end(), 0);

// 按聚类标签对索引排序
std::sort(cell_indices.begin(), cell_indices.end(),
    [&](Index i, Index j) { return cluster_labels[i] < cluster_labels[j]; });

// 创建排列数组
Array<Index> permutation(cell_indices.data(), n_cells);

// 重排序行（细胞）
Sparse<Real, true> sorted_expression;
sorted_expression = Sparse<Real, true>::create(n_cells, expression.cols(),
                                               expression.nnz());
scl::kernel::reorder::reorder_rows(expression, permutation, n_cells,
                                   sorted_expression);
```

### 按方差排序基因

```cpp
// 计算基因方差
Array<Real> gene_vars(expression.cols());
// ... 计算方差 ...

// 创建排列：按方差排序（降序）
std::vector<Index> gene_indices(expression.cols());
std::iota(gene_indices.begin(), gene_indices.end(), 0);
std::sort(gene_indices.begin(), gene_indices.end(),
    [&](Index i, Index j) { return gene_vars[i] > gene_vars[j]; });

Array<Index> permutation(gene_indices.data(), expression.cols());

// 重排序列（基因）
Sparse<Real, true> sorted_by_variance;
sorted_by_variance = Sparse<Real, true>::create(expression.rows(),
                                                expression.cols(),
                                                expression.nnz());
scl::kernel::reorder::reorder_columns(expression, permutation,
                                     expression.cols(),
                                     sorted_by_variance);
```

### 批次排序

```cpp
Array<Index> batch_labels = /* ... */;  // 每个细胞的批次 ID

// 按批次排序细胞
std::vector<Index> cell_indices(n_cells);
std::iota(cell_indices.begin(), cell_indices.end(), 0);
std::sort(cell_indices.begin(), cell_indices.end(),
    [&](Index i, Index j) { return batch_labels[i] < batch_labels[j]; });

Array<Index> permutation(cell_indices.data(), n_cells);

Sparse<Real, true> batch_ordered;
batch_ordered = Sparse<Real, true>::create(n_cells, expression.cols(),
                                           expression.nnz());
scl::kernel::reorder::reorder_rows(expression, permutation, n_cells,
                                   batch_ordered);
```

## 性能考虑

### 并行化

- 操作跨主维度并行化
- 阈值：`PARALLEL_THRESHOLD = 256` 行/列
- 小矩阵可能使用顺序处理以获得更好的缓存行为

### 内存效率

- 输出矩阵必须预分配
- 估算 nnz 进行分配（通常与输入相同）
- 保留稀疏结构（不密集化）

### 排列验证

- 调用者必须确保排列有效：
  - 所有值在范围 [0, n-1] 内
  - 无重复
  - 行排列：大小 = n_rows
  - 列排列：大小 = n_cols

---

::: tip 预分配
始终使用适当的维度预分配输出矩阵。在大多数情况下，nnz 将与输入相同，但考虑排列可能影响结构的边缘情况。
:::

