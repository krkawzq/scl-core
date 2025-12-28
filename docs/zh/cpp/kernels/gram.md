# Gram 矩阵

用于稀疏矩阵的高效 Gram 矩阵计算，采用自适应点积算法。

## 概览

Gram 矩阵内核提供：

- **Gram 矩阵** - 计算内积矩阵 G[i,j] = dot(row_i, row_j)
- **自适应算法** - 根据稀疏模式选择最优点积算法
- **高性能** - SIMD 优化和并行处理
- **对称计算** - 仅计算上三角，对称写入

## Gram 矩阵

### gram

计算稀疏矩阵行的 Gram 矩阵（内积矩阵）：

```cpp
#include "scl/kernel/gram.hpp"

Sparse<Real, true> matrix = /* ... */;  // 输入矩阵 [n_rows x n_cols]
Index n_rows = matrix.rows();
Array<Real> output(n_rows * n_rows);    // 预分配输出

scl::kernel::gram::gram(matrix, output);

// output[i * n_rows + j] = dot(row_i, row_j)
```

**参数：**
- `matrix`: 输入稀疏矩阵，形状 (n_rows, n_cols)
- `output`: 输出 Gram 矩阵，必须预分配，大小 = n_rows × n_rows

**后置条件：**
- `output` 是对称的：output[i,j] == output[j,i]
- 对角线：output[i,i] = 行 i 的平方 L2 范数
- output[i,j] = dot(row_i, row_j)
- 行主序布局：output[i * n_rows + j] = dot(row_i, row_j)

**算法：**
关键优化：
1. 对称计算：仅计算上三角
2. 对角线通过 vectorize::sum_squared（SIMD 优化）
3. 基于大小比的自适应稀疏点积选择：
   - ratio < 32：线性合并，8/4 路跳跃
   - ratio < 256：二分搜索，范围缩小
   - ratio >= 256：跳跃（指数）搜索
4. 在任何计算之前进行 O(1) 范围不相交检查

**复杂度：**
- 时间：O(n_rows^2 * avg_nnz_per_row / n_threads)
- 空间：O(1) 超出输出

**线程安全：**
- 安全 - 跨行并行，对称写入

**用例：**
- 核方法（RBF 核、多项式核）
- 相似性矩阵
- 主成分分析
- 距离计算

---

::: tip 自适应算法
实现根据向量对的稀疏比自动选择最优点积算法，确保在不同输入模式下的最佳性能。
:::

