# 置换检验

用于统计显著性的置换检验和多重比较校正内核。

## 概述

置换模块提供：

- **通用置换检验** - 用户自定义检验统计量
- **相关性置换检验** - Pearson 相关性显著性
- **FDR 校正** - Benjamini-Hochberg 和 Benjamini-Yekutieli
- **FWER 校正** - Bonferroni 和 Holm-Bonferroni
- **批量检验** - 多个特征的并行置换检验

## 通用置换检验

### permutation_test

使用用户自定义检验统计量的通用置换检验：

```cpp
#include "scl/kernel/permutation.hpp"

Array<Index> labels = /* ... */;  // 要置换的组标签
Real observed_statistic = /* ... */;

Real p_value = scl::kernel::permutation::permutation_test(
    [&](Array<const Index> permuted_labels) -> Real {
        // 在置换标签上计算检验统计量
        return compute_statistic(permuted_labels);
    },
    labels,
    observed_statistic,
    config::DEFAULT_N_PERMUTATIONS,  // n_permutations = 1000
    true,                            // two_sided = true
    42                               // seed
);
```

**参数：**
- `compute_statistic`: 可调用函数，接受置换标签并返回检验统计量
- `labels`: 要置换的组标签，大小 = n
- `observed_statistic`: 观察到的检验统计量值
- `n_permutations`: 置换次数（默认：1000）
- `two_sided`: 使用双侧检验（默认：true）
- `seed`: 用于可重现性的随机种子

**返回：** 范围 [1/(n_perm+1), 1] 内的 P 值

**算法：**
1. 将标签复制到置换缓冲区
2. 对于每次置换：
   - 使用 Fisher-Yates 算法打乱
   - 在打乱的标签上计算检验统计量
   - 存储在零分布中
3. 从零分布计算 p 值

**复杂度：**
- 时间：O(n_permutations * (n + compute_statistic 的成本))
- 空间：O(n_permutations + n)

**使用场景：**
- 自定义检验统计量
- 非参数假设检验
- 当分布假设未知时

## 相关性置换检验

### permutation_correlation_test

Pearson 相关性显著性的置换检验：

```cpp
Array<const Real> x = /* ... */;
Array<const Real> y = /* ... */;
Real observed_correlation = /* ... */;

Real p_value = scl::kernel::permutation::permutation_correlation_test(
    x,
    y,
    observed_correlation,
    config::DEFAULT_N_PERMUTATIONS,
    42  // seed
);
```

**参数：**
- `x`: 第一个变量，大小 = n
- `y`: 第二个变量，大小 = n
- `observed_correlation`: 观察到的 Pearson 相关性
- `n_permutations`: 置换次数
- `seed`: 随机种子

**返回：** 相关性检验的双侧 p 值

**算法：**
1. 预计算 x 的均值和标准差（在置换中保持不变）
2. 对于每次置换：
   - 打乱索引
   - 使用置换的 y 计算相关性
   - 存储在零分布中
3. 比较 |observed| 与 |null| 进行双侧检验

**复杂度：**
- 时间：O(n_permutations * n)
- 空间：O(n_permutations + n)

**使用场景：**
- 相关性显著性检验
- 非参数相关性分析
- 当正态性假设失败时

## FDR 校正

### fdr_correction_bh

Benjamini-Hochberg FDR 校正用于多重检验：

```cpp
Array<const Real> p_values = /* ... */;
Array<Real> q_values(n);  // 预分配

scl::kernel::permutation::fdr_correction_bh(
    p_values,
    q_values
);
```

**参数：**
- `p_values`: 原始 p 值，大小 = n
- `q_values`: 输出的 FDR 调整 q 值，大小 = n，预分配

**后置条件：**
- `q_values[i]` 包含检验 i 的 FDR 调整 p 值
- `q_values[i] >= p_values[i]`
- `q_values[i]` 在 [0, 1] 范围内

**算法：**
1. 对 p 值排序并获取顺序
2. 从最大到最小排名：
   - adjusted = p * n / rank
   - q = min(adjusted, cumulative_min)
3. 映射回原始顺序

**复杂度：**
- 时间：O(n log n) 用于排序
- 空间：O(n) 用于排序索引

**参考：**
Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing.

**使用场景：**
- 多重假设检验
- 基因表达分析
- 当许多检验预期为真时

### fdr_correction_by

Benjamini-Yekutieli FDR 校正用于相关检验：

```cpp
Array<Real> q_values(n);  // 预分配

scl::kernel::permutation::fdr_correction_by(
    p_values,
    q_values
);
```

**参数：**
- `p_values`: 原始 p 值，大小 = n
- `q_values`: 输出的调整 q 值，大小 = n，预分配

**后置条件：**
- `q_values` 在任意依赖下控制 FDR
- 比 BH 校正更保守

**算法：**
与 BH 相同，但乘以调和和 c_n = 1 + 1/2 + ... + 1/n

**参考：**
Benjamini, Y. and Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency.

**使用场景：**
- 相关检验（例如，相关基因）
- 当不能假设检验独立性时
- 更保守的 FDR 控制

## FWER 校正

### bonferroni_correction

Bonferroni 校正用于多重检验（FWER 控制）：

```cpp
Array<Real> adjusted_p_values(n);  // 预分配

scl::kernel::permutation::bonferroni_correction(
    p_values,
    adjusted_p_values
);
```

**参数：**
- `p_values`: 原始 p 值，大小 = n
- `adjusted_p_values`: Bonferroni 调整 p 值，预分配

**后置条件：**
- `adjusted_p_values[i] = min(p_values[i] * n, 1)`
- 如果在 adjusted_p < alpha 时拒绝，则在 alpha 处控制 FWER

**复杂度：**
- 时间：O(n)
- 空间：O(1) 辅助空间

**使用场景：**
- 族错误率控制
- 非常保守的校正
- 当少数检验预期为真时

### holm_correction

Holm-Bonferroni 逐步下降校正（比 Bonferroni 更不保守）：

```cpp
Array<Real> adjusted_p_values(n);  // 预分配

scl::kernel::permutation::holm_correction(
    p_values,
    adjusted_p_values
);
```

**参数：**
- `p_values`: 原始 p 值，大小 = n
- `adjusted_p_values`: Holm 调整 p 值，预分配

**后置条件：**
- 在 alpha 处控制 FWER
- 比 Bonferroni 更强大

**算法：**
1. 按升序对 p 值排序
2. 对于 i = 1 到 n：
   - adjusted = p_(i) * (n - i + 1)
   - result_(i) = max(adjusted, result_(i-1))
3. 映射回原始顺序

**复杂度：**
- 时间：O(n log n)
- 空间：O(n)

**参考：**
Holm, S. (1979). A simple sequentially rejective multiple test procedure.

**使用场景：**
- 比 Bonferroni 更好的 FWER 控制
- 逐步下降程序
- 当 Bonferroni 过于保守时

## 工具函数

### count_significant

统计低于显著性阈值的 p 值数量：

```cpp
Size n_sig = scl::kernel::permutation::count_significant(
    p_values,
    Real(0.05)  // alpha
);
```

**参数：**
- `p_values`: 要检验的 p 值，大小 = n
- `alpha`: 显著性阈值（默认：0.05）

**返回：** p 值 < alpha 的数量

**复杂度：**
- 时间：O(n)
- 空间：O(1)

### get_significant_indices

获取显著检验的索引：

```cpp
Array<Index> indices(max_results);
Size n_significant;

scl::kernel::permutation::get_significant_indices(
    p_values,
    Real(0.05),  // alpha
    indices,
    n_significant
);
```

**参数：**
- `p_values`: P 值，大小 = n
- `alpha`: 显著性阈值
- `indices`: 显著索引的输出缓冲区，预分配
- `n_significant`: 找到的显著检验数量

**后置条件：**
- `indices[0..n_significant-1]` 包含 p < alpha 的索引

**复杂度：**
- 时间：O(n)
- 空间：O(1) 辅助空间

## 批量置换检验

### batch_permutation_test

多个特征（稀疏矩阵的行）的并行置换检验：

```cpp
Sparse<Real, true> matrix = /* ... */;  // 特征 x 样本
Array<const Index> group_labels = /* ... */;  // 0/1 组分配
Array<Real> p_values(matrix.rows());  // 预分配

scl::kernel::permutation::batch_permutation_test(
    matrix,
    group_labels,
    config::DEFAULT_N_PERMUTATIONS,
    p_values,
    42  // seed
);
```

**参数：**
- `matrix`: CSR 稀疏矩阵，形状 (n_features x n_samples)
- `group_labels`: 组分配（0/1），大小 = n_samples
- `n_permutations`: 每个特征的置换次数
- `p_values`: 输出的 p 值，大小 = n_features，预分配
- `seed`: 随机种子

**后置条件：**
- `p_values[i]` 包含行 i 的双侧 p 值
- 没有非零值的行得到 p 值 = 1.0

**算法：**
在特征上并行：
1. 计算行的观察均值差
2. 对于每次置换：
   - 打乱组标签（线程本地 RNG）
   - 计算置换均值差
3. 计算双侧 p 值

**复杂度：**
- 时间：O(n_features * n_permutations * avg_nnz_per_row)
- 空间：O(n_threads * (n_permutations + n_samples))

**使用场景：**
- 差异表达分析
- 多特征检验
- 大规模置换检验

## 配置

`scl::kernel::permutation::config` 中的默认参数：

```cpp
namespace config {
    constexpr Size DEFAULT_N_PERMUTATIONS = 1000;
    constexpr Size MIN_PERMUTATIONS = 100;
    constexpr Size MAX_PERMUTATIONS = 100000;
    constexpr Size PARALLEL_THRESHOLD = 500;
}
```

## 性能考虑

### 并行化

- `batch_permutation_test`: 在特征上并行
- 使用 WorkspacePool 进行线程本地缓冲区
- 线程本地 RNG 用于独立置换

### 内存效率

- 预分配的输出缓冲区
- 在并行循环中重用工作空间
- 最少的临时分配

## 最佳实践

### 1. 选择适当的置换次数

```cpp
// 对于 0.001 的 p 值分辨率，使用至少 1000 次置换
Size n_perm = 1000;

// 对于更高精度，使用更多
Size n_perm = 10000;

Real p_value = scl::kernel::permutation::permutation_test(
    compute_statistic, labels, observed, n_perm
);
```

### 2. 对多重检验使用 FDR

```cpp
// 批量检验后
Array<Real> p_values(n_features);
scl::kernel::permutation::batch_permutation_test(/* ... */, p_values);

// 应用 FDR 校正
Array<Real> q_values(n_features);
scl::kernel::permutation::fdr_correction_bh(p_values, q_values);

// 按 FDR 过滤
Size n_sig = scl::kernel::permutation::count_significant(q_values, 0.05);
```

### 3. 使用 Holm 进行 FWER 控制

```cpp
// 当族错误率至关重要时
Array<Real> adjusted_p(n_tests);
scl::kernel::permutation::holm_correction(p_values, adjusted_p);

// 比 Bonferroni 更强大
```

---

::: tip 置换次数
对于 0.001 的 p 值分辨率，使用至少 1000 次置换。更多置换会增加精度，但也会增加计算时间。
:::

::: warning 多重检验
在检验多个假设时，始终应用 FDR 或 FWER 校正以控制错误发现。
:::

