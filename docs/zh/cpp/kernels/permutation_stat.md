# permutation_stat.hpp

> scl/kernel/stat/permutation_stat.hpp · 带排序重用优化的优化置换检验

## 概述

本文件提供优化的置换检验，用于比较两组。关键创新是在置换之间重用排序数据结构，避免每次置换重新排序，实现显著加速。

**头文件**: `#include "scl/kernel/stat/permutation_stat.hpp"`

---

## 主要 API

### batch_permutation_reuse_sort

::: source_code file="scl/kernel/stat/permutation_stat.hpp" symbol="batch_permutation_reuse_sort" collapsed
:::

**算法说明**

通过重用排序数据结构优化的批量置换检验：

1. 对于每个特征并行处理：
   - 提取带索引的非零值
   - 使用 argsort 排序一次（保留到 group_ids 的索引映射）
   - 使用原始 group_ids 计算观测统计量：
     - MWU: 从排序秩计算 Mann-Whitney U 统计量
     - MeanDiff: 均值差（组0 - 组1）
     - KS: Kolmogorov-Smirnov D 统计量（未来）
   - 对于每次置换：
     - 使用 Fisher-Yates 打乱 group_ids（O(n) 时间）
     - 使用排序数据 + 打乱的组重新计算统计量：
       - 遍历排序值
       - 根据打乱的 group_ids 分配组
       - 无需重新排序即可计算统计量
     - 每 100 次置换进行自适应早停检查：
       - 如果 p < 0.001 或 p > 0.5，早停
   - 计算双侧 p 值：
     - P 值 = (|stat_perm| >= |stat_obs| 的计数 + 1) / (n_perms + 1)

2. **关键优化**:
   - 标准方法: 每次置换排序数据 = O(P * n log n)
   - 此方法: 排序一次，置换组 = O(n log n + P * n)
   - 加速因子: 对于大 n 约为 log(n)

3. 使用带 jump() 的 Xoshiro256++ PRNG 进行并行独立流

**边界条件**

- **组 0 或组 1 为空**: 抛出 ArgumentError
- **n_permutations < 100**: 使用最小 100 次置换
- **n_permutations > 100000**: 限制为最大 100000
- **所有值相同**: 所有置换统计量 = 观测值，p 值 = 1.0
- **组间无重叠**: p 值接近 0
- **早停**: 如果 p < 0.001 或 p > 0.5，可能使用少于 n_permutations 的次数

**数据保证（前置条件）**

- 两组（0 和 1）必须至少有一个成员
- 输出数组必须具有大小 >= matrix.primary_dim()
- `n_permutations` 应在范围 [100, 100000] 内（如果超出则限制）
- `group_ids` 必须仅包含 0 或 1 值
- 矩阵必须是有效的 CSR 或 CSC 格式

**复杂度分析**

- **时间**: O(features * (nnz * log(nnz) + n_permutations * nnz))
  - 排序一次: O(nnz * log(nnz))
  - 每次置换: O(nnz)（打乱 + 统计量计算）
- **空间**: O(threads * (max_row_length + n_samples + n_permutations))
  - 排序数据、打乱的组、置换统计量

**示例**

```cpp
#include "scl/kernel/stat/permutation_stat.hpp"

// 准备数据
Sparse<Real, true> matrix = /* 特征 x 样本 */;
Array<int32_t> group_ids = /* 二值组分配 (0 或 1) */;
Size n_permutations = 10000;

// 预分配输出
Size n_features = matrix.rows();
Array<Real> p_values(n_features);

// 使用 MWU 统计量计算置换检验
scl::kernel::stat::permutation_stat::batch_permutation_reuse_sort(
    matrix, group_ids, n_permutations,
    p_values,
    stat_type = scl::kernel::stat::permutation_stat::PermStatType::MWU,
    seed = 42
);

// 解释结果
for (Size i = 0; i < n_features; ++i) {
    if (p_values[i] < 0.05) {
        std::cout << "特征 " << i 
                  << ": p = " << p_values[i]
                  << " (显著)\n";
    }
}

// 与均值差统计量比较
Array<Real> p_values_mean(n_features);
scl::kernel::stat::permutation_stat::batch_permutation_reuse_sort(
    matrix, group_ids, n_permutations,
    p_values_mean,
    stat_type = scl::kernel::stat::permutation_stat::PermStatType::MeanDiff,
    seed = 42
);
```

---

### permutation_test_single

::: source_code file="scl/kernel/stat/permutation_stat.hpp" symbol="permutation_test_single" collapsed
:::

**算法说明**

带排序重用优化的单特征置换检验：

1. 与 batch_permutation_reuse_sort 相同的算法，但针对单特征
2. 提取特征值
3. 排序一次，然后为每次置换打乱组
4. 计算经验双侧 p 值

**边界条件**

- **组 0 或组 1 为空**: 抛出 ArgumentError
- **n_permutations < 100**: 使用最小 100 次置换
- **所有值相同**: p 值 = 1.0

**数据保证（前置条件）**

- `values.len == group_ids.len`
- 两组必须至少有一个成员
- `n_permutations` 在范围 [100, 100000] 内

**复杂度分析**

- **时间**: O(n * log(n) + n_permutations * n)
- **空间**: O(n + n_permutations)

**示例**

```cpp
#include "scl/kernel/stat/permutation_stat.hpp"

// 单特征检验
Array<Real> values = /* 特征值 */;
Array<int32_t> group_ids = /* 二值组分配 */;
Size n_permutations = 10000;

Real p_value = scl::kernel::stat::permutation_stat::permutation_test_single(
    values, group_ids, n_permutations,
    stat_type = scl::kernel::stat::permutation_stat::PermStatType::MWU,
    seed = 42
);

std::cout << "P 值: " << p_value << "\n";
```

---

## 统计量类型

### PermStatType

支持的统计量类型枚举：

- **MWU**: Mann-Whitney U 统计量（非参数秩检验）
- **MeanDiff**: 均值差（组0 - 组1，类似 t 检验）
- **KS**: Kolmogorov-Smirnov D 统计量（未来实现）

---

## 注意事项

**何时使用**: 置换检验适用于：
- 需要精确 p 值（无分布假设）
- 小样本量
- 非正态数据
- 可以使用任何统计量（不限于标准检验）

**优势**:
- **精确**: 无分布假设
- **灵活**: 适用于任何检验统计量
- **稳健**: 对任何样本量有效
- **优化**: 排序重用提供 log(n) 加速

**性能优化**:
- 标准方法: O(P * n log n) - 每次置换排序
- 此方法: O(n log n + P * n) - 排序一次，置换组
- 加速: 对于大 n 约为 log(n)
- 示例: n=1000, log(n)≈10, 10 倍加速

**自适应早停**:
- 每 100 次置换检查一次
- 如果 p < 0.001（非常显著）或 p > 0.5（不显著）则早停
- 减少明显情况的计算

**随机数生成**:
- 使用 Xoshiro256++ PRNG（快速、高质量）
- jump() 方法用于并行独立流
- Lemire 的几乎无除法的有界随机用于打乱

**线程安全**: 使用线程局部工作空间和 RNG 在特征上并行处理，对并发执行安全。

**与参数检验的比较**:
- **置换检验**: 精确，无假设，统计量灵活
- **t 检验**: 假设正态性，更快，灵活性较低
- **Mann-Whitney U**: 非参数，更快，灵活性较低

---

## 相关内容

- [Mann-Whitney U](/zh/cpp/kernels/mwu) - 非参数秩检验
- [T-test](/zh/cpp/kernels/ttest) - 参数均值比较
- [KS Test](/zh/cpp/kernels/ks) - 分布比较

