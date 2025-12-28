# multiple_testing.hpp

> scl/kernel/multiple_testing.hpp · 控制错误发现率的多种多重检验校正方法

## 概述

本文件提供在进行多重假设检验时校正 p 值的各种方法。多重检验校正在同时检验多个假设时对于控制错误发现率（FDR）或族错误率（FWER）至关重要。

主要特性：
- Benjamini-Hochberg (BH) FDR 控制
- Bonferroni 和 Holm-Bonferroni FWER 控制
- Storey 的 q 值估计
- 使用核密度估计的局部 FDR
- 来自置换的经验 FDR
- 适用于依赖检验的 Benjamini-Yekutieli 校正

**头文件**: `#include "scl/kernel/multiple_testing.hpp"`

---

## 主要 API

### benjamini_hochberg

对 p 值应用 Benjamini-Hochberg FDR 校正。这是最常用的 FDR 控制方法。

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="benjamini_hochberg" collapsed
:::

**算法说明**

Benjamini-Hochberg 程序控制错误发现率：

1. 按升序对 p 值排序
2. 对于秩 i，计算调整后的 p = p_value[i] * n / i
3. 从右到左强制单调性：如果 adjusted[i] > adjusted[i+1]，则设置 adjusted[i] = adjusted[i+1]
4. 映射回原始顺序

这确保在被拒绝的假设中，错误发现的预期比例控制在指定的 FDR 水平。

**边界条件**

- **所有 p 值 > FDR 水平**：所有调整后的 p 值仍高于阈值，无发现
- **所有 p 值非常小**：如果检验高度显著，调整后的值可能仍然很小
- **p 值中的并列**：通过排序正确处理

**数据保证（前置条件）**

- `p_values.len == adjusted_p_values.len`
- 所有 p 值在 [0, 1] 范围内
- `fdr_level` 通常为 0.05

**复杂度分析**

- **时间**：排序 O(n log n)
- **空间**：排序索引的辅助空间 O(n)

**示例**

```cpp
#include "scl/kernel/multiple_testing.hpp"

Array<Real> p_values(n_tests);  // 来自检验的原始 p 值
Array<Real> adjusted(n_tests);  // 预分配的输出

// 应用 BH 校正，FDR 水平 0.05
scl::kernel::multiple_testing::benjamini_hochberg(
    p_values, adjusted, 0.05
);

// 查找显著检验（调整后的 p < 0.05）
Index indices[n_tests];
Size count;
scl::kernel::multiple_testing::significant_indices(
    adjusted, 0.05, indices, count
);
```

---

### storey_qvalue

使用 Storey 方法和 pi0（真零假设比例）估计来估计 q 值。

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="storey_qvalue" collapsed
:::

**算法说明**

Storey 的 q 值是比 BH 调整后的 p 值更强大的替代方法：

1. 使用 lambda 调优参数估计 pi0（真零假设比例）：
   pi0 = (# p 值 > lambda) / ((1 - lambda) * n)
2. 按升序对 p 值排序
3. 从右到左计算 q 值并保持单调性：
   q[i] = min(pi0 * p[i] * n / i, q[i+1])
4. 映射回原始顺序

lambda 参数（默认 0.5）控制 pi0 估计。当许多检验确实为零时，Storey 方法比 BH 更强大。

**边界条件**

- **所有 p 值都很小**：pi0 估计可能保守
- **很少零假设**：pi0 可能被高估，使 q 值更保守
- **Lambda 过高**：可能高估 pi0

**数据保证（前置条件）**

- `p_values.len == q_values.len`
- 所有 p 值在 [0, 1] 范围内
- `lambda` 在 (0, 1) 范围内，通常为 0.5

**复杂度分析**

- **时间**：排序 O(n log n)
- **空间**：辅助空间 O(n)

**示例**

```cpp
Array<Real> p_values(n_tests);
Array<Real> q_values(n_tests);

// 使用默认 lambda = 0.5 估计 q 值
scl::kernel::multiple_testing::storey_qvalue(p_values, q_values);

// 或使用自定义 lambda
Real lambda = 0.75;
scl::kernel::multiple_testing::storey_qvalue(p_values, q_values, lambda);

// q_values[i] 估计如果检验 i 被称为显著时的 FDR
```

---

### local_fdr

使用 z 分数上的核密度估计来估计局部错误发现率。

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="local_fdr" collapsed
:::

**算法说明**

局部 FDR 估计特定检验是错误发现的概率：

1. 将 p 值转换为 z 分数：z = -Phi^(-1)(p)
2. 使用核密度估计（KDE）估计密度 f(z)
3. 计算 f0(z)（零假设密度，标准正态 N(0,1)）
4. 从 z 分数分布的右尾估计 pi0
5. 计算 lfdr = pi0 * f0(z) / f(z)

局部 FDR 提供特定检验的 FDR 估计，而不是全局校正。

**边界条件**

- **所有 p 值非常大**：z 分数接近 0，lfdr 接近 pi0
- **非常小的 p 值**：z 分数非常大，lfdr 非常小
- **KDE 带宽**：影响密度估计质量

**数据保证（前置条件）**

- `p_values.len == lfdr.len`
- 所有 p 值在 [0, 1] 范围内

**复杂度分析**

- **时间**：KDE 估计 O(n^2)
- **空间**：辅助空间 O(n)

**示例**

```cpp
Array<Real> p_values(n_tests);
Array<Real> lfdr_values(n_tests);

scl::kernel::multiple_testing::local_fdr(p_values, lfdr_values);

// lfdr_values[i] 是检验 i 是错误发现的估计概率
// 较低的 lfdr 表示检验真正显著的可能性更高
```

---

### bonferroni

应用 Bonferroni 校正（乘以检验次数）。最保守的 FWER 控制。

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="bonferroni" collapsed
:::

**算法说明**

Bonferroni 校正控制族错误率（FWER）：

1. 将每个 p 值乘以 n（检验次数）
2. 钳制到 [0, 1]

这是最保守的校正方法，控制至少一个错误发现的概率。

**边界条件**

- **n * p > 1**：钳制到 1.0
- **非常大的 n**：大多数 p 值变为 1.0，非常保守

**数据保证（前置条件）**

- `p_values.len == adjusted_p_values.len`
- 所有 p 值在 [0, 1] 范围内

**复杂度分析**

- **时间**：使用 SIMD 优化操作 O(n)
- **空间**：辅助空间 O(1)

**示例**

```cpp
Array<Real> p_values(n_tests);
Array<Real> adjusted(n_tests);

scl::kernel::multiple_testing::bonferroni(p_values, adjusted);

// adjusted[i] = min(p_values[i] * n, 1.0)
```

---

### holm_bonferroni

应用 Holm-Bonferroni 逐步下降校正。比 Bonferroni 更强大，同时仍控制 FWER。

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="holm_bonferroni" collapsed
:::

**算法说明**

Holm-Bonferroni 是一个逐步下降程序：

1. 按升序对 p 值排序
2. 对于秩 i：adjusted = p_value[i] * (n - i + 1)
3. 从左到右强制单调性

比 Bonferroni 更强大，但仍控制 FWER。

**边界条件**

- **逐步下降性质**：从最小的 p 值开始顺序拒绝假设
- **比 Bonferroni 更强大**：特别是对于 p 值较小的检验

**数据保证（前置条件）**

- `p_values.len == adjusted_p_values.len`
- 所有 p 值在 [0, 1] 范围内

**复杂度分析**

- **时间**：排序 O(n log n)
- **空间**：辅助空间 O(n)

**示例**

```cpp
Array<Real> p_values(n_tests);
Array<Real> adjusted(n_tests);

scl::kernel::multiple_testing::holm_bonferroni(p_values, adjusted);

// 比 Bonferroni 更强大，仍控制 FWER
```

---

### hochberg

应用 Hochberg 逐步上升校正。比 Holm 更强大，同时控制 FWER。

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="hochberg" collapsed
:::

**算法说明**

Hochberg 程序是一个逐步上升方法：

1. 按升序对 p 值排序
2. 对于秩 i：adjusted = p_value[i] * (n - i + 1)
3. 从右到左强制单调性

比 Bonferroni 和 Holm 都更强大，在独立性下控制 FWER。

**边界条件**

- **逐步上升性质**：比逐步下降（Holm）更激进
- **独立性假设**：在依赖下可能不控制 FWER

**数据保证（前置条件）**

- `p_values.len == adjusted_p_values.len`
- 所有 p 值在 [0, 1] 范围内

**复杂度分析**

- **时间**：排序 O(n log n)
- **空间**：辅助空间 O(n)

**示例**

```cpp
Array<Real> p_values(n_tests);
Array<Real> adjusted(n_tests);

scl::kernel::multiple_testing::hochberg(p_values, adjusted);

// 在独立性下最强大的 FWER 控制
```

---

### benjamini_yekutieli

应用 Benjamini-Yekutieli FDR 校正。在任意依赖下工作。

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="benjamini_yekutieli" collapsed
:::

**算法说明**

类似于 BH，但使用校正因子 C(n) = sum(1/i) for i=1..n 来处理依赖：

1. 计算校正因子 C(n)
2. 对 p 值排序
3. 使用校正因子应用类似 BH 的程序

比 BH 更保守，但在任意依赖结构下工作。

**边界条件**

- **独立检验**：比标准 BH 略保守
- **依赖检验**：与标准 BH 不同，正确控制 FDR

**数据保证（前置条件）**

- `p_values.len == adjusted_p_values.len`
- 所有 p 值在 [0, 1] 范围内

**复杂度分析**

- **时间**：排序 O(n log n)
- **空间**：辅助空间 O(n)

**示例**

```cpp
Array<Real> p_values(n_tests);
Array<Real> adjusted(n_tests);

scl::kernel::multiple_testing::benjamini_yekutieli(p_values, adjusted);

// 在依赖下工作的保守 FDR 控制
```

---

### empirical_fdr

使用基于置换的经验零分布估计 FDR。

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="empirical_fdr" collapsed
:::

**算法说明**

从置换检验结果计算 FDR：

1. 对于每个检验 i，计算置换分数 >= 观察分数[i] 的置换次数
2. 计算 FDR[i] = (permutation_count + 1) / (n_permutations + 1)

这提供了无需分布假设的经验 FDR 估计。

**边界条件**

- **没有置换超过观察值**：FDR = 1/(n_perm+1)
- **所有置换都超过观察值**：FDR = 1.0
- **置换次数少**：估计不太可靠

**数据保证（前置条件）**

- `observed_scores.len == fdr.len`
- 所有 permuted_scores 数组的长度与 observed_scores 相同

**复杂度分析**

- **时间**：O(n_tests * n_permutations)，跨检验并行化
- **空间**：辅助空间 O(1)

**示例**

```cpp
Array<Real> observed(n_tests);
std::vector<Array<Real>> permuted(n_permutations);
for (auto& p : permuted) {
    p = Array<Real>(n_tests);
}

// 用置换结果填充...

Array<Real> fdr(n_tests);
scl::kernel::multiple_testing::empirical_fdr(observed, permuted, fdr);

// fdr[i] 是检验 i 的经验 FDR 估计
```

---

## 工具函数

### significant_indices

获取 p 值低于阈值的检验索引。

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="significant_indices" collapsed
:::

**复杂度**

- 时间：O(n)
- 空间：辅助空间 O(1)

---

### neglog10_pvalues

计算 p 值的负对数（以 10 为底）用于可视化。

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="neglog10_pvalues" collapsed
:::

**复杂度**

- 时间：O(n)
- 空间：辅助空间 O(1)

---

### fisher_combine

使用 Fisher 方法组合 p 值。返回卡方检验统计量。

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="fisher_combine" collapsed
:::

**复杂度**

- 时间：O(n)
- 空间：辅助空间 O(1)

---

## 配置

`scl::kernel::multiple_testing::config` 中的默认配置值：

- `DEFAULT_FDR_LEVEL = 0.05`
- `DEFAULT_LAMBDA = 0.5`（用于 Storey 方法）
- `MIN_PVALUE = 1e-300`
- `MAX_PVALUE = 1.0`
- `SPLINE_KNOTS = 10`
- `MIN_TESTS_FOR_STOREY = 100`

## 相关内容

- [MWU 检验](/zh/cpp/kernels/mwu) - 产生 p 值的 Mann-Whitney U 检验
- [T 检验](/zh/cpp/kernels/ttest) - 产生 p 值的 T 检验

