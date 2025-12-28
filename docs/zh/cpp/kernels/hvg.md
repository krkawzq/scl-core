# hvg.hpp

> scl/kernel/hvg.hpp · 高变基因选择内核

## 概述

本文件为单细胞 RNA-seq 分析提供高效的高变基因（HVG）选择方法。HVG 选择是识别具有高生物变异性的基因用于下游分析的关键预处理步骤。

本文件提供：
- 基于离散度的基因选择（方差/均值比）
- 方差稳定化变换（VST）方法
- SIMD 加速计算
- 用于高效 top-k 选择的部分排序

**头文件**: `#include "scl/kernel/hvg.hpp"`

---

## 主要 API

### select_by_dispersion

::: source_code file="scl/kernel/hvg.hpp" symbol="select_by_dispersion" collapsed
:::

**算法说明**

通过离散度（方差/均值比）选择高变基因：

1. **计算矩**：对每个基因并行计算均值和方差
2. **计算离散度**：dispersion[g] = var[g] / mean[g]
3. **选择 top k**：使用部分排序选择 n_top 个离散度最高的基因

**边界条件**

- 零均值基因：mean <= epsilon 的基因离散度 = 0（被排除）
- 常数值基因：零方差的基因离散度 = 0
- n_top > n_genes：限制为 n_genes

**数据保证（前置条件）**

- `out_indices.len >= n_top`
- `out_mask.len >= n_genes`
- `out_dispersions.len >= n_genes`

**复杂度分析**

- **时间**：O(nnz + n_genes * log(n_top))
- **空间**：O(n_genes) 用于中间缓冲区

**示例**

```cpp
#include "scl/kernel/hvg.hpp"

Sparse<Real, true> expression = /* ... */;
Size n_top = 2000;

Array<Index> selected_indices(n_top);
Array<uint8_t> mask(n_genes);
Array<Real> dispersions(n_genes);

scl::kernel::hvg::select_by_dispersion(
    expression, n_top,
    selected_indices, mask, dispersions
);
```

---

### select_by_vst

::: source_code file="scl/kernel/hvg.hpp" symbol="select_by_vst" collapsed
:::

**算法说明**

使用方差稳定化变换（VST）方法选择高变基因：

1. **裁剪值**：将表达值裁剪到 clip_vals[g]
2. **计算裁剪后的矩**：计算裁剪后的均值和方差
3. **选择 top k**：选择 n_top 个裁剪方差最高的基因

**边界条件**

- 零裁剪值：clip_val[g] = 0 时，所有值裁剪为 0，方差 = 0

**数据保证（前置条件）**

- `clip_vals.len >= n_genes`
- 所有输出数组预分配

**复杂度分析**

- **时间**：O(nnz + n_genes * log(n_top))
- **空间**：O(n_genes)

---

## 工具函数

### detail::dispersion_simd

使用 SIMD 优化计算离散度。

::: source_code file="scl/kernel/hvg.hpp" symbol="detail::dispersion_simd" collapsed
:::

**复杂度**

- 时间：O(n)
- 空间：O(1)

---

### detail::normalize_dispersion_simd

在均值范围内对离散度进行 z 分数归一化。

::: source_code file="scl/kernel/hvg.hpp" symbol="detail::normalize_dispersion_simd" collapsed
:::

**复杂度**

- 时间：O(n)
- 空间：O(1)

---

### detail::select_top_k_partial

使用部分排序选择 top k 元素。

::: source_code file="scl/kernel/hvg.hpp" symbol="detail::select_top_k_partial" collapsed
:::

**复杂度**

- 时间：O(n + k log k)
- 空间：O(k)

---

### detail::compute_moments

计算每个基因的均值和方差。

::: source_code file="scl/kernel/hvg.hpp" symbol="detail::compute_moments" collapsed
:::

**复杂度**

- 时间：O(nnz)
- 空间：O(n_genes)

---

### detail::compute_clipped_moments

使用每基因值裁剪计算均值和方差。

::: source_code file="scl/kernel/hvg.hpp" symbol="detail::compute_clipped_moments" collapsed
:::

**复杂度**

- 时间：O(nnz)
- 空间：O(n_genes)

---

## 注意事项

**离散度 vs VST**：
- 离散度方法：简单的方差/均值比，快速有效
- VST 方法：在方差计算前裁剪高值，对异常值更稳健

**性能**：
- SIMD 加速均值和方差计算
- 部分排序用于高效 top-k 选择
- 按基因并行化

## 相关内容

- [Feature Selection](/zh/cpp/kernels/feature) - 其他特征选择方法
- [Statistics](/zh/cpp/kernels/statistics) - 统计操作
