# lineage.hpp

> scl/kernel/lineage.hpp · 谱系追踪与命运映射

## 概述

本文件提供谱系追踪和命运映射分析功能。它计算克隆与细胞类型之间的耦合矩阵，并量化克隆对不同细胞类型的命运偏向。

**头文件**: `#include "scl/kernel/lineage.hpp"`

---

## 主要 API

### lineage_coupling

::: source_code file="scl/kernel/lineage.hpp" symbol="lineage_coupling" collapsed
:::

**算法说明**

计算克隆与细胞类型之间的耦合矩阵，表示每个克隆中属于每种细胞类型的比例：

1. 将耦合矩阵初始化为零
2. 对于每个细胞：
   - 获取克隆 ID 和细胞类型
   - 原子地增加 coupling[clone_id * n_types + cell_type]
3. 将每行（克隆）归一化为总和为 1.0
4. 结果: coupling[c * n_types + t] = P(类型 t | 克隆 c)

耦合矩阵表示给定克隆的细胞类型的条件概率分布。

**边界条件**

- **空输入**: 返回零耦合矩阵
- **单个克隆**: 所有细胞在一个克隆中，该克隆的耦合总和为 1.0
- **单个细胞类型**: 所有细胞为相同类型，所有克隆的耦合为 1.0
- **无细胞的克隆**: 行总和为 0.0（实践中不应发生）
- **无效克隆 ID**: 负数或超出范围的 ID 导致未定义行为

**数据保证（前置条件）**

- `clone_ids` 必须具有长度 == n_cells（从数组大小隐式）
- `cell_types` 必须具有长度 == n_cells
- `coupling_matrix` 必须具有容量 >= n_clones * n_types
- 克隆 ID 应在范围 [0, n_clones) 内
- 细胞类型 ID 应在范围 [0, n_types) 内
- 数组必须有效（非空指针）

**复杂度分析**

- **时间**: O(n_cells) - 单次遍历所有细胞
- **空间**: O(n_clones * n_types) 辅助空间 - 耦合矩阵存储

**示例**

```cpp
#include "scl/kernel/lineage.hpp"

// 准备数据
Array<Index> clone_ids = /* 克隆 ID [n_cells] */;
Array<Index> cell_types = /* 细胞类型标签 [n_cells] */;
Size n_clones = 10;  // 唯一克隆数量
Size n_types = 5;    // 唯一细胞类型数量

// 预分配耦合矩阵
Array<Real> coupling(n_clones * n_types);

// 计算耦合
scl::kernel::lineage::lineage_coupling(
    clone_ids, cell_types,
    coupling.ptr, n_clones, n_types
);

// 分析克隆到类型的分布
for (Index c = 0; c < n_clones; ++c) {
    std::cout << "克隆 " << c << ":\n";
    for (Index t = 0; t < n_types; ++t) {
        Real frac = coupling[c * n_types + t];
        if (frac > 0.1) {  // 至少 10%
            std::cout << "  类型 " << t << ": " << frac * 100 << "%\n";
        }
    }
}
```

---

### fate_bias

::: source_code file="scl/kernel/lineage.hpp" symbol="fate_bias" collapsed
:::

**算法说明**

计算克隆对不同细胞类型的命运偏向，衡量相对于背景的富集：

1. 计算耦合矩阵（如 lineage_coupling）
2. 计算背景类型分布: P(类型) = count(类型) / n_cells
3. 对于每个克隆 c 和类型 t：
   - 计算偏向 = coupling[c, t] / P(类型 t)
   - 更高的值表示更强的偏好
4. 结果: fate_bias[c * n_types + t] = 克隆 c 对类型 t 的富集

命运偏向量化克隆相对于总体群体对细胞类型的偏好程度。

**边界条件**

- **空输入**: 返回零偏向矩阵
- **均匀分布**: 所有克隆的偏向 = 1.0（无偏好）
- **克隆在类型中不存在**: 偏向 = 0.0
- **背景中罕见类型**: 可能产生非常高的偏向值
- **零背景概率**: 根据实现返回 0.0 或 NaN

**数据保证（前置条件）**

- `clone_ids` 必须具有长度 == n_cells
- `cell_types` 必须具有长度 == n_cells
- `fate_bias` 必须具有容量 >= n_clones * n_types
- 克隆 ID 应在范围 [0, n_clones) 内
- 细胞类型 ID 应在范围 [0, n_types) 内
- 数组必须有效（非空指针）

**复杂度分析**

- **时间**: O(n_cells + n_clones * n_types) - 遍历细胞，然后计算所有克隆-类型对的偏向
- **空间**: O(n_clones * n_types) 辅助空间 - 耦合矩阵和偏向矩阵

**示例**

```cpp
#include "scl/kernel/lineage.hpp"

// 准备数据
Array<Index> clone_ids = /* ... */;
Array<Index> cell_types = /* ... */;
Size n_clones = 10;
Size n_types = 5;

// 预分配偏向矩阵
Array<Real> bias(n_clones * n_types);

// 计算命运偏向
scl::kernel::lineage::fate_bias(
    clone_ids, cell_types,
    n_clones, n_types,
    bias.ptr
);

// 找到对特定类型有强偏向的克隆
Index target_type = 3;  // 例如，神经元
Real bias_threshold = 2.0;  // 2 倍富集

std::vector<Index> biased_clones;
for (Index c = 0; c < n_clones; ++c) {
    Real b = bias[c * n_types + target_type];
    if (b > bias_threshold) {
        biased_clones.push_back(c);
    }
}

std::cout << "找到 " << biased_clones.size()
          << " 个对类型 " << target_type << " 有偏向的克隆\n";

// 识别多能克隆（存在于多种类型中）
Real multipotency_threshold = 0.1;
std::vector<Index> multipotent_clones;
for (Index c = 0; c < n_clones; ++c) {
    Size n_types_present = 0;
    for (Index t = 0; t < n_types; ++t) {
        Real coupling_val = /* 从耦合矩阵获取 */;
        if (coupling_val > multipotency_threshold) {
            n_types_present++;
        }
    }
    if (n_types_present >= 3) {
        multipotent_clones.push_back(c);
    }
}
```

---

## 配置

### 默认参数

```cpp
namespace scl::kernel::lineage::config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Index NO_PARENT = -1;
    constexpr Size MIN_CLONE_SIZE = 2;
}
```

**NO_PARENT 常量**: 用于没有父细胞的特殊值（谱系树的根），用于谱系树构建。

**最小克隆大小**: 用于过滤小克隆以确保统计可靠性。

---

## 注意事项

**耦合 vs. 偏向**: 使用耦合获取原始克隆到类型的分布。使用命运偏向进行相对于背景的富集分析。偏向更适合识别命运偏好和比较克隆。

**多能性**: 如果克隆产生多种细胞类型，则它是多能的。使用耦合矩阵识别存在于 >= 3 种类型且具有显著比例 (>10%) 的克隆。

**统计考虑**: 小克隆可能具有不可靠的耦合估计。在分析前考虑过滤大小 < MIN_CLONE_SIZE 的克隆。

**线程安全**: 使用原子操作进行并行累加，对并发执行安全。

---

## 相关内容

- [Subpopulation](/zh/cpp/kernels/subpopulation) - 亚群分析
- [Clustering](/zh/cpp/kernels/clustering) - 细胞类型聚类
