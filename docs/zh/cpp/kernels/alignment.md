# 对齐

用于数据集整合的多模态数据对齐和批次整合内核。

## 概述

对齐模块提供：

- **锚点查找** - 查找数据集之间的锚点对（Seurat 风格）
- **MNN 校正** - 互最近邻批次校正
- **整合评分** - 数据集整合的质量评分

## 锚点查找

### find_anchors

查找两个数据集之间的锚点对（Seurat 风格整合）：

```cpp
#include "scl/kernel/alignment.hpp"

Sparse<Real, true> query_data = /* ... */;  // 查询表达 [n_query x n_genes]
Sparse<Real, true> reference_data = /* ... */;  // 参考表达 [n_ref x n_genes]
Index* anchor_pairs = /* 分配 max_anchors * 2 */;
Real* anchor_scores = /* 分配 max_anchors */;

Index n_anchors = scl::kernel::alignment::find_anchors(
    query_data,
    reference_data,
    n_query,
    n_ref,
    config::DEFAULT_K,  // k = 30
    anchor_pairs,
    anchor_scores,
    max_anchors
);
```

**参数：**
- `query_data`: 查询表达矩阵（细胞 x 基因，CSR）
- `reference_data`: 参考表达矩阵（细胞 x 基因，CSR）
- `n_query`: 查询细胞数量
- `n_ref`: 参考细胞数量
- `k`: 用于锚点评分的邻居数量
- `anchor_pairs`: 输出的锚点对 [max_anchors * 2]
- `anchor_scores`: 输出的锚点评分 [max_anchors]
- `max_anchors`: 最大锚点数量

**返回：** 找到的锚点数量

**后置条件：**
- `anchor_pairs[i * 2]` = 查询索引，`anchor_pairs[i * 2 + 1]` = 参考索引
- 锚点评分表示相似性强度

**算法：**
对于每个查询细胞并行：
1. 在参考中查找 k 个最近邻
2. 检查查询细胞是否在参考的 k 个最近邻中（MNN）
3. 基于互最近邻状态对锚点对评分
4. 按评分选择顶级锚点

**复杂度：**
- 时间：O(n_query * n_ref * log(n_ref))
- 空间：O(n_query * k) 辅助空间

**使用场景：**
- 数据集整合
- 批次校正
- 参考映射

## MNN 校正

### mnn_correction

应用互最近邻（MNN）批次校正：

```cpp
Sparse<Real, true> query_data = /* ... */;  // 原地修改
const Sparse<Real, true> reference_data = /* ... */;
const Index* mnn_pairs = /* ... */;  // MNN 对 [n_mnn * 2]
Index n_mnn = /* ... */;

scl::kernel::alignment::mnn_correction(
    query_data,
    reference_data,
    mnn_pairs,
    n_mnn,
    n_genes
);
```

**参数：**
- `query_data`: 查询表达矩阵，原地修改
- `reference_data`: 参考表达矩阵
- `mnn_pairs`: MNN 对 [n_mnn * 2]
- `n_mnn`: MNN 对数量
- `n_genes`: 基因数量

**后置条件：**
- `query_data` 向参考方向校正
- MNN 对指导校正方向

**可变性：**
INPLACE - 修改 query_data.values()

**算法：**
对于每个 MNN 对并行：
1. 计算查询与参考之间的差异向量
2. 对查询细胞表达应用校正
3. 在邻居之间平滑校正

**复杂度：**
- 时间：O(n_mnn * n_genes)
- 空间：O(n_genes) 辅助空间

**使用场景：**
- 批次效应校正
- 数据集协调
- 整合预处理

## 整合评分

### integration_score

计算数据集之间的整合质量评分：

```cpp
Real score;

scl::kernel::alignment::integration_score(
    query_data,
    reference_data,
    n_query,
    n_ref,
    config::DEFAULT_K,  // k = 30
    score
);
```

**参数：**
- `query_data`: 查询表达矩阵（细胞 x 基因，CSR）
- `reference_data`: 参考表达矩阵（细胞 x 基因，CSR）
- `n_query`: 查询细胞数量
- `n_ref`: 参考细胞数量
- `k`: 用于评分的邻居数量
- `score`: 输出的整合评分

**后置条件：**
- `score` 包含整合质量（越高越好）
- 评分基于数据集之间的邻居混合

**算法：**
1. 对于两个数据集中的每个细胞，查找 k 个最近邻
2. 统计来自相同 vs 不同数据集的邻居
3. 计算混合评分
4. 对所有细胞求平均

**复杂度：**
- 时间：O((n_query + n_ref) * k * log(n_ref))
- 空间：O(k) 辅助空间（每个细胞）

**使用场景：**
- 整合质量评估
- 批次校正评估
- 整合参数调整

## 配置

`scl::kernel::alignment::config` 中的默认参数：

```cpp
namespace config {
    constexpr Real EPSILON = 1e-10;
    constexpr Size DEFAULT_K = 30;
    constexpr Real ANCHOR_SCORE_THRESHOLD = 0.5;
    constexpr Size MAX_ANCHORS_PER_CELL = 10;
    constexpr Size PARALLEL_THRESHOLD = 32;
}
```

## 性能考虑

### 并行化

- `find_anchors`: 在查询细胞上并行
- `mnn_correction`: 在 MNN 对上并行
- `integration_score`: 在细胞上并行

### 内存效率

- 高效的稀疏矩阵访问
- 预分配的输出缓冲区
- 最少的临时分配

---

::: tip 锚点质量
按评分阈值（默认：0.5）过滤锚点，仅使用高质量锚点对进行校正。
:::

::: warning 整合评分
使用整合评分评估校正质量并调整参数（k、阈值）。
:::

