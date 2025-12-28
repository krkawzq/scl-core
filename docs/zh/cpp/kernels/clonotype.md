# 克隆型分析

用于免疫库研究的 TCR/BCR 克隆多样性和扩增分析。

## 概览

克隆型分析内核提供：

- **克隆多样性** - Shannon 熵、Simpson 指数、Gini 系数
- **克隆扩增** - 识别扩增的克隆
- **表型关联** - 将克隆与表型关联
- **免疫库分析** - 量化克隆分布

## 克隆多样性

### clonal_diversity

计算克隆多样性指标（Shannon 熵、Simpson 指数、Gini）：

```cpp
#include "scl/kernel/clonotype.hpp"

Array<Index> clone_ids = /* ... */;  // 每个细胞的克隆 ID [n_cells]
Size n_cells = clone_ids.len;

Real shannon_entropy, simpson_index, gini_coeff;

scl::kernel::clonotype::clonal_diversity(
    clone_ids, n_cells,
    shannon_entropy, simpson_index, gini_coeff);

// shannon_entropy: H = -sum(p_i * log(p_i))
// simpson_index: 1 - sum(p_i^2)
// gini_coeff: 不平等度量
```

**参数：**
- `clone_ids`: 每个细胞的克隆 ID，大小 = n_cells
- `n_cells`: 细胞数量
- `shannon_entropy`: 输出 Shannon 熵
- `simpson_index`: 输出 Simpson 多样性指数
- `gini_coeff`: 输出 Gini 系数

**后置条件：**
- `shannon_entropy` = H = -sum(p_i * log(p_i))，其中 p_i = 克隆 i 的频率
- `simpson_index` = 1 - sum(p_i^2)，范围 [0, 1)
- `gini_coeff` = 不平等度量，范围 [0, 1]

**多样性指标：**
- **Shannon 熵**：更高 = 更多样，对稀有克隆敏感
- **Simpson 指数**：更高 = 更多样，权重常见克隆
- **Gini 系数**：更高 = 更不平等（少数大克隆），更低 = 更均匀

**复杂度：**
- 时间：O(n_cells) - 单次遍历数据
- 空间：O(n_clones) 辅助空间用于频率计数

**线程安全：**
- 安全 - 并行化处理

**用例：**
- 库多样性量化
- 跨样本比较多样性
- 跟踪多样性随时间变化
- 评估免疫反应幅度

## 克隆扩增

### clone_expansion

基于大小阈值识别扩增的克隆：

```cpp
Array<Index> clone_ids = /* ... */;
Size n_cells = clone_ids.len;

Index max_results = 100;
Array<Index> expanded_clones(max_results);  // 预分配
Array<Size> clone_sizes(max_results);       // 预分配

Index n_expanded = scl::kernel::clonotype::clone_expansion(
    clone_ids, n_cells,
    expanded_clones.ptr, clone_sizes.ptr,
    min_size = 5,                           // 最小克隆大小
    max_results                             // 最大返回数
);

// expanded_clones[0..n_expanded-1] 包含克隆 ID
// clone_sizes[0..n_expanded-1] 包含对应大小
```

**参数：**
- `clone_ids`: 克隆 ID，大小 = n_cells
- `n_cells`: 细胞数量
- `expanded_clones`: 输出扩增的克隆 ID，必须预分配
- `clone_sizes`: 输出克隆大小，必须预分配
- `min_size`: 扩增的最小大小（默认：2）
- `max_results`: 最大返回结果数

**返回：**
- 找到的扩增克隆数量（可能少于 max_results）

**后置条件：**
- `expanded_clones[0..n_expanded-1]` 包含大小 >= min_size 的克隆 ID
- `clone_sizes[0..n_expanded-1]` 包含对应大小
- 结果按大小排序（最大优先）

**复杂度：**
- 时间：O(n_cells + n_clones) - 计数和排序
- 空间：O(n_clones) 辅助空间用于频率计数

**线程安全：**
- 安全 - 无共享可变状态

**用例：**
- 识别优势克隆
- 查找扩增的 T 细胞/B 细胞克隆
- 跟踪克隆扩增
- 过滤顶级克隆进行分析

## 表型关联

### clone_phenotype_association

测试克隆与表型之间的关联：

```cpp
Array<Index> clone_ids = /* ... */;      // 克隆 ID [n_cells]
Array<Index> phenotypes = /* ... */;      // 表型标签 [n_cells]
Size n_clones = /* 唯一克隆数量 */;
Size n_phenotypes = /* 唯一表型数量 */;

Array<Real> association(n_clones * n_phenotypes);  // 预分配

scl::kernel::clonotype::clone_phenotype_association(
    clone_ids, phenotypes,
    n_clones, n_phenotypes,
    association.ptr);

// association[c * n_phenotypes + p] 包含关联强度
```

**参数：**
- `clone_ids`: 克隆 ID，大小 = n_cells
- `phenotypes`: 表型标签，大小 = n_cells
- `n_clones`: 唯一克隆数量
- `n_phenotypes`: 唯一表型数量
- `association_matrix`: 输出关联矩阵，必须预分配，大小 = n_clones × n_phenotypes

**后置条件：**
- `association_matrix[c * n_phenotypes + p]` 包含关联强度
- 关联通常表示条件概率或富集
- 更高值表示更强的克隆-表型关联

**算法：**
- 计数克隆和表型的共现
- 计算关联指标（例如，条件概率）
- 按克隆归一化跨表型

**复杂度：**
- 时间：O(n_cells + n_clones * n_phenotypes)
- 空间：O(n_clones * n_phenotypes) 辅助空间

**线程安全：**
- 安全 - 并行化处理

**用例：**
- 将 TCR/BCR 克隆与细胞类型关联
- 识别表型特异性克隆
- 克隆功能预测
- 免疫反应表征

## 配置

### 默认参数

```cpp
namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Index NO_CLONE = -1;
    constexpr Size MIN_CLONE_SIZE = 2;
}
```

**NO_CLONE 常量：**
- 特殊值，表示细胞没有克隆分配
- 用于未分配的细胞
- 从多样性计算中排除

**最小克隆大小：**
- 克隆扩增的默认阈值
- 根据数据集大小和生物学调整

## 示例

### 多样性比较

```cpp
#include "scl/kernel/clonotype.hpp"

// 样本 1
Array<Index> clones1 = /* ... */;
Real shannon1, simpson1, gini1;
scl::kernel::clonotype::clonal_diversity(
    clones1, clones1.len, shannon1, simpson1, gini1);

// 样本 2
Array<Index> clones2 = /* ... */;
Real shannon2, simpson2, gini2;
scl::kernel::clonotype::clonal_diversity(
    clones2, clones2.len, shannon2, simpson2, gini2);

std::cout << "样本 1 - Shannon: " << shannon1
          << ", Simpson: " << simpson1 << "\n";
std::cout << "样本 2 - Shannon: " << shannon2
          << ", Simpson: " << simpson2 << "\n";
```

### 查找扩增克隆

```cpp
Array<Index> clone_ids = /* ... */;
Size n_cells = clone_ids.len;

Index max_top = 20;
Array<Index> top_clones(max_top);
Array<Size> sizes(max_top);

Index n_top = scl::kernel::clonotype::clone_expansion(
    clone_ids, n_cells,
    top_clones.ptr, sizes.ptr,
    min_size = 10,                           // 至少 10 个细胞
    max_top
);

std::cout << "前 " << n_top << " 个扩增克隆：\n";
for (Index i = 0; i < n_top; ++i) {
    std::cout << "克隆 " << top_clones[i]
              << ": " << sizes[i] << " 个细胞\n";
}
```

### 克隆-细胞类型关联

```cpp
Array<Index> clone_ids = /* ... */;
Array<Index> cell_types = /* ... */;  // 细胞类型标签

Size n_clones = /* ... */;
Size n_types = /* ... */;

Array<Real> association(n_clones * n_types);
scl::kernel::clonotype::clone_phenotype_association(
    clone_ids, cell_types,
    n_clones, n_types,
    association.ptr);

// 查找与特定细胞类型强关联的克隆
Index target_type = 5;  // 例如，CD8+ T 细胞
Real threshold = 0.5;

for (Index c = 0; c < n_clones; ++c) {
    Real strength = association[c * n_types + target_type];
    if (strength > threshold) {
        std::cout << "克隆 " << c
                  << " 与类型 " << target_type << " 强关联"
                  << " (强度: " << strength << ")\n";
    }
}
```

---

::: tip 多样性指标
一起使用多个多样性指标：Shannon 熵用于整体多样性，Simpson 指数用于常见克隆多样性，Gini 系数用于分布不平等。
:::

