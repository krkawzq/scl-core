# 细胞生态位分析

用于组织中细胞类型组成的空间邻域分析。

## 概览

生态位分析内核提供：

- **生态位组成** - 计算空间邻域中的细胞类型组成
- **空间图集成** - 与空间邻居图配合工作
- **并行处理** - 针对大型空间数据集高效
- **组织上下文分析** - 理解局部细胞环境

## 生态位组成

### niche_composition

计算空间邻域中的细胞类型组成：

```cpp
#include "scl/kernel/niche.hpp"

Sparse<Real, true> spatial_graph = /* ... */;  // 空间邻居图 [n_cells x n_cells]
Array<Index> cell_types = /* ... */;            // 细胞类型标签 [n_cells]
Index n_cells = spatial_graph.rows();
Index n_types = /* 唯一细胞类型的数量 */;

Array<Real> composition(n_cells * n_types);     // 预分配输出

scl::kernel::niche::niche_composition(spatial_graph, cell_types,
                                     n_cells, n_types, composition.ptr);

// composition[i * n_types + t] 包含细胞 i 的生态位中类型 t 的比例
```

**参数：**
- `spatial_graph`: 空间邻居图（CSR 格式），大小 = n_cells × n_cells
- `cell_types`: 每个细胞的细胞类型标签，大小 = n_cells
- `n_cells`: 细胞数量
- `n_types`: 唯一细胞类型的数量
- `composition`: 输出组成矩阵，必须预分配，大小 = n_cells × n_types

**后置条件：**
- `composition[i * n_types + t]` 包含细胞 `i` 的生态位中类型 `t` 的比例
- 每行总和为 1.0（归一化组成）
- 没有邻居的细胞具有均匀组成（1/n_types）

**算法：**
对每个细胞 i 并行：
1. 从空间图获取邻居
2. 在邻域中计数细胞类型
3. 将计数归一化以获得比例
4. 存储在 composition[i * n_types + 0..n_types-1]

**复杂度：**
- 时间：O(nnz + n_cells * n_types)，其中 nnz = 空间图中的边数
- 空间：O(n_types) 每个细胞的辅助空间用于计数

**线程安全：**
- 安全 - 跨细胞并行化
- 每个线程处理独立的细胞

**用例：**
- 空间转录组学分析
- 组织结构表征
- 细胞-细胞相互作用研究
- 生态位特异性表达分析
- 肿瘤微环境分析

## 配置

### 默认参数

```cpp
namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size DEFAULT_K = 15;
    constexpr Size PARALLEL_THRESHOLD = 500;
}
```

**K 参数：**
- 考虑的默认邻居数（如果图是基于 kNN 的）
- 根据空间分辨率调整

**并行化：**
- 对于 > `PARALLEL_THRESHOLD` 细胞的数据集进行并行处理
- 对大型空间数据集高效

## 示例

### 基本生态位组成

```cpp
#include "scl/kernel/niche.hpp"
#include "scl/kernel/spatial.hpp"

// 构建空间邻居图（空间坐标中的 kNN）
Sparse<Real, true> spatial_knn = /* ... */;  // 来自 spatial::neighbors
Array<Index> cell_type_labels = /* ... */;   // 来自聚类/注释

// 计数唯一类型
Index n_types = *std::max_element(cell_type_labels.begin(),
                                  cell_type_labels.end()) + 1;
Index n_cells = spatial_knn.rows();

// 计算生态位组成
Array<Real> niche_comp(n_cells * n_types);
scl::kernel::niche::niche_composition(spatial_knn, cell_type_labels,
                                     n_cells, n_types, niche_comp.ptr);

// 分析每个细胞的组成
for (Index i = 0; i < n_cells; ++i) {
    Real* comp = &niche_comp[i * n_types];
    // comp[t] 是细胞 i 的邻域中类型 t 的比例
}
```

### 查找具有特定生态位的细胞

```cpp
// 查找生态位中类型 5 比例高的细胞
Index target_type = 5;
Real threshold = 0.3;  // 至少 30% 的邻居是类型 5

std::vector<Index> enriched_cells;
for (Index i = 0; i < n_cells; ++i) {
    Real fraction = niche_comp[i * n_types + target_type];
    if (fraction >= threshold) {
        enriched_cells.push_back(i);
    }
}

std::cout << "找到 " << enriched_cells.size()
          << " 个在生态位中富集类型 " << target_type << " 的细胞\n";
```

### 生态位异质性

```cpp
// 计算每个细胞的生态位多样性（熵）
Array<Real> niche_entropy(n_cells);
for (Index i = 0; i < n_cells; ++i) {
    Real entropy = 0.0;
    for (Index t = 0; t < n_types; ++t) {
        Real p = niche_comp[i * n_types + t];
        if (p > 0) {
            entropy -= p * std::log2(p);
        }
    }
    niche_entropy[i] = entropy;
}

// 高熵 = 多样化生态位，低熵 = 同质生态位
```

## 性能考虑

### 空间图构建

- 首先构建空间邻居图（例如，使用 `scl::kernel::spatial::neighbors`）
- kNN 图高效（O(n log n) 构建）
- 考虑距离阈值 vs kNN（kNN 更快）

### 内存

- 组成矩阵需要 `n_cells * n_types * sizeof(Real)` 字节
- 对于大型数据集（100万细胞，50类型）：约 200 MB
- 如果组成非常稀疏，考虑稀疏表示

### 并行化

- 高效地跨细胞并行化
- 每个细胞的计算是独立的
- 随着线程数扩展良好

---

::: tip 空间图
生态位组成的质量取决于空间邻居图。根据您的空间分辨率使用适当的距离度量和邻居选择（kNN、距离阈值）。
:::

