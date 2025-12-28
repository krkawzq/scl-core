# 图神经网络

用于在稀疏图上学习节点表示的图神经网络操作。

## 概览

`gnn` 模块为单细胞和空间转录组学数据提供高效的 GNN 操作：

- **图卷积** - 带归一化的标准 GCN 层
- **图注意力** - GAT 风格的注意力机制

所有操作均：
- 针对稀疏邻接矩阵优化
- 并行化以提高计算效率
- 内存高效，分配最少

## 核心函数

### graph_convolution

应用图卷积层：H' = (D^-1 A) H W

```cpp
#include "scl/kernel/gnn.hpp"

Sparse<Real, true> adjacency = /* 邻接矩阵 [n_nodes x n_nodes] */;
const Real* node_features = /* 节点特征 [n_nodes * n_features] */;
const Real* weights = /* 权重矩阵 [n_features * n_output_features] */;
Real* output = /* 输出特征 [n_nodes * n_output_features] */;

scl::kernel::gnn::graph_convolution(
    adjacency, node_features, n_nodes, n_features,
    weights, n_output_features, output
);
```

**参数：**
- `adjacency` [in] - 邻接矩阵（CSR）[n_nodes x n_nodes]
- `node_features` [in] - 节点特征 [n_nodes * n_features]，行主序
- `n_nodes` [in] - 节点数
- `n_features` [in] - 输入特征数
- `weights` [in] - 权重矩阵 [n_features * n_output_features]，行主序
- `n_output_features` [in] - 输出特征数
- `output` [out] - 输出特征 [n_nodes * n_output_features]，行主序

**前置条件：**
- `output` 容量 >= n_nodes * n_output_features
- 邻接矩阵应行归一化（可选，内部应用归一化）

**后置条件：**
- `output` 包含卷积后的特征

**算法：**
1. 归一化邻接：D^-1 A（逆度归一化）
2. 应用图卷积：(D^-1 A) H
3. 应用线性变换：H' = H W

**复杂度：**
- 时间：O(nnz * n_features + n_nodes * n_features * n_output_features)
- 空间：O(n_nodes * n_output_features) 辅助空间

**线程安全：** 安全 - 并行化

### graph_attention

应用图注意力层（GAT 风格）。

```cpp
const Real* attention_weights = /* 注意力权重矩阵 [n_features * n_features] */;

scl::kernel::gnn::graph_attention(
    adjacency, node_features, n_nodes, n_features,
    attention_weights, output, 0.5  // alpha: LeakyReLU 斜率
);
```

**参数：**
- `adjacency` [in] - 邻接矩阵（CSR）[n_nodes x n_nodes]
- `node_features` [in] - 节点特征 [n_nodes * n_features]，行主序
- `n_nodes` [in] - 节点数
- `n_features` [in] - 特征数
- `attention_weights` [in] - 注意力权重矩阵 [n_features * n_features]，行主序
- `output` [out] - 注意力输出 [n_nodes * n_features]，行主序
- `alpha` [in] - 注意力系数（LeakyReLU 斜率，默认：0.5）

**前置条件：**
- `output` 容量 >= n_nodes * n_features

**后置条件：**
- `output` 包含注意力后的特征

**算法：**
1. 为每条边计算注意力分数：a_ij = LeakyReLU(W h_i, W h_j)
2. 在邻居上使用 softmax 归一化注意力分数
3. 按注意力加权聚合邻居特征：h'_i = sum_j a_ij W h_j

**复杂度：**
- 时间：O(nnz * n_features + n_nodes * n_features^2)
- 空间：O(n_nodes * n_features) 辅助空间

**线程安全：** 安全 - 并行化

## 配置

```cpp
namespace scl::kernel::gnn::config {
    constexpr Real DEFAULT_ALPHA = Real(0.5);
    constexpr Real EPSILON = Real(1e-15);
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size SIMD_THRESHOLD = 16;
}
```

## 使用示例

### 多层 GCN

```cpp
// 第 1 层：n_features -> n_hidden
Real* hidden = allocate(n_nodes * n_hidden);
scl::kernel::gnn::graph_convolution(
    adjacency, node_features, n_nodes, n_features,
    weights1, n_hidden, hidden
);

// 应用激活函数（例如，ReLU）
apply_relu(hidden, n_nodes * n_hidden);

// 第 2 层：n_hidden -> n_output
Real* output = allocate(n_nodes * n_output);
scl::kernel::gnn::graph_convolution(
    adjacency, hidden, n_nodes, n_hidden,
    weights2, n_output, output
);
```

### 图注意力网络

```cpp
// 单层 GAT
const Real* attention_weights = /* 可学习的注意力权重 */;

scl::kernel::gnn::graph_attention(
    adjacency, node_features, n_nodes, n_features,
    attention_weights, output, 0.2  // LeakyReLU alpha
);

// 应用激活和归一化
apply_relu(output, n_nodes * n_features);
normalize_rows(output, n_nodes, n_features);
```

### 细胞-细胞图学习

```cpp
// 从表达数据构建 k-NN 图
Sparse<Real, true> knn_graph = build_knn_graph(X, k=15);

// 使用 GCN 平滑基因表达
const Real* expression = /* 基因表达 [n_cells * n_genes] */;
const Real* weights = /* 学习的权重 */;
Real* smoothed_expression = allocate(n_cells * n_genes);

scl::kernel::gnn::graph_convolution(
    knn_graph, expression, n_cells, n_genes,
    weights, n_genes, smoothed_expression
);
```

## 性能说明

- 邻接矩阵应为 CSR 格式以获得最佳性能
- 图卷积内部使用归一化邻接（D^-1 A）
- 注意力机制更具表现力但计算成本更高
- 对于大型图，考虑使用近似注意力或采样
- 特征维度应为 2 的幂以获得最佳 SIMD 性能

---

::: tip 图构建
GNN 操作需要图作为输入。常见的图构建方法：从表达数据的 k-NN 图、空间邻域图或学习的相似度图。
:::

