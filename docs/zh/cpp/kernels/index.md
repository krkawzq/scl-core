# 内核函数

`scl/kernel/` 目录包含 400+ 个按功能组织的计算算子。

## 概述

内核函数提供：

- **稀疏工具** - 矩阵转换、验证、清理、优化
- **线性代数** - 稀疏矩阵向量乘法、Gram 矩阵
- **预处理** - 归一化、缩放、对数变换、标准化
- **特征选择** - 高变基因、质量控制指标
- **统计** - 统计检验、相关性、多重检验校正
- **邻居搜索** - KNN、批次平衡 KNN
- **聚类** - Leiden、Louvain 社区检测
- **空间分析** - 空间模式、热点检测、组织分析
- **富集分析** - 基因集富集分析
- **图操作** - 扩散、传播、中心性
- **单细胞分析** - 标记基因、双细胞检测、速度、拟时序
- **数据操作** - 合并、切片、重排序、采样、排列

## 分类

### 稀疏工具

矩阵基础设施和工具：

```cpp
#include "scl/kernel/sparse.hpp"

// 转换为连续 CSR/CSC
auto arrays = scl::kernel::sparse::to_contiguous_arrays(matrix);

// 转换为 COO 格式
auto coo = scl::kernel::sparse::to_coo_arrays(matrix);

// 验证矩阵结构
auto result = scl::kernel::sparse::validate(matrix);

// 获取内存信息
auto info = scl::kernel::sparse::memory_info(matrix);

// 消除零值
scl::kernel::sparse::eliminate_zeros(matrix, tolerance);

// 修剪小值
scl::kernel::sparse::prune(matrix, threshold, keep_structure);
```

### 归一化

行/列归一化和缩放：

```cpp
#include "scl/kernel/normalize.hpp"

// 计算行范数
scl::kernel::normalize::row_norms(matrix, NormMode::L2, output);

// 原地归一化行
scl::kernel::normalize::normalize_rows_inplace(matrix, NormMode::L2);

// 缩放矩阵
scl::kernel::normalize::scale(matrix, factor);
```

### Softmax

带温度缩放的 Softmax 归一化：

```cpp
#include "scl/kernel/softmax.hpp"

// 标准 softmax
scl::kernel::softmax::softmax_inplace(values, len);

// 带温度
scl::kernel::softmax::softmax_inplace(values, len, 0.5);

// Log-softmax
scl::kernel::softmax::log_softmax_inplace(values, len);

// 稀疏矩阵
scl::kernel::softmax::softmax_inplace(matrix);
```

### 缩放与变换

标准化、缩放和对数变换：

```cpp
#include "scl/kernel/scale.hpp"
#include "scl/kernel/log1p.hpp"

// 标准化矩阵
scl::kernel::scale::standardize(matrix, means, stds, max_value, zero_center);

// 缩放行
scl::kernel::scale::scale_rows(matrix, scales);

// Log1p 变换
scl::kernel::log1p::log1p_inplace(matrix);

// Log2(1+x) 变换
scl::kernel::log1p::log2p1_inplace(matrix);

// 反向变换
scl::kernel::log1p::expm1_inplace(matrix);
```

### 线性代数

稀疏矩阵向量乘法和 Gram 矩阵：

```cpp
#include "scl/kernel/algebra.hpp"
#include "scl/kernel/gram.hpp"

// 稀疏矩阵向量乘法: y = alpha * A * x + beta * y
scl::kernel::algebra::spmv(A, x, y, alpha, beta);

// 简单形式: y = A * x
scl::kernel::algebra::spmv_simple(A, x, y);

// Gram 矩阵: G[i,j] = dot(row_i, row_j)
scl::kernel::gram::gram(matrix, output);
```

### 统计

统计检验、相关性和多重检验校正：

```cpp
#include "scl/kernel/ttest.hpp"
#include "scl/kernel/mwu.hpp"
#include "scl/kernel/correlation.hpp"
#include "scl/kernel/multiple_testing.hpp"

// T 检验
scl::kernel::ttest::ttest(matrix, group_ids, t_stats, p_values, log2_fc);

// Mann-Whitney U 检验
scl::kernel::mwu::mwu_test(matrix, group_ids, u_stats, p_values, log2_fc);

// Pearson 相关性
scl::kernel::correlation::pearson(matrix, output);

// 多重检验校正
scl::kernel::multiple_testing::benjamini_hochberg(p_values, adjusted_p_values);
scl::kernel::multiple_testing::bonferroni(p_values, adjusted_p_values);
```

### 分组聚合

每组统计：

```cpp
#include "scl/kernel/group.hpp"

// 计算组均值和方差
scl::kernel::group::group_stats(
    matrix, group_ids, n_groups, group_sizes, means, vars
);
```

### 矩阵操作

合并和切片：

```cpp
#include "scl/kernel/merge.hpp"
#include "scl/kernel/slice.hpp"

// 垂直堆叠
auto result = scl::kernel::merge::vstack(matrix1, matrix2);

// 水平堆叠
auto result = scl::kernel::merge::hstack(matrix1, matrix2);

// 切片行
auto sliced = scl::kernel::slice::slice_primary(matrix, keep_indices);

// 过滤列
auto filtered = scl::kernel::slice::filter_secondary(matrix, mask);
```

### 邻居搜索

K 近邻：

```cpp
#include "scl/kernel/neighbors.hpp"

// 预计算范数
scl::kernel::neighbors::compute_norms(matrix, norms_sq);

// 计算 KNN
scl::kernel::neighbors::knn(matrix, norms_sq, k, indices, distances);
```

### 批次平衡 KNN

用于跨批次数据整合的批次感知 KNN：

```cpp
#include "scl/kernel/bbknn.hpp"

// 预计算范数（可选）
scl::kernel::bbknn::compute_norms(matrix, norms_sq);

// 计算 BBKNN（从每个批次找到 k 个邻居）
scl::kernel::bbknn::bbknn(
    matrix, batch_labels, n_batches, k, indices, distances, norms_sq
);
```

### MMD

用于分布比较的最大均值差异：

```cpp
#include "scl/kernel/mmd.hpp"

// 比较两个分布
scl::kernel::mmd::mmd_rbf(mat_x, mat_y, output, gamma);
```

### 聚类

社区检测：

```cpp
#include "scl/kernel/leiden.hpp"
#include "scl/kernel/louvain.hpp"

// Leiden 聚类
auto labels = scl::kernel::leiden::leiden(graph, resolution);

// Louvain 聚类
auto labels = scl::kernel::louvain::louvain(graph, resolution);
```

### 特征选择

高变基因和质量控制：

```cpp
#include "scl/kernel/hvg.hpp"
#include "scl/kernel/qc.hpp"

// 通过离散度选择高变基因
scl::kernel::hvg::select_by_dispersion(
    matrix, n_top, out_indices, out_mask, out_dispersions
);

// 质量控制指标
scl::kernel::qc::compute_basic_qc(matrix, out_n_genes, out_total_counts);
scl::kernel::qc::compute_subset_pct(matrix, subset_mask, out_pcts);
```

### 标记基因

标记基因识别和特异性评分：

```cpp
#include "scl/kernel/markers.hpp"

// 为每个簇寻找标记基因
scl::kernel::markers::find_markers(
    expression, cluster_labels, n_cells, n_genes, n_clusters,
    marker_genes, marker_scores, max_markers, min_fc, max_pval
);
```

### 数据操作

重排序、采样和排列：

```cpp
#include "scl/kernel/reorder.hpp"
#include "scl/kernel/sampling.hpp"
#include "scl/kernel/permutation.hpp"

// 重排序行
scl::kernel::reorder::reorder_rows(matrix, permutation, n_rows, output);

// 几何草图采样用于降采样
scl::kernel::sampling::geometric_sketching(
    data, target_size, selected_indices, n_selected, seed
);

// 排列检验
scl::kernel::permutation::permutation_test(
    data, labels, test_func, n_permutations, p_value, seed
);
```

## 设计模式

### 函数式 API

无副作用的纯函数：

```cpp
// 纯函数 - 返回结果
auto result = compute_something(input);

// 原地修改 - 明确命名
modify_something_inplace(data);
```

### 基于模板的多态

适用于任何兼容类型：

```cpp
template <CSRLike MatrixT>
void process_matrix(const MatrixT& matrix) {
    // 适用于任何类似 CSR 的类型
}
```

### 显式并行化

```cpp
// 默认并行处理大输入
parallel_for(Size(0), n, [&](size_t i) {
    process(data[i]);
});
```

## 性能

### SIMD 优化

热路径使用 SIMD：

```cpp
namespace s = scl::simd;
const s::Tag d;

for (size_t i = 0; i < n; i += s::Lanes(d)) {
    auto v = s::Load(d, data + i);
    // SIMD 操作
}
```

### 最小分配

使用工作空间而不是分配：

```cpp
// 预分配工作空间
WorkspacePool<Real> pool(num_threads, workspace_size);

// 在并行循环中重用
parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    Real* workspace = pool.get(thread_rank);
    // 使用工作空间
});
```

## 下一步

探索特定的内核类别：

### 核心基础设施
- [稀疏工具](/zh/cpp/kernels/sparse-tools) - 矩阵工具和优化
- [代数](/zh/cpp/kernels/algebra) - 稀疏线性代数操作
- [Gram](/zh/cpp/kernels/gram) - Gram 矩阵计算

### 预处理
- [归一化](/zh/cpp/kernels/normalization) - 归一化和缩放
- [缩放](/zh/cpp/kernels/scale) - 标准化和行缩放
- [Log1p](/zh/cpp/kernels/log1p) - 对数变换
- [Softmax](/zh/cpp/kernels/softmax) - 带温度缩放的 Softmax 归一化

### 特征选择
- [HVG](/zh/cpp/kernels/hvg) - 高变基因选择
- [QC](/zh/cpp/kernels/qc) - 质量控制指标
- [标记基因](/zh/cpp/kernels/markers) - 标记基因识别

### 统计
- [T 检验](/zh/cpp/kernels/ttest) - 参数统计检验
- [Mann-Whitney U](/zh/cpp/kernels/mwu) - 非参数统计检验
- [相关性](/zh/cpp/kernels/correlation) - Pearson 相关性
- [多重检验](/zh/cpp/kernels/multiple_testing) - FDR 和 FWER 校正

### 邻居搜索与聚类
- [邻居搜索](/zh/cpp/kernels/neighbors) - KNN 算法
- [BBKNN](/zh/cpp/kernels/bbknn) - 用于批次整合的批次平衡 KNN
- [Leiden](/zh/cpp/kernels/leiden) - Leiden 聚类
- [Louvain](/zh/cpp/kernels/louvain) - Louvain 聚类

### 数据操作
- [合并](/zh/cpp/kernels/merge) - 矩阵合并操作
- [切片](/zh/cpp/kernels/slice) - 矩阵切片操作
- [重排序](/zh/cpp/kernels/reorder) - 矩阵重排序
- [采样](/zh/cpp/kernels/sampling) - 降采样方法
- [排列](/zh/cpp/kernels/permutation) - 排列检验

### 高级分析
- [MMD](/zh/cpp/kernels/mmd) - 最大均值差异
- [分组](/zh/cpp/kernels/group) - 分组聚合统计

---

::: tip 高性能
所有内核函数都通过 SIMD、并行化和最小分配进行了性能优化。
:::

