# SCL-Core 算子迁移计划

> **版本**: v0.4 → v0.5
> **更新日期**: 2025-12-31
> **原则**: 基础设施已完善，禁止改动。仅迁移算子层（kernel/、math/）

---

## 迁移优先级说明

- **P0 - 最高优先级**: 核心统计算子，差异分析基础
- **P1 - 高优先级**: 常用生物信息学算子
- **P2 - 中优先级**: 高级分析算子
- **P3 - 低优先级**: 特殊场景算子

每个算子包含三个开发阶段：
1. **C++ 实现** (`scl/kernel/*.hpp`)
2. **C-API 接口** (`scl/api/kernel/*.h` + `*.cpp`)
3. **文档开发** (`docs/api/*.md`)

---

## P0 - 最高优先级（核心统计与差异分析）

### 1. 统计基础设施

#### 1.1 stat_base - 统计基础
- [x] C++ 实现 (`scl/math/stat_base.hpp`) ✅ 2025-12-31
- [x] C-API 接口 (N/A - 数学工具，无需封装)
- [x] 文档开发 (内联文档已完成)

**功能**: 统计常量、p值计算、分组常量、基础统计工具
**依赖**: `scl/math/stats.hpp` (已有)
**优先级**: P0 - 所有统计算子的基础
**状态**: ✅ 已完成

---

#### 1.2 rank_utils - 秩计算工具
- [x] C++ 实现 (`scl/math/rank_utils.hpp`) ✅ 2025-12-31
- [x] C-API 接口 (N/A - 数学工具，无需封装)
- [x] 文档开发 (内联文档已完成)

**功能**: 秩计算、平均秩处理、秩变换（MWU/AUROC/Spearman 共用）
**依赖**: `stat_base.hpp`, `scl/core/sort.hpp`
**优先级**: P0 - 非参数检验基础
**状态**: ✅ 已完成

---

#### 1.3 group_partition - 分组分区
- [x] C++ 实现 (`scl/math/group_partition.hpp`) ✅ 2025-12-31
- [x] C-API 接口 (N/A - 数学工具，无需封装)
- [x] 文档开发 (内联文档已完成)

**功能**: 按组标签分区数据、组内索引管理、矩累加
**依赖**: `scl/core/type.hpp`
**优先级**: P0 - 多组比较基础
**状态**: ✅ 已完成

---

### 2. 核心统计检验

#### 2.1 auroc - ROC 曲线下面积
- [ ] C++ 实现 (`scl/math/auroc.hpp`)
- [ ] C-API 接口 (`scl/api/math/auroc.h`)
- [ ] 文档开发

**功能**: AUROC 计算、ROC 曲线、分类性能评估
**依赖**: `rank_utils.hpp`
**优先级**: P0 - 差异基因检测核心指标

---

#### 2.2 effect_size - 效应量
- [x] C++ 实现 (`scl/math/effect_size.hpp`) ✅ 2025-12-31
- [x] C-API 接口 (N/A - 数学工具，无需封装)
- [x] 文档开发 (内联文档已完成)

**功能**: Cohen's d, Hedges' g, Glass' Δ, CLES
**依赖**: `stat_base.hpp`
**优先级**: P0 - 差异显著性量化
**状态**: ✅ 已完成

---

#### 2.3 ttest - t 检验
- [x] C++ 实现 (`scl/math/ttest.hpp`) ✅ 2025-12-31
- [x] C-API 接口 (N/A - 数学工具，无需封装)
- [x] 文档开发 (内联文档已完成)

**功能**: Welch's t-test, Student's t-test, 配对 t 检验
**依赖**: `stat_base.hpp`, `scl/math/stats.hpp`
**优先级**: P0 - 最常用参数检验
**状态**: ✅ 已完成

---

#### 2.4 mwu - Mann-Whitney U 检验
- [x] C++ 实现 (`scl/math/mwu.hpp`) ✅ 2025-12-31
- [x] C-API 接口 (N/A - 数学工具，无需封装)
- [x] 文档开发 (内联文档已完成)

**功能**: Mann-Whitney U 检验（非参数）、Wilcoxon 秩和检验
**依赖**: `rank_utils.hpp`, `stat_base.hpp`
**优先级**: P0 - 最常用非参数检验
**状态**: ✅ 已完成（更新现有实现以符合新规范）

---

### 3. 差异分析

#### 3.1 markers - 差异标记基因
- [ ] C++ 实现 (`scl/kernel/markers.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/markers.h`)
- [ ] 文档开发

**功能**: 差异基因检测、多组比较、统计量聚合
**依赖**: `ttest.hpp`, `mwu.hpp`, `auroc.hpp`, `effect_size.hpp`
**优先级**: P0 - 单细胞分析核心功能

---

#### 3.2 multiple_testing - 多重检验校正
- [x] C++ 实现 (`scl/math/multiple_testing.hpp`) ✅ 2025-12-31
- [x] C-API 接口 (N/A - 数学工具，无需封装)
- [x] 文档开发 (内联文档已完成)

**功能**: Bonferroni, Benjamini-Hochberg FDR, Benjamini-Yekutieli FDR
**依赖**: `scl/core/sort.hpp` (使用 std::sort)
**优先级**: P0 - 多重检验必备
**状态**: ✅ 已完成（使用现代 C++20 std::span 接口）

---

### 4. 基础预处理

#### 4.1 log1p - log1p 变换
- [x] C++ 实现 (`scl/kernel/log1p.hpp`) ✅ 2025-12-31
- [x] C-API 接口 (`scl/api/kernel/log1p.h` + `log1p.cpp`) ✅ 2025-12-31
- [ ] 文档开发

**功能**: log(1+x) 变换、log2(1+x) 变换、expm1(x) 变换、8-way SIMD 优化、管道预取、自动并行化
**依赖**: `scl/core/simd.hpp`, `scl/threading/parallel_for.hpp`
**优先级**: P0 - 标准化前置步骤
**状态**: ✅ 已完成（C++ + C-API）

---

#### 4.2 normalize - 标准化
- [ ] C++ 实现 (`scl/kernel/normalize.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/normalize.h`)
- [ ] 文档开发

**功能**: CPM, TPM, log-normalization, library size normalization
**依赖**: `log1p.hpp`
**优先级**: P0 - 数据预处理核心

---

#### 4.3 scale - 缩放
- [ ] C++ 实现 (`scl/kernel/scale.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/scale.h`)
- [ ] 文档开发

**功能**: z-score, min-max scaling, robust scaling
**依赖**: `scl/core/simd.hpp`
**优先级**: P0 - 数据预处理核心

---

#### 4.4 qc - 质量控制
- [ ] C++ 实现 (`scl/kernel/qc.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/qc.h`)
- [ ] 文档开发

**功能**: 质控指标计算（nGenes, nCounts, mito%, ribo%）
**依赖**: `scl/core/sparse.hpp`
**优先级**: P0 - 数据过滤基础

---

#### 4.5 hvg - 高变基因选择
- [ ] C++ 实现 (`scl/kernel/hvg.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/hvg.h`)
- [ ] 文档开发

**功能**: Seurat, Scanpy, Pearson residuals 方法
**依赖**: `scl/core/sparse.hpp`
**优先级**: P0 - 特征选择核心

---

### 5. 邻域图

#### 5.1 neighbors - K 近邻图
- [ ] C++ 实现 (`scl/kernel/neighbors.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/neighbors.h`)
- [ ] 文档开发

**功能**: KNN 图构建、距离计算、稀疏图表示
**依赖**: `scl/core/sparse.hpp`, `scl/math/metrics.hpp`
**优先级**: P0 - 聚类/降维基础

---

## P1 - 高优先级（常用生物信息学算子）

### 6. 统计检验扩展

#### 6.1 kruskal_wallis - Kruskal-Wallis H 检验
- [ ] C++ 实现 (`scl/math/kruskal_wallis.hpp`)
- [ ] C-API 接口 (`scl/api/math/kruskal_wallis.h`)
- [ ] 文档开发

**功能**: 非参数单因素方差分析（多组比较）
**依赖**: `rank_utils.hpp`
**优先级**: P1

---

#### 6.2 ks - Kolmogorov-Smirnov 检验
- [ ] C++ 实现 (`scl/math/ks.hpp`)
- [ ] C-API 接口 (`scl/api/math/ks.h`)
- [ ] 文档开发

**功能**: KS 双样本检验、分布差异检测
**依赖**: `rank_utils.hpp`
**优先级**: P1

---

#### 6.3 oneway_anova - 单因素方差分析
- [ ] C++ 实现 (`scl/math/oneway_anova.hpp`)
- [ ] C-API 接口 (`scl/api/math/oneway_anova.h`)
- [ ] 文档开发

**功能**: ANOVA F 检验、组间方差分析
**依赖**: `stat_base.hpp`
**优先级**: P1

---

#### 6.4 comparison - 多组比较
- [ ] C++ 实现 (`scl/kernel/comparison.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/comparison.h`)
- [ ] 文档开发

**功能**: 一对多、多对多比较、批量统计检验
**依赖**: `ttest.hpp`, `mwu.hpp`, `kruskal_wallis.hpp`
**优先级**: P1

---

### 7. 质控与过滤

#### 7.1 doublet - 双细胞检测
- [ ] C++ 实现 (`scl/kernel/doublet.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/doublet.h`)
- [ ] 文档开发

**功能**: Doublet 检测（Scrublet, DoubletFinder 算法）
**依赖**: `neighbors.hpp`
**优先级**: P1

---

#### 7.2 outlier - 离群值检测
- [ ] C++ 实现 (`scl/kernel/outlier.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/outlier.h`)
- [ ] 文档开发

**功能**: MAD, IQR, isolation forest 方法
**依赖**: `scl/core/sort.hpp`
**优先级**: P1

---

#### 7.3 feature - 特征选择
- [ ] C++ 实现 (`scl/kernel/feature.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/feature.h`)
- [ ] 文档开发

**功能**: 特征选择算法（方差、相关性、信息增益）
**依赖**: `scl/core/sparse.hpp`
**优先级**: P1

---

### 8. 降维与聚类

#### 8.1 projection - 降维投影
- [ ] C++ 实现 (`scl/kernel/projection.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/projection.h`)
- [ ] 文档开发

**功能**: PCA, t-SNE, UMAP 降维
**依赖**: `algebra.hpp`
**优先级**: P1

---

#### 8.2 leiden - Leiden 聚类
- [ ] C++ 实现 (`scl/kernel/leiden.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/leiden.h`)
- [ ] 文档开发

**功能**: Leiden 社区检测算法
**依赖**: `neighbors.hpp`
**优先级**: P1

---

#### 8.3 louvain - Louvain 聚类
- [ ] C++ 实现 (`scl/kernel/louvain.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/louvain.h`)
- [ ] 文档开发

**功能**: Louvain 社区检测算法
**依赖**: `neighbors.hpp`
**优先级**: P1

---

### 9. 细胞注释

#### 9.1 annotation - 细胞类型注释
- [ ] C++ 实现 (`scl/kernel/annotation.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/annotation.h`)
- [ ] 文档开发

**功能**: 基于标记基因的细胞类型注释
**依赖**: `markers.hpp`
**优先级**: P1

---

#### 9.2 scoring - 基因集打分
- [ ] C++ 实现 (`scl/kernel/scoring.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/scoring.h`)
- [ ] 文档开发

**功能**: 基因集评分（Seurat, Scanpy 方法）
**依赖**: `scl/core/sparse.hpp`
**优先级**: P1

---

#### 9.3 enrichment - 富集分析
- [ ] C++ 实现 (`scl/kernel/enrichment.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/enrichment.h`)
- [ ] 文档开发

**功能**: GO/KEGG 富集分析、超几何检验
**依赖**: `multiple_testing.hpp`
**优先级**: P1

---

### 10. 空间转录组

#### 10.1 spatial - 空间分析
- [ ] C++ 实现 (`scl/kernel/spatial.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/spatial.h`)
- [ ] 文档开发

**功能**: 空间邻域、空间自相关（Moran's I, Geary's C）
**依赖**: `neighbors.hpp`
**优先级**: P1

---

### 11. 通用算法

#### 11.1 correlation - 相关系数
- [ ] C++ 实现 (`scl/math/correlation.hpp`)
- [ ] C-API 接口 (`scl/api/math/correlation.h`)
- [ ] 文档开发

**功能**: Pearson, Spearman, Kendall 相关系数
**依赖**: `rank_utils.hpp`
**优先级**: P1

---

#### 11.2 metrics - 距离度量
- [ ] C++ 实现 (`scl/math/metrics.hpp`)
- [ ] C-API 接口 (`scl/api/math/metrics.h`)
- [ ] 文档开发

**功能**: 欧氏距离、余弦相似度、Jaccard 系数
**依赖**: `scl/core/simd.hpp`
**优先级**: P1

---

#### 11.3 components - 连通分量
- [ ] C++ 实现 (`scl/kernel/components.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/components.h`)
- [ ] 文档开发

**功能**: 图连通分量检测（Union-Find）
**依赖**: `scl/core/sparse.hpp`
**优先级**: P1

---

#### 11.4 merge - 数据合并
- [ ] C++ 实现 (`scl/kernel/merge.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/merge.h`)
- [ ] 文档开发

**功能**: 数据集合并、批次整合
**依赖**: `scl/core/sparse.hpp`
**优先级**: P1

---

#### 11.5 reorder - 重排序
- [ ] C++ 实现 (`scl/kernel/reorder.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/reorder.h`)
- [ ] 文档开发

**功能**: 按索引重排序、层次聚类排序
**依赖**: `scl/core/sparse.hpp`
**优先级**: P1

---

#### 11.6 group - 分组操作
- [ ] C++ 实现 (`scl/kernel/group.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/group.h`)
- [ ] 文档开发

**功能**: 分组聚合、组内统计
**依赖**: `scl/core/sparse.hpp`
**优先级**: P1

---

#### 11.7 sampling - 采样
- [ ] C++ 实现 (`scl/kernel/sampling.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/sampling.h`)
- [ ] 文档开发

**功能**: 随机采样、分层采样、下采样
**依赖**: `scl/core/type.hpp`
**优先级**: P1

---

#### 11.8 sparse_opt - 稀疏矩阵优化
- [ ] C++ 实现 (`scl/kernel/sparse_opt.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/sparse_opt.h`)
- [ ] 文档开发

**功能**: 稀疏矩阵压缩、格式转换优化
**依赖**: `scl/core/sparse.hpp`
**优先级**: P1

---

#### 11.9 algebra - 线性代数
- [ ] C++ 实现 (`scl/math/algebra.hpp`)
- [ ] C-API 接口 (`scl/api/math/algebra.h`)
- [ ] 文档开发

**功能**: 矩阵乘法、SVD、特征值分解
**依赖**: `scl/core/sparse.hpp`
**优先级**: P1

---

#### 11.10 softmax - Softmax
- [ ] C++ 实现 (`scl/math/softmax.hpp`)
- [ ] C-API 接口 (`scl/api/math/softmax.h`)
- [ ] 文档开发

**功能**: Softmax、log-softmax、数值稳定版本
**依赖**: `scl/core/simd.hpp`
**优先级**: P1

---

## P2 - 中优先级（高级分析算子）

### 12. 轨迹推断

#### 12.1 pseudotime - 拟时序分析
- [ ] C++ 实现 (`scl/kernel/pseudotime.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/pseudotime.h`)
- [ ] 文档开发

**功能**: 拟时序推断、轨迹排序
**依赖**: `neighbors.hpp`, `diffusion.hpp`
**优先级**: P2

---

#### 12.2 velocity - RNA 速率
- [ ] C++ 实现 (`scl/kernel/velocity.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/velocity.h`)
- [ ] 文档开发

**功能**: RNA 速率分析、动态建模
**依赖**: `neighbors.hpp`
**优先级**: P2

---

#### 12.3 lineage - 谱系追踪
- [ ] C++ 实现 (`scl/kernel/lineage.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/lineage.h`)
- [ ] 文档开发

**功能**: 细胞谱系推断、分化路径
**依赖**: `pseudotime.hpp`
**优先级**: P2

---

#### 12.4 transition - 转换概率
- [ ] C++ 实现 (`scl/kernel/transition.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/transition.h`)
- [ ] 文档开发

**功能**: 转换概率矩阵、马尔可夫链
**依赖**: `neighbors.hpp`
**优先级**: P2

---

### 13. 细胞通讯与调控

#### 13.1 communication - 细胞通讯
- [ ] C++ 实现 (`scl/kernel/communication.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/communication.h`)
- [ ] 文档开发

**功能**: 配体-受体分析、细胞间通讯
**依赖**: `scl/core/sparse.hpp`
**优先级**: P2

---

#### 13.2 association - 关联分析
- [ ] C++ 实现 (`scl/kernel/association.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/association.h`)
- [ ] 文档开发

**功能**: 基因-基因关联、共表达分析
**依赖**: `correlation.hpp`
**优先级**: P2

---

#### 13.3 grn - 基因调控网络
- [ ] C++ 实现 (`scl/kernel/grn.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/grn.h`)
- [ ] 文档开发

**功能**: GRN 推断、转录因子-靶基因
**依赖**: `correlation.hpp`
**优先级**: P2

---

#### 13.4 coexpression - 共表达网络
- [ ] C++ 实现 (`scl/kernel/coexpression.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/coexpression.h`)
- [ ] 文档开发

**功能**: 共表达模块检测、WGCNA
**依赖**: `correlation.hpp`
**优先级**: P2

---

### 14. 空间转录组高级分析

#### 14.1 spatial_pattern - 空间模式
- [ ] C++ 实现 (`scl/kernel/spatial_pattern.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/spatial_pattern.h`)
- [ ] 文档开发

**功能**: 空间表达模式识别
**依赖**: `spatial.hpp`
**优先级**: P2

---

#### 14.2 hotspot - 空间热点
- [ ] C++ 实现 (`scl/kernel/hotspot.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/hotspot.h`)
- [ ] 文档开发

**功能**: 空间热点检测、局部聚集
**依赖**: `spatial.hpp`
**优先级**: P2

---

#### 14.3 niche - 微环境分析
- [ ] C++ 实现 (`scl/kernel/niche.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/niche.h`)
- [ ] 文档开发

**功能**: 细胞微环境、邻域组成分析
**依赖**: `spatial.hpp`
**优先级**: P2

---

#### 14.4 tissue - 组织结构
- [ ] C++ 实现 (`scl/kernel/tissue.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/tissue.h`)
- [ ] 文档开发

**功能**: 组织结构分析、区域划分
**依赖**: `spatial.hpp`
**优先级**: P2

---

### 15. 图算法

#### 15.1 centrality - 中心性度量
- [ ] C++ 实现 (`scl/kernel/centrality.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/centrality.h`)
- [ ] 文档开发

**功能**: 度中心性、介数中心性、特征向量中心性
**依赖**: `scl/core/sparse.hpp`
**优先级**: P2

---

#### 15.2 diffusion - 扩散算法
- [ ] C++ 实现 (`scl/kernel/diffusion.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/diffusion.h`)
- [ ] 文档开发

**功能**: 图扩散、随机游走、扩散映射
**依赖**: `scl/core/sparse.hpp`
**优先级**: P2

---

#### 15.3 propagation - 标签传播
- [ ] C++ 实现 (`scl/kernel/propagation.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/propagation.h`)
- [ ] 文档开发

**功能**: 标签传播算法、半监督学习
**依赖**: `scl/core/sparse.hpp`
**优先级**: P2

---

### 16. 核方法

#### 16.1 kernel - 核函数
- [ ] C++ 实现 (`scl/math/kernel.hpp`)
- [ ] C-API 接口 (`scl/api/math/kernel.h`)
- [ ] 文档开发

**功能**: RBF、多项式、线性核
**依赖**: `scl/core/simd.hpp`
**优先级**: P2

---

#### 16.2 gram - Gram 矩阵
- [ ] C++ 实现 (`scl/math/gram.hpp`)
- [ ] C-API 接口 (`scl/api/math/gram.h`)
- [ ] 文档开发

**功能**: Gram 矩阵计算、核矩阵
**依赖**: `kernel.hpp`
**优先级**: P2

---

#### 16.3 mmd - 最大均值差异
- [ ] C++ 实现 (`scl/math/mmd.hpp`)
- [ ] C-API 接口 (`scl/api/math/mmd.h`)
- [ ] 文档开发

**功能**: MMD 距离、分布比较
**依赖**: `kernel.hpp`
**优先级**: P2

---

#### 16.4 sparse_kernel - 稀疏核
- [ ] C++ 实现 (`scl/math/sparse_kernel.hpp`)
- [ ] C-API 接口 (`scl/api/math/sparse_kernel.h`)
- [ ] 文档开发

**功能**: 稀疏数据核函数
**依赖**: `kernel.hpp`, `scl/core/sparse.hpp`
**优先级**: P2

---

### 17. 其他统计与工具

#### 17.1 permutation_stat - 置换检验
- [ ] C++ 实现 (`scl/math/permutation_stat.hpp`)
- [ ] C-API 接口 (`scl/api/math/permutation_stat.h`)
- [ ] 文档开发

**功能**: 置换检验、随机化检验
**依赖**: `stat_base.hpp`
**优先级**: P2

---

#### 17.2 resample - 重采样
- [ ] C++ 实现 (`scl/kernel/resample.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/resample.h`)
- [ ] 文档开发

**功能**: Bootstrap、Jackknife
**依赖**: `sampling.hpp`
**优先级**: P2

---

#### 17.3 permutation - 置换
- [ ] C++ 实现 (`scl/kernel/permutation.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/permutation.h`)
- [ ] 文档开发

**功能**: 随机置换、排列生成
**依赖**: `scl/core/type.hpp`
**优先级**: P2

---

#### 17.4 entropy - 熵计算
- [ ] C++ 实现 (`scl/math/entropy.hpp`)
- [ ] C-API 接口 (`scl/api/math/entropy.h`)
- [ ] 文档开发

**功能**: Shannon 熵、互信息、KL 散度
**依赖**: `scl/core/simd.hpp`
**优先级**: P2

---

#### 17.5 impute - 缺失值填充
- [ ] C++ 实现 (`scl/kernel/impute.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/impute.h`)
- [ ] 文档开发

**功能**: KNN 填充、MAGIC、均值填充
**依赖**: `neighbors.hpp`
**优先级**: P2

---

#### 17.6 state - 细胞状态
- [ ] C++ 实现 (`scl/kernel/state.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/state.h`)
- [ ] 文档开发

**功能**: 细胞状态分析、状态转换
**依赖**: `scl/core/sparse.hpp`
**优先级**: P2

---

#### 17.7 subpopulation - 亚群分析
- [ ] C++ 实现 (`scl/kernel/subpopulation.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/subpopulation.h`)
- [ ] 文档开发

**功能**: 亚群识别、层次结构
**依赖**: `leiden.hpp`, `louvain.hpp`
**优先级**: P2

---

#### 17.8 bbknn - 批次平衡 KNN
- [ ] C++ 实现 (`scl/kernel/bbknn.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/bbknn.h`)
- [ ] 文档开发

**功能**: 批次效应校正的 KNN
**依赖**: `neighbors.hpp`
**优先级**: P2

---

## P3 - 低优先级（特殊场景算子）

### 18. 图神经网络

#### 18.1 gnn - 图神经网络
- [ ] C++ 实现 (`scl/kernel/gnn.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/gnn.h`)
- [ ] 文档开发

**功能**: GCN、GAT、GraphSAGE
**依赖**: `neighbors.hpp`, `algebra.hpp`
**优先级**: P3

---

### 19. 免疫组库分析

#### 19.1 clonotype - 克隆型分析
- [ ] C++ 实现 (`scl/kernel/clonotype.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/clonotype.h`)
- [ ] 文档开发

**功能**: TCR/BCR 克隆型分析、多样性指数
**依赖**: `alignment.hpp`
**优先级**: P3

---

#### 19.2 alignment - 序列比对
- [ ] C++ 实现 (`scl/kernel/alignment.hpp`)
- [ ] C-API 接口 (`scl/api/kernel/alignment.h`)
- [ ] 文档开发

**功能**: Smith-Waterman、Needleman-Wunsch
**依赖**: `scl/core/type.hpp`
**优先级**: P3

---

## 迁移统计总结

### 按优先级统计

| 优先级 | 算子数量 | 说明 |
|--------|---------|------|
| **P0** | 15 个 | 核心统计检验、差异分析、基础预处理 |
| **P1** | 26 个 | 常用生物信息学算子、通用算法 |
| **P2** | 28 个 | 高级分析、轨迹推断、空间分析 |
| **P3** | 3 个 | 特殊场景（GNN、免疫组库） |
| **总计** | **72 个** | 完整算子库 |

### 按类别统计

| 类别 | 算子数量 | 主要内容 |
|------|---------|----------|
| **统计分析** | 13 个 | 统计检验、效应量、多重校正 |
| **预处理** | 8 个 | 标准化、质控、特征选择 |
| **聚类降维** | 5 个 | KNN、Leiden、Louvain、PCA |
| **差异分析** | 4 个 | Markers、比较、注释、富集 |
| **空间分析** | 5 个 | 空间邻域、模式、热点、微环境 |
| **轨迹推断** | 4 个 | 拟时序、速率、谱系、转换 |
| **细胞通讯** | 4 个 | 通讯、关联、GRN、共表达 |
| **图算法** | 4 个 | 连通分量、中心性、扩散、传播 |
| **核方法** | 4 个 | 核函数、Gram、MMD、稀疏核 |
| **通用工具** | 18 个 | 相关性、距离、代数、采样等 |
| **特殊场景** | 3 个 | GNN、免疫组库 |

---

## 关键依赖链

### 核心依赖路径

```
基础层（已完成）
├── scl/core/type.hpp ✅
├── scl/core/error.hpp ✅
├── scl/core/memory.hpp ✅
├── scl/core/sparse.hpp ✅
├── scl/core/simd.hpp ✅
├── scl/core/sort.hpp ✅
└── scl/math/stats.hpp ✅

统计基础层（P0 优先）
├── stat_base.hpp → 所有统计算子
├── rank_utils.hpp → 非参数检验
└── group_partition.hpp → 多组比较

核心统计层（P0 优先）
├── auroc.hpp → markers
├── effect_size.hpp → markers
├── ttest.hpp → markers, comparison
├── mwu.hpp → markers, comparison
└── multiple_testing.hpp → enrichment

预处理层（P0 优先）
├── log1p.hpp → normalize
├── normalize.hpp → 下游分析
├── scale.hpp → 下游分析
├── qc.hpp → 质控过滤
└── hvg.hpp → 特征选择

图算法层（P0-P1）
├── neighbors.hpp → leiden, louvain, spatial, doublet
├── metrics.hpp → neighbors
└── correlation.hpp → grn, coexpression

高级分析层（P1-P2）
├── markers.hpp → annotation
├── spatial.hpp → spatial_pattern, hotspot, niche
├── diffusion.hpp → pseudotime
└── kernel.hpp → gram, mmd, sparse_kernel
```

---

## 建议的迁移顺序

### 第一阶段：统计基础（P0，约 2-3 周）

**目标**: 建立完整的统计检验框架

1. **统计基础三件套**
   - `stat_base.hpp` - 统计常量和工具
   - `rank_utils.hpp` - 秩计算
   - `group_partition.hpp` - 分组工具

2. **核心统计检验**
   - `auroc.hpp` - AUROC 计算
   - `effect_size.hpp` - 效应量
   - `ttest.hpp` - t 检验
   - `mwu.hpp` - Mann-Whitney U 检验

3. **差异分析核心**
   - `markers.hpp` - 差异标记基因
   - `multiple_testing.hpp` - 多重检验校正

**验收标准**: 能够完成基本的差异基因分析流程

---

### 第二阶段：预处理与质控（P0，约 1-2 周）

**目标**: 完善数据预处理流程

1. **基础变换**
   - `log1p.hpp` - log1p 变换
   - `normalize.hpp` - 标准化
   - `scale.hpp` - 缩放

2. **质控与特征选择**
   - `qc.hpp` - 质量控制
   - `hvg.hpp` - 高变基因选择

**验收标准**: 能够完成从原始数据到标准化数据的完整流程

---

### 第三阶段：邻域图与聚类（P0-P1，约 2 周）

**目标**: 支持聚类和降维分析

1. **基础算法**
   - `metrics.hpp` - 距离度量
   - `neighbors.hpp` - K 近邻图

2. **聚类算法**
   - `leiden.hpp` - Leiden 聚类
   - `louvain.hpp` - Louvain 聚类
   - `components.hpp` - 连通分量

**验收标准**: 能够完成 KNN 图构建和社区检测

---

### 第四阶段：扩展统计与通用工具（P1，约 2-3 周）

**目标**: 丰富统计检验和数据操作

1. **扩展统计检验**
   - `kruskal_wallis.hpp` - Kruskal-Wallis 检验
   - `ks.hpp` - KS 检验
   - `oneway_anova.hpp` - 单因素方差分析
   - `comparison.hpp` - 多组比较

2. **通用工具**
   - `correlation.hpp` - 相关系数
   - `algebra.hpp` - 线性代数
   - `softmax.hpp` - Softmax
   - `merge.hpp`, `reorder.hpp`, `group.hpp`, `sampling.hpp`

**验收标准**: 支持多种统计检验和数据操作

---

### 第五阶段：生物信息学算子（P1，约 3-4 周）

**目标**: 完善单细胞分析流程

1. **质控扩展**
   - `doublet.hpp` - 双细胞检测
   - `outlier.hpp` - 离群值检测
   - `feature.hpp` - 特征选择

2. **细胞注释**
   - `annotation.hpp` - 细胞类型注释
   - `scoring.hpp` - 基因集打分
   - `enrichment.hpp` - 富集分析

3. **降维与空间**
   - `projection.hpp` - 降维投影
   - `spatial.hpp` - 空间分析

4. **稀疏矩阵优化**
   - `sparse_opt.hpp` - 稀疏矩阵优化

**验收标准**: 支持完整的单细胞分析流程

---

### 第六阶段：高级分析（P2，按需开发）

**目标**: 支持高级生物信息学分析

1. **轨迹推断** (4 个算子)
2. **细胞通讯与调控** (4 个算子)
3. **空间高级分析** (4 个算子)
4. **图算法** (3 个算子)
5. **核方法** (4 个算子)
6. **其他工具** (9 个算子)

**开发策略**: 根据用户需求优先级动态调整

---

### 第七阶段：特殊场景（P3，按需开发）

**目标**: 支持特殊应用场景

1. **图神经网络** - `gnn.hpp`
2. **免疫组库** - `clonotype.hpp`, `alignment.hpp`

**开发策略**: 仅在有明确需求时开发

---

## 开发规范与注意事项

### 迁移原则

1. **禁止改动基础设施**
   - `scl/core/` 模块已完善，禁止修改
   - `scl/math/stats.hpp` 等基础数学库已完善
   - 所有线程工具已集成在 `scl/core/threading.hpp`

2. **代码风格严格遵循 CLAUDE.md**
   - 函数声明格式：`template` → 属性/修饰符 → `auto f(...) -> T`
   - 每个属性独占一行
   - 必须使用 trailing return type
   - 完整的 Doxygen 文档

3. **错误处理层次**
   - 编译时检查：`static_assert`, `SCL_STATIC_CHECK`
   - 运行时检查：`SCL_CHECK` (throws)
   - 调试断言：`SCL_DEBUG_ASSERT` (debug only)

4. **性能优化要求**
   - 所有访问器必须 `SCL_FORCE_INLINE`
   - 使用 `[[likely]]`/`[[unlikely]]` 标注分支
   - 热循环使用 SIMD 优化
   - 使用 `SCL_RESTRICT` 标注非别名指针

### 每个算子的开发流程

#### 阶段 1: C++ 实现 (`scl/kernel/*.hpp` 或 `scl/math/*.hpp`)

**步骤**:
1. 阅读 v0.4 对应文件，理解算法逻辑
2. 按照 CLAUDE.md 规范重写代码
3. 添加完整的 Doxygen 文档
4. 实现 Config 类（继承 `ConfigBase`）
5. 编写单元测试（C++ 层）

**检查清单**:
- [ ] 函数声明格式正确
- [ ] 完整的 Doxygen 文档
- [ ] 错误处理完善
- [ ] 性能优化到位
- [ ] Config 验证逻辑
- [ ] 单元测试通过

#### 阶段 2: C-API 接口 (`scl/api/kernel/*.h` + `*.cpp`)

**步骤**:
1. 设计不透明句柄类型（如 `scl_xxx_t`）
2. 实现创建/销毁函数
3. 实现核心操作函数
4. 添加错误处理（线程本地错误状态）
5. 编写 C API 测试

**检查清单**:
- [ ] 句柄类型定义
- [ ] 内存管理正确（RAII）
- [ ] 错误状态正确传播
- [ ] C API 测试通过
- [ ] 文档注释完整

#### 阶段 3: 文档开发 (`docs/api/*.md`)

**步骤**:
1. 编写算子功能说明
2. 提供使用示例（C++ 和 C API）
3. 说明参数和返回值
4. 列出依赖和性能特性
5. 添加参考文献（如适用）

**检查清单**:
- [ ] 功能描述清晰
- [ ] 代码示例可运行
- [ ] 参数说明完整
- [ ] 性能特性说明
- [ ] 参考文献链接

---

## 快速参考

### P0 算子清单（15 个，必须优先完成）

**统计基础** (3):
- stat_base, rank_utils, group_partition

**核心统计** (4):
- auroc, effect_size, ttest, mwu

**差异分析** (2):
- markers, multiple_testing

**预处理** (5):
- log1p, normalize, scale, qc, hvg

**邻域图** (1):
- neighbors

---

### 常用命令

```bash
# 查看旧版本算子
tree -L 2 scl_v0.4/kernel/

# 创建新算子文件
touch scl/math/stat_base.hpp
touch scl/api/math/stat_base.h
touch scl/api/math/stat_base.cpp

# 运行测试
cd test/C && cmake -B build && cmake --build build && cd build && ctest

# 格式化代码
make format

# 构建文档
make docs-dev
```

---

### 文件路径约定

| 类型 | 路径模板 | 示例 |
|------|---------|------|
| C++ 数学库 | `scl/math/*.hpp` | `scl/math/stat_base.hpp` |
| C++ 算子 | `scl/kernel/*.hpp` | `scl/kernel/markers.hpp` |
| C API 头文件 | `scl/api/math/*.h` 或 `scl/api/kernel/*.h` | `scl/api/math/stat_base.h` |
| C API 实现 | `scl/api/math/*.cpp` 或 `scl/api/kernel/*.cpp` | `scl/api/math/stat_base.cpp` |
| 文档 | `docs/api/*.md` | `docs/api/stat_base.md` |
| 测试 | `test/C/src/test_*.cpp` | `test/C/src/test_stat.cpp` |

---

### 依赖检查清单

开发新算子前，确认以下依赖已就绪：

**统计算子依赖**:
- [ ] `scl/math/stats.hpp` (已有)
- [ ] `scl/core/sort.hpp` (已有)
- [ ] `stat_base.hpp` (需开发)
- [ ] `rank_utils.hpp` (需开发)

**预处理算子依赖**:
- [ ] `scl/core/sparse.hpp` (已有)
- [ ] `scl/core/simd.hpp` (已有)
- [ ] `log1p.hpp` (需开发)

**图算法依赖**:
- [ ] `scl/core/sparse.hpp` (已有)
- [ ] `metrics.hpp` (需开发)
- [ ] `neighbors.hpp` (需开发)

---

### 联系与反馈

- **项目仓库**: `/home/wzq/Code/Projects/scl-core`
- **迁移计划**: `MIGRATION_PLAN.md` (本文档)
- **开发指南**: `CLAUDE.md`
- **版本**: v0.4 → v0.5

---

**最后更新**: 2025-12-31
**总算子数**: 72 个
**P0 优先级**: 15 个
**预计完成时间**: 第一阶段 2-3 周，完整 P0+P1 约 10-12 周
