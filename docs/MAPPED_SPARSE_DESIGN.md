# Mapped Sparse Backend 调研与设计文档

## 1. 现有架构分析

### 1.1 类型系统 (scl/core/type.hpp)

SCL 采用**概念约束 + 统一访问器**的设计：

```cpp
// SparseLike concept - 核心接口约束
template <typename M, bool IsCSR>
concept SparseLike = requires(const M& m, Index i) {
    typename M::ValueType;
    typename M::Tag;
    { scl::rows(m) } -> std::convertible_to<Index>;
    { scl::cols(m) } -> std::convertible_to<Index>;
    { scl::nnz(m) } -> std::convertible_to<Index>;
    // CSR: row_values(i), row_indices(i), row_length(i)
    // CSC: col_values(j), col_indices(j), col_length(j)
};

// 统一访问器 - 支持 POD 成员和方法访问
template <typename M>
SCL_NODISCARD SCL_FORCE_INLINE auto primary_values(const M& mat, Index i) {
    if constexpr (tag_is_csr_v<typename M::Tag>) {
        return mat.row_values(i);
    } else {
        return mat.col_values(i);
    }
}
```

**关键特性:**
- 编译期静态分派（零开销抽象）
- POD 结构体与方法访问统一
- CSR/CSC 模板统一处理

### 1.2 稀疏矩阵类型 (scl/core/sparse.hpp)

| 类型 | 存储模式 | 特点 |
|------|---------|------|
| `ISparse<T, IsCSR>` | 虚接口 | 多态分派，惰性加载 |
| `CustomSparse<T, IsCSR>` | 连续数组 | data/indices/indptr 直接访问 |
| `VirtualSparse<T, IsCSR>` | 指针数组 | data_ptrs/indices_ptrs/lengths |

### 1.3 Kernel 算子模式 (scl/kernel/)

现有 kernel 采用**双路径优化**：

```cpp
// 统一分发器
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void primary_sums_fast(const MatrixT& matrix, Array<T> output) {
    if constexpr (MappedSparseLike<MatrixT, IsCSR>) {
        primary_sums_mapped_dispatch(matrix, output);   // Mapped 流式
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        primary_sums_custom_fast(matrix, output);       // 批量 SIMD
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        primary_sums_virtual_fast(matrix, output);      // 行级 SIMD
    }
}
```

**现有 Kernel 列表:**
| 模块 | 功能 | 备注 |
|------|-----|------|
| sparse.hpp | 行/列统计（sum, mean, var, nnz） | ✅ 已有 Custom/Virtual/Mapped 优化 |
| normalize.hpp | 归一化、缩放 | ✅ 已有 Custom/Virtual 优化 |
| slice.hpp | 切片操作 | ✅ unsafe 版本 |
| log1p.hpp | log(1+x) 变换 | |
| scale.hpp | 标准化 | |
| hvg.hpp | 高变异基因选择 | |
| qc.hpp | 质量控制 | |
| resample.hpp | 重采样 | |
| ttest.hpp | T检验 | |
| mwu.hpp | Mann-Whitney U 检验 | |
| neighbors.hpp | K近邻 | |
| correlation.hpp | 相关性计算 | |
| algebra.hpp | 矩阵代数运算 | |
| gram.hpp | Gram 矩阵 | |
| mmd.hpp | MMD 距离 | |

---

## 2. IO 模块架构 (scl/io/)

### 2.1 MappedSparse 类型 (mmatrix.hpp)

```cpp
// 内存映射稀疏矩阵
template <typename T, bool IsCSR>
class MappedCustomSparse {
    MappedArray<T> _data;
    MappedArray<Index> _indices;
    MappedArray<Index> _indptr;

    // 满足 SparseLike - 可直接用于泛型算法
    Array<T> row_values(Index i) const;
    Array<Index> row_indices(Index i) const;
    Index row_length(Index i) const;

    // 转换方法
    CustomSparse<T, IsCSR> as_view() const;   // 零拷贝视图
    OwnedSparse<T, IsCSR> materialize() const; // 深拷贝到堆
};

// 虚拟切片（间接寻址）
template <typename T, bool IsCSR>
class MappedVirtualSparse {
    const Index* _map;  // 逻辑行 -> 物理行映射
    // 支持 O(1) 行切片
};
```

### 2.2 HDF5 高级查询优化 (h5_tools.hpp)

#### Zone Map 过滤
```cpp
struct ZoneMapEntry {
    Index min_col, max_col;  // 块内列范围
    Index nnz;
    hsize_t chunk_idx;

    bool may_contain(Index q_min, Index q_max) const {
        return !(max_col < q_min || min_col > q_max);
    }
};
```

#### 自适应交集算法
```cpp
enum class IntersectionStrategy {
    Galloping,     // |query| << |chunk|, O(Q·log(C/Q))
    BinarySearch,  // |query| < |chunk|,  O(Q·log C)
    LinearMerge    // |query| ≈ |chunk|,  O(Q + C)
};
```

### 2.3 自适应调度器 (scheduler.hpp)

基于**污染指数模型 (Contamination Index)**:

```
CI = β · (Q/N) · C
```

- β: 聚类因子 [0.2, 1.0]
- Q: 查询列数
- N: 总列数
- C: 块大小

---

## 3. Mapped Backend 设计方案

### 3.1 核心问题

**Mapped 数据的特点:**
1. 随机 IO 代价高昂（HDD 100x, SSD 10x 延迟）
2. 顺序读取接近内存速度
3. 页面缓存机制影响实际性能
4. 不支持原地修改（需 materialize）

### 3.2 设计原则

```
┌─────────────────────────────────────────────────────────────┐
│                    一套算子，全 Backend 通用                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │ CustomSparse │    │VirtualSparse│    │ MappedSparse    │ │
│  │   (连续)     │    │  (指针)      │    │ (内存映射)      │ │
│  └──────┬──────┘    └──────┬──────┘    └────────┬────────┘ │
│         │                  │                     │          │
│         └──────────────────┼─────────────────────┘          │
│                            │                                │
│                    ┌───────▼───────┐                        │
│                    │ SparseLike<T> │  ← 统一概念约束         │
│                    │   Concept     │                        │
│                    └───────┬───────┘                        │
│                            │                                │
│              ┌─────────────┼─────────────┐                  │
│              │             │             │                  │
│        ┌─────▼────┐  ┌─────▼────┐  ┌─────▼──────┐          │
│        │ Custom   │  │ Virtual  │  │ Mapped     │          │
│        │ Fast Path│  │ Fast Path│  │ Fast Path  │          │
│        └──────────┘  └──────────┘  └────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Mapped 专用优化策略

#### 策略一: 流式单遍扫描

```cpp
// 传统算法 (2遍扫描)
for row: sum += values
mu = sum / N
for row: var += (val - mu)^2

// Mapped 优化 (1遍扫描 - Welford)
for row:
    delta = val - mean
    mean += delta / n
    m2 += delta * (val - mean)
var = m2 / (n - ddof)
```

#### 策略二: 块级处理 + 预取

```cpp
constexpr Size CHUNK_SIZE = 256;  // 行

// 预取提示
hint_prefetch(matrix);

// 按块并行处理
for (chunk : chunks) {
    parallel_for(chunk.start, chunk.end, [&](Index i) {
        // SIMD 处理
    });
}
```

### 3.4 算子分类与实现策略

| 算子类型 | Mapped 策略 | 示例 |
|---------|------------|------|
| 只读统计 | 流式单遍扫描 | sum, mean, var, nnz |
| 行级操作 | 顺序扫描 + 缓存 | row_normalize, scale |
| 列级操作 | 块级缓存 + 交集算法 | col_sum, feature_select |
| 切片操作 | 延迟物化 + Zone Map | slice_rows, slice_cols |
| 修改操作 | 物化后处理 | log1p, normalize |

---

## 4. 已实现组件

### 4.1 mapped_common.hpp

- `MappedSparseLike<M, IsCSR>` concept
- `BackendType` 枚举
- `StreamConfig` 配置
- `ChunkIterator` 分块迭代
- `WelfordState<T>` 单遍方差
- `RowStats<T>` 行统计

### 4.2 sparse_mapped_impl.hpp

- `primary_sums_mapped()` - 行求和
- `primary_means_mapped()` - 行均值
- `primary_variances_mapped()` - 行方差（单遍）
- `primary_nnz_mapped()` - 行非零计数

### 4.3 sparse_fast_impl.hpp 更新

统一调度器已添加 Mapped 分支：
```cpp
if constexpr (MappedSparseLike<MatrixT, IsCSR>) {
    primary_sums_mapped_dispatch(...);
} else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
    primary_sums_custom_fast(...);
} else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
    primary_sums_virtual_fast(...);
}
```

---

## 5. 预期性能提升

| 场景 | 传统方法 | Mapped 优化 | 提升 |
|------|---------|------------|------|
| 行统计（1M 行） | 随机扫描 | 顺序流式 | 5-10x |
| 列切片（1% 列） | 全量读取 | Zone Map 过滤 | 50-100x |
| 子矩阵提取 | 物化全部 | 延迟物化 | 10-50x |

---

## 6. 下一步行动

1. [x] 实现 `mapped_common.hpp` - MappedSparseLike concept
2. [x] 实现 `sparse_mapped_impl.hpp` - Mapped 统计算子
3. [x] 更新 `sparse_fast_impl.hpp` - 添加 Mapped 分支
4. [ ] 更新 C API - 添加 Mapped 导出
5. [ ] Python 集成 - `_kernel_bridge.py` 添加 Mapped 支持
6. [ ] 其他 kernel 添加 Mapped 版本
