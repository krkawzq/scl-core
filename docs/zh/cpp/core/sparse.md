# sparse.hpp

> scl/core/sparse.hpp · 使用指针数组的非连续存储稀疏矩阵

## 概述

本文件提供 `Sparse<T, IsCSR>` 结构，SCL-Core 的主要稀疏矩阵数据结构。与传统的连续 CSR/CSC 格式不同，Sparse 使用指针数组，其中每一行/列可以存储在单独的分配中，从而实现与外部数据源的灵活集成。

主要特性：
- 非连续存储（每一行/列可以在单独的分配中）
- 非拥有视图（包装现有数据而不复制）
- 灵活集成（包装 NumPy 数组、内存映射文件）
- 延迟加载支持（按需加载行/列）
- 注册表管理的元数据数组

**头文件**: `#include "scl/core/sparse.hpp"`

---

## 主要 API

### Sparse<T, IsCSR>

具有非连续存储的稀疏矩阵结构。

::: source_code file="scl/core/sparse.hpp" symbol="Sparse" collapsed
:::

**算法说明**

使用指针数组的稀疏矩阵结构：

**内存布局**（CSR 示例）：
```
data_ptrs    = [ptr_to_row0_vals, ptr_to_row1_vals, ptr_to_row2_vals]
indices_ptrs = [ptr_to_row0_cols, ptr_to_row1_cols, ptr_to_row2_cols]
lengths      = [len0, len1, len2]
```

每一行/列可以在单独的内存分配中，与所有数据都在单个数组中的传统连续格式不同。

**设计理念**：
- **非拥有视图**：不管理底层数据的生命周期
- **非连续**：每一行/列可以在单独的分配中
- **灵活**：可以包装外部数据（NumPy、内存映射文件）
- **高效**：通过指针间接访问实现 O(1) 行/列访问

**边界条件**

- **空矩阵**：rows_ == 0 OR cols_ == 0 OR nnz_ == 0
- **空指针**：默认构造函数创建无效矩阵（所有指针为 nullptr）
- **越界访问**：调试版本断言，发布版本未定义行为
- **无效矩阵**：调用 `valid()` 或 `operator bool()` 检查有效性

**数据保证（前置条件）**

- 如果使用指针数组构造，它们必须比 Sparse 对象存活更久
- 行/列索引必须在有效范围 [0, primary_dim()) 内
- 对于 CSR：行索引 i 必须在 [0, rows_) 内
- 对于 CSC：列索引 j 必须在 [0, cols_) 内
- 如果使用 `new_registered()`，元数据数组已注册到 HandlerRegistry

**复杂度分析**

- **构造**：O(1) - 仅存储指针
- **行/列访问**：O(1) - 指针解引用
- **内存**：(2*primary_dim + 1) 指针 + nnz 元素（vs 连续格式的 2*nnz + primary_dim+1）

**示例**

```cpp
#include "scl/core/sparse.hpp"

// 从零开始创建 CSR 矩阵
auto matrix = scl::Sparse<Real, true>::new_registered(
    1000,    // 行数
    500,     // 列数
    10000    // 总非零元素数
);

// 访问行
Index row_idx = 0;
auto vals = matrix.primary_values(row_idx);
auto idxs = matrix.primary_indices(row_idx);
Index len = matrix.primary_length(row_idx);

// 迭代行中的非零元素
for (Index j = 0; j < len; ++j) {
    Real value = vals[j];
    Index col = idxs[j];
    // 处理 (row_idx, col) 处的值
}

// 包装现有数据（非拥有）
Real* row0_data = ...;
Index* row0_indices = ...;
Pointer* data_ptrs = ...;
Pointer* indices_ptrs = ...;
Index* lengths = ...;

scl::Sparse<Real, true> wrapper(
    data_ptrs, indices_ptrs, lengths,
    1000, 500, 10000
);
// Wrapper 指向现有数据，不拥有它
```

---

### primary_values / row_values / col_values

获取行（CSR）或列（CSC）中的值的数组视图。

::: source_code file="scl/core/sparse.hpp" symbol="primary_values" collapsed
:::

**算法说明**

返回指定行（CSR）或列（CSC）中的值的 Array<T> 视图：
- CSR: `primary_values(i)` = 行 i 中的值
- CSC: `primary_values(j)` = 列 j 中的值

该方法执行指针解引用：`data_ptrs[i]` 转换为 `T*` 并用 `Array<T>` 包装，长度来自 `lengths[i]`。

**边界条件**

- **越界索引**：调试版本断言，发布版本未定义行为
- **data_ptrs 中的空指针**：调试版本断言，发布版本未定义行为
- **零长度行/列**：返回空 Array<T>（ptr 可能为 nullptr）

**数据保证（前置条件）**

- 索引必须在 [0, primary_dim()) 内
- `data_ptrs` 不得为 nullptr
- `data_ptrs[i]` 必须指向至少 `lengths[i]` 个元素的有效数组

**复杂度分析**

- **时间**：O(1) - 指针解引用和 Array 构造
- **空间**：O(1) - 返回 Array 视图（非拥有）

**示例**

```cpp
Sparse<Real, true> matrix = ...;  // CSR

Index row = 5;
Array<Real> vals = matrix.primary_values(row);
// vals 是行 5 中值的视图

// 迭代
for (Index j = 0; j < vals.size(); ++j) {
    Real value = vals[j];
    // 处理值
}
```

---

### primary_indices / row_indices / col_indices

获取列索引（CSR）或行索引（CSC）的数组视图。

::: source_code file="scl/core/sparse.hpp" symbol="primary_indices" collapsed
:::

**算法说明**

返回列索引（CSR）或行索引（CSC）的 Array<Index> 视图：
- CSR: `primary_indices(i)` = 行 i 中的列索引
- CSC: `primary_indices(j)` = 列 j 中的行索引

**边界条件**

- **越界索引**：调试版本断言，发布版本未定义行为
- **indices_ptrs 中的空指针**：调试版本断言，发布版本未定义行为
- **零长度行/列**：返回空 Array<Index>

**数据保证（前置条件）**

- 索引必须在 [0, primary_dim()) 内
- `indices_ptrs` 不得为 nullptr
- `indices_ptrs[i]` 必须指向至少 `lengths[i]` 个 Index 元素的有效数组

**复杂度分析**

- **时间**：O(1) - 指针解引用和 Array 构造
- **空间**：O(1) - 返回 Array 视图

**示例**

```cpp
Sparse<Real, true> matrix = ...;  // CSR

Index row = 5;
Array<Real> vals = matrix.primary_values(row);
Array<Index> cols = matrix.primary_indices(row);

// 同时访问值和列
for (Index j = 0; j < vals.size(); ++j) {
    Real value = vals[j];
    Index col = cols[j];
    // (row, col) 处的值是 value
}
```

---

### primary_length / row_length / col_length

获取行（CSR）或列（CSC）中的非零元素数。

::: source_code file="scl/core/sparse.hpp" symbol="primary_length" collapsed
:::

**算法说明**

返回指定行（CSR）或列（CSC）中的非零元素数：
- CSR: `primary_length(i)` = 行 i 中的非零元素数
- CSC: `primary_length(j)` = 列 j 中的非零元素数

简单返回 `lengths[i]`，其中 i 是主维度索引。

**边界条件**

- **越界索引**：调试版本断言，发布版本未定义行为
- **lengths 中的空指针**：调试版本断言，发布版本未定义行为

**数据保证（前置条件）**

- 索引必须在 [0, primary_dim()) 内
- `lengths` 不得为 nullptr

**复杂度分析**

- **时间**：O(1) - 数组访问
- **空间**：O(1)

**示例**

```cpp
Sparse<Real, true> matrix = ...;  // CSR

Index row = 5;
Index nnz_in_row = matrix.primary_length(row);
// nnz_in_row 是行 5 中的非零元素数
```

---

### new_registered

创建具有注册表管理的元数据数组的 Sparse 矩阵的工厂方法。

::: source_code file="scl/core/sparse.hpp" symbol="new_registered" collapsed
:::

**算法说明**

分配元数据数组（data_ptrs, indices_ptrs, lengths）并在 HandlerRegistry 中注册：
1. 为指针数组分配对齐内存（每个 primary_dim 元素）
2. 为 lengths 数组分配对齐内存（primary_dim 个 Index 元素）
3. 将所有数组初始化为零
4. 将所有数组注册到 HandlerRegistry
5. 返回指向已分配数组的 Sparse 视图

**边界条件**

- **分配失败**：返回空 Sparse（所有指针为 nullptr）
- **零维度**：返回有效但为空的矩阵
- **大维度**：如果分配超过系统限制可能失败

**数据保证（前置条件）**

- rows >= 0, cols >= 0, total_nnz >= 0
- HandlerRegistry 必须可用（通过 `scl::get_registry()`）

**复杂度分析**

- **时间**：零初始化 O(primary_dim)
- **空间**：元数据数组 O(primary_dim)

**示例**

```cpp
#include "scl/core/sparse.hpp"

// 创建具有注册元数据的 CSR 矩阵
auto matrix = scl::Sparse<Real, true>::new_registered(
    1000,    // 行数
    500,     // 列数
    10000    // 总非零元素数（信息性，不用于分配）
);

// 矩阵现在具有有效的元数据数组
// 但实际的行数据数组必须单独分配
// 并分配给 data_ptrs[i] 和 indices_ptrs[i]

// 完成后，在转移到 Python 之前取消注册
matrix.unregister_metadata();
```

---

### unregister_metadata

从 HandlerRegistry 取消注册元数据数组。

::: source_code file="scl/core/sparse.hpp" symbol="unregister_metadata" collapsed
:::

**算法说明**

从 HandlerRegistry 取消注册三个元数据数组（data_ptrs, indices_ptrs, lengths）并将所有指针设置为 nullptr，使矩阵无效。

在将元数据数组的所有权转移到 Python 或其他外部代码之前使用。

**边界条件**

- **已无效**：安全调用，不执行任何操作
- **未注册**：安全调用，可能记录警告但不会崩溃

**数据保证（前置条件）**

- 无 - 即使矩阵无效也可以安全调用

**复杂度分析**

- **时间**：O(1) - 注册表查找和指针重置
- **空间**：O(1)

**示例**

```cpp
auto matrix = scl::Sparse<Real, true>::new_registered(1000, 500, 10000);

// 使用矩阵...

// 在将所有权转移到 Python 之前
matrix.unregister_metadata();
// 矩阵现在无效（所有指针为 nullptr）
// 元数据数组仍已分配但不再由注册表跟踪
```

---

## 工具方法

### rows / cols / nnz / primary_dim / secondary_dim

维度访问器，全部 O(1)。

**复杂度**：所有方法 O(1)

### empty / valid

状态查询，全部 O(1)。

**复杂度**：所有方法 O(1)

---

## 类型别名

```cpp
using CSR = Sparse<Real, true>;   // 具有 Real 值的 CSR 矩阵
using CSC = Sparse<Real, false>;  // 具有 Real 值的 CSC 矩阵
```

## 设计说明

### 非拥有视图

Sparse 是非拥有视图 - 它不：
- 分配或释放数据数组
- 管理底层内存的生命周期
- 复制数据

调用者负责：
- 分配数据数组
- 管理它们的生命周期
- 确保它们比 Sparse 对象存活更久

### 非连续 vs 连续

**连续格式**（传统 CSR）：
- 单个数据数组：`[v0, v1, v2, ...]`
- 单个索引数组：`[i0, i1, i2, ...]`
- Indptr 数组：`[0, 3, 5, 9, ...]`
- 对顺序访问缓存友好
- 对异构数据灵活性较低

**非连续格式**（SCL Sparse）：
- 指针数组：`[ptr0, ptr1, ptr2, ...]`
- 每个指针指向单独的分配
- 长度数组：`[len0, len1, len2, ...]`
- 更灵活（包装外部数据）
- 可能有缓存未命中（指针间接访问）

### 注册表集成

使用 `new_registered()` 时：
- 元数据数组注册到 HandlerRegistry
- 启用自动跟踪和清理
- 在转移到 Python 之前调用 `unregister_metadata()`
- 如果需要，可以单独注册实际数据数组

## 相关内容

- [类型系统](./types) - 用于视图的 Array<T> 类型
- [注册表](./registry) - 用于内存跟踪的 HandlerRegistry
- [内存管理](./memory) - 对齐分配函数

