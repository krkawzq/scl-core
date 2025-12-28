# sort.hpp

> scl/core/sort.hpp · 通过 Google Highway VQSort 实现高性能排序

## 概述

本文件提供使用 Google Highway VQSort 后端的 SIMD 加速排序。针对数值计算优化，具有架构无关的 SIMD 加速。

主要特性：
- SIMD 加速排序（比 std::sort 快 2-5 倍）
- 单数组排序（升序/降序）
- 键值对排序
- 架构无关（AVX2/AVX-512/NEON）

**头文件**: `#include "scl/core/sort.hpp"`

---

## 主要 API

### sort

使用 SIMD 优化的 VQSort 按升序排序数组。

::: source_code file="scl/core/sort.hpp" symbol="sort" collapsed
:::

**算法说明**

使用 Google Highway VQSort（向量化快速排序变体）排序数组：
- 使用 SIMD 分区和比较
- 针对现代 CPU 缓存层次结构优化
- 对于 > 100 个元素的数组性能最佳

**边界条件**

- **空数组**：无操作
- **单个元素**：无操作
- **已排序**：最坏情况 O(n log n)，通常更快

**数据保证（前置条件）**

- data.ptr 必须有效或为 nullptr（如果 data.len == 0）
- T 必须可排序（具有 < 运算符）

**复杂度分析**

- **时间**：平均和最坏情况 O(n log n)
- **空间**：递归栈 O(log n)

**示例**

```cpp
#include "scl/core/sort.hpp"

Array<Real> data = ...;
scl::sort::sort(data);  // 按升序排序
```

---

### sort_descending

按降序排序数组。

::: source_code file="scl/core/sort.hpp" symbol="sort_descending" collapsed
:::

**复杂度**：O(n log n) 时间，O(log n) 空间

---

### sort_key_value

排序键值对，保持对应关系。

::: source_code file="scl/core/sort.hpp" symbol="sort_key_value" collapsed
:::

**复杂度**：O(n log n) 时间，O(log n) 空间

## 相关内容

- [Argsort](./argsort) - 参数排序（返回排序后的索引）
- [SIMD](./simd) - 底层 SIMD 抽象

