# argsort.hpp

> scl/core/argsort.hpp · 参数排序（返回排序后的索引）

## 概述

本文件提供参数排序操作，返回对数组进行排序的置换索引，而不是对数组本身进行排序。适用于 top-K 选择、排名和间接排序。

主要特性：
- 返回排序后的索引而不修改原始数据
- SIMD 优化的索引初始化
- 多种变体（就地、缓冲、间接）
- 比使用 lambda 的 std::sort 快 5-10 倍

**头文件**: `#include "scl/core/argsort.hpp"`

---

## 主要 API

### argsort_inplace

排序键并返回对应的索引（升序）。修改键数组。

::: source_code file="scl/core/argsort.hpp" symbol="argsort_inplace" collapsed
:::

**算法说明**

排序键并返回置换索引：
1. 使用 SIMD 将索引初始化为 [0, 1, 2, ..., n-1]
2. 使用键值排序按键排序（键，索引）对
3. 结果：键已排序，索引包含原始位置

**边界条件**

- **空数组**：无操作
- **单个元素**：indices[0] = 0

**数据保证（前置条件）**

- keys.len == indices.len
- 必须分配 indices 缓冲区

**复杂度分析**

- **时间**：O(n log n)
- **空间**：O(1) 辅助空间

**示例**

```cpp
#include "scl/core/argsort.hpp"

Array<Real> keys = ...;
Array<Index> indices(keys.len);

scl::sort::argsort_inplace(keys, indices);
// keys 现在已排序
// indices[i] 包含 keys[i] 的原始位置
```

---

### argsort_inplace_descending

按降序排序键并返回索引。

**复杂度**：O(n log n) 时间，O(1) 空间

---

### argsort_indirect

排序索引而不修改键数组（需要缓冲区）。

**复杂度**：O(n log n) 时间，缓冲区 O(n) 空间

## 相关内容

- [Sort](./sort) - 直接数组排序
- [SIMD](./simd) - 底层 SIMD 抽象

