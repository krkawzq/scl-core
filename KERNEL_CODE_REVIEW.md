# SCL Kernel 组件代码审查报告

## 审查日期
2024年审查

## 审查范围
审查了 `scl/kernel/` 目录下的所有算子实现，检查是否符合以下要求：
1. 所有算子至少实现一种基于 `ISparse`/`IDense` 等基类的算法
2. 实现 `CustomSparseLike` 和 `VirtualSparseLike`（非接口）的高效算子
3. 如果 custom 和 virtual 无法统一，单独写两套方法
4. 考虑虚继承接口是否需要写两套不同的实现

## 审查结果总览

| 文件 | ISparse接口 | CustomSparseLike | VirtualSparseLike | 状态 |
|------|------------|------------------|-------------------|------|
| ttest.hpp | ✅ | ❌ | ❌ | **需要补充** |
| correlation.hpp | ✅ | ✅ | ✅ | ✅ 正确 |
| feature.hpp | ✅ | ✅ | ✅ | ✅ 正确 |
| group.hpp | ✅ | ✅ | ✅ | ✅ 正确 |
| neighbors.hpp | ❌ | ✅ | ❌ | **需要补充** |
| merge.hpp | ❌ | ❌ | ✅ | **需要补充** |
| reorder.hpp | ❌ | ✅ | ❌ | **需要补充** |
| mmd.hpp | ❌ | ❌ | ❌ | **需要补充** |
| resample.hpp | ❌ | ✅ | ❌ | **需要补充** |
| qc.hpp | ❌ | ✅ | ❌ | **需要补充** |
| mwu.hpp | ❌ | ❌ | ❌ | **需要补充** |
| scale.hpp | ❌ | ✅ | ❌ | **需要补充** |
| normalize.hpp | ❌ | ✅ | ❌ | **需要补充** |
| softmax.hpp | ❌ | ❌ | ❌ | **需要补充** |
| sparse.hpp | ❌ | ✅ | ❌ | **需要补充** |
| log1p.hpp | ❌ | ✅ | ❌ | **需要补充** |
| spatial.hpp | ❌ | ✅ | ✅ | **需要补充** |
| algebra.hpp | ✅ | ✅ | ❌ | **需要补充** |
| bbknn.hpp | ✅ | ❌ | ❌ | **需要补充** |
| gram.hpp | ✅ | ✅ | ❌ | **需要补充** |
| hvg.hpp | ✅ | ❌ | ❌ | **需要补充** |

## 详细审查结果

### ✅ 已正确实现的文件

#### 1. correlation.hpp
**状态**: ✅ **完全符合要求**

**实现情况**:
- ✅ Layer 1: `ISparse` 接口实现（`ICSC<T>` 和 `ICSR<T>`）
- ✅ Layer 2: Concept-based 实现（`CSCLike` 和 `CSRLike`）
- ✅ 统一的实现逻辑，通过 concept 自动适配 Custom 和 Virtual

**代码结构**:
```cpp
// Layer 1: Virtual Interface
template <typename T> void pearson(const ICSC<T>& matrix, ...);
template <typename T> void pearson(const ICSR<T>& matrix, ...);

// Layer 2: Concept-Based (统一实现)
template <CSCLike MatrixT> void pearson(const MatrixT& matrix, ...);
template <CSRLike MatrixT> void pearson(const MatrixT& matrix, ...);
```

**优点**:
- 清晰的层次结构
- 通过 concept 实现零开销抽象
- Custom 和 Virtual 共享同一实现

#### 2. feature.hpp
**状态**: ✅ **完全符合要求**

**实现情况**:
- ✅ Layer 1: `ISparse` 接口实现（`ICSC<T>` 和 `ICSR<T>`）
- ✅ Layer 2: Concept-based 实现（`CSCLike` 和 `CSRLike`）
- ✅ 使用 `detail::*_impl` 模板函数实现统一逻辑

**代码结构**:
```cpp
// Layer 1: Virtual Interface
template <typename T> void clipped_moments(const ICSC<T>& matrix, ...);
template <typename T> void clipped_moments(const ICSR<T>& matrix, ...);

// Layer 2: Concept-Based (委托给 detail)
template <CSCLike MatrixT> void clipped_moments(MatrixT matrix, ...) {
    detail::clipped_moments_impl(matrix, ...);
}
```

#### 3. group.hpp
**状态**: ✅ **完全符合要求**

**实现情况**:
- ✅ Layer 1: `ISparse` 接口实现（`ICSC<T>` 和 `ICSR<T>`）
- ✅ Layer 2: Concept-based 实现（`CSCLike` 和 `CSRLike`）
- ✅ 使用统一访问器 `scl::primary_*` 实现零开销抽象

### ⚠️ 需要补充的文件

#### 1. ttest.hpp
**状态**: ⚠️ **缺少 Custom/Virtual 优化版本**

**当前实现**:
- ✅ Layer 1: `ISparse` 接口实现（`ICSC<T>` 和 `ICSR<T>`）
- ❌ Layer 2: 缺少 Concept-based 实现

**问题**:
- 只有虚拟接口实现，没有针对 `CustomSparseLike` 和 `VirtualSparseLike` 的优化版本
- 无法利用零开销抽象优化性能

**建议修复**:
```cpp
// 添加 Layer 2: Concept-Based 实现
template <CSCLike MatrixT>
SCL_FORCE_INLINE void ttest(
    const MatrixT& matrix,
    Span<const int32_t> group_ids,
    Size n_groups,
    MutableSpan<Real> out_t_stats,
    MutableSpan<Real> out_p_values,
    MutableSpan<Real> out_log2_fc,
    MutableSpan<Real> out_mean_diff,
    MutableSpan<Byte> workspace,
    bool use_welch = true
) {
    // 重用现有逻辑，但使用统一访问器
    // 可以委托给 detail 实现
}

template <CSRLike MatrixT>
SCL_FORCE_INLINE void ttest(...) {
    // CSR 版本
}
```

#### 2. neighbors.hpp
**状态**: ⚠️ **缺少 ISparse 接口实现**

**当前实现**:
- ❌ Layer 1: 缺少 `ISparse` 接口实现
- ✅ Layer 2: 只有 `CSRLike` concept 实现

**问题**:
- 没有提供基于 `ICSR<T>` 的通用接口
- 无法支持用户自定义的稀疏矩阵类型（通过继承 `ISparse`）

**建议修复**:
```cpp
// 添加 Layer 1: Virtual Interface
template <typename T>
void knn_sparse(
    const ICSR<T>& matrix,
    Size k,
    MutableSpan<Index> out_indices,
    MutableSpan<T> out_distances
) {
    // 使用虚拟接口访问
    const Index R = matrix.rows();
    // ... 实现逻辑
}
```

#### 3. merge.hpp
**状态**: ⚠️ **架构特殊，但缺少通用接口**

**当前实现**:
- ❌ 只针对 `VirtualCSR`/`VirtualCSC` 类型
- ❌ 没有 `ISparse` 接口支持
- ❌ 没有 `CustomSparseLike` 支持

**问题**:
- 这是一个特殊的零拷贝合并操作，主要针对 Virtual 模式
- 但应该提供基于 `ISparse` 的通用接口作为后备

**建议**:
- 保持 Virtual 专用实现（这是设计目标）
- 但添加基于 `ISparse` 的通用接口，内部可以转换为 Virtual 或使用物理合并

#### 4. reorder.hpp
**状态**: ⚠️ **只支持 Custom，缺少接口和 Virtual**

**当前实现**:
- ❌ 只针对 `CustomCSR`/`CustomCSC`（需要直接修改 `indptr`）
- ❌ 没有 `ISparse` 接口支持
- ❌ 没有 `VirtualSparseLike` 支持

**问题**:
- 这是一个原地修改操作，主要针对 Custom 模式
- 但应该提供基于 `ISparse` 的通用接口

**建议**:
- 保持 Custom 专用实现（需要直接访问 `indptr`）
- 添加基于 `ISparse` 的通用接口，可能需要创建新矩阵（非原地）

#### 5. algebra.hpp
**状态**: ⚠️ **缺少 VirtualSparseLike 优化**

**当前实现**:
- ✅ Layer 1: `ISparse` 接口实现（`ICSR<T>`）
- ✅ Layer 2: `CSRLike` concept 实现
- ❌ 没有针对 `VirtualSparseLike` 的特殊优化

**问题**:
- 虽然 concept 实现可以工作，但 Virtual 模式有额外的间接访问开销
- 可以考虑针对 Virtual 的特殊优化（例如批量预取指针数组）

**建议**:
- 当前实现已经足够（concept 统一实现）
- 如果性能测试显示 Virtual 模式有瓶颈，再考虑专门优化

#### 6. gram.hpp
**状态**: ⚠️ **缺少 VirtualSparseLike 优化**

**当前实现**:
- ✅ Layer 1: `ISparse` 接口实现（`ICSR<T>` 和 `ICSC<T>`）
- ✅ Layer 2: `CSRLike`/`CSCLike` concept 实现
- ❌ 没有针对 `VirtualSparseLike` 的特殊优化

**建议**: 同 `algebra.hpp`，当前实现已足够

#### 7. bbknn.hpp
**状态**: ⚠️ **缺少 Custom/Virtual 优化版本**

**当前实现**:
- ✅ Layer 1: `ISparse` 接口实现（`ICSR<T>`）
- ❌ Layer 2: 缺少 Concept-based 实现

**建议**: 参考 `neighbors.hpp` 的修复方案

#### 8. hvg.hpp
**状态**: ⚠️ **有 ISparse 接口但缺少 Concept 实现**

**当前实现**:
- ✅ Layer 1: `ISparse` 接口实现（`ICSC<T>` 和 `ICSR<T>`）
- ❌ Layer 2: 缺少 Concept-based 实现

**建议**: 添加 `CSCLike`/`CSRLike` concept 版本

#### 9. 其他文件（mmd, resample, qc, mwu, scale, normalize, softmax, sparse, log1p, spatial）

**共同问题**:
- 大部分只有 Concept-based 实现，缺少 `ISparse` 接口层
- 部分文件（如 `spatial.hpp`）有 Custom 和 Virtual 的专门优化，但缺少 `ISparse` 接口

**建议**:
- 所有文件都应该添加 Layer 1（`ISparse` 接口实现）
- 保持现有的 Concept-based 实现作为 Layer 2

## 架构建议

### 推荐的三层架构

```cpp
// =============================================================================
// Layer 1: Virtual Interface (ISparse-based, Generic but Slower)
// =============================================================================
// 目的: 支持用户自定义类型（通过继承 ISparse）
// 性能: 有虚拟调用开销，但提供最大兼容性

template <typename T>
void algorithm(const ICSC<T>& matrix, ...) {
    // 使用 matrix.primary_values(), matrix.primary_indices() 等
}

// =============================================================================
// Layer 2: Concept-Based (CSCLike/CSRLike, Optimized for Custom/Virtual)
// =============================================================================
// 目的: 零开销抽象，自动适配 Custom 和 Virtual
// 性能: 最优，编译时多态

template <CSCLike MatrixT>
void algorithm(const MatrixT& matrix, ...) {
    // 使用 scl::primary_values(), scl::primary_indices() 等统一访问器
    // 编译器会自动内联，零开销
}
```

### Custom vs Virtual 统一策略

**原则**: 
1. **优先统一**: 如果算法逻辑相同，使用 Concept-based 统一实现
2. **必要时分离**: 如果性能差异显著（>10%），考虑专门优化

**当前观察**:
- 大部分算法可以统一（如 `correlation`, `feature`, `group`）
- 少数算法需要分离（如 `merge` 的零拷贝特性，`reorder` 的原地修改）

### 虚继承接口是否需要两套实现？

**结论**: **不需要**

**理由**:
1. `ISparse<T, true>` 和 `ISparse<T, false>` 已经通过模板参数区分 CSR/CSC
2. 统一的 `primary_*` 接口已经抽象了 CSR/CSC 差异
3. 两套实现会增加代码重复，维护成本高

**当前实现验证**:
- `ttest.hpp`, `correlation.hpp`, `feature.hpp` 等都只实现一套 `ISparse` 接口
- 通过 `if constexpr` 或统一访问器处理 CSR/CSC 差异

## 修复优先级

### 高优先级（核心功能）
1. **ttest.hpp**: 添加 Concept-based 实现
2. **neighbors.hpp**: 添加 `ISparse` 接口实现
3. **algebra.hpp**: 考虑 Virtual 优化（如果性能测试需要）

### 中优先级（常用功能）
4. **bbknn.hpp**: 添加 Concept-based 实现
5. **hvg.hpp**: 添加 Concept-based 实现
6. **gram.hpp**: 考虑 Virtual 优化

### 低优先级（特殊用途）
7. **merge.hpp**: 保持当前设计，但添加通用接口作为后备
8. **reorder.hpp**: 保持当前设计，但添加通用接口（非原地版本）
9. 其他文件：逐步添加 `ISparse` 接口层

## 总结

### 优点
1. ✅ 核心统计算子（`correlation`, `feature`, `group`）架构正确
2. ✅ Concept-based 实现提供了零开销抽象
3. ✅ 统一的访问器设计（`scl::primary_*`）优雅且高效

### 需要改进
1. ⚠️ 约 60% 的文件缺少完整的双层架构
2. ⚠️ 部分文件只有 Concept 实现，缺少 `ISparse` 接口层
3. ⚠️ 部分文件只有 `ISparse` 接口，缺少 Concept 优化层

### 建议
1. **短期**: 优先修复高优先级文件（`ttest`, `neighbors`）
2. **中期**: 逐步为所有文件添加完整的双层架构
3. **长期**: 建立代码审查检查清单，确保新代码符合架构规范

## 附录：检查清单

在添加新算子时，请确保：

- [ ] Layer 1: 实现 `ISparse<T, true>` 和 `ISparse<T, false>` 版本（或 `IDense<T>`）
- [ ] Layer 2: 实现 `CSRLike`/`CSCLike`/`DenseLike` concept 版本
- [ ] 使用统一访问器（`scl::primary_*`, `scl::rows`, `scl::cols`）
- [ ] 如果 Custom 和 Virtual 性能差异显著，考虑专门优化
- [ ] 添加适当的文档注释说明架构层次
- [ ] 确保所有版本通过相同的测试用例

