# 文档标准

SCL-Core 使用双文件文档系统，严格分离实现和规范。这种分离使代码保持可读性，同时提供全面的 API 文档。

## 哲学

**清晰代码 + 全面文档 = 可维护代码库**

- 实现文件（.hpp）应该在没有大量注释的情况下可读
- API 文档文件（.h）应该是完整的规范
- 文档是机器可解析和人类可读的
- 快速查询应该提取接口而不是噪音

## 双文件系统

### 实现文件 (.hpp)

实现文件包含实际代码和最少的内联注释。代码本身应该通过清晰的命名和结构自我文档化。

**文件结构：**

```cpp
// =============================================================================
// FILE: scl/kernel/normalize.hpp
// BRIEF: 行/列归一化内核
// =============================================================================
#pragma once

#include "scl/kernel/normalize.h"
#include "scl/core/sparse.hpp"
#include "scl/threading/parallel_for.hpp"

namespace scl::kernel::normalize {

template <CSRLike MatrixT>
void normalize_rows_inplace(MatrixT& matrix, NormMode mode, Real eps) {
    // 使用 Kahan 求和以提高数值稳定性
    parallel_for(Size(0), matrix.rows(), [&](Index i) {
        auto vals = matrix.primary_values(i);
        const Index len = matrix.primary_length(i);
        
        if (len == 0) return;
        
        Real norm = compute_norm(vals.ptr, len, mode);
        
        if (norm > eps) {
            const Real inv_norm = Real(1) / norm;
            for (Index j = 0; j < len; ++j) {
                vals.ptr[j] *= inv_norm;
            }
        }
    });
}

template <CSRLike MatrixT>
void row_norms(const MatrixT& matrix, NormMode mode, MutableSpan<Real> output) {
    // 并行计算行范数
    parallel_for(Size(0), matrix.rows(), [&](Index i) {
        auto vals = matrix.primary_values(i);
        const Index len = matrix.primary_length(i);
        output[i] = (len > 0) ? compute_norm(vals.ptr, len, mode) : Real(0);
    });
}

} // namespace scl::kernel::normalize
```

**注释指南：**

仅包含以下注释：

1. **非显而易见的算法选择：**
   ```cpp
   // 使用 Kahan 求和以提高数值稳定性
   ```

2. **性能关键决策：**
   ```cpp
   // 4 路展开以利用 FMA 流水线
   ```

3. **微妙的正确性问题：**
   ```cpp
   // 必须检查长度 > 0 以避免除零
   ```

**不要包含：**
- 函数目的（从名称和签名显而易见）
- 参数描述（在 .h 文件中记录）
- 算法解释（属于 .h 文件）
- 使用示例（不在实现中）

### API 文档文件 (.h)

文档文件使用带有全面块注释的 C++ 语法。这些文件不编译 - .h 扩展名支持语法高亮，同时避免模板实例化问题。

**为什么是 .h 扩展名？**
- 在所有编辑器中启用 C++ 语法高亮
- 避免意外包含在构建中
- 将文档与实现区分开
- 约定: .h = 仅文档，.hpp = 实现

**文件结构：**

```cpp
// =============================================================================
// FILE: scl/kernel/normalize.h
// BRIEF: 归一化内核的 API 参考
// NOTE: 仅文档 - 不要包含在构建中
// =============================================================================
#pragma once

namespace scl::kernel::normalize {

/* -----------------------------------------------------------------------------
 * FUNCTION: normalize_rows_inplace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     将 CSR 矩阵的每一行就地归一化为单位范数。
 *
 * PARAMETERS:
 *     matrix   [in,out] 可变 CSR 矩阵，就地修改
 *     mode     [in]     归一化的范数类型
 *     epsilon  [in]     防止除零的小常数
 *
 * PRECONDITIONS:
 *     - matrix 必须是有效的 CSR 格式
 *     - matrix 值必须可变
 *     - epsilon > 0
 *
 * POSTCONDITIONS:
 *     - 每行范数 > epsilon 的具有单位范数
 *     - 范数 <= epsilon 的行保持不变
 *     - 矩阵结构（索引、indptr）不变
 *     - 无内存分配
 *
 * MUTABILITY:
 *     INPLACE - 直接修改 matrix.values()
 *
 * ALGORITHM:
 *     对每行 i 并行:
 *         1. 提取行值和长度
 *         2. 如果长度 == 0 则跳过
 *         3. 使用指定模式计算范数:
 *            - L1:  sum(|x_j|)
 *            - L2:  sqrt(sum(x_j^2))
 *            - Max: max(|x_j|)
 *         4. 如果范数 > epsilon:
 *            - 计算 inv_norm = 1 / norm
 *            - 将每个元素乘以 inv_norm
 *         5. 否则: 保持行不变
 *
 * COMPLEXITY:
 *     时间:  O(nnz) 其中 nnz = 非零元素数
 *     空间: O(1) 辅助
 *
 * NUMERICAL NOTES:
 *     - 使用 epsilon 优雅地处理零/接近零的行
 *     - L2 范数使用补偿求和以提高精度
 *     - 除以范数转换为乘以倒数
 *     - 主循环中无分支处理空行
 *
 * THREAD SAFETY:
 *     安全 - 按行并行化，无共享可变状态
 *
 * PERFORMANCE:
 *     - 对大型矩阵自动并行化
 *     - SIMD 优化的范数计算
 *     - 缓存友好的按行访问模式
 *     - 典型吞吐量: 5-10 GB/s 内存带宽
 * -------------------------------------------------------------------------- */
template <CSRLike MatrixT>
void normalize_rows_inplace(
    MatrixT& matrix,               // CSR 矩阵，就地修改
    NormMode mode,                 // 归一化类型: L1, L2, 或 Max
    Real epsilon = 1e-12           // 零范数阈值 (默认: 1e-12)
);

/* -----------------------------------------------------------------------------
 * FUNCTION: row_norms
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     计算 CSR 稀疏矩阵中每行的范数。
 *
 * PARAMETERS:
 *     matrix   [in]  CSR 稀疏矩阵，形状 (n_rows, n_cols)
 *     mode     [in]  范数类型: L1, L2, Max, 或 Sum
 *     output   [out] 预分配缓冲区，大小 = n_rows
 *
 * PRECONDITIONS:
 *     - matrix 必须是有效的 CSR 格式（有序索引，无重复）
 *     - output.size() == matrix.rows()
 *     - output 必须可写
 *
 * POSTCONDITIONS:
 *     - output[i] 包含行 i 的范数
 *     - output[i] == 0 对于空行
 *     - matrix 不变（const 操作）
 *
 * ALGORITHM:
 *     对每行 i 并行:
 *         1. 提取行值和长度
 *         2. 如果长度 == 0: output[i] = 0
 *         3. 否则: output[i] = compute_norm(values, length, mode)
 *
 * COMPLEXITY:
 *     时间:  O(nnz)
 *     空间: O(1) 辅助
 *
 * THREAD SAFETY:
 *     安全 - 每个线程写入不相交的输出位置
 * -------------------------------------------------------------------------- */
template <CSRLike MatrixT>
void row_norms(
    const MatrixT& matrix,         // CSR 矩阵输入（只读）
    NormMode mode,                 // 范数类型: L1, L2, Max, 或 Sum
    MutableSpan<Real> output       // 输出缓冲区 [n_rows]
);

/* -----------------------------------------------------------------------------
 * ENUM: NormMode
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     归一化操作的范数类型。
 *
 * VALUES:
 *     L1   - 曼哈顿范数: sum(|x_i|)
 *     L2   - 欧几里得范数: sqrt(sum(x_i^2))
 *     Max  - 最大绝对值: max(|x_i|)
 *     Sum  - 简单求和（有符号）: sum(x_i)
 *
 * NOTES:
 *     - L1: 对异常值鲁棒，鼓励稀疏性
 *     - L2: 标准欧几里得距离，对异常值敏感
 *     - Max: 极其鲁棒，用于稳定性分析
 *     - Sum: 不是真正的范数（不总是正数），用于中心化
 * -------------------------------------------------------------------------- */
enum class NormMode {
    L1,    // sum(|x_i|)
    L2,    // sqrt(sum(x_i^2))
    Max,   // max(|x_i|)
    Sum    // sum(x_i) - 有符号，不是真正的范数
};

} // namespace scl::kernel::normalize
```

## 文档部分

每个函数文档块必须按顺序包含这些部分：

### 必需部分

| 部分 | 目的 | 总是必需 |
|---------|---------|-----------------|
| SUMMARY | 单行目的声明 | 是 |
| PARAMETERS | 带方向标签的参数列表 | 是 |
| PRECONDITIONS | 调用前的要求 | 是 |
| POSTCONDITIONS | 执行后的保证 | 是 |
| COMPLEXITY | 时间和空间分析 | 是 |
| THREAD SAFETY | 并发保证 | 是 |

### 条件部分

| 部分 | 目的 | 必需时 |
|---------|---------|---------------|
| MUTABILITY | 状态修改类型 | 函数修改输入 |
| ALGORITHM | 逐步描述 | 非平凡算法 |
| THROWS | 异常规范 | 函数可能抛出 |
| NUMERICAL NOTES | 精度/稳定性注释 | 数值计算 |
| PERFORMANCE | 性能特征 | 性能关键 |
| RELATED | 相关函数 | 函数族的一部分 |

## 部分规范

### SUMMARY

函数目的的单行描述。应回答"此函数做什么？"

```cpp
/* -----------------------------------------------------------------------------
 * FUNCTION: compute_pca
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     对稀疏数据矩阵计算主成分分析。
```

**指南：**
- 以动词开头（计算、归一化等）
- 具体（不是"处理数据"而是"将行归一化为单位 L2 范数"）
- 提及关键约束（稀疏、就地、并行等）
- 保持在一行

### PARAMETERS

列出每个参数及其方向标签和描述：

```cpp
 * PARAMETERS:
 *     matrix   [in]     输入稀疏矩阵，形状 (n_samples, n_features)
 *     n_comps  [in]     要计算的成分数 (1 <= n_comps <= min(n_samples, n_features))
 *     output   [out]    预分配输出缓冲区，形状 (n_samples, n_comps)
 *     scratch  [in,out] 工作空间缓冲区，计算期间修改
```

**方向标签：**
- `[in]` - 输入参数，只读，不修改
- `[out]` - 输出参数，只写，调用者必须分配
- `[in,out]` - 就地修改，进入时必须有效

**指南：**
- 对齐标签和描述以提高可读性
- 包含数组的维度（形状、大小）
- 指定范围约束（0 < x < 1 等）
- 提及分配要求（预分配等）

### PRECONDITIONS

调用前必须为真的要求。违反表示调用者错误。

```cpp
 * PRECONDITIONS:
 *     - matrix 必须是有效的 CSR 格式（有序索引，无重复）
 *     - output.size() == matrix.rows() * n_comps
 *     - scratch.size() >= required_workspace_size(matrix, n_comps)
 *     - n_comps > 0 且 n_comps <= min(matrix.rows(), matrix.cols())
 *     - matrix 必须包含至少 n_comps 个线性独立的行
```

**指南：**
- 列出所有有效性要求
- 包含大小/维度约束
- 指定格式要求（有序、唯一等）
- 提及数学约束（正定等）
- 从最重要到最不重要排序

### POSTCONDITIONS

成功执行后为真的保证。

```cpp
 * POSTCONDITIONS:
 *     - output 包含前 n_comps 个主成分
 *     - 成分是正交归一的 (dot(output[i], output[j]) = delta_ij)
 *     - 成分按解释方差递减排序
 *     - matrix 不变（const 操作）
 *     - scratch 可能被修改但仍可重用
```

**指南：**
- 说明计算/写入的内容
- 指定维护的不变量
- 提及对输入的副作用
- 相关时包含数学属性

### MUTABILITY

分类函数如何影响程序状态：

```cpp
 * MUTABILITY:
 *     INPLACE - 直接修改 matrix.values()，无分配
```

**值：**
- `CONST` - 无修改，纯计算
- `INPLACE` - 就地修改输入
- `ALLOCATES` - 分配新内存（指定什么）
- `MIXED` - 一些输入被修改，一些 const（指定哪些）

### ALGORITHM

算法的逐步描述。包括足够的细节以理解复杂性和正确性。

```cpp
 * ALGORITHM:
 *     阶段 1: 计算行均值
 *         对每行 i 并行:
 *             1. 对所有非零元素求和
 *             2. 除以非零数
 *             3. 存储在 means[i] 中
 *     
 *     阶段 2: 中心化矩阵
 *         对每行 i 并行:
 *             对每个非零元素 (i, j):
 *                 matrix[i,j] -= means[i]
 *     
 *     阶段 3: 计算协方差矩阵
 *         result = matrix^T * matrix / (n_samples - 1)
 *     
 *     阶段 4: 特征分解
 *         使用幂迭代计算协方差矩阵的特征向量
```

**指南：**
- 分解为逻辑阶段
- 指定并行化策略
- 提及关键优化技术
- 包含迭代算法的终止条件

### COMPLEXITY

使用 Big-O 符号的时间和空间复杂度：

```cpp
 * COMPLEXITY:
 *     时间:  O(nnz * n_comps * n_iter) 其中 n_iter 是收敛迭代
 *     空间: O(n_features * n_comps) 辅助
```

**指南：**
- 以输入维度表示
- 定义使用的变量（nnz、n_features 等）
- 如果显著不同，指定最佳/平均/最坏情况
- 包含辅助空间（不包括输入/输出）

### THREAD SAFETY

并发执行的线程安全保证：

```cpp
 * THREAD SAFETY:
 *     安全 - 内部并行化，带有每线程工作空间
```

**值：**
- `安全` - 可以从多个线程并发调用
- `不安全` - 需要外部同步
- `条件` - 在特定条件下安全（描述它们）

**示例：**

```cpp
 * THREAD SAFETY:
 *     条件 - 如果不同线程操作不同矩阵则安全
```

```cpp
 * THREAD SAFETY:
 *     不安全 - 修改全局注册表，需要外部锁
```

### THROWS

可能抛出的异常：

```cpp
 * THROWS:
 *     DimensionError - 如果 output.size() != matrix.rows() * n_comps
 *     ConvergenceError - 如果算法在 max_iter 迭代内未收敛
 *     std::bad_alloc - 如果工作空间分配失败
```

**指南：**
- 列出异常类型和条件
- 按可能性排序（最常见的优先）
- 包含标准异常（bad_alloc 等）

### NUMERICAL NOTES

精度、稳定性和边缘情况处理：

```cpp
 * NUMERICAL NOTES:
 *     - 使用补偿求和以提高大 n 的精度
 *     - L2 范数的相对误差界限为 O(epsilon * sqrt(n))
 *     - 接近零范数的除法替换为零以避免无穷大
 *     - 空行产生范数 = 0 而不出错
 *     - 次正规数被刷新为零以提高性能
```

**指南：**
- 提及使用的数值算法（Kahan 求和等）
- 指定已知的误差界限
- 描述边缘情况的处理（零、无穷大、NaN）
- 注意任何精度损失

### PERFORMANCE

预期性能特征：

```cpp
 * PERFORMANCE:
 *     - 对具有 >10,000 行的矩阵自动并行化
 *     - SIMD 优化，带有 4 路累加器模式
 *     - 典型吞吐量: 在现代 CPU 上为 8-15 GB/s 内存带宽
 *     - 线性扩展到核心数，直到内存带宽限制
 *     - 缓存友好: 顺序处理行
```

**指南：**
- 指定并行化行为
- 提及 SIMD 优化
- 在可用时给出吞吐量数字
- 描述扩展特性

## 声明中的内联注释

在文档块之后，包含实际的函数声明及对齐的内联注释：

```cpp
template <CSRLike MatrixT>
void normalize_rows_inplace(
    MatrixT& matrix,               // CSR 矩阵，就地修改
    NormMode mode,                 // 归一化类型: L1, L2, 或 Max
    Real epsilon = 1e-12           // 零范数阈值 (默认: 1e-12)
);
```

**指南：**
- 每个参数的简短注释（5-10 个词）
- 垂直对齐注释以提高可读性
- 在注释中包含默认值
- 提及关键约束（只读、预分配等）

这些内联注释在快速查询和 IDE 工具提示中可见。

## 快速查询命令

仅提取函数签名而不提取文档：

```bash
sed '/^\/\*/,/\*\/$/d' scl/kernel/normalize.h
```

**输出：**

```cpp
#pragma once

namespace scl::kernel::normalize {

template <CSRLike MatrixT>
void normalize_rows_inplace(
    MatrixT& matrix,
    NormMode mode,
    Real epsilon = 1e-12
);

template <CSRLike MatrixT>
void row_norms(
    const MatrixT& matrix,
    NormMode mode,
    MutableSpan<Real> output
);

enum class NormMode {
    L1, L2, Max, Sum
};

} // namespace scl::kernel::normalize
```

这允许 API 用户快速检查接口。

## 语言和格式规则

### 严格的纯文本

**关键规则：** 文档必须仅使用纯文本。不使用标记语言。

**禁止的语法：**

❌ **不使用 Markdown：**
```cpp
/* SUMMARY:
 *     使用 `sqrt(sum(x^2))` 计算 **L2 范数**。
 *     
 *     - 第一步
 *     - 第二步
 */
```

❌ **不使用 LaTeX：**
```cpp
/* SUMMARY:
 *     计算范数: $\|x\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$
 */
```

❌ **不使用示例：**
```cpp
/* SUMMARY:
 *     归一化行。
 *     
 *     示例:
 *         normalize_rows_inplace(matrix, NormMode::L2);
 */
```

**正确的纯文本：**

✅ **好：**
```cpp
/* SUMMARY:
 *     使用公式计算 L2 范数: norm = sqrt(sum(x_i^2))
 *     
 * ALGORITHM:
 *     1. 使用补偿求和计算平方和
 *     2. 取结果的平方根
 *     3. 处理零情况: 如果 sum < epsilon^2 则返回 0
 */
```

**理由：**
- Markdown 破坏结构化解析工具
- LaTeX 在纯编辑器中不可读
- 示例会过时并增加维护负担
- 纯文本是通用和明确的

### 纯文本中的公式

使用 ASCII 符号表示数学表达式：

| 概念 | 纯文本 |
|---------|------------|
| L1 范数 | `sum(\|x_i\|)` |
| L2 范数 | `sqrt(sum(x_i^2))` |
| 点积 | `sum(a_i * b_i)` |
| 矩阵乘法 | `C[i,j] = sum_k A[i,k] * B[k,j]` |
| 求和 | `sum_{i=1}^{n} x_i` |
| Argmax | `argmax_i f(x_i)` |

## 不同构造的文档

### 函数

上面显示的标准模板。

### 枚举

```cpp
/* -----------------------------------------------------------------------------
 * ENUM: AllocType
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     用于正确清理的内存分配类型。
 *
 * VALUES:
 *     ArrayNew      - 使用 new[] 分配，使用 delete[] 清理
 *     ScalarNew     - 使用 new 分配，使用 delete 清理
 *     AlignedAlloc  - 使用 aligned_alloc 分配，使用 free 清理
 *     Custom        - 提供自定义删除器函数
 *
 * USAGE:
 *     传递给注册表以指定应如何释放内存。
 * -------------------------------------------------------------------------- */
enum class AllocType {
    ArrayNew,      // new[] → delete[]
    ScalarNew,     // new → delete
    AlignedAlloc,  // aligned_alloc → free
    Custom         // 用户提供的删除器
};
```

### 结构体/类

```cpp
/* -----------------------------------------------------------------------------
 * STRUCT: ContiguousArraysT
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     具有注册表管理内存的连续 CSR/CSC 数组。
 *
 * FIELDS:
 *     data        - 非零值，大小 = nnz
 *     indices     - 列/行索引，大小 = nnz
 *     indptr      - 行/列偏移，大小 = primary_dim + 1
 *     nnz         - 非零元素数
 *     primary_dim - 行数（CSR）或列数（CSC）
 *
 * INVARIANTS:
 *     - indptr[0] == 0
 *     - indptr[primary_dim] == nnz
 *     - indptr 单调递增
 *     - 所有指针已向 scl::Registry 注册
 *
 * LIFETIME:
 *     调用者必须取消注册指针以释放内存。
 * -------------------------------------------------------------------------- */
template <typename T>
struct ContiguousArraysT {
    T* data;             // 值数组 [nnz]
    Index* indices;      // 索引数组 [nnz]
    Index* indptr;       // 偏移数组 [primary_dim + 1]
    Index nnz;           // 非零数
    Index primary_dim;   // 行（CSR）或列（CSC）
};
```

### 类型别名

```cpp
/* -----------------------------------------------------------------------------
 * ALIAS: Real
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     用于数值计算的浮点类型。
 *
 * CONFIGURATION:
 *     - SCL_REAL_FLOAT32: float (32 位，默认)
 *     - SCL_REAL_FLOAT64: double (64 位)
 *     - SCL_REAL_FLOAT16: _Float16 (16 位，实验性)
 *
 * NOTES:
 *     类型在编译时配置。所有算术使用此类型。
 * -------------------------------------------------------------------------- */
using Real = /* float | double | _Float16 */;
```

## 工作流程要求

**关键：** 修改任何 .hpp 文件后，更新相应的 .h 文件。

### 检查清单

在标记任务完成之前：

- [ ] .hpp 中的实现正确且可编译
- [ ] 代码使用最少的内联注释
- [ ] 相应的 .h 文件存在
- [ ] .h 包含实际函数声明（不仅仅是文档）
- [ ] .h 中的所有函数签名与实现完全匹配
- [ ] .h 声明中的内联注释简洁且对齐
- [ ] 块注释包含所有必需部分
- [ ] PRECONDITIONS 记录所有输入要求
- [ ] POSTCONDITIONS 记录所有保证
- [ ] MUTABILITY 对于更改状态的函数正确指定
- [ ] 任何注释中没有 Markdown、LaTeX 或示例
- [ ] 快速查询有效: `sed '/^\/\*/,/\*\/$/d' file.h` 显示清晰的签名

## 新函数模板

### 实现 (.hpp)

```cpp
// scl/module/feature.hpp
#pragma once

#include "scl/module/feature.h"
#include "scl/core/type.hpp"

namespace scl::module::feature {

template <typename T>
void my_function(const T* input, T* output, size_t n) {
    // 仅当算法非显而易见时才简短注释
    for (size_t i = 0; i < n; ++i) {
        output[i] = process(input[i]);
    }
}

} // namespace scl::module::feature
```

### 文档 (.h)

```cpp
// scl/module/feature.h
#pragma once

namespace scl::module::feature {

/* -----------------------------------------------------------------------------
 * FUNCTION: my_function
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     函数功能的简短单行描述。
 *
 * PARAMETERS:
 *     input  [in]  输入数组，大小 n
 *     output [out] 输出数组，大小 n（预分配）
 *     n      [in]  数组大小
 *
 * PRECONDITIONS:
 *     - input 和 output 是有效指针
 *     - output 有 n 个元素的空间
 *     - n > 0
 *
 * POSTCONDITIONS:
 *     - output[i] 包含 input[i] 的处理值
 *     - input 不变
 *
 * COMPLEXITY:
 *     时间:  O(n)
 *     空间: O(1) 辅助
 *
 * THREAD SAFETY:
 *     安全 - 无共享可变状态
 * -------------------------------------------------------------------------- */
template <typename T>
void my_function(
    const T* input,    // 输入数组 [n]
    T* output,         // 输出数组 [n]（预分配）
    size_t n           // 数组大小
);

} // namespace scl::module::feature
```

---

::: tip 作为契约的文档
API 文档是实现和用户之间的契约。前置条件定义用户必须确保的内容，后置条件定义实现保证的内容。这个契约使得能够在不阅读实现的情况下推理正确性。
:::

