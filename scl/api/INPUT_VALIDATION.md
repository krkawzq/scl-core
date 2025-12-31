# SCL C-API 输入验证规范

## 设计哲学

SCL C-API 遵循**防御性编程 + 性能平衡**原则：

1. **必须检查**：明显的用户错误（NULL、负数、无效枚举）
2. **前置条件**：性能敏感的深度验证（数组内容、索引范围）
3. **清晰文档**：所有前置条件必须在 API 文档中明确说明

## 验证级别

### Level 1: 必须检查（Always Validate）

这些检查**必须**在所有 C-API 函数中执行，失败时抛出错误：

#### 1.1 NULL 指针检查

```c
// ✅ 必须检查
SCL_CHECK_NOT_NULL(handle);      // 句柄
SCL_CHECK_NOT_NULL(row_indices); // 输入数组
SCL_CHECK_NOT_NULL(values);      // 数据指针
SCL_CHECK_NOT_NULL(output);      // 输出参数

// 📝 错误码: NullPointer (1)
```

**理由**：NULL 指针解引用会立即崩溃，必须拦截。

#### 1.2 维度非负检查

```c
// ✅ 必须检查
SCL_CHECK_ARG(rows >= 0, "rows must be non-negative");
SCL_CHECK_ARG(cols >= 0, "cols must be non-negative");
SCL_CHECK_ARG(n >= 0, "dimension must be non-negative");
SCL_CHECK_ARG(nnz >= 0, "nnz must be non-negative");

// 📝 错误码: InvalidArgument (4)
```

**理由**：负维度没有语义，是明显的用户错误。

#### 1.3 枚举值有效性

```c
// ✅ 必须检查
SCL_CHECK_ARG(scl_is_valid_value_type(value_type), "invalid value type");
SCL_CHECK_ARG(scl_is_valid_index_type(index_type), "invalid index type");
SCL_CHECK_ARG(scl_is_valid_layout(layout), "invalid layout");

// 📝 错误码: InvalidArgument (4) 或 TypeMismatch (302)
```

**理由**：无效枚举值会导致分派宏失败或未定义行为。

#### 1.4 布局兼容性检查

```c
// ✅ 必须检查
// row_data 只能用于 CSR
SCL_CHECK_ARG(handle->layout == SCL_LAYOUT_CSR, "row_data requires CSR layout");

// col_data 只能用于 CSC
SCL_CHECK_ARG(handle->layout == SCL_LAYOUT_CSC, "col_data requires CSC layout");

// 📝 错误码: InvalidArgument (4)
```

**理由**：布局不匹配会返回错误数据。

#### 1.5 容差非负检查

```c
// ✅ 必须检查
SCL_CHECK_ARG(tolerance >= 0, "tolerance must be non-negative");

// 📝 错误码: InvalidArgument (4)
```

**理由**：负容差没有数学意义。

---

### Level 2: 条件检查（Conditional Validate）

这些检查**应该**在 Debug 模式或启用验证时执行：

#### 2.1 索引越界检查（at/get/exists）

```c
// ⚠️ 当前状态：未检查
// 建议：添加检查

// 当前实现
auto scl_sparse_at(handle, row, col, value) {
    SCL_CHECK_NOT_NULL(handle);
    SCL_CHECK_NOT_NULL(value);
    // ❌ 缺失：row < 0 || row >= rows || col < 0 || col >= cols
    ...
}

// 建议实现
auto scl_sparse_at(handle, row, col, value) {
    SCL_CHECK_NOT_NULL(handle);
    SCL_CHECK_NOT_NULL(value);
    
    // ✅ 添加边界检查
    SCL_CHECK_ARG(row >= 0 && row < scl_sparse_rows(handle), 
                  "row index out of bounds");
    SCL_CHECK_ARG(col >= 0 && col < scl_sparse_cols(handle), 
                  "col index out of bounds");
    ...
}

// 📝 错误码: IndexOutOfBounds (152)
```

**理由**：
- ✅ **应该检查**：防止缓冲区溢出，安全第一
- ⚠️ **性能影响**：每次访问多2个比较操作（可接受）

#### 2.2 NNZ 上界检查

```c
// ⚠️ 建议添加
auto scl_sparse_from_coo(..., rows, cols, nnz, ...) {
    SCL_CHECK_ARG(rows >= 0 && cols >= 0 && nnz >= 0, ...);
    
    // ✅ 添加合理性检查
    SCL_CHECK_ARG(nnz <= rows * cols, 
                  "nnz cannot exceed rows*cols");
    ...
}

// 📝 错误码: InvalidArgument (4)
```

**理由**：超过矩阵总元素数是明显错误。

---

### Level 3: 前置条件（Preconditions - Document Only）

这些条件**不检查**，在文档中明确说明用户责任：

#### 3.1 数组内容有效性

```c
/**
 * @brief Create sparse matrix from COO format
 * 
 * @pre Row and column indices must be within [0, rows) and [0, cols)
 * @pre Array lengths must match nnz parameter
 * @pre Behavior is undefined if indices are out of range
 * 
 * @note This function does NOT validate index ranges for performance.
 *       Use scl_sparse_validate() to check data integrity if needed.
 */
scl_sparse_t scl_sparse_from_coo(...);
```

**不检查的内容**：
- ❌ `row_indices[i] < 0` 或 `>= rows`（每个元素）
- ❌ `col_indices[i] < 0` 或 `>= cols`（每个元素）
- ❌ 数组长度是否真的等于 nnz

**理由**：
- 性能：检查 nnz 个元素会显著增加开销
- 合约：这是用户责任，文档化即可
- 备选：提供 `scl_sparse_validate()` 可选验证函数

#### 3.2 CSR/CSC 格式正确性

```c
/**
 * @brief Create sparse matrix from CSR format
 * 
 * @pre indptr must be monotonically increasing
 * @pre indptr[0] == 0
 * @pre indptr[rows] == nnz
 * @pre Column indices must be in [0, cols)
 * @pre Indices within each row should be sorted (recommended but not required)
 * 
 * @note This function assumes input data is valid CSR format.
 *       Invalid data may cause crashes or incorrect results.
 */
scl_sparse_t scl_sparse_from_csr(...);
```

**不检查的内容**：
- ❌ `indptr[i+1] >= indptr[i]`（单调性）
- ❌ `indptr[0] == 0`（起始条件）
- ❌ 列索引范围和排序

**理由**：
- 性能：验证 CSR 格式完整性需要 O(rows + nnz) 时间
- 信任：假设用户提供正确格式数据
- 调试：提供 `scl_sparse_check_csr()` 调试工具

#### 3.3 数据类型匹配

```c
/**
 * @brief Create sparse matrix from COO format
 * 
 * @param values Pointer to value array (must match value_type)
 * 
 * @pre If value_type==SCL_REAL64, values must point to double array
 * @pre If value_type==SCL_INT32, values must point to int32_t array
 * 
 * @warning Passing wrong type causes undefined behavior (type punning)
 */
scl_sparse_t scl_sparse_from_coo(
    ...,
    const void* values,  // Type-erased, user must pass correct type
    ...
);
```

**不检查的内容**：
- ❌ `void*` 指针实际指向的类型

**理由**：
- 无法检查：C 语言无运行时类型信息
- 文档化：明确说明用户责任
- 类型安全：C++ 用户应使用类型安全的包装

---

## 当前实现分析

### ✅ 已正确实现的检查

| 函数 | NULL | 维度非负 | 类型有效 | 其他 |
|------|------|---------|---------|------|
| scl_sparse_zeros | ✅ | ✅ | ✅ | ✅ layout |
| scl_sparse_identity | ✅ | ✅ | ✅ | |
| scl_sparse_from_coo | ✅ | ✅ | ❌ | |
| scl_sparse_from_csr | ✅ | ✅ | ❌ | |
| scl_sparse_from_csc | ✅ | ✅ | ❌ | |
| scl_sparse_from_dense | ✅ | ✅ | ❌ | ✅ tolerance≥0 |
| scl_sparse_at | ✅ | ❌ | - | |
| scl_sparse_get | ✅ | ❌ | - | |
| scl_sparse_exists | ✅ | ❌ | - | |
| scl_sparse_scale | ✅ | - | - | |
| scl_sparse_clone | ✅ | - | - | |
| scl_sparse_transpose | ✅ | - | - | |

### ❌ 缺失的关键检查

#### 1. 索引越界（at/get/exists）

**风险**：高 - 可能导致缓冲区溢出

**建议**：立即添加

```cpp
SCL_API
auto scl_sparse_at(..., row, col, ...) -> std::int32_t {
    SCL_C_API_BEGIN
    SCL_CHECK_NOT_NULL(handle);
    SCL_CHECK_NOT_NULL(value);
    
    // ✅ 添加边界检查
    auto rows = scl_sparse_rows(handle);
    auto cols = scl_sparse_cols(handle);
    SCL_CHECK_ARG(row >= 0 && row < rows, "row index out of bounds");
    SCL_CHECK_ARG(col >= 0 && col < cols, "column index out of bounds");
    
    ...
}
```

#### 2. 类型验证（from_* 函数）

**风险**：中 - 无效类型会导致分派失败

**建议**：在创建函数中添加（已部分完成）

```cpp
// ✅ zeros/identity 已添加
SCL_CHECK_ARG(scl_is_valid_value_type(value_type), "invalid value type");

// ❌ from_coo/csr/csc 需要添加
```

#### 3. NNZ 合理性检查

**风险**：低 - 不影响安全性，但可以提前发现错误

**建议**：添加快速检查

```cpp
auto scl_sparse_from_coo(..., rows, cols, nnz, ...) {
    ...
    // ✅ 添加上界检查（快速，防止明显错误）
    SCL_CHECK_ARG(nnz <= static_cast<std::int64_t>(rows) * cols, 
                  "nnz exceeds maximum possible elements");
    ...
}
```

---

## 建议的验证策略

### 策略A：严格模式（推荐用于 Debug）

```c
// 编译选项：-DSCL_STRICT_VALIDATION=1

// 所有函数都做完整验证
#ifdef SCL_STRICT_VALIDATION
    #define SCL_VALIDATE_BOUNDS(cond, msg) SCL_CHECK_ARG(cond, msg)
#else
    #define SCL_VALIDATE_BOUNDS(cond, msg) ((void)0)
#endif

// 使用
SCL_VALIDATE_BOUNDS(row >= 0 && row < rows, "row out of bounds");
```

### 策略B：可选验证函数

```c
/**
 * @brief Validate sparse matrix data integrity
 * @param handle Sparse handle
 * @return 0 on success, error code on validation failure
 * 
 * Checks:
 *   - All indices in valid range
 *   - CSR/CSC format correctness
 *   - No duplicate entries (if sorted)
 */
int32_t scl_sparse_validate(scl_sparse_t handle);

/**
 * @brief Validate COO data before creation
 * @return 0 if valid, error code otherwise
 */
int32_t scl_validate_coo_data(
    int64_t rows, int64_t cols, int64_t nnz,
    const void* row_indices,
    const void* col_indices
);
```

**用法**：

```c
// 生产环境：快速创建，不验证
auto mat = scl_sparse_from_coo(...);

// 调试/测试：创建并验证
auto mat = scl_sparse_from_coo(...);
if (scl_sparse_validate(mat) != 0) {
    fprintf(stderr, "Invalid sparse data: %s\n", scl_get_error_message());
}
```

---

## 具体函数的验证决策

### scl_sparse_zeros()

**当前检查**：✅ 完整
```c
SCL_CHECK_ARG(rows >= 0, ...);
SCL_CHECK_ARG(cols >= 0, ...);
SCL_CHECK_ARG(scl_is_valid_value_type(value_type), ...);
SCL_CHECK_ARG(scl_is_valid_index_type(index_type), ...);
SCL_CHECK_ARG(scl_is_valid_layout(layout), ...);
```

**建议**：保持不变 ✅

---

### scl_sparse_from_coo()

**当前检查**：
```c
✅ rows >= 0, cols >= 0, nnz >= 0
✅ NULL pointers
❌ 类型有效性
❌ nnz 上界
❌ 索引范围（前置条件）
```

**建议添加**：

```cpp
SCL_API
auto scl_sparse_from_coo(...) -> scl_sparse_t {
    SCL_C_API_BEGIN
    
    // Level 1: 必须检查
    SCL_CHECK_ARG(rows >= 0, "rows must be non-negative");
    SCL_CHECK_ARG(cols >= 0, "cols must be non-negative");
    SCL_CHECK_ARG(nnz >= 0, "nnz must be non-negative");
    SCL_CHECK_NOT_NULL(row_indices);
    SCL_CHECK_NOT_NULL(col_indices);
    SCL_CHECK_NOT_NULL(values);
    
    // ✅ 添加：类型有效性
    SCL_CHECK_ARG(scl_is_valid_value_type(value_type), "invalid value type");
    SCL_CHECK_ARG(scl_is_valid_index_type(index_type), "invalid index type");
    SCL_CHECK_ARG(scl_is_valid_layout(layout), "invalid layout");
    
    // ✅ 添加：NNZ 上界（快速检查，防止明显错误）
    if (rows > 0 && cols > 0) {
        SCL_CHECK_ARG(nnz <= rows * cols, "nnz exceeds matrix size");
    }
    
    // ❌ 不检查（前置条件）：
    //   - 索引值范围 [0, rows), [0, cols)
    //   - 重复索引（允许，会自动合并）
    
    ...
}
```

**文档更新**：

```c
/**
 * @brief Create sparse matrix from COO (Coordinate) format
 * 
 * @param row_indices Array of row indices (length = nnz)
 * @param col_indices Array of column indices (length = nnz)
 * @param values Array of values (length = nnz)
 * @param nnz Number of non-zero elements
 * 
 * @pre Array pointers must be valid for `nnz` elements
 * @pre Index arrays must be castable to index_type (int32_t or int64_t)
 * @pre Value array must be castable to value_type
 * 
 * ## Index Range (Precondition - NOT Validated)
 * 
 * User must ensure:
 *   - `row_indices[i]` ∈ [0, rows) for all i
 *   - `col_indices[i]` ∈ [0, cols) for all i
 * 
 * Behavior with out-of-range indices:
 *   - May crash (buffer overflow)
 *   - May produce incorrect results
 *   - NOT validated for performance reasons
 * 
 * @note Use scl_validate_coo_data() to verify data before creation
 * 
 * ## Duplicate Indices
 * 
 * Duplicate (row, col) pairs are allowed and will be merged (summed).
 * 
 * ## Unsorted Data
 * 
 * Input does not need to be sorted. The function will sort internally.
 * 
 * @return New sparse handle, or NULL on error
 */
scl_sparse_t scl_sparse_from_coo(...);
```

---

### scl_sparse_at/get/exists()

**当前检查**：
```c
✅ NULL handle
✅ NULL output pointer (at only)
❌ 索引越界
```

**建议添加**：

```cpp
SCL_API
auto scl_sparse_at(handle, row, col, value) -> std::int32_t {
    SCL_C_API_BEGIN
    SCL_CHECK_NOT_NULL(handle);
    SCL_CHECK_NOT_NULL(value);
    
    // ✅ 必须添加：边界检查
    const auto rows = scl_sparse_rows(handle);
    const auto cols = scl_sparse_cols(handle);
    SCL_CHECK_ARG(row >= 0 && row < rows, "row index out of bounds");
    SCL_CHECK_ARG(col >= 0 && col < cols, "column index out of bounds");
    
    ...
    SCL_C_API_END
}

SCL_API
auto scl_sparse_get(handle, row, col) -> double {
    // ⚠️ 返回值型，无法返回错误码
    // 选项1：越界返回 0.0（静默失败）
    // 选项2：越界返回 NaN（可检测）
    
    if (!handle) return 0.0;
    
    auto rows = scl_sparse_rows(handle);
    auto cols = scl_sparse_cols(handle);
    
    // ✅ 建议：越界返回 NaN
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        scl::set_thread_error(scl::ErrorCode::IndexOutOfBounds);
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    ...
}

SCL_API
auto scl_sparse_exists(handle, row, col) -> std::int32_t {
    if (!handle) return 0;
    
    auto rows = scl_sparse_rows(handle);
    auto cols = scl_sparse_cols(handle);
    
    // ✅ 越界：返回 0 (不存在)
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        return 0;
    }
    
    ...
}
```

---

### scl_sparse_row_data/col_data()

**当前检查**：
```c
✅ NULL handle
✅ 布局检查
✅ NULL 输出指针
❌ 行/列索引范围
```

**建议添加**：

```cpp
SCL_API
auto scl_sparse_row_data(handle, row, ...) -> std::int32_t {
    SCL_C_API_BEGIN
    SCL_CHECK_NOT_NULL(handle);
    SCL_CHECK_ARG(handle->layout == SCL_LAYOUT_CSR, ...);
    SCL_CHECK_NOT_NULL(values);
    SCL_CHECK_NOT_NULL(indices);
    SCL_CHECK_NOT_NULL(length);
    
    // ✅ 添加：行索引检查
    auto rows = scl_sparse_rows(handle);
    SCL_CHECK_ARG(row >= 0 && row < rows, "row index out of bounds");
    
    ...
}
```

---

## 实施建议

### 立即修复（High Priority）

1. **scl_sparse_at/get/exists**: 添加边界检查
2. **scl_sparse_from_***: 添加类型有效性检查
3. **scl_sparse_row_data/col_data**: 添加索引范围检查

### 文档改进（Medium Priority）

1. 为所有函数添加 `@pre` 前置条件
2. 明确说明哪些不验证
3. 建议使用 `scl_sparse_validate()` 调试

### 可选增强（Low Priority）

1. 实现 `scl_sparse_validate()` 完整验证
2. 实现 `scl_validate_coo_data()` 等辅助函数
3. 添加 `SCL_STRICT_VALIDATION` 编译选项

---

## 错误码分配

| 检查类型 | 错误码 | 名称 | 说明 |
|---------|--------|------|------|
| NULL 指针 | 1 | NullPointer | 必须立即修复 |
| 无效参数 | 4 | InvalidArgument | 维度、tolerance等 |
| 索引越界 | 152 | IndexOutOfBounds | 访问操作 |
| 维度不匹配 | 201 | DimensionMismatch | 形状不兼容 |
| 类型不匹配 | 302 | TypeMismatch | 类型不兼容 |

---

## 示例：完整的验证实现

```cpp
SCL_API
auto scl_sparse_at(
    scl_sparse_t handle,
    std::int64_t row,
    std::int64_t col,
    void* value
) -> std::int32_t {
    SCL_C_API_BEGIN
    
    // Level 1: 必须检查（Always）
    SCL_CHECK_NOT_NULL(handle);
    SCL_CHECK_NOT_NULL(value);
    
    // Level 2: 边界检查（Recommended）
    const auto rows = scl_sparse_rows(handle);
    const auto cols = scl_sparse_cols(handle);
    SCL_CHECK_ARG(row >= 0, "row index cannot be negative");
    SCL_CHECK_ARG(row < rows, "row index out of bounds");
    SCL_CHECK_ARG(col >= 0, "column index cannot be negative");
    SCL_CHECK_ARG(col < cols, "column index out of bounds");
    
    // Execute operation
    visit_sparse(handle, [row, col, value](const auto& mat) {
        using ValueT = typename std::decay_t<decltype(mat)>::ValueType;
        auto* value_typed = static_cast<ValueT*>(value);
        *value_typed = mat.at(static_cast<decltype(mat.rows())>(row),
                             static_cast<decltype(mat.cols())>(col));
    });
    
    SCL_C_API_END
}
```

---

## 测试建议

为每个验证点添加测试：

```cpp
// 边界测试
TEST(at_negative_row) {
    auto m = scl_sparse_identity(5, ...);
    double v;
    auto r = scl_sparse_at(m, -1, 0, &v);
    SCL_ASSERT_NE(r, 0);  // Should fail
    SCL_ASSERT_ERROR_CODE(IndexOutOfBounds);
}

TEST(at_row_equals_rows) {
    auto m = scl_sparse_identity(5, ...);
    double v;
    auto r = scl_sparse_at(m, 5, 0, &v);  // rows=5, max index=4
    SCL_ASSERT_NE(r, 0);
    SCL_ASSERT_ERROR_CODE(IndexOutOfBounds);
}
```

---

## 总结

| 验证类型 | 检查 | 性能影响 | 优先级 |
|---------|------|---------|--------|
| NULL 指针 | ✅ 已实现 | 极小 | P0 |
| 维度非负 | ✅ 已实现 | 极小 | P0 |
| 类型有效 | ⚠️ 部分实现 | 极小 | P0 |
| 索引越界 | ❌ 未实现 | 小 | P1 |
| NNZ 上界 | ❌ 未实现 | 极小 | P2 |
| 数组内容 | ❌ 不检查 | 大 | - |
| 格式完整性 | ❌ 不检查 | 大 | - |

**建议行动**：
1. ✅ 立即添加索引越界检查到 at/get/exists
2. ✅ 补全 from_* 函数的类型验证
3. ✅ 添加 NNZ 上界检查
4. 📝 完善 API 文档的 @pre 条件
5. 🔮 未来：实现可选的 validate() 函数

