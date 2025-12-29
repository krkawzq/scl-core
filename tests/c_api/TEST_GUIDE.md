# SCL C API 测试构建指南

## 📋 测试编写规范

### 1. 文件命名和组织

```
tests/c_api/src/
├── test_<module>.cpp          # 单个模块完整测试
├── test_<module>_<feature>.cpp  # 特定功能测试
└── test_<category>_<name>.cpp   # 分类测试
```

**命名规则：**
- `test_core.cpp` - core.h 所有函数
- `test_sparse_complete.cpp` - sparse.h 完整测试
- `test_sparse_spmv.cpp` - SpMV 专项测试
- `test_kernel_gemm.cpp` - GEMM kernel 测试

### 2. 测试结构模板

```cpp
// =============================================================================
// SCL Core - <模块名> Tests
// =============================================================================
//
// 测试范围: <描述>
// 
// 函数列表:
//   ✓ function1
//   ✓ function2
//   ...
//
// =============================================================================

#include "test.hpp"

using namespace scl::test;

// Helper functions (if needed)
static void setup_data() {
    // ...
}

SCL_TEST_BEGIN

// =============================================================================
// <功能组1>
// =============================================================================

SCL_TEST_SUITE(feature_group_1)

SCL_TEST_CASE(basic_functionality) {
    // 1. Setup test data
    // 2. Call function
    // 3. Verify results
    // 4. Check error codes
}

SCL_TEST_CASE(null_pointer_safety) {
    // Test NULL inputs
    SCL_ASSERT_EQ(some_func(nullptr, ...), SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(invalid_arguments) {
    // Test invalid args
    SCL_ASSERT_NE(some_func(...), SCL_OK);
}

SCL_TEST_CASE(boundary_conditions) {
    // Edge cases: 0, 1, max values
}

SCL_TEST_SUITE_END

// =============================================================================
// <功能组2 - 需要随机测试>
// =============================================================================

SCL_TEST_SUITE(randomized_tests)

SCL_TEST_RETRY(statistical_correctness, 3)  // 重试3次
{
    Random rng(12345);
    
    // Generate random data
    auto shape = random_shape(10, 100, rng);
    auto matrix = random_sparse_random_shape(10, 100, 0.01, 0.1, rng);
    
    // Test with random data
    // Compare with reference implementation
}

SCL_TEST_SUITE_END

// =============================================================================
// <大规模测试>
// =============================================================================

SCL_TEST_TAGGED(large_scale_test, "slow", "integration")
{
    // Large matrices
}

SCL_TEST_END

SCL_TEST_MAIN()
```

### 3. 必须覆盖的边界情况

**对于每个函数，测试：**

#### 3.1 NULL 指针安全
```cpp
// 输入句柄
SCL_ASSERT_EQ(func(nullptr, ...), SCL_ERROR_NULL_POINTER);

// 输出指针
SCL_ASSERT_EQ(func(handle, nullptr, ...), SCL_ERROR_NULL_POINTER);

// 数组指针
SCL_ASSERT_EQ(func(handle, nullptr), SCL_ERROR_NULL_POINTER);
```

#### 3.2 无效参数
```cpp
// 负数
SCL_ASSERT_NE(func(handle, -1, ...), SCL_OK);

// 零（如果不允许）
SCL_ASSERT_NE(func(handle, 0, ...), SCL_OK);

// 超出范围
SCL_ASSERT_NE(func(handle, HUGE_VALUE, ...), SCL_OK);
```

#### 3.3 维度不匹配
```cpp
// 矩阵维度不一致
SCL_ASSERT_EQ(func(mat1, mat2), SCL_ERROR_DIMENSION_MISMATCH);
```

#### 3.4 边界值
```cpp
// 最小值
test_with_value(1);

// 最大值  
test_with_value(MAX_INDEX);

// 空矩阵
test_with_value(0);

// 单元素
test_with_value(1, 1, 1);
```

### 4. 随机测试要求

#### 4.1 使用随机数据生成器
```cpp
SCL_TEST_RETRY(random_test, 5)  // 重试5次确保稳定性
{
    Random rng(seed);  // 可复现的种子
    
    // 随机 shape
    auto [rows, cols] = random_shape(10, 100, rng);
    
    // 随机密度
    double density = random_density(0.01, 0.2, rng);
    
    // 生成矩阵
    auto mat = random_sparse_csr(rows, cols, density, rng);
    
    // 测试...
}
```

#### 4.2 批量随机测试
```cpp
SCL_TEST_UNIT(monte_carlo_verification) {
    std::vector<double> errors;
    
    for (int trial = 0; trial < 100; ++trial) {
        Random rng(trial);
        auto mat = random_sparse_random_shape(10, 100, 0.01, 0.1, rng);
        
        // Compute and record error
        double error = compute_error(mat);
        errors.push_back(error);
    }
    
    // Statistical verification
    auto stats = precision::compute_statistics(errors);
    SCL_ASSERT_TRUE(precision::error_stats_acceptable(stats));
}
```

### 5. 数值精度验证

#### 5.1 使用参考实现对比
```cpp
#include "precision.hpp"

SCL_TEST_UNIT(numerical_correctness) {
    auto mat = random_sparse_csr(50, 50, 0.1);
    
    // SCL implementation
    auto result_scl = compute_scl(mat);
    
    // Reference (Eigen or BLAS)
    auto result_ref = compute_eigen(mat);
    
    // Compare with tolerance
    using precision::Tolerance;
    SCL_ASSERT_TRUE(matrices_equal(result_scl, result_ref, Tolerance::normal()));
}
```

#### 5.2 精度容差选择
```cpp
// 严格（直接计算）
Tolerance::strict()    // rtol=1e-12, atol=1e-15

// 正常（大多数情况）
Tolerance::normal()    // rtol=1e-9, atol=1e-12

// 宽松（迭代算法）
Tolerance::iterative() // rtol=1e-6, atol=1e-9

// 统计方法
Tolerance::statistical() // rtol=1e-4, atol=1e-6

// 近似算法
Tolerance::approximate() // rtol=1e-2, atol=1e-4
```

### 6. 测试标签使用

```cpp
// 基础测试（必须通过）
SCL_TEST_UNIT(basic_test) { ... }

// 慢测试（可选运行）
SCL_TEST_TAGGED(slow_test, "slow") { ... }

// 集成测试
SCL_TEST_TAGGED(integration_test, "integration", "slow") { ... }

// 跳过的测试（临时）
SCL_TEST_SKIP(broken_test, "Bug #123") { ... }

// 预期失败（已知bug）
SCL_TEST_XFAIL(known_bug, "Waiting for fix") { ... }

// 需要重试（随机测试）
SCL_TEST_RETRY(statistical_test, 5) { ... }
```

## 🏗️ 编译和运行

### 独立编译单个测试
```bash
cd tests/c_api

# 编译并运行
make test-<name>

# 只编译
make build-<name>

# 只运行（必须先编译）
make run-<name>

# 详细输出
make test-<name> VERBOSE=1

# 调试版本
make test-<name> DEBUG=1
```

### 批量运行
```bash
# 运行所有测试
make test-all

# 详细输出
make test-all VERBOSE=1

# 只运行快速测试（排除 slow 标签）
./units/test_<name> --exclude-tag slow

# 只运行特定标签
./units/test_<name> --tag unit
```

### 测试过滤和调试
```bash
# 运行匹配的测试
./units/test_sparse --filter "transpose"

# 排除某些测试
./units/test_sparse --exclude "slow"

# 列出所有测试
./units/test_sparse --list

# 失败即停止
./units/test_sparse --fail-fast

# 生成报告
./units/test_sparse --json report.json --xml results.xml
```

## 📝 任务分发模板

### 任务模板

```markdown
## 任务: 测试 <模块名>

### 目标
- 文件: `tests/c_api/src/test_<name>.cpp`
- 覆盖: `scl/binding/c_api/<path>/<name>.h`
- 函数数: X 个
- 预期测试数: Y 个

### 函数列表
- [ ] function1(...)
- [ ] function2(...)
- [ ] ...

### 要求
1. ✅ 所有函数覆盖
2. ✅ NULL 指针检查
3. ✅ 无效参数检查
4. ✅ 边界值测试
5. ✅ 随机数据测试（至少3次 retry）
6. ✅ 参考实现对比（Eigen/BLAS）
7. ✅ 精度验证（使用 Tolerance）

### 验收标准
- 编译通过: `make build-<name>`
- 测试通过: `make test-<name>`
- 覆盖率: 100%
- 通过率: 100%

### 交付
- 测试文件: `test_<name>.cpp`
- 测试数量: Y+ 个
- 运行时间: <1秒
```

## 📦 可用工具

### 测试框架 (core.hpp)
```cpp
// 断言
SCL_ASSERT(expr)
SCL_ASSERT_EQ(expected, actual)
SCL_ASSERT_NEAR(expected, actual, tolerance)
SCL_ASSERT_NULL(ptr)
SCL_ASSERT_NOT_NULL(ptr)

// 组织
SCL_TEST_UNIT(name) { ... }
SCL_TEST_SUITE(name) { ... }
SCL_TEST_RETRY(name, count) { ... }
SCL_TEST_TAGGED(name, "tag1", "tag2") { ... }

// 控制
SCL_SKIP("reason")
SCL_FAIL("message")
```

### RAII 守卫 (guard.hpp)
```cpp
Sparse mat = make_sparse_csr(...);
Dense view = wrap_dense(...);
// 自动清理，无需手动 destroy
```

### 数据生成 (data.hpp)
```cpp
Random rng(seed);

// 随机 shape
auto [rows, cols] = random_shape(10, 100, rng);

// 随机矩阵
auto mat = random_sparse_csr(rows, cols, density, rng);
auto mat2 = random_sparse_random_shape(10, 100, 0.01, 0.1, rng);

// 批量生成
auto matrices = batch_random_shapes(10, 10, 100, 0.05, rng);

// 结构化矩阵
auto identity = identity_csr(n);
auto diagonal = diagonal_csr(diag_values);
auto symmetric = property::symmetric(n, density, rng);
```

### Eigen 参考 (oracle.hpp)
```cpp
// 转换
auto eigen_mat = to_eigen_csr(scl_mat);
auto csr_arrays = from_eigen_csr(eigen_mat);

// 参考操作
auto transposed = oracle::transpose_csr_to_csc(mat);
auto cloned = oracle::clone_csr(mat);
auto result = oracle::add_csr(A, B);
```

### 精度比较 (precision.hpp)
```cpp
using precision::Tolerance;

// 标量比较
SCL_ASSERT_TRUE(precision::approx_equal(a, b, Tolerance::normal()));

// 向量比较
SCL_ASSERT_TRUE(precision::vectors_equal(v1, v2, Tolerance::strict()));

// 矩阵比较
SCL_ASSERT_TRUE(precision::matrices_equal(A, B, Tolerance::relaxed()));

// 相对误差
double rel_err = precision::relative_error(result, expected);
SCL_ASSERT_LT(rel_err, 1e-9);

// 统计验证
auto stats = precision::compute_statistics(errors);
SCL_ASSERT_TRUE(precision::error_stats_acceptable(stats));
```

### BLAS 参考 (blas.hpp)
```cpp
// 向量操作
double dot = blas::dot(x, y);
double norm = blas::norm2(x);

// 矩阵向量
blas::gemv(false, m, n, alpha, A, x, beta, y);

// 矩阵矩阵
blas::gemm(false, false, m, n, k, alpha, A, B, beta, C);
```

## 🎯 具体任务列表

### Core 模块 (已完成 ✅)

- [x] test_core.cpp - core.h (41测试) ✅
- [x] test_dense_complete.cpp - dense.h (39测试) ✅
- [ ] test_sparse_complete.cpp - sparse.h (需完善)
- [ ] test_unsafe.cpp - unsafe.h

### Kernel 模块 (待分配)

#### 线性代数 Kernel
- [ ] test_kernel_spmv.cpp - 稀疏矩阵向量乘
- [ ] test_kernel_gemm.cpp - 稠密矩阵乘
- [ ] test_kernel_gemv.cpp - 稠密矩阵向量乘

#### 统计 Kernel
- [ ] test_kernel_sum.cpp - 求和归约
- [ ] test_kernel_mean.cpp - 均值方差
- [ ] test_kernel_norm.cpp - 范数计算

#### 元素操作 Kernel
- [ ] test_kernel_elementwise.cpp - 元素级操作
- [ ] test_kernel_comparison.cpp - 比较操作
- [ ] test_kernel_reduction.cpp - 归约操作

## 📐 示例：完整测试文件

参考 `test_core.cpp` (41个测试，覆盖所有函数和边界情况)：

**结构：**
- 6 个 test suite
- 41 个 test case  
- 覆盖所有边界：NULL、无效参数、边界值
- 100% 通过率

**运行：**
```bash
make test-core                # 快速运行
make test-core VERBOSE=1      # 详细输出
```

## 🔧 调试失败的测试

### 1. 查看详细错误
```bash
make test-<name> VERBOSE=1
```

### 2. 运行单个测试
```bash
./units/test_<name> --filter "test_name"
```

### 3. 使用 GDB 调试
```bash
gdb ./units/test_<name>
(gdb) run --filter "failing_test"
(gdb) bt  # 查看调用栈
```

### 4. 检查段错误
```bash
# 编译 debug 版本
make build-<name> DEBUG=1

# 运行特定测试
./units/test_<name> --filter "problem_test"
```

## ⚠️ 常见问题

### Q1: 测试编译失败
```bash
# 确保主库已编译
cd ../..
make compile-cpp

# 清理重新编译
cd tests/c_api
make clean
make build-<name>
```

### Q2: 链接错误 (undefined reference)
```bash
# 检查库是否存在
ls -la ../../python/scl/libs/libscl_f64_i64.so

# 如不存在，编译主库
cd ../..
make compile-cpp
```

### Q3: 运行时段错误
```bash
# 使用过滤器隔离问题
./units/test_<name> --list  # 查看所有测试
./units/test_<name> --filter "specific_test"  # 只运行一个

# 逐个测试
for test in $(./units/test_<name> --list | grep "  " | awk '{print $1}'); do
    echo "Testing: $test"
    ./units/test_<name> --filter "$test" || echo "FAILED: $test"
done
```

### Q4: 数值精度问题
```cpp
// 放宽容差
SCL_ASSERT_TRUE(matrices_equal(A, B, Tolerance::relaxed()));

// 或使用相对误差
double rel_err = relative_error(result, expected);
SCL_ASSERT_LT(rel_err, 1e-6);  // 更宽松的阈值
```

## 📊 测试报告

### 生成测试报告
```bash
# JSON 报告
./units/test_<name> --json report.json

# JUnit XML（CI）
./units/test_<name> --xml results.xml

# HTML 报告
./units/test_<name> --html report.html

# Markdown
./units/test_<name> --markdown report.md

# TAP 格式
./units/test_<name> --tap
```

## 🚀 提交清单

提交测试前确认：

- [ ] 编译通过: `make build-<name>`
- [ ] 所有测试通过: `make test-<name>`
- [ ] 详细模式通过: `make test-<name> VERBOSE=1`
- [ ] 覆盖所有函数
- [ ] 覆盖所有边界情况
- [ ] 包含随机测试（带 retry）
- [ ] 包含参考实现对比
- [ ] 精度验证正确
- [ ] 代码注释清晰
- [ ] 测试名称描述性强

## 📚 参考资料

- **测试框架文档**: `README.md`
- **API 参考**: `scl/binding/c_api/core/*.h`
- **示例测试**: `test_core.cpp`, `test_dense_complete.cpp`
- **工具文档**: 各 `.hpp` 文件头部注释

---

**版本**: 1.0
**更新**: 2025-12-30
**维护**: SCL Core Team

