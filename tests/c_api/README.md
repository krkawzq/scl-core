# SCL Core - C API Test Framework

基于 `core.hpp` 的现代 C++ 测试框架，pytest 风格输出，专为 CI/CD 集成设计。

## ✨ 特性

### 🎨 输出格式
- **Human** - 彩色 pytest 风格输出（默认）
- **TAP** - Test Anything Protocol (标准 CI 格式)
- **JSON** - 机器可读报告
- **JUnit XML** - Jenkins/GitLab 集成
- **HTML** - 美观的网页报告（深色主题）
- **Markdown** - 文档友好
- **GitHub Actions** - 原生注解
- **TeamCity** - 服务消息
- **Minimal** - 点号输出

### 🎯 核心功能
- ✅ 自动测试注册（`__COUNTER__` 魔法）
- ✅ 丰富的断言宏（`SCL_ASSERT_*`）
- ✅ 测试套件（Suites）
- ✅ 标签过滤（Tags）
- ✅ Skip/XFail 标记
- ✅ Fixtures（setup/teardown）
- ✅ 参数化测试
- ✅ 性能基准测试
- ✅ 超时控制
- ✅ 失败重试
- ✅ 随机顺序执行
- ✅ 进度条
- ✅ CI 环境自动检测

## 🚀 快速开始

### 1. 编写测试

```cpp
#include "test.hpp"

SCL_TEST_BEGIN

SCL_TEST_UNIT(my_first_test) {
    SCL_ASSERT_EQ(1 + 1, 2);
    SCL_ASSERT_TRUE(42 > 0);
}

SCL_TEST_UNIT(another_test) {
    SCL_ASSERT_NE(1, 2);
}

SCL_TEST_END

SCL_TEST_MAIN()
```

### 2. 编译测试

```bash
g++ -std=c++20 -I. -Itests/c_api/include -o my_test my_test.cpp
```

### 3. 运行测试

```bash
# 基本运行
./my_test

# 详细输出
./my_test --verbose

# TAP 格式（CI）
./my_test --tap

# 导出报告
./my_test --json report.json --xml results.xml --html report.html

# 过滤测试
./my_test --filter "matrix"

# 失败即停
./my_test --fail-fast

# 随机顺序
./my_test --shuffle --seed 42

# GitHub Actions 模式
./my_test --github --fail-fast
```

## 📊 输出示例

### Human 模式（默认）

```
═══════════════════════════════════════════════════════════════════════════════
  🧪 SCL Core Test Suite
═══════════════════════════════════════════════════════════════════════════════
  Tests: 2
───────────────────────────────────────────────────────────────────────────────

  ✓ test_example_1 (1.23ms)
  ✓ test_example_2 (0.45ms)

═══════════════════════════════════════════════════════════════════════════════
  ✅ All tests passed!
═══════════════════════════════════════════════════════════════════════════════

  2 passed in 1.68ms
```

### TAP 模式

```
TAP version 14
1..2
ok 1 - test_example_1
ok 2 - test_example_2
```

### Minimal 模式

```
..

2/2 passed (0.00s)
```

## 🔧 断言宏

### 基本断言
```cpp
SCL_ASSERT(expr)
SCL_ASSERT_MSG(expr, "custom message")
```

### 比较断言
```cpp
SCL_ASSERT_EQ(expected, actual)
SCL_ASSERT_NE(a, b)
SCL_ASSERT_LT(a, b)   // a < b
SCL_ASSERT_LE(a, b)   // a <= b
SCL_ASSERT_GT(a, b)   // a > b
SCL_ASSERT_GE(a, b)   // a >= b
```

### 布尔断言
```cpp
SCL_ASSERT_TRUE(expr)
SCL_ASSERT_FALSE(expr)
```

### 指针断言
```cpp
SCL_ASSERT_NULL(ptr)
SCL_ASSERT_NOT_NULL(ptr)
```

### 浮点断言
```cpp
SCL_ASSERT_NEAR(expected, actual, tolerance)
// Example: SCL_ASSERT_NEAR(3.14159, 3.14, 0.01)
```

### 字符串断言
```cpp
SCL_ASSERT_STR_EQ("hello", str)
SCL_ASSERT_STR_CONTAINS("hello world", "world")
```

### 异常断言
```cpp
SCL_ASSERT_THROWS(expr, exception_type)
SCL_ASSERT_NO_THROW(expr)
```

### 失败/跳过
```cpp
SCL_FAIL("reason")
SCL_SKIP("reason")
SCL_SKIP_IF(condition, "reason")
```

## 🛠️ CLI 选项

### 输出格式
```bash
--human               # 人类可读（默认）
--tap                 # TAP 格式
--json <file>         # JSON 报告
--xml <file>          # JUnit XML
--html <file>         # HTML 报告
--markdown <file>     # Markdown
--github              # GitHub Actions
--teamcity            # TeamCity
--minimal             # 点号输出
--quiet, -q           # 仅失败
```

### 过滤
```bash
--filter <pattern>    # 名称匹配
--exclude <pattern>   # 排除名称
--tag <tag>           # 按标签
--exclude-tag <tag>   # 排除标签
--suite <name>        # 按套件
--list                # 列出测试
--list-tags           # 列出标签
```

### 执行控制
```bash
--fail-fast, -x       # 首次失败即停
--shuffle             # 随机顺序
--seed <n>            # 随机种子
--repeat <n>          # 重复 n 次
--timeout <ms>        # 超时（毫秒）
--retry <n>           # 重试次数
--dry-run             # 模拟运行
```

### 输出控制
```bash
--verbose, -v         # 详细输出
-vv                   # 调试输出
--no-color            # 禁用颜色
--no-progress         # 禁用进度条
--no-time             # 禁用计时
--show-all            # 显示所有
--show-slow <ms>      # 慢测试阈值
--capture             # 捕获 stdout/stderr
```

### 日志
```bash
--log <file>          # 详细日志
--tap-file <file>     # TAP 文件
```

## 🌍 环境变量

```bash
SCL_TEST_FILTER=pattern    # 默认过滤
SCL_TEST_TIMEOUT=30000     # 默认超时（毫秒）
SCL_TEST_COLOR=1           # 强制颜色（1/0）
CI=1                       # CI 友好模式
```

## ✅ 已修复的问题

### 宏参数冲突 ✅

**问题：** 宏参数 `name` 与 `TestInfo.name` 成员冲突。

**解决方案：** 将 `TestInfo.name` 重命名为 `TestInfo.name_str`，完全避免冲突。

**状态：** ✅ 已修复并验证通过。

## 🎯 未来改进

- [x] 修复宏参数冲突 ✅
- [ ] 添加 RAII 守卫（`guard.hpp`）
- [ ] 添加 Eigen 参考实现（`oracle.hpp`）
- [ ] 添加测试数据生成器（`data.hpp`）
- [ ] 实际 C API 测试用例
- [ ] 参数化测试（`SCL_TEST_P` - 需要重构）

## 📝 文件结构

```
tests/c_api/
├── include/
│   ├── test.hpp          # 主入口
│   └── core.hpp          # 完整测试框架（2955行）
├── src/
│   └── test_demo.cpp     # 完整功能演示 ✅
├── CMakeLists.txt        # CMake 配置
└── README.md             # 本文档

## 📜 许可

与主项目相同。

## 🙏 致谢

- pytest - 输出风格灵感
- Catch2 - API 设计参考
- Google Test - 断言设计
```

