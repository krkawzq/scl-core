# scl/core/error.hpp 详细笔记

## 文件介绍
本文件为 `scl/core/error.hpp`，提供了 SCL（据上下文可能为科学/矩阵/张量库）项目全局的**异常处理、错误码、断言、调试与线程局部错误状态**支持。覆盖了错误检查、异常类、错误码与跨 ABI（C/C++）交互的详尽机制。

---

## 1. 错误码（ErrorCode）

### 枚举说明

`enum class ErrorCode : std::int32_t`

- 设计为 32 位整型枚举，确保跨 C-ABI 兼容。
- 按照不同错误类型分区，数值稳定，发布后不能随意更改。
- 主体区间划分如下（详见后表）：

| 区间        | 数值范围      | 含义           | 举例              |
|------------|--------------|----------------|-------------------|
| Success    | 0            | 成功           | `Success` |
| General    | 1-99         | 通用/未分类错误 | `Unknown`, `NotImplemented`, ... |
| Memory     | 100-199      | 内存相关       | `OutOfMemory`,... |
| Dimension  | 200-299      | 张量维/形状相关 | `ShapeMismatch`, ...|
| Type       | 300-399      | 类型与精度相关 | `TypeMismatch` ...|
| Value      | 400-499      | 参数与数值相关 | `InvalidArgument`,...|
| IO         | 500-599      | 输入输出       | `FileNotFound`,...|
| Algorithm  | 600-699      | 算法数值相关   | `ConvergenceError`, ...|
| Threading  | 700-799      | 线程相关       | `ThreadError`,...|
| Hardware   | 800-899      | 硬件平台相关   | `HardwareError`,...|
| Internal   | 900-999      | 内部错误       | `InternalError`,...|

### 常用错误码举例

- `Success`：操作成功，无错误。
- `Unknown`：未知错误，捕获异常但类型未知。
- `OutOfMemory`：内存分配失败。
- `ShapeMismatch`：张量形状不一致。
- `TypeMismatch`：类型不匹配。
- `InvalidArgument`：参数非法。
- `FileNotFound`：文件不存在。
- `ComputationError`：一般算法错误。
- `ThreadError`：线程异常。
- `InternalError`：内部逻辑错误。

### 方法说明

- `constexpr auto error_code_name(ErrorCode code) noexcept -> const char*`
    - 作用：将 `ErrorCode` 转为字符串，如 "ShapeMismatch"。
    - 用法：调试/报错时，明确输出错误类型。
- `constexpr auto error_code_category(ErrorCode code) noexcept -> const char*`
    - 作用：返回错误码所属大类，如 "Memory"、"Type"。
- `constexpr auto is_success(ErrorCode code) noexcept -> bool`
    - 作用：代码是否表示成功（即 `ErrorCode::Success`）。
- `constexpr auto is_error(ErrorCode code) noexcept -> bool`
    - 作用：代码是否代表错误（非常用 `Success`）。
- `constexpr auto is_recoverable(ErrorCode code) noexcept -> bool`
    - 作用：是否为可恢复性错误（不包括硬件/内部）。

---

## 2. 异常体系（Exception Hierarchy）

### 基础类

- `class Error : public std::exception`
    - 全部 SCL 异常基类。
    - 主要成员：
        - `ErrorCode code_`：错误码，跨 ABI 与 C 对应。
        - `std::string message_`：异常详细消息。
        - `source_location location_`：异常源代码定位（文件、行号、函数）。
        - `std::string full_message_`：线程安全的完整消息（含位置信息）。
    - 常用虚方法说明：
        - `const char* what() const noexcept override`
            - 标准 C++ 异常接口，返回完整消息。
        - `ErrorCode code() const noexcept`
            - 取错误码，方便进一步处理。
        - `const std::string& message() const noexcept`
            - 获取消息体。
        - `const source_location& location() const noexcept`
            - 源代码位置信息。

### 主要子类说明（部分）

- `ValueError`：参数值相关错误，基类。
- `IndexError`：索引超界。
- `RangeError`：数值范围相关错误。
- `DimensionError`：维度相关错误（如形状混淆）。
- `ShapeMismatchError`：张量形状不一致。
- `TypeError`：类型错误，主要由模板推导自动抛出。
- `DtypeMismatchError`：数据类型不一致。
- `NullPointerError`：空指针解引用。
- `MemoryError`：内存错误的基类。
- `OutOfMemoryError`：内存分配不足。
- `ComputationError`：一般数值/算法出错。
- `ConvergenceError`：算法未收敛问题。
- `NotImplementedError`：功能未实现（用于占位或抽象接口）。
- `AssertionError`：显式断言失败。
- `IoError`：输入输出相关。
- `FileNotFoundError`：无法找到文件。
- 其余见代码详细注释。

每个异常都支持（通常）接收错误内容和定位，便于精细化诊断。

---

## 3. 线程局部错误状态（ThreadErrorState）

- 设计为**每线程一个实例**（`thread_local`），无锁高效。
- 类静态方法 `instance()` 获得 per-thread 单例。
- 主要成员：  
    - `ErrorCode code_`：本线程最新错误码。
    - `std::array<char, kMaxMessageLength> message_`：错误消息，最大长度限制。
- 主要方法：
    - `void clear()`：清除错误状态，置为 Success。
    - `void set(ErrorCode, const char* message = nullptr)`：设置错误与消息。
    - `void set_from_exception(const Error& error)`：从 SCL 异常复制错误+消息。
    - `ErrorCode code() const`：获取当前错误码。
    - `const char* message() const`：获取消息（C 风格字符串）。
    - `bool has_error() const`：当前是否有错误。
    - `const char* category() const`：分类字符串。
    - `bool is_recoverable() const`：是否为可恢复。
- 全局辅助函数（inline）：
    - `get_thread_error()`：取本线程 error state 单例引用。
    - `clear_thread_error()`：重置本线程错误。
    - `set_thread_error(ErrorCode, const char*)`：设置。
    - `set_thread_error_from_exception(const Error&)`：SCL 异常转为错误状态。
    - `thread_has_error()`：是否有错。
    - `get_thread_error_code()`、`get_thread_error_message()`：当前错误码与消息。

---

## 4. 错误格式化与断言工具

### 格式化工具

- `detail::format_error`：
    - C++20 支持自动用 `std::format` 否则 fallback 到 `snprintf`。
    - 通常用来格式化错误消息，结合 source_location 使用。
- `detail::format_message`：不用定位，仅格式化文本参数。

### 断言失败处理

- `detail::debug_assert_fail(expr, location)`：调试断言失败，直接标准错误输出并 abort。
- `detail::debug_assert_fail_msg(expr, msg, location)`：调试断言失败，带消息补充，abort。

---

## 5. 编译时类型检查模板

提供适用于模板参数的类型专用静态检查。异常不用，编译“断言”：

- `check_floating<T>()`: `T` 必须为浮点型（`static_assert`）。
- `check_integral<T>()`: `T` 必须为整数型。
- `check_arithmetic<T>()`: `T` 必须为算术型。
- `check_signed<T>()` / `check_unsigned<T>()`: 有无符号检查。
- `check_size<T, min>()`: 指定类型尺寸不能过小。
- `check_same<T, U>()`: 类型强制一致。

---

## 6. 运行时异常检查工具

### 参数检查模板（抛出对应异常）

- `check_arg(bool cond, ...)`：参数检查，否抛 `ValueError`。
- `check_dim(bool cond, ...)`：维空间/形状检查，否抛 `DimensionError`。
- `check_range(bool cond, ...)`：范围类，否抛 `RangeError`。
- `check_mem(bool cond, ...)`：内存相关，否抛 `MemoryError`。
- `check_type(bool cond, ...)`：类型相关，否抛 `TypeError`。
- `check_io(bool cond, ...)`：I/O 检查，否抛 `IoError`。
- `check_compute(bool cond, ...)`：数值算法错误，否抛 `ComputationError`。

### 专用检查

- `check_not_null(const T* ptr, name)`：非空指针，否抛 `NullPointerError`。
- `check_index(index, size)`：索引越界检测，否抛 `IndexError`。
- `check_size_match(a, b)`：尺寸不一致，否抛 `DimensionError`。
- `check_positive(value, name)`：必须大于 0，否抛 `ValueError`。
- `check_non_negative(value, name)`：必须 >= 0，否抛 `ValueError`。
- `check_alignment<T, N>(ptr)`：指针对齐（N 字节），否抛 `AlignmentError`。
- `check_finite(val)`：必须有限（不能为 +/-Inf/NaN），否则抛相应 `NaNError/ValueError`。
- `not_implemented(feature)`：直接抛 `NotImplementedError`。
- `unreachable()`：不可达代码抛 `InternalError`。

---

## 7. 调试模式断言（Debug-Only）

### 仅调试版本有效，release 下为 no-op，失败直接 abort

- `debug_check(cond, msg, loc)`：条件检查。
- `debug_check_size_match(a, b, msg, loc)`：尺寸一致性。
- `debug_check_not_null(ptr, name, loc)`：非空检查。
- `debug_check_index(index, size, loc)`：下标越界检查。
- `debug_check_positive(value, name, loc)`：正值检查。
- `debug_check_non_negative(value, name, loc)`：非负值。
- `debug_check_alignment<N>(ptr, loc)`：指针对齐检查。

---

## 8. 其他说明

- 源文件依赖 `scl/core/source_location.hpp`。全部错误都可定位到具体源代码位置。
- 尽量避免直接写 `throw std::runtime_error`，而统一经上述类型和工具抛出。
- 自动区分“开发调试期/部署生产环境”的断言与容错策略。
- 与线程相关的错误及内部错误单独分区，避免误用。

---

## 最佳实践

- 通常仅用异常基类与检查模板，不直接操作 `ErrorCode`，除非需要与 C API 交互。
- 编写算法/函数时尽量用 `SCL_CHECK` 或各类 `check_*`，统一异常体系。
- 接口暴露给 C/C++/Python 多端时，使用统一 ErrorCode 保证兼容。

---


