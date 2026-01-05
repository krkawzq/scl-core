# scl/core/type.hpp 详解笔记

`scl/core/type.hpp` 是 SCL 基础类型配置的核心文件，定义了跨平台、精度可配的数值类型体系。内容涵盖类型别名、扩展类型自动适配、元信息、Traits/Concepts 及类型工具。下面按类型定义逐项详细说明。

---

## 1. 扩展原生类型（C++平台相关特殊类型）

> **命名空间：** `scl::detail`

- **int128_native_t, uint128_native_t**  
  条件编译，若平台/编译器原生支持使用 `__int128` 和 `unsigned __int128`，则定义这两种128位整数型，用于SCL的128位整数封装。
- **float16_native_t**  
  若硬件原生支持`_Float16`（如x86/ARM gcc/clang泛用符号），或 `__fp16` (ARM专用)，封装为`float16_native_t`。
- **float128_native_t**  
  若支持 IEEE 754 binary128（如GCC的`__float128`），则定义为`float128_native_t`。

---

## 2. C风格类型别名（带 `scl_` 前缀，分为三大类）

> **命名空间：** `scl`

### 2.1 整型（有符号和无符号）

**有符号（signed）**
- `scl_int8_t`     → `std::int8_t`      （8位有符号，范围-128~127）
- `scl_int16_t`    → `std::int16_t`     （16位有符号）
- `scl_int32_t`    → `std::int32_t`     （32位有符号）
- `scl_int64_t`    → `std::int64_t`     （64位有符号）
- `scl_int128_t`   → `__int128`/`int128_native_t`（128位有符号，依赖平台，SCL_ENABLE_INT128控制）

**无符号（unsigned）**
- `scl_uint8_t`    → `std::uint8_t`     （8位无符号，0~255）
- `scl_uint16_t`   → `std::uint16_t`    （16位无符号）
- `scl_uint32_t`   → `std::uint32_t`    （32位无符号）
- `scl_uint64_t`   → `std::uint64_t`    （64位无符号）
- `scl_uint128_t`  → `unsigned __int128`/`uint128_native_t`（128位无符号）

**注意:**  
- 128位整数类型仅在`SCL_ENABLE_INT128`为1（平台/编译器支持时）才定义，否则无对应类型。

### 2.2 浮点型（IEEE标准）

- `scl_float16_t`  → 16位半精度浮点（需平台支持，SCL_ENABLE_FLOAT16控制）
- `scl_float32_t`  → 32位单精度浮点（通常为`float`）
- `scl_float64_t`  → 64位双精度浮点（通常为`double`）
- `scl_float128_t` → 128位四精度浮点（需平台支持，SCL_ENABLE_FLOAT128控制）

### 2.3 其他基础类型

- `scl_null_t`     → `std::nullptr_t`  
  用于不支持高精度类型时的类型占位。
- `scl_size_t`     → `std::size_t`       （无符号，数据/内存块计数）
- `scl_ptrdiff_t`  → `std::ptrdiff_t`    （有符号，指针差值、跨度）
- `scl_byte_t`     → `std::byte`         （原始字节）
- `scl_mask_t`     → `std::uint32_t`     （掩码用）

---

## 3. C++风格类型别名（更现代化，简明命名）

### 3.1 整型（大写 IntX/UIntX）

- `Int8`, `Int16`, `Int32`, `Int64`, (`Int128`)  
- `UInt8`, `UInt16`, `UInt32`, `UInt64`, (`UInt128`)
- `Int128`/`UInt128` 若平台不支持则为 `scl_null_t`

### 3.2 浮点型（RealX）

- `Real16`    → 16位半精度浮点，若不支持为`scl_null_t`
- `Real32`    → 32位浮点 (等价于`float`)
- `Real64`    → 64位浮点 (等价于`double`)
- `Real128`   → 128位浮点，若不支持为`scl_null_t`

**别名兼容:**  
- `Float32` = `Real32`；`Float64` = `Real64`（兼容老风格）

### 3.3 索引类与工具

- `Index32`   → 32位整型索引/下标
- `Index64`   → 64位整型索引  
- `Size`      → `size_t`，用于容器长度
- `Stride`    → `ptrdiff_t`，用于步幅（如跨数组跳步）
- `Offset`    → `ptrdiff_t`，指针偏移
- `Byte`      → 字节类型
- `Mask`      → 掩码

---

## 4. 默认类型配置（支持编译开关，灵活控制默认数值精度）

- `Index`   默认等于`Index32`，可用 `-DSCL_DEFAULT_INDEX64`/`-DSCL_DEFAULT_INDEX32` 切换
- `Real`    默认等于`Real64`，可切换为`Real32`
- `Int`     默认等于`Int32`，可切换为`Int64`
- `Uint`    默认等于`UInt32`，可切换为`UInt64`

相关的 `kDefaultIndexBits`, `kDefaultRealBits`, `kDefaultIntBits`, `kDefaultUintBits` 为编译期常量，便于模板分支/调度等用途。

---

## 5. TypeInfo 类型信息结构

- `TypeInfo` 结构体收集了库内相关类型的字节数、名字、是否支持扩展精度等元信息
    - `kRealTypeName` / `kIndexTypeName` ：当前默认 Real、Index 类型对应名字
    - `kHasFloat16/128`、`kHasInt128`    ：是否有相关扩展类型
    - `kInt8Size`, `kInt32Size`, `kReal64Size` ... 各基础类型字节数
    - 若有扩展类型，包含 `kReal16Size`, `kReal128Size`, `kInt128Size`
- 用于元编程，或性能/精度动态调度

---

## 6. 类型特征与 Traits

常用的静态布尔模板（变量模板）：
- `is_int_v<T>`           ：标准有符号整数（Int8~Int64）
- `is_uint_v<T>`          ：标准无符号整数
- `is_real_v<T>`          ：标准浮点
- `is_index_v<T>`         ：标准下标类型
- `is_int_extended_v<T>`  ：128位有符号整数
- `is_real_extended_v<T>` ：16/128位浮点
- `is_int_any_v<T>`       ：所有有符号整数（含128位）
- `is_uint_any_v<T>`      ：所有无符号整数（含128位）
- `is_real_any_v<T>`      ：所有浮点（含16/128）
- `is_numeric_v<T>`       ：任何数值类型（整数/浮点全含）
- `is_supported_value_type_v<T>` ：标准可用于算法/向量化的类型

---

## 7. 类型 Concepts（C++20概念）

标准及SCL定制概念，如：
- `Arithmetic<T>`, `FloatingPoint<T>`, `Integral<T>`           // 标准
- `Integer`, `Unsigned`, `Floating`, `Indexing`                // SCL基本类型
- `IntegerAny`, `UnsignedAny`, `FloatingAny`, `Numeric`        // 含扩展类型

用于高层泛型约束、模板筛选，提高类型安全与自动文档。

---

## 8. 值类型分类（ValueCategory）

- 枚举类型 `ValueCategory`：`kReal`（浮点）、`kInt`（有符号整型）、`kUint`（无符号整型）
- 函数模板 `value_category<T>()` 编译期返回T归属的值类别
- 用于算法分支（如混合精度、向量化分派）

---

## 9. 类型工具（scl::type 命名空间）

- `size_bytes<T>()`         ：类型T的字节数
- `alignment_bytes<T>()`    ：类型T的对齐字节
- `is_compatible_precision<T,U>()`  ：T和U字节数是否一致
- `wider_t<T,U>`            ：获取T/U中更宽（高精度）类型

均为constexpr/编译期常量，用于高性能模板运算和SFINAE。

---

## 10. 精度支持检测与SFINAE工具

- 静态布尔变量 `kHasReal16Support`, `kHasReal128Support`, `kHasInt128Support`, `kHasUInt128Support` 判断相关扩展类型是否可用
- 模板 `is_precision_supported<P>()` : 精度类型P是否可用
- SFINAE工具 `enable_if_supported_t<P>` / `enable_if_unsupported_t<P>`
- C++20概念 `SupportedPrecision<P>`、`UnsupportedPrecision<P>`

主要用于模板特化、分支和静态断言，防止在不支持平台误用高精度类型。

---

## 结论

`scl/core/type.hpp` 通过大量的条件编译、类型封装和泛型工具，保证了SCL的所有代码均可安全、高效、可扩展地在不同平台/编译器下使用最优的基础数值类型，同时便于静态调度和类型安全编程，是 SCL 数值与容器体系的根基之一。

**建议查阅:**  
- 类型相关的“默认值类型配置”，配合编译选项灵活切换（影响数据精度/内存占用）。
- 所有结构和类型都可按本文件备注或源码注释查看详细介绍，实用工具和traits建议优先用。

---

