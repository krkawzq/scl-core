# 统一Fast Path设计 - 最终方案

## 核心理念

**不需要手动分派Custom vs Virtual!**

编译器通过 `SparseLike` 概念和 `primary_values()` 统一访问器,
会自动为不同类型生成最优代码。

## 设计模式

### 之前的错误做法 ❌

```cpp
// 手动分派 - 代码重复!
template <typename T, bool IsCSR>
    requires CustomSparseLike<...>
void algorithm_custom_fast(...) { /* 实现A */ }

template <typename T, bool IsCSR>
    requires VirtualSparseLike<...>
void algorithm_virtual_fast(...) { /* 实现B - 几乎相同! */ }

template <typename MatrixT, bool IsCSR>
void algorithm_fast(MatrixT& mat) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        algorithm_custom_fast(mat);
    } else {
        algorithm_virtual_fast(mat);
    }
}
```

### 正确的做法 ✅

```cpp
// 统一实现 - 编译器自动优化!
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void algorithm_fast(MatrixT& mat) {
    const Index primary_dim = scl::primary_size(mat);
    
    scl::threading::parallel_for(0, primary_dim, [&](size_t p) {
        // 使用统一访问器
        auto vals = scl::primary_values(mat, p);
        
        // 访问底层指针 - 编译器会内联
        // CustomSparse: vals.ptr = mat.data + mat.indptr[p]
        // VirtualSparse: vals.ptr = mat.data_ptrs[p]
        auto* data_ptr = vals.ptr;
        Size len = vals.len;
        
        // SIMD处理 - 对两种类型都高效
        // ...
    });
}
```

## 为什么这样更好?

### 1. 编译器优化

**CustomSparse 调用**:
```cpp
CustomSparse<Real, true> mat;
log1p_inplace_fast(mat);

// 编译器内联后:
// auto vals = mat.primary_values(i);
// → return Array(mat.data + mat.indptr[i], len);
// → vals.ptr 直接是 mat.data + offset
// 零开销!
```

**VirtualSparse 调用**:
```cpp
VirtualSparse<Real, true> mat;
log1p_inplace_fast(mat);

// 编译器内联后:
// auto vals = mat.primary_values(i);
// → return Array(mat.data_ptrs[i], mat.lengths[i]);
// → vals.ptr 是单次指针解引用
// 最优!
```

### 2. 代码简洁

- 单一实现,无重复
- 编译器看到完整上下文,优化更好
- 维护成本低

### 3. 性能相同

编译器的内联和优化能力足够强大:
- 对CustomSparse: 生成批量访问代码
- 对VirtualSparse: 生成行内访问代码
- 零运行时分派开销

## 实施策略

### 所有fast_impl统一使用此模式:

```cpp
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void xxx_fast(MatrixT& mat, ...) {
    scl::threading::parallel_for(0, primary_dim, [&](size_t p) {
        auto vals = scl::primary_values(mat, p);
        auto* data = vals.ptr;  // 编译器优化
        Size len = vals.len;
        
        // 4-way unrolled SIMD
        // 对Custom和Virtual都高效
    });
}
```

## 结论

**SparseLike概念 + 统一访问器 = 完美抽象**

- ✅ 零运行时开销
- ✅ 编译器自动优化
- ✅ 代码简洁
- ✅ Custom和Virtual都快
- ✅ 易于维护

这就是现代C++的威力! 🚀
