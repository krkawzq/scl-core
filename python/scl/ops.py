"""
SCL Ops - 算子封装

提供高级操作接口，自动处理后端选择和延迟执行。
"""

from __future__ import annotations

from typing import Optional, Union, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from scl.sparse import SclCSR, SclCSC

from scl._ffi import get_lib_with_signatures, check_error, Backend
from scl.sparse import SclCSR, SclCSC
from scl.lazy import LazyView, LazyReorder, PairSparse, LazyVStack, hstack, vstack


def _ensure_numpy():
    import numpy as np
    return np


# =============================================================================
# 归一化
# =============================================================================

def normalize(mat: SclCSR, norm: str = "l2", axis: int = 1,
              inplace: bool = False) -> SclCSR:
    """
    归一化

    Args:
        mat: 输入矩阵
        norm: "l1", "l2", "max"
        axis: 0=列, 1=行
        inplace: 是否原地修改

    Returns:
        归一化后的矩阵
    """
    np = _ensure_numpy()

    mat.materialize()

    if axis != 1:
        raise NotImplementedError("Only axis=1 (row normalization) supported")

    if mat._handle is not None:
        lib = get_lib_with_signatures()

        new_data = np.empty(mat.nnz, dtype=np.float32)
        new_indices = np.empty(mat.nnz, dtype=np.int32)
        new_indptr = np.empty(mat.shape[0] + 1, dtype=np.int32)

        if norm == "l1":
            check_error(lib.scl_mmap_csr_normalize_l1(
                mat._handle,
                new_data.ctypes.data,
                new_indices.ctypes.data,
                new_indptr.ctypes.data
            ))
        elif norm == "l2":
            check_error(lib.scl_mmap_csr_normalize_l2(
                mat._handle,
                new_data.ctypes.data,
                new_indices.ctypes.data,
                new_indptr.ctypes.data
            ))
        else:
            raise ValueError(f"Unknown norm: {norm}")

        if inplace:
            mat._data = new_data
            mat._indices = new_indices
            mat._indptr = new_indptr
            return mat
        else:
            return SclCSR.from_arrays(new_data, new_indices, new_indptr, mat.shape)
    else:
        # 纯 Python 实现
        new_data = mat._data.copy()

        for i in range(mat.shape[0]):
            start, end = mat._indptr[i], mat._indptr[i + 1]
            if start == end:
                continue

            vals = new_data[start:end]

            if norm == "l1":
                n = np.sum(np.abs(vals))
            elif norm == "l2":
                n = np.sqrt(np.sum(vals ** 2))
            elif norm == "max":
                n = np.max(np.abs(vals))
            else:
                raise ValueError(f"Unknown norm: {norm}")

            if n > 0:
                new_data[start:end] = vals / n

        if inplace:
            mat._data = new_data
            return mat
        else:
            return SclCSR.from_arrays(
                new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
            )


# =============================================================================
# 变换
# =============================================================================

def log1p(mat: SclCSR, inplace: bool = False) -> SclCSR:
    """
    log1p 变换: log(1 + x)

    Args:
        mat: 输入矩阵
        inplace: 是否原地修改
    """
    np = _ensure_numpy()

    mat.materialize()

    if mat._handle is not None:
        lib = get_lib_with_signatures()

        new_data = np.empty(mat.nnz, dtype=np.float32)
        new_indices = np.empty(mat.nnz, dtype=np.int32)
        new_indptr = np.empty(mat.shape[0] + 1, dtype=np.int32)

        check_error(lib.scl_mmap_csr_log1p(
            mat._handle,
            new_data.ctypes.data,
            new_indices.ctypes.data,
            new_indptr.ctypes.data
        ))

        if inplace:
            mat._data = new_data
            mat._indices = new_indices
            mat._indptr = new_indptr
            return mat
        else:
            return SclCSR.from_arrays(new_data, new_indices, new_indptr, mat.shape)
    else:
        new_data = np.log1p(mat._data)

        if inplace:
            mat._data = new_data
            return mat
        else:
            return SclCSR.from_arrays(
                new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
            )


def scale(mat: SclCSR, row_factors: Optional["np.ndarray"] = None,
          col_factors: Optional["np.ndarray"] = None,
          inplace: bool = False) -> SclCSR:
    """
    缩放

    Args:
        mat: 输入矩阵
        row_factors: 行缩放因子
        col_factors: 列缩放因子
        inplace: 是否原地修改
    """
    np = _ensure_numpy()

    mat.materialize()

    new_data = mat._data.copy() if not inplace else mat._data
    new_indices = mat._indices if inplace else mat._indices.copy()
    new_indptr = mat._indptr if inplace else mat._indptr.copy()

    if mat._handle is not None and row_factors is not None:
        lib = get_lib_with_signatures()
        row_factors = np.ascontiguousarray(row_factors, dtype=np.float32)

        check_error(lib.scl_mmap_csr_scale_rows(
            mat._handle,
            row_factors.ctypes.data,
            new_data.ctypes.data,
            new_indices.ctypes.data,
            new_indptr.ctypes.data
        ))
    elif row_factors is not None:
        for i in range(mat.shape[0]):
            start, end = new_indptr[i], new_indptr[i + 1]
            new_data[start:end] *= row_factors[i]

    if mat._handle is not None and col_factors is not None:
        lib = get_lib_with_signatures()
        col_factors = np.ascontiguousarray(col_factors, dtype=np.float32)

        check_error(lib.scl_mmap_csr_scale_cols(
            mat._handle,
            col_factors.ctypes.data,
            new_data.ctypes.data,
            new_indices.ctypes.data,
            new_indptr.ctypes.data
        ))
    elif col_factors is not None:
        for i in range(mat.shape[0]):
            start, end = new_indptr[i], new_indptr[i + 1]
            for k in range(start, end):
                new_data[k] *= col_factors[new_indices[k]]

    if inplace:
        mat._data = new_data
        return mat
    else:
        return SclCSR.from_arrays(new_data, new_indices, new_indptr, mat.shape)


# =============================================================================
# 过滤
# =============================================================================

def filter_threshold(mat: SclCSR, threshold: float) -> SclCSR:
    """
    过滤小于阈值的元素

    Args:
        mat: 输入矩阵
        threshold: 阈值 (绝对值)

    Returns:
        过滤后的矩阵
    """
    np = _ensure_numpy()
    from ctypes import c_int64, byref

    mat.materialize()

    if mat._handle is not None:
        lib = get_lib_with_signatures()

        # 分配最大可能的空间
        new_data = np.empty(mat.nnz, dtype=np.float32)
        new_indices = np.empty(mat.nnz, dtype=np.int32)
        new_indptr = np.empty(mat.shape[0] + 1, dtype=np.int32)

        out_nnz = c_int64()
        check_error(lib.scl_mmap_csr_filter_threshold(
            mat._handle,
            threshold,
            new_data.ctypes.data,
            new_indices.ctypes.data,
            new_indptr.ctypes.data,
            byref(out_nnz)
        ))

        nnz = out_nnz.value
        return SclCSR.from_arrays(
            new_data[:nnz], new_indices[:nnz], new_indptr, mat.shape
        )
    else:
        new_data = []
        new_indices = []
        new_indptr = [0]

        for i in range(mat.shape[0]):
            start, end = mat._indptr[i], mat._indptr[i + 1]
            for k in range(start, end):
                if abs(mat._data[k]) >= threshold:
                    new_data.append(mat._data[k])
                    new_indices.append(mat._indices[k])
            new_indptr.append(len(new_data))

        return SclCSR.from_arrays(
            np.array(new_data, dtype=np.float32),
            np.array(new_indices, dtype=np.int32),
            np.array(new_indptr, dtype=np.int32),
            mat.shape
        )


def top_k(mat: SclCSR, k: int) -> SclCSR:
    """
    每行保留 top-k 元素 (按绝对值)

    Args:
        mat: 输入矩阵
        k: 保留的元素数

    Returns:
        过滤后的矩阵
    """
    np = _ensure_numpy()

    mat.materialize()

    if mat._handle is not None:
        lib = get_lib_with_signatures()

        # 分配空间 (最多 k * rows)
        max_nnz = k * mat.shape[0]
        new_data = np.empty(max_nnz, dtype=np.float32)
        new_indices = np.empty(max_nnz, dtype=np.int32)
        new_indptr = np.empty(mat.shape[0] + 1, dtype=np.int32)

        check_error(lib.scl_mmap_csr_top_k(
            mat._handle,
            k,
            new_data.ctypes.data,
            new_indices.ctypes.data,
            new_indptr.ctypes.data
        ))

        nnz = new_indptr[mat.shape[0]]
        return SclCSR.from_arrays(
            new_data[:nnz], new_indices[:nnz], new_indptr, mat.shape
        )
    else:
        new_data = []
        new_indices = []
        new_indptr = [0]

        for i in range(mat.shape[0]):
            start, end = mat._indptr[i], mat._indptr[i + 1]
            row_data = list(zip(mat._data[start:end], mat._indices[start:end]))

            # 按绝对值排序取 top-k
            row_data.sort(key=lambda x: abs(x[0]), reverse=True)
            row_data = row_data[:k]

            # 按索引排序
            row_data.sort(key=lambda x: x[1])

            for val, idx in row_data:
                new_data.append(val)
                new_indices.append(idx)

            new_indptr.append(len(new_data))

        return SclCSR.from_arrays(
            np.array(new_data, dtype=np.float32),
            np.array(new_indices, dtype=np.int32),
            np.array(new_indptr, dtype=np.int32),
            mat.shape
        )


# =============================================================================
# 矩阵运算
# =============================================================================

def spmv(mat: SclCSR, x: "np.ndarray") -> "np.ndarray":
    """
    稀疏矩阵-向量乘法: y = A * x

    Args:
        mat: 稀疏矩阵
        x: 向量

    Returns:
        结果向量
    """
    np = _ensure_numpy()

    mat.materialize()
    x = np.ascontiguousarray(x, dtype=np.float32)

    if len(x) != mat.shape[1]:
        raise ValueError(f"Vector length {len(x)} != matrix cols {mat.shape[1]}")

    y = np.empty(mat.shape[0], dtype=np.float32)

    if mat._handle is not None:
        lib = get_lib_with_signatures()
        check_error(lib.scl_mmap_csr_spmv(
            mat._handle,
            x.ctypes.data,
            y.ctypes.data
        ))
    else:
        for i in range(mat.shape[0]):
            start, end = mat._indptr[i], mat._indptr[i + 1]
            y[i] = np.dot(mat._data[start:end], x[mat._indices[start:end]])

    return y


def dot(a: SclCSR, b: SclCSR) -> SclCSR:
    """
    稀疏矩阵乘法: C = A * B

    Args:
        a: 左矩阵
        b: 右矩阵

    Returns:
        结果矩阵
    """
    # 使用 scipy 实现
    a.materialize()
    b.materialize()

    sp_a = a.to_scipy()
    sp_b = b.to_scipy()
    sp_c = sp_a.dot(sp_b)

    return SclCSR.from_scipy(sp_c)


# =============================================================================
# 统计
# =============================================================================

def var(mat: SclCSR, axis: Optional[int] = None,
        ddof: int = 0) -> Union[float, "np.ndarray"]:
    """
    计算方差

    Args:
        mat: 输入矩阵
        axis: None=全局, 0=列, 1=行
        ddof: 自由度修正
    """
    np = _ensure_numpy()

    mat.materialize()

    if axis == 1:
        # 行方差
        means = mat.mean(axis=1)
        result = np.empty(mat.shape[0], dtype=np.float32)

        if mat._handle is not None:
            lib = get_lib_with_signatures()
            check_error(lib.scl_mmap_csr_row_var(
                mat._handle,
                result.ctypes.data,
                means.ctypes.data,
                1  # count_zeros
            ))
        else:
            for i in range(mat.shape[0]):
                start, end = mat._indptr[i], mat._indptr[i + 1]
                vals = mat._data[start:end]
                mean = means[i]

                sum_sq = np.sum((vals - mean) ** 2)
                # 零值的贡献
                num_zeros = mat.shape[1] - (end - start)
                sum_sq += num_zeros * mean ** 2

                n = mat.shape[1] - ddof
                result[i] = sum_sq / n if n > 0 else 0

        return result
    elif axis == 0:
        # 列方差 - 使用 scipy
        return mat.to_scipy().toarray().var(axis=0, ddof=ddof)
    else:
        # 全局方差
        mean = mat.mean()
        total = np.sum((mat._data - mean) ** 2)
        num_zeros = mat.shape[0] * mat.shape[1] - mat.nnz
        total += num_zeros * mean ** 2
        n = mat.shape[0] * mat.shape[1] - ddof
        return total / n if n > 0 else 0


def std(mat: SclCSR, axis: Optional[int] = None,
        ddof: int = 0) -> Union[float, "np.ndarray"]:
    """计算标准差"""
    np = _ensure_numpy()
    return np.sqrt(var(mat, axis, ddof))


# =============================================================================
# 工具函数
# =============================================================================

def concatenate(matrices: List[SclCSR], axis: int = 0) -> SclCSR:
    """
    拼接矩阵

    Args:
        matrices: 矩阵列表
        axis: 0=垂直, 1=水平

    Returns:
        拼接后的矩阵 (延迟执行)
    """
    if axis == 0:
        return vstack(matrices)
    else:
        return hstack(matrices)


def copy(mat: SclCSR) -> SclCSR:
    """复制矩阵"""
    mat.materialize()

    np = _ensure_numpy()
    return SclCSR.from_arrays(
        mat._data.copy(),
        mat._indices.copy(),
        mat._indptr.copy(),
        mat.shape
    )


def issparse(x) -> bool:
    """检查是否为稀疏矩阵"""
    return isinstance(x, (SclCSR, SclCSC, LazyView, LazyReorder, PairSparse, LazyVStack))


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 归一化
    "normalize",
    # 变换
    "log1p",
    "scale",
    # 过滤
    "filter_threshold",
    "top_k",
    # 矩阵运算
    "spmv",
    "dot",
    # 统计
    "var",
    "std",
    # 工具
    "concatenate",
    "copy",
    "issparse",
    # 从 lazy 模块重导出
    "hstack",
    "vstack",
]
