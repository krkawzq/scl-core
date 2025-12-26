"""
SCL Sparse - 智能稀疏矩阵

提供自动管理后端的稀疏矩阵类，支持:
- 内存模式: 全量数据在 RAM
- 映射模式: 数据在磁盘，按需加载
- 延迟操作: 切片、重排等操作延迟执行
"""

from __future__ import annotations

import ctypes
from ctypes import c_int64, c_uint8, POINTER, byref
from typing import Optional, Union, Tuple, TYPE_CHECKING
import weakref

from scl._ffi import (
    get_lib_with_signatures, check_error, Backend,
    estimate_memory, suggest_backend
)

if TYPE_CHECKING:
    import numpy as np

# =============================================================================
# 类型别名
# =============================================================================

ArrayLike = Union["np.ndarray", list, tuple]
MaskLike = Union["np.ndarray", list, None]


# =============================================================================
# 辅助函数
# =============================================================================

def _ensure_numpy():
    """延迟导入 numpy"""
    import numpy as np
    return np


def _to_contiguous(arr, dtype):
    """转换为连续数组"""
    np = _ensure_numpy()
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=dtype)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def _to_mask(mask, length: int) -> Optional[ctypes.Array]:
    """转换为 uint8 掩码"""
    if mask is None:
        return None
    np = _ensure_numpy()
    mask = np.asarray(mask, dtype=np.uint8)
    if len(mask) != length:
        raise ValueError(f"Mask length {len(mask)} != expected {length}")
    if not mask.flags["C_CONTIGUOUS"]:
        mask = np.ascontiguousarray(mask)
    return mask.ctypes.data_as(POINTER(c_uint8))


# =============================================================================
# SclCSR - 智能 CSR 稀疏矩阵
# =============================================================================

class SclCSR:
    """
    智能 CSR 稀疏矩阵

    自动管理后端:
    - 小数据: 内存模式
    - 大数据: 映射模式 (按需加载)

    支持延迟操作:
    - 切片 (行/列选择)
    - 重排序
    - 虚拟堆叠

    示例:
        # 从 scipy 创建
        mat = SclCSR.from_scipy(sp_matrix)

        # 从文件创建 (映射模式)
        mat = SclCSR.from_file("data.sclm")

        # 切片 (延迟执行)
        subset = mat[mask]

        # 强制加载到内存
        mat.materialize()

        # 转换为 scipy
        sp = mat.to_scipy()
    """

    __slots__ = (
        "_handle",           # C++ 句柄
        "_rows", "_cols", "_nnz",  # 维度
        "_data", "_indices", "_indptr",  # 内存数据 (如果有)
        "_row_mask", "_col_mask",  # 延迟行列掩码
        "_row_order", "_col_order",  # 延迟重排序
        "_parent",           # 父对象引用 (用于视图)
        "_is_view",          # 是否为视图
        "_backend",          # 后端类型
        "__weakref__",
    )

    def __init__(self):
        self._handle: Optional[int] = None
        self._rows: int = 0
        self._cols: int = 0
        self._nnz: int = 0
        self._data = None
        self._indices = None
        self._indptr = None
        self._row_mask = None
        self._col_mask = None
        self._row_order = None
        self._col_order = None
        self._parent = None
        self._is_view = False
        self._backend = Backend.IN_MEMORY

    def __del__(self):
        self._release()

    def _release(self):
        """释放 C++ 资源"""
        if self._handle is not None:
            try:
                lib = get_lib_with_signatures()
                lib.scl_mmap_release(self._handle)
            except:
                pass
            self._handle = None

    # -------------------------------------------------------------------------
    # 工厂方法
    # -------------------------------------------------------------------------

    @classmethod
    def from_scipy(cls, sp_matrix, max_pages: int = 64) -> "SclCSR":
        """
        从 scipy.sparse 创建

        Args:
            sp_matrix: scipy.sparse.csr_matrix
            max_pages: 映射模式的最大驻留页数

        Returns:
            SclCSR 实例
        """
        np = _ensure_numpy()

        # 延迟导入 scipy
        from scipy import sparse
        if not sparse.isspmatrix_csr(sp_matrix):
            sp_matrix = sp_matrix.tocsr()

        obj = cls()
        obj._rows = sp_matrix.shape[0]
        obj._cols = sp_matrix.shape[1]
        obj._nnz = sp_matrix.nnz

        # 检查是否需要映射模式
        mem_bytes = estimate_memory(obj._rows, obj._nnz)
        backend = suggest_backend(mem_bytes)

        if backend == Backend.IN_MEMORY:
            # 直接存储数据
            obj._data = np.ascontiguousarray(sp_matrix.data, dtype=np.float32)
            obj._indices = np.ascontiguousarray(sp_matrix.indices, dtype=np.int32)
            obj._indptr = np.ascontiguousarray(sp_matrix.indptr, dtype=np.int32)
            obj._backend = Backend.IN_MEMORY
        else:
            # 创建映射句柄
            obj._data = np.ascontiguousarray(sp_matrix.data, dtype=np.float32)
            obj._indices = np.ascontiguousarray(sp_matrix.indices, dtype=np.int32)
            obj._indptr = np.ascontiguousarray(sp_matrix.indptr, dtype=np.int32)

            lib = get_lib_with_signatures()
            handle = c_int64()
            check_error(lib.scl_mmap_create_csr_from_ptr(
                obj._data.ctypes.data,
                obj._indices.ctypes.data,
                obj._indptr.ctypes.data,
                obj._rows, obj._cols, obj._nnz,
                max_pages,
                byref(handle)
            ))
            obj._handle = handle.value
            obj._backend = Backend.MAPPED

        return obj

    @classmethod
    def from_arrays(cls, data, indices, indptr, shape: Tuple[int, int],
                    max_pages: int = 64) -> "SclCSR":
        """
        从原始数组创建

        Args:
            data: 非零值数组
            indices: 列索引数组
            indptr: 行指针数组
            shape: (rows, cols)
            max_pages: 映射模式的最大驻留页数
        """
        np = _ensure_numpy()

        obj = cls()
        obj._rows, obj._cols = shape
        obj._data = _to_contiguous(data, np.float32)
        obj._indices = _to_contiguous(indices, np.int32)
        obj._indptr = _to_contiguous(indptr, np.int32)
        obj._nnz = len(obj._data)

        # 检查后端
        mem_bytes = estimate_memory(obj._rows, obj._nnz)
        backend = suggest_backend(mem_bytes)

        if backend != Backend.IN_MEMORY:
            lib = get_lib_with_signatures()
            handle = c_int64()
            check_error(lib.scl_mmap_create_csr_from_ptr(
                obj._data.ctypes.data,
                obj._indices.ctypes.data,
                obj._indptr.ctypes.data,
                obj._rows, obj._cols, obj._nnz,
                max_pages,
                byref(handle)
            ))
            obj._handle = handle.value
            obj._backend = Backend.MAPPED

        return obj

    @classmethod
    def from_file(cls, filepath: str, max_pages: int = 64) -> "SclCSR":
        """
        从文件创建 (映射模式)

        Args:
            filepath: SCL 二进制文件路径
            max_pages: 最大驻留页数
        """
        lib = get_lib_with_signatures()

        obj = cls()
        handle = c_int64()
        check_error(lib.scl_mmap_open_csr_file(
            filepath.encode("utf-8"),
            max_pages,
            byref(handle)
        ))
        obj._handle = handle.value
        obj._backend = Backend.MAPPED

        # 获取维度
        rows, cols, nnz = c_int64(), c_int64(), c_int64()
        check_error(lib.scl_mmap_csr_shape(
            obj._handle, byref(rows), byref(cols), byref(nnz)
        ))
        obj._rows = rows.value
        obj._cols = cols.value
        obj._nnz = nnz.value

        return obj

    # -------------------------------------------------------------------------
    # 属性
    # -------------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        """矩阵形状 (rows, cols)"""
        return (self._rows, self._cols)

    @property
    def nnz(self) -> int:
        """非零元素数"""
        return self._nnz

    @property
    def dtype(self):
        """数据类型"""
        np = _ensure_numpy()
        return np.float32

    @property
    def is_materialized(self) -> bool:
        """数据是否已加载到内存"""
        return self._data is not None and not self._has_pending_ops()

    @property
    def backend(self) -> int:
        """当前后端类型"""
        return self._backend

    def _has_pending_ops(self) -> bool:
        """是否有待执行的延迟操作"""
        return (self._row_mask is not None or
                self._col_mask is not None or
                self._row_order is not None or
                self._col_order is not None)

    # -------------------------------------------------------------------------
    # 切片操作 (延迟执行)
    # -------------------------------------------------------------------------

    def __getitem__(self, key) -> "SclCSR":
        """
        切片操作 (延迟执行)

        支持:
        - mat[row_mask]: 行选择
        - mat[row_mask, col_mask]: 行列选择
        - mat[:, col_mask]: 列选择
        """
        np = _ensure_numpy()

        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key = key
            col_key = None

        # 创建视图
        view = SclCSR()
        view._parent = weakref.ref(self)
        view._is_view = True
        view._handle = self._handle
        view._backend = self._backend

        # 处理行选择
        if row_key is not None and not (isinstance(row_key, slice) and row_key == slice(None)):
            if isinstance(row_key, slice):
                # 转换 slice 为掩码
                mask = np.zeros(self._rows, dtype=np.uint8)
                mask[row_key] = 1
                view._row_mask = mask
            else:
                # 假设是布尔/索引数组
                row_key = np.asarray(row_key)
                if row_key.dtype == bool:
                    view._row_mask = row_key.astype(np.uint8)
                else:
                    # 索引数组转掩码
                    mask = np.zeros(self._rows, dtype=np.uint8)
                    mask[row_key] = 1
                    view._row_mask = mask

        # 处理列选择
        if col_key is not None and not (isinstance(col_key, slice) and col_key == slice(None)):
            if isinstance(col_key, slice):
                mask = np.zeros(self._cols, dtype=np.uint8)
                mask[col_key] = 1
                view._col_mask = mask
            else:
                col_key = np.asarray(col_key)
                if col_key.dtype == bool:
                    view._col_mask = col_key.astype(np.uint8)
                else:
                    mask = np.zeros(self._cols, dtype=np.uint8)
                    mask[col_key] = 1
                    view._col_mask = mask

        # 合并父视图的掩码
        if self._row_mask is not None:
            if view._row_mask is not None:
                # 子视图的掩码基于父视图
                parent_indices = np.where(self._row_mask)[0]
                child_indices = np.where(view._row_mask)[0]
                new_mask = np.zeros(self._rows, dtype=np.uint8)
                new_mask[parent_indices[child_indices]] = 1
                view._row_mask = new_mask
            else:
                view._row_mask = self._row_mask.copy()

        if self._col_mask is not None:
            if view._col_mask is not None:
                parent_indices = np.where(self._col_mask)[0]
                child_indices = np.where(view._col_mask)[0]
                new_mask = np.zeros(self._cols, dtype=np.uint8)
                new_mask[parent_indices[child_indices]] = 1
                view._col_mask = new_mask
            else:
                view._col_mask = self._col_mask.copy()

        # 计算新维度
        view._rows = int(np.sum(view._row_mask)) if view._row_mask is not None else self._rows
        view._cols = int(np.sum(view._col_mask)) if view._col_mask is not None else self._cols
        # nnz 延迟计算

        return view

    # -------------------------------------------------------------------------
    # 重排序 (延迟执行)
    # -------------------------------------------------------------------------

    def reorder_rows(self, order: ArrayLike) -> "SclCSR":
        """
        按顺序重排行 (延迟执行)

        Args:
            order: 新顺序的索引数组
        """
        np = _ensure_numpy()

        view = SclCSR()
        view._parent = weakref.ref(self)
        view._is_view = True
        view._handle = self._handle
        view._backend = self._backend
        view._row_order = np.asarray(order, dtype=np.int32)
        view._col_mask = self._col_mask
        view._rows = len(view._row_order)
        view._cols = self._cols

        return view

    def reorder_cols(self, order: ArrayLike) -> "SclCSR":
        """
        按顺序重排列 (延迟执行)

        Args:
            order: 新顺序的索引数组
        """
        np = _ensure_numpy()

        view = SclCSR()
        view._parent = weakref.ref(self)
        view._is_view = True
        view._handle = self._handle
        view._backend = self._backend
        view._row_mask = self._row_mask
        view._col_order = np.asarray(order, dtype=np.int32)
        view._rows = self._rows
        view._cols = len(view._col_order)

        return view

    # -------------------------------------------------------------------------
    # 物化 (执行所有延迟操作)
    # -------------------------------------------------------------------------

    def materialize(self) -> "SclCSR":
        """
        物化: 执行所有延迟操作，加载到内存

        Returns:
            self (支持链式调用)
        """
        if self.is_materialized:
            return self

        np = _ensure_numpy()
        lib = get_lib_with_signatures()

        # 确定源数据
        if self._handle is not None:
            # 从映射句柄加载
            if self._has_pending_ops():
                # 有延迟操作，使用掩码加载
                if self._row_order is not None or self._col_order is not None:
                    # 重排序操作
                    self._materialize_reorder()
                else:
                    # 掩码操作
                    self._materialize_masked()
            else:
                # 全量加载
                self._data = np.empty(self._nnz, dtype=np.float32)
                self._indices = np.empty(self._nnz, dtype=np.int32)
                self._indptr = np.empty(self._rows + 1, dtype=np.int32)

                check_error(lib.scl_mmap_csr_load_full(
                    self._handle,
                    self._data.ctypes.data,
                    self._indices.ctypes.data,
                    self._indptr.ctypes.data
                ))
        elif self._parent is not None:
            # 从父视图物化
            parent = self._parent()
            if parent is None:
                raise RuntimeError("Parent object has been garbage collected")
            parent.materialize()
            self._materialize_from_parent(parent)

        # 清理延迟操作状态
        self._row_mask = None
        self._col_mask = None
        self._row_order = None
        self._col_order = None
        self._parent = None
        self._is_view = False
        self._backend = Backend.IN_MEMORY

        return self

    def _materialize_masked(self):
        """执行掩码加载"""
        np = _ensure_numpy()
        lib = get_lib_with_signatures()

        # 计算 nnz
        out_nnz = c_int64()
        row_mask_ptr = _to_mask(self._row_mask, self._rows) if self._row_mask is not None else None
        col_mask_ptr = _to_mask(self._col_mask, self._cols) if self._col_mask is not None else None

        check_error(lib.scl_mmap_csr_compute_masked_nnz(
            self._handle,
            row_mask_ptr,
            col_mask_ptr,
            byref(out_nnz)
        ))

        # 分配内存
        new_rows = int(np.sum(self._row_mask)) if self._row_mask is not None else self._rows
        nnz = out_nnz.value

        self._data = np.empty(nnz, dtype=np.float32)
        self._indices = np.empty(nnz, dtype=np.int32)
        self._indptr = np.empty(new_rows + 1, dtype=np.int32)

        out_rows, out_cols, out_nnz = c_int64(), c_int64(), c_int64()
        check_error(lib.scl_mmap_csr_load_masked(
            self._handle,
            row_mask_ptr,
            col_mask_ptr,
            self._data.ctypes.data,
            self._indices.ctypes.data,
            self._indptr.ctypes.data,
            byref(out_rows), byref(out_cols), byref(out_nnz)
        ))

        self._rows = out_rows.value
        self._cols = out_cols.value
        self._nnz = out_nnz.value

    def _materialize_reorder(self):
        """执行重排序加载"""
        np = _ensure_numpy()
        lib = get_lib_with_signatures()

        if self._row_order is not None:
            # 行重排序
            new_rows = len(self._row_order)
            # 估算 nnz (上界)
            self._data = np.empty(self._nnz, dtype=np.float32)
            self._indices = np.empty(self._nnz, dtype=np.int32)
            self._indptr = np.empty(new_rows + 1, dtype=np.int32)

            check_error(lib.scl_mmap_csr_reorder_rows(
                self._handle,
                self._row_order.ctypes.data,
                new_rows,
                self._data.ctypes.data,
                self._indices.ctypes.data,
                self._indptr.ctypes.data
            ))

            self._nnz = self._indptr[new_rows]
            self._data = self._data[:self._nnz]
            self._indices = self._indices[:self._nnz]
            self._rows = new_rows

        if self._col_order is not None:
            # 列重排序
            new_cols = len(self._col_order)
            new_data = np.empty(self._nnz, dtype=np.float32)
            new_indices = np.empty(self._nnz, dtype=np.int32)
            new_indptr = np.empty(self._rows + 1, dtype=np.int32)

            check_error(lib.scl_mmap_csr_reorder_cols(
                self._handle,
                self._col_order.ctypes.data,
                new_cols,
                new_data.ctypes.data,
                new_indices.ctypes.data,
                new_indptr.ctypes.data
            ))

            self._data = new_data
            self._indices = new_indices
            self._indptr = new_indptr
            self._cols = new_cols

    def _materialize_from_parent(self, parent: "SclCSR"):
        """从父视图物化"""
        np = _ensure_numpy()

        if self._row_mask is None and self._col_mask is None:
            # 无掩码，直接复制
            self._data = parent._data.copy()
            self._indices = parent._indices.copy()
            self._indptr = parent._indptr.copy()
            self._nnz = parent._nnz
        else:
            # 有掩码，需要过滤
            # 这里使用纯 Python 实现
            row_indices = np.where(self._row_mask)[0] if self._row_mask is not None else np.arange(parent._rows)
            col_set = set(np.where(self._col_mask)[0]) if self._col_mask is not None else None

            new_data = []
            new_indices = []
            new_indptr = [0]

            if self._col_mask is not None:
                col_remap = {old: new for new, old in enumerate(np.where(self._col_mask)[0])}

            for i in row_indices:
                start, end = parent._indptr[i], parent._indptr[i + 1]
                for k in range(start, end):
                    col = parent._indices[k]
                    if col_set is None or col in col_set:
                        new_data.append(parent._data[k])
                        if self._col_mask is not None:
                            new_indices.append(col_remap[col])
                        else:
                            new_indices.append(col)
                new_indptr.append(len(new_data))

            self._data = np.array(new_data, dtype=np.float32)
            self._indices = np.array(new_indices, dtype=np.int32)
            self._indptr = np.array(new_indptr, dtype=np.int32)
            self._nnz = len(self._data)

    # -------------------------------------------------------------------------
    # 转换方法
    # -------------------------------------------------------------------------

    def to_scipy(self):
        """转换为 scipy.sparse.csr_matrix"""
        self.materialize()

        # 延迟导入
        from scipy import sparse
        return sparse.csr_matrix(
            (self._data, self._indices, self._indptr),
            shape=self.shape
        )

    def to_dense(self):
        """转换为稠密 numpy 数组"""
        self.materialize()

        np = _ensure_numpy()
        dense = np.zeros(self.shape, dtype=np.float32)

        for i in range(self._rows):
            start, end = self._indptr[i], self._indptr[i + 1]
            for k in range(start, end):
                dense[i, self._indices[k]] = self._data[k]

        return dense

    def to_csc(self) -> "SclCSC":
        """转换为 CSC 格式"""
        self.materialize()

        np = _ensure_numpy()
        lib = get_lib_with_signatures()

        csc_data = np.empty(self._nnz, dtype=np.float32)
        csc_indices = np.empty(self._nnz, dtype=np.int32)
        csc_indptr = np.empty(self._cols + 1, dtype=np.int32)

        if self._handle is not None:
            check_error(lib.scl_mmap_csr_to_csc(
                self._handle,
                csc_data.ctypes.data,
                csc_indices.ctypes.data,
                csc_indptr.ctypes.data
            ))
        else:
            # 使用 scipy 转换
            sp = self.to_scipy()
            csc = sp.tocsc()
            csc_data = csc.data.astype(np.float32)
            csc_indices = csc.indices.astype(np.int32)
            csc_indptr = csc.indptr.astype(np.int32)

        return SclCSC.from_arrays(
            csc_data, csc_indices, csc_indptr, self.shape
        )

    # -------------------------------------------------------------------------
    # 统计方法
    # -------------------------------------------------------------------------

    def sum(self, axis: Optional[int] = None):
        """
        求和

        Args:
            axis: None=全局, 0=列, 1=行
        """
        self.materialize()
        np = _ensure_numpy()
        lib = get_lib_with_signatures()

        if axis is None:
            result = np.empty(1, dtype=np.float32)
            if self._handle is not None:
                check_error(lib.scl_mmap_csr_global_sum(
                    self._handle, result.ctypes.data
                ))
            else:
                result[0] = np.sum(self._data)
            return result[0]
        elif axis == 1:
            result = np.empty(self._rows, dtype=np.float32)
            if self._handle is not None:
                check_error(lib.scl_mmap_csr_row_sum(
                    self._handle, result.ctypes.data
                ))
            else:
                for i in range(self._rows):
                    start, end = self._indptr[i], self._indptr[i + 1]
                    result[i] = np.sum(self._data[start:end])
            return result
        elif axis == 0:
            result = np.zeros(self._cols, dtype=np.float32)
            if self._handle is not None:
                check_error(lib.scl_mmap_csr_col_sum(
                    self._handle, result.ctypes.data
                ))
            else:
                for i in range(self._rows):
                    start, end = self._indptr[i], self._indptr[i + 1]
                    for k in range(start, end):
                        result[self._indices[k]] += self._data[k]
            return result

    def mean(self, axis: Optional[int] = None):
        """求均值"""
        sums = self.sum(axis)
        if axis is None:
            return sums / (self._rows * self._cols)
        elif axis == 1:
            return sums / self._cols
        else:
            return sums / self._rows

    # -------------------------------------------------------------------------
    # 表示方法
    # -------------------------------------------------------------------------

    def __repr__(self):
        status = "materialized" if self.is_materialized else "lazy"
        backend = ["in_memory", "mapped", "streaming"][self._backend]
        return f"SclCSR(shape={self.shape}, nnz={self._nnz}, {status}, {backend})"


# =============================================================================
# SclCSC - 智能 CSC 稀疏矩阵
# =============================================================================

class SclCSC:
    """
    智能 CSC 稀疏矩阵

    API 与 SclCSR 类似，但针对列操作优化。
    """

    __slots__ = (
        "_rows", "_cols", "_nnz",
        "_data", "_indices", "_indptr",
        "_backend",
    )

    def __init__(self):
        self._rows = 0
        self._cols = 0
        self._nnz = 0
        self._data = None
        self._indices = None
        self._indptr = None
        self._backend = Backend.IN_MEMORY

    @classmethod
    def from_scipy(cls, sp_matrix) -> "SclCSC":
        """从 scipy.sparse 创建"""
        np = _ensure_numpy()
        from scipy import sparse

        if not sparse.isspmatrix_csc(sp_matrix):
            sp_matrix = sp_matrix.tocsc()

        obj = cls()
        obj._rows, obj._cols = sp_matrix.shape
        obj._nnz = sp_matrix.nnz
        obj._data = np.ascontiguousarray(sp_matrix.data, dtype=np.float32)
        obj._indices = np.ascontiguousarray(sp_matrix.indices, dtype=np.int32)
        obj._indptr = np.ascontiguousarray(sp_matrix.indptr, dtype=np.int32)

        return obj

    @classmethod
    def from_arrays(cls, data, indices, indptr, shape: Tuple[int, int]) -> "SclCSC":
        """从原始数组创建"""
        np = _ensure_numpy()

        obj = cls()
        obj._rows, obj._cols = shape
        obj._data = _to_contiguous(data, np.float32)
        obj._indices = _to_contiguous(indices, np.int32)
        obj._indptr = _to_contiguous(indptr, np.int32)
        obj._nnz = len(obj._data)

        return obj

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._rows, self._cols)

    @property
    def nnz(self) -> int:
        return self._nnz

    def to_scipy(self):
        """转换为 scipy.sparse.csc_matrix"""
        from scipy import sparse
        return sparse.csc_matrix(
            (self._data, self._indices, self._indptr),
            shape=self.shape
        )

    def to_csr(self) -> SclCSR:
        """转换为 CSR 格式"""
        sp = self.to_scipy()
        return SclCSR.from_scipy(sp.tocsr())

    def __repr__(self):
        return f"SclCSC(shape={self.shape}, nnz={self._nnz})"
