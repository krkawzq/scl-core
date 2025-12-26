"""
SCL Lazy - 延迟操作组合器

提供零成本的延迟操作:
- LazyView: 延迟切片视图
- LazyReorder: 延迟重排序
- PairSparse: 虚拟水平堆叠
- LazyVStack: 虚拟垂直堆叠
"""

from __future__ import annotations

from typing import Optional, Union, Tuple, List, TYPE_CHECKING
import weakref

if TYPE_CHECKING:
    import numpy as np
    from scl.sparse import SclCSR, SclCSC


def _ensure_numpy():
    import numpy as np
    return np


# =============================================================================
# LazyView - 延迟切片视图
# =============================================================================

class LazyView:
    """
    延迟切片视图

    不复制数据，只存储选择信息。
    支持链式切片。

    示例:
        view = LazyView(mat, row_mask, col_mask)
        sub_view = view[sub_mask]  # 链式
        result = view.materialize()  # 执行
    """

    __slots__ = (
        "_base",          # 底层数据
        "_row_mask",      # 行掩码
        "_col_mask",      # 列掩码
        "_shape",         # 结果形状
    )

    def __init__(self, base: Union["SclCSR", "LazyView"],
                 row_mask: Optional["np.ndarray"] = None,
                 col_mask: Optional["np.ndarray"] = None):
        """
        创建延迟视图

        Args:
            base: 底层矩阵或视图
            row_mask: 行选择掩码 (uint8)
            col_mask: 列选择掩码 (uint8)
        """
        np = _ensure_numpy()

        self._base = base
        self._row_mask = row_mask
        self._col_mask = col_mask

        # 计算形状
        base_shape = base.shape if hasattr(base, 'shape') else base._shape
        rows = int(np.sum(row_mask)) if row_mask is not None else base_shape[0]
        cols = int(np.sum(col_mask)) if col_mask is not None else base_shape[1]
        self._shape = (rows, cols)

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    def __getitem__(self, key) -> "LazyView":
        """链式切片"""
        np = _ensure_numpy()

        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key = key
            col_key = None

        # 构建新掩码
        new_row_mask = self._apply_key(row_key, self._row_mask, self._shape[0])
        new_col_mask = self._apply_key(col_key, self._col_mask, self._shape[1])

        return LazyView(self._base, new_row_mask, new_col_mask)

    def _apply_key(self, key, existing_mask, length):
        """应用索引到现有掩码"""
        np = _ensure_numpy()

        if key is None or (isinstance(key, slice) and key == slice(None)):
            return existing_mask

        # 获取当前有效索引
        if existing_mask is not None:
            valid_indices = np.where(existing_mask)[0]
        else:
            valid_indices = np.arange(length)

        # 应用新选择
        if isinstance(key, slice):
            selected = valid_indices[key]
        else:
            key = np.asarray(key)
            if key.dtype == bool:
                selected = valid_indices[key]
            else:
                selected = valid_indices[key]

        # 构建新掩码
        new_mask = np.zeros(length, dtype=np.uint8)
        new_mask[selected] = 1
        return new_mask

    def materialize(self) -> "SclCSR":
        """执行切片，返回物化的矩阵"""
        from scl.sparse import SclCSR

        # 获取根矩阵
        base = self._base
        while isinstance(base, LazyView):
            base = base._base

        # 组合所有掩码
        row_mask = self._combine_masks_up()
        col_mask = self._combine_col_masks_up()

        # 创建视图并物化
        if isinstance(base, SclCSR):
            result = base[row_mask, col_mask] if row_mask is not None or col_mask is not None else base
            return result.materialize()

        raise TypeError(f"Unsupported base type: {type(base)}")

    def _combine_masks_up(self):
        """向上合并行掩码"""
        np = _ensure_numpy()

        if not isinstance(self._base, LazyView):
            return self._row_mask

        parent_mask = self._base._combine_masks_up()
        if self._row_mask is None:
            return parent_mask
        if parent_mask is None:
            return self._row_mask

        # 合并
        parent_indices = np.where(parent_mask)[0]
        child_selected = np.where(self._row_mask)[0]

        new_mask = np.zeros(len(parent_mask), dtype=np.uint8)
        new_mask[parent_indices[child_selected]] = 1
        return new_mask

    def _combine_col_masks_up(self):
        """向上合并列掩码"""
        np = _ensure_numpy()

        if not isinstance(self._base, LazyView):
            return self._col_mask

        parent_mask = self._base._combine_col_masks_up()
        if self._col_mask is None:
            return parent_mask
        if parent_mask is None:
            return self._col_mask

        parent_indices = np.where(parent_mask)[0]
        child_selected = np.where(self._col_mask)[0]

        new_mask = np.zeros(len(parent_mask), dtype=np.uint8)
        new_mask[parent_indices[child_selected]] = 1
        return new_mask

    def __repr__(self):
        return f"LazyView(shape={self._shape})"


# =============================================================================
# LazyReorder - 延迟重排序
# =============================================================================

class LazyReorder:
    """
    延迟重排序

    只存储重排序向量，不复制数据。

    示例:
        reordered = LazyReorder(mat, order)
        result = reordered.materialize()
    """

    __slots__ = (
        "_base",
        "_row_order",
        "_col_order",
        "_shape",
    )

    def __init__(self, base: Union["SclCSR", "LazyView", "LazyReorder"],
                 row_order: Optional["np.ndarray"] = None,
                 col_order: Optional["np.ndarray"] = None):
        """
        创建延迟重排序

        Args:
            base: 底层矩阵
            row_order: 行重排序向量
            col_order: 列重排序向量
        """
        np = _ensure_numpy()

        self._base = base
        self._row_order = np.asarray(row_order, dtype=np.int32) if row_order is not None else None
        self._col_order = np.asarray(col_order, dtype=np.int32) if col_order is not None else None

        base_shape = base.shape if hasattr(base, 'shape') else base._shape
        rows = len(self._row_order) if self._row_order is not None else base_shape[0]
        cols = len(self._col_order) if self._col_order is not None else base_shape[1]
        self._shape = (rows, cols)

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    def reorder_rows(self, order: "np.ndarray") -> "LazyReorder":
        """链式行重排序"""
        np = _ensure_numpy()

        if self._row_order is not None:
            # 组合重排序
            new_order = self._row_order[order]
        else:
            new_order = np.asarray(order, dtype=np.int32)

        return LazyReorder(self._base, new_order, self._col_order)

    def reorder_cols(self, order: "np.ndarray") -> "LazyReorder":
        """链式列重排序"""
        np = _ensure_numpy()

        if self._col_order is not None:
            new_order = self._col_order[order]
        else:
            new_order = np.asarray(order, dtype=np.int32)

        return LazyReorder(self._base, self._row_order, new_order)

    def __getitem__(self, key) -> "LazyView":
        """切片后返回 LazyView"""
        return LazyView(self, *self._parse_key(key))

    def _parse_key(self, key):
        """解析索引"""
        np = _ensure_numpy()

        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key = key
            col_key = None

        row_mask = None
        col_mask = None

        if row_key is not None and not (isinstance(row_key, slice) and row_key == slice(None)):
            if isinstance(row_key, slice):
                mask = np.zeros(self._shape[0], dtype=np.uint8)
                mask[row_key] = 1
                row_mask = mask
            else:
                row_key = np.asarray(row_key)
                if row_key.dtype == bool:
                    row_mask = row_key.astype(np.uint8)
                else:
                    mask = np.zeros(self._shape[0], dtype=np.uint8)
                    mask[row_key] = 1
                    row_mask = mask

        if col_key is not None and not (isinstance(col_key, slice) and col_key == slice(None)):
            if isinstance(col_key, slice):
                mask = np.zeros(self._shape[1], dtype=np.uint8)
                mask[col_key] = 1
                col_mask = mask
            else:
                col_key = np.asarray(col_key)
                if col_key.dtype == bool:
                    col_mask = col_key.astype(np.uint8)
                else:
                    mask = np.zeros(self._shape[1], dtype=np.uint8)
                    mask[col_key] = 1
                    col_mask = mask

        return row_mask, col_mask

    def materialize(self) -> "SclCSR":
        """执行重排序"""
        from scl.sparse import SclCSR

        # 首先物化基础矩阵
        if isinstance(self._base, (LazyView, LazyReorder)):
            base_mat = self._base.materialize()
        else:
            base_mat = self._base
            if not base_mat.is_materialized:
                base_mat.materialize()

        # 应用重排序
        if self._row_order is not None:
            base_mat = base_mat.reorder_rows(self._row_order)

        if self._col_order is not None:
            base_mat = base_mat.reorder_cols(self._col_order)

        return base_mat.materialize()

    def __repr__(self):
        return f"LazyReorder(shape={self._shape})"


# =============================================================================
# PairSparse - 虚拟水平堆叠
# =============================================================================

class PairSparse:
    """
    虚拟水平堆叠 (hstack)

    不复制数据，只在访问时合并。
    适用于需要合并多个数据源但不想立即物化的场景。

    示例:
        pair = PairSparse(left_mat, right_mat)
        print(pair.shape)  # (rows, left_cols + right_cols)
        result = pair.materialize()  # 真正合并
    """

    __slots__ = (
        "_left",
        "_right",
        "_shape",
    )

    def __init__(self, left: Union["SclCSR", "PairSparse"],
                 right: Union["SclCSR", "PairSparse"]):
        """
        创建虚拟 hstack

        Args:
            left: 左矩阵
            right: 右矩阵 (行数必须相同)
        """
        left_shape = left.shape
        right_shape = right.shape

        if left_shape[0] != right_shape[0]:
            raise ValueError(
                f"Row count mismatch: {left_shape[0]} vs {right_shape[0]}"
            )

        self._left = left
        self._right = right
        self._shape = (left_shape[0], left_shape[1] + right_shape[1])

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    def __getitem__(self, key) -> Union["PairSparse", "LazyView"]:
        """切片"""
        np = _ensure_numpy()

        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key = key
            col_key = None

        # 列切片需要分发到左右矩阵
        if col_key is not None:
            left_cols = self._left.shape[1]

            if isinstance(col_key, slice):
                # 处理 slice
                start, stop, step = col_key.indices(self._shape[1])
                left_slice = slice(
                    max(0, start),
                    min(left_cols, stop),
                    step
                )
                right_slice = slice(
                    max(0, start - left_cols),
                    max(0, stop - left_cols),
                    step
                )

                left_view = self._left[row_key, left_slice] if left_slice.stop > left_slice.start else None
                right_view = self._right[row_key, right_slice] if right_slice.stop > right_slice.start else None

                if left_view is None:
                    return right_view
                if right_view is None:
                    return left_view
                return PairSparse(left_view, right_view)
            else:
                # 掩码或索引数组
                col_key = np.asarray(col_key)
                if col_key.dtype == bool:
                    col_indices = np.where(col_key)[0]
                else:
                    col_indices = col_key

                left_mask = col_indices < left_cols
                right_mask = col_indices >= left_cols

                left_indices = col_indices[left_mask]
                right_indices = col_indices[right_mask] - left_cols

                # 创建掩码
                left_col_mask = np.zeros(left_cols, dtype=np.uint8)
                left_col_mask[left_indices] = 1

                right_col_mask = np.zeros(self._right.shape[1], dtype=np.uint8)
                right_col_mask[right_indices] = 1

                left_view = self._left[row_key, left_col_mask] if np.any(left_mask) else None
                right_view = self._right[row_key, right_col_mask] if np.any(right_mask) else None

                if left_view is None:
                    return right_view
                if right_view is None:
                    return left_view
                return PairSparse(left_view, right_view)
        else:
            # 只有行切片
            return PairSparse(
                self._left[row_key] if row_key is not None else self._left,
                self._right[row_key] if row_key is not None else self._right
            )

    def materialize(self) -> "SclCSR":
        """执行 hstack"""
        from scl.sparse import SclCSR
        np = _ensure_numpy()

        # 物化左右矩阵
        left = self._left
        if isinstance(left, (PairSparse, LazyView, LazyReorder)):
            left = left.materialize()
        elif not left.is_materialized:
            left.materialize()

        right = self._right
        if isinstance(right, (PairSparse, LazyView, LazyReorder)):
            right = right.materialize()
        elif not right.is_materialized:
            right.materialize()

        # 合并数据
        left_cols = left.shape[1]
        new_nnz = left.nnz + right.nnz

        new_data = np.empty(new_nnz, dtype=np.float32)
        new_indices = np.empty(new_nnz, dtype=np.int32)
        new_indptr = np.empty(self._shape[0] + 1, dtype=np.int32)

        offset = 0
        new_indptr[0] = 0

        for i in range(self._shape[0]):
            # 左矩阵数据
            l_start, l_end = left._indptr[i], left._indptr[i + 1]
            l_len = l_end - l_start
            if l_len > 0:
                new_data[offset:offset + l_len] = left._data[l_start:l_end]
                new_indices[offset:offset + l_len] = left._indices[l_start:l_end]
                offset += l_len

            # 右矩阵数据 (列索引需要偏移)
            r_start, r_end = right._indptr[i], right._indptr[i + 1]
            r_len = r_end - r_start
            if r_len > 0:
                new_data[offset:offset + r_len] = right._data[r_start:r_end]
                new_indices[offset:offset + r_len] = right._indices[r_start:r_end] + left_cols
                offset += r_len

            new_indptr[i + 1] = offset

        return SclCSR.from_arrays(new_data, new_indices, new_indptr, self._shape)

    def __repr__(self):
        return f"PairSparse(shape={self._shape})"


# =============================================================================
# LazyVStack - 虚拟垂直堆叠
# =============================================================================

class LazyVStack:
    """
    虚拟垂直堆叠 (vstack)

    示例:
        stack = LazyVStack([mat1, mat2, mat3])
        result = stack.materialize()
    """

    __slots__ = (
        "_matrices",
        "_shape",
        "_row_offsets",
    )

    def __init__(self, matrices: List[Union["SclCSR", "LazyView", "LazyReorder"]]):
        """
        创建虚拟 vstack

        Args:
            matrices: 矩阵列表 (列数必须相同)
        """
        if not matrices:
            raise ValueError("At least one matrix required")

        cols = matrices[0].shape[1]
        for mat in matrices[1:]:
            if mat.shape[1] != cols:
                raise ValueError(f"Column count mismatch: {mat.shape[1]} vs {cols}")

        self._matrices = matrices

        # 计算行偏移
        total_rows = 0
        self._row_offsets = [0]
        for mat in matrices:
            total_rows += mat.shape[0]
            self._row_offsets.append(total_rows)

        self._shape = (total_rows, cols)

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    def __getitem__(self, key) -> Union["LazyVStack", "LazyView"]:
        """切片"""
        np = _ensure_numpy()

        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key = key
            col_key = None

        if row_key is None or (isinstance(row_key, slice) and row_key == slice(None)):
            # 只有列切片
            if col_key is not None:
                return LazyVStack([mat[:, col_key] for mat in self._matrices])
            return self

        # 行切片需要分发到各矩阵
        if isinstance(row_key, slice):
            start, stop, step = row_key.indices(self._shape[0])
            row_indices = np.arange(start, stop, step)
        else:
            row_key = np.asarray(row_key)
            if row_key.dtype == bool:
                row_indices = np.where(row_key)[0]
            else:
                row_indices = row_key

        # 找出每个矩阵的行选择
        selected_mats = []
        for i, mat in enumerate(self._matrices):
            start_row = self._row_offsets[i]
            end_row = self._row_offsets[i + 1]

            mask = (row_indices >= start_row) & (row_indices < end_row)
            if np.any(mask):
                local_indices = row_indices[mask] - start_row
                local_mask = np.zeros(mat.shape[0], dtype=np.uint8)
                local_mask[local_indices] = 1

                if col_key is not None:
                    selected_mats.append(mat[local_mask, col_key])
                else:
                    selected_mats.append(mat[local_mask])

        if len(selected_mats) == 1:
            return selected_mats[0]
        return LazyVStack(selected_mats)

    def materialize(self) -> "SclCSR":
        """执行 vstack"""
        from scl.sparse import SclCSR
        np = _ensure_numpy()

        # 物化所有矩阵
        materialized = []
        total_nnz = 0
        for mat in self._matrices:
            if isinstance(mat, (LazyView, LazyReorder, PairSparse, LazyVStack)):
                mat = mat.materialize()
            elif not mat.is_materialized:
                mat.materialize()
            materialized.append(mat)
            total_nnz += mat.nnz

        # 合并
        new_data = np.empty(total_nnz, dtype=np.float32)
        new_indices = np.empty(total_nnz, dtype=np.int32)
        new_indptr = np.empty(self._shape[0] + 1, dtype=np.int32)

        offset = 0
        row_offset = 0
        new_indptr[0] = 0

        for mat in materialized:
            for i in range(mat.shape[0]):
                start, end = mat._indptr[i], mat._indptr[i + 1]
                length = end - start

                if length > 0:
                    new_data[offset:offset + length] = mat._data[start:end]
                    new_indices[offset:offset + length] = mat._indices[start:end]
                    offset += length

                new_indptr[row_offset + i + 1] = offset

            row_offset += mat.shape[0]

        return SclCSR.from_arrays(new_data, new_indices, new_indptr, self._shape)

    def __repr__(self):
        return f"LazyVStack(shape={self._shape}, n_matrices={len(self._matrices)})"


# =============================================================================
# 便捷函数
# =============================================================================

def hstack(matrices: List[Union["SclCSR", "PairSparse"]]) -> Union["SclCSR", "PairSparse"]:
    """
    水平堆叠多个矩阵 (延迟执行)

    Args:
        matrices: 矩阵列表

    Returns:
        PairSparse 或 SclCSR
    """
    if len(matrices) == 0:
        raise ValueError("At least one matrix required")
    if len(matrices) == 1:
        return matrices[0]

    result = matrices[0]
    for mat in matrices[1:]:
        result = PairSparse(result, mat)
    return result


def vstack(matrices: List[Union["SclCSR", "LazyView"]]) -> Union["SclCSR", "LazyVStack"]:
    """
    垂直堆叠多个矩阵 (延迟执行)

    Args:
        matrices: 矩阵列表

    Returns:
        LazyVStack 或 SclCSR
    """
    if len(matrices) == 0:
        raise ValueError("At least one matrix required")
    if len(matrices) == 1:
        return matrices[0]

    return LazyVStack(matrices)
