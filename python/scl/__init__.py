"""
SCL - Sparse Computing Library

高性能稀疏矩阵计算库，支持超大规模数据集的延迟加载。

主要组件:
- SclCSR: 智能 CSR 稀疏矩阵
- SclCSC: 智能 CSC 稀疏矩阵
- LazyView: 延迟视图
- PairSparse: 虚拟水平堆叠
"""

from scl._ffi import get_lib, check_error
from scl.sparse import SclCSR, SclCSC
from scl.lazy import LazyView, LazyReorder, PairSparse
from scl import ops

__version__ = "0.1.0"
__all__ = [
    "SclCSR",
    "SclCSC",
    "LazyView",
    "LazyReorder",
    "PairSparse",
    "ops",
]
