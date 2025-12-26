"""
SCL FFI - ctypes 绑定层

提供与 C++ 库的底层交互接口。
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (
    c_int32, c_int64, c_uint8, c_double, c_void_p, c_char_p,
    POINTER, byref, Structure
)
from functools import lru_cache
from typing import Optional

# =============================================================================
# 错误码
# =============================================================================

OK = 0
ERR_INVALID_HANDLE = -1
ERR_INVALID_PARAM = -2
ERR_OUT_OF_MEMORY = -3
ERR_IO = -4
ERR_TYPE_MISMATCH = -5
ERR_UNKNOWN = -99

_ERROR_MESSAGES = {
    OK: "Success",
    ERR_INVALID_HANDLE: "Invalid handle",
    ERR_INVALID_PARAM: "Invalid parameter",
    ERR_OUT_OF_MEMORY: "Out of memory",
    ERR_IO: "I/O error",
    ERR_TYPE_MISMATCH: "Type mismatch",
    ERR_UNKNOWN: "Unknown error",
}


class SclError(Exception):
    """SCL 库错误"""
    def __init__(self, code: int, message: str = ""):
        self.code = code
        self.message = message or _ERROR_MESSAGES.get(code, f"Error code {code}")
        super().__init__(self.message)


def check_error(code: int) -> None:
    """检查错误码并抛出异常"""
    if code != OK:
        raise SclError(code)


# =============================================================================
# 库加载
# =============================================================================

@lru_cache(maxsize=1)
def get_lib():
    """获取 SCL 共享库"""
    # 查找库路径
    lib_names = []

    if sys.platform == "linux":
        lib_names = ["libscl.so", "libscl_mmap.so"]
    elif sys.platform == "darwin":
        lib_names = ["libscl.dylib", "libscl_mmap.dylib"]
    elif sys.platform == "win32":
        lib_names = ["scl.dll", "scl_mmap.dll"]

    # 搜索路径
    search_paths = [
        os.path.dirname(__file__),
        os.path.join(os.path.dirname(__file__), ".."),
        os.path.join(os.path.dirname(__file__), "..", "lib"),
        os.path.join(os.path.dirname(__file__), "..", "..", "build"),
        os.path.join(os.path.dirname(__file__), "..", "..", "build", "Release"),
        "/usr/local/lib",
        "/usr/lib",
    ]

    # 环境变量
    if "SCL_LIB_PATH" in os.environ:
        search_paths.insert(0, os.environ["SCL_LIB_PATH"])

    for path in search_paths:
        for name in lib_names:
            lib_path = os.path.join(path, name)
            if os.path.exists(lib_path):
                return ctypes.CDLL(lib_path)

    raise RuntimeError(
        f"Cannot find SCL library. Searched: {search_paths}. "
        "Set SCL_LIB_PATH environment variable or install the library."
    )


# =============================================================================
# 函数签名
# =============================================================================

def _setup_functions(lib):
    """设置函数签名"""

    # --- Lifecycle ---
    lib.scl_mmap_create_csr_from_ptr.argtypes = [
        c_void_p, c_void_p, c_void_p,  # data, indices, indptr
        c_int64, c_int64, c_int64,      # rows, cols, nnz
        c_int64,                         # max_pages
        POINTER(c_int64)                 # out_handle
    ]
    lib.scl_mmap_create_csr_from_ptr.restype = c_int32

    lib.scl_mmap_open_csr_file.argtypes = [c_char_p, c_int64, POINTER(c_int64)]
    lib.scl_mmap_open_csr_file.restype = c_int32

    lib.scl_mmap_release.argtypes = [c_int64]
    lib.scl_mmap_release.restype = c_int32

    lib.scl_mmap_type.argtypes = [c_int64]
    lib.scl_mmap_type.restype = c_char_p

    # --- Properties ---
    lib.scl_mmap_csr_shape.argtypes = [
        c_int64, POINTER(c_int64), POINTER(c_int64), POINTER(c_int64)
    ]
    lib.scl_mmap_csr_shape.restype = c_int32

    # --- Load Operations ---
    lib.scl_mmap_csr_load_full.argtypes = [c_int64, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_load_full.restype = c_int32

    lib.scl_mmap_csr_load_masked.argtypes = [
        c_int64,                         # handle
        POINTER(c_uint8), POINTER(c_uint8),  # row_mask, col_mask
        c_void_p, c_void_p, c_void_p,    # data, indices, indptr
        POINTER(c_int64), POINTER(c_int64), POINTER(c_int64)  # out dims
    ]
    lib.scl_mmap_csr_load_masked.restype = c_int32

    lib.scl_mmap_csr_compute_masked_nnz.argtypes = [
        c_int64, POINTER(c_uint8), POINTER(c_uint8), POINTER(c_int64)
    ]
    lib.scl_mmap_csr_compute_masked_nnz.restype = c_int32

    lib.scl_mmap_csr_load_indexed.argtypes = [
        c_int64,                         # handle
        c_void_p, c_int64,               # row_indices, num_rows
        c_void_p, c_int64,               # col_indices, num_cols
        c_void_p, c_void_p, c_void_p,    # data, indices, indptr
        POINTER(c_int64)                 # out_nnz
    ]
    lib.scl_mmap_csr_load_indexed.restype = c_int32

    # --- View Operations ---
    lib.scl_mmap_csr_create_view.argtypes = [
        c_int64, POINTER(c_uint8), POINTER(c_uint8), POINTER(c_int64)
    ]
    lib.scl_mmap_csr_create_view.restype = c_int32

    lib.scl_mmap_view_shape.argtypes = [
        c_int64, POINTER(c_int64), POINTER(c_int64), POINTER(c_int64)
    ]
    lib.scl_mmap_view_shape.restype = c_int32

    # --- Reorder Operations ---
    lib.scl_mmap_csr_reorder_rows.argtypes = [
        c_int64, c_void_p, c_int64, c_void_p, c_void_p, c_void_p
    ]
    lib.scl_mmap_csr_reorder_rows.restype = c_int32

    lib.scl_mmap_csr_reorder_cols.argtypes = [
        c_int64, c_void_p, c_int64, c_void_p, c_void_p, c_void_p
    ]
    lib.scl_mmap_csr_reorder_cols.restype = c_int32

    # --- Format Conversion ---
    lib.scl_mmap_csr_to_csc.argtypes = [c_int64, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_to_csc.restype = c_int32

    lib.scl_mmap_csr_to_dense.argtypes = [c_int64, c_void_p]
    lib.scl_mmap_csr_to_dense.restype = c_int32

    # --- Statistics ---
    lib.scl_mmap_csr_row_sum.argtypes = [c_int64, c_void_p]
    lib.scl_mmap_csr_row_sum.restype = c_int32

    lib.scl_mmap_csr_row_mean.argtypes = [c_int64, c_void_p, c_int32]
    lib.scl_mmap_csr_row_mean.restype = c_int32

    lib.scl_mmap_csr_row_var.argtypes = [c_int64, c_void_p, c_void_p, c_int32]
    lib.scl_mmap_csr_row_var.restype = c_int32

    lib.scl_mmap_csr_col_sum.argtypes = [c_int64, c_void_p]
    lib.scl_mmap_csr_col_sum.restype = c_int32

    lib.scl_mmap_csr_global_sum.argtypes = [c_int64, c_void_p]
    lib.scl_mmap_csr_global_sum.restype = c_int32

    # --- Normalization ---
    lib.scl_mmap_csr_normalize_l1.argtypes = [c_int64, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_normalize_l1.restype = c_int32

    lib.scl_mmap_csr_normalize_l2.argtypes = [c_int64, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_normalize_l2.restype = c_int32

    # --- Transforms ---
    lib.scl_mmap_csr_log1p.argtypes = [c_int64, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_log1p.restype = c_int32

    lib.scl_mmap_csr_scale_rows.argtypes = [c_int64, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_scale_rows.restype = c_int32

    lib.scl_mmap_csr_scale_cols.argtypes = [c_int64, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_scale_cols.restype = c_int32

    # --- SpMV ---
    lib.scl_mmap_csr_spmv.argtypes = [c_int64, c_void_p, c_void_p]
    lib.scl_mmap_csr_spmv.restype = c_int32

    # --- Filtering ---
    lib.scl_mmap_csr_filter_threshold.argtypes = [
        c_int64, c_double, c_void_p, c_void_p, c_void_p, POINTER(c_int64)
    ]
    lib.scl_mmap_csr_filter_threshold.restype = c_int32

    lib.scl_mmap_csr_top_k.argtypes = [c_int64, c_int64, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_top_k.restype = c_int32

    # --- Utility ---
    lib.scl_mmap_get_config.argtypes = [POINTER(c_int64), POINTER(c_int64)]
    lib.scl_mmap_get_config.restype = c_int32

    lib.scl_mmap_estimate_memory.argtypes = [c_int64, c_int64, POINTER(c_int64)]
    lib.scl_mmap_estimate_memory.restype = c_int32

    lib.scl_mmap_suggest_backend.argtypes = [c_int64, c_int64, POINTER(c_int32)]
    lib.scl_mmap_suggest_backend.restype = c_int32

    return lib


@lru_cache(maxsize=1)
def get_lib_with_signatures():
    """获取带签名的库"""
    lib = get_lib()
    return _setup_functions(lib)


# =============================================================================
# 配置查询
# =============================================================================

def get_config() -> tuple[int, int]:
    """获取 mmap 配置: (page_size, default_pool_size)"""
    lib = get_lib_with_signatures()
    page_size = c_int64()
    pool_size = c_int64()
    check_error(lib.scl_mmap_get_config(byref(page_size), byref(pool_size)))
    return page_size.value, pool_size.value


def estimate_memory(rows: int, nnz: int) -> int:
    """估算稀疏矩阵内存需求 (bytes)"""
    lib = get_lib_with_signatures()
    result = c_int64()
    check_error(lib.scl_mmap_estimate_memory(rows, nnz, byref(result)))
    return result.value


class Backend:
    """后端类型"""
    IN_MEMORY = 0
    MAPPED = 1
    STREAMING = 2


def suggest_backend(data_bytes: int, available_mb: int = 4096) -> int:
    """建议使用的后端类型"""
    lib = get_lib_with_signatures()
    result = c_int32()
    check_error(lib.scl_mmap_suggest_backend(data_bytes, available_mb, byref(result)))
    return result.value
