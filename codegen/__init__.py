"""
SCL-Core Code Generator

A toolchain for generating Python bindings and documentation from C API headers.
"""

__version__ = "0.1.0"
__author__ = "SCL-Core Team"

from .config import CodegenConfig

__all__ = ["CodegenConfig", "__version__"]
