"""
Python binding generator using type annotations.

Generates clean Python modules with:
- Type-annotated function signatures
- @register decorator for binding
- IDE-friendly code with full type hints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from .base import Generator, GeneratedFile
from ..parser.c_types import ParsedHeader, CFunction, CStruct, CEnum
from ..config import CodegenConfig


# Python reserved keywords that need renaming
PYTHON_KEYWORDS = {
    "lambda", "class", "def", "return", "yield", "import", "from", "as",
    "if", "elif", "else", "for", "while", "break", "continue", "pass",
    "try", "except", "finally", "raise", "with", "assert", "global",
    "nonlocal", "del", "in", "is", "not", "and", "or", "True", "False",
    "None", "async", "await", "type", "match", "case",
}


def _safe_param_name(name: str) -> str:
    """Convert parameter name to safe Python identifier."""
    if name in PYTHON_KEYWORDS:
        return f"{name}_"
    return name


# C type to Python type marker mapping
TYPE_MARKER_MAP = {
    # SCL dynamic types
    "scl_real_t": "Real",
    "scl_index_t": "Index",
    "scl_size_t": "Size",
    "scl_error_t": "Error",
    "scl_bool_t": "Bool",
    
    # Handle types
    "scl_sparse_t": "SparseHandle",
    "scl_dense_t": "DenseHandle",
    
    # Fixed-width integers
    "int8_t": "Int8",
    "int16_t": "Int16",
    "int32_t": "Int32",
    "int64_t": "Int64",
    "uint8_t": "UInt8",
    "uint16_t": "UInt16",
    "uint32_t": "UInt32",
    "uint64_t": "UInt64",
    
    # Basic types
    "int": "Int32",
    "long": "Int64",
    "float": "Float",
    "double": "Double",
    "char": "Char",
    "bool": "Bool",
    "_Bool": "Bool",
    "void": "Void",
    "size_t": "Size",
}


@dataclass
class FunctionStub:
    """Function stub for .pyi generation."""
    name: str
    params: List[str]  # ["A: c_void_p", "x: POINTER(c_double)", ...]
    return_type: str   # "c_int32"


@dataclass
class GeneratedBindingFile(GeneratedFile):
    """Extended result with function stubs for .pyi generation."""
    function_stubs: List[FunctionStub] = field(default_factory=list)


class PythonBindingGenerator(Generator):
    """
    Generator for type-annotated Python bindings.
    """

    def __init__(self, config: CodegenConfig):
        super().__init__(config)

    def get_output_path(self, parsed: ParsedHeader) -> Path:
        c_api_dir = self.config.c_api_dir_abs
        try:
            rel_path = parsed.path.relative_to(c_api_dir)
        except ValueError:
            rel_path = Path(parsed.path.name)
        return self.config.python_output_abs / rel_path.with_suffix(".py")

    def generate(self, parsed: ParsedHeader) -> GeneratedBindingFile:
        """Generate binding module."""
        output_path = self.get_output_path(parsed)
        
        # Calculate import depth
        try:
            rel = output_path.relative_to(self.config.python_output_abs)
            depth = len(rel.parts) - 1
        except ValueError:
            depth = 0
        
        prefix = "." * (depth + 1) if depth > 0 else "."
        
        lines = []
        function_names = []
        
        # Header
        lines.extend(self._gen_header(parsed))
        lines.append("")
        
        # Imports
        lines.extend(self._gen_imports(parsed, prefix))
        lines.append("")
        lines.append("")
        
        # Enums
        if parsed.enums:
            lines.append("# " + "=" * 75)
            lines.append("# Enums")
            lines.append("# " + "=" * 75)
            lines.append("")
            for enum in parsed.enums:
                lines.extend(self._gen_enum(enum))
                lines.append("")
        
        # Functions
        function_stubs = []
        if parsed.functions:
            lines.append("# " + "=" * 75)
            lines.append("# Functions")
            lines.append("# " + "=" * 75)
            lines.append("")
            for func in parsed.functions:
                lines.extend(self._gen_function(func))
                lines.append("")
                function_stubs.append(self._gen_function_stub(func))
        
        # Exports
        lines.append("# " + "=" * 75)
        lines.append("# Exports")
        lines.append("# " + "=" * 75)
        lines.append("")
        lines.extend(self._gen_exports(parsed))
        
        return GeneratedBindingFile(
            path=output_path,
            content="\n".join(lines),
            source=parsed.path,
            function_stubs=function_stubs,
        )

    def _gen_header(self, parsed: ParsedHeader) -> List[str]:
        """Generate module header."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return [
            '"""',
            f"Auto-generated Python bindings for {parsed.module_name}",
            "",
            f"Source: {parsed.path}",
            f"Generated: {ts}",
            "",
            "Usage:",
            "    from scl._bindings._loader import lib",
            f"    from scl._bindings import {parsed.module_name}",
            "",
            "    result = lib.scl_...(args)",
            '"""',
        ]

    def _gen_imports(self, parsed: ParsedHeader, prefix: str) -> List[str]:
        """Generate imports based on used types."""
        # Collect used type markers
        used_types = set()
        has_pointers = False
        
        def extract_types(marker: str) -> None:
            """Extract base types from a marker, handling Ptr[T] generics."""
            nonlocal has_pointers
            if marker and marker.startswith("Ptr["):
                has_pointers = True
                inner = marker[4:-1]
                used_types.add(inner)
            elif marker:
                used_types.add(marker)
        
        for func in parsed.functions:
            for param in func.parameters:
                type_str = str(param.type)
                marker = self._get_type_marker(type_str)
                extract_types(marker)
            
            ret_marker = self._get_type_marker(str(func.return_type))
            extract_types(ret_marker)
        
        # Build import list
        type_imports = sorted(used_types)
        if has_pointers:
            type_imports = ["Ptr"] + type_imports
        
        lines = ["from __future__ import annotations", ""]
        
        if type_imports:
            imports_str = ", ".join(type_imports)
            lines.append(f"from {prefix}_types import {imports_str}")
        
        lines.append(f"from {prefix}_loader import register")
        
        return lines

    def _gen_enum(self, enum: CEnum) -> List[str]:
        """Generate enum class."""
        lines = []
        name = self._to_class_name(enum.name)
        
        lines.append(f"class {name}:")
        if enum.doc:
            lines.append(f'    """{enum.doc}"""')
        
        if enum.values:
            for val in enum.values:
                lines.append(f"    {val.name} = {val.value if val.value is not None else 'None'}")
        else:
            lines.append("    pass")
        
        return lines

    def _gen_function(self, func: CFunction) -> List[str]:
        """Generate function with type annotations."""
        lines = []
        
        # Build parameter list with type annotations
        params = []
        for param in func.parameters:
            safe_name = _safe_param_name(param.name)
            type_marker = self._get_type_marker(str(param.type))
            if type_marker:
                params.append(f"{safe_name}: {type_marker}")
            else:
                params.append(safe_name)
        
        # Return type
        ret_marker = self._get_type_marker(str(func.return_type))
        ret_annotation = f" -> {ret_marker}" if ret_marker else ""
        
        # Format parameters (multi-line if many)
        if len(params) > 3:
            params_str = ",\n    ".join(params)
            sig = f"def {func.name}(\n    {params_str},\n){ret_annotation}:"
        else:
            params_str = ", ".join(params)
            sig = f"def {func.name}({params_str}){ret_annotation}:"
        
        # Decorator and signature
        lines.append("@register")
        lines.append(sig)
        
        # Docstring
        lines.append('    """')
        if func.doc:
            lines.append(f"    {func.doc}")
            lines.append("")
        
        lines.append("    Args:")
        for param in func.parameters:
            lines.append(f"        {param.name}: {param.type}")
        
        lines.append("")
        lines.append("    Returns:")
        lines.append(f"        {func.return_type}")
        lines.append('    """')
        
        return lines

    def _gen_function_stub(self, func: CFunction) -> FunctionStub:
        """Generate function stub for .pyi file."""
        params = []
        for param in func.parameters:
            safe_name = _safe_param_name(param.name)
            pyi_type = self._get_pyi_type(str(param.type))
            params.append(f"{safe_name}: {pyi_type}")
        
        ret_type = self._get_pyi_type(str(func.return_type)) or "None"
        
        return FunctionStub(
            name=func.name,
            params=params,
            return_type=ret_type,
        )

    def _get_pyi_type(self, c_type: str) -> str:
        """Convert C type to .pyi ctypes type."""
        c_type = c_type.strip()
        
        # Handle void
        if c_type == "void":
            return "None"
        
        # Handle const
        if c_type.startswith("const "):
            c_type = c_type[6:].strip()
        
        # Count pointers
        ptr_depth = 0
        while c_type.endswith("*"):
            ptr_depth += 1
            c_type = c_type[:-1].strip()
        
        # Handle trailing const
        if c_type.endswith(" const"):
            c_type = c_type[:-6].strip()
        
        # Map to ctypes
        ctypes_map = {
            "scl_real_t": "float",  # Will be c_double or c_float
            "scl_index_t": "int",   # Will be c_int32 or c_int64
            "scl_size_t": "int",
            "scl_error_t": "int",
            "scl_bool_t": "bool",
            "scl_sparse_t": "c_void_p",
            "scl_dense_t": "c_void_p",
            "int8_t": "int",
            "int16_t": "int",
            "int32_t": "int",
            "int64_t": "int",
            "uint8_t": "int",
            "uint16_t": "int",
            "uint32_t": "int",
            "uint64_t": "int",
            "int": "int",
            "long": "int",
            "float": "float",
            "double": "float",
            "char": "bytes",
            "bool": "bool",
            "_Bool": "bool",
            "size_t": "int",
        }
        
        base_type = ctypes_map.get(c_type, "Any")
        
        if ptr_depth > 0:
            # Pointer types are passed as ctypes objects or Python objects
            return "Any"  # Flexible for pointers
        
        return base_type

    def _gen_exports(self, parsed: ParsedHeader) -> List[str]:
        """Generate __all__ list."""
        exports = []
        
        for enum in parsed.enums:
            if enum.name:
                exports.append(self._to_class_name(enum.name))
        
        for func in parsed.functions:
            exports.append(func.name)
        
        lines = ["__all__ = ["]
        for name in sorted(exports):
            lines.append(f'    "{name}",')
        lines.append("]")
        
        return lines

    def _get_type_marker(self, c_type: str) -> Optional[str]:
        """Convert C type string to Python type marker."""
        c_type = c_type.strip()
        
        # Handle void
        if c_type == "void":
            return None
        
        # Handle const
        is_const = c_type.startswith("const ")
        if is_const:
            c_type = c_type[6:].strip()
        
        # Count pointers
        ptr_depth = 0
        while c_type.endswith("*"):
            ptr_depth += 1
            c_type = c_type[:-1].strip()
        
        # Handle trailing const
        if c_type.endswith(" const"):
            c_type = c_type[:-6].strip()
        
        # Map base type
        base_marker = TYPE_MARKER_MAP.get(c_type)
        
        if base_marker is None:
            # Unknown type - use handle if it's a struct pointer
            if ptr_depth > 0:
                return "SparseHandle"  # Default to generic handle
            return None
        
        # Apply pointer wrapper
        if ptr_depth > 0:
            return f"Ptr[{base_marker}]"
        
        return base_marker

    def _to_class_name(self, name: str) -> str:
        """Convert C name to Python class name."""
        if name.startswith("scl_"):
            name = name[4:]
        if name.endswith("_t"):
            name = name[:-2]
        return "".join(p.capitalize() for p in name.split("_"))
