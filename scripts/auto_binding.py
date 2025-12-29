#!/usr/bin/env python3
"""
SCL Python Bindings Generator
==============================

Automatically generates Python ctypes bindings from C API header files
using libclang for accurate C parsing.

Usage:
    python generate_bindings.py [options]

Requirements:
    pip install clang libclang

Author: SCL Team
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

try:
    from clang.cindex import (
        Config,
        CursorKind,
        Index,
        TranslationUnit,
        TypeKind,
    )
except ImportError:
    print("Error: clang package not found. Install with: pip install libclang")
    sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================

# Default paths (relative to project root)
DEFAULT_C_API_DIR = "scl/binding/c_api"
DEFAULT_OUTPUT_DIR = "python/scl/_bindings"
DEFAULT_LIB_NAME = "scl"

# Type mapping: C types -> ctypes
C_TO_CTYPES: dict[str, str] = {
    # Basic types
    "void": "None",
    "bool": "c_bool",
    "char": "c_char",
    "signed char": "c_byte",
    "unsigned char": "c_ubyte",
    "short": "c_short",
    "unsigned short": "c_ushort",
    "int": "c_int",
    "unsigned int": "c_uint",
    "long": "c_long",
    "unsigned long": "c_ulong",
    "long long": "c_longlong",
    "unsigned long long": "c_ulonglong",
    "float": "c_float",
    "double": "c_double",
    "long double": "c_longdouble",
    # Size types
    "size_t": "c_size_t",
    "ssize_t": "c_ssize_t",
    "ptrdiff_t": "c_ptrdiff_t",
    # Fixed-width integers
    "int8_t": "c_int8",
    "int16_t": "c_int16",
    "int32_t": "c_int32",
    "int64_t": "c_int64",
    "uint8_t": "c_uint8",
    "uint16_t": "c_uint16",
    "uint32_t": "c_uint32",
    "uint64_t": "c_uint64",
    # Pointer types (base)
    "void *": "c_void_p",
    "char *": "c_char_p",
    "const char *": "c_char_p",
    "wchar_t *": "c_wchar_p",
    "const wchar_t *": "c_wchar_p",
}

# SCL-specific type aliases
SCL_TYPE_ALIASES: dict[str, str] = {
    "SCL_Float": "c_float",  # or c_double depending on precision
    "SCL_Double": "c_double",
    "SCL_Int": "c_int32",
    "SCL_Int32": "c_int32",
    "SCL_Int64": "c_int64",
    "SCL_UInt": "c_uint32",
    "SCL_Size": "c_size_t",
    "SCL_Index": "c_int64",
    "SCL_Bool": "c_bool",
    "SCL_Status": "c_int",
    "SCL_Handle": "c_void_p",
    "SCL_Context": "c_void_p",
    "SCL_Error": "c_int",
}

# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class Parameter:
    """Represents a function parameter."""

    name: str
    c_type: str
    ctypes_type: str
    is_pointer: bool = False
    is_const: bool = False
    is_array: bool = False
    array_size: Optional[int] = None


@dataclass
class Function:
    """Represents a C function declaration."""

    name: str
    return_type: str
    return_ctypes: str
    parameters: list[Parameter] = field(default_factory=list)
    docstring: str = ""
    is_variadic: bool = False
    source_file: str = ""
    line_number: int = 0


@dataclass
class Enum:
    """Represents a C enum."""

    name: str
    values: dict[str, int] = field(default_factory=dict)
    docstring: str = ""


@dataclass
class Struct:
    """Represents a C struct."""

    name: str
    fields: list[tuple[str, str]] = field(default_factory=list)  # (name, ctypes_type)
    is_opaque: bool = False
    docstring: str = ""


@dataclass
class TypeDef:
    """Represents a typedef."""

    name: str
    underlying_type: str
    ctypes_type: str


@dataclass
class ParsedHeader:
    """Contains all parsed declarations from a header file."""

    file_path: Path
    functions: list[Function] = field(default_factory=list)
    enums: list[Enum] = field(default_factory=list)
    structs: list[Struct] = field(default_factory=list)
    typedefs: list[TypeDef] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)


# =============================================================================
# Type Conversion
# =============================================================================


class TypeConverter:
    """Converts C types to ctypes equivalents."""

    def __init__(self, custom_types: Optional[dict[str, str]] = None):
        self.type_map = {**C_TO_CTYPES, **SCL_TYPE_ALIASES}
        if custom_types:
            self.type_map.update(custom_types)
        self._registered_structs: set[str] = set()

    def register_struct(self, name: str) -> None:
        """Register a struct type for pointer resolution."""
        self._registered_structs.add(name)

    def convert(self, c_type: str) -> str:
        """Convert a C type string to ctypes equivalent."""
        # Normalize whitespace
        c_type = " ".join(c_type.split())

        # Direct lookup
        if c_type in self.type_map:
            return self.type_map[c_type]

        # Handle const qualifier
        if c_type.startswith("const "):
            base_type = c_type[6:]
            return self.convert(base_type)

        # Handle pointers
        if c_type.endswith("*"):
            base_type = c_type[:-1].strip()
            if base_type.endswith("const"):
                base_type = base_type[:-5].strip()

            # void* -> c_void_p
            if base_type == "void":
                return "c_void_p"

            # char* -> c_char_p
            if base_type in ("char", "const char"):
                return "c_char_p"

            # Known struct pointer
            if base_type in self._registered_structs:
                return f"POINTER({base_type})"

            # Generic pointer
            base_ctypes = self.convert(base_type)
            if base_ctypes == "None":
                return "c_void_p"
            return f"POINTER({base_ctypes})"

        # Handle arrays (e.g., "int[10]")
        array_match = re.match(r"(.+)\[(\d+)\]$", c_type)
        if array_match:
            base_type = array_match.group(1).strip()
            size = int(array_match.group(2))
            base_ctypes = self.convert(base_type)
            return f"{base_ctypes} * {size}"

        # Handle function pointers (simplified)
        if "(*)" in c_type or c_type.startswith("void (*"):
            return "c_void_p"  # Simplified handling

        # Unknown type - return as-is with warning
        logging.warning(f"Unknown C type: '{c_type}', using c_void_p")
        return "c_void_p"


# =============================================================================
# Clang Parser
# =============================================================================


class ClangParser:
    """Parses C header files using libclang."""

    def __init__(
        self,
        include_paths: Optional[list[str]] = None,
        defines: Optional[dict[str, str]] = None,
    ):
        self.index = Index.create()
        self.include_paths = include_paths or []
        self.defines = defines or {}
        self.type_converter = TypeConverter()

    def _get_compile_args(self) -> list[str]:
        """Generate clang compile arguments."""
        args = ["-x", "c", "-std=c11"]

        for path in self.include_paths:
            args.append(f"-I{path}")

        for name, value in self.defines.items():
            if value:
                args.append(f"-D{name}={value}")
            else:
                args.append(f"-D{name}")

        return args

    def _get_docstring(self, cursor) -> str:
        """Extract documentation comment from cursor."""
        if cursor.raw_comment:
            comment = cursor.raw_comment
            # Clean up comment markers
            lines = []
            for line in comment.split("\n"):
                line = line.strip()
                # Remove comment markers
                line = re.sub(r"^/\*\*?|\*/$|^\*\s?|^///?", "", line)
                if line:
                    lines.append(line)
            return "\n".join(lines)
        return ""

    def _parse_type(self, clang_type) -> tuple[str, str]:
        """Parse a clang type and return (c_type, ctypes_type)."""
        c_type = clang_type.spelling

        # Handle elaborated types (struct X, enum Y)
        if clang_type.kind == TypeKind.ELABORATED:
            c_type = clang_type.get_canonical().spelling

        # Handle typedefs
        if clang_type.kind == TypeKind.TYPEDEF:
            typedef_name = clang_type.spelling
            if typedef_name in self.type_converter.type_map:
                return typedef_name, self.type_converter.type_map[typedef_name]
            # Get underlying type
            c_type = clang_type.get_canonical().spelling

        ctypes_type = self.type_converter.convert(c_type)
        return c_type, ctypes_type

    def _parse_function(self, cursor, source_file: str) -> Optional[Function]:
        """Parse a function declaration."""
        if cursor.kind != CursorKind.FUNCTION_DECL:
            return None

        # Skip if not from our header
        if cursor.location.file and str(cursor.location.file) != source_file:
            return None

        name = cursor.spelling
        return_c_type, return_ctypes = self._parse_type(cursor.result_type)

        params = []
        is_variadic = False

        for child in cursor.get_children():
            if child.kind == CursorKind.PARM_DECL:
                param_name = child.spelling or f"arg{len(params)}"
                param_c_type, param_ctypes = self._parse_type(child.type)

                is_pointer = child.type.kind == TypeKind.POINTER
                is_const = "const" in param_c_type

                params.append(
                    Parameter(
                        name=param_name,
                        c_type=param_c_type,
                        ctypes_type=param_ctypes,
                        is_pointer=is_pointer,
                        is_const=is_const,
                    )
                )

        # Check for variadic
        if cursor.type.is_function_variadic():
            is_variadic = True

        return Function(
            name=name,
            return_type=return_c_type,
            return_ctypes=return_ctypes,
            parameters=params,
            docstring=self._get_docstring(cursor),
            is_variadic=is_variadic,
            source_file=source_file,
            line_number=cursor.location.line,
        )

    def _parse_enum(self, cursor) -> Optional[Enum]:
        """Parse an enum declaration."""
        if cursor.kind != CursorKind.ENUM_DECL:
            return None

        name = cursor.spelling
        if not name:
            return None  # Anonymous enum

        values = {}
        for child in cursor.get_children():
            if child.kind == CursorKind.ENUM_CONSTANT_DECL:
                values[child.spelling] = child.enum_value

        return Enum(
            name=name,
            values=values,
            docstring=self._get_docstring(cursor),
        )

    def _parse_struct(self, cursor) -> Optional[Struct]:
        """Parse a struct declaration."""
        if cursor.kind != CursorKind.STRUCT_DECL:
            return None

        name = cursor.spelling
        if not name:
            return None  # Anonymous struct

        # Check if opaque (forward declaration)
        is_opaque = not cursor.is_definition()

        fields = []
        if not is_opaque:
            for child in cursor.get_children():
                if child.kind == CursorKind.FIELD_DECL:
                    field_name = child.spelling
                    _, field_ctypes = self._parse_type(child.type)
                    fields.append((field_name, field_ctypes))

        self.type_converter.register_struct(name)

        return Struct(
            name=name,
            fields=fields,
            is_opaque=is_opaque,
            docstring=self._get_docstring(cursor),
        )

    def _parse_typedef(self, cursor) -> Optional[TypeDef]:
        """Parse a typedef declaration."""
        if cursor.kind != CursorKind.TYPEDEF_DECL:
            return None

        name = cursor.spelling
        underlying = cursor.underlying_typedef_type
        underlying_c_type = underlying.spelling
        ctypes_type = self.type_converter.convert(underlying_c_type)

        # Register in type map for future use
        self.type_converter.type_map[name] = ctypes_type

        return TypeDef(
            name=name,
            underlying_type=underlying_c_type,
            ctypes_type=ctypes_type,
        )

    def parse_header(self, header_path: Path) -> ParsedHeader:
        """Parse a single header file."""
        logging.info(f"Parsing: {header_path}")

        args = self._get_compile_args()
        tu = self.index.parse(
            str(header_path),
            args=args,
            options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            | TranslationUnit.PARSE_SKIP_FUNCTION_BODIES,
        )

        # Check for errors
        for diag in tu.diagnostics:
            if diag.severity >= 3:  # Error or Fatal
                logging.error(f"Clang error in {header_path}: {diag.spelling}")

        result = ParsedHeader(file_path=header_path)
        source_file = str(header_path)

        def visit(cursor):
            # Only process items from this file
            if cursor.location.file and str(cursor.location.file) == source_file:
                if cursor.kind == CursorKind.FUNCTION_DECL:
                    func = self._parse_function(cursor, source_file)
                    if func:
                        result.functions.append(func)

                elif cursor.kind == CursorKind.ENUM_DECL:
                    enum = self._parse_enum(cursor)
                    if enum:
                        result.enums.append(enum)

                elif cursor.kind == CursorKind.STRUCT_DECL:
                    struct = self._parse_struct(cursor)
                    if struct:
                        result.structs.append(struct)

                elif cursor.kind == CursorKind.TYPEDEF_DECL:
                    typedef = self._parse_typedef(cursor)
                    if typedef:
                        result.typedefs.append(typedef)

                elif cursor.kind == CursorKind.INCLUSION_DIRECTIVE:
                    result.includes.append(cursor.spelling)

            for child in cursor.get_children():
                visit(child)

        visit(tu.cursor)

        logging.info(
            f"  Found: {len(result.functions)} functions, "
            f"{len(result.enums)} enums, "
            f"{len(result.structs)} structs, "
            f"{len(result.typedefs)} typedefs"
        )

        return result


# =============================================================================
# Code Generator
# =============================================================================


class BindingGenerator:
    """Generates Python binding code from parsed headers."""

    def __init__(
        self,
        lib_name: str = DEFAULT_LIB_NAME,
        precision: str = "f32_i32",
    ):
        self.lib_name = lib_name
        self.precision = precision

    def _generate_header(self, module_name: str, source_file: str) -> str:
        """Generate module header with imports."""
        return f'''"""
Auto-generated Python bindings for {module_name}

Source: {source_file}
Generated by: generate_bindings.py

DO NOT EDIT - This file is automatically generated.
"""

from __future__ import annotations

from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_bool,
    c_char,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_int8,
    c_int16,
    c_int32,
    c_int64,
    c_long,
    c_longlong,
    c_short,
    c_size_t,
    c_ssize_t,
    c_ubyte,
    c_uint,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_ulong,
    c_ulonglong,
    c_ushort,
    c_void_p,
    c_wchar_p,
)
from typing import Any, Optional

from .._loader import get_library

# Load the native library
_lib = get_library()

'''

    def _generate_enum(self, enum: Enum) -> str:
        """Generate enum class definition."""
        lines = []

        # Add docstring
        if enum.docstring:
            lines.append(f'"""{enum.docstring}"""')
            lines.append("")

        lines.append(f"class {enum.name}:")
        lines.append(f'    """C enum: {enum.name}"""')

        for name, value in enum.values.items():
            lines.append(f"    {name} = {value}")

        lines.append("")
        return "\n".join(lines)

    def _generate_struct(self, struct: Struct) -> str:
        """Generate struct class definition."""
        lines = []

        if struct.is_opaque:
            # Opaque struct - just a type marker
            lines.append(f"class {struct.name}(Structure):")
            lines.append(f'    """Opaque struct: {struct.name}"""')
            lines.append("    pass")
        else:
            lines.append(f"class {struct.name}(Structure):")
            if struct.docstring:
                lines.append(f'    """{struct.docstring}"""')
            else:
                lines.append(f'    """C struct: {struct.name}"""')

            if struct.fields:
                lines.append("    _fields_ = [")
                for field_name, field_type in struct.fields:
                    lines.append(f'        ("{field_name}", {field_type}),')
                lines.append("    ]")
            else:
                lines.append("    pass")

        lines.append("")
        return "\n".join(lines)

    def _generate_typedef(self, typedef: TypeDef) -> str:
        """Generate typedef alias."""
        return f"{typedef.name} = {typedef.ctypes_type}  # typedef {typedef.underlying_type}\n"

    def _generate_function(self, func: Function) -> str:
        """Generate function binding."""
        lines = []

        # Function signature comment
        param_strs = [f"{p.c_type} {p.name}" for p in func.parameters]
        c_sig = f"{func.return_type} {func.name}({', '.join(param_strs)})"
        lines.append(f"# C: {c_sig}")

        # argtypes
        if func.parameters:
            arg_types = ", ".join(p.ctypes_type for p in func.parameters)
            lines.append(f"_lib.{func.name}.argtypes = [{arg_types}]")
        else:
            lines.append(f"_lib.{func.name}.argtypes = []")

        # restype
        lines.append(f"_lib.{func.name}.restype = {func.return_ctypes}")

        lines.append("")

        # Python wrapper function
        py_params = []
        for p in func.parameters:
            py_params.append(p.name)

        params_str = ", ".join(py_params) if py_params else ""

        lines.append(f"def {func.name}({params_str}):")

        # Docstring
        doc_lines = []
        if func.docstring:
            doc_lines.append(func.docstring)
            doc_lines.append("")

        doc_lines.append("Args:")
        for p in func.parameters:
            doc_lines.append(f"    {p.name}: {p.c_type}")

        doc_lines.append("")
        doc_lines.append(f"Returns:")
        doc_lines.append(f"    {func.return_type}")

        lines.append('    """')
        for dl in doc_lines:
            lines.append(f"    {dl}")
        lines.append('    """')

        # Function call
        call_args = ", ".join(py_params)
        lines.append(f"    return _lib.{func.name}({call_args})")

        lines.append("")
        lines.append("")
        return "\n".join(lines)

    def generate_module(self, parsed: ParsedHeader) -> str:
        """Generate complete module code."""
        module_name = parsed.file_path.stem
        source_file = str(parsed.file_path)

        parts = [self._generate_header(module_name, source_file)]

        # Enums
        if parsed.enums:
            parts.append("# " + "=" * 75)
            parts.append("# Enums")
            parts.append("# " + "=" * 75)
            parts.append("")
            for enum in parsed.enums:
                parts.append(self._generate_enum(enum))

        # Structs
        if parsed.structs:
            parts.append("# " + "=" * 75)
            parts.append("# Structures")
            parts.append("# " + "=" * 75)
            parts.append("")
            for struct in parsed.structs:
                parts.append(self._generate_struct(struct))

        # Typedefs
        if parsed.typedefs:
            parts.append("# " + "=" * 75)
            parts.append("# Type Aliases")
            parts.append("# " + "=" * 75)
            parts.append("")
            for typedef in parsed.typedefs:
                parts.append(self._generate_typedef(typedef))
            parts.append("")

        # Functions
        if parsed.functions:
            parts.append("# " + "=" * 75)
            parts.append("# Functions")
            parts.append("# " + "=" * 75)
            parts.append("")
            for func in parsed.functions:
                parts.append(self._generate_function(func))

        # __all__ export list
        all_names = []
        all_names.extend(e.name for e in parsed.enums)
        all_names.extend(s.name for s in parsed.structs)
        all_names.extend(t.name for t in parsed.typedefs)
        all_names.extend(f.name for f in parsed.functions)

        if all_names:
            parts.append("# " + "=" * 75)
            parts.append("# Exports")
            parts.append("# " + "=" * 75)
            parts.append("")
            parts.append("__all__ = [")
            for name in sorted(all_names):
                parts.append(f'    "{name}",')
            parts.append("]")

        return "\n".join(parts)


# =============================================================================
# Library Loader Generator
# =============================================================================


def generate_loader(output_dir: Path, lib_name: str) -> None:
    """Generate the library loader module."""
    loader_code = f'''"""
Native Library Loader for SCL

Handles loading the correct shared library based on platform and precision.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path
from typing import Optional

_lib_cache: dict[str, ctypes.CDLL] = {{}}

def _get_lib_path() -> Path:
    """Get the path to the native library directory."""
    # Check environment variable first
    if "SCL_LIB_PATH" in os.environ:
        return Path(os.environ["SCL_LIB_PATH"])
    
    # Default: relative to this file
    return Path(__file__).parent.parent / "libs"


def _get_lib_name(precision: str = "f32_i32") -> str:
    """Get the library filename for current platform."""
    system = platform.system()
    
    base_name = f"{lib_name}_{{precision}}"
    
    if system == "Windows":
        return f"{{base_name}}.dll"
    elif system == "Darwin":
        return f"lib{{base_name}}.dylib"
    else:  # Linux and others
        return f"lib{{base_name}}.so"


def get_library(precision: str = "f32_i32") -> ctypes.CDLL:
    """
    Load and return the native SCL library.
    
    Args:
        precision: Library variant to load ("f32_i32" or "f64_i64")
    
    Returns:
        Loaded ctypes.CDLL instance
    
    Raises:
        OSError: If library cannot be found or loaded
    """
    if precision in _lib_cache:
        return _lib_cache[precision]
    
    lib_dir = _get_lib_path()
    lib_name = _get_lib_name(precision)
    lib_path = lib_dir / lib_name
    
    if not lib_path.exists():
        # Try finding in system paths
        try:
            lib = ctypes.CDLL(lib_name)
            _lib_cache[precision] = lib
            return lib
        except OSError:
            pass
        
        raise OSError(
            f"Cannot find SCL library: {{lib_path}}\\n"
            f"Searched in: {{lib_dir}}\\n"
            f"Set SCL_LIB_PATH environment variable to specify custom path."
        )
    
    lib = ctypes.CDLL(str(lib_path))
    _lib_cache[precision] = lib
    return lib


def get_available_precisions() -> list[str]:
    """Get list of available library precisions."""
    lib_dir = _get_lib_path()
    precisions = []
    
    for variant in ["f32_i32", "f64_i64", "f32_i64", "f64_i32"]:
        lib_name = _get_lib_name(variant)
        if (lib_dir / lib_name).exists():
            precisions.append(variant)
    
    return precisions


# Convenience: default library instance
_default_lib: Optional[ctypes.CDLL] = None

def get_default_library() -> ctypes.CDLL:
    """Get the default library (f32_i32 precision)."""
    global _default_lib
    if _default_lib is None:
        _default_lib = get_library("f32_i32")
    return _default_lib
'''

    loader_path = output_dir / "_loader.py"
    loader_path.write_text(loader_code)
    logging.info(f"Generated: {loader_path}")


def generate_init(output_dir: Path, modules: list[str]) -> None:
    """Generate __init__.py for the bindings package."""
    init_code = '''"""
SCL Python Bindings
===================

Auto-generated ctypes bindings for the SCL native library.

Usage:
    from scl._bindings import algebra, neighbors, ...
    
    # Or import specific functions
    from scl._bindings.algebra import scl_matrix_multiply
"""

from __future__ import annotations

from ._loader import get_library, get_available_precisions

'''

    # Add module imports
    for module in sorted(modules):
        init_code += f"from . import {module}\n"

    init_code += "\n__all__ = [\n"
    init_code += '    "get_library",\n'
    init_code += '    "get_available_precisions",\n'
    for module in sorted(modules):
        init_code += f'    "{module}",\n'
    init_code += "]\n"

    init_path = output_dir / "__init__.py"
    init_path.write_text(init_code)
    logging.info(f"Generated: {init_path}")


# =============================================================================
# Main Processing
# =============================================================================


class BindingPipeline:
    """Orchestrates the binding generation process."""

    def __init__(
        self,
        c_api_dir: Path,
        output_dir: Path,
        include_paths: Optional[list[Path]] = None,
        lib_name: str = DEFAULT_LIB_NAME,
    ):
        self.c_api_dir = c_api_dir
        self.output_dir = output_dir
        self.lib_name = lib_name

        # Setup include paths
        inc_paths = [str(c_api_dir.parent.parent.parent)]  # Project root
        if include_paths:
            inc_paths.extend(str(p) for p in include_paths)

        self.parser = ClangParser(include_paths=inc_paths)
        self.generator = BindingGenerator(lib_name=lib_name)

    def _get_relative_path(self, header_path: Path) -> Path:
        """Get path relative to c_api directory."""
        return header_path.relative_to(self.c_api_dir)

    def _get_output_path(self, header_path: Path) -> Path:
        """Calculate output path for a header file."""
        rel_path = self._get_relative_path(header_path)
        # Change extension from .h to .py
        py_path = rel_path.with_suffix(".py")
        return self.output_dir / py_path

    def process_header(self, header_path: Path) -> Optional[Path]:
        """Process a single header file."""
        try:
            # Parse
            parsed = self.parser.parse_header(header_path)

            # Skip if no declarations found
            if not (
                parsed.functions
                or parsed.enums
                or parsed.structs
                or parsed.typedefs
            ):
                logging.warning(f"No declarations found in: {header_path}")
                return None

            # Generate
            code = self.generator.generate_module(parsed)

            # Write output
            output_path = self._get_output_path(header_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(code)

            logging.info(f"Generated: {output_path}")
            return output_path

        except Exception as e:
            logging.error(f"Error processing {header_path}: {e}")
            raise

    def run(self) -> dict[str, Any]:
        """Run the full binding generation pipeline."""
        logging.info(f"C API Directory: {self.c_api_dir}")
        logging.info(f"Output Directory: {self.output_dir}")

        # Find all header files
        headers = list(self.c_api_dir.rglob("*.h"))
        logging.info(f"Found {len(headers)} header files")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Process each header
        generated_modules = []
        errors = []

        for header in sorted(headers):
            try:
                output = self.process_header(header)
                if output:
                    # Get module name (relative path without extension)
                    rel_path = output.relative_to(self.output_dir)
                    module_name = str(rel_path.with_suffix("")).replace("/", ".")
                    generated_modules.append(module_name)
            except Exception as e:
                errors.append((header, str(e)))

        # Generate loader
        generate_loader(self.output_dir, self.lib_name)

        # Generate __init__.py
        # Convert module paths to simple names for top-level modules
        top_modules = set()
        for m in generated_modules:
            top_name = m.split(".")[0]
            top_modules.add(top_name)

        generate_init(self.output_dir, list(top_modules))

        # Generate sub-package __init__.py files
        for subdir in self.output_dir.rglob("*"):
            if subdir.is_dir() and subdir != self.output_dir:
                sub_init = subdir / "__init__.py"
                if not sub_init.exists():
                    # Find all .py files in this directory
                    py_files = [
                        f.stem for f in subdir.glob("*.py") if f.stem != "__init__"
                    ]
                    if py_files:
                        init_content = f'"""Auto-generated sub-package."""\n\n'
                        for py_file in sorted(py_files):
                            init_content += f"from .{py_file} import *\n"
                        sub_init.write_text(init_content)
                        logging.info(f"Generated: {sub_init}")

        # Summary
        result = {
            "total_headers": len(headers),
            "generated_modules": len(generated_modules),
            "errors": len(errors),
            "modules": generated_modules,
            "error_details": errors,
        }

        return result


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate Python ctypes bindings from C API headers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate bindings with defaults
    python generate_bindings.py
    
    # Specify custom paths
    python generate_bindings.py -c ./scl/binding/c_api -o ./python/scl/_bindings
    
    # Verbose output
    python generate_bindings.py -v
    
    # Generate JSON report
    python generate_bindings.py --json-report report.json
""",
    )

    parser.add_argument(
        "-c",
        "--c-api-dir",
        type=Path,
        default=Path(DEFAULT_C_API_DIR),
        help=f"Path to C API headers (default: {DEFAULT_C_API_DIR})",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for Python bindings (default: {DEFAULT_OUTPUT_DIR})",
    )

    parser.add_argument(
        "-I",
        "--include",
        type=Path,
        action="append",
        default=[],
        help="Additional include paths for clang",
    )

    parser.add_argument(
        "--lib-name",
        type=str,
        default=DEFAULT_LIB_NAME,
        help=f"Base name of the native library (default: {DEFAULT_LIB_NAME})",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--json-report",
        type=Path,
        help="Write JSON report to file",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse headers but don't write output files",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    # Validate inputs
    if not args.c_api_dir.exists():
        logging.error(f"C API directory not found: {args.c_api_dir}")
        sys.exit(1)

    # Run pipeline
    pipeline = BindingPipeline(
        c_api_dir=args.c_api_dir,
        output_dir=args.output_dir,
        include_paths=args.include,
        lib_name=args.lib_name,
    )

    if args.dry_run:
        logging.info("Dry run mode - no files will be written")
        # Just parse headers
        headers = list(args.c_api_dir.rglob("*.h"))
        for header in headers:
            try:
                parsed = pipeline.parser.parse_header(header)
                logging.info(
                    f"  {header.name}: "
                    f"{len(parsed.functions)} functions, "
                    f"{len(parsed.enums)} enums"
                )
            except Exception as e:
                logging.error(f"  {header.name}: {e}")
        sys.exit(0)

    result = pipeline.run()

    # Print summary
    print("\n" + "=" * 60)
    print("Generation Summary")
    print("=" * 60)
    print(f"  Headers processed: {result['total_headers']}")
    print(f"  Modules generated: {result['generated_modules']}")
    print(f"  Errors: {result['errors']}")

    if result["errors"] > 0:
        print("\nErrors:")
        for header, error in result["error_details"]:
            print(f"  - {header}: {error}")

    # Write JSON report
    if args.json_report:
        with open(args.json_report, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nJSON report written to: {args.json_report}")

    print("=" * 60)

    sys.exit(0 if result["errors"] == 0 else 1)


if __name__ == "__main__":
    main()