"""
Clang-based C header parser.

Uses libclang to parse C headers and extract function declarations,
struct definitions, enums, and typedefs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Iterator

try:
    from clang.cindex import (
        Index,
        TranslationUnit,
        Cursor,
        CursorKind,
        TypeKind,
        Config,
    )
    HAS_CLANG = True
except ImportError:
    HAS_CLANG = False

from .c_types import (
    CType,
    CFunction,
    CParameter,
    CStruct,
    CField,
    CEnum,
    CEnumValue,
    CTypedef,
    ParsedHeader,
    SourceLocation,
)


class ClangParser:
    """
    Parser for C headers using libclang.

    Extracts:
    - Function declarations (extern "C" or in global scope)
    - Struct definitions (including opaque forward declarations)
    - Enum definitions
    - Typedef declarations
    """

    def __init__(
        self,
        include_dirs: Optional[list[Path]] = None,
        defines: Optional[dict[str, str]] = None,
        clang_args: Optional[list[str]] = None,
    ):
        """
        Initialize the parser.

        Args:
            include_dirs: Additional include directories
            defines: Preprocessor macro definitions
            clang_args: Extra arguments to pass to clang
        """
        if not HAS_CLANG:
            raise ImportError(
                "libclang is required for parsing. "
                "Install with: pip install libclang"
            )

        self.include_dirs = include_dirs or []
        self.defines = defines or {}
        self.clang_args = clang_args or []

        self._index = Index.create()

    def _build_args(self) -> list[str]:
        """Build clang argument list."""
        args = ["-x", "c", "-std=c99"]

        # Add include directories
        for inc in self.include_dirs:
            args.append(f"-I{inc}")

        # Add defines
        for name, value in self.defines.items():
            if value:
                args.append(f"-D{name}={value}")
            else:
                args.append(f"-D{name}")

        # Add extra args
        args.extend(self.clang_args)

        return args

    def parse(self, header_path: Path) -> ParsedHeader:
        """
        Parse a single C header file.

        Args:
            header_path: Path to the header file

        Returns:
            ParsedHeader containing all extracted declarations
        """
        args = self._build_args()

        tu = self._index.parse(
            str(header_path),
            args=args,
            options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )

        if tu is None:
            raise RuntimeError(f"Failed to parse {header_path}")

        # Check for errors
        errors = [d for d in tu.diagnostics if d.severity >= 3]
        if errors:
            error_msgs = "\n".join(str(e) for e in errors[:5])
            raise RuntimeError(f"Parse errors in {header_path}:\n{error_msgs}")

        result = ParsedHeader(path=header_path)

        # Walk the AST
        self._process_cursor(tu.cursor, result, header_path)

        return result

    def _process_cursor(
        self,
        cursor: Cursor,
        result: ParsedHeader,
        source_file: Path,
    ) -> None:
        """Process a cursor and its children."""
        for child in cursor.get_children():
            # Only process items from the main file
            if child.location.file is None:
                continue
            child_file = Path(child.location.file.name)
            if child_file != source_file:
                continue

            kind = child.kind

            if kind == CursorKind.FUNCTION_DECL:
                func = self._parse_function(child)
                if func is not None:
                    result.functions.append(func)

            elif kind == CursorKind.STRUCT_DECL:
                struct = self._parse_struct(child)
                if struct is not None:
                    result.structs.append(struct)

            elif kind == CursorKind.ENUM_DECL:
                enum = self._parse_enum(child)
                if enum is not None:
                    result.enums.append(enum)

            elif kind == CursorKind.TYPEDEF_DECL:
                typedef = self._parse_typedef(child)
                if typedef is not None:
                    result.typedefs.append(typedef)

    def _parse_function(self, cursor: Cursor) -> Optional[CFunction]:
        """Parse a function declaration."""
        name = cursor.spelling

        # Skip unnamed functions
        if not name:
            return None

        # Get return type
        return_type = self._parse_type(cursor.result_type)

        # Get parameters
        params = []
        for arg in cursor.get_arguments():
            param = CParameter(
                name=arg.spelling or f"arg{len(params)}",
                type=self._parse_type(arg.type),
            )
            params.append(param)

        # Get doc comment
        doc = cursor.brief_comment or cursor.raw_comment
        if doc:
            doc = self._clean_doc(doc)

        return CFunction(
            name=name,
            return_type=return_type,
            parameters=params,
            doc=doc,
            location=self._get_location(cursor),
        )

    def _parse_struct(self, cursor: Cursor) -> Optional[CStruct]:
        """Parse a struct declaration."""
        name = cursor.spelling

        # Skip anonymous structs
        if not name:
            return None

        # Check if this is a forward declaration (no definition)
        is_opaque = not cursor.is_definition()

        fields = []
        if not is_opaque:
            for child in cursor.get_children():
                if child.kind == CursorKind.FIELD_DECL:
                    field = CField(
                        name=child.spelling,
                        type=self._parse_type(child.type),
                        doc=child.brief_comment,
                    )
                    fields.append(field)

        doc = cursor.brief_comment or cursor.raw_comment
        if doc:
            doc = self._clean_doc(doc)

        return CStruct(
            name=name,
            fields=fields,
            is_opaque=is_opaque,
            doc=doc,
            location=self._get_location(cursor),
        )

    def _parse_enum(self, cursor: Cursor) -> Optional[CEnum]:
        """Parse an enum declaration."""
        name = cursor.spelling

        # Allow anonymous enums (they may have typedef'd names)
        values = []
        for child in cursor.get_children():
            if child.kind == CursorKind.ENUM_CONSTANT_DECL:
                value = CEnumValue(
                    name=child.spelling,
                    value=child.enum_value,
                    doc=child.brief_comment,
                )
                values.append(value)

        doc = cursor.brief_comment or cursor.raw_comment
        if doc:
            doc = self._clean_doc(doc)

        return CEnum(
            name=name or "",
            values=values,
            doc=doc,
            location=self._get_location(cursor),
        )

    def _parse_typedef(self, cursor: Cursor) -> Optional[CTypedef]:
        """Parse a typedef declaration."""
        name = cursor.spelling

        if not name:
            return None

        # Get the underlying type
        underlying = cursor.underlying_typedef_type
        target_type = self._parse_type(underlying)

        doc = cursor.brief_comment
        if doc:
            doc = self._clean_doc(doc)

        return CTypedef(
            name=name,
            target_type=target_type,
            doc=doc,
            location=self._get_location(cursor),
        )

    def _parse_type(self, clang_type) -> CType:
        """Convert a clang type to our CType representation."""
        type_name = clang_type.spelling

        # Determine pointer depth
        pointer_depth = 0
        current = clang_type
        while current.kind == TypeKind.POINTER:
            pointer_depth += 1
            current = current.get_pointee()

        # Get base type name
        base_name = current.spelling

        # Check for const qualifier
        is_const = current.is_const_qualified()

        # Check if it's a struct or enum
        is_struct = "struct " in base_name
        is_enum = "enum " in base_name

        # Clean up the name
        base_name = base_name.replace("struct ", "").replace("enum ", "")
        base_name = base_name.replace("const ", "").strip()

        return CType(
            name=base_name,
            is_pointer=pointer_depth > 0,
            pointer_depth=pointer_depth,
            is_const=is_const,
            is_struct=is_struct,
            is_enum=is_enum,
        )

    def _get_location(self, cursor: Cursor) -> Optional[SourceLocation]:
        """Get source location from cursor."""
        loc = cursor.location
        if loc.file is None:
            return None
        return SourceLocation(
            file=Path(loc.file.name),
            line=loc.line,
            column=loc.column,
        )

    def _clean_doc(self, doc: str) -> str:
        """Clean up documentation string."""
        if not doc:
            return ""

        # Remove C comment markers
        doc = re.sub(r"^/\*\*?", "", doc)
        doc = re.sub(r"\*/$", "", doc)
        doc = re.sub(r"^///?", "", doc, flags=re.MULTILINE)

        # Remove leading asterisks from each line
        lines = []
        for line in doc.split("\n"):
            line = re.sub(r"^\s*\*\s?", "", line)
            lines.append(line)

        doc = "\n".join(lines).strip()

        return doc

    def parse_headers(self, header_paths: list[Path]) -> list[ParsedHeader]:
        """
        Parse multiple header files.

        Args:
            header_paths: List of header file paths

        Returns:
            List of ParsedHeader results
        """
        return [self.parse(p) for p in header_paths]
