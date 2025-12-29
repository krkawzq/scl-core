"""
C type AST data structures.

These dataclasses represent parsed C constructs (functions, structs, enums)
in a language-agnostic way for code generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SourceLocation:
    """Source code location for error reporting."""
    file: Path
    line: int
    column: int

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


@dataclass
class CType:
    """
    Representation of a C type.

    Handles basic types, pointers, qualifiers, and type modifiers.
    """
    name: str
    is_pointer: bool = False
    pointer_depth: int = 0
    is_const: bool = False
    is_volatile: bool = False
    is_struct: bool = False
    is_enum: bool = False

    def __str__(self) -> str:
        """Return the C type string representation."""
        parts = []
        if self.is_const:
            parts.append("const")
        if self.is_volatile:
            parts.append("volatile")
        if self.is_struct:
            parts.append("struct")
        if self.is_enum:
            parts.append("enum")
        parts.append(self.name)
        result = " ".join(parts)
        result += "*" * self.pointer_depth
        return result

    @property
    def base_type(self) -> str:
        """Get the base type name without qualifiers."""
        return self.name


@dataclass
class CParameter:
    """Function parameter."""
    name: str
    type: CType
    doc: Optional[str] = None

    # Parameter direction annotation from doc comments
    # [in], [out], [in,out]
    direction: str = "in"


@dataclass
class CFunction:
    """C function declaration."""
    name: str
    return_type: CType
    parameters: list[CParameter] = field(default_factory=list)
    doc: Optional[str] = None
    location: Optional[SourceLocation] = None

    # Whether this function is marked with SCL_EXPORT
    is_exported: bool = True

    @property
    def signature(self) -> str:
        """Generate C function signature string."""
        params = ", ".join(
            f"{p.type} {p.name}" for p in self.parameters
        )
        return f"{self.return_type} {self.name}({params})"


@dataclass
class CField:
    """Struct or union field."""
    name: str
    type: CType
    doc: Optional[str] = None
    bit_width: Optional[int] = None  # For bit fields


@dataclass
class CStruct:
    """C struct definition."""
    name: str
    fields: list[CField] = field(default_factory=list)
    doc: Optional[str] = None
    location: Optional[SourceLocation] = None

    # Whether this is an opaque/forward-declared struct
    is_opaque: bool = False


@dataclass
class CEnumValue:
    """Enum constant."""
    name: str
    value: Optional[int] = None
    doc: Optional[str] = None


@dataclass
class CEnum:
    """C enum definition."""
    name: str
    values: list[CEnumValue] = field(default_factory=list)
    doc: Optional[str] = None
    location: Optional[SourceLocation] = None


@dataclass
class CTypedef:
    """C typedef declaration."""
    name: str
    target_type: CType
    doc: Optional[str] = None
    location: Optional[SourceLocation] = None


@dataclass
class ParsedHeader:
    """
    Complete parsed representation of a C header file.

    Contains all exported symbols: functions, structs, enums, and typedefs.
    """
    path: Path
    functions: list[CFunction] = field(default_factory=list)
    structs: list[CStruct] = field(default_factory=list)
    enums: list[CEnum] = field(default_factory=list)
    typedefs: list[CTypedef] = field(default_factory=list)

    # Raw doc comment at file level
    file_doc: Optional[str] = None

    # Include dependencies
    includes: list[Path] = field(default_factory=list)

    @property
    def module_name(self) -> str:
        """Derive module name from file path."""
        return self.path.stem

    @property
    def has_content(self) -> bool:
        """Check if header has any exported content."""
        return bool(
            self.functions or self.structs or self.enums or self.typedefs
        )

    def get_all_types(self) -> set[str]:
        """Get all type names defined in this header."""
        types = set()
        for s in self.structs:
            types.add(s.name)
        for e in self.enums:
            types.add(e.name)
        for t in self.typedefs:
            types.add(t.name)
        return types
