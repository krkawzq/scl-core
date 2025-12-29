"""
Configuration system for codegen.

Supports:
- TOML configuration files
- CLI argument overrides
- Environment variable fallbacks
- Multi-variant library configuration
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


# =============================================================================
# Type Variants Configuration
# =============================================================================

# Supported real types: float16, float32, float64
REAL_TYPES = ["float16", "float32", "float64"]

# Supported index types: int16, int32, int64
INDEX_TYPES = ["int16", "int32", "int64"]

# Mapping from type name to ctypes type
CTYPES_REAL_MAP = {
    "float16": "c_uint16",  # No native float16 in ctypes, use uint16 for storage
    "float32": "c_float",
    "float64": "c_double",
}

CTYPES_INDEX_MAP = {
    "int16": "c_int16",
    "int32": "c_int32",
    "int64": "c_int64",
}

# Mapping from type name to numpy dtype
NUMPY_REAL_MAP = {
    "float16": "np.float16",
    "float32": "np.float32",
    "float64": "np.float64",
}

NUMPY_INDEX_MAP = {
    "int16": "np.int16",
    "int32": "np.int32",
    "int64": "np.int64",
}


@dataclass
class LibraryVariant:
    """Represents a single library variant."""
    
    real_type: str  # float16 | float32 | float64
    index_type: str  # int16 | int32 | int64
    
    @property
    def suffix(self) -> str:
        """Get library suffix (e.g., 'f64_i64')."""
        real_suffix = self.real_type.replace("float", "f")
        index_suffix = self.index_type.replace("int", "i")
        return f"{real_suffix}_{index_suffix}"
    
    @property
    def ctypes_real(self) -> str:
        """Get ctypes type for real values."""
        return CTYPES_REAL_MAP[self.real_type]
    
    @property
    def ctypes_index(self) -> str:
        """Get ctypes type for index values."""
        return CTYPES_INDEX_MAP[self.index_type]
    
    @property
    def numpy_real(self) -> str:
        """Get numpy dtype for real values."""
        return NUMPY_REAL_MAP[self.real_type]
    
    @property
    def numpy_index(self) -> str:
        """Get numpy dtype for index values."""
        return NUMPY_INDEX_MAP[self.index_type]
    
    def __hash__(self):
        return hash((self.real_type, self.index_type))
    
    def __eq__(self, other):
        if not isinstance(other, LibraryVariant):
            return False
        return self.real_type == other.real_type and self.index_type == other.index_type


@dataclass
class PathsConfig:
    """Path configuration."""

    c_api_dir: Path = field(default_factory=lambda: Path("scl/binding/c_api"))
    python_output: Path = field(default_factory=lambda: Path("python/scl/_bindings"))
    docs_output: Path = field(default_factory=lambda: Path("docs/api/c-api"))
    libs_dir: Path = field(default_factory=lambda: Path("python/scl/libs"))
    core_headers: list[str] = field(default_factory=lambda: ["core/core.h"])


@dataclass
class VariantsConfig:
    """Library variants configuration."""
    
    # List of enabled variants
    variants: list[LibraryVariant] = field(default_factory=lambda: [
        LibraryVariant("float32", "int32"),
        LibraryVariant("float64", "int64"),
        LibraryVariant("float32", "int64"),
        LibraryVariant("float64", "int32"),
    ])
    
    # Default variant to use
    default_variant: str = "f64_i64"
    
    @property
    def variant_suffixes(self) -> list[str]:
        """Get list of variant suffixes."""
        return [v.suffix for v in self.variants]
    
    def get_variant(self, suffix: str) -> Optional[LibraryVariant]:
        """Get variant by suffix."""
        for v in self.variants:
            if v.suffix == suffix:
                return v
        return None


@dataclass
class GenerationConfig:
    """Generation options."""

    overwrite: bool = False
    generate_docstrings: bool = True
    generate_type_hints: bool = True


@dataclass
class CodegenConfig:
    """Main configuration container."""

    project_root: Path = field(default_factory=Path.cwd)
    paths: PathsConfig = field(default_factory=PathsConfig)
    variants: VariantsConfig = field(default_factory=VariantsConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def __post_init__(self):
        """Ensure project_root is a Path."""
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)

    @classmethod
    def from_file(cls, path: Path) -> "CodegenConfig":
        """Load configuration from a TOML file."""
        if tomllib is None:
            raise ImportError(
                "tomli is required for Python < 3.11. "
                "Install with: pip install tomli"
            )

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls._from_dict(data, path.parent)

    @classmethod
    def _from_dict(cls, data: dict, base_path: Path) -> "CodegenConfig":
        """Create config from dictionary."""
        paths_data = data.get("paths", {})
        variants_data = data.get("variants", {})
        gen_data = data.get("generation", {})

        paths = PathsConfig(
            c_api_dir=Path(paths_data.get("c_api_dir", "scl/binding/c_api")),
            python_output=Path(paths_data.get("python_output", "python/scl/_bindings")),
            docs_output=Path(paths_data.get("docs_output", "docs/api/c-api")),
            libs_dir=Path(paths_data.get("libs_dir", "python/scl/libs")),
            core_headers=paths_data.get("core_headers", ["core/core.h"]),
        )

        # Parse variants
        variant_list = []
        for v_str in variants_data.get("enabled", ["f32_i32", "f64_i64", "f32_i64", "f64_i32"]):
            # Parse suffix like "f32_i64" to real_type and index_type
            parts = v_str.split("_")
            if len(parts) == 2:
                real_type = parts[0].replace("f", "float")
                index_type = parts[1].replace("i", "int")
                variant_list.append(LibraryVariant(real_type, index_type))
        
        variants = VariantsConfig(
            variants=variant_list if variant_list else [
                LibraryVariant("float32", "int32"),
                LibraryVariant("float64", "int64"),
                LibraryVariant("float32", "int64"),
                LibraryVariant("float64", "int32"),
            ],
            default_variant=variants_data.get("default", "f64_i64"),
        )

        generation = GenerationConfig(
            overwrite=gen_data.get("overwrite", False),
            generate_docstrings=gen_data.get("generate_docstrings", True),
            generate_type_hints=gen_data.get("generate_type_hints", True),
        )

        return cls(
            project_root=base_path,
            paths=paths,
            variants=variants,
            generation=generation,
        )

    @classmethod
    def find_config(cls, start_path: Optional[Path] = None) -> Optional[Path]:
        """Find codegen.toml in current or parent directories."""
        if start_path is None:
            start_path = Path.cwd()

        current = start_path.resolve()

        for _ in range(10):  # Max 10 levels up
            config_path = current / "codegen.toml"
            if config_path.exists():
                return config_path

            parent = current.parent
            if parent == current:
                break
            current = parent

        return None

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "CodegenConfig":
        """Load configuration, auto-discovering if path not provided."""
        if config_path is None:
            config_path = cls.find_config()

        if config_path is not None and config_path.exists():
            return cls.from_file(config_path)

        # Return default config with current directory as root
        return cls(project_root=Path.cwd())

    def resolve_path(self, path: Path) -> Path:
        """Resolve a relative path against project root."""
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def c_api_dir_abs(self) -> Path:
        """Absolute path to C API directory."""
        return self.resolve_path(self.paths.c_api_dir)

    @property
    def python_output_abs(self) -> Path:
        """Absolute path to Python output directory."""
        return self.resolve_path(self.paths.python_output)

    @property
    def docs_output_abs(self) -> Path:
        """Absolute path to docs output directory."""
        return self.resolve_path(self.paths.docs_output)
    
    @property
    def libs_dir_abs(self) -> Path:
        """Absolute path to libs directory."""
        return self.resolve_path(self.paths.libs_dir)
