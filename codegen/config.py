"""
Configuration system for codegen.

Supports:
- TOML configuration files
- CLI argument overrides
- Environment variable fallbacks
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


@dataclass
class PathsConfig:
    """Path configuration."""

    c_api_dir: Path = field(default_factory=lambda: Path("scl/binding/c_api"))
    python_output: Path = field(default_factory=lambda: Path("python/scl/_bindings"))
    docs_output: Path = field(default_factory=lambda: Path("docs/api/c-api"))
    core_headers: list[str] = field(default_factory=lambda: ["core/core.h"])


@dataclass
class TypesConfig:
    """Type configuration matching compile-time settings."""

    real_type: str = "float64"   # float32 | float64
    index_type: str = "int64"    # int16 | int32 | int64


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
    types: TypesConfig = field(default_factory=TypesConfig)
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
        types_data = data.get("types", {})
        gen_data = data.get("generation", {})

        paths = PathsConfig(
            c_api_dir=Path(paths_data.get("c_api_dir", "scl/binding/c_api")),
            python_output=Path(paths_data.get("python_output", "python/scl/_bindings")),
            docs_output=Path(paths_data.get("docs_output", "docs/api/c-api")),
            core_headers=paths_data.get("core_headers", ["core/core.h"]),
        )

        types = TypesConfig(
            real_type=types_data.get("real_type", "float64"),
            index_type=types_data.get("index_type", "int64"),
        )

        generation = GenerationConfig(
            overwrite=gen_data.get("overwrite", False),
            generate_docstrings=gen_data.get("generate_docstrings", True),
            generate_type_hints=gen_data.get("generate_type_hints", True),
        )

        return cls(
            project_root=base_path,
            paths=paths,
            types=types,
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
