"""
Base classes for code generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from jinja2 import Environment, PackageLoader, select_autoescape

from ..parser.c_types import ParsedHeader
from ..config import CodegenConfig


@dataclass
class GeneratedFile:
    """Represents a generated file."""
    path: Path
    content: str
    source: Optional[Path] = None  # Original source file


class Generator(ABC):
    """
    Abstract base class for code generators.

    Subclasses implement specific generation logic for different
    output formats (Python bindings, documentation, etc.).
    """

    def __init__(self, config: CodegenConfig):
        """
        Initialize the generator.

        Args:
            config: Codegen configuration
        """
        self.config = config
        self._env: Optional[Environment] = None

    @property
    def env(self) -> Environment:
        """Lazy-load Jinja2 environment."""
        if self._env is None:
            self._env = self._create_jinja_env()
        return self._env

    def _create_jinja_env(self) -> Environment:
        """Create and configure Jinja2 environment."""
        try:
            env = Environment(
                loader=PackageLoader("codegen", "templates"),
                autoescape=select_autoescape(default=False),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True,
            )
        except Exception:
            # Fallback: try loading from file system
            from jinja2 import FileSystemLoader
            templates_dir = Path(__file__).parent.parent / "templates"
            env = Environment(
                loader=FileSystemLoader(str(templates_dir)),
                autoescape=select_autoescape(default=False),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True,
            )

        # Add custom filters
        env.filters["snake_case"] = self._to_snake_case
        env.filters["pascal_case"] = self._to_pascal_case
        env.filters["upper_case"] = lambda s: s.upper()

        return env

    @abstractmethod
    def generate(self, parsed: ParsedHeader) -> GeneratedFile:
        """
        Generate output from a parsed header.

        Args:
            parsed: Parsed header content

        Returns:
            Generated file content
        """
        pass

    @abstractmethod
    def get_output_path(self, parsed: ParsedHeader) -> Path:
        """
        Determine output path for a parsed header.

        Args:
            parsed: Parsed header content

        Returns:
            Output file path
        """
        pass

    def generate_all(self, headers: list[ParsedHeader]) -> list[GeneratedFile]:
        """
        Generate output for multiple headers.

        Args:
            headers: List of parsed headers

        Returns:
            List of generated files
        """
        return [self.generate(h) for h in headers if h.has_content]

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert name to snake_case."""
        import re
        # Insert underscore before uppercase letters
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    @staticmethod
    def _to_pascal_case(name: str) -> str:
        """Convert name to PascalCase."""
        return "".join(word.capitalize() for word in name.split("_"))
