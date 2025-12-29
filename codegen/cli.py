"""
Command-line interface for codegen.

Usage:
    python -m codegen <command> [options]

Commands:
    python-bindings    Generate Python ctypes bindings
    c-api-docs         Generate C API documentation skeletons
    all                Generate all outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import CodegenConfig
from .parser import ClangParser, ParsedHeader
from .generators import PythonBindingGenerator, CApiDocsGenerator


def find_headers(
    c_api_dir: Path,
    pattern: str = "*.h",
    exclude_dirs: Optional[set[str]] = None,
) -> list[Path]:
    """Find all header files in C API directory."""
    if exclude_dirs is None:
        exclude_dirs = {"__pycache__", ".git"}

    headers = []
    for path in c_api_dir.rglob(pattern):
        # Skip excluded directories
        if any(excl in path.parts for excl in exclude_dirs):
            continue
        headers.append(path)

    return sorted(headers)


def generate_python_bindings(
    config: CodegenConfig,
    input_path: Optional[Path] = None,
    verbose: bool = False,
) -> int:
    """Generate Python bindings."""
    # Find headers
    if input_path:
        if input_path.is_file():
            headers = [input_path]
        else:
            headers = find_headers(input_path)
    else:
        headers = find_headers(config.c_api_dir_abs)

    if not headers:
        print("No header files found.")
        return 1

    if verbose:
        print(f"Found {len(headers)} header files")

    # Parse headers
    parser = ClangParser(
        include_dirs=[config.project_root],
    )

    generator = PythonBindingGenerator(config)

    success_count = 0
    error_count = 0

    for header in headers:
        try:
            if verbose:
                print(f"Parsing: {header}")

            parsed = parser.parse(header)

            if not parsed.has_content:
                if verbose:
                    print(f"  Skipping (no content)")
                continue

            result = generator.generate(parsed)

            # Create output directory
            result.path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists and overwrite is disabled
            if result.path.exists() and not config.generation.overwrite:
                if verbose:
                    print(f"  Skipping (exists): {result.path}")
                continue

            # Write file
            result.path.write_text(result.content)

            if verbose:
                print(f"  Generated: {result.path}")

            success_count += 1

        except Exception as e:
            print(f"Error processing {header}: {e}", file=sys.stderr)
            error_count += 1

    print(f"\nGenerated {success_count} files, {error_count} errors")
    return 0 if error_count == 0 else 1


def generate_c_api_docs(
    config: CodegenConfig,
    input_path: Optional[Path] = None,
    verbose: bool = False,
) -> int:
    """Generate C API documentation skeletons."""
    # Find headers
    if input_path:
        if input_path.is_file():
            headers = [input_path]
        else:
            headers = find_headers(input_path)
    else:
        headers = find_headers(config.c_api_dir_abs)

    if not headers:
        print("No header files found.")
        return 1

    if verbose:
        print(f"Found {len(headers)} header files")

    # Parse headers
    parser = ClangParser(
        include_dirs=[config.project_root],
    )

    generator = CApiDocsGenerator(config)

    success_count = 0
    error_count = 0

    for header in headers:
        try:
            if verbose:
                print(f"Parsing: {header}")

            parsed = parser.parse(header)

            if not parsed.has_content:
                if verbose:
                    print(f"  Skipping (no content)")
                continue

            result = generator.generate(parsed)

            # Create output directory
            result.path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists and overwrite is disabled
            if result.path.exists() and not config.generation.overwrite:
                if verbose:
                    print(f"  Skipping (exists): {result.path}")
                continue

            # Write file
            result.path.write_text(result.content)

            if verbose:
                print(f"  Generated: {result.path}")

            success_count += 1

        except Exception as e:
            print(f"Error processing {header}: {e}", file=sys.stderr)
            error_count += 1

    print(f"\nGenerated {success_count} files, {error_count} errors")
    return 0 if error_count == 0 else 1


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="codegen",
        description="SCL-Core Code Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all Python bindings
  python -m codegen python-bindings

  # Generate bindings for a single file
  python -m codegen python-bindings -i scl/binding/c_api/hvg.h

  # Generate C API documentation
  python -m codegen c-api-docs

  # Use custom config file
  python -m codegen --config codegen.toml all
""",
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration file (codegen.toml)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # python-bindings subcommand
    py_parser = subparsers.add_parser(
        "python-bindings",
        help="Generate Python ctypes bindings",
    )
    py_parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input header file or directory",
    )
    py_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory",
    )

    # c-api-docs subcommand
    docs_parser = subparsers.add_parser(
        "c-api-docs",
        help="Generate C API documentation skeletons",
    )
    docs_parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input header file or directory",
    )
    docs_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory",
    )

    # all subcommand
    all_parser = subparsers.add_parser(
        "all",
        help="Generate all outputs",
    )

    args = parser.parse_args(argv)

    # Load configuration
    config = CodegenConfig.load(args.config)

    # Apply CLI overrides
    if args.overwrite:
        config.generation.overwrite = True

    # Handle output path overrides
    if hasattr(args, "output") and args.output:
        if args.command == "python-bindings":
            config.paths.python_output = args.output
        elif args.command == "c-api-docs":
            config.paths.docs_output = args.output

    # Execute command
    if args.command == "python-bindings":
        return generate_python_bindings(
            config,
            input_path=args.input,
            verbose=args.verbose,
        )
    elif args.command == "c-api-docs":
        return generate_c_api_docs(
            config,
            input_path=args.input,
            verbose=args.verbose,
        )
    elif args.command == "all":
        ret1 = generate_python_bindings(config, verbose=args.verbose)
        ret2 = generate_c_api_docs(config, verbose=args.verbose)
        return max(ret1, ret2)

    return 1


if __name__ == "__main__":
    sys.exit(main())
