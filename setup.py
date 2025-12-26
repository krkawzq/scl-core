"""
Setup script for scl-core

This setup.py is primarily for compatibility. The main configuration is in pyproject.toml.
However, this script handles:
1. Copying compiled libraries from build/lib/ to src/scl/libs/ before packaging
2. Ensuring both f32 and f64 libraries are included
"""

import os
import sys
import shutil
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info


class BuildPyWithLibs(build_py):
    """Custom build command that copies compiled libraries into the package."""
    
    def run(self):
        # First run the standard build
        super().run()
        
        # Copy libraries from build/lib/ to src/scl/libs/
        build_dir = Path("build/lib")
        target_dir = Path("src/scl/libs")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Library names to look for
        lib_names = ["scl_core_f32", "scl_core_f64"]
        
        # Platform-specific extensions
        if os.name == 'nt':  # Windows
            extensions = [".dll"]
        elif sys.platform == 'darwin':  # macOS
            extensions = [".dylib"]
        else:  # Linux
            extensions = [".so"]
        
        copied = False
        for lib_name in lib_names:
            for ext in extensions:
                src = build_dir / f"lib{lib_name}{ext}" if ext != ".dll" else build_dir / f"{lib_name}{ext}"
                if src.exists():
                    # Copy without lib prefix on Windows
                    if ext == ".dll":
                        dst = target_dir / f"{lib_name}{ext}"
                    else:
                        # Keep lib prefix for .so and .dylib
                        dst = target_dir / f"lib{lib_name}{ext}"
                    
                    shutil.copy2(src, dst)
                    print(f"Copied {src} -> {dst}")
                    copied = True
        
        if not copied:
            print("Warning: No compiled libraries found in build/lib/")
            print("Make sure to run 'make build' or 'cmake --build build' first.")


class EggInfoWithLibs(egg_info):
    """Custom egg_info that includes library files."""
    
    def run(self):
        # Ensure libs directory is recognized
        libs_dir = Path("src/scl/libs")
        if libs_dir.exists():
            # Add library files to package data
            if not hasattr(self, 'distribution') or not hasattr(self.distribution, 'package_data'):
                return
            super().run()


# Read version from src/__init__.py
def get_version():
    version_file = Path("src/__init__.py")
    if version_file.exists():
        for line in version_file.read_text().splitlines():
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"


# Read long description from README
def get_long_description():
    readme = Path("README.md")
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return ""


# Configuration is primarily in pyproject.toml
# This setup.py mainly handles library copying
setup(
    name="scl-core",
    version=get_version(),
    description="High-performance single-cell analysis library",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"scl": "src"},
    package_data={
        "scl": [
            "libs/*.so",
            "libs/*.dylib",
            "libs/*.dll",
        ],
    },
    include_package_data=True,
    cmdclass={
        "build_py": BuildPyWithLibs,
        "egg_info": EggInfoWithLibs,
    },
    zip_safe=False,  # Cannot be zipped due to shared libraries
)

