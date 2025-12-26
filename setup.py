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
        
        # Copy libraries from build/ or build/lib/ to src/scl/libs/
        # Try multiple possible locations
        possible_build_dirs = [
            Path("build/lib"),  # If CMake outputs to build/lib/
            Path("build"),      # If CMake outputs directly to build/
        ]
        
        target_dir = Path("src/scl/libs")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Library names (without lib prefix, as set in CMakeLists OUTPUT_NAME)
        lib_names = ["scl_core_f32", "scl_core_f64"]
        
        # Platform-specific extensions
        if os.name == 'nt':  # Windows
            extensions = [".dll"]
        elif sys.platform == 'darwin':  # macOS
            extensions = [".dylib"]
        else:  # Linux
            extensions = [".so"]
        
        copied_count = 0
        found_build_dir = None
        
        # Try to find libraries in any of the possible build directories
        for build_dir in possible_build_dirs:
            if not build_dir.exists():
                continue
            
            # Copy all libraries found in this build directory
            for lib_name in lib_names:
                for ext in extensions:
                    # Try with lib prefix first (standard Unix convention)
                    src_with_prefix = build_dir / f"lib{lib_name}{ext}"
                    # Try without prefix (CMake OUTPUT_NAME may remove it)
                    src_without_prefix = build_dir / f"{lib_name}{ext}"
                    
                    src = None
                    if src_with_prefix.exists():
                        src = src_with_prefix
                    elif src_without_prefix.exists():
                        src = src_without_prefix
                    
                    if src is not None:
                        found_build_dir = build_dir
                        # Use the same filename as source (preserve naming convention)
                        # lib_loader.py will look for lib{lib_name}{ext}
                        # So we add lib prefix when copying (standard Unix convention)
                        if ext == ".dll":
                            # Windows doesn't use lib prefix
                            dst = target_dir / f"{lib_name}{ext}"
                        else:
                            # Unix systems: add lib prefix if source doesn't have it
                            if src.name.startswith("lib"):
                                dst = target_dir / src.name
                            else:
                                dst = target_dir / f"lib{lib_name}{ext}"
                        
                        shutil.copy2(src, dst)
                        print(f"✓ Copied {src.name} -> {dst}")
                        copied_count += 1
                        # Don't break here - continue to copy all libraries
                        break  # Only break from ext loop, continue to next lib_name
                # Continue to next lib_name (don't break)
            
            # If we found a build directory with libraries, we're done searching
            if copied_count > 0:
                break
        
        if copied_count == 0:
            print("=" * 60)
            print("WARNING: No compiled libraries found!")
            print("=" * 60)
            print("Searched in:")
            for build_dir in possible_build_dirs:
                print(f"  - {build_dir.absolute()}")
            print("\nPlease build the libraries first:")
            print("  mkdir -p build && cd build")
            print("  cmake ..")
            print("  cmake --build .")
            print("=" * 60)
        elif copied_count < len(lib_names):
            print(f"WARNING: Only copied {copied_count}/{len(lib_names)} libraries")
            print(f"Expected: {', '.join(lib_names)}")


class EggInfoWithLibs(egg_info):
    """Custom egg_info that includes library files."""
    
    def run(self):
        # Always call super().run() to ensure egg_info is created
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
# Note: Most configuration should be in pyproject.toml to avoid conflicts
setup(
    cmdclass={
        "build_py": BuildPyWithLibs,
        "egg_info": EggInfoWithLibs,
    },
    zip_safe=False,  # Cannot be zipped due to shared libraries
)

