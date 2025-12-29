#!/bin/bash
# Fix single file workflow script
# Usage: ./scripts/fix_single_file.sh <source_file>

set -e

SOURCE_FILE="$1"

if [ -z "$SOURCE_FILE" ]; then
    echo "Usage: $0 <source_file>"
    echo "Example: $0 scl/binding/c_api/spatial.cpp"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "========================================="
echo "Fixing: $SOURCE_FILE"
echo "========================================="
echo ""

# Step 1: Try to compile just this file
echo "Step 1: Compiling $SOURCE_FILE..."
echo "-----------------------------------------"

# Extract base name for object file
BASENAME=$(basename "$SOURCE_FILE" .cpp)
OBJ_FILE_F64="build/cmake/CMakeFiles/scl_core_f64.dir/${SOURCE_FILE}.o"
OBJ_FILE_F32="build/cmake/CMakeFiles/scl_core_f32.dir/${SOURCE_FILE}.o"

# Try to build just this object
cd build/cmake
make "$OBJ_FILE_F64" 2>&1 | tee "/tmp/${BASENAME}_f64.log" | tail -50
echo ""
echo "Errors for f64 version:"
grep "error:" "/tmp/${BASENAME}_f64.log" | head -20 || echo "No errors found"
echo ""

# Step 2: Run clangd linting
echo "Step 2: Running clangd linting..."
echo "-----------------------------------------"
cd "$PROJECT_ROOT"
clangd --check="$SOURCE_FILE" 2>&1 | tee "/tmp/${BASENAME}_clangd.log" | grep -A 5 "error:" | head -50 || echo "No clangd errors"
echo ""

# Step 3: Summary
echo "========================================="
echo "Summary for $SOURCE_FILE:"
echo "========================================="
echo "Compile log: /tmp/${BASENAME}_f64.log"
echo "Clangd log: /tmp/${BASENAME}_clangd.log"
echo ""
echo "Next steps:"
echo "  1. Review errors above"
echo "  2. Fix issues in $SOURCE_FILE"
echo "  3. Run this script again to verify"

