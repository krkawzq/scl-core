# Source Code Block Test

This page tests the `::: source_code` container functionality.

## Test 1: Extract Array Struct

::: source_code file="scl/core/type.hpp" symbol="Array" title="Array View Struct"
:::

## Test 2: Extract AlignedBuffer Class

::: source_code file="scl/core/memory.hpp" symbol="AlignedBuffer" title="Aligned Buffer RAII Wrapper"
:::

## Test 3: Extract a Template Function

::: source_code file="scl/core/memory.hpp" symbol="fill" title="SIMD Fill Function"
:::

## Test 4: Extract TagSparse Struct

::: source_code file="scl/core/type.hpp" symbol="TagSparse"
:::

## Expected Behavior

Each block above should:
1. Find the specified file in the SCL project
2. Extract the source code for the specified symbol
3. Display it in a styled code block with:
   - File path and line numbers
   - Syntax highlighting (via VitePress/Shiki)

## Error Handling Test

This should show an error message:

::: source_code file="nonexistent.hpp" symbol="foo"
:::
