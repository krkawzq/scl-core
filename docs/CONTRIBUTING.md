# SCL-Core Documentation Contributing Guide

This guide explains the documentation standards and conventions for SCL-Core C++ developer documentation.

## Table of Contents

- [Documentation Philosophy](#documentation-philosophy)
- [Structure Overview](#structure-overview)
- [File Organization](#file-organization)
- [Writing Standards](#writing-standards)
- [Source Code Integration](#source-code-integration)
- [Templates](#templates)
- [Bilingual Requirements](#bilingual-requirements)

---

## Documentation Philosophy

SCL-Core documentation follows these principles:

1. **File-Oriented, Not Title-Oriented**: Documentation is organized by source files, not by abstract topics
2. **Developer-First**: Written for C++ developers implementing or extending SCL-Core
3. **Two-Tier System**: Overview documents + detailed per-file documentation
4. **Show the Code**: Use collapsible source code blocks to show actual implementations
5. **Bilingual**: Maintain parallel English and Chinese documentation

---

## Structure Overview

```
docs/
├── cpp/
│   ├── index.md                    # C++ Developer Overview
│   ├── core/
│   │   ├── index.md               # Core Module Overview (brief)
│   │   ├── type.md                # type.hpp Detailed Documentation
│   │   ├── sparse.md              # sparse.hpp Detailed Documentation
│   │   └── ...
│   ├── kernel/
│   │   ├── index.md               # Kernel Module Overview
│   │   ├── normalize.md           # normalize.hpp Detailed
│   │   └── ...
│   └── ...
├── zh/cpp/                         # Chinese mirror
│   ├── index.md
│   ├── core/
│   └── ...
└── _templates/                     # Documentation templates
    ├── overview-en.md
    ├── overview-zh.md
    ├── detail-en.md
    └── detail-zh.md
```

---

## File Organization

### Module Overview Documents (index.md)

Overview documents provide a quick reference table of all files in a module.

**Purpose**:
- List all header files in the module
- Provide brief one-line descriptions
- Link to detailed documentation

**Format**:

```markdown
# Core Module

| File | Description | Main APIs |
|------|-------------|-----------|
| [type.hpp](./type) | Type system | Array, Span, Index |
| [sparse.hpp](./sparse) | Sparse matrices | Sparse class |
```

**Template**: [`_templates/overview-en.md`](_templates/overview-en.md) or [`_templates/overview-zh.md`](_templates/overview-zh.md)

### Detailed File Documents

Each `.hpp` file gets its own detailed documentation page.

**Naming Convention**: `filename.md` (without the `.hpp` extension)

**Example**: `scl/core/type.hpp` → `docs/cpp/core/type.md`

---

## Writing Standards

### Detailed Documentation Structure

Every detailed file document MUST follow this structure:

#### 1. Header Section

```markdown
# filename.hpp

> scl/module/filename.hpp · Brief one-line description

## Overview

Detailed description of file's purpose and functionality.
```

#### 2. Main API Sections

For each **primary** function, class, or struct:

**Required Components**:

| Component | Required | Description |
|-----------|----------|-------------|
| Source Code | ✅ | Collapsible `source_code` block |
| Algorithm Description | ✅ | Explain the core algorithm |
| Edge Cases | ✅ | Handle empty input, nulls, limits |
| Data Guarantees | ✅ | Preconditions (sorted, valid, etc.) |
| Complexity Analysis | ✅ | Time and space complexity |
| Example | ✅ | Working code example |

**Format**:

```markdown
### FunctionName

::: source_code file="scl/module/file.hpp" symbol="FunctionName" collapsed
:::

**Algorithm Description**

1. Step 1: ...
2. Step 2: ...

**Edge Cases**

- **Empty input**: Behavior
- **Null pointer**: Behavior

**Data Guarantees (Preconditions)**

- Input must be sorted
- Pointers must be valid or null

**Complexity Analysis**

- **Time**: O(n log n)
- **Space**: O(1)

**Example**

\`\`\`cpp
// Working example
\`\`\`
```

#### 3. Utility Functions Section

For **helper** or **utility** functions:

**Required Components**:

| Component | Required | Description |
|-----------|----------|-------------|
| Source Code | ✅ | Collapsible block |
| Brief Description | ✅ | One-line explanation |
| Complexity | Optional | For simple functions |

**Format**:

```markdown
## Utility Functions

### helper_func

Brief description of what it does.

::: source_code file="scl/module/file.hpp" symbol="helper_func" collapsed
:::

**Complexity**

- Time: O(1)
- Space: O(1)
```

---

## Source Code Integration

### Using Collapsible Code Blocks

**Syntax**:

```markdown
::: source_code file="scl/core/type.hpp" symbol="Array" collapsed
:::
```

**Parameters**:

| Parameter | Required | Description |
|-----------|----------|-------------|
| `file` | ✅ | Path to `.hpp` file relative to project root |
| `symbol` | ✅ | Function/class/struct name to extract |
| `collapsed` | ❌ | Include to default to collapsed state |
| `title` | ❌ | Custom title (defaults to symbol name) |

**Examples**:

```markdown
<!-- Default collapsed -->
::: source_code file="scl/core/memory.hpp" symbol="aligned_alloc" collapsed
:::

<!-- Default expanded -->
::: source_code file="scl/core/memory.hpp" symbol="aligned_alloc"
:::

<!-- Custom title -->
::: source_code file="scl/core/memory.hpp" symbol="aligned_alloc" title="Aligned Memory Allocation" collapsed
:::
```

### Best Practices

1. **Always collapse in detailed docs**: Keep documentation scannable
2. **Show full definitions**: Extract complete function/class definitions
3. **One symbol per block**: Don't try to extract multiple symbols at once
4. **Verify extraction**: Ensure symbol names match exactly (case-sensitive)

---

## Templates

Use the provided templates as starting points:

### Overview Templates

- **English**: [`_templates/overview-en.md`](_templates/overview-en.md)
- **Chinese**: [`_templates/overview-zh.md`](_templates/overview-zh.md)

### Detailed Documentation Templates

- **English**: [`_templates/detail-en.md`](_templates/detail-en.md)
- **Chinese**: [`_templates/detail-zh.md`](_templates/detail-zh.md)

### Customizing Templates

1. Copy the appropriate template
2. Replace placeholders:
   - `[Module Name]` → Actual module name
   - `filename.hpp` → Actual file name
   - `ClassName` → Actual API name
3. Fill in descriptions, algorithms, examples
4. Add all main APIs and utility functions

---

## Bilingual Requirements

All documentation MUST be provided in both English and Chinese.

### File Mirroring

```
docs/cpp/core/type.md        ← English
docs/zh/cpp/core/type.md     ← Chinese (mirror)
```

### Translation Guidelines

1. **Maintain structure**: Same headings, same order
2. **Preserve code blocks**: Don't translate code, keep as-is
3. **Translate descriptions**: Algorithm descriptions, edge cases, etc.
4. **Keep formatting**: Same markdown formatting and layout
5. **Link consistency**: Update relative links to point to Chinese versions

### Translation Priority

When creating new documentation:

1. Write English version first
2. Create Chinese version immediately after
3. Both versions should be complete before PR

---

## Quality Checklist

Before submitting documentation:

### Content Checklist

- [ ] File overview is clear and concise
- [ ] All main APIs documented
- [ ] All sections include required components
- [ ] Algorithms are clearly explained
- [ ] Edge cases are covered
- [ ] Complexity analysis is accurate
- [ ] Examples are tested and working
- [ ] Utility functions are documented

### Format Checklist

- [ ] Uses appropriate template
- [ ] Source code blocks use `collapsed` flag
- [ ] All `source_code` blocks extract correctly
- [ ] No syntax errors in markdown
- [ ] Code examples have syntax highlighting
- [ ] Links are valid and working

### Bilingual Checklist

- [ ] Both English and Chinese versions exist
- [ ] Structure matches between versions
- [ ] Translations are accurate
- [ ] Links point to correct language versions
- [ ] No untranslated content

---

## Examples

### Good Example: Main API Documentation

```markdown
### normalize_rows

::: source_code file="scl/kernel/normalize.hpp" symbol="normalize_rows" collapsed
:::

**Algorithm Description**

Normalize each row of a sparse matrix to unit norm:

1. Compute row sum using parallel reduction
2. Divide each element by the row sum
3. Handle near-zero rows (< epsilon) by leaving unchanged

**Edge Cases**

- **Empty matrix**: Returns immediately without changes
- **Zero rows**: Rows with sum < epsilon are left unchanged
- **NaN/Inf**: Replaced with zero before normalization

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR format
- Indices must be sorted within rows
- No duplicate indices

**Complexity Analysis**

- **Time**: O(nnz) where nnz is number of non-zeros
- **Space**: O(1) auxiliary space

**Example**

\`\`\`cpp
#include "scl/kernel/normalize.hpp"

// Create sparse matrix
scl::Sparse<Real, true> matrix = ...;

// Normalize rows to L2 norm
scl::kernel::normalize_rows(matrix, scl::kernel::NormMode::L2);

// Each row now has unit L2 norm
\`\`\`
```

### Good Example: Utility Function

```markdown
## Utility Functions

### is_csr_format

Check if a sparse matrix is in CSR format.

::: source_code file="scl/core/sparse.hpp" symbol="is_csr_format" collapsed
:::

**Complexity**

- Time: O(1)
- Space: O(1)
```

---

## Getting Help

If you have questions about documentation standards:

1. Check existing documentation for examples
2. Review the templates in `_templates/`
3. Ask in GitHub Discussions
4. Open an issue with the `documentation` label

---

## Appendix: Markdown Features

### Custom Containers

VitePress supports custom containers:

```markdown
::: tip
Helpful tip
:::

::: warning
Warning message
:::

::: danger
Critical warning
:::
```

### Code Highlighting

Specify language for syntax highlighting:

```markdown
\`\`\`cpp
// C++ code
\`\`\`

\`\`\`python
# Python code
\`\`\`
```

### Links

```markdown
[Link text](./relative/path)
[External](https://example.com)
```

---

**Last Updated**: December 2024
**Maintainers**: SCL-Core Team

