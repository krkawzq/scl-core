# Contributing Guide

Thank you for your interest in contributing to SCL-Core! This guide will help you understand our development workflow and coding standards.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/scl-core.git
cd scl-core

# Add upstream remote
git remote add upstream https://github.com/krkawzq/scl-core.git
```

### 2. Create Branch

```bash
# Update main
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/my-feature
```

### 3. Make Changes

Follow our [Code Standards](#code-standards) and [Documentation Standards](#documentation-standards).

### 4. Test

```bash
# Build and test
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
ctest --output-on-failure
```

### 5. Commit

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature: brief description

Detailed description of changes:
- What was changed
- Why it was changed
- Any breaking changes"
```

### 6. Push and Create PR

```bash
# Push to your fork
git push origin feature/my-feature

# Create Pull Request on GitHub
```

## Code Standards

### Language

**English only** for all code, comments, and documentation.

### File Organization

**Dual-file system:**

- `.hpp` files - Implementation with minimal comments
- `.h` files - Comprehensive API documentation

**Example:**

```cpp
// scl/kernel/myfeature.h - API documentation
#pragma once

namespace scl::kernel::myfeature {

/* -----------------------------------------------------------------------------
 * FUNCTION: my_function
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Brief description of what the function does.
 *
 * PARAMETERS:
 *     input  [in]  Input array, size n
 *     output [out] Output array, size n (pre-allocated)
 *     n      [in]  Array size
 *
 * PRECONDITIONS:
 *     - input and output must be valid pointers
 *     - output must have space for n elements
 *
 * POSTCONDITIONS:
 *     - output[i] contains processed value of input[i]
 *     - input is unchanged
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
template <typename T>
void my_function(
    const T* input,    // Input array
    T* output,         // Output array
    size_t n           // Array size
);

} // namespace scl::kernel::myfeature
```

```cpp
// scl/kernel/myfeature.hpp - Implementation
#pragma once

#include "scl/kernel/myfeature.h"
#include "scl/core/type.hpp"

namespace scl::kernel::myfeature {

template <typename T>
void my_function(const T* input, T* output, size_t n) {
    // Brief comment about algorithm if non-obvious
    for (size_t i = 0; i < n; ++i) {
        output[i] = process(input[i]);
    }
}

} // namespace scl::kernel::myfeature
```

### Naming Conventions

**Files:**
- `snake_case.hpp` for implementation
- `snake_case.h` for documentation

**Namespaces:**
- `scl::module::feature`
- All lowercase, nested

**Types:**
- `PascalCase` for classes/structs
- `snake_case` for type aliases

**Functions:**
- `snake_case` for all functions
- Descriptive names (e.g., `normalize_rows_inplace`)

**Variables:**
- `snake_case` for local variables
- `trailing_underscore_` for member variables
- `UPPER_CASE` for constants

**Example:**

```cpp
namespace scl::kernel::normalize {

// Type
struct NormalizationOptions {
    Real epsilon_;
    bool inplace_;
};

// Function
void normalize_rows_inplace(Sparse<Real, true>& matrix, NormMode mode);

// Constants
constexpr Real DEFAULT_EPSILON = 1e-12;

} // namespace scl::kernel::normalize
```

### Code Formatting

**Indentation:**
- 4 spaces (no tabs)

**Line Length:**
- Prefer < 100 characters
- Max 120 characters

**Braces:**
- Opening brace on same line
- Always use braces for control structures

**Example:**

```cpp
if (condition) {
    do_something();
} else {
    do_something_else();
}

for (Index i = 0; i < n; ++i) {
    process(i);
}
```

### Header Guards

Use `#pragma once`:

```cpp
#pragma once

// ... content ...
```

### Includes

Order:
1. Corresponding header
2. C++ standard library
3. Third-party libraries
4. SCL-Core headers

**Example:**

```cpp
#include "scl/kernel/normalize.h"

#include <vector>
#include <algorithm>

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/threading/parallel_for.hpp"
```

### Error Handling

**Debug assertions:**

```cpp
SCL_ASSERT(i >= 0 && i < n, "Index out of bounds");
```

**Runtime checks:**

```cpp
SCL_CHECK_ARG(data != nullptr, "Null pointer");
SCL_CHECK_DIM(output.size() == n, "Size mismatch");
```

**Exceptions:**

```cpp
if (invalid_condition) {
    throw std::invalid_argument("Descriptive error message");
}
```

### Performance Guidelines

**Use SIMD:**

```cpp
namespace s = scl::simd;
const s::Tag d;

for (size_t i = 0; i < n; i += s::Lanes(d)) {
    auto v = s::Load(d, data + i);
    // ... SIMD operations ...
}
```

**Parallelize:**

```cpp
parallel_for(Size(0), n, [&](size_t i) {
    // Parallel work
});
```

**Avoid allocations in hot loops:**

```cpp
// BAD
for (size_t i = 0; i < n; ++i) {
    std::vector<Real> temp(100);  // Allocation!
}

// GOOD
std::vector<Real> temp(100);
for (size_t i = 0; i < n; ++i) {
    // Reuse temp
}
```

## Documentation Standards

### API Documentation

Every public function must have comprehensive documentation in the `.h` file.

**Required sections:**

- SUMMARY
- PARAMETERS
- PRECONDITIONS
- POSTCONDITIONS
- COMPLEXITY
- THREAD SAFETY

**Optional sections:**

- MUTABILITY (for in-place operations)
- ALGORITHM (for non-trivial algorithms)
- THROWS (if function can throw)
- NUMERICAL NOTES (for numerical algorithms)

### Plain Text Only

**NO Markdown or LaTeX:**

```cpp
// BAD
/* SUMMARY:
 *     Computes the **L2 norm** using $$\|x\|_2 = \sqrt{\sum x_i^2}$$
 */

// GOOD
/* SUMMARY:
 *     Computes the L2 norm using norm = sqrt(sum(x_i^2))
 */
```

### No Examples in API Docs

Describe behavior precisely instead of showing examples:

```cpp
// BAD
/* EXAMPLE:
 *     Real data[] = {1, 2, 3};
 *     Real norm = compute_norm(data, 3, NormMode::L2);
 */

// GOOD
/* ALGORITHM:
 *     For L2 norm:
 *         1. Compute sum of squares: s = sum(x_i^2)
 *         2. Return square root: sqrt(s)
 */
```

## Testing

### Write Tests

Every new feature must include tests:

```cpp
// tests/test_myfeature.cpp
#include <gtest/gtest.h>
#include "scl/kernel/myfeature.hpp"

TEST(MyFeature, BasicFunctionality) {
    // Arrange
    std::vector<Real> input = {1, 2, 3};
    std::vector<Real> output(3);
    
    // Act
    scl::kernel::myfeature::my_function(
        input.data(), output.data(), 3);
    
    // Assert
    EXPECT_NEAR(output[0], expected_value, 1e-6);
}

TEST(MyFeature, EdgeCases) {
    // Test empty input
    // Test single element
    // Test large input
}
```

### Run Tests

```bash
cd build
ctest --output-on-failure
```

### Test Coverage

Aim for high test coverage:
- Happy path
- Edge cases (empty, single element, large)
- Error conditions
- Thread safety (if applicable)

## Pull Request Guidelines

### PR Title

Use descriptive titles:

```
Add feature: Sparse matrix validation
Fix bug: Memory leak in Registry
Improve: SIMD performance for L2 norm
Docs: Update contributing guide
```

### PR Description

Include:

1. **What** - What changes were made
2. **Why** - Why the changes were necessary
3. **How** - How the changes were implemented
4. **Testing** - How the changes were tested
5. **Breaking Changes** - Any breaking changes (if applicable)

**Example:**

```markdown
## What
Add sparse matrix validation function to check structural integrity.

## Why
Users need a way to verify that sparse matrices are well-formed before
processing, especially when loading from external sources.

## How
- Implemented `validate()` function in `scl/kernel/sparse.hpp`
- Checks index bounds, sorted indices, and NNZ consistency
- Returns `ValidationResult` struct with detailed error info

## Testing
- Added unit tests for valid and invalid matrices
- Tested with various edge cases (empty, single element, large)
- All tests pass

## Breaking Changes
None
```

### Review Process

1. **Automated checks** - CI must pass
2. **Code review** - At least one maintainer approval
3. **Testing** - All tests must pass
4. **Documentation** - Documentation must be updated

### Addressing Feedback

```bash
# Make requested changes
git add .
git commit -m "Address review feedback: fix memory leak"
git push origin feature/my-feature
```

## Commit Message Guidelines

### Format

```
<type>: <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Example

```
feat: Add sparse matrix validation

Implement validate() function to check structural integrity of sparse
matrices. This is useful for verifying matrices loaded from external
sources.

Changes:
- Add validate() function to scl/kernel/sparse.hpp
- Add ValidationResult struct
- Add comprehensive tests
- Update documentation

Closes #123
```

## Code Review Checklist

Before submitting PR, verify:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated (`.h` files)
- [ ] No compiler warnings
- [ ] No memory leaks (run with sanitizers)
- [ ] Performance benchmarks (if applicable)
- [ ] Commit messages are clear

## Community Guidelines

### Be Respectful

- Be kind and courteous
- Respect different viewpoints
- Accept constructive criticism gracefully

### Ask Questions

- No question is too simple
- Use GitHub Discussions for questions
- Use Issues for bug reports

### Help Others

- Review pull requests
- Answer questions in Discussions
- Improve documentation

## Getting Help

- **Questions**: [GitHub Discussions](https://github.com/krkawzq/scl-core/discussions)
- **Bug Reports**: [GitHub Issues](https://github.com/krkawzq/scl-core/issues)
- **Email**: maintainers@scl-core.dev

---

::: tip First Contribution?
Start with issues labeled "good first issue". These are beginner-friendly tasks that will help you get familiar with the codebase.
:::

