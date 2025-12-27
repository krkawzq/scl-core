# SCL-Core Documentation

This directory contains the documentation for SCL-Core, built with VitePress.

## Quick Start

### Development Server

Start the development server with hot reload:

```bash
npm run docs:dev
```

The documentation will be available at `http://localhost:5173`

### Build for Production

Build the documentation for production:

```bash
npm run docs:build
```

Built files will be in `docs/.vitepress/dist/`

### Preview Production Build

Preview the production build locally:

```bash
npm run docs:preview
```

## Directory Structure

```
docs/
├── .vitepress/
│   ├── config.ts           # VitePress configuration
│   ├── theme/
│   │   ├── index.ts        # Theme customization
│   │   └── style.css       # Custom styles
│   └── components/         # Custom Vue components
│       ├── ApiCard.vue
│       ├── AlgorithmBlock.vue
│       ├── ParameterTable.vue
│       ├── ComplexityBadge.vue
│       └── PerformanceChart.vue
├── index.md                # Homepage
├── guide/                  # User guides
│   ├── getting-started.md
│   ├── installation.md
│   ├── architecture.md
│   └── ...
├── api/                    # API reference
│   ├── overview.md
│   └── kernels/
│       ├── normalize.md
│       ├── neighbors.md
│       └── ...
└── examples/              # Examples and tutorials
    └── ...
```

## Writing Documentation

### Basic Markdown

VitePress supports standard Markdown with enhancements:

```markdown
# Heading 1
## Heading 2

**Bold text** and *italic text*

- List item 1
- List item 2

Code: `inline code`

\`\`\`python
# Code block
import scl_core as scl
\`\`\`
```

### Custom Containers

```markdown
::: tip
This is a tip
:::

::: warning
This is a warning
:::

::: danger
This is a danger message
:::

::: info
This is an info message
:::
```

### Using Custom Components

#### ApiCard

Display API function information:

```vue
<ApiCard
  name="normalize_total"
  signature="normalize_total(X, target_sum=1e4, inplace=True)"
  summary="Normalize counts per cell to a target sum"
  time-complexity="O(nnz)"
  space-complexity="O(1)"
>

Additional content goes here...

</ApiCard>
```

#### ParameterTable

Display function parameters:

```vue
<ParameterTable :parameters="[
  {
    name: 'X',
    type: 'sparse matrix',
    direction: 'in,out',
    description: 'Count matrix (cells × genes)'
  },
  {
    name: 'target_sum',
    type: 'float',
    direction: 'in',
    description: 'Target sum for normalization'
  }
]" />
```

#### AlgorithmBlock

Describe algorithms:

```vue
<AlgorithmBlock
  title="Normalization Algorithm"
  summary="Parallel row-wise normalization"
  :complexity="{ time: 'O(nnz)', space: 'O(1)' }"
>

1. Compute row sum
2. Divide each element by row sum
3. Multiply by target sum

</AlgorithmBlock>
```

#### ComplexityBadge

Display complexity badges:

```vue
<ComplexityBadge type="time" value="O(n log n)" />
<ComplexityBadge type="space" value="O(n)" />
```

#### PerformanceChart

Placeholder for performance charts:

```vue
<PerformanceChart 
  title="Benchmark Results"
  description="Performance comparison on 10k cells"
/>
```

### Code Highlighting

VitePress supports syntax highlighting for many languages:

````markdown
```python
import scl_core as scl
scl.normalize_total(X)
```

```cpp
template <typename T>
void normalize(T* data, size_t n) {
    // Implementation
}
```

```bash
pip install scl-core
```
````

### Line Numbers

Code blocks automatically show line numbers (configured in `config.ts`).

### Math Equations

Use LaTeX syntax for math (requires markdown-it-katex plugin):

```markdown
Inline math: $E = mc^2$

Block math:
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$
```

## Configuration

Edit `.vitepress/config.ts` to customize:

- Site title and description
- Navigation menu
- Sidebar structure
- Theme colors
- Search settings
- Social links

## Adding New Pages

1. Create a new `.md` file in the appropriate directory
2. Add front matter if needed:

```markdown
---
title: Page Title
description: Page description
---

# Content here
```

3. Update sidebar in `config.ts` if needed

## Styling

Custom styles can be added to `.vitepress/theme/style.css`.

CSS variables available:

```css
:root {
  --vp-c-brand-1: #5f67ee;
  --vp-c-brand-2: #7b82f0;
  --vp-c-brand-3: #9ca3f2;
  /* ... more variables */
}
```

## Deployment

### GitHub Pages

Add to `.github/workflows/deploy.yml`:

```yaml
name: Deploy Docs

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
      - run: npm ci
      - run: npm run docs:build
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/.vitepress/dist
```

### Netlify / Vercel

Build command: `npm run docs:build`
Publish directory: `docs/.vitepress/dist`

## Resources

- [VitePress Documentation](https://vitepress.dev/)
- [Vue 3 Documentation](https://vuejs.org/)
- [Markdown Guide](https://www.markdownguide.org/)

