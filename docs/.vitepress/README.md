# VitePress Documentation Setup

This directory contains the VitePress configuration and custom components for the SCL-Core documentation.

## Structure

```
.vitepress/
├── components/          # Custom Vue components
│   ├── ApiCard.vue
│   ├── AlgorithmBlock.vue
│   ├── ParameterTable.vue
│   ├── ComplexityBadge.vue
│   ├── PerformanceChart.vue
│   └── SourceCode.vue   # NEW: Automatic source code extraction
├── plugins/             # Markdown-it plugins
│   └── sourceCodePlugin.ts  # NEW: ::: source_code block handler
├── utils/               # Utility functions
│   └── cppParser.ts     # NEW: C++ source code parser
├── theme/               # Theme customization
│   ├── index.ts         # Theme entry point
│   └── style.css        # Custom styles
├── config.ts            # VitePress configuration
└── cache/               # Build cache (gitignored)
```

## Custom Components

### ApiCard
Displays API function/method information with complexity badges.

### AlgorithmBlock
Highlights algorithm descriptions with complexity information.

### ParameterTable
Displays function parameters in a structured table.

### ComplexityBadge
Shows time/space complexity with color coding.

### PerformanceChart
Displays performance metrics and benchmarks.

### SourceCode (NEW)
Automatically extracts and displays C++ source code from the SCL codebase.

**Usage:**
```markdown
::: source_code file="scl/core/memory.hpp" symbol="aligned_alloc"
:::
```

See [SOURCE_CODE_USAGE.md](./SOURCE_CODE_USAGE.md) for detailed documentation.

## Plugins

### sourceCodePlugin
Markdown-it plugin that processes `::: source_code` blocks at build time.

**Features:**
- Automatic file location
- C++ symbol extraction (functions, classes, structs, enums)
- Intelligent brace matching
- Error reporting
- GitHub link generation

## Development

### Install Dependencies

```bash
npm install
```

### Run Dev Server

```bash
npm run docs:dev
```

### Build Documentation

```bash
npm run docs:build
```

### Preview Build

```bash
npm run docs:preview
```

## Adding New Components

1. Create component in `components/`:
   ```vue
   <template>
     <!-- Your component -->
   </template>
   
   <script setup lang="ts">
   // Component logic
   </script>
   
   <style scoped>
   /* Component styles */
   </style>
   ```

2. Register in `theme/index.ts`:
   ```typescript
   import YourComponent from '../components/YourComponent.vue'
   
   app.component('YourComponent', YourComponent)
   ```

3. Use in markdown:
   ```markdown
   <YourComponent prop="value" />
   ```

## Adding New Plugins

1. Create plugin in `plugins/`:
   ```typescript
   export function yourPlugin(md: MarkdownIt, options: YourOptions) {
     // Plugin logic
   }
   ```

2. Register in `config.ts`:
   ```typescript
   markdown: {
     config: (md) => {
       md.use(yourPlugin, options)
     }
   }
   ```

## Styling

Global styles are in `theme/style.css`. Component-specific styles use scoped CSS in Vue components.

### CSS Variables

VitePress provides CSS variables for theming:
- `--vp-c-brand-1`: Primary brand color
- `--vp-c-bg-soft`: Soft background
- `--vp-c-divider`: Divider color
- `--vp-c-text-1`: Primary text
- `--vp-c-text-2`: Secondary text

See [VitePress Theme Documentation](https://vitepress.dev/guide/extending-default-theme) for more.

## Troubleshooting

### Build Errors

**TypeScript errors:**
- Ensure `@types/node` and `@types/markdown-it` are installed
- Check `tsconfig.json` in project root

**Plugin errors:**
- Check console output for detailed error messages
- Verify file paths and symbol names in source_code blocks

**Component errors:**
- Ensure components are properly registered in `theme/index.ts`
- Check Vue component syntax

### Performance Issues

**Slow builds:**
- Clear cache: `rm -rf .vitepress/cache`
- Reduce number of source_code blocks
- Use manual code blocks for frequently changed code

**Large bundle size:**
- Check for large dependencies
- Use dynamic imports for heavy components
- Optimize images and assets

## Resources

- [VitePress Documentation](https://vitepress.dev/)
- [Markdown-it Documentation](https://markdown-it.github.io/)
- [Vue 3 Documentation](https://vuejs.org/)

