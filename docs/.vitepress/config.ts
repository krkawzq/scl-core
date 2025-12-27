import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "SCL-Core",
  description: "High-Performance Biological Operator Library",
  
  // Language and locale
  lang: 'en-US',
  
  // Base URL (adjust if deploying to subdirectory)
  base: '/',
  
  // Clean URLs (remove .html extension)
  cleanUrls: true,
  
  // Theme configuration
  themeConfig: {
    // Logo
    logo: '/logo.svg',
    
    // Site title
    siteTitle: 'SCL-Core',
    
    // Navigation bar
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/getting-started' },
      { 
        text: 'API Reference',
        items: [
          { text: 'Overview', link: '/api/' },
          { text: 'Python API', link: '/api/python/' },
          { text: 'C API', link: '/api/c-api/' }
        ]
      },
      { text: 'Examples', link: '/examples/basic-usage' },
      {
        text: 'v0.2',
        items: [
          { text: 'Changelog', link: '/changelog' },
          { text: 'Contributing', link: '/contributing' }
        ]
      }
    ],
    
    // Sidebar
    sidebar: {
      '/guide/': [
        {
          text: 'Introduction',
          items: [
            { text: 'What is SCL-Core?', link: '/guide/what-is-scl' },
            { text: 'Getting Started', link: '/guide/getting-started' },
            { text: 'Installation', link: '/guide/installation' }
          ]
        },
        {
          text: 'Core Concepts',
          items: [
            { text: 'Architecture', link: '/guide/architecture' },
            { text: 'Performance', link: '/guide/performance' },
            { text: 'C-ABI Interface', link: '/guide/c-abi' }
          ]
        }
      ],
      '/api/': [
        {
          text: 'API Overview',
          items: [
            { text: 'Introduction', link: '/api/' },
            { text: 'Python API', link: '/api/python/' },
            { text: 'C API', link: '/api/c-api/' }
          ]
        },
        {
          text: 'Python API',
          collapsed: false,
          items: [
            { text: 'Overview', link: '/api/python/' },
            { text: 'Preprocessing', link: '/api/python/preprocessing/' },
            { text: 'Neighbors', link: '/api/python/neighbors/' },
            { text: 'Statistics', link: '/api/python/stats/' },
            { text: 'Utilities', link: '/api/python/utilities/' }
          ]
        },
        {
          text: 'C API',
          collapsed: false,
          items: [
            { text: 'Overview', link: '/api/c-api/' },
            { text: 'Core Types', link: '/api/c-api/core/types' },
            { text: 'Error Handling', link: '/api/c-api/core/error' },
            { text: 'Sparse Matrices', link: '/api/c-api/core/sparse' },
            { text: 'Memory Management', link: '/api/c-api/memory' },
            { text: 'Kernels', link: '/api/c-api/kernels/' },
            { text: 'Normalize', link: '/api/c-api/kernels/normalize' }
          ]
        }
      ],
      '/examples/': [
        {
          text: 'Examples',
          items: [
            { text: 'Basic Usage', link: '/examples/basic-usage' },
            { text: 'Performance Tuning', link: '/examples/performance' },
            { text: 'Custom Operators', link: '/examples/custom-operators' }
          ]
        }
      ]
    },
    
    // Social links
    socialLinks: [
      { icon: 'github', link: 'https://github.com/krkawzq/scl-core' }
    ],
    
    // Footer
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-present SCL-Core Contributors'
    },
    
    // Search (requires algolia or local search plugin)
    search: {
      provider: 'local'
    },
    
    // Edit link
    editLink: {
      pattern: 'https://github.com/krkawzq/scl-core/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },
    
    // Last updated
    lastUpdated: {
      text: 'Last updated',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short'
      }
    },
    
    // Outline (table of contents)
    outline: {
      level: [2, 3],
      label: 'On this page'
    }
  },
  
  // Markdown configuration
  markdown: {
    // Line numbers in code blocks
    lineNumbers: true,
    
    // Theme for code highlighting
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  },
  
  // Head tags (for custom styles, fonts, etc.)
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/logo.svg' }],
    ['link', { rel: 'icon', type: 'image/png', href: '/logo.png' }],
    ['meta', { name: 'theme-color', content: '#5f67ee' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:locale', content: 'en' }],
    ['meta', { property: 'og:title', content: 'SCL-Core | High-Performance Biological Operator Library' }],
    ['meta', { property: 'og:site_name', content: 'SCL-Core' }],
    ['meta', { property: 'og:url', content: 'https://scl-core.dev/' }]
  ]
})

