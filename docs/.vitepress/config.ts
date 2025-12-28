import { defineConfig } from 'vitepress'
import path from 'path'
import { sourceCodePlugin } from './plugins/sourceCodePlugin'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "SCL-Core",
  description: "High-Performance Biological Operator Library",
  
  // Base URL (adjust if deploying to subdirectory)
  base: '/',
  
  // Clean URLs (remove .html extension)
  cleanUrls: true,
  
  // Markdown configuration
  markdown: {
    // Line numbers in code blocks
    lineNumbers: true,
    
    // Theme for code highlighting
    // Options: 'one-dark-pro', 'dracula', 'tokyo-night', 'vitesse-dark', 'nord', 'catppuccin-mocha'
    theme: {
      light: 'vitesse-light',
      dark: 'one-dark-pro'
    },
    
    // Custom markdown-it plugins
    config: (md) => {
      md.use(sourceCodePlugin, {
        projectRoot: path.resolve(__dirname, '../..')
      })
    }
  },
  
  // Head tags (for custom styles, fonts, etc.)
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/logo.svg' }],
    ['link', { rel: 'icon', type: 'image/png', href: '/logo.png' }],
    ['meta', { name: 'theme-color', content: '#5f67ee' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'SCL-Core' }],
    ['meta', { property: 'og:url', content: 'https://scl-core.dev/' }]
  ],
  
  // Locales configuration
  locales: {
    root: {
      label: 'English',
      lang: 'en-US',
      description: 'High-Performance Biological Operator Library',
      
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
          { text: 'C++ Developer', link: '/cpp/' },
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
          ],
          '/cpp/': [
            {
              text: 'C++ Developer Guide',
              items: [
                { text: 'Overview', link: '/cpp/' },
                { text: 'Getting Started', link: '/cpp/getting-started/' },
                { text: 'Architecture', link: '/cpp/architecture/' }
              ]
            },
            {
              text: 'Core Modules',
              collapsed: false,
              items: [
                { text: 'Overview', link: '/cpp/core/' },
                { text: 'Types', link: '/cpp/core/types' },
                { text: 'Sparse Matrix', link: '/cpp/core/sparse' },
                { text: 'Registry', link: '/cpp/core/registry' },
                { text: 'SIMD', link: '/cpp/core/simd' },
                { text: 'Memory', link: '/cpp/core/memory' },
                { text: 'Vectorize', link: '/cpp/core/vectorize' },
                { text: 'Sort', link: '/cpp/core/sort' },
                { text: 'Argsort', link: '/cpp/core/argsort' },
                { text: 'Error Handling', link: '/cpp/core/error' },
                { text: 'Macros', link: '/cpp/core/macros' }
              ]
            },
            {
              text: 'Threading',
              collapsed: false,
              items: [
                { text: 'Overview', link: '/cpp/threading/' },
                { text: 'Parallel For', link: '/cpp/threading/parallel-for' },
                { text: 'Scheduler', link: '/cpp/threading/scheduler' },
                { text: 'Workspace', link: '/cpp/threading/workspace' }
              ]
            },
            {
              text: 'Kernels',
              collapsed: true,
              items: [
                { text: 'Overview', link: '/cpp/kernels/' },
                { text: 'Sparse Tools', link: '/cpp/kernels/sparse-tools' },
                { text: 'Linear Algebra', link: '/cpp/kernels/algebra' },
                { text: 'Feature Statistics', link: '/cpp/kernels/feature' },
                { text: 'Scaling', link: '/cpp/kernels/scale' },
                { text: 'Normalization', link: '/cpp/kernels/normalization' },
                { text: 'Softmax', link: '/cpp/kernels/softmax' },
                { text: 'Mann-Whitney U', link: '/cpp/kernels/mwu' },
                { text: 'T-test', link: '/cpp/kernels/ttest' },
                { text: 'Neighbors', link: '/cpp/kernels/neighbors' },
                { text: 'BBKNN', link: '/cpp/kernels/bbknn' },
                { text: 'MMD', link: '/cpp/kernels/mmd' },
                { text: 'Correlation', link: '/cpp/kernels/correlation' },
                { text: 'Highly Variable Genes', link: '/cpp/kernels/hvg' },
                { text: 'Reorder', link: '/cpp/kernels/reorder' },
                { text: 'Louvain', link: '/cpp/kernels/louvain' },
                { text: 'Sparse Optimization', link: '/cpp/kernels/sparse-opt' },
                { text: 'Niche Analysis', link: '/cpp/kernels/niche' },
                { text: 'Merge', link: '/cpp/kernels/merge' },
                { text: 'Slice', link: '/cpp/kernels/slice' },
                { text: 'Group', link: '/cpp/kernels/group' },
                { text: 'Log1p Transform', link: '/cpp/kernels/log1p' },
                { text: 'Quality Control', link: '/cpp/kernels/qc' },
                { text: 'Resampling', link: '/cpp/kernels/resample' },
                { text: 'Entropy', link: '/cpp/kernels/entropy' },
                { text: 'Gene Regulatory Network', link: '/cpp/kernels/grn' },
                { text: 'Hotspot Detection', link: '/cpp/kernels/hotspot' },
                { text: 'Imputation', link: '/cpp/kernels/impute' },
                { text: 'Connected Components', link: '/cpp/kernels/components' },
                { text: 'Clustering Metrics', link: '/cpp/kernels/metrics' },
                { text: 'Outlier Detection', link: '/cpp/kernels/outlier' },
                { text: 'Sampling', link: '/cpp/kernels/sampling' },
                { text: 'Statistics', link: '/cpp/kernels/statistics' },
                { text: 'Clustering', link: '/cpp/kernels/clustering' },
                { text: 'Spatial', link: '/cpp/kernels/spatial' },
                { text: 'Subpopulation', link: '/cpp/kernels/subpopulation' },
                { text: 'Clonotype', link: '/cpp/kernels/clonotype' },
                { text: 'Lineage', link: '/cpp/kernels/lineage' },
                { text: 'Spatial Pattern', link: '/cpp/kernels/spatial-pattern' },
                { text: 'Tissue Architecture', link: '/cpp/kernels/tissue' }
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
        
        // Search
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
        
        // Outline
        outline: {
          level: [2, 3],
          label: 'On this page'
        }
      }
    },
    
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      description: '高性能生物算子库',
      link: '/zh/',
      
      themeConfig: {
        // Logo
        logo: '/logo.svg',
        
        // Site title
        siteTitle: 'SCL-Core',
        
        // Navigation bar
        nav: [
          { text: '首页', link: '/zh/' },
          { text: '指南', link: '/zh/guide/getting-started' },
          { 
            text: 'API 参考',
            items: [
              { text: '概览', link: '/zh/api/' },
              { text: 'Python API', link: '/zh/api/python/' },
              { text: 'C API', link: '/zh/api/c-api/' }
            ]
          },
          { text: 'C++ 开发者', link: '/zh/cpp/' },
          { text: '示例', link: '/zh/examples/basic-usage' },
          {
            text: 'v0.2',
            items: [
              { text: '更新日志', link: '/zh/changelog' },
              { text: '贡献指南', link: '/zh/contributing' }
            ]
          }
        ],
        
        // Sidebar
        sidebar: {
          '/zh/guide/': [
            {
              text: '介绍',
              items: [
                { text: '什么是 SCL-Core?', link: '/zh/guide/what-is-scl' },
                { text: '快速开始', link: '/zh/guide/getting-started' },
                { text: '安装', link: '/zh/guide/installation' }
              ]
            },
            {
              text: '核心概念',
              items: [
                { text: '架构', link: '/zh/guide/architecture' },
                { text: '性能', link: '/zh/guide/performance' },
                { text: 'C-ABI 接口', link: '/zh/guide/c-abi' }
              ]
            }
          ],
          '/zh/api/': [
            {
              text: 'API 概览',
              items: [
                { text: '介绍', link: '/zh/api/' },
                { text: 'Python API', link: '/zh/api/python/' },
                { text: 'C API', link: '/zh/api/c-api/' }
              ]
            },
            {
              text: 'Python API',
              collapsed: false,
              items: [
                { text: '概览', link: '/zh/api/python/' },
                { text: '预处理', link: '/zh/api/python/preprocessing/' },
                { text: '邻居搜索', link: '/zh/api/python/neighbors/' },
                { text: '统计', link: '/zh/api/python/stats/' },
                { text: '工具', link: '/zh/api/python/utilities/' }
              ]
            },
            {
              text: 'C API',
              collapsed: false,
              items: [
                { text: '概览', link: '/zh/api/c-api/' },
                { text: '核心类型', link: '/zh/api/c-api/core/types' },
                { text: '错误处理', link: '/zh/api/c-api/core/error' },
                { text: '稀疏矩阵', link: '/zh/api/c-api/core/sparse' },
                { text: '内存管理', link: '/zh/api/c-api/memory' },
                { text: '内核函数', link: '/zh/api/c-api/kernels/' },
                { text: '归一化', link: '/zh/api/c-api/kernels/normalize' }
              ]
            }
          ],
          '/zh/examples/': [
            {
              text: '示例',
              items: [
                { text: '基本用法', link: '/zh/examples/basic-usage' },
                { text: '性能优化', link: '/zh/examples/performance' },
                { text: '自定义算子', link: '/zh/examples/custom-operators' }
              ]
            }
          ],
          '/zh/cpp/': [
            {
              text: 'C++ 开发者指南',
              items: [
                { text: '概览', link: '/zh/cpp/' },
                { text: '快速开始', link: '/zh/cpp/getting-started/' },
                { text: '架构设计', link: '/zh/cpp/architecture/' }
              ]
            },
            {
              text: '核心模块',
              collapsed: false,
              items: [
                { text: '概览', link: '/zh/cpp/core/' },
                { text: '类型系统', link: '/zh/cpp/core/types' },
                { text: '稀疏矩阵', link: '/zh/cpp/core/sparse' },
                { text: '注册表', link: '/zh/cpp/core/registry' },
                { text: 'SIMD', link: '/zh/cpp/core/simd' },
                { text: '内存', link: '/zh/cpp/core/memory' },
                { text: '向量化', link: '/zh/cpp/core/vectorize' },
                { text: '排序', link: '/zh/cpp/core/sort' },
                { text: '参数排序', link: '/zh/cpp/core/argsort' },
                { text: '错误处理', link: '/zh/cpp/core/error' },
                { text: '宏定义', link: '/zh/cpp/core/macros' }
              ]
            },
            {
              text: '并行处理',
              collapsed: false,
              items: [
                { text: '概览', link: '/zh/cpp/threading/' },
                { text: '并行循环', link: '/zh/cpp/threading/parallel-for' },
                { text: '调度器', link: '/zh/cpp/threading/scheduler' },
                { text: '工作空间', link: '/zh/cpp/threading/workspace' }
              ]
            },
            {
              text: '内核函数',
              collapsed: true,
              items: [
                { text: '概览', link: '/zh/cpp/kernels/' },
                { text: '稀疏工具', link: '/zh/cpp/kernels/sparse-tools' },
                { text: '线性代数', link: '/zh/cpp/kernels/algebra' },
                { text: '特征统计', link: '/zh/cpp/kernels/feature' },
                { text: '缩放', link: '/zh/cpp/kernels/scale' },
                { text: '归一化', link: '/zh/cpp/kernels/normalization' },
                { text: 'Softmax', link: '/zh/cpp/kernels/softmax' },
                { text: 'Mann-Whitney U', link: '/zh/cpp/kernels/mwu' },
                { text: 'T 检验', link: '/zh/cpp/kernels/ttest' },
                { text: '邻居搜索', link: '/zh/cpp/kernels/neighbors' },
                { text: 'BBKNN', link: '/zh/cpp/kernels/bbknn' },
                { text: 'MMD', link: '/zh/cpp/kernels/mmd' },
                { text: '相关性', link: '/zh/cpp/kernels/correlation' },
                { text: '高变基因', link: '/zh/cpp/kernels/hvg' },
                { text: '重排序', link: '/zh/cpp/kernels/reorder' },
                { text: 'Louvain', link: '/zh/cpp/kernels/louvain' },
                { text: '稀疏优化', link: '/zh/cpp/kernels/sparse-opt' },
                { text: '生态位分析', link: '/zh/cpp/kernels/niche' },
                { text: '合并', link: '/zh/cpp/kernels/merge' },
                { text: '切片', link: '/zh/cpp/kernels/slice' },
                { text: '分组', link: '/zh/cpp/kernels/group' },
                { text: 'Log1p 变换', link: '/zh/cpp/kernels/log1p' },
                { text: '质量控制', link: '/zh/cpp/kernels/qc' },
                { text: '重采样', link: '/zh/cpp/kernels/resample' },
                { text: '扩散算法', link: '/zh/cpp/kernels/diffusion' },
                { text: '双联体检测', link: '/zh/cpp/kernels/doublet' },
                { text: '富集分析', link: '/zh/cpp/kernels/enrichment' },
                { text: '图神经网络', link: '/zh/cpp/kernels/gnn' },
                { text: '熵', link: '/zh/cpp/kernels/entropy' },
                { text: '基因调控网络', link: '/zh/cpp/kernels/grn' },
                { text: '热点检测', link: '/zh/cpp/kernels/hotspot' },
                { text: '插补', link: '/zh/cpp/kernels/impute' },
                { text: '连通分量', link: '/zh/cpp/kernels/components' },
                { text: '聚类度量', link: '/zh/cpp/kernels/metrics' },
                { text: '异常值检测', link: '/zh/cpp/kernels/outlier' },
                { text: '采样', link: '/zh/cpp/kernels/sampling' },
                { text: '统计', link: '/zh/cpp/kernels/statistics' },
                { text: '聚类', link: '/zh/cpp/kernels/clustering' },
                { text: '空间分析', link: '/zh/cpp/kernels/spatial' },
                { text: '亚群分析', link: '/zh/cpp/kernels/subpopulation' },
                { text: '克隆型', link: '/zh/cpp/kernels/clonotype' },
                { text: '谱系追踪', link: '/zh/cpp/kernels/lineage' },
                { text: '空间模式', link: '/zh/cpp/kernels/spatial-pattern' },
                { text: '组织结构', link: '/zh/cpp/kernels/tissue' }
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
          message: '基于 MIT 许可证发布',
          copyright: 'Copyright © 2024-present SCL-Core Contributors'
        },
        
        // Search
        search: {
          provider: 'local',
          options: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档'
              },
              modal: {
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭'
                }
              }
            }
          }
        },
        
        // Edit link
        editLink: {
          pattern: 'https://github.com/krkawzq/scl-core/edit/main/docs/:path',
          text: '在 GitHub 上编辑此页'
        },
        
        // Last updated
        lastUpdated: {
          text: '最后更新',
          formatOptions: {
            dateStyle: 'short',
            timeStyle: 'short'
          }
        },
        
        // Outline
        outline: {
          level: [2, 3],
          label: '本页目录'
        }
      }
    }
  }
})
