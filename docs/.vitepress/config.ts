import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'SCL Core',
  description: 'High-performance biological operator library',

  lang: 'en-US',

  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
  ],

  markdown: {
    math: true,
    lineNumbers: true,
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  },

  themeConfig: {
    logo: '/logo.svg',

    nav: [
      { text: 'Home', link: '/' },
      { text: 'C++ API', link: '/cpp/' },
      { text: 'Python API', link: '/python/' },
    ],

    sidebar: {
      '/cpp/': [
        {
          text: 'C++ API',
          items: [
            { text: 'Overview', link: '/cpp/' },
          ]
        }
      ],
      '/python/': [
        {
          text: 'Python API',
          items: [
            { text: 'Overview', link: '/python/' },
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/krkawzq/scl-core' }
    ],

    search: {
      provider: 'local'
    },

    outline: {
      level: [2, 3],
      label: 'On this page'
    },

    lastUpdated: {
      text: 'Last updated',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short'
      }
    }
  }
})
