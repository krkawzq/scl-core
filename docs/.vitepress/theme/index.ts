import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import './style.css'

// Import custom components
import ApiCard from '../components/ApiCard.vue'
import AlgorithmBlock from '../components/AlgorithmBlock.vue'
import ParameterTable from '../components/ParameterTable.vue'
import ComplexityBadge from '../components/ComplexityBadge.vue'
import PerformanceChart from '../components/PerformanceChart.vue'
import SourceCode from '../components/SourceCode.vue'
import CollapsibleCode from '../components/CollapsibleCode.vue'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      // https://vitepress.dev/guide/extending-default-theme#layout-slots
    })
  },
  enhanceApp({ app, router, siteData }) {
    // Register global components
    app.component('ApiCard', ApiCard)
    app.component('AlgorithmBlock', AlgorithmBlock)
    app.component('ParameterTable', ParameterTable)
    app.component('ComplexityBadge', ComplexityBadge)
    app.component('PerformanceChart', PerformanceChart)
    app.component('SourceCode', SourceCode)
    app.component('CollapsibleCode', CollapsibleCode)
  }
} satisfies Theme

