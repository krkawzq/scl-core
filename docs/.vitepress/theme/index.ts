import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import './style.css'

// Import custom components
import ApiSignature from '../components/ApiSignature.vue'
import ParamTable from '../components/ParamTable.vue'
import AlgoCard from '../components/AlgoCard.vue'
import ComplexityBadge from '../components/ComplexityBadge.vue'
import SourceLink from '../components/SourceLink.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    // Register global components
    app.component('ApiSignature', ApiSignature)
    app.component('ParamTable', ParamTable)
    app.component('AlgoCard', AlgoCard)
    app.component('ComplexityBadge', ComplexityBadge)
    app.component('SourceLink', SourceLink)
  }
} satisfies Theme
