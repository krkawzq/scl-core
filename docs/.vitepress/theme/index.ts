import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import './style.css'

// Base components
import Badge from '../components/base/Badge.vue'
import Callout from '../components/base/Callout.vue'
import CodeTabs from '../components/base/CodeTabs.vue'

// API components
import ApiSignature from '../components/api/ApiSignature.vue'
import ParamTable from '../components/api/ParamTable.vue'
import SupportMatrix from '../components/api/SupportMatrix.vue'
import ApiFunctionRenderer from '../components/api/ApiFunctionRenderer.vue'

// Content components
import AlgoCard from '../components/content/AlgoCard.vue'
import Steps from '../components/content/Steps.vue'
import Step from '../components/content/Step.vue'

// Navigation components
import SeeAlso from '../components/nav/SeeAlso.vue'
import ModuleNav from '../components/nav/ModuleNav.vue'

// Meta components
import SourceLink from '../components/meta/SourceLink.vue'
import Since from '../components/meta/Since.vue'
import Deprecated from '../components/meta/Deprecated.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    // Base
    app.component('Badge', Badge)
    app.component('Callout', Callout)
    app.component('CodeTabs', CodeTabs)

    // API
    app.component('ApiSignature', ApiSignature)
    app.component('ParamTable', ParamTable)
    app.component('SupportMatrix', SupportMatrix)
    app.component('ApiFunctionRenderer', ApiFunctionRenderer)

    // Content
    app.component('AlgoCard', AlgoCard)
    app.component('Steps', Steps)
    app.component('Step', Step)

    // Navigation
    app.component('SeeAlso', SeeAlso)
    app.component('ModuleNav', ModuleNav)

    // Meta
    app.component('SourceLink', SourceLink)
    app.component('Since', Since)
    app.component('Deprecated', Deprecated)
  }
} satisfies Theme
