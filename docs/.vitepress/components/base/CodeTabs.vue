<script setup lang="ts">
import { ref, computed, useSlots, onMounted } from 'vue'

interface Props {
  tabs: string[]
  syncKey?: string
}

const props = defineProps<Props>()
const slots = useSlots()

const activeIndex = ref(0)

// Generate slot names from tab labels
const slotNames = computed(() => {
  return props.tabs.map(tab =>
    tab.toLowerCase().replace(/[^a-z0-9]/g, '-')
  )
})

const setActive = (index: number) => {
  activeIndex.value = index
  // Save preference if syncKey is provided
  if (props.syncKey && typeof localStorage !== 'undefined') {
    localStorage.setItem(`scl-tabs-${props.syncKey}`, String(index))
  }
}

// Restore preference on mount
onMounted(() => {
  if (props.syncKey && typeof localStorage !== 'undefined') {
    const saved = localStorage.getItem(`scl-tabs-${props.syncKey}`)
    if (saved !== null) {
      const index = parseInt(saved, 10)
      if (index >= 0 && index < props.tabs.length) {
        activeIndex.value = index
      }
    }
  }
})
</script>

<template>
  <div class="scl-code-tabs">
    <div class="scl-code-tabs__header">
      <button
        v-for="(tab, index) in tabs"
        :key="tab"
        class="scl-code-tabs__tab"
        :class="{ 'scl-code-tabs__tab--active': activeIndex === index }"
        @click="setActive(index)"
      >
        {{ tab }}
      </button>
    </div>
    <div class="scl-code-tabs__content">
      <div
        v-for="(name, index) in slotNames"
        :key="name"
        class="scl-code-tabs__panel"
        :class="{ 'scl-code-tabs__panel--active': activeIndex === index }"
      >
        <slot :name="name"></slot>
      </div>
    </div>
  </div>
</template>
