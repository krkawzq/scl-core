<script setup lang="ts">
type SupportLevel = boolean | 'partial'

interface Feature {
  name: string
  numpy?: SupportLevel
  sparse?: SupportLevel
  dask?: SupportLevel
  gpu?: SupportLevel
}

interface Props {
  features: Feature[]
  columns?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  columns: () => ['NumPy', 'Sparse', 'Dask', 'GPU']
})

const columnKeys = ['numpy', 'sparse', 'dask', 'gpu'] as const

const getIcon = (level?: SupportLevel) => {
  if (level === true) return '✓'
  if (level === 'partial') return '◐'
  if (level === false) return '✗'
  return '—'
}

const getClass = (level?: SupportLevel) => {
  if (level === true) return 'scl-support-matrix__yes'
  if (level === 'partial') return 'scl-support-matrix__partial'
  if (level === false) return 'scl-support-matrix__no'
  return ''
}
</script>

<template>
  <table class="scl-support-matrix">
    <thead>
      <tr>
        <th>Function</th>
        <th v-for="col in columns" :key="col">{{ col }}</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="feature in features" :key="feature.name">
        <td>{{ feature.name }}</td>
        <td
          v-for="(key, index) in columnKeys"
          :key="key"
          :class="getClass(feature[key])"
        >
          {{ getIcon(feature[key]) }}
        </td>
      </tr>
    </tbody>
  </table>
</template>
