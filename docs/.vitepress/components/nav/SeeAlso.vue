<script setup lang="ts">
interface Link {
  href: string
  text: string
}

interface Props {
  links?: Link[]
}

defineProps<Props>()
</script>

<template>
  <div class="scl-see-also">
    <div class="scl-see-also__title">See Also</div>
    <ul class="scl-see-also__list">
      <li v-if="links" v-for="link in links" :key="link.href" class="scl-see-also__item">
        <a :href="link.href">{{ link.text }}</a>
      </li>
      <slot></slot>
    </ul>
  </div>
</template>
