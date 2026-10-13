<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  option: { type: Object, required: true },
  height: { type: String, default: '300px' }
})

const container = ref(null)
let chart = null

function render() {
  if (!container.value) return
  if (!chart) {
    chart = echarts.init(container.value, 'dark', { renderer: 'canvas' })
    chart.getZr().setCursorStyle('default')
  }
  chart.setOption(props.option, true)
}

function resize() {
  chart?.resize()
}

onMounted(() => {
  render()
  window.addEventListener('resize', resize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', resize)
  chart?.dispose()
  chart = null
})

watch(() => props.option, render, { deep: true })
</script>

<template>
  <div ref="container" :style="{ width: '100%', height }"></div>
</template>
