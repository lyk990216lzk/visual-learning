<script setup>
import { labelText, percent } from '../utils/format'

defineProps({
  items: { type: Array, default: () => [] },
  max: { type: Number, default: 1 }
})
</script>

<template>
  <div>
    <div v-for="(item, index) in items" :key="item.label" class="prob-row">
      <div class="prob-label" :title="item.label">{{ labelText(item.label) }}</div>
      <div class="prob-track">
        <div
          class="prob-fill"
          :class="{ top: index === 0 }"
          :style="{ width: `${Math.min(100, (item.probability / max) * 100)}%` }"
        ></div>
      </div>
      <div class="prob-value">{{ percent(item.probability, 1) }}</div>
    </div>
    <div v-if="!items.length" class="dim" style="font-size: 13px">暂无结果</div>
  </div>
</template>
