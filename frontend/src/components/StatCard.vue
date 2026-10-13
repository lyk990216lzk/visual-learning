<script setup>
defineProps({
  title: { type: String, required: true },
  value: { type: [String, Number], default: '--' },
  suffix: { type: String, default: '' },
  hint: { type: String, default: '' },
  icon: { type: String, default: 'DataLine' },
  color: { type: String, default: '#4f8cff' }
})
</script>

<template>
  <div class="glass section stat-card">
    <div class="stat-icon" :style="{ background: `${color}22`, color }">
      <el-icon :size="20"><component :is="icon" /></el-icon>
    </div>
    <div class="stat-body">
      <div class="stat-title">{{ title }}</div>
      <div class="stat-value" :style="{ color }">
        {{ value }}<span v-if="suffix" class="stat-suffix">{{ suffix }}</span>
      </div>
      <div v-if="hint" class="stat-hint">{{ hint }}</div>
    </div>
  </div>
</template>

<style scoped>
.stat-card {
  display: flex;
  align-items: center;
  gap: 14px;
}

.stat-icon {
  width: 44px;
  height: 44px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex: none;
}

.stat-title {
  font-size: 13px;
  color: var(--ir-text-dim);
}

.stat-value {
  font-size: 24px;
  font-weight: 600;
  line-height: 1.3;
}

.stat-suffix {
  font-size: 13px;
  margin-left: 4px;
  color: var(--ir-text-dim);
}

.stat-hint {
  font-size: 12px;
  color: var(--ir-text-dim);
  margin-top: 2px;
}
</style>
