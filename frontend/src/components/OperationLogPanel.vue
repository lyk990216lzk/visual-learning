<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useAuthStore } from '../store/auth'
import { clearLogs, exportLogs, operationLogs } from '../utils/operationLog'

defineProps({
  embedded: {
    type: Boolean,
    default: false
  },
  collapsed: {
    type: Boolean,
    default: false
  }
})

const auth = useAuthStore()
const drawer = ref(false)
const keyword = ref('')
const moduleFilter = ref('')
const autoScroll = ref(true)
const listBox = ref(null)
const drawerSize = ref('560px')

const modules = computed(() => {
  const set = new Set(operationLogs.map((item) => item.module).filter(Boolean))
  return Array.from(set)
})

const visibleLogs = computed(() => {
  const word = keyword.value.trim().toLowerCase()
  return operationLogs.filter((entry) => {
    if (moduleFilter.value && entry.module !== moduleFilter.value) return false
    if (!word) return true
    return [entry.action, entry.detail, entry.error, entry.user, entry.module]
      .filter(Boolean)
      .some((text) => String(text).toLowerCase().includes(word))
  })
})

const successCount = computed(
  () => operationLogs.filter((item) => item.level === 'success').length
)
const errorCount = computed(
  () => operationLogs.filter((item) => item.level === 'error').length
)
const runningCount = computed(
  () => operationLogs.filter((item) => item.level === 'running').length
)

function statusText(entry) {
  if (entry.level === 'running') return '执行中'
  if (entry.level === 'error') return '失败'
  if (entry.level === 'info') return '记录'
  return entry.duration != null ? `成功 · ${entry.duration}ms` : '成功'
}

function updateDrawerSize() {
  drawerSize.value = window.innerWidth < 640 ? '100%' : '560px'
}

function scrollToBottom() {
  if (!autoScroll.value) return
  nextTick(() => {
    if (listBox.value) listBox.value.scrollTop = listBox.value.scrollHeight
  })
}

function openDrawer() {
  drawer.value = true
}

async function handleClear() {
  try {
    await ElMessageBox.confirm('确定清空当前会话的全部操作日志吗？', '清空日志', {
      type: 'warning',
      confirmButtonText: '清空',
      cancelButtonText: '取消'
    })
  } catch (error) {
    return
  }
  clearLogs()
  ElMessage.success('操作日志已清空')
}

function download() {
  if (!visibleLogs.value.length) {
    ElMessage.warning('当前没有可导出的日志')
    return
  }
  const now = new Date()
  const stamp = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}-${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}`
  const blob = new Blob([exportLogs(visibleLogs.value)], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `操作日志-${stamp}.txt`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

watch(
  () => operationLogs.length,
  () => scrollToBottom()
)

watch(
  () => visibleLogs.value[visibleLogs.value.length - 1]?.level,
  () => scrollToBottom()
)

watch(drawer, (open) => {
  if (open) scrollToBottom()
})

watch(autoScroll, (enabled) => {
  if (enabled) scrollToBottom()
})

onMounted(() => {
  updateDrawerSize()
  window.addEventListener('resize', updateDrawerSize)
})

onUnmounted(() => {
  window.removeEventListener('resize', updateDrawerSize)
})
</script>

<template>
  <div class="oplog-root" :class="{ 'is-embedded': embedded, 'is-collapsed': collapsed }">
    <transition name="oplog-pop">
      <button v-show="embedded || !drawer" type="button"
              :class="[
                embedded ? 'oplog-embedded-trigger' : 'oplog-fab',
                { 'is-active': embedded && drawer }
              ]"
              title="查看操作日志" aria-label="查看操作日志"
              @click="openDrawer">
        <el-icon :size="18"><Tickets /></el-icon>
        <span v-if="!embedded || !collapsed" class="oplog-fab-text">操作日志</span>
      </button>
    </transition>

    <el-drawer v-model="drawer" :size="drawerSize" :with-header="false" append-to-body
               modal-class="oplog-overlay" destroy-on-close>
      <div class="oplog-body">
        <header class="oplog-head">
          <div class="oplog-head-main">
            <div class="oplog-head-title">
              <el-icon :size="18"><Tickets /></el-icon>
              <span>操作日志</span>
              <span class="oplog-live"><i />实时</span>
            </div>
            <div class="oplog-head-sub">你在系统中的页面访问与操作记录</div>
          </div>
          <el-button text circle @click="drawer = false">
            <el-icon><Close /></el-icon>
          </el-button>
        </header>

        <div class="oplog-stats">
          <span class="oplog-stat"><b>{{ operationLogs.length }}</b> 条</span>
          <span class="oplog-stat is-ok"><b>{{ successCount }}</b> 成功</span>
          <span class="oplog-stat is-err"><b>{{ errorCount }}</b> 失败</span>
          <span v-if="runningCount" class="oplog-stat is-running"><b>{{ runningCount }}</b> 进行中</span>
          <span class="oplog-spacer" />
          <span class="oplog-user">
            <el-icon><User /></el-icon>{{ auth.displayName }}
          </span>
        </div>

        <div class="oplog-toolbar">
          <el-select v-model="moduleFilter" size="small" clearable placeholder="全部模块"
                     class="oplog-module">
            <el-option v-for="item in modules" :key="item" :label="item" :value="item" />
          </el-select>
          <el-input v-model="keyword" size="small" clearable placeholder="搜索动作 / 详情"
                    class="oplog-search">
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
          <el-tooltip content="自动滚动到最新日志" placement="top">
            <el-switch v-model="autoScroll" size="small" />
          </el-tooltip>
          <el-button size="small" text :icon="'Download'" :disabled="!visibleLogs.length"
                     @click="download">导出</el-button>
          <el-button size="small" text :icon="'Delete'" :disabled="!operationLogs.length"
                     @click="handleClear">清空</el-button>
        </div>

        <div ref="listBox" class="oplog-list">
          <div v-for="entry in visibleLogs" :key="entry.id" class="oplog-row"
               :class="`is-${entry.level}`">
            <div class="oplog-time mono">{{ entry.time }}</div>
            <span class="oplog-mark" />
            <div class="oplog-main">
              <div class="oplog-line">
                <span class="oplog-action">{{ entry.action }}</span>
                <span class="oplog-chip">{{ entry.module }}</span>
                <span class="oplog-status">{{ statusText(entry) }}</span>
              </div>
              <div v-if="entry.detail" class="oplog-detail mono">{{ entry.detail }}</div>
              <div v-if="entry.error" class="oplog-error mono">{{ entry.error }}</div>
            </div>
          </div>
          <el-empty v-if="!visibleLogs.length" :image-size="72"
                    description="暂无日志，操作后会实时出现在这里" />
        </div>

        <footer class="oplog-foot">
          <span>仅记录当前浏览器会话，最多保留 300 条</span>
          <span v-if="visibleLogs.length !== operationLogs.length">
            已筛选 {{ visibleLogs.length }} 条
          </span>
        </footer>
      </div>
    </el-drawer>
  </div>
</template>

<style scoped>
.oplog-fab {
  position: fixed;
  right: 22px;
  bottom: 22px;
  z-index: 1900;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  border-radius: 999px;
  border: 1px solid var(--ir-border);
  background: rgba(20, 27, 48, 0.94);
  color: var(--ir-text);
  font-size: 13px;
  cursor: pointer;
  box-shadow: 0 10px 26px rgba(4, 8, 20, 0.45);
  backdrop-filter: blur(10px);
  transition: transform 0.18s ease, border-color 0.18s ease;
}

.oplog-root.is-embedded {
  width: 100%;
}

.oplog-embedded-trigger {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-height: 50px;
  padding: 0 20px;
  border: 0;
  border-right: 2px solid transparent;
  color: var(--ir-text);
  background: transparent;
  font-size: 14px;
  text-align: left;
  transition: color 0.18s ease, background 0.18s ease;
}

.oplog-embedded-trigger:hover {
  color: var(--ir-text);
  background: rgba(79, 140, 255, 0.08);
}

.oplog-embedded-trigger.is-active {
  border-right-color: var(--ir-primary);
  color: var(--ir-text);
  background: linear-gradient(90deg, rgba(79, 140, 255, 0.24), rgba(79, 140, 255, 0.02));
}

.oplog-root.is-embedded.is-collapsed .oplog-embedded-trigger {
  justify-content: center;
  padding: 0 8px;
}

.oplog-fab:hover {
  transform: translateY(-2px);
  border-color: rgba(79, 140, 255, 0.65);
}

.oplog-pop-enter-active,
.oplog-pop-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.oplog-pop-enter-from,
.oplog-pop-leave-to {
  opacity: 0;
  transform: translateY(6px);
}

.oplog-body {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #0b1020;
  color: var(--ir-text);
}

.oplog-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 18px 20px 12px;
  border-bottom: 1px solid var(--ir-border);
}

.oplog-head-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
}

.oplog-head-title .el-icon {
  color: var(--ir-primary);
}

.oplog-live {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  margin-left: 4px;
  padding: 1px 8px;
  border-radius: 999px;
  border: 1px solid rgba(54, 209, 196, 0.35);
  color: var(--ir-accent);
  font-size: 11px;
  font-weight: 400;
}

.oplog-live i {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--ir-accent);
  box-shadow: 0 0 0 3px rgba(54, 209, 196, 0.14);
}

.oplog-head-sub {
  margin-top: 6px;
  font-size: 12px;
  color: var(--ir-text-dim);
}

.oplog-stats {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  padding: 10px 20px;
  font-size: 12px;
  color: var(--ir-text-dim);
}

.oplog-stat b {
  color: var(--ir-text);
  font-weight: 600;
}

.oplog-stat.is-ok b {
  color: var(--ir-accent);
}

.oplog-stat.is-err b {
  color: var(--ir-danger);
}

.oplog-stat.is-running b {
  color: var(--ir-primary);
}

.oplog-spacer {
  flex: 1;
}

.oplog-user {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.oplog-toolbar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  padding: 10px 20px;
  border-top: 1px solid var(--ir-border);
  border-bottom: 1px solid var(--ir-border);
  background: rgba(20, 27, 48, 0.5);
}

.oplog-module {
  width: 118px;
}

.oplog-search {
  flex: 1;
  min-width: 130px;
}

.oplog-list {
  flex: 1;
  overflow: auto;
  padding: 10px 12px 16px;
}

.oplog-row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 9px;
  border-left: 2px solid transparent;
  transition: background 0.15s ease;
}

.oplog-row:hover {
  background: rgba(79, 140, 255, 0.06);
}

.oplog-row.is-error {
  background: rgba(255, 107, 129, 0.06);
  border-left-color: rgba(255, 107, 129, 0.5);
}

.oplog-row.is-running {
  border-left-color: rgba(79, 140, 255, 0.5);
}

.oplog-time {
  width: 56px;
  flex: none;
  padding-top: 2px;
  color: #7c88a8;
  font-size: 11px;
}

.oplog-mark {
  width: 6px;
  height: 6px;
  flex: none;
  margin-top: 7px;
  border-radius: 50%;
  background: var(--ir-text-dim);
}

.oplog-row.is-success .oplog-mark {
  background: var(--ir-accent);
}

.oplog-row.is-error .oplog-mark {
  background: var(--ir-danger);
  box-shadow: 0 0 0 3px rgba(255, 107, 129, 0.16);
}

.oplog-row.is-running .oplog-mark {
  background: var(--ir-primary);
  animation: oplog-pulse 1.1s ease-in-out infinite;
}

.oplog-main {
  flex: 1;
  min-width: 0;
}

.oplog-line {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 7px;
  line-height: 1.45;
}

.oplog-action {
  font-size: 13px;
  font-weight: 500;
}

.oplog-chip {
  padding: 0 6px;
  border-radius: 5px;
  border: 1px solid var(--ir-border);
  color: var(--ir-text-dim);
  font-size: 10px;
  line-height: 16px;
}

.oplog-status {
  margin-left: auto;
  color: var(--ir-text-dim);
  font-size: 11px;
  white-space: nowrap;
}

.oplog-row.is-success .oplog-status {
  color: rgba(54, 209, 196, 0.85);
}

.oplog-row.is-error .oplog-status {
  color: var(--ir-danger);
}

.oplog-row.is-running .oplog-status {
  color: var(--ir-primary);
}

.oplog-detail {
  margin-top: 2px;
  color: #8592b3;
  font-size: 11px;
  word-break: break-all;
}

.oplog-error {
  margin-top: 3px;
  color: #ff8f9f;
  font-size: 11px;
  word-break: break-all;
}

.oplog-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 8px 20px;
  border-top: 1px solid var(--ir-border);
  color: #6f7b9b;
  font-size: 11px;
}

@keyframes oplog-pulse {
  0%,
  100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.45;
    transform: scale(0.75);
  }
}

@media (max-width: 640px) {
  .oplog-fab {
    right: 14px;
    bottom: 14px;
  }

  .oplog-fab-text {
    display: none;
  }
}
</style>
