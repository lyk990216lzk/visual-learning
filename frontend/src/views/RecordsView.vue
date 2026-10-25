<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search } from '@element-plus/icons-vue'
import api from '../api'
import EChart from '../components/EChart.vue'
import StatCard from '../components/StatCard.vue'
import { useAuthStore } from '../store/auth'
import { dateTime, labelText, percent } from '../utils/format'

const auth = useAuthStore()
const loading = ref(false)
const optionLoading = ref(false)
const records = ref([])
const stats = ref(null)
const total = ref(0)
const taskOptions = ref([])
const optionArchitectures = ref([])
const optionLabels = ref([])

const EMPTY_FILTER = {
  days: '14',
  task_type: '',
  architecture: '',
  training_run: '',
  label: '',
  source: '',
  search: ''
}

const filter = reactive({ ...EMPTY_FILTER })
const appliedFilter = reactive({ ...EMPTY_FILTER, page: 1, page_size: 12 })

const sourceText = {
  upload: '单图识别',
  compare: '多模型对比',
  batch: '批量识别',
  sample: '样例识别'
}

const TASK_TEXT = {
  classification: '图片识别',
  detection: '目标检测',
  segmentation: '语义分割'
}

const availableArchitectures = computed(() => {
  const rows = taskOptions.value.filter(
    (item) => !filter.task_type || item.task_type === filter.task_type
  )
  const keys = new Set(
    filter.task_type
      ? rows.map((item) => item.architecture)
      : [...rows.map((item) => item.architecture), ...optionArchitectures.value]
  )
  return [...keys]
    .filter(Boolean)
    .map((key) => ({
      value: key,
      label: rows.find((item) => item.architecture === key)?.architecture_name || key
    }))
    .sort((a, b) => a.label.localeCompare(b.label, 'zh-CN'))
})

const availableTasks = computed(() => taskOptions.value.filter(
  (item) =>
    (!filter.task_type || item.task_type === filter.task_type) &&
    (!filter.architecture || item.architecture === filter.architecture)
))

const selectedTask = computed(
  () => taskOptions.value.find((item) => item.id === filter.training_run) || null
)

const availableLabels = computed(() => {
  const pool = selectedTask.value
    ? selectedTask.value.classes
    : availableTasks.value.flatMap((item) => item.classes || [])
  const labels = new Set(pool || [])
  if (!selectedTask.value && !filter.task_type && !filter.architecture) {
    optionLabels.value.forEach((item) => labels.add(item))
  }
  return [...labels]
    .filter(Boolean)
    .sort((a, b) => labelText(a).localeCompare(labelText(b), 'zh-CN'))
})

const activeChips = computed(() => {
  const chips = [{ key: 'days', text: timeRangeText(appliedFilter.days) }]
  if (appliedFilter.task_type) {
    chips.push({ key: 'task_type', text: TASK_TEXT[appliedFilter.task_type] })
  }
  if (appliedFilter.architecture) {
    chips.push({ key: 'architecture', text: architectureName(appliedFilter.architecture) })
  }
  if (appliedFilter.training_run) {
    const task = taskOptions.value.find((item) => item.id === appliedFilter.training_run)
    chips.push({ key: 'training_run', text: task?.name || `任务 #${appliedFilter.training_run}` })
  }
  if (appliedFilter.label) chips.push({ key: 'label', text: labelText(appliedFilter.label) })
  if (appliedFilter.source) chips.push({ key: 'source', text: sourceText[appliedFilter.source] })
  if (appliedFilter.search) chips.push({ key: 'search', text: `文件名：${appliedFilter.search}` })
  return chips
})

const dailyOption = computed(() => {
  const rows = stats.value?.by_day || []
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 40, right: 18, top: 24, bottom: 30 },
    xAxis: {
      type: 'category',
      data: rows.map((item) => item.date.slice(5)),
      axisLabel: { color: '#7f8eaa', fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      minInterval: 1,
      axisLabel: { color: '#7f8eaa', fontSize: 11 },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } }
    },
    series: [
      {
        name: '识别批次',
        type: 'line',
        smooth: true,
        symbolSize: 7,
        data: rows.map((item) => item.count),
        itemStyle: { color: '#4f8cff' },
        lineStyle: { width: 2 },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(79,140,255,0.42)' },
              { offset: 1, color: 'rgba(79,140,255,0.02)' }
            ]
          }
        }
      }
    ]
  }
})

const functionOption = computed(() => {
  const rows = (stats.value?.by_task_type || []).map((item) => ({
    label: TASK_TEXT[item.task_type] || item.task_type,
    value: item.count
  }))
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 74, right: 35, top: 16, bottom: 20 },
    xAxis: {
      type: 'value',
      minInterval: 1,
      axisLabel: { color: '#7f8eaa', fontSize: 11 },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } }
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: rows.map((item) => item.label),
      axisLabel: { color: '#a7b3cd', fontSize: 11 }
    },
    series: [
      {
        name: '结果记录',
        type: 'bar',
        barWidth: 15,
        data: rows.map((item) => item.value),
        label: { show: true, position: 'right', color: '#c7d3e9', fontSize: 11 },
        itemStyle: {
          borderRadius: [0, 7, 7, 0],
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 1,
            y2: 0,
            colorStops: [
              { offset: 0, color: '#4f8cff' },
              { offset: 1, color: '#36d1c4' }
            ]
          }
        }
      }
    ]
  }
})

const architectureOption = computed(() => {
  const rows = (stats.value?.by_architecture || []).slice(0, 6)
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 92, right: 32, top: 16, bottom: 20 },
    xAxis: {
      type: 'value',
      minInterval: 1,
      axisLabel: { color: '#7f8eaa', fontSize: 11 },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } }
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: rows.map((item) => item.architecture),
      axisLabel: { color: '#a7b3cd', fontSize: 11 }
    },
    series: [
      {
        name: '结果记录',
        type: 'bar',
        barWidth: 15,
        data: rows.map((item) => item.count),
        label: { show: true, position: 'right', color: '#c7d3e9', fontSize: 11 },
        itemStyle: {
          borderRadius: [0, 7, 7, 0],
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 1,
            y2: 0,
            colorStops: [
              { offset: 0, color: '#249f97' },
              { offset: 1, color: '#45d6ca' }
            ]
          }
        }
      }
    ]
  }
})

function timeRangeText(value) {
  return {
    '1': '今天',
    '7': '近 7 天',
    '14': '近 14 天',
    '30': '近 30 天',
    all: '全部时间'
  }[value] || '近 14 天'
}

function architectureName(key) {
  return availableArchitectures.value.find((item) => item.value === key)?.label || key
}

function queryParams(source = appliedFilter) {
  const params = {
    days: source.days,
    task_type: source.task_type || undefined,
    architecture: source.architecture || undefined,
    training_run: source.training_run || undefined,
    label: source.label || undefined,
    source: source.source || undefined,
    search: source.search?.trim() || undefined
  }
  if ('page' in source) {
    params.page = source.page
    params.page_size = source.page_size
  }
  return Object.fromEntries(Object.entries(params).filter(([, value]) => value !== undefined && value !== ''))
}

async function loadOptions() {
  optionLoading.value = true
  try {
    const data = await api.recordOptions()
    taskOptions.value = data.tasks || []
    optionArchitectures.value = data.architectures || []
    optionLabels.value = data.labels || []
  } finally {
    optionLoading.value = false
  }
}

async function load() {
  loading.value = true
  try {
    const params = queryParams()
    const [list, stat] = await Promise.all([
      api.records({ ...params, grouped: 1, ordering: '-created_at' }),
      api.recordStats(params)
    ])
    records.value = list.results || []
    total.value = list.count || records.value.length
    stats.value = stat
  } finally {
    loading.value = false
  }
}

async function runQuery() {
  if (filter.architecture && !availableArchitectures.value.some(
    (item) => item.value === filter.architecture
  )) {
    filter.architecture = ''
  }
  if (filter.training_run && !availableTasks.value.some((item) => item.id === filter.training_run)) {
    filter.training_run = ''
  }
  if (filter.label && !availableLabels.value.includes(filter.label)) {
    filter.label = ''
  }
  Object.assign(appliedFilter, {
    days: filter.days,
    task_type: filter.task_type,
    architecture: filter.architecture,
    training_run: filter.training_run,
    label: filter.label,
    source: filter.source,
    search: filter.search.trim(),
    page: 1
  })
  await load()
}

async function reset() {
  Object.assign(filter, EMPTY_FILTER)
  Object.assign(appliedFilter, EMPTY_FILTER, { page: 1, page_size: 12 })
  await load()
}

function removeChip(key) {
  if (key === 'days') {
    filter.days = '14'
    appliedFilter.days = '14'
  } else {
    filter[key] = ''
    appliedFilter[key] = ''
  }
  if (key === 'task_type' || key === 'architecture') {
    filter.training_run = ''
    filter.label = ''
    appliedFilter.training_run = ''
    appliedFilter.label = ''
  }
  appliedFilter.page = 1
  load()
}

function changePage(page) {
  appliedFilter.page = page
  load()
}

async function removeRecord(record) {
  await api.deleteRecord(record.id)
  ElMessage.success('记录已删除')
  await Promise.all([loadOptions(), load()])
}

async function clearAll() {
  try {
    await ElMessageBox.confirm('确定清空全部识别记录吗？该操作不可恢复。', '提示', {
      type: 'warning'
    })
  } catch {
    return
  }
  const data = await api.clearRecords({ all: true })
  ElMessage.success(`已清空 ${data.count} 条记录`)
  await Promise.all([loadOptions(), load()])
}

function exportRecords() {
  const rows = records.value.flatMap((group) =>
    (group.records || []).map((record) => ({
      batch: group.group_id,
      file: record.original_name,
      taskType: TASK_TEXT[record.task_type] || record.task_type,
      architecture: record.architecture,
      trainingRun: record.training_run_name || '',
      model: record.model_name,
      label: labelText(record.predicted_label),
      confidence: percent(record.confidence, 2),
      source: sourceText[record.source] || record.source,
      latency: record.latency_ms,
      operator: record.username,
      time: dateTime(record.created_at)
    }))
  )
  if (!rows.length) {
    ElMessage.warning('当前没有可导出的记录')
    return
  }
  const headers = [
    ['batch', '识别批次'],
    ['file', '文件名'],
    ['taskType', '识别功能'],
    ['architecture', '网络结构'],
    ['trainingRun', '训练任务'],
    ['model', '模型名称'],
    ['label', '预测类别'],
    ['confidence', '置信度'],
    ['source', '识别来源'],
    ['latency', '耗时(ms)'],
    ['operator', '操作人'],
    ['time', '识别时间']
  ]
  const escape = (value) => `"${String(value ?? '').replaceAll('"', '""')}"`
  const csv = [
    headers.map(([, title]) => escape(title)).join(','),
    ...rows.map((row) => headers.map(([key]) => escape(row[key])).join(','))
  ].join('\n')
  const blob = new Blob([`\uFEFF${csv}`], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `识别历史_${new Date().toISOString().slice(0, 10)}.csv`
  link.click()
  URL.revokeObjectURL(url)
}

watch(
  [() => filter.task_type, () => filter.architecture],
  () => {
    if (filter.architecture && !availableArchitectures.value.some(
      (item) => item.value === filter.architecture
    )) {
      filter.architecture = ''
    }
    if (filter.training_run && !availableTasks.value.some(
      (item) => item.id === filter.training_run
    )) {
      filter.training_run = ''
    }
    if (filter.label && !availableLabels.value.includes(filter.label)) {
      filter.label = ''
    }
  }
)

watch(
  () => filter.training_run,
  () => {
    if (filter.label && !availableLabels.value.includes(filter.label)) {
      filter.label = ''
    }
  }
)

onMounted(async () => {
  await Promise.allSettled([loadOptions(), load()])
})
</script>

<template>
  <div class="page records-page" v-loading="loading">
    <div class="records-heading">
      <div>
        <div class="eyebrow"><el-icon><Filter /></el-icon>记录查询</div>
        <h1 class="records-title">识别历史</h1>
        <p>按顺序逐层收窄查询范围，统计卡片、分布图与识别记录同步更新</p>
      </div>
      <div class="heading-actions">
        <el-button @click="exportRecords"><el-icon><Download /></el-icon>导出当前页</el-button>
        <el-button type="primary" @click="load"><el-icon><Refresh /></el-icon>刷新</el-button>
      </div>
    </div>

    <section class="glass filter-panel">
      <div class="filter-head">
        <div>
          <strong><el-icon><Operation /></el-icon>筛选条件</strong>
          <span>任务名称会与识别功能、网络结构共同约束预测类别</span>
        </div>
        <div class="filter-path" aria-label="筛选顺序">
          <span>时间范围</span><el-icon><ArrowRight /></el-icon>
          <span>识别功能</span><el-icon><ArrowRight /></el-icon>
          <span>网络结构</span><el-icon><ArrowRight /></el-icon>
          <span>任务名称</span><el-icon><ArrowRight /></el-icon>
          <span>预测类别</span><el-icon><ArrowRight /></el-icon>
          <span>识别来源</span><el-icon><ArrowRight /></el-icon>
          <span>文件名</span>
        </div>
      </div>

      <div class="filter-grid">
        <label class="filter-field">
          <span><b>1</b>时间范围</span>
          <el-select v-model="filter.days">
            <el-option label="今天" value="1" />
            <el-option label="近 7 天" value="7" />
            <el-option label="近 14 天" value="14" />
            <el-option label="近 30 天" value="30" />
            <el-option label="全部时间" value="all" />
          </el-select>
        </label>
        <label class="filter-field">
          <span><b>2</b>识别功能</span>
          <el-select v-model="filter.task_type" clearable placeholder="全部功能">
            <el-option label="图片识别" value="classification" />
            <el-option label="目标检测" value="detection" />
            <el-option label="语义分割" value="segmentation" />
          </el-select>
        </label>
        <label class="filter-field">
          <span><b>3</b>网络结构</span>
          <el-select v-model="filter.architecture" clearable placeholder="全部结构">
            <el-option
              v-for="item in availableArchitectures"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </label>
        <label class="filter-field">
          <span><b>4</b>任务名称</span>
          <el-select
            v-model="filter.training_run"
            clearable
            filterable
            :loading="optionLoading"
            placeholder="全部训练任务"
            no-match-text="当前功能或结构下没有训练任务"
          >
            <el-option
              v-for="item in availableTasks"
              :key="item.id"
              :label="item.name"
              :value="item.id"
            >
              <div class="task-option">
                <span>{{ item.name }}</span>
                <small>#{{ item.id }} · {{ item.architecture_name }}</small>
              </div>
            </el-option>
          </el-select>
        </label>
        <label class="filter-field">
          <span><b>5</b>预测类别</span>
          <el-select v-model="filter.label" clearable filterable placeholder="全部类别">
            <el-option
              v-for="item in availableLabels"
              :key="item"
              :label="labelText(item)"
              :value="item"
            />
          </el-select>
        </label>
        <label class="filter-field">
          <span><b>6</b>识别来源</span>
          <el-select v-model="filter.source" clearable placeholder="全部来源">
            <el-option
              v-for="(text, key) in sourceText"
              :key="key"
              :label="text"
              :value="key"
            />
          </el-select>
        </label>
        <label class="filter-field">
          <span><b>7</b>文件名</span>
          <el-input
            v-model="filter.search"
            clearable
            placeholder="输入图片文件名"
            :prefix-icon="Search"
            @keyup.enter="runQuery"
          />
        </label>
        <div class="filter-actions">
          <el-button @click="reset">重置</el-button>
          <el-button type="primary" @click="runQuery">
            <el-icon><Search /></el-icon>查询
          </el-button>
        </div>
      </div>

      <div class="active-filter-row">
        <span class="active-filter-label"><el-icon><CollectionTag /></el-icon>当前条件</span>
        <div class="active-filter-chips">
          <el-tag
            v-for="item in activeChips"
            :key="item.key"
            closable
            effect="plain"
            @close="removeChip(item.key)"
          >{{ item.text }}</el-tag>
        </div>
        <el-button link type="primary" @click="reset">清除全部</el-button>
      </div>
    </section>

    <div class="grid grid-4 records-metrics">
      <StatCard title="识别批次" :value="stats?.group_count || 0" suffix="个" icon="Layers" />
      <StatCard
        title="结果记录"
        :value="stats?.total || 0"
        suffix="条"
        color="#36d1c4"
        icon="Files"
      />
      <StatCard
        title="平均置信度"
        :value="percent(stats?.avg_confidence, 2)"
        color="#ffb454"
        icon="TrendCharts"
      />
      <StatCard
        title="平均耗时"
        :value="stats?.avg_latency_ms || 0"
        suffix="ms"
        color="#ff7187"
        icon="Timer"
      />
    </div>

    <div class="grid records-analysis">
      <div class="glass section chart-card">
        <div class="section-title">
          <span>识别量趋势</span>
          <span class="sub">按结果记录统计</span>
        </div>
        <EChart v-if="stats?.by_day?.length" :option="dailyOption" height="220px" />
        <el-empty v-else description="当前条件下暂无数据" :image-size="70" />
      </div>
      <div class="glass section chart-card">
        <div class="section-title">
          <span>识别功能构成</span>
          <span class="sub">当前查询结果</span>
        </div>
        <EChart v-if="stats?.by_task_type?.length" :option="functionOption" height="220px" />
        <el-empty v-else description="当前条件下暂无数据" :image-size="70" />
      </div>
      <div class="glass section chart-card">
        <div class="section-title">
          <span>常用网络结构</span>
          <span class="sub">最多展示前 6 项</span>
        </div>
        <EChart v-if="stats?.by_architecture?.length" :option="architectureOption" height="220px" />
        <el-empty v-else description="当前条件下暂无数据" :image-size="70" />
      </div>
    </div>

    <section class="glass section records-list-panel">
      <div class="section-title records-list-title">
        <div>
          <span>识别记录</span>
          <p>默认按识别批次聚合，展开后可查看该批次内的模型或图片结果</p>
        </div>
        <div class="record-title-actions">
          <span class="sub">共 {{ total }} 个批次</span>
          <el-button v-if="auth.isAdmin" type="danger" plain size="small" @click="clearAll">
            清空全部
          </el-button>
        </div>
      </div>

      <el-table
        :data="records"
        row-key="group_id"
        size="small"
        class="records-table"
        empty-text="没有符合当前条件的识别记录"
      >
        <el-table-column type="expand" width="42">
          <template #default="{ row }">
            <div class="group-detail">
              <div class="detail-head">
                <span>批次明细</span>
                <span>批次编号 {{ row.group_id }}</span>
              </div>
              <div
                v-for="record in row.records"
                :key="record.id"
                class="detail-record"
              >
                <el-image
                  v-if="record.image_url"
                  :src="record.image_url"
                  fit="cover"
                  :preview-src-list="[record.image_url]"
                  preview-teleported
                  class="detail-image"
                />
                <div v-else class="detail-image placeholder"><el-icon><Picture /></el-icon></div>
                <div class="detail-main">
                  <strong>{{ record.original_name || '未命名图片' }}</strong>
                  <span>{{ record.model_name || record.architecture }}</span>
                </div>
                <div class="detail-result">
                  <el-tag size="small" type="success" effect="plain">
                    {{ labelText(record.predicted_label) || '未检出目标' }}
                  </el-tag>
                  <span>{{ percent(record.confidence, 1) }} · {{ record.latency_ms }} ms</span>
                </div>
                <span class="detail-user">{{ record.username || '—' }}</span>
                <el-button
                  v-if="auth.isAdmin"
                  link
                  type="danger"
                  size="small"
                  @click.stop="removeRecord(record)"
                >删除</el-button>
              </div>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="识别图片" min-width="180">
          <template #default="{ row }">
            <div class="record-file-cell">
              <el-image
                v-if="row.image_url"
                :src="row.image_url"
                fit="cover"
                class="record-thumb"
              />
              <div v-else class="record-thumb placeholder"><el-icon><Picture /></el-icon></div>
              <div>
                <strong>{{ row.original_name || '未命名图片' }}</strong>
                <span class="mono">{{ row.group_id.slice(0, 8) }}</span>
              </div>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="识别功能" width="104">
          <template #default="{ row }">
            <span :class="['task-type-dot', row.task_type]">{{ row.task_type_label }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="architecture" label="网络结构" width="116" />
        <el-table-column label="训练任务" min-width="160" show-overflow-tooltip>
          <template #default="{ row }">
            <div class="training-run-cell">
              <strong>{{ row.training_run_name || '未关联任务' }}</strong>
              <span v-if="row.training_run_id">#{{ row.training_run_id }}</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="识别结果" min-width="150">
          <template #default="{ row }">
            <div class="label-tags">
              <el-tag
                v-for="item in row.labels.slice(0, 3)"
                :key="item"
                size="small"
                effect="plain"
              >{{ labelText(item) }}</el-tag>
              <el-tag v-if="row.labels.length > 3" size="small" type="info" effect="plain">
                +{{ row.labels.length - 3 }}
              </el-tag>
              <span v-if="!row.labels.length" class="dim">未检出目标</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="识别来源" width="104">
          <template #default="{ row }">{{ row.source_label || sourceText[row.source] || row.source }}</template>
        </el-table-column>
        <el-table-column label="数量 / 置信度" width="128">
          <template #default="{ row }">
            <div class="count-confidence">
              <strong>{{ row.images_count }} 张 / {{ row.result_count }} 条</strong>
              <span>{{ percent(row.confidence, 1) }}</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="识别时间" width="150">
          <template #default="{ row }">
            <div class="time-cell">
              <span>{{ dateTime(row.created_at) }}</span>
              <small>{{ Number(row.latency_ms).toFixed(1) }} ms</small>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="操作人" width="100">
          <template #default="{ row }">
            <span class="operator">{{ row.username || '—' }}</span>
          </template>
        </el-table-column>
      </el-table>

      <el-pagination
        class="records-pagination"
        layout="total, prev, pager, next"
        :total="total"
        :page-size="appliedFilter.page_size"
        :current-page="appliedFilter.page"
        @current-change="changePage"
      />
    </section>
  </div>
</template>

<style scoped>
.records-page {
  min-width: 0;
}

.records-heading {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 16px;
}

.eyebrow {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 7px;
  color: #78a9fb;
  font-size: 12px;
  letter-spacing: 0.05em;
}

.records-title {
  margin: 0;
  font-size: 24px;
  font-weight: 600;
  letter-spacing: 0.02em;
}

.records-heading p {
  margin: 6px 0 0;
  color: var(--ir-text-dim);
  font-size: 13px;
}

.heading-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.filter-panel {
  padding: 16px 18px;
}

.filter-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(120, 145, 220, 0.12);
}

.filter-head strong {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 14px;
  font-weight: 500;
}

.filter-head strong :deep(.el-icon) {
  color: var(--ir-primary);
}

.filter-head > div:first-child > span {
  display: block;
  margin-top: 5px;
  color: var(--ir-text-dim);
  font-size: 12px;
}

.filter-path {
  display: flex;
  max-width: 58%;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 3px;
  color: var(--ir-text-dim);
  font-size: 11px;
}

.filter-path span {
  padding: 3px 6px;
  border-radius: 5px;
  background: rgba(79, 140, 255, 0.07);
  white-space: nowrap;
}

.filter-path :deep(.el-icon) {
  font-size: 10px;
  color: rgba(151, 163, 196, 0.55);
}

.filter-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 13px;
}

.filter-field {
  display: grid;
  min-width: 0;
  gap: 6px;
}

.filter-field > span {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--ir-text-dim);
  font-size: 12px;
}

.filter-field > span b {
  display: inline-grid;
  width: 17px;
  height: 17px;
  place-items: center;
  border: 1px solid rgba(79, 140, 255, 0.25);
  border-radius: 5px;
  color: #7babf8;
  background: rgba(79, 140, 255, 0.08);
  font-size: 10px;
  font-weight: 400;
}

.filter-field :deep(.el-select),
.filter-field :deep(.el-input) {
  width: 100%;
}

.task-option {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
}

.task-option small {
  overflow: hidden;
  color: var(--ir-text-dim);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.filter-actions {
  display: flex;
  align-items: flex-end;
  justify-content: flex-end;
  gap: 8px;
}

.active-filter-row {
  display: flex;
  min-height: 36px;
  align-items: center;
  gap: 9px;
  margin-top: 13px;
  padding: 8px 10px;
  border: 1px solid rgba(79, 140, 255, 0.11);
  border-radius: 9px;
  background: rgba(79, 140, 255, 0.035);
}

.active-filter-label {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 5px;
  color: var(--ir-text-dim);
  font-size: 12px;
}

.active-filter-chips {
  display: flex;
  flex: 1;
  flex-wrap: wrap;
  gap: 6px;
}

.active-filter-chips :deep(.el-tag) {
  --el-tag-bg-color: rgba(79, 140, 255, 0.08);
  --el-tag-border-color: rgba(79, 140, 255, 0.2);
  --el-tag-text-color: #a6c6fa;
}

.records-metrics {
  margin-top: 14px;
}

.records-analysis {
  grid-template-columns: 1.2fr 0.9fr 0.9fr;
  margin-top: 14px;
}

.chart-card {
  min-width: 0;
}

.records-list-panel {
  margin-top: 14px;
}

.records-list-title {
  align-items: flex-start;
}

.records-list-title p {
  margin: 5px 0 0;
  color: var(--ir-text-dim);
  font-size: 12px;
  font-weight: 400;
}

.record-title-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.records-table {
  width: 100%;
}

.records-table :deep(.el-table__inner-wrapper::before) {
  display: none;
}

.records-table :deep(th.el-table__cell) {
  color: var(--ir-text-dim);
  font-size: 12px;
  font-weight: 400;
  background: rgba(8, 13, 28, 0.28);
}

.records-table :deep(td.el-table__cell) {
  vertical-align: middle;
}

.record-file-cell {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.record-file-cell > div:last-child {
  min-width: 0;
}

.record-file-cell strong {
  display: block;
  overflow: hidden;
  color: var(--ir-text);
  font-size: 12px;
  font-weight: 500;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.record-file-cell span {
  display: block;
  margin-top: 4px;
  color: var(--ir-text-dim);
  font-size: 10px;
}

.record-thumb {
  width: 46px;
  height: 42px;
  flex: 0 0 auto;
  border: 1px solid var(--ir-border);
  border-radius: 8px;
}

.record-thumb.placeholder,
.detail-image.placeholder {
  display: grid;
  place-items: center;
  color: var(--ir-text-dim);
  background: rgba(9, 14, 29, 0.65);
}

.task-type-dot {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  white-space: nowrap;
}

.task-type-dot::before {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--ir-primary);
  content: '';
}

.task-type-dot.detection::before {
  background: var(--ir-warn);
}

.task-type-dot.segmentation::before {
  background: #a38cff;
}

.training-run-cell strong,
.training-run-cell span,
.count-confidence strong,
.count-confidence span,
.time-cell span,
.time-cell small {
  display: block;
}

.training-run-cell strong {
  overflow: hidden;
  color: #c7d4e9;
  font-size: 12px;
  font-weight: 500;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.training-run-cell span,
.time-cell small {
  margin-top: 4px;
  color: var(--ir-text-dim);
  font-size: 10px;
}

.label-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

.count-confidence strong {
  color: #cbd8ea;
  font-size: 11px;
  font-weight: 500;
}

.count-confidence span {
  margin-top: 5px;
  color: var(--ir-accent);
  font-size: 10px;
}

.time-cell span {
  color: #a8b5cd;
  font-size: 11px;
}

.operator {
  color: #aab7cf;
  white-space: nowrap;
}

.group-detail {
  margin: 2px 12px 10px;
  padding: 12px;
  border: 1px solid rgba(79, 140, 255, 0.13);
  border-radius: 10px;
  background: rgba(8, 14, 30, 0.45);
}

.detail-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding-bottom: 9px;
  border-bottom: 1px solid rgba(120, 145, 220, 0.1);
  color: var(--ir-text-dim);
  font-size: 11px;
}

.detail-head span:first-child {
  color: #c5d3e9;
  font-weight: 500;
}

.detail-record {
  display: grid;
  grid-template-columns: 46px minmax(130px, 1fr) auto 90px 42px;
  align-items: center;
  gap: 10px;
  padding: 9px 0;
  border-bottom: 1px solid rgba(120, 145, 220, 0.07);
}

.detail-record:last-child {
  padding-bottom: 0;
  border-bottom: 0;
}

.detail-image {
  width: 46px;
  height: 40px;
  border: 1px solid var(--ir-border);
  border-radius: 7px;
}

.detail-main {
  min-width: 0;
}

.detail-main strong,
.detail-main span {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.detail-main strong {
  color: #c9d6ea;
  font-size: 11px;
  font-weight: 500;
}

.detail-main span {
  margin-top: 4px;
  color: var(--ir-text-dim);
  font-size: 10px;
}

.detail-result {
  display: flex;
  align-items: center;
  gap: 8px;
}

.detail-result > span,
.detail-user {
  color: var(--ir-text-dim);
  font-size: 10px;
}

.records-pagination {
  justify-content: flex-end;
  margin-top: 14px;
}

@media (max-width: 1200px) {
  .filter-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .records-analysis {
    grid-template-columns: 1fr 1fr;
  }

  .records-analysis .chart-card:first-child {
    grid-column: 1 / -1;
  }
}

@media (max-width: 820px) {
  .records-heading,
  .filter-head {
    align-items: stretch;
    flex-direction: column;
  }

  .heading-actions {
    justify-content: flex-end;
  }

  .filter-path {
    max-width: none;
    justify-content: flex-start;
  }

  .filter-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .records-analysis {
    grid-template-columns: 1fr;
  }

  .records-analysis .chart-card:first-child {
    grid-column: auto;
  }
}

@media (max-width: 560px) {
  .filter-grid {
    grid-template-columns: 1fr;
  }

  .filter-actions {
    justify-content: stretch;
  }

  .filter-actions :deep(.el-button) {
    flex: 1;
  }

  .active-filter-row,
  .records-list-title,
  .record-title-actions {
    align-items: flex-start;
    flex-direction: column;
  }

  .detail-record {
    grid-template-columns: 46px minmax(0, 1fr);
  }

  .detail-result,
  .detail-user,
  .detail-record > :deep(.el-button) {
    grid-column: 2;
  }
}
</style>
