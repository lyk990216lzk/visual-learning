<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import ProbBars from '../components/ProbBars.vue'
import { useAppStore } from '../store'
import { labelText, percent } from '../utils/format'
import { logAction } from '../utils/operationLog'

const store = useAppStore()

const TASK_OPTIONS = [
  { value: 'classification', label: '图片识别', desc: '判断图片所属类别', icon: 'Picture' },
  { value: 'detection', label: '目标检测', desc: '定位并识别多个目标', icon: 'Aim' },
  { value: 'segmentation', label: '语义分割', desc: '生成逐像素类别掩膜', icon: 'Crop' }
]
const TASK_LABELS = Object.fromEntries(TASK_OPTIONS.map((item) => [item.value, item.label]))
const RECOGNITION_MODES = [
  {
    value: 'single',
    label: '单图识别',
    desc: '每次选择一张图片，快速查看识别结果',
    icon: 'Picture'
  },
  {
    value: 'batch',
    label: '多图识别',
    desc: '一次选择多张图片，统一使用当前任务模型',
    icon: 'Files'
  }
]

const currentPage = ref('config')
const activeTask = ref('classification')
const recognitionMode = ref('single')

const tasks = ref([])
const selectedArchitecture = ref('')
const selectedTaskId = ref(null)
const taskLoading = ref(false)
const taskError = ref('')

const samples = ref([])
const sampleLoading = ref(false)
const sampleError = ref('')
const selectedSampleKey = ref('')
const batchSelectedKeys = ref([])

const singleFile = ref(null)
const localPreviewUrl = ref('')
const batchFileList = ref([])
const batchLocalFiles = ref([])

const topK = ref(5)
const confidence = ref(0.25)
const maskOpacity = ref(0.7)
const loading = ref(false)
const batchLoading = ref(false)
const result = ref(null)
const batchResult = ref(null)

let taskRequestId = 0
let sampleRequestId = 0

const taskTypeMeta = computed(
  () => TASK_OPTIONS.find((item) => item.value === activeTask.value) || TASK_OPTIONS[0]
)
const modeMeta = computed(
  () => RECOGNITION_MODES.find((item) => item.value === recognitionMode.value) || RECOGNITION_MODES[0]
)
const isBatch = computed(() => recognitionMode.value === 'batch')
const hasResult = computed(() => Boolean(result.value || batchResult.value))
const modelGroups = computed(() => {
  const groups = new Map()
  tasks.value.forEach((task) => {
    if (!groups.has(task.architecture)) {
      groups.set(task.architecture, {
        key: task.architecture,
        name: task.architecture_name || task.architecture,
        count: 0
      })
    }
    groups.get(task.architecture).count += 1
  })
  return Array.from(groups.values())
})
const tasksForModel = computed(
  () => tasks.value.filter((item) => item.architecture === selectedArchitecture.value)
)
const selectedTask = computed(
  () => tasks.value.find((item) => item.id === selectedTaskId.value) || null
)
const maxTopK = computed(() => Math.max(1, Number(selectedTask.value?.trained_model_num_classes || 1)))
const selectedSample = computed(
  () => samples.value.find((item) => item.key === selectedSampleKey.value) || null
)
const batchSelectedSamples = computed(
  () => samples.value.filter((item) => batchSelectedKeys.value.includes(item.key))
)
const singleSourceFile = computed(() => singleFile.value || selectedSample.value?.file || null)
const singleSourceName = computed(() => singleSourceFile.value?.name || '')
const displayPreviewUrl = computed(() => localPreviewUrl.value || selectedSample.value?.previewUrl || '')
const batchTotalCount = computed(
  () => batchSelectedSamples.value.length + batchLocalFiles.value.length
)
const resultImageUrl = computed(() => result.value?.image_url || displayPreviewUrl.value)
const inferenceDevice = computed(() => {
  const device = String(store.trainingDevice || store.system?.device || 'cpu').toLowerCase()
  return device === 'gpu' ? 'cuda' : device
})
const batchFileListSafe = computed(() => batchFileList.value)

const probabilityOption = computed(() => {
  const entries = Object.entries(result.value?.all_probabilities || {})
  return {
    tooltip: {
      trigger: 'item',
      formatter: (params) => `${labelText(params.name)}: ${percent(params.value, 1)}`
    },
    legend: {
      bottom: 0,
      textStyle: { color: '#97a3c4', fontSize: 11 },
      itemWidth: 10,
      itemHeight: 10
    },
    series: [
      {
        type: 'pie',
        radius: ['42%', '68%'],
        center: ['50%', '44%'],
        avoidLabelOverlap: true,
        label: { show: false },
        data: entries.map(([name, value]) => ({ name, value: Number(value.toFixed(4)) })),
        itemStyle: { borderColor: '#0b1020', borderWidth: 2 }
      }
    ]
  }
})

const detectionBoxes = computed(() => {
  if (result.value?.task_type !== 'detection') return []
  const width = Number(result.value.width || 0)
  const height = Number(result.value.height || 0)
  if (!width || !height) return []
  const colors = ['#36d1c4', '#ffb454', '#b499ff', '#ff7187', '#5c95ff']
  return (result.value.detections || []).map((item, index) => {
    const [x1, y1, x2, y2] = item.bbox || [0, 0, 0, 0]
    return {
      ...item,
      color: colors[index % colors.length],
      style: {
        left: `${Math.max(0, (x1 / width) * 100)}%`,
        top: `${Math.max(0, (y1 / height) * 100)}%`,
        width: `${Math.max(0.5, ((x2 - x1) / width) * 100)}%`,
        height: `${Math.max(0.5, ((y2 - y1) / height) * 100)}%`
      }
    }
  })
})

const segmentationMaskUrl = computed(() => (
  result.value?.mask_base64 ? `data:image/png;base64,${result.value.mask_base64}` : ''
))

onMounted(async () => {
  if (!store.system) store.bootstrap().catch(() => {})
  await loadRecognitionTasks()
})

watch(activeTask, async () => {
  invalidateResult()
  selectedArchitecture.value = ''
  selectedTaskId.value = null
  await loadRecognitionTasks()
})

watch(selectedTaskId, async (taskId) => {
  await loadTaskSamples(taskId)
})

watch(maxTopK, (value) => {
  if (topK.value > value) topK.value = value
})

onBeforeUnmount(() => {
  taskRequestId += 1
  sampleRequestId += 1
  releaseSampleUrls(samples.value)
  if (localPreviewUrl.value) URL.revokeObjectURL(localPreviewUrl.value)
})

async function loadRecognitionTasks() {
  const requestId = ++taskRequestId
  taskLoading.value = true
  taskError.value = ''
  try {
    const data = await api.recognizableTrainingRuns({
      task_type: activeTask.value,
      limit: 100
    })
    if (requestId !== taskRequestId) return
    tasks.value = data.results || []
    if (!tasks.value.length) {
      selectedArchitecture.value = ''
      selectedTaskId.value = null
      return
    }
    const firstTask = tasks.value[0]
    selectedArchitecture.value = firstTask.architecture
    selectedTaskId.value = firstTask.id
  } catch (error) {
    if (requestId !== taskRequestId) return
    tasks.value = []
    selectedArchitecture.value = ''
    selectedTaskId.value = null
    taskError.value = error?.message || '已完成训练任务加载失败'
  } finally {
    if (requestId === taskRequestId) taskLoading.value = false
  }
}

function selectArchitecture(key) {
  const firstTask = tasks.value.find((item) => item.architecture === key)
  selectedArchitecture.value = key
  selectedTaskId.value = firstTask?.id ?? null
  invalidateResult()
}

async function loadTaskSamples(taskId = selectedTaskId.value) {
  const requestId = ++sampleRequestId
  releaseSampleUrls(samples.value)
  samples.value = []
  singleFile.value = null
  batchFileList.value = []
  batchLocalFiles.value = []
  if (localPreviewUrl.value) {
    URL.revokeObjectURL(localPreviewUrl.value)
    localPreviewUrl.value = ''
  }
  selectedSampleKey.value = ''
  batchSelectedKeys.value = []
  sampleError.value = ''
  invalidateResult()
  sampleLoading.value = false
  if (!taskId) return

  const task = tasks.value.find((item) => item.id === taskId)
  if (!task) return
  sampleLoading.value = true
  try {
    const data = await api.trainingRunSamples(taskId, { limit: 6, t: Date.now() })
    const loaded = await Promise.all(
      (data.results || []).map(async (item, index) => {
        try {
          const blob = await api.datasetImageBlob(task.dataset, {
            label: item.label,
            name: item.name,
            split: item.split || ''
          })
          const fileName = String(item.name).split('/').pop()
          const raw = new File([blob], fileName, { type: blob.type || 'image/jpeg' })
          return {
            ...item,
            index,
            key: `${item.split || ''}|${item.label}|${item.name}`,
            file: raw,
            previewUrl: URL.createObjectURL(blob)
          }
        } catch (error) {
          return null
        }
      })
    )
    if (requestId !== sampleRequestId) {
      loaded.filter(Boolean).forEach((item) => URL.revokeObjectURL(item.previewUrl))
      return
    }
    samples.value = loaded.filter(Boolean)
    if (samples.value.length) {
      selectedSampleKey.value = samples.value[0].key
      batchSelectedKeys.value = samples.value.slice(0, Math.min(4, samples.value.length)).map((item) => item.key)
      topK.value = Math.min(5, maxTopK.value)
    } else {
      sampleError.value = '该任务数据集中没有可抽取的图片'
    }
  } catch (error) {
    if (requestId === sampleRequestId) sampleError.value = error?.message || '任务样例加载失败'
  } finally {
    if (requestId === sampleRequestId) sampleLoading.value = false
  }
}

function releaseSampleUrls(items) {
  items.forEach((item) => {
    if (item.previewUrl) URL.revokeObjectURL(item.previewUrl)
  })
}

async function refreshSamples() {
  if (!selectedTask.value) return
  await loadTaskSamples(selectedTask.value.id)
  if (samples.value.length) ElMessage.success('已从任务数据集重新抽取 6 张图片')
}

function selectSample(sample) {
  if (isBatch.value) {
    const selected = new Set(batchSelectedKeys.value)
    if (selected.has(sample.key)) selected.delete(sample.key)
    else selected.add(sample.key)
    batchSelectedKeys.value = Array.from(selected)
  } else {
    selectedSampleKey.value = sample.key
    singleFile.value = null
    if (localPreviewUrl.value) {
      URL.revokeObjectURL(localPreviewUrl.value)
      localPreviewUrl.value = ''
    }
  }
  invalidateResult()
}

function isSampleSelected(sample) {
  return isBatch.value
    ? batchSelectedKeys.value.includes(sample.key)
    : selectedSampleKey.value === sample.key
}

function selectAllSamples() {
  batchSelectedKeys.value = samples.value.map((item) => item.key)
  invalidateResult()
}

function clearSampleSelection() {
  batchSelectedKeys.value = []
  invalidateResult()
}

function setRecognitionMode(mode) {
  recognitionMode.value = mode
  invalidateResult()
}

function pickFile(uploadFile) {
  const raw = uploadFile.raw || uploadFile
  if (!raw) return false
  singleFile.value = raw
  selectedSampleKey.value = ''
  if (localPreviewUrl.value) URL.revokeObjectURL(localPreviewUrl.value)
  localPreviewUrl.value = URL.createObjectURL(raw)
  invalidateResult()
  logAction({
    action: '选择识别图片',
    detail: `${raw.name}${raw.size ? ` · ${formatSize(raw.size)}` : ''}`,
    module: '识别',
    level: 'info'
  })
  return false
}

function onBatchChange(_file, fileList) {
  batchFileList.value = fileList || []
  batchLocalFiles.value = batchFileList.value.map((item) => item.raw).filter(Boolean)
  invalidateResult()
}

function removeBatchFile(_file, fileList) {
  batchFileList.value = fileList || []
  batchLocalFiles.value = batchFileList.value.map((item) => item.raw).filter(Boolean)
  invalidateResult()
}

function formatSize(size) {
  if (size >= 1024 * 1024) return `${(size / 1024 / 1024).toFixed(2)} MB`
  return `${Math.max(1, Math.round(size / 1024))} KB`
}

function appendTaskOptions(formData, task) {
  formData.append('trained_model', task.trained_model_id)
  if (task.task_type === 'classification') {
    formData.append('top_k', topK.value)
  } else if (task.task_type === 'detection') {
    formData.append('confidence', confidence.value)
    formData.append('iou', 0.45)
    formData.append('max_det', 100)
    formData.append('device', inferenceDevice.value)
  } else if (task.task_type === 'segmentation') {
    formData.append('device', inferenceDevice.value)
  }
}

async function submitSingle() {
  const task = selectedTask.value
  if (!task) {
    ElMessage.warning('请先选择一个已完成训练任务')
    return
  }
  const file = singleSourceFile.value
  if (!file) {
    ElMessage.warning('请从任务数据集选择一张图片，或上传本地图片')
    return
  }

  const formData = new FormData()
  formData.append('image', file)
  appendTaskOptions(formData, task)
  loading.value = true
  try {
    const request = task.task_type === 'detection'
      ? api.detect
      : task.task_type === 'segmentation'
        ? api.segment
        : api.recognize
    result.value = await request(formData)
    batchResult.value = null
    currentPage.value = 'result'
    ElMessage.success(`识别完成：${labelText(result.value.predicted_label) || '未检出目标'}`)
    logAction({
      action: '单图识别完成',
      detail: `${TASK_LABELS[task.task_type]} · 任务 #${task.id} · ${task.architecture_name} · ${Math.round(result.value.latency_ms)}ms`,
      module: '识别'
    })
  } catch (error) {
    result.value = null
  } finally {
    loading.value = false
  }
}

async function submitBatch() {
  const task = selectedTask.value
  if (!task) {
    ElMessage.warning('请先选择一个已完成训练任务')
    return
  }
  const files = [
    ...batchSelectedSamples.value.map((item) => item.file),
    ...batchLocalFiles.value
  ]
  if (!files.length) {
    ElMessage.warning('请至少选择一张任务数据集图片或上传本地图片')
    return
  }
  if (files.length > 20) {
    ElMessage.warning('多图识别单次最多 20 张图片')
    return
  }

  const formData = new FormData()
  files.forEach((file) => formData.append('images', file))
  appendTaskOptions(formData, task)
  batchLoading.value = true
  try {
    batchResult.value = await api.recognizeBatch(formData)
    result.value = null
    currentPage.value = 'result'
    ElMessage.success(`多图识别完成，成功 ${batchResult.value.total} 张`)
    logAction({
      action: '多图识别完成',
      detail: `${TASK_LABELS[task.task_type]} · 任务 #${task.id} · ${batchResult.value.total} 张图片`,
      module: '识别'
    })
  } catch (error) {
    batchResult.value = null
  } finally {
    batchLoading.value = false
  }
}

function runCurrentMode() {
  if (isBatch.value) submitBatch()
  else submitSingle()
}

function invalidateResult() {
  result.value = null
  batchResult.value = null
  currentPage.value = 'config'
}

function taskMetric(task) {
  const metrics = task?.metrics || {}
  if (task?.task_type === 'detection') {
    return `mAP50 ${percent(metrics.map50 ?? task.best_accuracy, 1)}`
  }
  if (task?.task_type === 'segmentation') {
    return `mIoU ${percent(metrics.miou ?? metrics.best_miou ?? task.best_accuracy, 1)}`
  }
  return `准确率 ${percent(task?.best_accuracy, 1)}`
}

function taskFileSize(task) {
  const bytes = Number(task?.trained_model_file_size || 0)
  if (!bytes) return '—'
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  return `${Math.max(1, Math.round(bytes / 1024))} KB`
}

function taskLabel(value) {
  return TASK_LABELS[value] || value
}

function taskTagType(value) {
  return { classification: 'success', detection: 'warning', segmentation: 'info' }[value] || 'info'
}

function batchResultText(row) {
  if (row.task_type === 'detection') {
    const labels = (row.detections || []).map((item) => item.label)
    return labels.length ? Array.from(new Set(labels)).join(' / ') : '未检出目标'
  }
  if (row.task_type === 'segmentation') {
    const stats = (row.class_stats || []).filter((item) => item.index !== 0)
    const top = [...stats].sort((a, b) => b.percentage - a.percentage)[0] || row.class_stats?.[0]
    return top ? `${top.label} ${percent(top.percentage, 1)}` : '无有效分割类别'
  }
  return labelText(row.predicted_label)
}

function batchResultSubtext(row) {
  if (row.task_type === 'detection') return `${row.object_count || 0} 个目标`
  if (row.task_type === 'segmentation') return '像素掩膜'
  return `Top-${(row.topk || []).length}`
}
</script>

<template>
  <div class="page recognize-page">
    <nav class="page-switch" aria-label="视觉识别页面">
      <button
        type="button"
        class="page-switch-item"
        :class="{ active: currentPage === 'config' }"
        :aria-current="currentPage === 'config' ? 'page' : undefined"
        @click="currentPage = 'config'"
      >
        <span>1</span>识别配置
      </button>
      <el-icon class="page-switch-arrow"><ArrowRight /></el-icon>
      <button
        type="button"
        class="page-switch-item"
        :class="{ active: currentPage === 'result' }"
        :aria-current="currentPage === 'result' ? 'page' : undefined"
        :disabled="!hasResult"
        @click="currentPage = 'result'"
      >
        <span>2</span>识别结果
      </button>
    </nav>

    <div v-if="currentPage === 'config'" class="step-stack">
      <section class="glass section workflow-step">
        <div class="step-head">
          <div class="step-title"><span>1</span>选择识别功能</div>
          <span class="step-note">图片识别 / 目标检测 / 语义分割</span>
        </div>
        <div class="function-grid">
          <button
            v-for="item in TASK_OPTIONS"
            :key="item.value"
            type="button"
            class="function-card"
            :class="{ active: activeTask === item.value }"
            :aria-pressed="activeTask === item.value"
            @click="activeTask = item.value"
          >
            <span class="function-icon"><el-icon><component :is="item.icon" /></el-icon></span>
            <span class="function-copy"><strong>{{ item.label }}</strong><small>{{ item.desc }}</small></span>
          </button>
        </div>
      </section>

      <section class="glass section workflow-step">
        <div class="step-head">
          <div class="step-title"><span>2</span>选择已完成的训练任务</div>
          <span class="step-note">先选模型，再选任务</span>
        </div>

        <el-skeleton v-if="taskLoading" :rows="3" animated />
        <el-alert
          v-else-if="taskError"
          :title="taskError"
          type="error"
          :closable="false"
          show-icon
        />
        <el-empty
          v-else-if="!tasks.length"
          :image-size="76"
          :description="`还没有可用的${taskTypeMeta.label}训练任务`"
        />
        <div v-else class="task-selection">
          <div class="task-selector-fields">
            <div class="field-row">
              <label for="recognition-model">模型</label>
              <el-select
                id="recognition-model"
                :model-value="selectedArchitecture"
                style="width: 100%"
                @change="selectArchitecture"
              >
                <el-option
                  v-for="group in modelGroups"
                  :key="group.key"
                  :value="group.key"
                  :label="`${group.name}（${group.count} 个已完成任务）`"
                />
              </el-select>
            </div>
            <div class="field-row">
              <label for="recognition-task">训练任务</label>
              <el-select id="recognition-task" v-model="selectedTaskId" style="width: 100%">
                <el-option
                  v-for="task in tasksForModel"
                  :key="task.id"
                  :value="task.id"
                  :label="`#${task.id} ${task.name}（${task.dataset_name} · ${taskMetric(task)}）`"
                />
              </el-select>
            </div>
          </div>

          <aside v-if="selectedTask" class="bound-model">
            <div class="bound-head">
              <span><el-icon><Connection /></el-icon>任务自动绑定的模型</span>
              <el-tag size="small" type="success" effect="dark">权重就绪</el-tag>
            </div>
            <div class="bound-main">
              <span class="model-cube"><el-icon><Cpu /></el-icon></span>
              <span class="bound-copy">
                <strong>{{ selectedTask.trained_model_name }}</strong>
                <small>任务 #{{ selectedTask.id }} · {{ selectedTask.name }}</small>
              </span>
            </div>
            <div class="bound-metrics">
              <div><span>验证指标</span><b>{{ taskMetric(selectedTask) }}</b></div>
              <div><span>数据集</span><b>{{ selectedTask.dataset_name }}</b></div>
              <div><span>权重体积</span><b>{{ taskFileSize(selectedTask) }}</b></div>
            </div>
            <div v-if="selectedTask.task_type === 'classification'" class="inference-param">
              <div class="param-head"><span>Top-K 输出</span><b>{{ topK }}</b></div>
              <el-slider
                v-model="topK"
                :min="1"
                :max="maxTopK"
                :step="1"
                @change="invalidateResult"
              />
              <div class="param-hint">最高值来自当前模型的类别数</div>
            </div>
            <div v-else-if="selectedTask.task_type === 'detection'" class="inference-param">
              <div class="param-head"><span>检测置信度阈值</span><b>{{ confidence.toFixed(2) }}</b></div>
              <el-slider
                v-model="confidence"
                :min="0.1"
                :max="0.9"
                :step="0.05"
                @change="invalidateResult"
              />
              <div class="param-hint">调高减少误检，调低可检出更多目标</div>
            </div>
          </aside>
        </div>
      </section>

      <section class="glass section workflow-step">
        <div class="step-head">
          <div class="step-title"><span>3</span>选择识别类型</div>
          <span class="step-note">单图识别或多图识别</span>
        </div>
        <div class="mode-grid">
          <button
            v-for="item in RECOGNITION_MODES"
            :key="item.value"
            type="button"
            class="mode-card"
            :class="{ active: recognitionMode === item.value }"
            :aria-pressed="recognitionMode === item.value"
            @click="setRecognitionMode(item.value)"
          >
            <span class="mode-icon"><el-icon><component :is="item.icon" /></el-icon></span>
            <span><strong>{{ item.label }}</strong><small>{{ item.desc }}</small></span>
          </button>
        </div>
      </section>

      <section class="glass section workflow-step">
        <div class="step-head">
          <div class="step-title">
            <span>4</span>{{ isBatch ? '选择多张图片' : '选择一张图片' }}
          </div>
          <el-button
            text
            type="primary"
            :loading="sampleLoading"
            :disabled="!selectedTask"
            @click="refreshSamples"
          >
            <el-icon><Refresh /></el-icon>
            换一组
          </el-button>
        </div>

        <div v-if="selectedTask" class="dataset-source">
          <span class="dataset-source-icon"><el-icon><DataBoard /></el-icon></span>
          <span class="dataset-source-copy">
            <strong>任务 #{{ selectedTask.id }} · {{ selectedTask.dataset_name }}</strong>
            <small>随机抽取 6 张，优先使用验证集样例</small>
          </span>
          <el-tag v-if="samples.length" size="small" effect="plain">{{ samples.length }} 张</el-tag>
        </div>

        <el-skeleton v-if="sampleLoading" :rows="3" animated />
        <el-alert
          v-else-if="sampleError"
          :title="sampleError"
          type="warning"
          :closable="false"
          show-icon
        />
        <template v-else>
          <div v-if="isBatch && samples.length" class="selection-toolbar">
            <span>已选择 <b>{{ batchTotalCount }}</b> 张图片</span>
            <span>
              <el-button text type="primary" size="small" @click="selectAllSamples">全选</el-button>
              <el-button text size="small" @click="clearSampleSelection">清空</el-button>
            </span>
          </div>
          <div v-if="samples.length" class="dataset-sample-grid" role="listbox" aria-label="任务数据集样例">
            <button
              v-for="(sample, index) in samples"
              :key="sample.key"
              type="button"
              class="dataset-sample-card"
              :class="{ active: isSampleSelected(sample) }"
              :aria-selected="isSampleSelected(sample)"
              role="option"
              @click="selectSample(sample)"
            >
              <img :src="sample.previewUrl" :alt="sample.name" />
              <span class="sample-order">{{ String(index + 1).padStart(2, '0') }}</span>
              <span v-if="isSampleSelected(sample)" class="sample-check">
                <el-icon><Check /></el-icon>
              </span>
              <span class="sample-name">{{ sample.name.split('/').pop() }}</span>
            </button>
          </div>
          <el-empty v-else :image-size="68" description="当前任务暂无可用样例" />
        </template>

        <div class="local-upload-row">
          <div>
            <strong>{{ isBatch ? '也可以补充上传多张本地图片' : '也可以上传本地图片' }}</strong>
            <small>JPG / PNG / WEBP · 单次最多 20 张</small>
          </div>
          <el-upload
            v-if="isBatch"
            multiple
            :auto-upload="false"
            :show-file-list="true"
            :file-list="batchFileListSafe"
            accept="image/*"
            :on-change="onBatchChange"
            :on-remove="removeBatchFile"
          >
            <el-button><el-icon><FolderAdd /></el-icon>选择图片</el-button>
          </el-upload>
          <el-upload
            v-else
            :auto-upload="false"
            :show-file-list="false"
            accept="image/*"
            :on-change="pickFile"
          >
            <el-button><el-icon><UploadFilled /></el-icon>本地上传</el-button>
          </el-upload>
        </div>

        <el-button
          type="primary"
          size="large"
          class="recognize-submit"
          :loading="isBatch ? batchLoading : loading"
          :disabled="!selectedTask || (isBatch ? !batchTotalCount : !singleSourceFile)"
          @click="runCurrentMode"
        >
          <el-icon><VideoPlay /></el-icon>
          {{ isBatch ? `开始多图识别（${batchTotalCount} 张）` : '开始单图识别' }}
        </el-button>
      </section>
    </div>

    <div v-else class="result-page">
      <div class="result-page-head">
        <div class="result-page-title">
          <el-button text @click="currentPage = 'config'">
            <el-icon><ArrowLeft /></el-icon>
            返回修改
          </el-button>
          <div>
            <h2>识别结果</h2>
            <p>本次识别使用的功能、训练任务、模型和识别类型</p>
          </div>
        </div>
        <div class="result-context">
          <el-tag effect="plain" type="warning">{{ modeMeta.label }}</el-tag>
          <el-tag effect="plain">{{ taskTypeMeta.label }}</el-tag>
          <el-tag v-if="selectedTask" effect="plain" type="success">
            任务 #{{ selectedTask.id }}
          </el-tag>
          <el-tag v-if="selectedTask" effect="plain">{{ selectedTask.architecture_name }}</el-tag>
        </div>
      </div>

      <template v-if="!isBatch && result">
        <div class="result-grid">
          <section class="glass section result-summary-card">
            <div class="section-title">
              <span>识别摘要</span>
              <span class="sub">{{ result.architecture }} · {{ result.latency_ms }} ms</span>
            </div>
            <div class="result-summary">
              <div>
                <div class="result-kicker">{{ taskTypeMeta.label }} · {{ isBatch ? '多图识别' : '单图识别' }}</div>
                <div class="result-main">{{ labelText(result.predicted_label) || '未检出目标' }}</div>
                <div class="result-note">结果已关联训练任务模型并写入识别历史</div>
              </div>
              <el-progress
                type="dashboard"
                :percentage="Number(((result.confidence || 0) * 100).toFixed(1))"
                :width="118"
                :stroke-width="10"
                color="#36d1c4"
              >
                <template #default="{ percentage }">
                  <div class="confidence-value">{{ percentage }}%</div>
                  <div class="confidence-label">置信度</div>
                </template>
              </el-progress>
            </div>
          </section>

          <section class="glass section result-detail-card">
            <template v-if="result.task_type === 'classification'">
              <div class="section-title">
                <span>Top-{{ result.topk.length }} 概率分布</span>
                <span class="sub">Softmax 输出</span>
              </div>
              <ProbBars :items="result.topk" />
              <EChart :option="probabilityOption" height="260px" />
            </template>

            <template v-else-if="result.task_type === 'detection'">
              <div class="section-title">
                <span>检测目标</span>
                <span class="sub">{{ result.object_count || 0 }} 个目标</span>
              </div>
              <div
                v-if="resultImageUrl"
                class="detection-preview"
                :style="{ aspectRatio: `${result.width || 4} / ${result.height || 3}` }"
              >
                <img :src="resultImageUrl" alt="目标检测结果" />
                <span
                  v-for="box in detectionBoxes"
                  :key="`${box.class_id}-${box.bbox.join('-')}`"
                  class="detection-box"
                  :style="{ ...box.style, '--box-color': box.color }"
                >
                  <b>{{ box.label }} {{ percent(box.confidence, 1) }}</b>
                </span>
              </div>
              <div class="object-list">
                <div v-for="(item, index) in result.detections" :key="`${item.label}-${index}`" class="object-row">
                  <span class="object-rank">{{ String(index + 1).padStart(2, '0') }}</span>
                  <strong>{{ item.label }}</strong>
                  <span>边界框 {{ item.bbox.map((value) => Math.round(value)).join(' · ') }}</span>
                  <b>{{ percent(item.confidence, 1) }}</b>
                </div>
                <el-empty v-if="!result.detections?.length" :image-size="58" description="当前阈值下未检出目标" />
              </div>
            </template>

            <template v-else-if="result.task_type === 'segmentation'">
              <div class="section-title">
                <span>语义分割掩膜</span>
                <span class="sub">识别类型 · {{ modeMeta.label }}</span>
              </div>
              <div
                v-if="resultImageUrl && segmentationMaskUrl"
                class="segmentation-preview"
                :style="{ aspectRatio: `${result.width || 4} / ${result.height || 3}` }"
              >
                <img :src="resultImageUrl" alt="原图" />
                <img :src="segmentationMaskUrl" alt="语义分割掩膜" :style="{ opacity: maskOpacity }" />
              </div>
              <div class="mask-opacity-row">
                <span>掩膜叠加透明度</span>
                <el-slider v-model="maskOpacity" :min="0" :max="1" :step="0.05" />
                <b>{{ Math.round(maskOpacity * 100) }}%</b>
              </div>
              <div class="seg-class-grid">
                <div v-for="item in result.class_stats" :key="item.index" class="seg-class-item">
                  <span>{{ item.label }}</span>
                  <b>{{ percent(item.percentage, 1) }}</b>
                </div>
              </div>
            </template>
          </section>
        </div>
      </template>

      <section v-else-if="isBatch && batchResult" class="glass section batch-results">
        <div class="section-title">
          <span>多图识别结果</span>
          <span class="sub">成功 {{ batchResult.total }} 张 · 失败 {{ batchResult.failed.length }} 张</span>
        </div>
        <el-alert
          v-if="batchResult.failed.length"
          :title="`有 ${batchResult.failed.length} 张图片识别失败`"
          type="warning"
          :closable="false"
          show-icon
          style="margin-bottom: 12px"
        />
        <el-table :data="batchResult.results" size="small" style="width: 100%">
          <el-table-column label="图片" width="86">
            <template #default="{ row }">
              <el-image :src="row.image_url" fit="cover" class="batch-thumb" />
            </template>
          </el-table-column>
          <el-table-column prop="name" label="文件名" min-width="145" show-overflow-tooltip />
          <el-table-column label="识别类型" width="95">
            <template #default>多图识别</template>
          </el-table-column>
          <el-table-column label="功能" width="92">
            <template #default="{ row }">
              <el-tag size="small" effect="plain" :type="taskTagType(row.task_type)">
                {{ taskLabel(row.task_type) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="识别结果" min-width="150">
            <template #default="{ row }">
              <div>{{ batchResultText(row) }}</div>
              <div class="table-subtext">{{ batchResultSubtext(row) }}</div>
            </template>
          </el-table-column>
          <el-table-column label="置信度" width="88">
            <template #default="{ row }">{{ percent(row.confidence, 1) }}</template>
          </el-table-column>
          <el-table-column label="耗时" width="88">
            <template #default="{ row }">{{ row.latency_ms }} ms</template>
          </el-table-column>
        </el-table>
      </section>

      <el-empty v-else :image-size="72" description="暂无识别结果，请返回配置页开始识别" />
    </div>
  </div>
</template>

<style scoped>
.recognize-page {
  min-width: 0;
}

.page-switch {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 14px;
  padding: 6px;
  border: 1px solid var(--ir-border);
  border-radius: 11px;
  background: rgba(15, 22, 40, 0.72);
}

.page-switch-item {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  min-height: 31px;
  padding: 4px 11px;
  border: 1px solid transparent;
  border-radius: 8px;
  color: var(--ir-text-dim);
  background: transparent;
  font-size: 12px;
}

.page-switch-item span {
  display: grid;
  place-items: center;
  width: 19px;
  height: 19px;
  border-radius: 5px;
  background: rgba(255, 255, 255, 0.06);
  font-size: 10px;
}

.page-switch-item.active {
  border-color: rgba(79, 140, 255, 0.28);
  color: var(--ir-text);
  background: rgba(79, 140, 255, 0.12);
}

.page-switch-item.active span {
  color: #07111e;
  background: var(--ir-accent);
}

.page-switch-item:disabled {
  opacity: 0.42;
}

.page-switch-arrow {
  color: var(--ir-text-dim);
}

.step-stack {
  display: grid;
  gap: 13px;
}

.workflow-step {
  padding: 0;
  overflow: hidden;
}

.workflow-step > :not(.step-head) {
  margin: 13px 16px 16px;
}

.step-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 47px;
  padding: 10px 15px;
  border-bottom: 1px solid var(--ir-border);
}

.step-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  font-weight: 600;
}

.step-title > span {
  display: grid;
  place-items: center;
  width: 24px;
  height: 24px;
  border-radius: 7px;
  color: #07111e;
  background: linear-gradient(135deg, var(--ir-primary), var(--ir-accent));
  font-size: 11px;
}

.step-note {
  color: var(--ir-text-dim);
  font-size: 11px;
}

.function-grid,
.mode-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.mode-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.function-card,
.mode-card {
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr);
  align-items: center;
  gap: 10px;
  min-width: 0;
  min-height: 68px;
  padding: 10px 12px;
  border: 1px solid var(--ir-border);
  border-radius: 11px;
  color: var(--ir-text);
  background: rgba(24, 33, 58, 0.65);
  text-align: left;
}

.function-card:hover,
.mode-card:hover {
  border-color: rgba(79, 140, 255, 0.55);
}

.function-card.active,
.mode-card.active {
  border-color: rgba(79, 140, 255, 0.78);
  background:
    linear-gradient(135deg, rgba(79, 140, 255, 0.2), rgba(54, 209, 196, 0.04)),
    rgba(24, 33, 58, 0.82);
}

.function-icon,
.mode-icon {
  display: grid;
  place-items: center;
  width: 38px;
  height: 38px;
  border-radius: 9px;
  color: var(--ir-primary);
  background: rgba(79, 140, 255, 0.12);
}

.function-icon :deep(svg),
.mode-icon :deep(svg) {
  width: 19px;
  height: 19px;
}

.function-copy,
.mode-card > span:last-child {
  display: grid;
  min-width: 0;
  gap: 3px;
}

.function-copy strong,
.mode-card strong {
  font-size: 13px;
  font-weight: 600;
}

.function-copy small,
.mode-card small {
  overflow: hidden;
  color: var(--ir-text-dim);
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-selection {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(250px, 0.55fr);
  gap: 12px;
}

.task-selector-fields {
  min-width: 0;
}

.field-row {
  display: grid;
  grid-template-columns: 62px minmax(0, 1fr);
  align-items: center;
  gap: 9px;
  margin-bottom: 9px;
}

.field-row label {
  color: var(--ir-text-dim);
  font-size: 12px;
}

.bound-model {
  padding: 11px;
  border: 1px solid rgba(54, 209, 196, 0.22);
  border-radius: 11px;
  background: rgba(54, 209, 196, 0.045);
}

.bound-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  color: var(--ir-text-dim);
  font-size: 11px;
}

.bound-head > span:first-child {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}

.bound-main {
  display: flex;
  align-items: center;
  gap: 9px;
  margin-top: 9px;
}

.model-cube {
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  flex: 0 0 32px;
  border-radius: 8px;
  color: var(--ir-accent);
  background: rgba(54, 209, 196, 0.1);
}

.bound-copy {
  display: grid;
  min-width: 0;
  gap: 2px;
}

.bound-copy strong,
.bound-copy small {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.bound-copy strong {
  font-size: 13px;
}

.bound-copy small {
  color: var(--ir-text-dim);
  font-size: 10px;
}

.bound-metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 7px;
  margin-top: 9px;
}

.bound-metrics > div {
  min-width: 0;
  padding: 7px 8px;
  border-radius: 8px;
  background: rgba(7, 11, 22, 0.38);
}

.bound-metrics span,
.bound-metrics b {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.bound-metrics span {
  color: var(--ir-text-dim);
  font-size: 10px;
}

.bound-metrics b {
  margin-top: 2px;
  font-size: 11px;
}

.inference-param {
  margin-top: 9px;
  padding-top: 8px;
  border-top: 1px solid var(--ir-border);
}

.param-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  color: var(--ir-text-dim);
  font-size: 11px;
}

.param-head b {
  color: var(--ir-text);
  font-weight: 500;
}

.param-hint {
  color: var(--ir-text-dim);
  font-size: 10px;
}

.dataset-source {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 9px;
  padding: 8px 9px;
  border: 1px solid rgba(79, 140, 255, 0.18);
  border-radius: 9px;
  background: rgba(79, 140, 255, 0.07);
}

.dataset-source-icon {
  display: grid;
  place-items: center;
  width: 25px;
  height: 25px;
  flex: 0 0 25px;
  border-radius: 7px;
  color: var(--ir-primary);
  background: rgba(79, 140, 255, 0.12);
}

.dataset-source-copy {
  display: grid;
  min-width: 0;
  flex: 1;
  gap: 2px;
}

.dataset-source-copy strong,
.dataset-source-copy small {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.dataset-source-copy strong {
  font-size: 11px;
}

.dataset-source-copy small {
  color: var(--ir-text-dim);
  font-size: 10px;
}

.selection-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 7px;
  color: var(--ir-text-dim);
  font-size: 11px;
}

.selection-toolbar b {
  color: var(--ir-text);
}

.dataset-sample-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 7px;
}

.dataset-sample-card {
  position: relative;
  min-width: 0;
  overflow: hidden;
  padding: 0;
  border: 1px solid var(--ir-border);
  border-radius: 9px;
  color: var(--ir-text);
  background: rgba(15, 22, 40, 0.8);
  text-align: left;
}

.dataset-sample-card:hover {
  border-color: rgba(79, 140, 255, 0.55);
}

.dataset-sample-card.active {
  border-color: var(--ir-primary);
  box-shadow: 0 0 0 1px rgba(79, 140, 255, 0.2);
}

.dataset-sample-card img {
  display: block;
  width: 100%;
  height: 86px;
  object-fit: cover;
}

.sample-order {
  position: absolute;
  top: 5px;
  left: 5px;
  padding: 1px 5px;
  border-radius: 4px;
  color: #fff;
  background: rgba(7, 11, 22, 0.7);
  font-size: 9px;
}

.sample-check {
  position: absolute;
  top: 5px;
  right: 5px;
  display: grid;
  place-items: center;
  width: 17px;
  height: 17px;
  border-radius: 4px;
  color: #fff;
  background: var(--ir-primary);
}

.sample-check :deep(svg) {
  width: 11px;
  height: 11px;
}

.sample-name {
  display: block;
  overflow: hidden;
  padding: 5px 6px;
  color: var(--ir-text-dim);
  font-size: 10px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.local-upload-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--ir-border);
}

.local-upload-row > div:first-child {
  display: grid;
  gap: 2px;
}

.local-upload-row strong {
  font-size: 11px;
}

.local-upload-row small {
  color: var(--ir-text-dim);
  font-size: 10px;
}

.recognize-submit {
  width: 100%;
  margin-top: 12px;
}

.result-page {
  display: grid;
  gap: 14px;
}

.result-page-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  padding: 13px 15px;
  border: 1px solid var(--ir-border);
  border-radius: 12px;
  background:
    linear-gradient(90deg, rgba(54, 209, 196, 0.07), transparent 55%),
    rgba(20, 27, 48, 0.86);
}

.result-page-title {
  display: flex;
  align-items: flex-start;
  gap: 10px;
}

.result-page-title h2 {
  margin: 0;
  font-size: 17px;
  font-weight: 600;
}

.result-page-title p {
  margin: 3px 0 0;
  color: var(--ir-text-dim);
  font-size: 11px;
}

.result-context {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 6px;
  flex-wrap: wrap;
}

.result-grid {
  display: grid;
  grid-template-columns: minmax(230px, 0.76fr) minmax(0, 1.24fr);
  gap: 14px;
}

.result-summary-card,
.result-detail-card,
.batch-results {
  min-width: 0;
}

.result-summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  min-height: 170px;
}

.result-kicker,
.result-note {
  color: var(--ir-text-dim);
  font-size: 11px;
}

.result-main {
  margin: 5px 0 8px;
  font-size: 25px;
  font-weight: 600;
}

.confidence-value {
  font-size: 20px;
  font-weight: 600;
}

.confidence-label {
  color: var(--ir-text-dim);
  font-size: 10px;
}

.detection-preview,
.segmentation-preview {
  position: relative;
  width: 100%;
  min-height: 0;
  overflow: hidden;
  border: 1px solid var(--ir-border);
  border-radius: 10px;
  background: #070b16;
}

.detection-preview > img,
.segmentation-preview > img:first-child {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: fill;
}

.detection-box {
  position: absolute;
  border: 1.5px solid var(--box-color);
  border-radius: 4px;
}

.detection-box b {
  position: absolute;
  top: -20px;
  left: -1px;
  padding: 2px 5px;
  border-radius: 3px 3px 0 0;
  color: #07111e;
  background: var(--box-color);
  font-size: 10px;
  font-weight: 500;
  white-space: nowrap;
}

.object-list {
  display: grid;
  gap: 6px;
  margin-top: 10px;
}

.object-row {
  display: grid;
  grid-template-columns: 24px minmax(90px, 0.8fr) minmax(120px, 1.2fr) auto;
  align-items: center;
  gap: 8px;
  padding: 7px 9px;
  border-radius: 8px;
  background: rgba(24, 33, 58, 0.65);
  color: var(--ir-text-dim);
  font-size: 11px;
}

.object-row strong {
  color: var(--ir-text);
}

.object-row b {
  color: var(--ir-accent);
  font-weight: 500;
}

.object-rank {
  color: #6d7c9f;
}

.segmentation-preview > img:last-child {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: fill;
  pointer-events: none;
  transition: opacity 0.12s linear;
}

.mask-opacity-row {
  display: grid;
  grid-template-columns: auto minmax(100px, 1fr) 42px;
  align-items: center;
  gap: 10px;
  margin-top: 10px;
  color: var(--ir-text-dim);
  font-size: 11px;
}

.mask-opacity-row b {
  color: var(--ir-text);
  text-align: right;
  font-weight: 500;
}

.seg-class-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
  margin-top: 8px;
}

.seg-class-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 7px 9px;
  border-radius: 8px;
  background: rgba(24, 33, 58, 0.65);
  color: var(--ir-text-dim);
  font-size: 11px;
}

.seg-class-item b {
  color: var(--ir-text);
  font-weight: 500;
}

.batch-thumb {
  width: 50px;
  height: 42px;
  border-radius: 7px;
}

.table-subtext {
  margin-top: 2px;
  color: var(--ir-text-dim);
  font-size: 10px;
}

@media (max-width: 880px) {
  .task-selection,
  .result-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .dataset-sample-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 680px) {
  .function-grid,
  .mode-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .step-head,
  .result-page-head,
  .result-summary,
  .local-upload-row {
    align-items: flex-start;
    flex-direction: column;
  }

  .field-row {
    grid-template-columns: minmax(0, 1fr);
  }

  .bound-metrics,
  .seg-class-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .dataset-sample-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
