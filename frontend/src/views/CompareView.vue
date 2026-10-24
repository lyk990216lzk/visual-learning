<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import ProbBars from '../components/ProbBars.vue'
import { useAppStore } from '../store'
import { fixed, labelText, percent } from '../utils/format'
import { logAction } from '../utils/operationLog'

const store = useAppStore()

const TASK_OPTIONS = [
  { value: 'classification', label: '图片识别', desc: '比较类别与概率', icon: 'Picture' },
  { value: 'detection', label: '目标检测', desc: '比较目标实例与检测框', icon: 'Aim' },
  { value: 'segmentation', label: '语义分割', desc: '比较像素类别与掩膜', icon: 'Crop' }
]
const TASK_LABELS = Object.fromEntries(TASK_OPTIONS.map((item) => [item.value, item.label]))
const PALETTE = ['#4f8cff', '#36d1c4', '#ffb454', '#ff7187', '#b499ff', '#58d6a2']

const currentPage = ref('config')
const activeTask = ref('classification')
const tasks = ref([])
const selectedDatasetId = ref(null)
const selectedArchitectures = ref([])
const selectedTaskByArchitecture = ref({})
const taskLoading = ref(false)
const taskError = ref('')

const samples = ref([])
const sampleLoading = ref(false)
const sampleError = ref('')
const selectedSampleKey = ref('')
const localFile = ref(null)
const localPreviewUrl = ref('')
const uploadRef = ref(null)

const loading = ref(false)
const result = ref(null)
const sampleRequestId = ref(0)

const taskTypeMeta = computed(
  () => TASK_OPTIONS.find((item) => item.value === activeTask.value) || TASK_OPTIONS[0]
)
const availableDatasets = computed(() => {
  const map = new Map()
  tasks.value.forEach((task) => {
    if (!map.has(task.dataset)) {
      map.set(task.dataset, {
        id: task.dataset,
        name: task.dataset_name,
        imageCount: task.dataset_image_count
      })
    }
  })
  return Array.from(map.values())
})
const datasetTasks = computed(
  () => tasks.value.filter((task) => task.dataset === selectedDatasetId.value)
)
const modelGroups = computed(() => {
  const groups = new Map()
  datasetTasks.value.forEach((task) => {
    if (!groups.has(task.architecture)) {
      groups.set(task.architecture, {
        key: task.architecture,
        name: task.architecture_name || task.architecture,
        tasks: []
      })
    }
    groups.get(task.architecture).tasks.push(task)
  })
  return Array.from(groups.values())
})
const selectedModels = computed(
  () => modelGroups.value.filter((item) => selectedArchitectures.value.includes(item.key))
)
const selectedTaskDetails = computed(
  () => selectedModels.value
    .map((model) => ({
      model,
      task: model.tasks.find((item) => item.id === selectedTaskByArchitecture.value[model.key])
    }))
    .filter((item) => item.task)
)
const selectedSample = computed(
  () => samples.value.find((item) => item.key === selectedSampleKey.value) || null
)
const compareFile = computed(() => localFile.value || selectedSample.value?.file || null)
const previewUrl = computed(() => localPreviewUrl.value || selectedSample.value?.previewUrl || '')
const compareImageUrl = computed(() => result.value?.image_url || previewUrl.value)
const canCompare = computed(
  () => selectedArchitectures.value.length >= 2
    && selectedArchitectures.value.length <= 6
    && selectedTaskDetails.value.length === selectedArchitectures.value.length
    && Boolean(compareFile.value)
)

const probabilityCompareOption = computed(() => {
  const outputs = result.value?.results || []
  const labels = Array.from(
    new Set(outputs.flatMap((item) => (item.topk || []).slice(0, 3).map((entry) => entry.label)))
  )
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { bottom: 0, textStyle: { color: '#97a3c4', fontSize: 11 } },
    grid: { left: 46, right: 16, top: 18, bottom: 56 },
    xAxis: {
      type: 'category',
      data: outputs.map((item) => item.architecture_name || item.architecture),
      axisLabel: { color: '#97a3c4', fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      max: 1,
      axisLabel: { color: '#97a3c4', formatter: (value) => `${value * 100}%` }
    },
    series: labels.map((label, index) => ({
      name: labelText(label),
      type: 'bar',
      barMaxWidth: 22,
      itemStyle: { color: PALETTE[index % PALETTE.length], borderRadius: [4, 4, 0, 0] },
      data: outputs.map((item) => (
        (item.topk || []).find((entry) => entry.label === label)?.probability || 0
      ))
    }))
  }
})

const referenceResult = computed(() => (
  result.value?.results?.find((item) => item.model_id === result.value?.reference_model_id) || null
))
const detectionComparison = computed(() => result.value?.detection_comparison || null)
const segmentationComparison = computed(() => result.value?.segmentation_comparison || null)
const referenceBoxes = computed(() => {
  const comparison = detectionComparison.value
  const reference = referenceResult.value
  if (!comparison || !reference?.width || !reference?.height) return []
  return (comparison.reference_instances || []).map((item, index) => {
    const [x1, y1, x2, y2] = item.bbox || [0, 0, 0, 0]
    return {
      ...item,
      color: PALETTE[index % PALETTE.length],
      style: {
        left: `${(x1 / reference.width) * 100}%`,
        top: `${(y1 / reference.height) * 100}%`,
        width: `${((x2 - x1) / reference.width) * 100}%`,
        height: `${((y2 - y1) / reference.height) * 100}%`
      }
    }
  })
})
const segmentationMaskUrl = computed(() => (
  segmentationComparison.value?.reference_mask_base64
    ? `data:image/png;base64,${segmentationComparison.value.reference_mask_base64}`
    : ''
))

onMounted(async () => {
  if (!store.system) store.bootstrap().catch(() => {})
  await loadTasks()
})

watch(activeTask, async () => {
  invalidateResult()
  selectedDatasetId.value = null
  selectedArchitectures.value = []
  selectedTaskByArchitecture.value = {}
  await loadTasks()
})

onBeforeUnmount(() => {
  sampleRequestId.value += 1
  releaseSampleUrls(samples.value)
  if (localPreviewUrl.value) URL.revokeObjectURL(localPreviewUrl.value)
})

async function loadTasks() {
  taskLoading.value = true
  taskError.value = ''
  try {
    const data = await api.recognizableTrainingRuns({
      task_type: activeTask.value,
      limit: 200
    })
    tasks.value = data.results || []
    const firstDataset = availableDatasets.value[0]
    if (!firstDataset) {
      selectedDatasetId.value = null
      samples.value = []
      return
    }
    selectedDatasetId.value = firstDataset.id
    resetDatasetModels()
    await loadSamples()
  } catch (error) {
    tasks.value = []
    selectedDatasetId.value = null
    samples.value = []
    taskError.value = error?.message || '可对比训练任务加载失败'
  } finally {
    taskLoading.value = false
  }
}

async function selectDataset(datasetId) {
  selectedDatasetId.value = datasetId
  resetDatasetModels()
  invalidateResult()
  await loadSamples()
}

function resetDatasetModels() {
  selectedArchitectures.value = modelGroups.value.slice(0, 4).map((item) => item.key)
  const mappings = {}
  selectedArchitectures.value.forEach((key) => {
    const model = modelGroups.value.find((item) => item.key === key)
    if (model) mappings[key] = model.tasks[0]?.id
  })
  selectedTaskByArchitecture.value = mappings
}

function toggleModel(model) {
  if (selectedArchitectures.value.includes(model.key)) {
    selectedArchitectures.value = selectedArchitectures.value.filter((key) => key !== model.key)
    const mappings = { ...selectedTaskByArchitecture.value }
    delete mappings[model.key]
    selectedTaskByArchitecture.value = mappings
  } else {
    if (selectedArchitectures.value.length >= 6) {
      ElMessage.warning('最多只能选择 6 个模型')
      return
    }
    selectedArchitectures.value = [...selectedArchitectures.value, model.key]
    selectedTaskByArchitecture.value = {
      ...selectedTaskByArchitecture.value,
      [model.key]: model.tasks[0]?.id
    }
  }
  invalidateResult()
}

async function loadSamples() {
  const requestId = sampleRequestId.value + 1
  sampleRequestId.value = requestId
  releaseSampleUrls(samples.value)
  samples.value = []
  selectedSampleKey.value = ''
  localFile.value = null
  if (localPreviewUrl.value) {
    URL.revokeObjectURL(localPreviewUrl.value)
    localPreviewUrl.value = ''
  }
  uploadRef.value?.clearFiles()
  sampleError.value = ''
  if (!selectedDatasetId.value) return
  sampleLoading.value = true
  try {
    const data = await api.datasetSamples(selectedDatasetId.value, { limit: 6, t: Date.now() })
    const loaded = await Promise.all(
      (data.results || []).map(async (item, index) => {
        try {
          const blob = await api.datasetImageBlob(selectedDatasetId.value, {
            label: item.label,
            name: item.name,
            split: item.split || ''
          })
          const fileName = String(item.name).split('/').pop()
          return {
            ...item,
            index,
            key: `${item.split || ''}|${item.label}|${item.name}`,
            file: new File([blob], fileName, { type: blob.type || 'image/jpeg' }),
            previewUrl: URL.createObjectURL(blob)
          }
        } catch (error) {
          return null
        }
      })
    )
    if (requestId !== sampleRequestId.value) {
      loaded.filter(Boolean).forEach((item) => URL.revokeObjectURL(item.previewUrl))
      return
    }
    samples.value = loaded.filter(Boolean)
    selectedSampleKey.value = samples.value[0]?.key || ''
  } catch (error) {
    if (requestId === sampleRequestId.value) {
      sampleError.value = error?.message || '对比图片加载失败'
    }
  } finally {
    if (requestId === sampleRequestId.value) sampleLoading.value = false
  }
}

function releaseSampleUrls(items) {
  items.forEach((item) => {
    if (item.previewUrl) URL.revokeObjectURL(item.previewUrl)
  })
}

function selectSample(sample) {
  selectedSampleKey.value = sample.key
  localFile.value = null
  uploadRef.value?.clearFiles()
  if (localPreviewUrl.value) {
    URL.revokeObjectURL(localPreviewUrl.value)
    localPreviewUrl.value = ''
  }
  invalidateResult()
}

function pickFile(uploadFile) {
  const raw = uploadFile.raw || uploadFile
  localFile.value = raw
  selectedSampleKey.value = ''
  if (localPreviewUrl.value) URL.revokeObjectURL(localPreviewUrl.value)
  localPreviewUrl.value = URL.createObjectURL(raw)
  invalidateResult()
  return false
}

async function finishCompare() {
  if (!selectedDatasetId.value) {
    ElMessage.warning('请选择数据集')
    return
  }
  if (selectedArchitectures.value.length < 2) {
    ElMessage.warning('至少选择 2 个模型')
    return
  }
  if (selectedArchitectures.value.length > 6) {
    ElMessage.warning('最多选择 6 个模型')
    return
  }
  if (!compareFile.value) {
    ElMessage.warning('请从数据集选择图片或上传本地图片')
    return
  }
  const modelIds = selectedTaskDetails.value.map((item) => item.task.trained_model_id)
  if (modelIds.length !== selectedArchitectures.value.length) {
    ElMessage.warning('请为每个模型选择训练任务')
    return
  }

  const formData = new FormData()
  formData.append('image', compareFile.value)
  formData.append('trained_models', modelIds.join(','))
  formData.append('task_type', activeTask.value)
  formData.append('top_k', '3')
  formData.append('confidence', '0.25')
  formData.append('iou_threshold', '0.5')
  formData.append('device', store.trainingDevice || store.system?.device || 'cpu')
  loading.value = true
  try {
    result.value = await api.recognizeCompare(formData)
    currentPage.value = 'result'
    ElMessage.success(`对比完成，共 ${result.value.successful_model_count} 个模型`)
    logAction({
      action: '多模型对比完成',
      detail: `${TASK_LABELS[activeTask.value]} · ${result.value.successful_model_count} 个模型`,
      module: '识别'
    })
  } catch (error) {
    result.value = null
  } finally {
    loading.value = false
  }
}

function invalidateResult() {
  result.value = null
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

function matchCell(instance, modelId) {
  return instance.matches?.[String(modelId)] || { status: 'missed', iou: 0, label: '' }
}

function matchCellText(instance, modelId) {
  if (modelId === result.value?.reference_model_id) {
    return `✓ 参考 ${percent(instance.confidence, 1)}`
  }
  const match = matchCell(instance, modelId)
  if (match.status === 'matched') return `✓ ${match.iou.toFixed(2)}`
  if (match.status === 'class_conflict') return `冲突 ${match.label}`
  return '—'
}

function matchCellClass(instance, modelId) {
  if (modelId === result.value?.reference_model_id) {
    return { matched: true, conflict: false, missed: false }
  }
  const match = matchCell(instance, modelId)
  return {
    matched: match.status === 'matched',
    conflict: match.status === 'class_conflict',
    missed: match.status === 'missed'
  }
}
</script>

<template>
  <div class="page compare-page">
    <nav class="compare-page-switch" aria-label="模型对比页面">
      <button
        type="button"
        :class="{ active: currentPage === 'config' }"
        @click="currentPage = 'config'"
      >
        <span>1</span>对比配置
      </button>
      <el-icon><ArrowRight /></el-icon>
      <button
        type="button"
        :class="{ active: currentPage === 'result' }"
        :disabled="!result"
        @click="currentPage = 'result'"
      >
        <span>2</span>对比结果
      </button>
    </nav>

    <div v-if="currentPage === 'config'" class="compare-steps">
      <section class="glass section compare-step">
        <div class="step-head">
          <div class="step-title"><span>1</span>选择对比功能</div>
          <span class="step-note">图片识别 / 目标检测 / 语义分割</span>
        </div>
        <div class="compare-function-grid">
          <button
            v-for="item in TASK_OPTIONS"
            :key="item.value"
            type="button"
            class="compare-function"
            :class="{ active: activeTask === item.value }"
            :aria-pressed="activeTask === item.value"
            @click="activeTask = item.value"
          >
            <span class="compare-function-icon"><el-icon><component :is="item.icon" /></el-icon></span>
            <span><strong>{{ item.label }}</strong><small>{{ item.desc }}</small></span>
          </button>
        </div>
      </section>

      <section class="glass section compare-step">
        <div class="step-head">
          <div class="step-title"><span>2</span>选择数据集</div>
          <span class="step-note">仅显示该功能下已有完成模型的数据集</span>
        </div>
        <el-skeleton v-if="taskLoading" :rows="2" animated />
        <el-alert v-else-if="taskError" :title="taskError" type="error" :closable="false" show-icon />
        <el-empty
          v-else-if="!availableDatasets.length"
          :image-size="72"
          :description="`还没有可用于${taskTypeMeta.label}对比的数据集`"
        />
        <div v-else class="dataset-field">
          <label for="compare-dataset">数据集</label>
          <el-select
            id="compare-dataset"
            :model-value="selectedDatasetId"
            style="width: 100%"
            @change="selectDataset"
          >
            <el-option
              v-for="dataset in availableDatasets"
              :key="dataset.id"
              :value="dataset.id"
              :label="`${dataset.name}（${dataset.imageCount} 张图片）`"
            />
          </el-select>
        </div>
      </section>

      <section class="glass section compare-step">
        <div class="step-head">
          <div class="step-title"><span>3</span>选择对比模型</div>
          <span class="step-note">至少 2 个，最多 6 个</span>
        </div>
        <div class="model-count-row">
          <span>同数据集已完成模型</span>
          <span>已选择 <b>{{ selectedArchitectures.length }}</b> / 6</span>
        </div>
        <div class="compare-model-grid">
          <button
            v-for="model in modelGroups"
            :key="model.key"
            type="button"
            class="compare-model-card"
            :class="{
              active: selectedArchitectures.includes(model.key),
              disabled: !selectedArchitectures.includes(model.key) && selectedArchitectures.length >= 6
            }"
            :disabled="!selectedArchitectures.includes(model.key) && selectedArchitectures.length >= 6"
            :aria-pressed="selectedArchitectures.includes(model.key)"
            @click="toggleModel(model)"
          >
            <span class="model-check">{{ selectedArchitectures.includes(model.key) ? '✓' : '' }}</span>
            <strong>{{ model.name }}</strong>
            <small>{{ model.tasks.length }} 个已完成任务 · {{ taskMetric(model.tasks[0]) }}</small>
          </button>
        </div>
        <el-alert
          v-if="selectedArchitectures.length < 2"
          type="warning"
          :closable="false"
          show-icon
          title="至少选择 2 个模型才能开始对比"
          style="margin-top: 10px"
        />
      </section>

      <section class="glass section compare-step">
        <div class="step-head">
          <div class="step-title"><span>4</span>选择对比图片</div>
          <el-button text type="primary" :loading="sampleLoading" @click="loadSamples">
            <el-icon><Refresh /></el-icon>换一组
          </el-button>
        </div>
        <el-skeleton v-if="sampleLoading" :rows="2" animated />
        <el-alert v-else-if="sampleError" :title="sampleError" type="warning" :closable="false" show-icon />
        <template v-else>
          <div class="sample-caption">
            <span>从当前数据集随机抽取 6 张图片，所有模型使用同一张图片</span>
            <span>{{ selectedSample?.name.split('/').pop() || '未选择' }}</span>
          </div>
          <div class="compare-sample-grid">
            <button
              v-for="(sample, index) in samples"
              :key="sample.key"
              type="button"
              class="compare-sample-card"
              :class="{ active: selectedSampleKey === sample.key }"
              @click="selectSample(sample)"
            >
              <img :src="sample.previewUrl" :alt="sample.name" />
              <span class="sample-index">{{ String(index + 1).padStart(2, '0') }}</span>
              <span v-if="selectedSampleKey === sample.key" class="sample-selected"><el-icon><Check /></el-icon></span>
              <span class="sample-file">{{ sample.name.split('/').pop() }}</span>
            </button>
          </div>
          <div class="local-image-row">
            <span>也可以上传本地图片</span>
            <el-upload
              ref="uploadRef"
              :auto-upload="false"
              :show-file-list="false"
              accept="image/*"
              :on-change="pickFile"
            >
              <el-button><el-icon><UploadFilled /></el-icon>本地上传</el-button>
            </el-upload>
          </div>
          <div v-if="previewUrl" class="compare-preview">
            <img :src="previewUrl" alt="当前对比图片" />
          </div>
        </template>
      </section>

      <section class="glass section compare-step">
        <div class="step-head">
          <div class="step-title"><span>5</span>选择模型下的训练任务</div>
          <span class="step-note">每个模型选择一个已完成任务</span>
        </div>
        <div v-if="selectedModels.length" class="task-binding-grid">
          <div v-for="model in selectedModels" :key="model.key" class="task-binding">
            <span class="binding-index">{{ model.name.slice(0, 2) }}</span>
            <div>
              <label>{{ model.name }} · 训练任务</label>
              <el-select v-model="selectedTaskByArchitecture[model.key]" style="width: 100%">
                <el-option
                  v-for="task in model.tasks"
                  :key="task.id"
                  :value="task.id"
                  :label="`#${task.id} ${task.name} · ${taskMetric(task)}`"
                />
              </el-select>
            </div>
          </div>
        </div>
        <el-empty v-else :image-size="60" description="先选择需要对比的模型" />
      </section>

      <section class="glass section compare-action">
        <div>
          <strong>{{ selectedArchitectures.length }} 个模型将对比同一张图片</strong>
          <span>功能与数据集：{{ taskTypeMeta.label }} · {{ availableDatasets.find((item) => item.id === selectedDatasetId)?.name || '未选择' }}</span>
        </div>
        <el-button
          type="primary"
          size="large"
          :loading="loading"
          :disabled="!canCompare"
          @click="finishCompare"
        >
          <el-icon><VideoPlay /></el-icon>
          6. 开始多模型对比
        </el-button>
      </section>
    </div>

    <div v-else class="compare-result-page">
      <div class="result-page-head">
        <div class="result-page-title">
          <el-button text @click="currentPage = 'config'">
            <el-icon><ArrowLeft /></el-icon>返回修改
          </el-button>
          <div>
            <h2>多模型对比结果</h2>
            <p>相同功能、相同数据集、相同图片</p>
          </div>
        </div>
        <div class="result-context">
          <el-tag effect="plain">{{ TASK_LABELS[result.task_type] }}</el-tag>
          <el-tag effect="plain" type="success">{{ result.dataset_name }}</el-tag>
          <el-tag effect="plain" type="warning">{{ result.successful_model_count }} 个模型</el-tag>
          <el-tag effect="plain">参考模型 {{ result.reference_model_name }}</el-tag>
        </div>
      </div>

      <template v-if="result.task_type === 'classification'">
        <div class="classification-summary-grid">
          <section class="glass section consensus-card">
            <div class="step-title"><span>✓</span>共识结果</div>
            <div class="consensus-label">{{ labelText(result.consensus_label) }}</div>
            <div class="consensus-meta">
              一致率 {{ percent(result.agreement, 1) }} ·
              最佳模型 {{ result.best_model_name }} ·
              {{ fixed(result.latency_ms, 1) }} ms
            </div>
          </section>
          <section class="glass section">
            <div class="section-title"><span>模型概率对比</span><span class="sub">Top-3 类别</span></div>
            <EChart :option="probabilityCompareOption" height="280px" />
          </section>
        </div>
        <div class="classification-model-grid">
          <section v-for="(item, index) in result.results" :key="item.model_id" class="glass section model-result-card">
            <div class="model-result-head">
              <span><i :style="{ background: PALETTE[index % PALETTE.length] }"></i>{{ item.architecture_name }}</span>
              <el-tag size="small" type="success">{{ percent(item.confidence, 1) }}</el-tag>
            </div>
            <div class="model-result-main">
              预测：<b>{{ labelText(item.predicted_label) }}</b>
              <span>{{ item.latency_ms }} ms · 任务 #{{ item.training_run_id }}</span>
            </div>
            <ProbBars :items="item.topk" />
          </section>
        </div>
      </template>

      <template v-else-if="result.task_type === 'detection' && detectionComparison">
        <div class="detection-summary-grid">
          <section class="glass section detection-image-panel">
            <div class="section-title">
              <span>参考模型检测框</span>
              <span class="sub">{{ detectionComparison.reference_model_name }} · IoU {{ detectionComparison.iou_threshold }}</span>
            </div>
            <div
              v-if="compareImageUrl && referenceResult"
              class="detection-image"
              :style="{ aspectRatio: `${referenceResult.width || 4} / ${referenceResult.height || 3}` }"
            >
              <img :src="compareImageUrl" alt="多模型检测对比图片" />
              <span
                v-for="box in referenceBoxes"
                :key="box.id"
                class="result-box"
                :style="{ ...box.style, '--box-color': box.color }"
              >
                <b>{{ box.label }} · {{ percent(box.confidence, 1) }}</b>
              </span>
            </div>
          </section>
          <section class="glass section comprehensive-panel">
            <div class="section-title">
              <span>综合检测结果</span>
              <span class="sub">以验证指标最佳模型为参考</span>
            </div>
            <div class="comprehensive-main">
              {{ detectionComparison.class_count }} 类 · {{ detectionComparison.object_count }} 个目标
            </div>
            <div class="class-result-grid">
              <div v-for="item in detectionComparison.classes" :key="item.name" class="class-result">
                <span>{{ item.name }}</span>
                <b>× {{ item.count }}</b>
                <small>模型覆盖 {{ item.covered_models }}/{{ item.total_models }} · 平均置信度 {{ percent(item.average_confidence, 1) }}</small>
              </div>
            </div>
            <div class="detection-metrics">
              <span>实例一致性 <b>{{ percent(detectionComparison.instance_consistency, 1) }}</b></span>
              <span>最佳模型 <b>{{ result.best_model_name }}</b></span>
              <span>类别冲突 <b>{{ detectionComparison.class_conflicts }}</b></span>
              <span>额外检测 <b>{{ detectionComparison.extra_detections.length }}</b></span>
            </div>
          </section>
        </div>

        <section class="glass section instance-matrix-panel">
          <div class="section-title">
            <span>目标级模型一致性</span>
            <span class="sub">相同类别优先，按检测框 IoU 一对一匹配</span>
          </div>
          <div class="instance-table-wrap">
            <table class="instance-table">
              <thead>
                <tr>
                  <th>参考实例</th>
                  <th v-for="item in result.results" :key="item.model_id">{{ item.architecture_name }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="instance in detectionComparison.reference_instances" :key="instance.id">
                  <th>{{ instance.label }} #{{ instance.id.split('-').pop() }}</th>
                  <td
                    v-for="model in result.results"
                    :key="model.model_id"
                    :class="matchCellClass(instance, model.model_id)"
                  >
                    {{ matchCellText(instance, model.model_id) }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <div class="detection-model-grid">
          <section v-for="(item, index) in result.results" :key="item.model_id" class="glass section detection-model-card">
            <div class="model-result-head">
              <span><i :style="{ background: PALETTE[index % PALETTE.length] }"></i>{{ item.architecture_name }}</span>
              <el-tag size="small" type="success">{{ item.object_count }} 个目标</el-tag>
            </div>
            <div class="detection-model-meta">
              任务 #{{ item.training_run_id }} · {{ item.latency_ms }} ms
            </div>
            <div class="mini-detections">
              <span v-for="detection in item.detections" :key="`${detection.label}-${detection.bbox.join('-')}`">
                {{ detection.label }} · {{ percent(detection.confidence, 1) }}
              </span>
              <span v-if="!item.detections?.length" class="missed">未检出目标</span>
            </div>
          </section>
        </div>
      </template>

      <template v-else-if="result.task_type === 'segmentation' && segmentationComparison">
        <div class="detection-summary-grid">
          <section class="glass section detection-image-panel">
            <div class="section-title"><span>参考模型分割掩膜</span><span class="sub">{{ segmentationComparison.reference_model_name }}</span></div>
            <div v-if="compareImageUrl" class="segmentation-image">
              <img :src="compareImageUrl" alt="分割对比图片" />
              <img
                v-if="segmentationMaskUrl"
                :src="segmentationMaskUrl"
                alt="参考分割掩膜"
                style="opacity:.7"
              />
            </div>
          </section>
          <section class="glass section comprehensive-panel">
            <div class="section-title"><span>综合分割结果</span><span class="sub">最佳模型为参考</span></div>
            <div class="comprehensive-main">{{ segmentationComparison.classes.length }} 个类别</div>
            <div class="class-result-grid">
              <div v-for="item in segmentationComparison.classes" :key="item.index" class="class-result">
                <span>{{ item.label }}</span>
                <b>{{ percent(item.percentage, 1) }}</b>
                <small>参考模型像素占比</small>
              </div>
            </div>
            <div class="detection-metrics">
              <span>平均像素一致率 <b>{{ percent(segmentationComparison.mean_pixel_agreement, 1) }}</b></span>
              <span>最佳模型 <b>{{ result.best_model_name }}</b></span>
            </div>
          </section>
        </div>
        <div class="classification-model-grid">
          <section
            v-for="item in segmentationComparison.model_results"
            :key="item.model_id"
            class="glass section model-result-card"
          >
            <div class="model-result-head">
              <span>{{ item.architecture_name }}</span>
              <el-tag size="small" type="success">{{ percent(item.pixel_agreement, 1) }}</el-tag>
            </div>
            <div class="model-result-main">
              像素一致率 · 任务 #{{ item.training_run_id }} · {{ item.latency_ms }} ms
            </div>
          </section>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.compare-page {
  min-width: 0;
}

.compare-page-switch {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 14px;
  padding: 6px;
  border: 1px solid var(--ir-border);
  border-radius: 11px;
  background: rgba(15, 22, 40, 0.72);
}

.compare-page-switch button {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  min-height: 31px;
  padding: 4px 11px;
  border: 1px solid transparent;
  border-radius: 8px;
  color: var(--ir-text-dim);
  background: transparent;
}

.compare-page-switch button span {
  display: grid;
  place-items: center;
  width: 19px;
  height: 19px;
  border-radius: 5px;
  background: rgba(255, 255, 255, 0.06);
  font-size: 10px;
}

.compare-page-switch button.active {
  border-color: rgba(79, 140, 255, 0.28);
  color: var(--ir-text);
  background: rgba(79, 140, 255, 0.12);
}

.compare-page-switch button.active span {
  color: #07111e;
  background: var(--ir-accent);
}

.compare-page-switch button:disabled {
  opacity: 0.4;
}

.compare-steps,
.compare-result-page {
  display: grid;
  gap: 13px;
}

.compare-step {
  padding: 0;
  overflow: hidden;
}

.compare-step > :not(.step-head) {
  margin: 12px 16px 15px;
}

.step-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 10px 14px;
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

.step-note,
.model-count-row,
.sample-caption {
  color: var(--ir-text-dim);
  font-size: 11px;
}

.compare-function-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 9px;
}

.compare-function {
  display: grid;
  grid-template-columns: 37px minmax(0, 1fr);
  align-items: center;
  gap: 9px;
  min-height: 66px;
  padding: 9px 11px;
  border: 1px solid var(--ir-border);
  border-radius: 11px;
  color: var(--ir-text);
  background: rgba(24, 33, 58, 0.65);
  text-align: left;
}

.compare-function.active {
  border-color: rgba(79, 140, 255, 0.74);
  background:
    linear-gradient(135deg, rgba(79, 140, 255, 0.18), rgba(54, 209, 196, 0.035)),
    rgba(24, 33, 58, 0.82);
}

.compare-function-icon {
  display: grid;
  place-items: center;
  width: 37px;
  height: 37px;
  border-radius: 9px;
  color: var(--ir-primary);
  background: rgba(79, 140, 255, 0.12);
}

.compare-function strong,
.compare-function small {
  display: block;
}

.compare-function strong {
  font-size: 13px;
}

.compare-function small {
  margin-top: 2px;
  color: var(--ir-text-dim);
  font-size: 11px;
}

.dataset-field {
  display: grid;
  grid-template-columns: 62px minmax(0, 1fr);
  align-items: center;
  gap: 9px;
}

.dataset-field label {
  color: var(--ir-text-dim);
}

.model-count-row,
.sample-caption {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
}

.model-count-row b {
  color: var(--ir-text);
}

.compare-model-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.compare-model-card {
  position: relative;
  min-width: 0;
  padding: 10px;
  border: 1px solid var(--ir-border);
  border-radius: 10px;
  color: var(--ir-text);
  background: rgba(24, 33, 58, 0.65);
  text-align: left;
}

.compare-model-card.active {
  border-color: rgba(79, 140, 255, 0.74);
  background:
    linear-gradient(135deg, rgba(79, 140, 255, 0.17), rgba(54, 209, 196, 0.035)),
    rgba(24, 33, 58, 0.82);
}

.compare-model-card.disabled {
  opacity: 0.45;
}

.compare-model-card strong,
.compare-model-card small {
  display: block;
  overflow: hidden;
  padding-right: 22px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.compare-model-card strong {
  font-size: 13px;
}

.compare-model-card small {
  margin-top: 3px;
  color: var(--ir-text-dim);
  font-size: 10px;
}

.model-check {
  position: absolute;
  top: 9px;
  right: 9px;
  display: grid;
  place-items: center;
  width: 16px;
  height: 16px;
  border: 1px solid var(--ir-border);
  border-radius: 4px;
  color: #fff;
  font-size: 9px;
}

.compare-model-card.active .model-check {
  border-color: var(--ir-primary);
  background: var(--ir-primary);
}

.compare-sample-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 7px;
}

.compare-sample-card {
  position: relative;
  min-width: 0;
  overflow: hidden;
  padding: 0;
  border: 1px solid var(--ir-border);
  border-radius: 9px;
  color: var(--ir-text);
  background: rgba(15, 22, 40, 0.72);
  text-align: left;
}

.compare-sample-card.active {
  border-color: var(--ir-primary);
}

.compare-sample-card img {
  display: block;
  width: 100%;
  height: 82px;
  object-fit: cover;
}

.sample-index {
  position: absolute;
  top: 4px;
  left: 4px;
  padding: 1px 4px;
  border-radius: 4px;
  color: #fff;
  background: rgba(7, 11, 22, 0.72);
  font-size: 9px;
}

.sample-selected {
  position: absolute;
  top: 4px;
  right: 4px;
  display: grid;
  place-items: center;
  width: 16px;
  height: 16px;
  border-radius: 4px;
  color: #fff;
  background: var(--ir-primary);
}

.sample-file {
  display: block;
  overflow: hidden;
  padding: 4px 5px;
  color: var(--ir-text-dim);
  font-size: 9px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.local-image-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-top: 9px;
  color: var(--ir-text-dim);
  font-size: 11px;
}

.compare-preview {
  max-height: 260px;
  margin-top: 9px;
  overflow: hidden;
  border: 1px solid var(--ir-border);
  border-radius: 10px;
  background: #070b16;
}

.compare-preview img {
  display: block;
  width: 100%;
  max-height: 260px;
  object-fit: contain;
}

.task-binding-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

.task-binding {
  display: grid;
  grid-template-columns: 36px minmax(0, 1fr);
  align-items: center;
  gap: 8px;
  padding: 8px;
  border: 1px solid var(--ir-border);
  border-radius: 9px;
  background: rgba(24, 33, 58, 0.55);
}

.binding-index {
  display: grid;
  place-items: center;
  width: 36px;
  height: 36px;
  border-radius: 8px;
  color: var(--ir-primary);
  background: rgba(79, 140, 255, 0.12);
  font-size: 10px;
}

.task-binding label {
  display: block;
  margin-bottom: 4px;
  color: var(--ir-text-dim);
  font-size: 10px;
}

.compare-action {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  padding: 14px 16px;
}

.compare-action > div {
  display: grid;
  gap: 3px;
}

.compare-action strong {
  font-size: 13px;
}

.compare-action span {
  color: var(--ir-text-dim);
  font-size: 11px;
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
  gap: 9px;
}

.result-page-title h2 {
  margin: 0;
  font-size: 17px;
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

.classification-summary-grid,
.detection-summary-grid {
  display: grid;
  grid-template-columns: minmax(250px, 0.7fr) minmax(0, 1.3fr);
  gap: 12px;
}

.consensus-label {
  margin-top: 12px;
  font-size: 36px;
  font-weight: 700;
  background: linear-gradient(120deg, #ffb454, #ff6b81);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.consensus-meta {
  margin-top: 8px;
  color: var(--ir-text-dim);
  font-size: 11px;
}

.classification-model-grid,
.detection-model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
}

.model-result-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.model-result-head > span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-weight: 600;
}

.model-result-head i {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.model-result-main {
  margin: 8px 0 10px;
  font-size: 12px;
}

.model-result-main span {
  color: var(--ir-text-dim);
  font-size: 10px;
}

.detection-image,
.segmentation-image {
  position: relative;
  overflow: hidden;
  width: 100%;
  border: 1px solid var(--ir-border);
  border-radius: 10px;
  background: #070b16;
}

.detection-image img,
.segmentation-image > img:first-child {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: fill;
}

.result-box {
  position: absolute;
  border: 1.5px solid var(--box-color);
  border-radius: 4px;
}

.result-box b {
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

.comprehensive-main {
  margin-top: 5px;
  font-size: 22px;
  font-weight: 600;
}

.class-result-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 7px;
  margin-top: 10px;
}

.class-result {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 4px 8px;
  padding: 8px 9px;
  border-radius: 9px;
  background: rgba(24, 33, 58, 0.65);
}

.class-result span {
  color: var(--ir-text);
  font-size: 12px;
}

.class-result b {
  color: var(--ir-accent);
  font-size: 15px;
  font-weight: 500;
}

.class-result small {
  grid-column: 1 / -1;
  color: var(--ir-text-dim);
  font-size: 10px;
}

.detection-metrics {
  display: flex;
  align-items: center;
  gap: 7px;
  flex-wrap: wrap;
  margin-top: 9px;
}

.detection-metrics span {
  padding: 4px 7px;
  border-radius: 7px;
  color: var(--ir-text-dim);
  background: rgba(15, 22, 40, 0.6);
  font-size: 10px;
}

.detection-metrics b {
  color: var(--ir-text);
}

.instance-table-wrap {
  overflow-x: auto;
}

.instance-table {
  width: 100%;
  min-width: 620px;
  border-collapse: collapse;
  font-size: 11px;
}

.instance-table th,
.instance-table td {
  padding: 9px 10px;
  border-bottom: 1px solid var(--ir-border);
  text-align: left;
}

.instance-table th {
  color: var(--ir-text-dim);
  font-weight: 400;
}

.instance-table td {
  color: var(--ir-text-dim);
}

.instance-table td.matched {
  color: var(--ir-accent);
}

.instance-table td.conflict {
  color: var(--ir-warn);
}

.instance-table td.missed {
  color: var(--ir-faint);
}

.detection-model-meta {
  margin-top: 7px;
  color: var(--ir-text-dim);
  font-size: 10px;
}

.mini-detections {
  display: flex;
  gap: 5px;
  flex-wrap: wrap;
  margin-top: 8px;
}

.mini-detections span {
  padding: 3px 6px;
  border-radius: 6px;
  color: var(--ir-text-dim);
  background: rgba(24, 33, 58, 0.72);
  font-size: 10px;
}

.mini-detections .missed {
  color: var(--ir-faint);
}

.segmentation-image > img:last-child {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: fill;
  pointer-events: none;
}

@media (max-width: 900px) {
  .classification-summary-grid,
  .detection-summary-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .compare-model-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .compare-sample-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .compare-function-grid,
  .compare-model-grid,
  .task-binding-grid,
  .class-result-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .compare-sample-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .step-head,
  .result-page-head,
  .compare-action {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
