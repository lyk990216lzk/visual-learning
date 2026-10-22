<script setup>
import { computed, onMounted, onUnmounted, reactive, ref, watch, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import { useAppStore } from '../store'
import { STATUS_META, fixed, percent } from '../utils/format'
import { logAction } from '../utils/operationLog'

const store = useAppStore()
const formRef = ref(null)
const submitting = ref(false)
const runs = ref([])
const activeRunId = ref(null)
const activeRun = ref(null)
const progress = ref(null)
const logLines = ref([])
const logBox = ref(null)
const autoScroll = ref(true)
const deviceChecking = ref(true)
let lastAvailableDevice = 'cpu'
let timer = null

const form = reactive({
  name: '',
  architecture: 'lenet5',
  dataset: null,
  epochs: 10,
  batch_size: 32,
  learning_rate: 0.01,
  weight_decay: 0.0001,
  optimizer: 'sgd',
  scheduler: 'cosine',
  image_size: 64,
  val_split: 0.2,
  label_smoothing: 0.0,
  device: 'cpu',
  augment: true,
  max_batches_per_epoch: 0,
  class_weighted_loss: false
})

const rules = {
  architecture: [{ required: true, message: '请选择网络结构', trigger: 'change' }],
  dataset: [{ required: true, message: '请选择数据集', trigger: 'change' }]
}

const currentArchitecture = computed(() => store.architectureMap[form.architecture])
const currentDataset = computed(() => store.datasets.find((item) => item.id === form.dataset))

function sortDeviceOptions(options) {
  return [...options].sort((a, b) => Number(!a.available) - Number(!b.available))
}

const deviceOptions = computed(() => {
  const detected = store.system?.device_options
  if (Array.isArray(detected) && detected.length) return sortDeviceOptions(detected)
  return sortDeviceOptions([
    {
      value: 'gpu',
      label: 'GPU',
      available: Boolean(store.system?.gpu_available ?? store.system?.cuda_available),
      detail: '当前环境未检测到可用的 CUDA GPU'
    },
    {
      value: 'cpu',
      label: 'CPU',
      available: true,
      detail: `使用 ${store.system?.cpu_count || '多'} 个 CPU 线程`
    },
    {
      value: 'mps',
      label: 'MPS',
      available: Boolean(store.system?.mps_available),
      detail: '当前环境未检测到可用的 Apple MPS'
    }
  ])
})
const currentDevice = computed(
  () => deviceOptions.value.find((item) => item.value === form.device) || deviceOptions.value[1]
)

const recommendations = {
  lenet5: { learning_rate: 0.02, image_size: 32, epochs: 10, optimizer: 'adam' },
  alexnet: { learning_rate: 0.01, image_size: 64, epochs: 10, optimizer: 'sgd' },
  vgg11: { learning_rate: 0.01, image_size: 64, epochs: 10, optimizer: 'sgd' },
  vgg16: { learning_rate: 0.005, image_size: 64, epochs: 8, optimizer: 'sgd' },
  googlenet: { learning_rate: 0.01, image_size: 96, epochs: 8, optimizer: 'sgd' },
  resnet18: { learning_rate: 0.01, image_size: 64, epochs: 12, optimizer: 'sgd' },
  resnet50: { learning_rate: 0.008, image_size: 64, epochs: 10, optimizer: 'sgd' },
  densenet121: { learning_rate: 0.01, image_size: 64, epochs: 10, optimizer: 'sgd' },
  mobilenetv2: { learning_rate: 0.01, image_size: 64, epochs: 12, optimizer: 'sgd' },
  squeezenet: { learning_rate: 0.01, image_size: 64, epochs: 12, optimizer: 'sgd' }
}

const curveOption = computed(() => {
  const history = progress.value?.history || []
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: ['训练损失', '验证损失', '训练准确率', '验证准确率'], bottom: 0, textStyle: { color: '#97a3c4', fontSize: 11 } },
    grid: { left: 50, right: 50, top: 24, bottom: 56 },
    xAxis: { type: 'category', data: history.map((item) => `E${item.epoch}`) },
    yAxis: [
      { type: 'value', name: 'Loss', splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
      { type: 'value', name: 'Acc', min: 0, max: 1, axisLabel: { formatter: (v) => `${v * 100}%` } }
    ],
    series: [
      { name: '训练损失', type: 'line', smooth: true, data: history.map((i) => i.train_loss), itemStyle: { color: '#4f8cff' } },
      { name: '验证损失', type: 'line', smooth: true, data: history.map((i) => i.val_loss), itemStyle: { color: '#ff6b81' } },
      { name: '训练准确率', type: 'line', smooth: true, yAxisIndex: 1, data: history.map((i) => i.train_accuracy), itemStyle: { color: '#36d1c4' } },
      { name: '验证准确率', type: 'line', smooth: true, yAxisIndex: 1, data: history.map((i) => i.val_accuracy), itemStyle: { color: '#ffb454' } }
    ]
  }
})

async function loadRuns() {
  const data = await api.trainingRuns({ page_size: 50 })
  runs.value = data.results || []
  if (!activeRunId.value && runs.value.length) {
    await selectRun(runs.value[0])
  }
}

async function selectRun(run) {
  activeRunId.value = run.id
  activeRun.value = await api.trainingRun(run.id)
  store.setTrainingDevice(activeRun.value.device)
  progress.value = {
    status: activeRun.value.status,
    progress: activeRun.value.progress,
    history: activeRun.value.history,
    best_accuracy: activeRun.value.best_accuracy,
    current_epoch: activeRun.value.current_epoch,
    epochs: activeRun.value.epochs,
    latest_metrics: activeRun.value.latest_metrics
  }
  await loadLog()
}

async function loadLog() {
  if (!activeRunId.value) return
  const data = await api.trainingLog(activeRunId.value, 200)
  logLines.value = data.lines || []
  if (autoScroll.value) {
    await nextTick()
    if (logBox.value) logBox.value.scrollTop = logBox.value.scrollHeight
  }
}

async function poll() {
  const running = runs.value.filter((item) => item.status === 'running' || item.status === 'pending')
  if (running.length) {
    await loadRuns()
  }
  if (activeRunId.value) {
    const data = await api.trainingProgress(activeRunId.value).catch(() => null)
    if (data) {
      progress.value = data
      await loadLog()
      const target = runs.value.find((item) => item.id === activeRunId.value)
      if (target) {
        target.progress = data.progress
        target.status = data.status
        target.best_accuracy = data.best_accuracy
      }
      if (data.status === 'success' && activeRun.value?.status !== 'success') {
        ElMessage.success(`训练完成，验证准确率 ${percent(data.best_accuracy, 2)}`)
        logAction({
          action: '训练完成',
          detail: `${activeRun.value?.name || '训练任务'} · 验证准确率 ${percent(data.best_accuracy, 2)}`,
          module: '训练'
        })
        activeRun.value.status = 'success'
        await store.fetchTrainedModels()
        await store.fetchArchitectures()
      }
    }
  }
}

function useFirstAvailableDevice() {
  const preferred = ['gpu', 'mps', 'cpu']
  const option = preferred
    .map((value) => deviceOptions.value.find((item) => item.value === value))
    .find((item) => item?.available)
  form.device = option?.value || 'cpu'
  lastAvailableDevice = form.device
  store.setTrainingDevice(form.device)
}

async function refreshDeviceAvailability() {
  deviceChecking.value = true
  try {
    await store.fetchSystem()
    useFirstAvailableDevice()
  } catch {
    form.device = 'cpu'
    lastAvailableDevice = 'cpu'
    store.setTrainingDevice(form.device)
  } finally {
    deviceChecking.value = false
  }
}

async function handleDeviceChange(value) {
  const option = deviceOptions.value.find((item) => item.value === value)
  if (option?.available) {
    lastAvailableDevice = value
    store.setTrainingDevice(value)
    return
  }

  form.device = lastAvailableDevice || 'cpu'
  store.setTrainingDevice(form.device)
  const label = option?.label || String(value).toUpperCase()
  const alternative = deviceOptions.value
    .filter((item) => item.available && item.value !== value)
    .map((item) => item.label)
    .join(' 或 ')
  await ElMessageBox.alert(
    `当前运行环境没有可用的 ${label}，本次选择未生效。请选择 ${alternative || 'CPU'}。`,
    `${label} 不可用`,
    { type: 'warning', confirmButtonText: '知道了' }
  )
}

onMounted(async () => {
  await refreshDeviceAvailability()
  if (!store.architectures.length) await store.bootstrap()
  if (!store.datasets.length) await store.fetchDatasets()
  form.dataset = store.datasets[0]?.id || null
  const preset = recommendations[form.architecture]
  if (preset) Object.assign(form, preset)
  await loadRuns()
  timer = setInterval(poll, 2000)
})

onUnmounted(() => timer && clearInterval(timer))

watch(() => form.architecture, (key) => {
  const preset = recommendations[key]
  if (preset) {
    Object.assign(form, preset)
  }
  const meta = store.architectureMap[key]
  if (meta && form.image_size < meta.min_input_size) {
    form.image_size = Math.max(meta.default_input_size <= 224 ? 64 : 224, meta.min_input_size)
  }
})

async function buildDemoDataset() {
  ElMessage.info('正在生成演示数据集，请稍候…')
  const dataset = await api.buildDemoDataset({ per_class: 240, image_size: 64 })
  await store.fetchDatasets()
  form.dataset = dataset.id
  ElMessage.success(`演示数据集已就绪：${dataset.image_count} 张图片 / ${dataset.num_classes} 个类别`)
  logAction({
    action: '演示数据集已生成',
    detail: `${dataset.name} · ${dataset.image_count} 张图片 · ${dataset.num_classes} 个类别`,
    module: '数据集'
  })
}

async function submit() {
  await formRef.value?.validate()
  if (!store.datasets.length) {
    ElMessage.warning('请先创建数据集')
    return
  }
  const selectedDevice = deviceOptions.value.find((item) => item.value === form.device)
  if (!selectedDevice?.available) {
    await handleDeviceChange(form.device)
    return
  }
  submitting.value = true
  try {
    const run = await api.createTraining({
      ...form,
      name: form.name || `${form.architecture}-${new Date().toLocaleTimeString('zh-CN')}`
    })
    ElMessage.success('训练任务已启动，可在下方查看实时进度')
    form.name = ''
    await loadRuns()
    await selectRun(run)
  } finally {
    submitting.value = false
  }
}

async function cancel(run) {
  await api.cancelTraining(run.id)
  ElMessage.warning('已发送中止指令')
  await loadRuns()
}

async function removeRun(run) {
  await ElMessageBox.confirm(`确定删除训练任务「${run.name}」吗？训练产出的模型也会一并删除。`, '提示', {
    type: 'warning'
  })
  await api.deleteTraining(run.id)
  ElMessage.success('已删除')
  if (activeRunId.value === run.id) {
    activeRunId.value = null
    progress.value = null
    logLines.value = []
  }
  await loadRuns()
  await store.fetchTrainedModels()
}

function rowClass({ row }) {
  return row.id === activeRunId.value ? 'active-row' : ''
}
</script>

<template>
  <div class="page">
    <div class="grid grid-2">
      <div class="glass section">
        <div class="section-title">
          <span>训练配置</span>
          <span class="sub">自动填充推荐超参数，可手动微调</span>
        </div>
        <el-form ref="formRef" :model="form" :rules="rules" label-width="110px" size="default">
          <el-form-item label="任务名称">
            <el-input v-model="form.name" placeholder="留空则自动命名" clearable />
          </el-form-item>
          <el-form-item label="网络结构" prop="architecture">
            <el-select v-model="form.architecture" style="width: 100%">
              <el-option
                v-for="item in store.architectures"
                :key="item.key"
                :value="item.key"
                :label="`${item.name}（${item.year} · ${item.family}）`"
              />
            </el-select>
          </el-form-item>
          <el-form-item label="数据集" prop="dataset">
            <div style="display: flex; gap: 8px; width: 100%">
              <el-select v-model="form.dataset" placeholder="选择数据集" style="flex: 1">
                <el-option
                  v-for="item in store.datasets"
                  :key="item.id"
                  :value="item.id"
                  :label="`${item.name}（${item.image_count} 张 / ${item.num_classes} 类）`"
                />
              </el-select>
              <el-button @click="buildDemoDataset">生成演示集</el-button>
            </div>
          </el-form-item>
          <el-form-item label="输入尺寸">
            <el-slider v-model="form.image_size" :min="currentArchitecture?.min_input_size || 16"
                       :max="224" :step="8" show-input
                       style="width: 100%" />
          </el-form-item>
          <el-form-item label="训练轮数">
            <el-input-number v-model="form.epochs" :min="1" :max="200" />
            <span class="dim" style="margin-left: 10px; font-size: 12px">
              {{ currentDataset ? `样本 ${currentDataset.image_count} 张` : '请选择数据集' }}
            </span>
          </el-form-item>
          <el-form-item label="批大小">
            <el-input-number v-model="form.batch_size" :min="1" :max="512" :step="8" />
            <span class="dim" style="margin-left: 10px; font-size: 12px">
              每轮 batch 数约
              {{ currentDataset && form.batch_size
                ? Math.ceil((currentDataset.image_count * (1 - form.val_split)) / form.batch_size)
                : '--' }}
            </span>
          </el-form-item>
          <el-form-item label="学习率">
            <el-input-number v-model="form.learning_rate" :min="0.00001" :max="1" :step="0.001"
                             :precision="5" style="width: 180px" />
            <span class="dim" style="margin-left: 10px; font-size: 12px">权重衰减 {{ form.weight_decay }}</span>
          </el-form-item>
          <el-form-item label="优化器">
            <el-radio-group v-model="form.optimizer">
              <el-radio-button value="sgd">SGD</el-radio-button>
              <el-radio-button value="adam">Adam</el-radio-button>
              <el-radio-button value="adamw">AdamW</el-radio-button>
            </el-radio-group>
          </el-form-item>
          <el-form-item label="学习率策略">
            <el-radio-group v-model="form.scheduler">
              <el-radio-button value="cosine">Cosine</el-radio-button>
              <el-radio-button value="step">Step</el-radio-button>
              <el-radio-button value="none">固定</el-radio-button>
            </el-radio-group>
          </el-form-item>
          <el-form-item label="验证集比例">
            <el-slider v-model="form.val_split" :min="0" :max="0.5" :step="0.05" show-input
                       style="width: 100%" />
          </el-form-item>
          <el-form-item label="标签平滑">
            <el-slider v-model="form.label_smoothing" :min="0" :max="0.3" :step="0.05" show-input
                       style="width: 100%" />
          </el-form-item>
          <el-form-item label="计算设备">
            <el-radio-group v-model="form.device" :disabled="deviceChecking"
                            @change="handleDeviceChange">
              <el-radio-button
                v-for="item in deviceOptions"
                :key="item.value"
                :value="item.value"
                :class="{ 'device-unavailable': !item.available }"
              >
                <span>{{ item.label }}</span>
                <span class="device-state">{{ item.available ? '可用' : '不可用' }}</span>
              </el-radio-button>
            </el-radio-group>
            <div class="device-hint">
              {{ deviceChecking ? '正在检测训练设备…' : currentDevice?.detail }}
            </div>
          </el-form-item>
          <el-form-item label="数据增强">
            <el-switch v-model="form.augment" />
            <span class="dim" style="margin-left: 10px; font-size: 12px">
              随机裁剪 / 翻转 / 旋转 / 色彩抖动 / 随机擦除
            </span>
          </el-form-item>
          <el-form-item label="快速模式">
            <el-input-number v-model="form.max_batches_per_epoch" :min="0" :max="200" />
            <span class="dim" style="margin-left: 10px; font-size: 12px">
              每轮只训练前 N 个 batch（0 = 全部），用于快速验证流程
            </span>
          </el-form-item>
          <el-button type="primary" size="large" style="width: 100%" :loading="submitting"
                     @click="submit">
            <el-icon><VideoPlay /></el-icon>
            <span style="margin-left: 6px">开始训练</span>
          </el-button>
        </el-form>
      </div>

      <div>
        <div class="glass section">
          <div class="section-title">
            <span>当前结构说明</span>
            <el-tag v-if="currentArchitecture" size="small" type="warning" effect="dark">
              {{ currentArchitecture.year }}
            </el-tag>
          </div>
          <template v-if="currentArchitecture">
            <div style="font-size: 16px; font-weight: 600; margin-bottom: 6px">
              {{ currentArchitecture.name }}
              <span class="dim" style="font-size: 12px">· {{ currentArchitecture.family }}</span>
            </div>
            <div class="dim" style="font-size: 12px; line-height: 1.9">
              {{ currentArchitecture.description }}
            </div>
            <div class="tag-row">
              <el-tag v-for="item in currentArchitecture.highlights" :key="item" size="small"
                      effect="plain" type="success">{{ item }}</el-tag>
            </div>
            <el-divider />
            <div class="dim" style="font-size: 12px; line-height: 1.9">
              <div>论文：{{ currentArchitecture.paper }}</div>
              <div>作者：{{ currentArchitecture.authors }}</div>
              <div>建议输入尺寸：{{ currentArchitecture.default_input_size }}px（最小
                {{ currentArchitecture.min_input_size }}px）</div>
              <div v-if="currentArchitecture.trained_count">
                已训练 {{ currentArchitecture.trained_count }} 次，最优准确率
                {{ percent(currentArchitecture.best_accuracy, 2) }}
              </div>
            </div>
          </template>
        </div>

        <div class="glass section" style="margin-top: 14px">
          <div class="section-title">
            <span>实时训练曲线</span>
            <span class="sub" v-if="progress">
              {{ STATUS_META[progress.status]?.text || progress.status }} ·
              第 {{ progress.current_epoch || 0 }}/{{ progress.epochs || 0 }} 轮 ·
              最优 {{ percent(progress.best_accuracy, 2) }}
            </span>
          </div>
          <el-progress
            :percentage="Number((progress?.progress || 0).toFixed(1))"
            :status="progress?.status === 'failed' ? 'exception' : progress?.status === 'success' ? 'success' : undefined"
            :stroke-width="12"
            striped
            :striped-flow="progress?.status === 'running'"
          />
          <EChart v-if="progress?.history?.length" :option="curveOption" height="280px"
                  style="margin-top: 10px" />
          <el-empty v-else description="选择任务后展示训练曲线" :image-size="70" />
        </div>

        <div class="glass section" style="margin-top: 14px">
          <div class="section-title">
            <span>训练日志</span>
            <div>
              <el-checkbox v-model="autoScroll" size="small">自动滚动</el-checkbox>
              <el-button text size="small" @click="loadLog">刷新</el-button>
            </div>
          </div>
          <div ref="logBox" class="log-panel">{{ logLines.join('\n') || '暂无日志' }}</div>
        </div>
      </div>
    </div>

    <div class="glass section" style="margin-top: 14px">
      <div class="section-title">
        <span>训练任务列表</span>
        <span class="sub">共 {{ runs.length }} 个任务</span>
      </div>
      <el-table :data="runs" style="width: 100%" size="small" :row-class-name="rowClass"
                @row-click="selectRun" highlight-current-row>
        <el-table-column prop="name" label="任务" min-width="150" show-overflow-tooltip />
        <el-table-column prop="architecture" label="结构" width="110" />
        <el-table-column prop="dataset_name" label="数据集" min-width="130" show-overflow-tooltip />
        <el-table-column label="状态" width="96">
          <template #default="{ row }">
            <el-tag size="small" :type="STATUS_META[row.status]?.type || 'info'">
              {{ STATUS_META[row.status]?.text || row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="进度" width="180">
          <template #default="{ row }">
            <el-progress :percentage="Number((row.progress || 0).toFixed(0))"
                         :status="row.status === 'failed' ? 'exception' : undefined"
                         :stroke-width="10" />
          </template>
        </el-table-column>
        <el-table-column label="准确率" width="90">
          <template #default="{ row }">{{ percent(row.best_accuracy, 1) }}</template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="170" />
        <el-table-column label="操作" width="140" fixed="right">
          <template #default="{ row }">
            <el-button v-if="row.status === 'running' || row.status === 'pending'" text size="small"
                       type="warning" @click.stop="cancel(row)">中止</el-button>
            <el-button text size="small" type="danger" @click.stop="removeRun(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<style scoped>
:deep(.active-row) {
  background: rgba(79, 140, 255, 0.14) !important;
}

:deep(.device-unavailable .el-radio-button__inner) {
  border-style: dashed;
  color: var(--ir-text-dim);
}

.device-state {
  margin-left: 6px;
  font-size: 10px;
  opacity: 0.72;
}

.device-hint {
  width: 100%;
  margin-top: 6px;
  color: var(--ir-text-dim);
  font-size: 12px;
  line-height: 1.5;
}
</style>
