<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import { useAppStore } from '../store'
import { dateTime } from '../utils/format'

const store = useAppStore()
const uploading = ref(false)
const uploadRef = ref(null)
const form = reactive({ name: '', task_type: 'classification', description: '' })
const uploadFile = ref(null)
const sampleDialog = ref(false)
const samples = ref([])
const sampleDataset = ref(null)
const validationErrors = ref([])
const validationHint = ref('')

const TASK_OPTIONS = [
  { value: 'classification', label: '图片识别', type: 'success' },
  { value: 'detection', label: '目标检测', type: 'warning' },
  { value: 'segmentation', label: '语义分割', type: 'info' }
]

const TASK_HINTS = {
  classification: '只提供 <类别>/*.jpg 时按 val_split 自动划分；提供 train/val/<类别>/*.jpg 时使用固定验证集。',
  detection: '固定验证集：data.yaml + images/train|val + labels/train|val；自动划分：只提供 images/train + labels/train。',
  segmentation: '固定验证集：images/train|val + masks/train|val + classes.json；自动划分：只提供 images/train + masks/train + classes.json。'
}

const taskLabelMap = Object.fromEntries(TASK_OPTIONS.map((item) => [item.value, item.label]))
const taskTypeMap = Object.fromEntries(TASK_OPTIONS.map((item) => [item.value, item.type]))
const taskHint = computed(() => TASK_HINTS[form.task_type] || '')

function clearValidationErrors() {
  validationErrors.value = []
  validationHint.value = ''
}

function showValidationError(error) {
  validationErrors.value = error?.errors?.length ? error.errors : [error?.message || '数据集格式不符合要求']
  validationHint.value = error?.hint || ''
}

watch(() => form.task_type, clearValidationErrors)

const classChartOption = computed(() => {
  const dataset = sampleDataset.value
  const counts = dataset?.class_counts || {}
  return {
    tooltip: { trigger: 'item' },
    legend: { bottom: 0, textStyle: { color: '#97a3c4', fontSize: 11 } },
    series: [
      {
        type: 'pie',
        radius: ['40%', '66%'],
        center: ['50%', '42%'],
        data: Object.entries(counts).map(([name, value]) => ({ name, value })),
        itemStyle: { borderColor: '#0b1020', borderWidth: 2 },
        label: { color: '#c9d4ee', fontSize: 11 }
      }
    ]
  }
})

async function load() {
  await store.fetchDatasets()
}

onMounted(load)

async function buildDemo() {
  ElMessage.info('正在生成演示数据集…')
  await api.buildDemoDataset({ per_class: 240, image_size: 64 })
  await load()
  ElMessage.success('演示数据集已生成（6 类几何图形，共 1440 张）')
}

function onFileChange(file) {
  uploadFile.value = file.raw
  clearValidationErrors()
}

async function upload() {
  if (!form.name) {
    ElMessage.warning('请填写数据集名称')
    return
  }
  if (!uploadFile.value) {
    ElMessage.warning('请选择 zip 压缩包')
    return
  }
  const formData = new FormData()
  formData.append('name', form.name)
  formData.append('task_type', form.task_type)
  formData.append('description', form.description)
  formData.append('file', uploadFile.value)
  clearValidationErrors()
  uploading.value = true
  try {
    await api.uploadDataset(formData, { silent: true })
    ElMessage.success('数据集上传成功')
    form.name = ''
    form.task_type = 'classification'
    form.description = ''
    uploadFile.value = null
    uploadRef.value?.clearFiles()
    await load()
  } catch (error) {
    showValidationError(error)
    ElMessage.error('数据集未通过格式校验，请根据错误列表调整目录')
  } finally {
    uploading.value = false
  }
}

async function refresh(dataset) {
  await api.refreshDataset(dataset.id)
  await load()
  ElMessage.success('统计数据已刷新')
}

async function remove(dataset) {
  await ElMessageBox.confirm(
    `确定删除数据集「${dataset.name}」吗？目录文件也会被移除。`,
    '提示',
    { type: 'warning' }
  )
  await api.deleteDataset(dataset.id)
  ElMessage.success('已删除')
  await load()
}

async function openSamples(dataset) {
  sampleDataset.value = dataset
  sampleDialog.value = true
  const data = await api.datasetSamples(dataset.id, { limit: 18 })
  samples.value = data.results || []
}
</script>

<template>
  <div class="page">
    <div class="grid grid-2">
      <div class="glass section">
        <div class="section-title">
          <span>快速开始</span>
          <span class="sub">没有数据集？一键生成</span>
        </div>
        <div class="dim" style="font-size: 13px; line-height: 1.9">
          演示数据集为系统合成的 6 类几何图形（圆形 / 正方形 / 三角形 / 十字 / 五角星 / 圆环），
          每张图片都带有随机颜色、旋转、尺寸与噪声扰动，可直接用于训练与对比所有经典模型，完全离线可用。
        </div>
        <el-button type="primary" style="margin-top: 12px" @click="buildDemo">
          <el-icon><MagicStick /></el-icon>
          <span style="margin-left: 6px">生成演示数据集</span>
        </el-button>
      </div>

      <div class="glass section">
        <div class="section-title">
          <span>上传自定义数据集</span>
          <span class="sub">先选择功能，再上传符合对应训练格式的 ZIP</span>
        </div>
        <el-form :model="form" label-width="80px" size="default">
          <el-form-item label="名称">
            <el-input v-model="form.name" placeholder="例如：水果分类" />
          </el-form-item>
          <el-form-item label="功能">
            <el-radio-group v-model="form.task_type">
              <el-radio-button
                v-for="item in TASK_OPTIONS"
                :key="item.value"
                :label="item.value"
              >
                {{ item.label }}
              </el-radio-button>
            </el-radio-group>
          </el-form-item>
          <el-form-item label="说明">
            <el-input v-model="form.description" placeholder="可选" />
          </el-form-item>
          <el-form-item label="数据包">
            <el-upload
              ref="uploadRef"
              :auto-upload="false"
              :limit="1"
              accept=".zip"
              :on-change="onFileChange"
              :show-file-list="true"
            >
              <el-button>选择 zip 文件</el-button>
            </el-upload>
          </el-form-item>
          <el-button type="primary" :loading="uploading" @click="upload">上传并解析</el-button>
          <div class="dataset-format-hint">
            {{ taskHint }}
          </div>
          <div v-if="validationErrors.length" class="dataset-validation-error" role="alert">
            <div class="dataset-validation-title">数据集不符合要求</div>
            <ul>
              <li v-for="item in validationErrors" :key="item">{{ item }}</li>
            </ul>
            <div v-if="validationHint" class="dataset-validation-hint">{{ validationHint }}</div>
          </div>
        </el-form>
      </div>
    </div>

    <div class="glass section" style="margin-top: 14px">
      <div class="section-title">
        <span>数据集列表（{{ store.datasets.length }}）</span>
        <span class="sub">图片识别、目标检测与语义分割分别按对应目录结构解析</span>
      </div>
      <el-table :data="store.datasets" style="width: 100%" size="small">
        <el-table-column prop="name" label="名称" min-width="160" show-overflow-tooltip />
        <el-table-column label="功能" width="100">
          <template #default="{ row }">
            <el-tag size="small" effect="plain" :type="taskTypeMap[row.task_type] || 'info'">
              {{ taskLabelMap[row.task_type] || '图片识别' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="类型" width="100">
          <template #default="{ row }">
            <el-tag size="small" :type="row.is_demo ? 'warning' : 'info'" effect="plain">
              {{ row.is_demo ? '演示数据' : '自定义' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="num_classes" label="类别数" width="90" />
        <el-table-column prop="image_count" label="图片总数" width="100" />
        <el-table-column label="训练/验证" width="120">
          <template #default="{ row }">{{ row.train_count }} / {{ row.val_count }}</template>
        </el-table-column>
        <el-table-column label="类别" min-width="220">
          <template #default="{ row }">
            <el-tag v-for="name in row.class_names" :key="name" size="small" effect="plain"
                    style="margin: 2px 4px 2px 0">{{ name }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="创建时间" width="170">
          <template #default="{ row }">{{ dateTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="190" fixed="right">
          <template #default="{ row }">
            <el-button text size="small" type="primary" @click="openSamples(row)">样本预览</el-button>
            <el-button text size="small" @click="refresh(row)">刷新</el-button>
            <el-button text size="small" type="danger" @click="remove(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-empty v-if="!store.datasets.length" description="还没有数据集，点击上方按钮生成或上传" :image-size="80" />
    </div>

    <el-dialog v-model="sampleDialog" :title="`样本预览 · ${sampleDataset?.name || ''}`" width="72%">
      <div class="grid grid-2">
        <div>
          <div class="section-title">
            <span>
              {{ sampleDataset?.task_type === 'segmentation'
                ? '类别像素分布'
                : sampleDataset?.task_type === 'detection'
                  ? '目标数量分布'
                  : '类别分布' }}
            </span>
          </div>
          <EChart v-if="sampleDataset?.class_counts" :option="classChartOption" height="260px" />
        </div>
        <div>
          <div class="section-title">
            <span>样本图片</span>
            <span class="sub">
              {{ sampleDataset?.task_type === 'classification' ? '每个类别抽样' : '训练/验证集抽样' }}
            </span>
          </div>
          <div class="sample-wall">
            <div v-for="item in samples" :key="item.url" class="wall-item">
              <el-image :src="item.url" fit="cover" style="width: 72px; height: 72px; border-radius: 8px">
                <template #error>
                  <div class="wall-fallback">无图</div>
                </template>
              </el-image>
              <div class="wall-label">{{ item.label }}</div>
            </div>
          </div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<style scoped>
.dataset-format-hint {
  margin-top: 9px;
  color: var(--ir-text-dim);
  font-size: 11px;
  line-height: 1.7;
}

.dataset-validation-error {
  margin-top: 12px;
  padding: 10px 12px;
  border: 1px solid rgba(255, 107, 129, 0.42);
  border-radius: 8px;
  background: rgba(255, 107, 129, 0.08);
  color: var(--ir-text);
  font-size: 12px;
}

.dataset-validation-title {
  color: #ff91a2;
  font-weight: 600;
}

.dataset-validation-error ul {
  margin: 7px 0 0;
  padding-left: 20px;
  line-height: 1.75;
}

.dataset-validation-hint {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid rgba(255, 107, 129, 0.2);
  color: var(--ir-text-dim);
}

.sample-wall {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(84px, 1fr));
  gap: 10px;
  max-height: 260px;
  overflow: auto;
}

.wall-item {
  text-align: center;
}

.wall-label {
  font-size: 11px;
  color: var(--ir-text-dim);
  margin-top: 4px;
}

.wall-fallback {
  width: 72px;
  height: 72px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.06);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  color: var(--ir-text-dim);
}
</style>
