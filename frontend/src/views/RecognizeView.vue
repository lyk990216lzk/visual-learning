<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import ProbBars from '../components/ProbBars.vue'
import { useAppStore } from '../store'
import { labelText, percent } from '../utils/format'
import { SAMPLE_LABELS, sampleFile, sampleObjectUrl } from '../utils/samples'

const store = useAppStore()
const tab = ref('single')

// ------------------------------------------------ 单图识别
const file = ref(null)
const previewUrl = ref('')
const architecture = ref('')
const topK = ref(5)
const loading = ref(false)
const result = ref(null)
const samples = SAMPLE_LABELS
const sampleUrls = ref({})

// ------------------------------------------------ 批量识别
const batchFiles = ref([])
const batchLoading = ref(false)
const batchResult = ref(null)
const history = ref([])

const readyArchitectures = computed(() => store.architectures.filter((item) => item.ready))

const currentArchitecture = computed(
  () => store.architectureMap[architecture.value] || null
)

const probabilityOption = computed(() => {
  const entries = Object.entries(result.value?.all_probabilities || {})
  return {
    tooltip: { trigger: 'item', formatter: (p) => `${labelText(p.name)}: ${percent(p.value, 1)}` },
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

async function loadHistory() {
  const data = await api.records({ page_size: 8, ordering: '-created_at' })
  history.value = data.results || []
}

onMounted(async () => {
  if (!store.architectures.length) await store.fetchArchitectures()
  if (!architecture.value) {
    architecture.value = readyArchitectures.value[0]?.key || store.architectures[0]?.key || ''
  }
  loadHistory()
  loadSampleThumbs()
})

// 样例图接口需要令牌，必须走 axios 取回二进制后转成 blob URL，<img src> 无法直接访问
async function loadSampleThumbs() {
  await Promise.all(
    samples.map(async (label) => {
      try {
        sampleUrls.value[label] = await sampleObjectUrl(label, 64)
      } catch (error) {
        sampleUrls.value[label] = ''
      }
    })
  )
}

function pickFile(uploadFile) {
  const raw = uploadFile.raw || uploadFile
  file.value = raw
  previewUrl.value = URL.createObjectURL(raw)
  result.value = null
  return false
}

async function useSample(label) {
  try {
    pickFile(await sampleFile(label, 64))
  } catch (error) {
    ElMessage.error('样例图片加载失败，请稍后重试')
  }
}

async function submit() {
  if (!file.value) {
    ElMessage.warning('请先选择或上传一张图片')
    return
  }
  if (!architecture.value) {
    ElMessage.warning('请选择识别模型')
    return
  }
  const formData = new FormData()
  formData.append('image', file.value)
  formData.append('architecture', architecture.value)
  formData.append('top_k', topK.value)
  loading.value = true
  try {
    result.value = await api.recognize(formData)
    ElMessage.success(`识别结果：${labelText(result.value.predicted_label)}`)
    loadHistory()
  } catch (error) {
    result.value = null
  } finally {
    loading.value = false
  }
}

async function submitBatch() {
  if (!batchFiles.value.length) {
    ElMessage.warning('请选择图片（可多选）')
    return
  }
  const formData = new FormData()
  batchFiles.value.forEach((item) => formData.append('images', item.raw))
  formData.append('architecture', architecture.value)
  formData.append('top_k', 3)
  batchLoading.value = true
  try {
    batchResult.value = await api.recognizeBatch(formData)
    ElMessage.success(`批量识别完成，共 ${batchResult.value.total} 张`)
    loadHistory()
  } finally {
    batchLoading.value = false
  }
}

function onBatchChange(fileList) {
  batchFiles.value = fileList
}
</script>

<template>
  <div class="page">
    <el-tabs v-model="tab">
      <el-tab-pane label="单图识别" name="single" />
      <el-tab-pane label="批量识别" name="batch" />
      <el-tab-pane label="识别历史" name="history" />
    </el-tabs>

    <div v-if="tab === 'single'" class="grid grid-2">
      <div class="glass section">
        <div class="section-title">
          <span>1. 选择待识别图片</span>
          <span class="sub">支持 JPG / PNG / BMP / WEBP，≤10MB</span>
        </div>
        <el-upload
          drag
          :auto-upload="false"
          :show-file-list="false"
          accept="image/*"
          :on-change="pickFile"
        >
          <div class="drop-zone" style="border: none">
            <el-icon :size="34" color="#4f8cff"><UploadFilled /></el-icon>
            <div style="margin-top: 8px">点击或拖拽图片到此处上传</div>
            <div class="dim" style="font-size: 12px; margin-top: 4px">
              也可以直接点击下方样例图片快速体验
            </div>
          </div>
        </el-upload>

        <div class="section-title" style="margin-top: 12px">
          <span>内置样例</span>
          <span class="sub">点击自动填充</span>
        </div>
        <div class="sample-grid">
          <div v-for="label in samples" :key="label" class="sample-chip" @click="useSample(label)">
            <img v-if="sampleUrls[label]" :src="sampleUrls[label]" :alt="labelText(label)" />
            <div v-else class="sample-ph" />
            {{ labelText(label) }}
          </div>
        </div>

        <div class="preview-box" style="margin-top: 12px">
          <img v-if="previewUrl" :src="previewUrl" alt="preview" />
          <div v-else class="dim">图片预览区</div>
        </div>
      </div>

      <div class="glass section">
        <div class="section-title">
          <span>2. 选择识别模型</span>
          <span class="sub">仅列出已训练完成的模型</span>
        </div>

        <el-select v-model="architecture" placeholder="选择网络结构" style="width: 100%">
          <el-option-group label="已就绪（推荐）">
            <el-option
              v-for="item in readyArchitectures"
              :key="item.key"
              :label="`${item.name} · 准确率 ${percent(item.best_accuracy, 1)}`"
              :value="item.key"
            />
          </el-option-group>
          <el-option-group label="尚未训练">
            <el-option
              v-for="item in store.architectures.filter((a) => !a.ready)"
              :key="item.key"
              :label="`${item.name}（需先训练）`"
              :value="item.key"
              disabled
            />
          </el-option-group>
        </el-select>

        <div v-if="currentArchitecture" class="arch-info">
          <div class="arch-info-title">
            {{ currentArchitecture.name }}
            <el-tag size="small" type="warning" effect="dark">{{ currentArchitecture.year }}</el-tag>
            <el-tag size="small" type="info" effect="plain">{{ currentArchitecture.family }}</el-tag>
          </div>
          <div class="dim" style="font-size: 12px; line-height: 1.7; margin-top: 6px">
            {{ currentArchitecture.description }}
          </div>
        </div>

        <div style="display: flex; align-items: center; gap: 14px; margin-top: 16px">
          <span class="dim" style="font-size: 13px">Top-K 结果</span>
          <el-slider v-model="topK" :min="1" :max="6" :step="1" style="flex: 1" />
          <span class="mono">{{ topK }}</span>
        </div>

        <el-button
          type="primary"
          size="large"
          style="width: 100%; margin-top: 14px"
          :loading="loading"
          :disabled="!readyArchitectures.length"
          @click="submit"
        >
          <el-icon><Search /></el-icon>
          <span style="margin-left: 6px">开始识别</span>
        </el-button>
        <el-alert
          v-if="!readyArchitectures.length"
          type="warning"
          :closable="false"
          show-icon
          title="还没有训练好的模型"
          description="请先到「模型训练」页面使用演示数据集训练一个模型，例如 LeNet-5 只需十几秒。"
          style="margin-top: 12px"
        />
      </div>
    </div>

    <div v-if="tab === 'single' && result" class="grid grid-2" style="margin-top: 14px">
      <div class="glass section">
        <div class="section-title">
          <span>识别结果</span>
          <span class="sub">
            {{ result.architecture }} · 输入 {{ result.image_size }}px · 耗时 {{ result.latency_ms }} ms
          </span>
        </div>
        <div class="result-head">
          <div class="result-label">{{ labelText(result.predicted_label) }}</div>
          <el-progress
            type="dashboard"
            :percentage="Number((result.confidence * 100).toFixed(1))"
            :width="128"
            :stroke-width="10"
            color="#36d1c4"
          >
            <template #default="{ percentage }">
              <div style="font-size: 20px; font-weight: 600">{{ percentage }}%</div>
              <div class="dim" style="font-size: 12px">置信度</div>
            </template>
          </el-progress>
        </div>
        <div class="section-title" style="margin-top: 6px">Top-{{ result.topk.length }} 概率分布</div>
        <ProbBars :items="result.topk" />
      </div>

      <div class="glass section">
        <div class="section-title">
          <span>全部类别概率</span>
          <span class="sub">Softmax 输出</span>
        </div>
        <EChart :option="probabilityOption" height="300px" />
      </div>
    </div>

    <div v-if="tab === 'batch'" class="glass section">
      <div class="section-title">
        <span>批量识别（单次最多 20 张）</span>
        <span class="sub">使用同一个模型逐张推理</span>
      </div>
      <div class="grid grid-2">
        <el-upload
          drag
          multiple
          :auto-upload="false"
          :show-file-list="true"
          accept="image/*"
          :on-change="onBatchChange"
          :file-list="batchFiles"
        >
          <el-icon :size="30" color="#4f8cff"><FolderAdd /></el-icon>
          <div style="margin-top: 6px">选择多张图片</div>
        </el-upload>
        <div>
          <el-select v-model="architecture" placeholder="选择模型" style="width: 100%">
            <el-option
              v-for="item in readyArchitectures"
              :key="item.key"
              :label="`${item.name} · ${percent(item.best_accuracy, 1)}`"
              :value="item.key"
            />
          </el-select>
          <el-button
            type="primary"
            style="width: 100%; margin-top: 12px"
            :loading="batchLoading"
            :disabled="!readyArchitectures.length"
            @click="submitBatch"
          >
            开始批量识别
          </el-button>
          <div v-if="batchResult" class="dim" style="font-size: 12px; margin-top: 10px">
            模型：{{ batchResult.model_name }} · 成功 {{ batchResult.total }} 张 ·
            失败 {{ batchResult.failed.length }} 张
          </div>
        </div>
      </div>

      <el-table
        v-if="batchResult"
        :data="batchResult.results"
        style="width: 100%; margin-top: 14px"
        size="small"
      >
        <el-table-column label="图片" width="90">
          <template #default="{ row }">
            <el-image :src="row.image_url" fit="cover" style="width: 56px; height: 56px; border-radius: 8px" />
          </template>
        </el-table-column>
        <el-table-column prop="name" label="文件名" min-width="160" show-overflow-tooltip />
        <el-table-column label="识别结果" width="120">
          <template #default="{ row }">{{ labelText(row.predicted_label) }}</template>
        </el-table-column>
        <el-table-column label="置信度" width="110">
          <template #default="{ row }">{{ percent(row.confidence, 1) }}</template>
        </el-table-column>
        <el-table-column label="耗时" width="100">
          <template #default="{ row }">{{ row.latency_ms }} ms</template>
        </el-table-column>
      </el-table>
    </div>

    <div v-if="tab === 'history'" class="glass section">
      <div class="section-title">
        <span>最近识别记录</span>
        <el-button text type="primary" @click="$router.push('/records')">进入历史记录页</el-button>
      </div>
      <el-table :data="history" style="width: 100%" size="small">
        <el-table-column label="图片" width="80">
          <template #default="{ row }">
            <el-image
              v-if="row.image_url"
              :src="row.image_url"
              fit="cover"
              style="width: 46px; height: 46px; border-radius: 8px"
            />
          </template>
        </el-table-column>
        <el-table-column prop="original_name" label="文件名" min-width="150" show-overflow-tooltip />
        <el-table-column prop="architecture" label="模型" width="120" />
        <el-table-column label="结果" width="110">
          <template #default="{ row }">{{ labelText(row.predicted_label) }}</template>
        </el-table-column>
        <el-table-column label="置信度" width="100">
          <template #default="{ row }">{{ percent(row.confidence, 1) }}</template>
        </el-table-column>
        <el-table-column prop="created_at" label="时间" width="170" />
      </el-table>
    </div>
  </div>
</template>

<style scoped>
.sample-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 10px;
}

.arch-info {
  margin-top: 12px;
  padding: 12px 14px;
  border-radius: 12px;
  background: rgba(79, 140, 255, 0.08);
  border: 1px solid var(--ir-border);
}

.arch-info-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.result-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
}

.result-label {
  font-size: 40px;
  font-weight: 700;
  background: linear-gradient(120deg, #36d1c4, #4f8cff);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
</style>
