<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import ProbBars from '../components/ProbBars.vue'
import { useAppStore } from '../store'
import { fixed, labelText, percent } from '../utils/format'
import { logAction } from '../utils/operationLog'
import { SAMPLE_LABELS, sampleFile, sampleObjectUrl } from '../utils/samples'

const store = useAppStore()
const selected = ref([])
const file = ref(null)
const previewUrl = ref('')
const loading = ref(false)
const result = ref(null)
const samples = SAMPLE_LABELS
const sampleUrls = ref({})
const palette = ['#4f8cff', '#36d1c4', '#ffb454', '#ff6b81', '#9b7bff', '#3ad47a', '#ff9f43']

const readyArchitectures = computed(() => store.architectures.filter((item) => item.ready))

const compareOption = computed(() => {
  const results = result.value?.results || []
  const labels = Array.from(
    new Set(results.flatMap((item) => item.topk.slice(0, 3).map((entry) => entry.label)))
  )
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { bottom: 0, textStyle: { color: '#97a3c4', fontSize: 11 } },
    grid: { left: 50, right: 20, top: 20, bottom: 56 },
    xAxis: {
      type: 'category',
      data: results.map((item) => item.architecture),
      axisLabel: { fontSize: 11 }
    },
    yAxis: { type: 'value', max: 1, axisLabel: { formatter: (v) => `${v * 100}%` } },
    series: labels.map((label, index) => ({
      name: labelText(label),
      type: 'bar',
      barMaxWidth: 22,
      itemStyle: { color: palette[index % palette.length], borderRadius: [5, 5, 0, 0] },
      data: results.map((item) => {
        const hit = item.topk.find((entry) => entry.label === label)
        return hit ? hit.probability : 0
      })
    }))
  }
})

onMounted(async () => {
  if (!store.architectures.length) await store.fetchArchitectures()
  selected.value = readyArchitectures.value.slice(0, 4).map((item) => item.key)
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
  logAction({
    action: '选择对比图片',
    detail: raw.name || '图片',
    module: '识别',
    level: 'info'
  })
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
    ElMessage.warning('请先选择一张图片')
    return
  }
  if (!selected.value.length) {
    ElMessage.warning('请至少选择一个已训练的模型')
    return
  }
  const formData = new FormData()
  formData.append('image', file.value)
  formData.append('architectures', selected.value.join(','))
  formData.append('top_k', 3)
  loading.value = true
  try {
    result.value = await api.recognizeCompare(formData)
    ElMessage.success(
      `对比完成：${result.value.results.length} 个模型，共识结果 ${labelText(result.value.consensus_label)}`
    )
    logAction({
      action: '多模型对比完成',
      detail: `${result.value.results.length} 个模型 · 共识结果 ${labelText(result.value.consensus_label)}`,
      module: '识别'
    })
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="page">
    <div class="grid grid-2">
      <div class="glass section">
        <div class="section-title">
          <span>1. 上传图片</span>
          <span class="sub">同一张图片会被送入所有选中模型</span>
        </div>
        <el-upload drag :auto-upload="false" :show-file-list="false" accept="image/*"
                   :on-change="pickFile">
          <div style="border: none">
            <el-icon :size="32" color="#4f8cff"><UploadFilled /></el-icon>
            <div style="margin-top: 6px">点击或拖拽图片到此处</div>
          </div>
        </el-upload>
        <div class="sample-grid" style="margin-top: 12px">
          <div v-for="label in samples" :key="label" class="sample-chip"
               @click="useSample(label)">
            <img v-if="sampleUrls[label]" :src="sampleUrls[label]" :alt="labelText(label)" />
            <div v-else class="sample-ph" />
            {{ labelText(label) }}
          </div>
        </div>
        <div class="preview-box" style="margin-top: 12px; min-height: 200px">
          <img v-if="previewUrl" :src="previewUrl" alt="preview" />
          <div v-else class="dim">图片预览区</div>
        </div>
      </div>

      <div class="glass section">
        <div class="section-title">
          <span>2. 选择对比模型</span>
          <span class="sub">已选 {{ selected.length }} / {{ readyArchitectures.length }}</span>
        </div>
        <el-checkbox-group v-model="selected" class="model-checks">
          <el-checkbox
            v-for="item in readyArchitectures"
            :key="item.key"
            :value="item.key"
            border
            class="model-check"
          >
            <div class="check-body">
              <span class="check-name">{{ item.name }}</span>
              <span class="dim" style="font-size: 11px">
                {{ item.family }} · 准确率 {{ percent(item.best_accuracy, 1) }}
              </span>
            </div>
          </el-checkbox>
        </el-checkbox-group>
        <el-empty v-if="!readyArchitectures.length" description="还没有已训练的模型" :image-size="70" />

        <el-button type="primary" size="large" style="width: 100%; margin-top: 14px"
                   :loading="loading" :disabled="!readyArchitectures.length" @click="submit">
          开始多模型对比
        </el-button>
        <el-alert
          type="info"
          :closable="false"
          show-icon
          style="margin-top: 12px"
          title="对比说明"
          description="不同经典模型在同一张图片上的预测概率分布会被并列展示，用于直观比较模型容量、结构与训练效果带来的差异。"
        />
      </div>
    </div>

    <template v-if="result">
      <div class="grid grid-3" style="margin-top: 14px">
        <div class="glass section">
          <div class="section-title"><span>共识结果</span><span class="sub">多数投票</span></div>
          <div class="consensus">{{ labelText(result.consensus_label) }}</div>
          <div class="dim" style="font-size: 13px">
            一致率 {{ percent(result.agreement, 1) }}（{{ result.results.length }} 个模型参与）
          </div>
          <el-divider />
          <div class="dim" style="font-size: 12px; line-height: 1.9">
            <div>总推理耗时：{{ fixed(result.latency_ms, 1) }} ms</div>
            <div>平均单模型耗时：{{ fixed(result.latency_ms / result.results.length, 1) }} ms</div>
            <div v-if="result.unavailable?.length">
              跳过未训练结构：{{ result.unavailable.join(', ') }}
            </div>
          </div>
        </div>

        <div class="glass section" style="grid-column: span 2">
          <div class="section-title">
            <span>各模型对候选类别的概率对比</span>
            <span class="sub">取每个模型 Top-3 涉及的类别</span>
          </div>
          <EChart :option="compareOption" height="300px" />
        </div>
      </div>

      <div class="grid grid-3" style="margin-top: 14px">
        <div v-for="(item, index) in result.results" :key="item.architecture" class="glass section">
          <div class="section-title">
            <span>
              <span class="dot" :style="{ background: palette[index % palette.length] }"></span>
              {{ item.architecture }}
            </span>
            <el-tag size="small" type="success" effect="dark">
              {{ percent(item.confidence, 1) }}
            </el-tag>
          </div>
          <div style="font-size: 13px; margin-bottom: 10px">
            预测：<b>{{ labelText(item.predicted_label) }}</b>
            <span class="dim" style="font-size: 12px"> · {{ item.latency_ms }} ms</span>
          </div>
          <ProbBars :items="item.topk" />
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.sample-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 10px;
}

.model-checks {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 10px;
}

.model-check {
  height: auto;
  margin: 0 !important;
  padding: 8px 10px;
  width: 100%;
}

.check-body {
  display: flex;
  flex-direction: column;
  line-height: 1.5;
}

.check-name {
  font-size: 13px;
  font-weight: 600;
}

.consensus {
  font-size: 34px;
  font-weight: 700;
  background: linear-gradient(120deg, #ffb454, #ff6b81);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 6px;
}
</style>
