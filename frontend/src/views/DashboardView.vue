<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api'
import EChart from '../components/EChart.vue'
import StatCard from '../components/StatCard.vue'
import { STATUS_META, fixed, labelText, percent } from '../utils/format'
import { useAuthStore } from '../store/auth'

const router = useRouter()
const auth = useAuthStore()
const loading = ref(true)
const data = ref(null)
let timer = null

async function load() {
  data.value = await api.dashboard({ days: 14 })
  loading.value = false
}

onMounted(() => {
  load()
  timer = setInterval(load, 15000)
})

onUnmounted(() => timer && clearInterval(timer))

const counts = computed(() => data.value?.counts || {})
const system = computed(() => data.value?.system || {})

const trendOption = computed(() => {
  const rows = data.value?.records_by_day || []
  return {
    grid: { left: 40, right: 18, top: 30, bottom: 30 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: rows.map((item) => item.date.slice(5)), boundaryGap: false },
    yAxis: { type: 'value', splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
    series: [
      {
        name: '识别次数',
        type: 'line',
        smooth: true,
        symbolSize: 7,
        data: rows.map((item) => item.count),
        itemStyle: { color: '#4f8cff' },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(79,140,255,0.55)' },
              { offset: 1, color: 'rgba(79,140,255,0.02)' }
            ]
          }
        }
      }
    ]
  }
})

const accuracyOption = computed(() => {
  const rows = data.value?.architecture_compare || []
  return {
    grid: { left: 110, right: 40, top: 20, bottom: 30 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: { type: 'value', max: 1, axisLabel: { formatter: (v) => `${v * 100}%` } },
    yAxis: {
      type: 'category',
      inverse: true,
      data: rows.map((item) => item.architecture),
      axisLabel: { fontSize: 12 }
    },
    series: [
      {
        name: '最优验证准确率',
        type: 'bar',
        barWidth: 14,
        data: rows.map((item) => item.best_accuracy),
        label: { show: true, position: 'right', formatter: (p) => `${(p.value * 100).toFixed(1)}%` },
        itemStyle: {
          borderRadius: [0, 7, 7, 0],
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 1,
            y2: 0,
            colorStops: [
              { offset: 0, color: '#36d1c4' },
              { offset: 1, color: '#4f8cff' }
            ]
          }
        }
      }
    ]
  }
})

const readinessOption = computed(() => {
  const rows = data.value?.accuracy_series || []
  return {
    grid: { left: 46, right: 20, top: 20, bottom: 40 },
    tooltip: { trigger: 'item' },
    xAxis: {
      type: 'category',
      data: rows.map((item) => item.architecture),
      axisLabel: { rotate: rows.length > 6 ? 30 : 0, fontSize: 11 }
    },
    yAxis: { type: 'value', min: 0, max: 1, axisLabel: { formatter: (v) => `${v * 100}%` } },
    series: [
      {
        name: '准确率',
        type: 'scatter',
        symbolSize: (value) => 12 + value[1] * 26,
        data: rows.map((item) => [item.architecture, item.accuracy]),
        itemStyle: { color: '#ffb454' }
      }
    ]
  }
})
</script>

<template>
  <div class="page" v-loading="loading">
    <div class="grid grid-4">
      <StatCard title="数据集" :value="counts.datasets || 0" suffix="个" icon="FolderOpened"
                hint="支持演示数据集与自定义上传" />
      <StatCard title="已训练模型" :value="counts.trained_models || 0" suffix="个" color="#36d1c4"
                icon="Files" :hint="`覆盖 ${counts.ready_architectures || 0} 种网络结构`" />
      <StatCard title="识别次数" :value="counts.records || 0" suffix="次" color="#ffb454"
                icon="Picture" :hint="`今日 ${counts.records_today || 0} 次`" />
      <StatCard title="平均验证准确率" :value="percent(data?.avg_accuracy, 2)" color="#ff6b81"
                icon="TrendCharts" :hint="`平均置信度 ${percent(data?.avg_confidence, 1)}`" />
    </div>

    <div class="grid grid-2" style="margin-top: 14px">
      <div class="glass section">
        <div class="section-title">
          <span>近 14 天识别趋势</span>
          <span class="sub">平均推理耗时 {{ fixed(data?.avg_latency_ms, 1) }} ms</span>
        </div>
        <EChart :option="trendOption" height="260px" />
      </div>

      <div class="glass section">
        <div class="section-title">
          <span>各网络结构最优准确率</span>
          <span class="sub">按验证集准确率排序</span>
        </div>
        <EChart v-if="data?.architecture_compare?.length" :option="accuracyOption" height="260px" />
        <el-empty v-else description="还没有训练完成的模型" :image-size="80" />
      </div>
    </div>

    <div class="grid grid-2" style="margin-top: 14px">
      <div class="glass section">
        <div class="section-title">
          <span>最优模型</span>
          <el-button text type="primary" @click="router.push('/models')">查看模型库</el-button>
        </div>
        <el-descriptions v-if="data?.best_model" :column="2" border size="small">
          <el-descriptions-item label="模型">{{ data.best_model.name }}</el-descriptions-item>
          <el-descriptions-item label="结构">{{ data.best_model.architecture }}</el-descriptions-item>
          <el-descriptions-item label="准确率">{{ percent(data.best_model.accuracy) }}</el-descriptions-item>
          <el-descriptions-item label="参数量">{{ fixed(data.best_model.params_million, 2) }} M</el-descriptions-item>
          <el-descriptions-item label="类别数">{{ data.best_model.classes }}</el-descriptions-item>
          <el-descriptions-item label="推理设备">{{ system.device }}</el-descriptions-item>
        </el-descriptions>
        <el-empty v-else description="还没有可用的模型，请先训练" :image-size="80">
          <el-button v-if="auth.isAdmin" type="primary" @click="router.push('/training')">去训练</el-button>
          <div v-else class="dim" style="font-size: 12px">请联系管理员训练模型</div>
        </el-empty>

        <div class="section-title" style="margin-top: 16px">
          <span>运行环境</span>
          <span class="sub">{{ system.platform }}</span>
        </div>
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="Python">{{ system.python }}</el-descriptions-item>
          <el-descriptions-item label="PyTorch">{{ system.torch }}</el-descriptions-item>
          <el-descriptions-item label="TorchVision">{{ system.torchvision }}</el-descriptions-item>
          <el-descriptions-item label="计算设备">{{ system.device }}</el-descriptions-item>
          <el-descriptions-item label="支持结构">{{ system.architectures }} 种</el-descriptions-item>
          <el-descriptions-item label="线程数">{{ system.cpu_count }}</el-descriptions-item>
          <el-descriptions-item label="当前数据库">
            {{ system.database?.label }} {{ system.database?.version }}
          </el-descriptions-item>
          <el-descriptions-item label="数据库位置">{{ system.database?.location }}</el-descriptions-item>
        </el-descriptions>
      </div>

      <div class="glass section">
        <div class="section-title">
          <span>模型准确率分布</span>
          <span class="sub">气泡大小代表准确率高低</span>
        </div>
        <EChart v-if="data?.accuracy_series?.length" :option="readinessOption" height="220px" />
        <el-empty v-else description="暂无数据" :image-size="80" />

        <div class="section-title" style="margin-top: 6px">
          <span>最近训练任务</span>
          <el-button text type="primary" @click="router.push('/training')">全部任务</el-button>
        </div>
        <el-table :data="data?.recent_runs || []" size="small" style="width: 100%">
          <el-table-column prop="name" label="任务" min-width="140" show-overflow-tooltip />
          <el-table-column prop="architecture" label="结构" width="110" />
          <el-table-column label="状态" width="90">
            <template #default="{ row }">
              <el-tag size="small" :type="STATUS_META[row.status]?.type || 'info'">
                {{ STATUS_META[row.status]?.text || row.status }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="准确率" width="90">
            <template #default="{ row }">{{ percent(row.best_accuracy, 1) }}</template>
          </el-table-column>
        </el-table>
      </div>
    </div>

    <div class="glass section" style="margin-top: 14px">
      <div class="section-title">
        <span>最近识别记录</span>
        <el-button text type="primary" @click="router.push('/records')">查看全部</el-button>
      </div>
      <div class="record-strip">
        <div v-for="item in data?.recent_records || []" :key="item.id" class="record-item">
          <img v-if="item.image_url" :src="item.image_url" alt="" />
          <div v-else class="record-placeholder">
            <el-icon><Picture /></el-icon>
          </div>
          <div class="record-label">{{ labelText(item.label) }}</div>
          <div class="record-meta">{{ percent(item.confidence, 1) }} · {{ item.architecture }}</div>
        </div>
        <el-empty v-if="!data?.recent_records?.length" description="还没有识别记录" :image-size="70" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.record-strip {
  display: flex;
  gap: 12px;
  overflow-x: auto;
  padding-bottom: 6px;
}

.record-item {
  width: 108px;
  flex: none;
  text-align: center;
}

.record-item img,
.record-placeholder {
  width: 108px;
  height: 92px;
  object-fit: cover;
  border-radius: 10px;
  border: 1px solid var(--ir-border);
  background: rgba(9, 13, 26, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
}

.record-label {
  font-size: 13px;
  margin-top: 6px;
}

.record-meta {
  font-size: 11px;
  color: var(--ir-text-dim);
}
</style>
