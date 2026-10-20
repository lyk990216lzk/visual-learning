<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import { useAppStore } from '../store'
import { useAuthStore } from '../store/auth'
import { bytes, dateTime, fixed, percent } from '../utils/format'
import { logAction } from '../utils/operationLog'

const store = useAppStore()
const auth = useAuthStore()
const tab = ref('architectures')
const archDrawer = ref(false)
const detail = ref(null)
const detailLoading = ref(false)
const modelDrawer = ref(false)
const activeModel = ref(null)

const models = computed(() => store.trainedModels)

const scatterOption = computed(() => {
  const rows = models.value.filter((item) => item.params_million)
  return {
    tooltip: {
      formatter: (p) => {
        const item = rows[p.dataIndex]
        return `${item.name}<br/>参数量 ${fixed(item.params_million, 2)}M<br/>准确率 ${percent(item.accuracy, 2)}`
      }
    },
    grid: { left: 60, right: 30, top: 30, bottom: 46 },
    xAxis: {
      type: 'value',
      name: '参数量 (M)',
      nameLocation: 'middle',
      nameGap: 28,
      scale: true
    },
    yAxis: { type: 'value', name: '准确率', min: 0, max: 1, axisLabel: { formatter: (v) => `${v * 100}%` } },
    series: [
      {
        type: 'scatter',
        symbolSize: 16,
        data: rows.map((item) => [item.params_million, item.accuracy, item.name]),
        itemStyle: { color: '#4f8cff', opacity: 0.85 }
      }
    ]
  }
})

const accuracyBarOption = computed(() => {
  const rows = [...models.value].sort((a, b) => b.accuracy - a.accuracy).slice(0, 12)
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 46, right: 20, top: 24, bottom: 60 },
    xAxis: {
      type: 'category',
      data: rows.map((item) => item.architecture),
      axisLabel: { rotate: 30, fontSize: 11 }
    },
    yAxis: { type: 'value', min: 0, max: 1, axisLabel: { formatter: (v) => `${v * 100}%` } },
    series: [
      {
        type: 'bar',
        barMaxWidth: 26,
        data: rows.map((item) => item.accuracy),
        label: { show: true, position: 'top', formatter: (p) => `${(p.value * 100).toFixed(1)}%`, fontSize: 10 },
        itemStyle: {
          borderRadius: [6, 6, 0, 0],
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
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

const confusionOption = computed(() => {
  const metrics = activeModel.value?.metrics || {}
  const matrix = metrics.confusion_matrix || []
  const names = metrics.class_names || activeModel.value?.classes || []
  if (!matrix.length) return {}
  const data = []
  let max = 0
  matrix.forEach((row, i) => {
    row.forEach((value, j) => {
      data.push([j, i, value])
      max = Math.max(max, value)
    })
  })
  return {
    tooltip: {
      formatter: (p) => `真实：${names[p.data[1]]}<br/>预测：${names[p.data[0]]}<br/>样本数：${p.data[2]}`
    },
    grid: { left: 80, right: 30, top: 20, bottom: 60 },
    xAxis: { type: 'category', data: names, axisLabel: { rotate: 30, fontSize: 11 } },
    yAxis: { type: 'category', data: names, axisLabel: { fontSize: 11 } },
    visualMap: {
      min: 0,
      max: max || 1,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      textStyle: { color: '#97a3c4', fontSize: 11 },
      inRange: { color: ['#0f1a35', '#2c5cff', '#36d1c4', '#ffb454'] }
    },
    series: [
      {
        type: 'heatmap',
        data,
        label: { show: true, fontSize: 10, color: '#0b1020' },
        emphasis: { itemStyle: { shadowBlur: 8, shadowColor: 'rgba(0,0,0,0.6)' } }
      }
    ]
  }
})

const trainingCurveOption = computed(() => {
  const history = activeModel.value?.metrics?.history || []
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: ['训练准确率', '验证准确率', '训练损失', '验证损失'], bottom: 0, textStyle: { color: '#97a3c4', fontSize: 11 } },
    grid: { left: 50, right: 50, top: 20, bottom: 56 },
    xAxis: { type: 'category', data: history.map((item) => `E${item.epoch}`) },
    yAxis: [
      { type: 'value', name: 'Acc', min: 0, max: 1, axisLabel: { formatter: (v) => `${v * 100}%` } },
      { type: 'value', name: 'Loss' }
    ],
    series: [
      { name: '训练准确率', type: 'line', smooth: true, data: history.map((i) => i.train_accuracy), itemStyle: { color: '#36d1c4' } },
      { name: '验证准确率', type: 'line', smooth: true, data: history.map((i) => i.val_accuracy), itemStyle: { color: '#ffb454' } },
      { name: '训练损失', type: 'line', smooth: true, yAxisIndex: 1, data: history.map((i) => i.train_loss), itemStyle: { color: '#4f8cff' } },
      { name: '验证损失', type: 'line', smooth: true, yAxisIndex: 1, data: history.map((i) => i.val_loss), itemStyle: { color: '#ff6b81' } }
    ]
  }
})

async function openArchitecture(item) {
  archDrawer.value = true
  detailLoading.value = true
  detail.value = { ...item, summary: null, trained_models: [] }
  try {
    const numClasses = item.num_classes || 6
    const data = await api.architectureDetail(item.key, {
      num_classes: numClasses,
      input_size: item.default_input_size
    })
    detail.value = { ...item, ...data }
  } finally {
    detailLoading.value = false
  }
}

function openModel(model) {
  activeModel.value = model
  modelDrawer.value = true
}

async function activate(model) {
  await api.activateModel(model.id)
  ElMessage.success(`已将「${model.name}」设为默认模型`)
  logAction({
    action: '切换默认模型',
    detail: `${model.name} · ${model.architecture}`,
    module: '模型'
  })
  await store.fetchTrainedModels()
  await store.fetchArchitectures()
}

async function evaluate(model) {
  ElMessage.info('正在重新评估模型，请稍候…')
  const data = await api.evaluateModel(model.id)
  ElMessage.success(`评估完成，准确率 ${percent(data.metrics.accuracy, 2)}`)
  logAction({
    action: '模型评估完成',
    detail: `${model.name} · 准确率 ${percent(data.metrics.accuracy, 2)}`,
    module: '模型'
  })
  await store.fetchTrainedModels()
  const refreshed = store.trainedModels.find((item) => item.id === model.id)
  if (refreshed) activeModel.value = refreshed
}

async function remove(model) {
  await ElMessageBox.confirm(`确定删除模型「${model.name}」及其权重文件吗？`, '提示', { type: 'warning' })
  await api.deleteModel(model.id)
  ElMessage.success('已删除')
  await store.fetchTrainedModels()
  await store.fetchArchitectures()
}

function download(model) {
  window.open(api.downloadModelUrl(model.id), '_blank')
}

onMounted(async () => {
  if (!store.architectures.length) await store.bootstrap()
  await store.fetchTrainedModels()
})
</script>

<template>
  <div class="page">
    <el-tabs v-model="tab">
      <el-tab-pane label="经典网络结构" name="architectures" />
      <el-tab-pane label="已训练模型" name="trained" />
      <el-tab-pane label="模型对比分析" name="compare" />
    </el-tabs>

    <div v-if="tab === 'architectures'">
      <div class="dim" style="font-size: 13px; margin-bottom: 12px">
        系统内置 {{ store.architectures.length }} 种经典卷积网络结构，全部为从零实现，无需下载预训练权重，可离线训练与推理。
      </div>
      <div class="grid grid-3">
        <div
          v-for="item in store.architectures"
          :key="item.key"
          class="glass arch-card"
          :class="{ ready: item.ready }"
          @click="openArchitecture(item)"
        >
          <div class="arch-head">
            <div class="arch-name">{{ item.name }}</div>
            <div class="arch-year">{{ item.year }}</div>
          </div>
          <div class="dim" style="font-size: 12px; margin-top: 4px">{{ item.family }}</div>
          <div class="tag-row">
            <el-tag v-for="tag in item.highlights" :key="tag" size="small" effect="plain">{{ tag }}</el-tag>
          </div>
          <div class="metrics-line">
            <span v-if="item.ready">已训练 {{ item.trained_count }} 次</span>
            <span v-else>尚未训练</span>
            <span v-if="item.ready">最优 {{ percent(item.best_accuracy, 1) }}</span>
          </div>
        </div>
      </div>
    </div>

    <div v-else-if="tab === 'trained'" class="glass section">
      <div class="section-title">
        <span>已训练模型（{{ models.length }}）</span>
        <span class="sub">点击任意一行查看混淆矩阵与训练曲线</span>
      </div>
      <el-table :data="models" style="width: 100%" size="small" @row-click="openModel">
        <el-table-column prop="name" label="模型名称" min-width="170" show-overflow-tooltip />
        <el-table-column prop="architecture" label="结构" width="110" />
        <el-table-column prop="dataset_name" label="数据集" min-width="130" show-overflow-tooltip />
        <el-table-column label="准确率" width="100" sortable :sort-method="(a, b) => a.accuracy - b.accuracy">
          <template #default="{ row }">
            <span style="color: #36d1c4">{{ percent(row.accuracy, 2) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="宏平均 F1" width="110">
          <template #default="{ row }">{{ percent(row.macro_f1, 2) }}</template>
        </el-table-column>
        <el-table-column label="参数量" width="100">
          <template #default="{ row }">{{ fixed(row.params_million, 2) }} M</template>
        </el-table-column>
        <el-table-column label="体积" width="100">
          <template #default="{ row }">{{ bytes(row.file_size) }}</template>
        </el-table-column>
        <el-table-column label="默认" width="80">
          <template #default="{ row }">
            <el-tag v-if="row.is_active" size="small" type="success" effect="dark">已激活</el-tag>
            <span v-else class="dim">—</span>
          </template>
        </el-table-column>
        <el-table-column v-if="auth.isAdmin" label="操作" width="240" fixed="right">
          <template #default="{ row }">
            <el-button text size="small" type="primary" @click.stop="activate(row)">设为默认</el-button>
            <el-button text size="small" @click.stop="evaluate(row)">重新评估</el-button>
            <el-button text size="small" @click.stop="download(row)">下载</el-button>
            <el-button text size="small" type="danger" @click.stop="remove(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-empty v-if="!models.length" description="还没有训练好的模型" :image-size="80" />
      <el-alert v-if="!auth.isAdmin" type="info" :closable="false" show-icon
                title="当前为普通用户视图"
                description="你可以查看模型指标与下载权重；激活、重新评估、删除以及发起训练等操作需要管理员权限。"
                style="margin-top: 12px" />
    </div>

    <div v-else class="grid grid-2">
      <div class="glass section">
        <div class="section-title">
          <span>参数量与准确率关系</span>
          <span class="sub">模型并非越大越好</span>
        </div>
        <EChart v-if="models.length" :option="scatterOption" height="320px" />
        <el-empty v-else description="暂无数据" :image-size="80" />
      </div>
      <div class="glass section">
        <div class="section-title">
          <span>模型准确率排行</span>
          <span class="sub">取前 12 个模型</span>
        </div>
        <EChart v-if="models.length" :option="accuracyBarOption" height="320px" />
        <el-empty v-else description="暂无数据" :image-size="80" />
      </div>
      <div class="glass section" style="grid-column: 1 / -1">
        <div class="section-title"><span>指标对照表</span></div>
        <el-table :data="[...models].sort((a, b) => b.accuracy - a.accuracy)" size="small">
          <el-table-column prop="architecture" label="结构" width="120" />
          <el-table-column prop="num_classes" label="类别数" width="90" />
          <el-table-column prop="input_size" label="输入尺寸" width="100" />
          <el-table-column label="准确率" width="110">
            <template #default="{ row }">{{ percent(row.accuracy, 2) }}</template>
          </el-table-column>
          <el-table-column label="宏平均精确率" width="130">
            <template #default="{ row }">{{ percent(row.metrics?.macro_precision, 2) }}</template>
          </el-table-column>
          <el-table-column label="宏平均召回率" width="130">
            <template #default="{ row }">{{ percent(row.metrics?.macro_recall, 2) }}</template>
          </el-table-column>
          <el-table-column label="参数量" width="110">
            <template #default="{ row }">{{ fixed(row.params_million, 2) }} M</template>
          </el-table-column>
          <el-table-column label="训练耗时" width="110">
            <template #default="{ row }">{{ fixed(row.metrics?.duration_seconds, 1) }} s</template>
          </el-table-column>
          <el-table-column label="创建时间" min-width="150">
            <template #default="{ row }">{{ dateTime(row.created_at) }}</template>
          </el-table-column>
        </el-table>
      </div>
    </div>

    <el-drawer v-model="archDrawer" size="46%" :title="detail?.name || '网络结构'">
      <div v-loading="detailLoading" style="padding: 0 4px">
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="提出年份">{{ detail?.year }}</el-descriptions-item>
          <el-descriptions-item label="所属类别">{{ detail?.family }}</el-descriptions-item>
          <el-descriptions-item label="作者">{{ detail?.authors }}</el-descriptions-item>
          <el-descriptions-item label="建议输入">
            {{ detail?.default_input_size }}px（最小 {{ detail?.min_input_size }}px）
          </el-descriptions-item>
        </el-descriptions>
        <p class="dim" style="font-size: 13px; line-height: 1.9">{{ detail?.description }}</p>
        <div class="tag-row">
          <el-tag v-for="tag in detail?.highlights || []" :key="tag" size="small" effect="plain" type="success">
            {{ tag }}
          </el-tag>
        </div>
        <el-divider content-position="left">结构统计（按 {{ detail?.num_classes || 6 }} 类计算）</el-divider>
        <el-descriptions v-if="detail?.summary" :column="2" border size="small">
          <el-descriptions-item label="总参数量">{{ detail.summary.total_params?.toLocaleString() }}</el-descriptions-item>
          <el-descriptions-item label="参数量">{{ fixed(detail.summary.params_million, 3) }} M</el-descriptions-item>
          <el-descriptions-item label="模型体积">{{ fixed(detail.summary.size_mb, 2) }} MB</el-descriptions-item>
          <el-descriptions-item label="CPU 推理耗时">{{ fixed(detail.summary.latency_ms, 2) }} ms</el-descriptions-item>
          <el-descriptions-item label="层数">{{ detail.summary.layers }}</el-descriptions-item>
          <el-descriptions-item label="论文">{{ detail?.paper }}</el-descriptions-item>
        </el-descriptions>
        <el-divider content-position="left">该结构已训练模型</el-divider>
        <el-table :data="detail?.trained_models || []" size="small">
          <el-table-column prop="name" label="名称" min-width="150" show-overflow-tooltip />
          <el-table-column label="准确率" width="100">
            <template #default="{ row }">{{ percent(row.accuracy, 2) }}</template>
          </el-table-column>
          <el-table-column label="默认" width="80">
            <template #default="{ row }">
              <el-tag v-if="row.is_active" size="small" type="success">是</el-tag>
              <span v-else class="dim">否</span>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </el-drawer>

    <el-drawer v-model="modelDrawer" size="60%" :title="activeModel?.name || '模型详情'">
      <div v-if="activeModel" style="padding: 0 4px">
        <el-descriptions :column="3" border size="small">
          <el-descriptions-item label="结构">{{ activeModel.architecture }}</el-descriptions-item>
          <el-descriptions-item label="数据集">{{ activeModel.dataset_name || '—' }}</el-descriptions-item>
          <el-descriptions-item label="输入尺寸">{{ activeModel.input_size }}px</el-descriptions-item>
          <el-descriptions-item label="验证准确率">{{ percent(activeModel.accuracy, 2) }}</el-descriptions-item>
          <el-descriptions-item label="宏平均 F1">{{ percent(activeModel.macro_f1, 2) }}</el-descriptions-item>
          <el-descriptions-item label="参数量">{{ fixed(activeModel.params_million, 2) }} M</el-descriptions-item>
          <el-descriptions-item label="权重体积">{{ bytes(activeModel.file_size) }}</el-descriptions-item>
          <el-descriptions-item label="权重文件">{{ activeModel.file_name }}</el-descriptions-item>
          <el-descriptions-item label="创建时间">{{ dateTime(activeModel.created_at) }}</el-descriptions-item>
        </el-descriptions>

        <el-divider content-position="left">混淆矩阵</el-divider>
        <EChart v-if="activeModel.metrics?.confusion_matrix" :option="confusionOption" height="320px" />
        <el-empty v-else description="该模型没有混淆矩阵数据，可点击「重新评估」生成" :image-size="70" />

        <el-divider content-position="left">分类别指标</el-divider>
        <el-table :data="activeModel.metrics?.per_class || []" size="small">
          <el-table-column prop="index" label="#" width="50" />
          <el-table-column prop="label" label="类别" min-width="110" />
          <el-table-column label="精确率" width="100">
            <template #default="{ row }">{{ percent(row.precision, 2) }}</template>
          </el-table-column>
          <el-table-column label="召回率" width="100">
            <template #default="{ row }">{{ percent(row.recall, 2) }}</template>
          </el-table-column>
          <el-table-column label="F1" width="100">
            <template #default="{ row }">{{ percent(row.f1, 2) }}</template>
          </el-table-column>
          <el-table-column prop="support" label="样本数" width="90" />
        </el-table>

        <el-divider content-position="left">训练曲线</el-divider>
        <EChart v-if="activeModel.metrics?.history?.length" :option="trainingCurveOption" height="300px" />
        <el-empty v-else description="暂无训练曲线数据" :image-size="70" />
      </div>
    </el-drawer>
  </div>
</template>
