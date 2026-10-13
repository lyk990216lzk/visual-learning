<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import StatCard from '../components/StatCard.vue'
import { useAppStore } from '../store'
import { useAuthStore } from '../store/auth'
import { labelText, percent } from '../utils/format'

const store = useAppStore()
const auth = useAuthStore()
const loading = ref(false)
const records = ref([])
const stats = ref(null)
const total = ref(0)
const filter = reactive({
  architecture: '',
  label: '',
  source: '',
  search: '',
  page: 1,
  page_size: 12
})

const sourceText = { upload: '单图上传', compare: '多模型对比', batch: '批量识别', sample: '内置样例' }

const dailyOption = computed(() => {
  const rows = stats.value?.by_day || []
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 44, right: 20, top: 24, bottom: 34 },
    xAxis: { type: 'category', data: rows.map((item) => item.date.slice(5)) },
    yAxis: { type: 'value' },
    series: [
      {
        type: 'bar',
        barMaxWidth: 22,
        data: rows.map((item) => item.count),
        itemStyle: {
          borderRadius: [5, 5, 0, 0],
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: '#4f8cff' },
              { offset: 1, color: 'rgba(79,140,255,0.15)' }
            ]
          }
        }
      }
    ]
  }
})

const labelOption = computed(() => ({
  tooltip: { trigger: 'item' },
  legend: { bottom: 0, textStyle: { color: '#97a3c4', fontSize: 11 } },
  series: [
    {
      type: 'pie',
      radius: ['42%', '68%'],
      center: ['50%', '42%'],
      data: (stats.value?.by_label || []).map((item) => ({
        name: labelText(item.label),
        value: item.count
      })),
      itemStyle: { borderColor: '#0b1020', borderWidth: 2 },
      label: { color: '#c9d4ee', fontSize: 11 }
    }
  ]
}))

const architectureOption = computed(() => {
  const rows = stats.value?.by_architecture || []
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 100, right: 30, top: 20, bottom: 30 },
    xAxis: { type: 'value' },
    yAxis: { type: 'category', data: rows.map((item) => item.architecture), inverse: true },
    series: [
      {
        type: 'bar',
        barWidth: 14,
        data: rows.map((item) => item.count),
        label: { show: true, position: 'right', fontSize: 11 },
        itemStyle: { borderRadius: [0, 6, 6, 0], color: '#36d1c4' }
      }
    ]
  }
})

async function load() {
  loading.value = true
  try {
    const [list, stat] = await Promise.all([
      api.records({ ...filter, ordering: '-created_at' }),
      api.recordStats({ days: 14 })
    ])
    records.value = list.results || []
    total.value = list.count || records.value.length
    stats.value = stat
  } finally {
    loading.value = false
  }
}

onMounted(load)

function reset() {
  Object.assign(filter, { architecture: '', label: '', source: '', search: '', page: 1 })
  load()
}

async function remove(row) {
  await api.deleteRecord(row.id)
  ElMessage.success('记录已删除')
  load()
}

async function clearAll() {
  await ElMessageBox.confirm('确定清空全部识别记录吗？该操作不可恢复。', '提示', { type: 'warning' })
  const data = await api.clearRecords({ all: true })
  ElMessage.success(`已清空 ${data.count} 条记录`)
  load()
}

</script>

<template>
  <div class="page" v-loading="loading">
    <div class="grid grid-4">
      <StatCard title="累计识别" :value="stats?.total || 0" suffix="次" icon="Picture" />
      <StatCard title="今日识别" :value="stats?.today || 0" suffix="次" color="#36d1c4" icon="Clock" />
      <StatCard title="平均置信度" :value="percent(stats?.avg_confidence, 2)" color="#ffb454"
                icon="TrendCharts" />
      <StatCard title="平均耗时" :value="stats?.avg_latency_ms || 0" suffix="ms" color="#ff6b81"
                icon="Timer" />
    </div>

    <div class="grid grid-3" style="margin-top: 14px">
      <div class="glass section">
        <div class="section-title"><span>近 14 天识别量</span></div>
        <EChart :option="dailyOption" height="240px" />
      </div>
      <div class="glass section">
        <div class="section-title"><span>识别类别分布</span></div>
        <EChart v-if="stats?.by_label?.length" :option="labelOption" height="240px" />
        <el-empty v-else description="暂无数据" :image-size="70" />
      </div>
      <div class="glass section">
        <div class="section-title"><span>模型使用分布</span></div>
        <EChart v-if="stats?.by_architecture?.length" :option="architectureOption" height="240px" />
        <el-empty v-else description="暂无数据" :image-size="70" />
      </div>
    </div>

    <div class="glass section" style="margin-top: 14px">
      <div class="section-title">
        <span>识别历史</span>
        <span class="sub">共 {{ total }} 条</span>
      </div>
      <div class="filters">
        <el-input v-model="filter.search" placeholder="按文件名搜索" clearable style="width: 200px"
                  @keyup.enter="load" />
        <el-select v-model="filter.architecture" placeholder="网络结构" clearable style="width: 150px">
          <el-option v-for="item in store.architectures" :key="item.key" :value="item.key"
                     :label="item.name" />
        </el-select>
        <el-select v-model="filter.label" placeholder="预测类别" clearable style="width: 150px">
          <el-option v-for="item in stats?.by_label || []" :key="item.label" :value="item.label"
                     :label="labelText(item.label)" />
        </el-select>
        <el-select v-model="filter.source" placeholder="来源" clearable style="width: 130px">
          <el-option v-for="(text, key) in sourceText" :key="key" :value="key" :label="text" />
        </el-select>
        <el-button type="primary" @click="load">查询</el-button>
        <el-button @click="reset">重置</el-button>
        <el-button v-if="auth.isAdmin" type="danger" plain @click="clearAll">清空全部</el-button>
      </div>

      <el-table :data="records" style="width: 100%; margin-top: 12px" size="small">
        <el-table-column label="图片" width="84">
          <template #default="{ row }">
            <el-image v-if="row.image_url" :src="row.image_url" fit="cover"
                      style="width: 48px; height: 48px; border-radius: 8px"
                      :preview-src-list="[row.image_url]" preview-teleported />
          </template>
        </el-table-column>
        <el-table-column prop="original_name" label="文件名" min-width="130" show-overflow-tooltip />
        <el-table-column prop="architecture" label="模型结构" width="110" />
        <el-table-column label="操作人" width="96">
          <template #default="{ row }">
            <el-tag size="small" effect="plain" type="info">{{ row.username || '—' }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="预测结果" width="96">
          <template #default="{ row }">
            <el-tag size="small" type="success" effect="plain">{{ labelText(row.predicted_label) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="置信度" width="88">
          <template #default="{ row }">{{ percent(row.confidence, 1) }}</template>
        </el-table-column>
        <el-table-column label="Top-3" min-width="180">
          <template #default="{ row }">
            <span class="dim" style="font-size: 12px">
              {{ (row.topk || []).slice(0, 3).map((item) => `${labelText(item.label)} ${percent(item.probability, 1)}`).join(' / ') }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="来源" width="100">
          <template #default="{ row }">{{ sourceText[row.source] || row.source }}</template>
        </el-table-column>
        <el-table-column label="耗时" width="82">
          <template #default="{ row }">{{ row.latency_ms }} ms</template>
        </el-table-column>
        <el-table-column prop="created_at" label="识别时间" width="152" />
        <el-table-column v-if="auth.isAdmin" label="操作" width="80" fixed="right">
          <template #default="{ row }">
            <el-button text size="small" type="danger" @click="remove(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>

      <el-pagination
        style="margin-top: 12px; justify-content: flex-end"
        layout="total, prev, pager, next"
        :total="total"
        :page-size="filter.page_size"
        :current-page="filter.page"
        @current-change="(page) => { filter.page = page; load() }"
      />
    </div>
  </div>
</template>

<style scoped>
.filters {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}
</style>
