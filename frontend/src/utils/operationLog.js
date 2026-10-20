/**
 * 前端操作日志：集中记录用户在当前浏览器会话里做过的操作。
 *
 * - 页面访问由 router.afterEach 写入；
 * - 接口调用由 axios 拦截器自动写入（含耗时、成功/失败）；
 * - 识别结果、训练完成等关键结果由页面补充写入。
 *
 * 日志保存在 sessionStorage 中，刷新页面不会丢，关闭标签页后自动清空。
 */

import { reactive } from 'vue'
import { getStoredUser } from './auth'

const STORAGE_KEY = 'ir_operation_logs'
const MAX_ENTRIES = 300

export const operationLogs = reactive([])

let sequence = 0

function pad(value) {
  return String(value).padStart(2, '0')
}

function formatTime(timestamp) {
  const date = new Date(timestamp)
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
}

function formatDateTime(timestamp) {
  const date = new Date(timestamp)
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
    `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
}

function nextId() {
  sequence += 1
  return `${Date.now()}-${sequence}`
}

export function currentUserName() {
  const user = getStoredUser()
  return user?.display_name || user?.username || '未登录'
}

function persist() {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(operationLogs))
  } catch (error) {
    // 存储空间不足或被禁用时忽略，日志仍保留在内存里
  }
}

function restore() {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY)
    if (!raw) return
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return
    parsed.slice(-MAX_ENTRIES).forEach((item) => {
      if (!item || typeof item !== 'object' || !item.action) return
      const timestamp = Number(item.ts) || Date.now()
      operationLogs.push({
        id: item.id || nextId(),
        ts: timestamp,
        time: formatTime(timestamp),
        level: item.level || 'info',
        module: item.module || '系统',
        action: item.action,
        detail: item.detail || '',
        error: item.error || '',
        duration: Number.isFinite(item.duration) ? item.duration : null,
        user: item.user || ''
      })
    })
  } catch (error) {
    // 损坏的缓存直接忽略
  }
}

restore()

/**
 * 写入一条已完成的日志，返回日志 id。
 * level: info | success | error
 */
export function logAction({
  action,
  detail = '',
  module = '系统',
  level = 'success',
  error = '',
  duration = null,
  user = null
}) {
  const ts = Date.now()
  const entry = {
    id: nextId(),
    ts,
    time: formatTime(ts),
    level,
    module,
    action,
    detail,
    error,
    duration: Number.isFinite(duration) ? Math.round(duration) : null,
    user: user || currentUserName()
  }
  operationLogs.push(entry)
  if (operationLogs.length > MAX_ENTRIES) {
    operationLogs.splice(0, operationLogs.length - MAX_ENTRIES)
  }
  persist()
  return entry.id
}

/** 请求开始时写入「执行中」日志，请求结束后原地更新为成功 / 失败。 */
export function beginOperation({ action, detail = '', module = '系统' }) {
  const ts = Date.now()
  const entry = {
    id: nextId(),
    ts,
    time: formatTime(ts),
    level: 'running',
    module,
    action,
    detail,
    error: '',
    duration: null,
    user: currentUserName()
  }
  operationLogs.push(entry)
  if (operationLogs.length > MAX_ENTRIES) {
    operationLogs.splice(0, operationLogs.length - MAX_ENTRIES)
  }
  persist()
  return entry.id
}

export function finishOperation(id, { level = 'success', error = '' } = {}) {
  const entry = operationLogs.find((item) => item.id === id)
  if (!entry) return
  entry.level = level
  entry.error = error
  entry.duration = Math.max(0, Date.now() - entry.ts)
  persist()
}

export function clearLogs() {
  operationLogs.splice(0, operationLogs.length)
  persist()
}

export function exportLogs(entries = operationLogs) {
  return entries
    .map((entry) => {
      const status = entry.level === 'error' ? '失败' : entry.level === 'running' ? '执行中' : '成功'
      const duration = entry.duration != null ? ` (${entry.duration}ms)` : ''
      const error = entry.error ? ` | 错误：${entry.error}` : ''
      const detail = entry.detail ? ` | ${entry.detail}` : ''
      return `[${formatDateTime(entry.ts)}] [${status}] [${entry.module}] ${entry.action}${detail}${error}${duration} | 用户：${entry.user || '-'}`
    })
    .join('\n')
}

// ---------------------------------------------------------------- 接口 -> 业务动作

/** 这些接口属于高频轮询 / 静默加载，不写入操作日志 */
const SILENT_URLS = [
  /^\/recognize\/sample$/,
  /^\/training-runs\/\d+\/progress$/,
  /^\/health$/,
  /^\/auth\/config$/,
  /^\/system$/
]

/**
 * method 为可选约束；read 表示该 GET 请求是用户主动查看，需要记录。
 * 匹配顺序从上到下，越具体的规则越靠前。
 */
const ACTION_RULES = [
  { method: 'POST', test: /^\/auth\/login$/, action: '登录系统', module: '账号' },
  { method: 'POST', test: /^\/auth\/logout$/, action: '退出登录', module: '账号' },
  { method: 'POST', test: /^\/auth\/password$/, action: '修改登录密码', module: '账号' },
  { method: 'PATCH', test: /^\/auth\/profile$/, action: '更新个人资料', module: '账号' },

  { method: 'POST', test: /^\/datasets\/build-demo$/, action: '生成演示数据集', module: '数据集' },
  { method: 'POST', test: /^\/datasets$/, action: '上传数据集', module: '数据集' },
  { method: 'POST', test: /^\/datasets\/\d+\/refresh$/, action: '刷新数据集统计', module: '数据集' },
  { method: 'DELETE', test: /^\/datasets\/\d+$/, action: '删除数据集', module: '数据集' },
  { method: 'GET', test: /^\/datasets\/\d+\/samples$/, action: '查看数据集样例', module: '数据集', read: true },

  { method: 'POST', test: /^\/training-runs$/, action: '创建训练任务', module: '训练' },
  { method: 'POST', test: /^\/training-runs\/\d+\/cancel$/, action: '中止训练任务', module: '训练' },
  { method: 'DELETE', test: /^\/training-runs\/\d+$/, action: '删除训练任务', module: '训练' },

  { method: 'POST', test: /^\/trained-models\/\d+\/activate$/, action: '设为默认模型', module: '模型' },
  { method: 'POST', test: /^\/trained-models\/\d+\/evaluate$/, action: '重新评估模型', module: '模型' },
  { method: 'DELETE', test: /^\/trained-models\/\d+$/, action: '删除模型', module: '模型' },
  { method: 'GET', test: /^\/architectures\/[\w-]+\/stat$/, action: '查看模型统计', module: '模型', read: true },
  { method: 'GET', test: /^\/architectures\/[\w-]+$/, action: '查看模型结构', module: '模型', read: true },

  { method: 'POST', test: /^\/recognize\/compare$/, action: '多模型对比识别', module: '识别' },
  { method: 'POST', test: /^\/recognize\/batch$/, action: '批量识别图片', module: '识别' },
  { method: 'POST', test: /^\/recognize$/, action: '单图识别', module: '识别' },

  { method: 'POST', test: /^\/records\/clear$/, action: '清空识别记录', module: '识别记录' },
  { method: 'DELETE', test: /^\/records\/\d+$/, action: '删除识别记录', module: '识别记录' },

  { method: 'POST', test: /^\/users$/, action: '新建用户', module: '用户' },
  { method: 'POST', test: /^\/users\/\d+\/reset-password$/, action: '重置用户密码', module: '用户' },
  { method: 'POST', test: /^\/users\/\d+\/toggle-active$/, action: '启用 / 禁用用户', module: '用户' },
  { method: 'PATCH', test: /^\/users\/\d+$/, action: '编辑用户信息', module: '用户' },
  { method: 'DELETE', test: /^\/users\/\d+$/, action: '删除用户', module: '用户' }
]

function normalizePath(url = '') {
  let path = String(url).split('?')[0]
  path = path.replace(/^https?:\/\/[^/]+/i, '')
  path = path.replace(/^\/api(?=\/|$)/, '')
  if (!path.startsWith('/')) path = `/${path}`
  if (path.length > 1) path = path.replace(/\/+$/, '')
  return path || '/'
}

function summarizeFormData(data) {
  const parts = []
  let fileCount = 0
  try {
    for (const [key, value] of data.entries()) {
      if (typeof File !== 'undefined' && value instanceof File) {
        if (key === 'images') {
          fileCount += 1
        } else {
          parts.push(`文件 ${value.name}`)
        }
      } else if (['name', 'architecture', 'architectures', 'top_k', 'image_size', 'label'].includes(key)) {
        if (value !== '') parts.push(`${key}=${value}`)
      }
    }
  } catch (error) {
    return ''
  }
  if (fileCount) parts.unshift(`${fileCount} 张图片`)
  return parts.slice(0, 4).join(' · ')
}

const DETAIL_KEYS = [
  'username', 'display_name', 'name', 'architecture', 'architectures', 'dataset',
  'epochs', 'learning_rate', 'per_class', 'all', 'days'
]

const SECRET_KEY = /password|token|secret|authorization/i

function summarizeData(data) {
  if (!data) return ''
  if (typeof FormData !== 'undefined' && data instanceof FormData) {
    return summarizeFormData(data)
  }
  if (typeof data === 'string') {
    try {
      data = JSON.parse(data)
    } catch (error) {
      return ''
    }
  }
  if (typeof data !== 'object' || Array.isArray(data)) return ''
  const parts = []
  for (const key of DETAIL_KEYS) {
    if (SECRET_KEY.test(key)) continue
    const value = data[key]
    if (value === undefined || value === null || value === '' || typeof value === 'object') continue
    parts.push(`${key}=${value}`)
    if (parts.length >= 4) break
  }
  return parts.join(' · ')
}

function summarizeParams(params) {
  if (!params || typeof params !== 'object') return ''
  const ignored = new Set(['page', 'page_size', 'ordering', 'format'])
  const parts = []
  for (const [key, value] of Object.entries(params)) {
    if (ignored.has(key) || value === undefined || value === null || value === '') continue
    if (typeof value === 'object') continue
    parts.push(`${key}=${value}`)
    if (parts.length >= 3) break
  }
  return parts.join(' · ')
}

/**
 * 根据 axios 请求配置生成日志描述。
 * 返回 null 表示该请求不需要记录（静默轮询、后台加载等）。
 */
export function describeRequest(config = {}) {
  const method = String(config.method || 'get').toUpperCase()
  const path = normalizePath(config.url)
  if (SILENT_URLS.some((pattern) => pattern.test(path))) return null

  const rule = ACTION_RULES.find((item) => {
    if (item.method && item.method !== method) return false
    return item.test.test(path)
  })

  if (!rule) {
    // 默认只记录写操作，GET 多为后台加载，避免刷屏
    if (method === 'GET') return null
    return {
      action: `${method} ${path}`,
      module: '系统',
      detail: `接口 ${path}`
    }
  }

  if (method === 'GET' && !rule.read) return null

  const pathText = path === '/' ? '/' : `${path}/`
  const summary = summarizeData(config.data)
  const params = summarizeParams(config.params)
  const detail = [summary, params].filter(Boolean).join(' · ')

  return {
    action: rule.action,
    module: rule.module,
    detail: detail ? `${method} ${pathText} · ${detail}` : `${method} ${pathText}`
  }
}

/** 从 axios 错误里提取后端返回的可读信息 */
export function extractErrorMessage(error) {
  const data = error?.response?.data
  let message = data?.detail
  if (!message && data && typeof data === 'object') {
    const first = Object.values(data)[0]
    message = Array.isArray(first) ? first[0] : first
  }
  message = message || error?.message || '请求失败'
  message = String(message)
  return message.length > 160 ? `${message.slice(0, 157)}…` : message
}
