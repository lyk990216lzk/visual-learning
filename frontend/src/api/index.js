import axios from 'axios'
import { ElMessage } from 'element-plus'
import { clearToken, getToken } from '../utils/auth'
import {
  beginOperation,
  describeRequest,
  extractErrorMessage,
  finishOperation
} from '../utils/operationLog'

const http = axios.create({
  baseURL: '/api',
  timeout: 120000
})

http.interceptors.request.use((config) => {
  const token = getToken()
  if (token) {
    config.headers = config.headers || {}
    config.headers.Authorization = `Token ${token}`
  }
  // 自动记录用户发起的业务请求，静默轮询（训练进度、样例图等）会被 describeRequest 过滤
  if (config.logOperation !== false) {
    const meta = describeRequest(config)
    if (meta) config.__operationLogId = beginOperation(meta)
  }
  return config
})

http.interceptors.response.use(
  (response) => {
    if (response.config?.__operationLogId) {
      finishOperation(response.config.__operationLogId, { level: 'success' })
    }
    return response.data
  },
  (error) => {
    if (error?.config?.__operationLogId) {
      finishOperation(error.config.__operationLogId, {
        level: 'error',
        error: extractErrorMessage(error)
      })
    }
    // 登录态失效：清理本地令牌并跳回登录页
    if (error?.response?.status === 401) {
      clearToken()
      if (!window.location.pathname.startsWith('/login')) {
        const redirect = encodeURIComponent(window.location.pathname + window.location.search)
        window.location.replace(`/login?redirect=${redirect}`)
      }
      return Promise.reject(new Error('登录状态已失效，请重新登录'))
    }
    const data = error?.response?.data
    let message = data?.detail
    if (!message && data && typeof data === 'object') {
      const first = Object.values(data)[0]
      message = Array.isArray(first) ? first[0] : first
    }
    message = message || error.message || '请求失败，请稍后重试'
    if (!error.config?.silent) {
      ElMessage.error(String(message))
    }
    const requestError = new Error(String(message))
    requestError.detail = data?.detail || String(message)
    requestError.errors = Array.isArray(data?.errors) ? data.errors : []
    requestError.hint = data?.hint || ''
    requestError.data = data
    requestError.response = error.response
    return Promise.reject(requestError)
  }
)

export const api = {
  // 认证
  authConfig: () => http.get('/auth/config/'),
  login: (payload) => http.post('/auth/login/', payload, { silent: true }),
  logout: () => http.post('/auth/logout/'),
  profile: () => http.get('/auth/profile/'),
  updateProfile: (payload) => http.patch('/auth/profile/', payload),
  changePassword: (payload) => http.post('/auth/password/', payload),

  // 用户管理（仅管理员）
  users: (params) => http.get('/users/', { params }),
  createUser: (payload) => http.post('/users/', payload),
  updateUser: (id, payload) => http.patch(`/users/${id}/`, payload),
  deleteUser: (id) => http.delete(`/users/${id}/`),
  resetUserPassword: (id, payload) => http.post(`/users/${id}/reset-password/`, payload),
  toggleUserActive: (id) => http.post(`/users/${id}/toggle-active/`),

  // 概览 / 系统
  dashboard: (params) => http.get('/dashboard/', { params }),
  system: () => http.get('/system/'),
  health: () => http.get('/health/'),

  // 模型结构
  architectures: () => http.get('/architectures/'),
  architectureDetail: (key, params) => http.get(`/architectures/${key}/`, { params }),
  architectureStat: (key, params) => http.get(`/architectures/${key}/stat/`, { params }),

  // 数据集
  datasets: () => http.get('/datasets/'),
  datasetSamples: (id, params) => http.get(`/datasets/${id}/samples/`, { params }),
  refreshDataset: (id) => http.post(`/datasets/${id}/refresh/`),
  deleteDataset: (id) => http.delete(`/datasets/${id}/`),
  buildDemoDataset: (payload) => http.post('/datasets/build-demo/', payload),
  uploadDataset: (formData, config = {}) => http.post('/datasets/', formData, config),

  // 训练
  trainingRuns: (params) => http.get('/training-runs/', { params }),
  trainingRun: (id) => http.get(`/training-runs/${id}/`),
  createTraining: (payload) => http.post('/training-runs/', payload),
  cancelTraining: (id) => http.post(`/training-runs/${id}/cancel/`),
  deleteTraining: (id) => http.delete(`/training-runs/${id}/`),
  trainingProgress: (id) => http.get(`/training-runs/${id}/progress/`, { silent: true }),
  trainingLog: (id, tail = 120) => http.get(`/training-runs/${id}/log/`, { params: { tail } }),

  // 已训练模型
  trainedModels: (params) => http.get('/trained-models/', { params }),
  activateModel: (id) => http.post(`/trained-models/${id}/activate/`),
  evaluateModel: (id, payload) => http.post(`/trained-models/${id}/evaluate/`, payload || {}),
  deleteModel: (id) => http.delete(`/trained-models/${id}/`),
  downloadModelUrl: (id) => `/api/trained-models/${id}/download/`,

  // 识别
  recognize: (formData) => http.post('/recognize/', formData),
  recognizeCompare: (formData) => http.post('/recognize/compare/', formData),
  recognizeBatch: (formData) => http.post('/recognize/batch/', formData),
  // 内置样例图：该接口同样需要 Token 鉴权，<img src> 无法携带 Authorization 请求头，
  // 因此统一用 axios 取回二进制内容，再由调用方转成 blob URL。
  sampleImageBlob: (label, size = 64, options = {}) =>
    http.get('/recognize/sample/', { params: { label, size }, responseType: 'blob', ...options }),

  // 识别记录
  records: (params) => http.get('/records/', { params }),
  recordStats: (params) => http.get('/records/stats/', { params }),
  deleteRecord: (id) => http.delete(`/records/${id}/`),
  clearRecords: (payload) => http.post('/records/clear/', payload)
}

export default api
