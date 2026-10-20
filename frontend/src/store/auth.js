import { defineStore } from 'pinia'
import api from '../api'
import { clearToken, getStoredUser, getToken, setStoredUser, setToken } from '../utils/auth'
import { logAction } from '../utils/operationLog'

export const DEFAULT_CONFIG = {
  site_name: '基于多种深度学习经典模型的图像识别系统',
  university: '新疆师范大学',
  university_en: 'Xinjiang Normal University',
  university_badge: 'XJNU · 1978',
  demo_accounts: []
}

export const useAuthStore = defineStore('auth', {
  state: () => ({
    token: getToken(),
    user: getStoredUser(),
    config: { ...DEFAULT_CONFIG },
    loading: false
  }),
  getters: {
    isAuthenticated: (state) => Boolean(state.token),
    isAdmin: (state) => Boolean(state.user?.is_staff),
    displayName: (state) => state.user?.display_name || state.user?.username || '未登录',
    role: (state) => state.user?.role || '',
    university: (state) => state.config?.university || DEFAULT_CONFIG.university
  },
  actions: {
    async loadConfig() {
      try {
        const data = await api.authConfig()
        this.config = { ...DEFAULT_CONFIG, ...data }
      } catch (error) {
        this.config = { ...DEFAULT_CONFIG }
      }
      return this.config
    },

    async login({ username, password, remember = true }) {
      this.loading = true
      try {
        const data = await api.login({ username, password })
        this.token = data.token
        this.user = data.user
        setToken(data.token, remember)
        setStoredUser(data.user, remember)
        logAction({
          action: '登录成功',
          detail: `账号 ${data.user?.username || username}`,
          module: '账号'
        })
        return data.user
      } catch (error) {
        logAction({
          action: '登录失败',
          detail: `账号 ${username}`,
          module: '账号',
          level: 'error',
          error: error?.message || '登录失败'
        })
        throw error
      } finally {
        this.loading = false
      }
    },

    async logout() {
      const username = this.user?.username
      try {
        if (this.token) await api.logout()
      } catch (error) {
        // 令牌已失效时忽略服务端错误，直接清理本地登录态
      }
      logAction({
        action: '退出登录',
        detail: username ? `账号 ${username}` : '',
        module: '账号',
        level: 'info',
        user: this.displayName
      })
      this.token = ''
      this.user = null
      clearToken()
    },

    async fetchProfile() {
      const user = await api.profile()
      this.user = user
      setStoredUser(user, Boolean(localStorage.getItem('ir_token')))
      return user
    },

    async updateProfile(payload) {
      const data = await api.updateProfile(payload)
      this.user = data.user
      setStoredUser(data.user, Boolean(localStorage.getItem('ir_token')))
      return data.user
    },

    async changePassword({ oldPassword, newPassword }) {
      const data = await api.changePassword({ old_password: oldPassword, new_password: newPassword })
      this.token = data.token
      this.user = data.user
      setToken(data.token, Boolean(localStorage.getItem('ir_token')))
      setStoredUser(data.user, Boolean(localStorage.getItem('ir_token')))
      logAction({
        action: '密码修改成功',
        detail: '登录令牌已刷新，其它设备上的登录状态已失效',
        module: '账号'
      })
      return data
    }
  }
})
