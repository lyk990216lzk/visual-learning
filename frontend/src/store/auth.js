import { defineStore } from 'pinia'
import api from '../api'
import { clearToken, getStoredUser, getToken, setStoredUser, setToken } from '../utils/auth'

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
        return data.user
      } finally {
        this.loading = false
      }
    },

    async logout() {
      try {
        if (this.token) await api.logout()
      } catch (error) {
        // 令牌已失效时忽略服务端错误，直接清理本地登录态
      }
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
      return data
    }
  }
})
