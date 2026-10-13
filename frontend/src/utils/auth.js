/**
 * 登录态存储：勾选「记住我」写入 localStorage，否则写入 sessionStorage（关闭标签页即失效）。
 */

const TOKEN_KEY = 'ir_token'
const USER_KEY = 'ir_user'

function stores() {
  return [localStorage, sessionStorage]
}

export function getToken() {
  for (const store of stores()) {
    const token = store.getItem(TOKEN_KEY)
    if (token) return token
  }
  return ''
}

export function setToken(token, remember = true) {
  clearToken()
  const store = remember ? localStorage : sessionStorage
  store.setItem(TOKEN_KEY, token)
}

export function getStoredUser() {
  for (const store of stores()) {
    const raw = store.getItem(USER_KEY)
    if (!raw) continue
    try {
      return JSON.parse(raw)
    } catch (error) {
      return null
    }
  }
  return null
}

export function setStoredUser(user, remember = true) {
  const store = remember ? localStorage : sessionStorage
  store.setItem(USER_KEY, JSON.stringify(user || {}))
}

export function clearToken() {
  for (const store of stores()) {
    store.removeItem(TOKEN_KEY)
    store.removeItem(USER_KEY)
  }
}
