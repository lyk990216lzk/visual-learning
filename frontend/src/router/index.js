import { createRouter, createWebHistory } from 'vue-router'
import MainLayout from '../layouts/MainLayout.vue'
import { ElMessage } from 'element-plus'
import { getStoredUser, getToken } from '../utils/auth'

const routes = [
  {
    path: '/login',
    name: 'login',
    component: () => import('../views/LoginView.vue'),
    meta: { title: '登录', public: true }
  },
  {
    path: '/',
    component: MainLayout,
    redirect: '/dashboard',
    children: [
      {
        path: 'dashboard',
        name: 'dashboard',
        component: () => import('../views/DashboardView.vue'),
        meta: { title: '系统概览', icon: 'DataAnalysis' }
      },
      {
        path: 'recognize',
        name: 'recognize',
        component: () => import('../views/RecognizeView.vue'),
        meta: { title: '图像识别', icon: 'Picture' }
      },
      {
        path: 'compare',
        name: 'compare',
        component: () => import('../views/CompareView.vue'),
        meta: { title: '模型对比', icon: 'Histogram' }
      },
      {
        path: 'training',
        name: 'training',
        component: () => import('../views/TrainingView.vue'),
        meta: { title: '模型训练', icon: 'Cpu', adminOnly: true }
      },
      {
        path: 'models',
        name: 'models',
        component: () => import('../views/ModelsView.vue'),
        meta: { title: '模型库', icon: 'Files' }
      },
      {
        path: 'datasets',
        name: 'datasets',
        component: () => import('../views/DatasetsView.vue'),
        meta: { title: '数据集', icon: 'FolderOpened', adminOnly: true }
      },
      {
        path: 'records',
        name: 'records',
        component: () => import('../views/RecordsView.vue'),
        meta: { title: '识别历史', icon: 'Clock' }
      },
      {
        path: 'users',
        name: 'users',
        component: () => import('../views/UsersView.vue'),
        meta: { title: '用户管理', icon: 'UserFilled', adminOnly: true }
      }
    ]
  },
  { path: '/:pathMatch(.*)*', redirect: '/dashboard' }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 登录守卫：未登录访问受保护页面时跳转到登录页，并记录原目标地址
router.beforeEach((to) => {
  const authenticated = Boolean(getToken())
  if (to.meta?.public) {
    if (to.name === 'login' && authenticated) return { path: '/dashboard' }
    return true
  }
  if (!authenticated) {
    const query = to.fullPath && to.fullPath !== '/' ? { redirect: to.fullPath } : {}
    return { name: 'login', query }
  }
  // 管理类页面仅管理员可进
  if (to.meta?.adminOnly && !getStoredUser()?.is_staff) {
    ElMessage.warning('该功能仅管理员可用')
    return { path: '/dashboard' }
  }
  return true
})

router.afterEach((to) => {
  document.title = to.meta?.title
    ? `${to.meta.title} · 深度学习图像识别系统`
    : '基于多种深度学习经典模型的图像识别系统'
})

export default router
