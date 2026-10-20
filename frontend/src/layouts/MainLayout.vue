<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useAppStore } from '../store'
import { useAuthStore } from '../store/auth'
import OperationLogPanel from '../components/OperationLogPanel.vue'
import { logAction } from '../utils/operationLog'

const route = useRoute()
const router = useRouter()
const store = useAppStore()
const auth = useAuthStore()
const collapsed = ref(false)
const passwordDialog = ref(false)
const passwordFormRef = ref(null)
const submitting = ref(false)

const passwordForm = reactive({ oldPassword: '', newPassword: '', confirmPassword: '' })

const passwordRules = {
  oldPassword: [{ required: true, message: '请输入原密码', trigger: 'blur' }],
  newPassword: [
    { required: true, message: '请输入新密码', trigger: 'blur' },
    { min: 6, message: '新密码至少 6 位', trigger: 'blur' }
  ],
  confirmPassword: [
    { required: true, message: '请再次输入新密码', trigger: 'blur' },
    {
      validator: (rule, value, callback) =>
        value === passwordForm.newPassword ? callback() : callback(new Error('两次输入的密码不一致')),
      trigger: 'blur'
    }
  ]
}

const menus = computed(() =>
  router.getRoutes()
    .filter((item) => item.meta?.title && !item.meta?.public)
    .filter((item) => !item.meta?.adminOnly || auth.isAdmin)
    .sort((a, b) => menuOrder(a.path) - menuOrder(b.path))
)

function menuOrder(path) {
  const order = ['/dashboard', '/recognize', '/compare', '/models', '/records', '/training', '/datasets', '/users']
  return order.indexOf(path)
}

const deviceText = computed(() => {
  const system = store.system
  if (!system) return ''
  const device = system.device?.toUpperCase?.() || 'CPU'
  return `${device} · PyTorch ${system.torch}`
})

const dbText = computed(() => {
  const db = store.system?.database
  if (!db) return ''
  return db.version ? `${db.label} ${db.version}` : db.label
})

const avatarText = computed(() => (auth.displayName || 'U').slice(0, 1).toUpperCase())

onMounted(() => {
  store.bootstrap()
  auth.fetchProfile().catch(() => {})
})

async function refreshAll() {
  logAction({
    action: '刷新系统数据',
    detail: '模型结构 / 数据集 / 模型库 / 系统信息',
    module: '系统',
    level: 'info'
  })
  await store.bootstrap()
}

async function handleCommand(command) {
  if (command === 'password') {
    passwordForm.oldPassword = ''
    passwordForm.newPassword = ''
    passwordForm.confirmPassword = ''
    passwordDialog.value = true
    return
  }
  if (command === 'logout') {
    await ElMessageBox.confirm('确定要退出登录吗？', '提示', { type: 'warning' })
    await auth.logout()
    ElMessage.success('已安全退出登录')
    router.replace({ name: 'login' })
  }
}

async function submitPassword() {
  await passwordFormRef.value?.validate()
  submitting.value = true
  try {
    await auth.changePassword({
      oldPassword: passwordForm.oldPassword,
      newPassword: passwordForm.newPassword
    })
    passwordDialog.value = false
    ElMessage.success('密码修改成功，其它设备上的登录状态已失效')
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <el-container style="height: 100vh">
    <el-aside :width="collapsed ? '64px' : '226px'" class="sidebar">
      <div class="logo">
        <div class="logo-mark">XJNU</div>
        <div v-if="!collapsed" class="logo-text">
          <div class="logo-title">图像识别系统</div>
          <div class="logo-sub">{{ auth.university }} · 经典深度模型</div>
        </div>
      </div>

      <el-menu :default-active="route.path" :collapse="collapsed" router class="sidebar-menu">
        <el-menu-item v-for="item in menus" :key="item.path" :index="item.path">
          <el-icon><component :is="item.meta.icon" /></el-icon>
          <template #title>{{ item.meta.title }}</template>
        </el-menu-item>
      </el-menu>

      <div class="sidebar-footer">
        <el-button text style="width: 100%" @click="collapsed = !collapsed">
          <el-icon><component :is="collapsed ? 'Expand' : 'Fold'" /></el-icon>
          <span v-if="!collapsed" style="margin-left: 8px">收起菜单</span>
        </el-button>
      </div>
    </el-aside>

    <el-container>
      <el-header class="topbar">
        <div>
          <div class="topbar-title">{{ route.meta.title }}</div>
          <div class="topbar-sub">
            {{ auth.university }} · 支持 LeNet-5 / AlexNet / VGG / GoogLeNet / ResNet / DenseNet / MobileNetV2
          </div>
        </div>
        <div class="topbar-right">
          <el-tag v-if="deviceText" type="info" effect="dark" round>{{ deviceText }}</el-tag>
          <el-tag v-if="dbText" type="warning" effect="dark" round>数据库 {{ dbText }}</el-tag>
          <el-tag type="success" effect="dark" round>已就绪模型 {{ store.trainedModels.length }}</el-tag>
          <el-tooltip content="刷新系统数据" placement="bottom">
            <el-button :icon="'Refresh'" circle @click="refreshAll" />
          </el-tooltip>

          <el-dropdown trigger="click" @command="handleCommand">
            <div class="user-chip">
              <el-avatar :size="30" class="user-avatar">{{ avatarText }}</el-avatar>
              <div class="user-meta">
                <span class="user-name">{{ auth.displayName }}</span>
                <span class="user-role">{{ auth.role }}</span>
              </div>
              <el-icon class="user-arrow"><ArrowDown /></el-icon>
            </div>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item disabled>
                  账号：{{ auth.user?.username || '—' }}
                </el-dropdown-item>
                <el-dropdown-item disabled v-if="auth.user?.last_login">
                  上次登录：{{ auth.user.last_login }}
                </el-dropdown-item>
                <el-dropdown-item divided command="password">
                  <el-icon><Lock /></el-icon>修改密码
                </el-dropdown-item>
                <el-dropdown-item command="logout">
                  <el-icon><SwitchButton /></el-icon>退出登录
                </el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </el-header>

      <el-main style="padding: 0; overflow-y: auto">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </el-main>
    </el-container>

    <el-dialog v-model="passwordDialog" title="修改密码" width="440px">
      <el-form ref="passwordFormRef" :model="passwordForm" :rules="passwordRules" label-width="90px">
        <el-form-item label="原密码" prop="oldPassword">
          <el-input v-model="passwordForm.oldPassword" type="password" show-password
                    placeholder="请输入当前密码" />
        </el-form-item>
        <el-form-item label="新密码" prop="newPassword">
          <el-input v-model="passwordForm.newPassword" type="password" show-password
                    placeholder="至少 6 位" />
        </el-form-item>
        <el-form-item label="确认密码" prop="confirmPassword">
          <el-input v-model="passwordForm.confirmPassword" type="password" show-password
                    placeholder="再次输入新密码" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="passwordDialog = false">取消</el-button>
        <el-button type="primary" :loading="submitting" @click="submitPassword">确认修改</el-button>
      </template>
    </el-dialog>

    <OperationLogPanel />
  </el-container>
</template>

<style scoped>
.sidebar {
  display: flex;
  flex-direction: column;
  background: rgba(12, 17, 32, 0.92);
  border-right: 1px solid var(--ir-border);
  transition: width 0.2s ease;
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 16px 14px;
  border-bottom: 1px solid var(--ir-border);
}

.logo-mark {
  width: 52px;
  height: 38px;
  border-radius: 11px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 12px;
  letter-spacing: 0.5px;
  color: #fff;
  background: linear-gradient(135deg, #b4232a, #4f8cff);
  flex: none;
}

.logo-title {
  font-size: 15px;
  font-weight: 600;
}

.logo-sub {
  font-size: 11px;
  color: var(--ir-text-dim);
}

.sidebar-menu {
  flex: 1;
  padding-top: 8px;
}

.sidebar-footer {
  padding: 10px;
  border-top: 1px solid var(--ir-border);
}

.topbar {
  height: 68px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 22px;
  border-bottom: 1px solid var(--ir-border);
  background: rgba(12, 17, 32, 0.7);
  backdrop-filter: blur(8px);
}

.topbar-title {
  font-size: 16px;
  font-weight: 600;
}

.topbar-sub {
  font-size: 12px;
  color: var(--ir-text-dim);
  margin-top: 2px;
}

.topbar-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.user-chip {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 10px 4px 4px;
  border-radius: 999px;
  border: 1px solid var(--ir-border);
  background: rgba(79, 140, 255, 0.08);
  cursor: pointer;
  outline: none;
}

.user-avatar {
  background: linear-gradient(135deg, #b4232a, #4f8cff);
  font-weight: 600;
}

.user-meta {
  display: flex;
  flex-direction: column;
  line-height: 1.25;
}

.user-name {
  font-size: 13px;
  font-weight: 600;
}

.user-role {
  font-size: 11px;
  color: var(--ir-text-dim);
}

.user-arrow {
  color: var(--ir-text-dim);
  font-size: 12px;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.18s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
