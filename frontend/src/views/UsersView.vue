<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import StatCard from '../components/StatCard.vue'
import { useAuthStore } from '../store/auth'
import { dateTime } from '../utils/format'

const auth = useAuthStore()
const loading = ref(false)
const users = ref([])
const total = ref(0)
const filter = reactive({ search: '', role: '' })

const dialogVisible = ref(false)
const dialogMode = ref('create')
const formRef = ref(null)
const submitting = ref(false)
const editingId = ref(null)
const form = reactive({
  username: '', display_name: '', email: '', password: '', role: 'user', is_active: true
})

const passwordDialog = ref(false)
const passwordTarget = ref(null)
const newPassword = ref('')
const passwordSubmitting = ref(false)

const rules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, max: 30, message: '用户名长度 3~30 位', trigger: 'blur' }
  ],
  password: [
    { validator: (rule, value, callback) => {
        if (dialogMode.value === 'create' && (!value || value.length < 6)) {
          return callback(new Error('新建用户需设置至少 6 位密码'))
        }
        if (value && value.length < 6) {
          return callback(new Error('密码至少 6 位'))
        }
        return callback()
      }, trigger: 'blur' }
  ],
  role: [{ required: true, message: '请选择角色', trigger: 'change' }]
}

const stats = computed(() => ({
  total: users.value.length,
  admin: users.value.filter((item) => item.role === 'admin').length,
  normal: users.value.filter((item) => item.role === 'user').length,
  disabled: users.value.filter((item) => !item.is_active).length
}))

async function load() {
  loading.value = true
  try {
    const data = await api.users({ page_size: 100, ...filter })
    users.value = data.results || []
    total.value = data.count || users.value.length
  } finally {
    loading.value = false
  }
}

onMounted(load)

function openCreate() {
  dialogMode.value = 'create'
  editingId.value = null
  Object.assign(form, { username: '', display_name: '', email: '', password: '', role: 'user', is_active: true })
  dialogVisible.value = true
  formRef.value?.clearValidate()
}

function openEdit(row) {
  dialogMode.value = 'edit'
  editingId.value = row.id
  Object.assign(form, {
    username: row.username,
    display_name: row.display_name,
    email: row.email,
    password: '',
    role: row.role,
    is_active: row.is_active
  })
  dialogVisible.value = true
  formRef.value?.clearValidate()
}

async function submitForm() {
  await formRef.value?.validate()
  submitting.value = true
  try {
    const payload = {
      username: form.username.trim(),
      display_name: form.display_name,
      email: form.email,
      role: form.role,
      is_active: form.is_active
    }
    if (form.password) payload.password = form.password
    if (dialogMode.value === 'create') {
      await api.createUser(payload)
      ElMessage.success(`用户「${payload.username}」创建成功`)
    } else {
      await api.updateUser(editingId.value, payload)
      ElMessage.success('用户信息已更新')
    }
    dialogVisible.value = false
    await load()
  } finally {
    submitting.value = false
  }
}

function openResetPassword(row) {
  passwordTarget.value = row
  newPassword.value = ''
  passwordDialog.value = true
}

async function submitResetPassword() {
  if (!newPassword.value || newPassword.value.length < 6) {
    ElMessage.warning('新密码至少 6 位')
    return
  }
  passwordSubmitting.value = true
  try {
    await api.resetUserPassword(passwordTarget.value.id, { new_password: newPassword.value })
    ElMessage.success(`已重置「${passwordTarget.value.username}」的密码，该账号需重新登录`)
    passwordDialog.value = false
  } finally {
    passwordSubmitting.value = false
  }
}

async function toggleActive(row) {
  const action = row.is_active ? '停用' : '启用'
  await ElMessageBox.confirm(
    `确定${action}账号「${row.username}」吗？${row.is_active ? '停用后该账号无法登录，已登录的会话会立即失效。' : ''}`,
    '提示',
    { type: 'warning' }
  )
  await api.toggleUserActive(row.id)
  ElMessage.success(`已${action}账号`)
  await load()
}

async function remove(row) {
  await ElMessageBox.confirm(`确定删除账号「${row.username}」吗？该操作不可恢复。`, '提示', { type: 'warning' })
  await api.deleteUser(row.id)
  ElMessage.success('账号已删除')
  await load()
}

const isSelf = (row) => row.id === auth.user?.id
</script>

<template>
  <div class="page" v-loading="loading">
    <div class="grid grid-4">
      <StatCard title="用户总数" :value="stats.total" suffix="个" icon="User" />
      <StatCard title="管理员" :value="stats.admin" suffix="个" color="#ffb454" icon="Star"
                hint="可管理数据集、训练、模型与用户" />
      <StatCard title="普通用户" :value="stats.normal" suffix="个" color="#36d1c4" icon="UserFilled"
                hint="可识别图片、查看模型与历史" />
      <StatCard title="已停用" :value="stats.disabled" suffix="个" color="#ff6b81" icon="CircleClose" />
    </div>

    <div class="glass section" style="margin-top: 14px">
      <div class="section-title">
        <span>账号列表（{{ total }}）</span>
        <span class="sub">角色决定权限：普通用户看不到「模型训练 / 数据集 / 用户管理」，且不能删除任何数据</span>
      </div>

      <div class="toolbar">
        <el-input v-model="filter.search" placeholder="按用户名搜索" clearable style="width: 200px"
                  @keyup.enter="load" />
        <el-select v-model="filter.role" placeholder="全部角色" clearable style="width: 150px">
          <el-option value="admin" label="管理员" />
          <el-option value="user" label="普通用户" />
        </el-select>
        <el-button type="primary" @click="load">查询</el-button>
        <div style="flex: 1"></div>
        <el-button type="primary" @click="openCreate">
          <el-icon><Plus /></el-icon>
          <span style="margin-left: 4px">新建用户</span>
        </el-button>
      </div>

      <el-table :data="users" style="width: 100%; margin-top: 12px" size="small">
        <el-table-column prop="username" label="用户名" width="130" />
        <el-table-column label="姓名 / 邮箱" min-width="150">
          <template #default="{ row }">
            <div>{{ row.display_name }}</div>
            <div class="dim" style="font-size: 11px">{{ row.email || '未填写邮箱' }}</div>
          </template>
        </el-table-column>
        <el-table-column label="角色" width="100">
          <template #default="{ row }">
            <el-tag size="small" :type="row.role === 'admin' ? 'warning' : 'success'" effect="dark">
              {{ row.role === 'admin' ? '管理员' : '普通用户' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="82">
          <template #default="{ row }">
            <el-tag size="small" :type="row.is_active ? 'success' : 'danger'" effect="plain">
              {{ row.is_active ? '正常' : '已停用' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="识别次数" width="90">
          <template #default="{ row }">{{ row.record_count }}</template>
        </el-table-column>
        <el-table-column label="最后登录" width="150">
          <template #default="{ row }">{{ dateTime(row.last_login) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="216" fixed="right">
          <template #default="{ row }">
            <el-button text size="small" type="primary" @click="openEdit(row)">编辑</el-button>
            <el-button text size="small" @click="openResetPassword(row)">改密码</el-button>
            <el-button text size="small" :type="row.is_active ? 'warning' : 'success'"
                       :disabled="isSelf(row)" @click="toggleActive(row)">
              {{ row.is_active ? '停用' : '启用' }}
            </el-button>
            <el-button text size="small" type="danger" :disabled="isSelf(row)" @click="remove(row)">
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <el-dialog v-model="dialogVisible" :title="dialogMode === 'create' ? '新建用户' : '编辑用户'"
               width="480px">
      <el-form ref="formRef" :model="form" :rules="rules" label-width="90px">
        <el-form-item label="用户名" prop="username">
          <el-input v-model="form.username" placeholder="登录账号，如 zhangsan" />
        </el-form-item>
        <el-form-item label="姓名">
          <el-input v-model="form.display_name" placeholder="显示名称，如 张三" />
        </el-form-item>
        <el-form-item label="邮箱">
          <el-input v-model="form.email" placeholder="可选，可同时用于登录" />
        </el-form-item>
        <el-form-item label="密码" prop="password">
          <el-input v-model="form.password" type="password" show-password
                    :placeholder="dialogMode === 'create' ? '至少 6 位' : '留空表示不修改密码'" />
        </el-form-item>
        <el-form-item label="角色" prop="role">
          <el-radio-group v-model="form.role">
            <el-radio-button value="user">普通用户</el-radio-button>
            <el-radio-button value="admin">管理员</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="账号状态">
          <el-switch v-model="form.is_active" active-text="启用" inactive-text="停用" />
        </el-form-item>
      </el-form>
      <el-alert type="info" :closable="false" show-icon
                title="权限说明"
                description="普通用户：识别图片、多模型对比、查看模型库与识别历史；管理员：额外可管理数据集、发起训练、管理模型、清空记录与维护用户。" />
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="submitting" @click="submitForm">保存</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="passwordDialog" title="重置密码" width="420px">
      <el-form label-width="90px">
        <el-form-item label="账号">
          <el-input :model-value="passwordTarget?.username" disabled />
        </el-form-item>
        <el-form-item label="新密码">
          <el-input v-model="newPassword" type="password" show-password placeholder="至少 6 位" />
        </el-form-item>
      </el-form>
      <el-alert type="warning" :closable="false" show-icon
                title="重置后该账号的所有登录状态会立即失效，需要重新登录。" />
      <template #footer>
        <el-button @click="passwordDialog = false">取消</el-button>
        <el-button type="primary" :loading="passwordSubmitting" @click="submitResetPassword">
          确认重置
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
}
</style>
