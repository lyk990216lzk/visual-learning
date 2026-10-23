<script setup>
import { onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '../store/auth'

const router = useRouter()
const route = useRoute()
const auth = useAuthStore()

const formRef = ref(null)
const form = reactive({ username: '', password: '', remember: true })
const logoFailed = ref(false)
const errorMessage = ref('')
// 如需使用学校官方校徽，把图片放到 frontend/public/logo.png 即可自动替换（缺失时回退到内置占位徽标）
const logoUrl = '/logo.png'

const rules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
}

const features = [
  { icon: 'Cpu', title: '10 种经典卷积网络', desc: 'LeNet-5 / AlexNet / VGG / GoogLeNet / ResNet / DenseNet / MobileNetV2 等' },
  { icon: 'TrendCharts', title: '在线训练与实时监控', desc: '图形化配置超参数，Loss/Acc 曲线与训练日志实时刷新' },
  { icon: 'Histogram', title: '多模型对比识别', desc: '同一张图片送入多个模型，给出共识结果与概率对比' },
  { icon: 'Grid', title: '完整评估体系', desc: '混淆矩阵、分类别精确率/召回率/F1、识别历史统计' }
]

onMounted(() => {
  auth.loadConfig()
})

function fillDemo(account) {
  form.username = account.username
  form.password = account.password
  errorMessage.value = ''
}

async function submit() {
  await formRef.value?.validate()
  errorMessage.value = ''
  try {
    const user = await auth.login({
      username: form.username.trim(),
      password: form.password,
      remember: form.remember
    })
    ElMessage.success(`欢迎回来，${user.display_name || user.username}`)
    const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '/dashboard'
    router.replace(redirect)
  } catch (error) {
    errorMessage.value = error.message || '登录失败，请检查用户名与密码'
  }
}
</script>

<template>
  <div class="login-page">
    <div class="glow glow-a"></div>
    <div class="glow glow-b"></div>

    <!-- 左侧：学校与系统介绍 -->
    <section class="brand-side">
      <div class="brand-top">
        <img v-if="!logoFailed" class="emblem-img" :src="logoUrl" alt="校徽" @error="logoFailed = true" />
        <svg v-else class="emblem-img" viewBox="0 0 120 120" aria-label="校徽占位图">
          <defs>
            <linearGradient id="emblemGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#ff6b81" />
              <stop offset="100%" stop-color="#4f8cff" />
            </linearGradient>
          </defs>
          <circle cx="60" cy="60" r="55" fill="rgba(255,255,255,0.03)" stroke="url(#emblemGrad)" stroke-width="3" />
          <circle cx="60" cy="60" r="46" fill="none" stroke="rgba(255,255,255,0.16)" stroke-width="1" />
          <text x="60" y="50" text-anchor="middle" font-size="14" fill="#e8edf8" letter-spacing="1.5">新疆师范大学</text>
          <text x="60" y="76" text-anchor="middle" font-size="18" font-weight="700" fill="url(#emblemGrad)" letter-spacing="2">XJNU</text>
          <text x="60" y="94" text-anchor="middle" font-size="9" fill="#97a3c4" letter-spacing="3">1978</text>
        </svg>
        <div class="brand-names">
          <h1 class="uni-name">{{ auth.config.university }}</h1>
          <div class="uni-en">{{ auth.config.university_en }}</div>
        </div>
      </div>

      <h2 class="system-name">{{ auth.config.site_name }}</h2>
      <p class="system-desc">
        面向视觉学习任务的多模型实验与识别平台，从零复现经典卷积神经网络，
        支持数据集管理、在线训练、模型评估与多模型对比推理。
      </p>

      <ul class="feature-list">
        <li v-for="item in features" :key="item.title">
          <el-icon class="feature-icon"><component :is="item.icon" /></el-icon>
          <div>
            <div class="feature-title">{{ item.title }}</div>
            <div class="feature-desc">{{ item.desc }}</div>
          </div>
        </li>
      </ul>

      <div class="brand-footer">
        © 2026 {{ auth.config.university }} · 计算机科学技术学院
      </div>
    </section>

    <!-- 右侧：登录表单 -->
    <section class="form-side">
      <div class="login-card">
        <div class="card-head">
          <div class="card-title">账号登录</div>
          <div class="card-sub">请使用系统分配的账号登录，未授权访问将被拦截</div>
        </div>

        <el-form ref="formRef" :model="form" :rules="rules" size="large" @submit.prevent>
          <el-form-item prop="username">
            <el-input v-model="form.username" placeholder="用户名 / 邮箱" clearable
                      :prefix-icon="'User'" @keyup.enter="submit" />
          </el-form-item>
          <el-form-item prop="password">
            <el-input v-model="form.password" type="password" placeholder="密码" show-password
                      :prefix-icon="'Lock'" @keyup.enter="submit" />
          </el-form-item>

          <div class="form-row">
            <el-checkbox v-model="form.remember">记住我</el-checkbox>
            <el-tooltip content="请联系系统管理员重置密码" placement="top">
              <span class="forget">忘记密码？</span>
            </el-tooltip>
          </div>

          <el-alert v-if="errorMessage" :title="errorMessage" type="error" show-icon :closable="false"
                    style="margin-bottom: 14px" />

          <el-button type="primary" size="large" style="width: 100%" :loading="auth.loading"
                     @click="submit">
            <el-icon v-if="!auth.loading"><Key /></el-icon>
            <span style="margin-left: 6px">登 录</span>
          </el-button>
        </el-form>

        <div v-if="auth.config.demo_accounts?.length" class="demo-box">
          <div class="demo-title">
            <el-icon><InfoFilled /></el-icon>
            <span>演示账号（点击自动填充）</span>
          </div>
          <div class="demo-list">
            <div v-for="account in auth.config.demo_accounts" :key="account.username" class="demo-chip"
                 @click="fillDemo(account)">
              <span class="demo-role">{{ account.role }}</span>
              <span class="demo-account">{{ account.username }} / {{ account.password }}</span>
              <span class="demo-note">{{ account.note }}</span>
            </div>
          </div>
        </div>

        <div class="card-foot">
          <span>{{ auth.config.university_badge }}</span>
          <span>系统版本 v1.0</span>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.login-page {
  position: relative;
  min-height: 100vh;
  display: grid;
  grid-template-columns: 1.15fr 0.85fr;
  align-items: center;
  gap: 40px;
  padding: 48px 6vw;
  overflow: hidden;
}

.glow {
  position: absolute;
  border-radius: 50%;
  filter: blur(90px);
  pointer-events: none;
}

.glow-a {
  width: 520px;
  height: 520px;
  background: rgba(180, 35, 42, 0.28);
  top: -160px;
  left: -120px;
}

.glow-b {
  width: 560px;
  height: 560px;
  background: rgba(79, 140, 255, 0.26);
  bottom: -200px;
  right: -140px;
}

.brand-side {
  position: relative;
  z-index: 1;
  max-width: 720px;
}

.brand-top {
  display: flex;
  align-items: center;
  gap: 18px;
}

.emblem-img {
  width: 96px;
  height: 96px;
  object-fit: contain;
  flex: none;
  filter: drop-shadow(0 6px 18px rgba(0, 0, 0, 0.35));
}

.uni-name {
  margin: 0;
  font-size: 30px;
  font-weight: 700;
  letter-spacing: 3px;
  background: linear-gradient(120deg, #ffffff, #cfe0ff);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.uni-en {
  margin-top: 4px;
  font-size: 13px;
  letter-spacing: 1.6px;
  text-transform: uppercase;
  color: var(--ir-text-dim);
}

.system-name {
  margin: 30px 0 12px;
  font-size: 34px;
  line-height: 1.32;
  font-weight: 700;
  background: linear-gradient(120deg, #ff8b9a, #4f8cff 55%, #36d1c4);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.system-desc {
  margin: 0;
  max-width: 560px;
  font-size: 14px;
  line-height: 1.9;
  color: var(--ir-text-dim);
}

.feature-list {
  list-style: none;
  padding: 0;
  margin: 30px 0 0;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 14px;
}

.feature-list li {
  display: flex;
  gap: 12px;
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid var(--ir-border);
  background: rgba(20, 27, 48, 0.6);
}

.feature-icon {
  color: #4f8cff;
  font-size: 18px;
  margin-top: 2px;
}

.feature-title {
  font-size: 14px;
  font-weight: 600;
}

.feature-desc {
  margin-top: 3px;
  font-size: 12px;
  line-height: 1.7;
  color: var(--ir-text-dim);
}

.brand-footer {
  margin-top: 34px;
  font-size: 12px;
  color: var(--ir-text-dim);
  letter-spacing: 0.3px;
}

.form-side {
  position: relative;
  z-index: 1;
  display: flex;
  justify-content: center;
}

.login-card {
  width: 100%;
  max-width: 420px;
  padding: 32px 30px 22px;
  border-radius: 18px;
  border: 1px solid var(--ir-border);
  background: rgba(18, 25, 45, 0.88);
  backdrop-filter: blur(14px);
  box-shadow: 0 24px 60px rgba(3, 7, 18, 0.55);
}

.card-title {
  font-size: 22px;
  font-weight: 700;
  letter-spacing: 1px;
}

.card-sub {
  margin: 8px 0 22px;
  font-size: 12.5px;
  line-height: 1.7;
  color: var(--ir-text-dim);
}

.form-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: -6px 0 16px;
}

.forget {
  font-size: 12px;
  color: var(--ir-text-dim);
  cursor: pointer;
}

.forget:hover {
  color: var(--ir-primary);
}

.demo-box {
  margin-top: 22px;
  padding: 14px;
  border-radius: 12px;
  border: 1px dashed rgba(120, 145, 220, 0.4);
  background: rgba(79, 140, 255, 0.06);
}

.demo-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--ir-text-dim);
  margin-bottom: 10px;
}

.demo-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.demo-chip {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: 9px;
  background: rgba(255, 255, 255, 0.04);
  cursor: pointer;
  transition: background 0.18s ease;
}

.demo-chip:hover {
  background: rgba(79, 140, 255, 0.18);
}

.demo-role {
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  color: #0b1020;
  background: linear-gradient(120deg, #36d1c4, #4f8cff);
}

.demo-account {
  font-family: 'SFMono-Regular', Menlo, Consolas, monospace;
  font-size: 12.5px;
}

.demo-note {
  margin-left: auto;
  font-size: 11px;
  color: var(--ir-text-dim);
}

.card-foot {
  display: flex;
  justify-content: space-between;
  margin-top: 20px;
  padding-top: 14px;
  border-top: 1px solid var(--ir-border);
  font-size: 11.5px;
  color: var(--ir-text-dim);
}

@media (max-width: 1080px) {
  .login-page {
    grid-template-columns: 1fr;
    padding: 32px 24px 48px;
    gap: 28px;
  }

  .system-name {
    font-size: 26px;
  }

  .emblem-img {
    width: 72px;
    height: 72px;
  }
}
</style>
