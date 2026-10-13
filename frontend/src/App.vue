<script setup>
import { onMounted } from 'vue'
import { useAuthStore } from './store/auth'

const auth = useAuthStore()

onMounted(() => {
  document.documentElement.classList.add('dark')
  auth.loadConfig()
})
</script>

<template>
  <router-view />
</template>
