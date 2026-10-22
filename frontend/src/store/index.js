import { defineStore } from 'pinia'
import api from '../api'

export const useAppStore = defineStore('app', {
  state: () => ({
    architectures: [],
    datasets: [],
    trainedModels: [],
    system: null,
    trainingDevice: '',
    loading: false
  }),
  getters: {
    readyArchitectures: (state) => state.architectures.filter((item) => item.ready),
    architectureMap: (state) =>
      state.architectures.reduce((acc, item) => {
        acc[item.key] = item
        return acc
      }, {}),
    activeModels: (state) => state.trainedModels.filter((item) => item.is_active)
  },
  actions: {
    setTrainingDevice(device) {
      this.trainingDevice = (device || '').toLowerCase()
    },
    async fetchArchitectures() {
      const data = await api.architectures()
      this.architectures = data.results || []
      return this.architectures
    },
    async fetchDatasets() {
      const data = await api.datasets()
      this.datasets = data.results || data || []
      return this.datasets
    },
    async fetchTrainedModels(params) {
      const data = await api.trainedModels({ page_size: 100, ...params })
      this.trainedModels = data.results || data || []
      return this.trainedModels
    },
    async fetchSystem() {
      this.system = await api.system()
      return this.system
    },
    async bootstrap() {
      if (this.loading) return
      this.loading = true
      try {
        await Promise.all([
          this.fetchArchitectures(),
          this.fetchDatasets(),
          this.fetchTrainedModels(),
          this.fetchSystem()
        ])
      } finally {
        this.loading = false
      }
    }
  }
})
