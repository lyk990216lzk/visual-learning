export function percent(value, digits = 2) {
  if (value === null || value === undefined) return '--'
  return `${(Number(value) * 100).toFixed(digits)}%`
}

export function fixed(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '--'
  return Number(value).toFixed(digits)
}

export function bytes(size) {
  if (!size) return '--'
  const units = ['B', 'KB', 'MB', 'GB']
  let index = 0
  let value = Number(size)
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024
    index += 1
  }
  return `${value.toFixed(1)} ${units[index]}`
}

export function params(million) {
  if (!million && million !== 0) return '--'
  if (million >= 1) return `${Number(million).toFixed(2)} M`
  return `${(Number(million) * 1000).toFixed(0)} K`
}

export function dateTime(value) {
  if (!value) return '--'
  return String(value).replace('T', ' ').slice(0, 19)
}

export const LABEL_TEXT = {
  circle: '圆形',
  square: '正方形',
  triangle: '三角形',
  cross: '十字',
  star: '五角星',
  ring: '圆环'
}

export function labelText(label) {
  return LABEL_TEXT[label] || label
}

export const STATUS_META = {
  pending: { text: '排队中', type: 'info' },
  running: { text: '训练中', type: 'warning' },
  success: { text: '已完成', type: 'success' },
  failed: { text: '失败', type: 'danger' },
  canceled: { text: '已中止', type: 'info' }
}
