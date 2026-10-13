/**
 * 内置样例图工具。
 *
 * 样例接口 `/api/recognize/sample/` 和其它业务接口一样需要 Token 鉴权，
 * 而 `<img src="...">` 这类浏览器原生请求无法附加 `Authorization` 请求头，
 * 直接写 <img src> 只会拿到 401，图片自然显示不出来。
 * 这里统一改为用 axios 带上令牌取回图片二进制，再转换成 blob URL 供 <img> 使用，
 * 并对结果做缓存，避免每次渲染都重复请求后端（样例图是后端实时绘制的）。
 */

import api from '../api'

/** 演示数据集内置的 6 个类别，与后端 ml/datasets.py 的 DEMO_CLASSES 保持一致。 */
export const SAMPLE_LABELS = ['circle', 'square', 'triangle', 'cross', 'star', 'ring']

const objectUrlCache = new Map()
const blobCache = new Map()

function cacheKey(label, size) {
  return `${label}@${size}`
}

/** 取回样例图二进制（带缓存），失败时不写入缓存，便于下次重试。 */
export function sampleImageBlob(label, size = 64) {
  const key = cacheKey(label, size)
  if (!blobCache.has(key)) {
    const pending = api
      .sampleImageBlob(label, size, { silent: true })
      .catch((error) => {
        blobCache.delete(key)
        throw error
      })
    blobCache.set(key, pending)
  }
  return blobCache.get(key)
}

/** 取回样例图的 blob URL（供 <img :src> 使用）。 */
export function sampleObjectUrl(label, size = 64) {
  const key = cacheKey(label, size)
  if (!objectUrlCache.has(key)) {
    const pending = sampleImageBlob(label, size)
      .then((blob) => URL.createObjectURL(blob))
      .catch((error) => {
        objectUrlCache.delete(key)
        throw error
      })
    objectUrlCache.set(key, pending)
  }
  return objectUrlCache.get(key)
}

/** 取回可直接上传给识别接口的样例文件。 */
export async function sampleFile(label, size = 64) {
  const blob = await sampleImageBlob(label, size)
  return new File([blob], `sample_${label}.png`, { type: 'image/png' })
}
