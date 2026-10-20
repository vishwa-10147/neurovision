/** Turn streamed run steps into chart series. Pure, so it is unit-tested. */
import type { RunPoint } from '../../types/runs'

/** Charts never draw more points than this; longer runs are evenly thinned. */
export const MAX_CHART_POINTS = 1000

export interface MetricGroup {
  title: string
  keys: string[]
  /** Accuracy-like metrics that stay within [0, 1] are shown as percentages. */
  percent: boolean
}

/** "loss" and "val_loss" share a chart; order follows first appearance. */
export function groupMetrics(points: RunPoint[]): MetricGroup[] {
  const groups = new Map<string, Set<string>>()
  for (const point of points) {
    for (const key of Object.keys(point.metrics)) {
      const base = key.replace(/^(val|valid|validation|test)[_.]/, '').replace(/[_.](val|valid|validation|test)$/, '')
      if (!groups.has(base)) groups.set(base, new Set())
      groups.get(base)?.add(key)
    }
  }
  return [...groups].map(([base, keys]) => {
    const values = points.flatMap((p) => [...keys].map((k) => p.metrics[k]).filter((v) => v !== undefined))
    return {
      title: base,
      keys: [...keys],
      percent: /acc/i.test(base) && values.every((v) => v >= 0 && v <= 1),
    }
  })
}

/** Points for Recharts, evenly thinned to at most `limit`, always keeping the last. */
export function chartRows(points: RunPoint[], limit = MAX_CHART_POINTS): RunPoint[] {
  if (points.length <= limit) return points
  const stride = Math.ceil(points.length / (limit - 1))
  return points.filter((_, i) => i % stride === 0 || i === points.length - 1)
}
