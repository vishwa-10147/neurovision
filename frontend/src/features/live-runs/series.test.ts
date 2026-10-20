import { describe, expect, it } from 'vitest'
import type { RunPoint } from '../../types/runs'
import { chartRows, groupMetrics } from './series'

const point = (step: number, metrics: Record<string, number>): RunPoint => ({ epoch: step, step, metrics })

describe('run chart series', () => {
  it('pairs training and validation metrics and detects percentages', () => {
    const groups = groupMetrics([
      point(1, { loss: 2, val_loss: 2.5, accuracy: 0.4, 'acc.val': 0.3, lr: 0.01 }),
      point(2, { loss: 1, val_loss: 1.5, accuracy: 0.6, lr: 0.01, top5_acc: 7 }),
    ])
    expect(groups).toEqual([
      { title: 'loss', keys: ['loss', 'val_loss'], percent: false },
      { title: 'accuracy', keys: ['accuracy'], percent: true },
      { title: 'acc', keys: ['acc.val'], percent: true },
      { title: 'lr', keys: ['lr'], percent: false },
      // Out of [0, 1], so not shown as a percentage even though it says acc.
      { title: 'top5_acc', keys: ['top5_acc'], percent: false },
    ])
  })

  it('thins long runs evenly and keeps the latest point', () => {
    const points = Array.from({ length: 5000 }, (_, i) => point(i, { loss: i }))
    const rows = chartRows(points, 1000)
    expect(rows.length).toBeLessThanOrEqual(1000)
    expect(rows[0].step).toBe(0)
    expect(rows[rows.length - 1].step).toBe(4999)
    expect(chartRows(points.slice(0, 10), 1000)).toHaveLength(10)
  })
})
