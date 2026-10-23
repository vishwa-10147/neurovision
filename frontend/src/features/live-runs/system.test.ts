import { describe, expect, it } from 'vitest'
import { formatBytes, formatDuration, systemRow } from './system'

describe('formatDuration', () => {
  it('uses seconds, minutes or hours', () => {
    expect(formatDuration(45.4)).toBe('45 s')
    expect(formatDuration(185)).toBe('3 min 05 s')
    expect(formatDuration(7620)).toBe('2 h 07 min')
  })
  it('shows a dash for missing values', () => {
    expect(formatDuration(null)).toBe('–')
    expect(formatDuration(Number.NaN)).toBe('–')
  })
})

describe('formatBytes', () => {
  it('uses binary units', () => {
    expect(formatBytes(512 * 2 ** 20)).toBe('512 MiB')
    expect(formatBytes(7.6 * 2 ** 30)).toBe('7.6 GiB')
    expect(formatBytes(undefined)).toBe('–')
  })
})

describe('systemRow', () => {
  it('names each measured series and skips missing ones', () => {
    expect(systemRow({ step: 3, cpu_percent: 40, gpu_util_percent: 95 })).toEqual({
      epoch: 0,
      step: 3,
      metrics: { 'CPU load %': 40, 'GPU load %': 95 },
    })
  })
})
