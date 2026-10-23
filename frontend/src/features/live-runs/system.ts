/** Formatting and chart rows for a run's resource use. Pure, so it is unit-tested. */
import type { SystemPoint } from '../../stores/liveRunStore'
import type { RunPoint } from '../../types/runs'

/** "45 s", "3 min 05 s", "2 h 07 min". */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '–'
  const s = Math.max(0, Math.round(seconds))
  if (s < 60) return `${s} s`
  if (s < 3600) return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, '0')} s`
  return `${Math.floor(s / 3600)} h ${String(Math.floor((s % 3600) / 60)).padStart(2, '0')} min`
}

/** Bytes in binary units: "512 MiB", "7.6 GiB". */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return '–'
  if (bytes >= 2 ** 30) return `${(bytes / 2 ** 30).toFixed(1)} GiB`
  if (bytes >= 2 ** 20) return `${Math.round(bytes / 2 ** 20)} MiB`
  return `${Math.round(bytes / 2 ** 10)} KiB`
}

/** Resource use as chartable series (one chart each). */
export function systemRow(point: SystemPoint): RunPoint {
  const metrics: Record<string, number> = {}
  if (point.cpu_percent !== undefined) metrics['CPU load %'] = point.cpu_percent
  if (point.gpu_util_percent !== undefined) metrics['GPU load %'] = point.gpu_util_percent
  if (point.gpu_memory_percent !== undefined) metrics['GPU memory %'] = point.gpu_memory_percent
  if (point.samples_per_s !== undefined) metrics['throughput (samples per s)'] = point.samples_per_s
  if (point.step_time_s !== undefined) metrics['step time (s)'] = point.step_time_s
  return { epoch: 0, step: point.step, metrics }
}
