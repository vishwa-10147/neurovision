/**
 * Resource use and timing of an external run, as measured by the training
 * script itself: elapsed time, ETA, step time, throughput, CPU, RAM and GPU.
 */
import type { StepSystem } from '../../types/events'
import { formatBytes, formatDuration } from './system'

function pct(value: number | null | undefined): string {
  return value === null || value === undefined ? '–' : `${Math.round(value)}%`
}

function rate(value: number | null | undefined): string {
  if (value === null || value === undefined) return '–'
  return value >= 100 ? Math.round(value).toLocaleString() : value.toFixed(1)
}

export function RunSystem({ system }: { system: { step: number; stats: StepSystem } | null }) {
  if (!system) {
    return (
      <section className="card run-system" aria-label="System and speed">
        <h3>System &amp; speed</h3>
        <p className="muted">
          No resource figures yet. The SDK sends timing with every step, and CPU, RAM and GPU use when{' '}
          <code>psutil</code> and <code>nvidia-ml-py</code> are installed.
        </p>
      </section>
    )
  }
  const s = system.stats
  const tiles: [string, string][] = [
    ['Elapsed', formatDuration(s.elapsed_s)],
    ['Time left (ETA)', formatDuration(s.eta_s)],
    ['Step time', s.step_time_s === null ? '–' : `${s.step_time_s.toFixed(s.step_time_s < 1 ? 3 : 1)} s`],
    ['Samples per second', rate(s.samples_per_s)],
    ['CPU load', pct(s.cpu_percent)],
    ['RAM', s.ram_total_bytes ? `${formatBytes(s.ram_used_bytes)} of ${formatBytes(s.ram_total_bytes)}` : '–'],
    ['Script memory', formatBytes(s.process_rss_bytes)],
  ]
  if (s.cuda_allocated_bytes !== null) {
    tiles.push([
      'PyTorch CUDA memory',
      `${formatBytes(s.cuda_allocated_bytes)} (reserved ${formatBytes(s.cuda_reserved_bytes)})`,
    ])
  }
  return (
    <section className="card run-system" aria-label="System and speed">
      <h3>System &amp; speed</h3>
      <p className="muted">
        Measured by the training script at step {system.step.toLocaleString()}
        {s.device && ` on ${s.device}`}.
      </p>
      <dl className="metric-tiles" aria-label="Resource use">
        {tiles.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      {s.gpus.map((gpu) => (
        <div key={gpu.index} className="gpu-row">
          <h4>
            GPU {gpu.index}
            {gpu.name && ` · ${gpu.name}`}
          </h4>
          <dl className="metric-tiles" aria-label={`GPU ${gpu.index}`}>
            <div>
              <dt>GPU load</dt>
              <dd>{pct(gpu.util_percent)}</dd>
            </div>
            <div>
              <dt>GPU memory</dt>
              <dd>
                {gpu.mem_total_bytes
                  ? `${formatBytes(gpu.mem_used_bytes)} of ${formatBytes(gpu.mem_total_bytes)}`
                  : '–'}
              </dd>
            </div>
            <div>
              <dt>Temperature</dt>
              <dd>{gpu.temperature_c === null ? '–' : `${Math.round(gpu.temperature_c)} °C`}</dd>
            </div>
            <div>
              <dt>Power</dt>
              <dd>{gpu.power_w === null ? '–' : `${Math.round(gpu.power_w)} W`}</dd>
            </div>
          </dl>
        </div>
      ))}
    </section>
  )
}
