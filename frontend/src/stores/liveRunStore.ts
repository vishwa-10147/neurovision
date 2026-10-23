/**
 * Live runs state: the external run being watched and the steps streamed from
 * it. Every value comes from the user's training script via the backend's
 * event stream; nothing is simulated. Commands are shown as the script
 * reports them (pending → applied/rejected/expired), never as assumed.
 */
import { create } from 'zustand'
import type { CommandKind, RunAlert, RunCommand, StepHealth, TrainingEvent } from '../types/events'
import type { RunInfo, RunPoint, RunStatus } from '../types/runs'
import type { StreamState } from './trainingStore'

export interface WatchedRun {
  runId: string
  name: string
  totalEpochs: number
  status: RunStatus
  /** Present when the run failed or was stopped. */
  failure: string | null
  /** Commands the script accepts. */
  controls: CommandKind[]
  paused: boolean
  learningRate: number | null
}

/** Global health numbers per step, for trend charts. */
export interface HealthPoint {
  step: number
  grad_norm?: number
  learning_rate?: number
}

/** Health trend charts keep at most this many points (evenly thinned). */
export const MAX_HEALTH_POINTS = 2000

interface LiveRunState {
  run: WatchedRun | null
  points: RunPoint[]
  /** Latest measured activations per layer, as sent by the script. */
  activations: Record<string, number[]>
  /** Latest step's per-layer health statistics. */
  health: { step: number; stats: StepHealth } | null
  healthPoints: HealthPoint[]
  /** Newest first. */
  alerts: RunAlert[]
  /** Latest known state of each command, by id. */
  commands: Record<number, RunCommand>
  stream: StreamState

  watch: (info: RunInfo) => void
  stopWatching: () => void
  applyEvents: (events: TrainingEvent[]) => void
  /** Record a command the dashboard just issued (before its event arrives). */
  recordCommand: (command: RunCommand) => void
  setStream: (stream: StreamState) => void
}

const EMPTY = {
  points: [] as RunPoint[],
  activations: {} as Record<string, number[]>,
  health: null,
  healthPoints: [] as HealthPoint[],
  alerts: [] as RunAlert[],
  commands: {} as Record<number, RunCommand>,
}

export const useLiveRunStore = create<LiveRunState>()((set, get) => ({
  run: null,
  ...EMPTY,
  stream: 'idle',

  watch: (info) => {
    if (get().run?.runId === info.run_id) return
    set({
      run: {
        runId: info.run_id,
        name: info.name,
        totalEpochs: info.total_epochs,
        // The stream replays from the start and brings the status up to date.
        status: 'running',
        failure: null,
        controls: info.controls,
        paused: false,
        learningRate: info.learning_rate,
      },
      ...EMPTY,
      stream: 'idle',
    })
  },
  stopWatching: () => set({ run: null, ...EMPTY, stream: 'idle' }),
  applyEvents: (events) => {
    const state = get()
    const { run } = state
    if (!run) return
    let nextRun = run
    let points = state.points
    let activations = state.activations
    let health = state.health
    let healthPoints = state.healthPoints
    let alerts = state.alerts
    let commands = state.commands
    for (const event of events) {
      if (event.job_id !== run.runId) continue
      switch (event.event_type) {
        case 'training.started':
          nextRun = { ...nextRun, totalEpochs: event.total_epochs }
          break
        case 'training.step': {
          // Steps strictly increase; replays after a reconnect repeat old ones.
          const last = points[points.length - 1]
          if (last && event.step <= last.step) break
          if (points === state.points) points = [...points]
          points.push({ epoch: event.epoch, step: event.step, metrics: event.metrics })
          if (Object.keys(event.selected_activations).length > 0) activations = event.selected_activations
          if (event.health) {
            const stats = event.health
            if (Object.keys(stats.parameters).length > 0 || Object.keys(stats.activations).length > 0) {
              health = { step: event.step, stats }
            }
            const point: HealthPoint = { step: event.step }
            if (stats.grad_norm !== null) point.grad_norm = stats.grad_norm
            if (stats.learning_rate !== null) {
              point.learning_rate = stats.learning_rate
              nextRun = { ...nextRun, learningRate: stats.learning_rate }
            }
            if (point.grad_norm !== undefined || point.learning_rate !== undefined) {
              if (healthPoints === state.healthPoints) healthPoints = [...healthPoints]
              healthPoints.push(point)
            }
          }
          break
        }
        case 'run.command': {
          const known = commands[event.command.command_id]
          // Replays repeat old updates; never move a resolved command back.
          if (known && known.status !== 'pending' && event.command.status === 'pending') break
          commands = { ...commands, [event.command.command_id]: event.command }
          nextRun = { ...nextRun, paused: event.paused, learningRate: event.learning_rate ?? nextRun.learningRate }
          break
        }
        case 'run.alert':
          if (alerts.some((a) => a.alert_id === event.alert.alert_id)) break
          alerts = [event.alert, ...alerts]
          break
        case 'training.completed':
          nextRun = { ...nextRun, status: 'completed', paused: false }
          break
        case 'training.cancelled':
          nextRun = { ...nextRun, status: 'cancelled', paused: false, failure: null }
          break
        case 'training.failed':
          nextRun = { ...nextRun, status: 'failed', paused: false, failure: event.message }
          break
      }
    }
    if (healthPoints.length > MAX_HEALTH_POINTS) healthPoints = thin(healthPoints, MAX_HEALTH_POINTS)
    set({ run: nextRun, points, activations, health, healthPoints, alerts, commands })
  },
  recordCommand: (command) => {
    const known = get().commands[command.command_id]
    if (known) return
    set({ commands: { ...get().commands, [command.command_id]: command } })
  },
  setStream: (stream) => set({ stream }),
}))

function thin<T>(rows: T[], limit: number): T[] {
  const stride = Math.ceil(rows.length / (limit - 1))
  return rows.filter((_, i) => i % stride === 0 || i === rows.length - 1)
}
