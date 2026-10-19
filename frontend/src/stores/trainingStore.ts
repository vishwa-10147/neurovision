/**
 * Training Lab state: the draft training settings, the current job, and the
 * metrics and activations streamed from it. Every value shown comes from the
 * backend's training events; nothing is simulated.
 */
import { create } from 'zustand'
import { cancelTraining } from '../api/training'
import type { TrainingEvent } from '../types/events'
import {
  DEFAULT_TRAINING_CONFIG,
  type EpochMetrics,
  type JobStatus,
  type TrainingConfig,
  type TrainingJobInfo,
} from '../types/training'
import { useModelStore } from './modelStore'

export type StreamState = 'idle' | 'connecting' | 'open' | 'closed' | 'error'

export interface TrainingJob {
  jobId: string
  modelId: string
  totalEpochs: number
  status: JobStatus
  /** Present when the job failed. */
  failure: string | null
}

interface TrainingState {
  config: TrainingConfig
  /** Layers to stream activations for; null means every learned layer. */
  captureLayers: string[] | null
  job: TrainingJob | null
  history: EpochMetrics[]
  /** Latest measured activations per layer for the probe example. */
  activations: Record<string, number[]>
  stream: StreamState

  updateConfig: (changes: Partial<TrainingConfig>) => void
  setCaptureLayers: (layers: string[] | null) => void
  beginJob: (info: TrainingJobInfo, history?: EpochMetrics[]) => void
  applyEvents: (events: TrainingEvent[]) => void
  setStream: (stream: StreamState) => void
}

export const useTrainingStore = create<TrainingState>()((set, get) => ({
  config: DEFAULT_TRAINING_CONFIG,
  captureLayers: null,
  job: null,
  history: [],
  activations: {},
  stream: 'idle',

  updateConfig: (changes) => set((s) => ({ config: { ...s.config, ...changes } })),
  setCaptureLayers: (captureLayers) => set({ captureLayers }),
  beginJob: (info, history = []) =>
    set({
      job: {
        jobId: info.job_id,
        modelId: info.model_id,
        totalEpochs: info.total_epochs,
        status: info.status,
        failure: null,
      },
      history,
      activations: {},
      stream: 'idle',
    }),
  applyEvents: (events) => {
    const { job, history, activations } = get()
    if (!job) return
    let nextJob = job
    let nextHistory = history
    let nextActivations = activations
    for (const event of events) {
      if (event.job_id !== job.jobId) continue
      switch (event.event_type) {
        case 'training.started':
          nextJob = { ...nextJob, status: 'running', totalEpochs: event.total_epochs }
          break
        case 'training.step': {
          // Replays after a reconnect repeat earlier epochs; keep each once.
          const last = nextHistory[nextHistory.length - 1]
          if (last && event.epoch <= last.epoch) break
          const m = event.metrics
          nextHistory = [
            ...nextHistory,
            {
              epoch: event.epoch,
              loss: m.loss,
              accuracy: m.accuracy,
              val_loss: m.val_loss,
              val_accuracy: m.val_accuracy,
            },
          ]
          if (Object.keys(event.selected_activations).length > 0) nextActivations = event.selected_activations
          break
        }
        case 'training.completed':
          nextJob = { ...nextJob, status: 'completed' }
          break
        case 'training.cancelled':
          nextJob = { ...nextJob, status: 'cancelled' }
          break
        case 'training.failed':
          nextJob = { ...nextJob, status: 'failed', failure: event.message }
          break
      }
    }
    set({ job: nextJob, history: nextHistory, activations: nextActivations })
    // Weights stop changing once a job ends; let views refetch them.
    if (job.status === 'running' && nextJob.status !== 'running') {
      useModelStore.getState().markWeightsChanged(job.modelId)
    }
  },
  setStream: (stream) => set({ stream }),
}))

// A new architecture means a new model: stop training the one it replaced.
useModelStore.subscribe((state, previous) => {
  if (state.modelId === previous.modelId) return
  const { job } = useTrainingStore.getState()
  if (job && job.status === 'running' && job.modelId !== state.modelId) {
    cancelTraining(job.jobId).catch(() => {
      // Best effort: the server also cancels jobs whose model is evicted.
    })
  }
})
