import { beforeEach, describe, expect, it, vi } from 'vitest'
import { stepEvent } from '../test/fakeBackend'
import type { TrainingEvent } from '../types/events'
import { DEFAULT_TRAINING_CONFIG } from '../types/training'
import { useModelStore } from './modelStore'
import { useTrainingStore } from './trainingStore'

const initialTraining = useTrainingStore.getState()
const initialModel = useModelStore.getState()
const job = {
  job_id: 'job_1',
  model_id: 'edu_1',
  status: 'running' as const,
  epoch: 0,
  total_epochs: 3,
  config: DEFAULT_TRAINING_CONFIG,
}
const ev = (e: object) => e as TrainingEvent

beforeEach(() => {
  useTrainingStore.setState(initialTraining, true)
  useModelStore.setState({ ...initialModel, modelId: 'edu_1', weightsVersion: 5 }, true)
  vi.unstubAllGlobals()
})

describe('applyEvents', () => {
  it('records each epoch once, even when a reconnect replays history', () => {
    const store = useTrainingStore.getState()
    store.beginJob(job)
    store.applyEvents([ev(stepEvent('job_1', 1)), ev(stepEvent('job_1', 2))])
    // Replay after reconnecting: epochs 1-2 again, then 3.
    store.applyEvents([ev(stepEvent('job_1', 1)), ev(stepEvent('job_1', 2)), ev(stepEvent('job_1', 3))])
    expect(useTrainingStore.getState().history.map((m) => m.epoch)).toEqual([1, 2, 3])
  })

  it('keeps the latest streamed activations', () => {
    const store = useTrainingStore.getState()
    store.beginJob(job)
    store.applyEvents([ev(stepEvent('job_1', 1, { selected_activations: { hidden_1: [0.1, 0.2] } }))])
    expect(useTrainingStore.getState().activations).toEqual({ hidden_1: [0.1, 0.2] })
  })

  it('ignores events for other jobs', () => {
    const store = useTrainingStore.getState()
    store.beginJob(job)
    store.applyEvents([ev(stepEvent('job_other', 1))])
    expect(useTrainingStore.getState().history).toEqual([])
  })

  it('marks weights as changed when a job ends', () => {
    const store = useTrainingStore.getState()
    store.beginJob(job)
    store.applyEvents([ev(stepEvent('job_1', 1))])
    expect(useModelStore.getState().weightsVersion).toBe(5)
    store.applyEvents([ev({ schema_version: '1.0', event_type: 'training.cancelled', job_id: 'job_1', step: 10 })])
    expect(useTrainingStore.getState().job?.status).toBe('cancelled')
    expect(useModelStore.getState().weightsVersion).toBe(6)
  })

  it('keeps the failure message', () => {
    const store = useTrainingStore.getState()
    store.beginJob(job)
    store.applyEvents([
      ev({ schema_version: '1.0', event_type: 'training.failed', job_id: 'job_1', message: 'Lower the learning rate.' }),
    ])
    expect(useTrainingStore.getState().job).toMatchObject({ status: 'failed', failure: 'Lower the learning rate.' })
  })
})

describe('model changes', () => {
  it('cancels a running job when the architecture produces a new model', () => {
    const fetchMock = vi.fn<typeof fetch>(async () => Response.json(job))
    vi.stubGlobal('fetch', fetchMock)
    useTrainingStore.getState().beginJob(job)
    useModelStore.setState({ modelId: 'edu_2' })
    expect(fetchMock).toHaveBeenCalledOnce()
    expect(String(fetchMock.mock.calls[0][0])).toMatch(/\/training\/job_1\/cancel$/)
  })
})
