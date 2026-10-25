import type { CommandKind, RunCommand } from '../types/events'
import { RunCommandSchema, RunListSchema, type RunInfo } from '../types/runs'
import { API_BASE_URL, request } from './client'

/** Every logged step's metrics as a CSV download. */
export function runMetricsCsvUrl(runId: string): string {
  return `${API_BASE_URL}/api/v1/runs/${encodeURIComponent(runId)}/metrics.csv`
}

export async function listRuns(signal?: AbortSignal): Promise<RunInfo[]> {
  return (await request('/runs', RunListSchema, { signal })).runs
}

/**
 * Ask a run's script to pause, resume, stop, checkpoint, or change its
 * learning rate. The script applies it at its next run.poll(); the outcome
 * arrives as a run.command event.
 */
export function sendCommand(runId: string, kind: CommandKind, value?: number): Promise<RunCommand> {
  return request(`/runs/${encodeURIComponent(runId)}/commands`, RunCommandSchema, {
    method: 'POST',
    body: value === undefined ? { kind } : { kind, value },
  })
}
