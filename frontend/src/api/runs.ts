import { RunListSchema, type RunInfo } from '../types/runs'
import { request } from './client'

export async function listRuns(signal?: AbortSignal): Promise<RunInfo[]> {
  return (await request('/runs', RunListSchema, { signal })).runs
}
