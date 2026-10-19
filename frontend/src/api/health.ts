import { z } from 'zod'
import { getJson } from './client'

export const HealthResponseSchema = z.object({
  status: z.literal('ok'),
  service: z.string(),
  version: z.string(),
  environment: z.string(),
})

export type HealthResponse = z.infer<typeof HealthResponseSchema>

export function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return getJson('/health', HealthResponseSchema, signal)
}
