import { z } from 'zod'
import type { DenseNetworkConfig } from '../types/architecture'
import { ModelGraphSchema } from '../types/graph'
import { request } from './client'

export const EducationalModelCreatedSchema = z.object({
  model_id: z.string(),
  graph: ModelGraphSchema,
})

export type EducationalModelCreated = z.infer<typeof EducationalModelCreatedSchema>

export function createEducationalModel(
  config: DenseNetworkConfig,
  signal?: AbortSignal,
): Promise<EducationalModelCreated> {
  return request('/models/educational', EducationalModelCreatedSchema, {
    method: 'POST',
    body: config,
    signal,
  })
}
