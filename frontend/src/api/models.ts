import { z } from 'zod'
import type { DenseNetworkConfig } from '../types/architecture'
import { ModelGraphSchema, type ModelGraph } from '../types/graph'
import { getJson, request } from './client'

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

/** The graph of any stored model, e.g. one an external run is linked to. */
export function getModelGraph(modelId: string, signal?: AbortSignal): Promise<ModelGraph> {
  return getJson(`/models/${encodeURIComponent(modelId)}/graph`, ModelGraphSchema, signal)
}
