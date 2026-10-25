import { z } from 'zod'
import { request } from './client'
import { DenseNetworkConfigSchema, type DenseNetworkConfig } from '../types/architecture'

const ProjectSummarySchema = z.object({
  project_id: z.string(),
  name: z.string(),
  description: z.string(),
  schema_version: z.number().int(),
  created_at: z.string(),
  updated_at: z.string(),
})

export const ProjectDocumentSchema = ProjectSummarySchema.extend({
  config: DenseNetworkConfigSchema,
})

export type ProjectSummary = z.infer<typeof ProjectSummarySchema>
export type ProjectDocument = z.infer<typeof ProjectDocumentSchema>

const ProjectListSchema = z.object({ projects: z.array(ProjectSummarySchema) })

export function listProjects(signal?: AbortSignal): Promise<{ projects: ProjectSummary[] }> {
  return request('/projects', ProjectListSchema, { signal })
}

export function saveProject(
  config: DenseNetworkConfig,
  description = '',
  signal?: AbortSignal,
): Promise<ProjectDocument> {
  return request('/projects', ProjectDocumentSchema, {
    method: 'POST',
    body: { name: config.name, description, config },
    signal,
  })
}

export function loadProject(projectId: string, signal?: AbortSignal): Promise<ProjectDocument> {
  return request(`/projects/${encodeURIComponent(projectId)}`, ProjectDocumentSchema, { signal })
}
