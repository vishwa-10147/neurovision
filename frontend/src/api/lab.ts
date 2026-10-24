/** Zod mirror of backend/app/schemas/lab.py. */
import { z } from 'zod'
import { getJson, request } from './client'

export const ParamSpecSchema = z.object({
  name: z.string(),
  label: z.string(),
  kind: z.enum(['int', 'float', 'choice']),
  default: z.union([z.number(), z.string()]),
  min: z.number().nullable().default(null),
  max: z.number().nullable().default(null),
  choices: z.array(z.string()).nullable().default(null),
  help: z.string().nullable().default(null),
})

export const TemplateSchema = z.object({
  id: z.string(),
  label: z.string(),
  family: z.string(),
  description: z.string(),
  shows: z.array(z.string()),
  params: z.array(ParamSpecSchema),
})

export const TemplateListSchema = z.object({
  templates: z.array(TemplateSchema),
  enabled: z.boolean(),
  python: z.string().nullable(),
})

export const LabJobSchema = z.object({
  job_id: z.string(),
  template: z.string(),
  run_name: z.string(),
  status: z.enum(['running', 'finished', 'failed', 'stopped']),
  exit_code: z.number().int().nullable(),
  started_at: z.number(),
  output: z.array(z.string()),
})

export type ParamSpec = z.infer<typeof ParamSpecSchema>
export type Template = z.infer<typeof TemplateSchema>
export type LabJob = z.infer<typeof LabJobSchema>

export function listTemplates(signal?: AbortSignal) {
  return getJson('/lab/templates', TemplateListSchema, signal)
}

export function launchLab(template: string, params: Record<string, number | string>, signal?: AbortSignal) {
  return request('/lab/runs', LabJobSchema, { method: 'POST', body: { template, params }, signal })
}

export function listLabJobs(signal?: AbortSignal) {
  return getJson('/lab/runs', z.object({ jobs: z.array(LabJobSchema) }), signal)
}

export function stopLabJob(jobId: string) {
  return request(`/lab/runs/${encodeURIComponent(jobId)}/stop`, LabJobSchema, { method: 'POST' })
}
