/** Zod mirror of backend/app/schemas/classical.py and the fit request. */
import { z } from 'zod'
import { request } from './client'

export const ALGORITHMS = [
  'logistic_regression',
  'linear_svm',
  'rbf_svm',
  'knn',
  'decision_tree',
  'random_forest',
  'gradient_boosting',
  'kmeans',
  'pca',
] as const
export type Algorithm = (typeof ALGORITHMS)[number]
export const CLASSIC_DATASETS = ['moons', 'circles', 'spiral', 'blobs', 'digits'] as const
export type ClassicDataset = (typeof CLASSIC_DATASETS)[number]

export interface ClassicRequest {
  algorithm: Algorithm
  dataset: ClassicDataset
  samples: number
  noise: number
  seed: number
  c: number
  gamma: number
  neighbors: number
  max_depth: number
  estimators: number
  learning_rate: number
  clusters: number
  epochs: number
}

const PointsSchema = z.object({ x: z.array(z.number()), y: z.array(z.number()), labels: z.array(z.number().int()) })
const Pair = z.tuple([z.number(), z.number()])

export const FrameSchema = z.object({
  label: z.string(),
  grid_class: z.array(z.number().int()),
  grid_confidence: z.array(z.number()),
  train_accuracy: z.number().nullable().default(null),
  test_accuracy: z.number().nullable().default(null),
  centroids: z.array(Pair).nullable().default(null),
  assignments: z.array(z.number().int()).nullable().default(null),
  inertia: z.number().nullable().default(null),
})

export const TreeNodeSchema = z.object({
  id: z.number().int(),
  depth: z.number().int(),
  feature: z.number().int().nullable(),
  threshold: z.number().nullable(),
  left: z.number().int().nullable(),
  right: z.number().int().nullable(),
  samples: z.number().int(),
  counts: z.array(z.number()),
  impurity: z.number(),
})

export const ClassicResultSchema = z.object({
  algorithm: z.enum(ALGORITHMS),
  dataset: z.enum(CLASSIC_DATASETS),
  classes: z.number().int(),
  features: z.number().int(),
  train: PointsSchema,
  test: PointsSchema,
  bounds: z.tuple([z.number(), z.number(), z.number(), z.number()]).nullable(),
  grid: z.number().int(),
  frames: z.array(FrameSchema),
  fit_ms: z.number(),
  support_vectors: z.array(Pair).nullable().default(null),
  tree: z.array(TreeNodeSchema).nullable().default(null),
  tree_note: z.string().nullable().default(null),
  feature_importance: z.array(z.number()).nullable().default(null),
  pca: z
    .object({
      explained_variance_ratio: z.array(z.number()),
      projected: PointsSchema,
      mean: Pair.nullable().default(null),
      axes: z.array(Pair).nullable().default(null),
    })
    .nullable()
    .default(null),
  notes: z.array(z.string()).default([]),
})

export type ClassicResult = z.infer<typeof ClassicResultSchema>
export type Frame = z.infer<typeof FrameSchema>
export type TreeNode = z.infer<typeof TreeNodeSchema>

export function fitClassical(body: ClassicRequest, signal?: AbortSignal): Promise<ClassicResult> {
  return request('/classical/fit', ClassicResultSchema, { method: 'POST', body, signal, timeoutMs: 60_000 })
}
