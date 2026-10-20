/** Pure helpers for the run panel: input drafts, validation, and top values. */
import { RUN_LIMITS, type Fill, type InputSpec, type RunInput, type TensorValues } from '../../types/imports'

export type Mode = Fill | 'custom'

export interface Draft {
  dims: string[]
  mode: Mode
  seed: string
  custom: string
}

const TOP_K = 5
const MAX_CLASSES = 1000

/** Default draft: fixed dims as declared, named/unknown dims set to 1. */
export function initialDraft(spec: InputSpec): Draft {
  const dims = (spec.shape ?? [1]).map((d) => (typeof d === 'number' ? String(d) : '1'))
  return { dims, mode: 'random', seed: '0', custom: '' }
}

/** Validate one input draft into a request body, or explain what is wrong. */
export function toRunInput(spec: InputSpec, draft: Draft): { input: RunInput } | { error: string } {
  const shape = draft.dims.map((d) => Number(d))
  if (shape.some((d) => !Number.isInteger(d) || d < 1 || d > RUN_LIMITS.maxDimension)) {
    return { error: `Each dimension of '${spec.name}' must be a whole number from 1 to ${RUN_LIMITS.maxDimension.toLocaleString()}.` }
  }
  const count = shape.reduce((a, b) => a * b, 1)
  if (count > RUN_LIMITS.maxElements) {
    return { error: `'${spec.name}' would have ${count.toLocaleString()} values; the limit is ${RUN_LIMITS.maxElements.toLocaleString()}.` }
  }
  if (draft.mode !== 'custom') {
    const seed = Number(draft.seed)
    if (draft.mode === 'random' && (!Number.isInteger(seed) || seed < 0)) return { error: 'The seed must be a whole number ≥ 0.' }
    return { input: { shape, fill: draft.mode, seed: draft.mode === 'random' ? seed : 0 } }
  }
  const values = draft.custom
    .split(/[\s,]+/)
    .filter(Boolean)
    .map(Number)
  if (values.some((v) => !Number.isFinite(v))) return { error: `Values for '${spec.name}' must be numbers.` }
  if (values.length !== count) {
    return {
      error: `'${spec.name}' with shape [${shape.join(' × ')}] needs ${count.toLocaleString()} values; you entered ${values.length.toLocaleString()}.`,
    }
  }
  return { input: { shape, values } }
}

/** For [batch, classes] outputs, the largest values of the first row. */
export function topValues(tensor: TensorValues): [number, number][] | null {
  if (tensor.shape.length !== 2) return null
  const classes = tensor.shape[1]
  if (classes < 2 || classes > MAX_CLASSES || tensor.values.length < classes) return null
  return tensor.values
    .slice(0, classes)
    .map((v, i) => [i, v] as [number, number | null])
    .filter((entry): entry is [number, number] => entry[1] !== null)
    .sort((a, b) => b[1] - a[1])
    .slice(0, TOP_K)
}
