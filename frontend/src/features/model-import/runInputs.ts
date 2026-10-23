/** Pure helpers for the run panel: input drafts, validation, and top values. */
import { RUN_LIMITS, type Fill, type InputSpec, type RunInput, type TensorValues } from '../../types/imports'

export type Mode = Fill | 'custom' | 'image'

/** A photo already cropped and resized to the input's height × width (RGBA). */
export interface Photo {
  name: string
  width: number
  height: number
  rgba: Uint8ClampedArray
}

export type Normalization = 'imagenet' | 'unit'

export interface Draft {
  dims: string[]
  mode: Mode
  seed: string
  custom: string
  photo: Photo | null
  normalization: Normalization
}

/** torchvision's ImageNet statistics, used by most pretrained vision models. */
export const IMAGENET_MEAN = [0.485, 0.456, 0.406]
export const IMAGENET_STD = [0.229, 0.224, 0.225]

const TOP_K = 5
const MAX_CLASSES = 1000

/** Default draft: fixed dims as declared, named/unknown dims set to 1. */
export function initialDraft(spec: InputSpec): Draft {
  const dims = (spec.shape ?? [1]).map((d) => (typeof d === 'number' ? String(d) : '1'))
  return { dims, mode: 'random', seed: '0', custom: '', photo: null, normalization: 'imagenet' }
}

/**
 * The [height, width] a photo must be resized to for this input, or null when
 * the input is not image-like: [batch, 1 or 3 channels, height, width].
 */
export function photoSize(dims: string[]): [number, number] | null {
  if (dims.length !== 4) return null
  const [, channels, height, width] = dims.map(Number)
  if (channels !== 1 && channels !== 3) return null
  if (![height, width].every((d) => Number.isInteger(d) && d >= 1 && d <= 4096)) return null
  return [height, width]
}

/**
 * RGBA pixels to row-major [batch, channels, height, width] values in 0–1,
 * optionally ImageNet-normalized. Every batch item gets the same photo.
 */
export function photoValues(photo: Photo, shape: number[], normalization: Normalization): number[] {
  const [batch, channels, height, width] = shape
  const plane = height * width
  const one = new Array<number>(channels * plane)
  for (let i = 0; i < plane; i++) {
    const r = photo.rgba[4 * i] / 255
    const g = photo.rgba[4 * i + 1] / 255
    const b = photo.rgba[4 * i + 2] / 255
    const rgb = channels === 1 ? [0.299 * r + 0.587 * g + 0.114 * b] : [r, g, b]
    rgb.forEach((v, c) => {
      const mean = channels === 1 ? 0.449 : IMAGENET_MEAN[c]
      const std = channels === 1 ? 0.226 : IMAGENET_STD[c]
      one[c * plane + i] = normalization === 'imagenet' ? (v - mean) / std : v
    })
  }
  return Array.from({ length: batch }, () => one).flat()
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
  if (draft.mode === 'image') {
    const size = photoSize(draft.dims)
    if (!size) return { error: `'${spec.name}' is not an image input ([batch, 1 or 3, height, width]).` }
    if (!draft.photo) return { error: `Choose a photo for '${spec.name}'.` }
    if (draft.photo.height !== size[0] || draft.photo.width !== size[1]) {
      return { error: `Choose the photo for '${spec.name}' again: its size changed.` }
    }
    return { input: { shape, values: photoValues(draft.photo, shape, draft.normalization) } }
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
