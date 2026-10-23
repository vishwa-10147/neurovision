/** Pure helpers that describe tensors for the map viewer. */
import type { AttentionCandidate } from '../../types/imports'
import type { ViewerTarget } from '../../types/maps'

/**
 * Names for each dimension. Attention candidates use their detected layout;
 * otherwise the common convention for the rank (NCHW for 4D; NCTHW for 5D,
 * as in PyTorch's Conv3d video models) is assumed and
 * shown as an assumption in the viewer.
 */
export function layoutFor(shape: number[], attention?: AttentionCandidate): string[] {
  if (attention && attention.layout.length === shape.length) return attention.layout
  if (shape.length === 5) return ['batch', 'channels', 'frames', 'height', 'width']
  if (shape.length === 4) return ['batch', 'channels', 'height', 'width']
  if (shape.length === 3) return ['batch', 'rows', 'columns']
  if (shape.length === 2) return ['rows', 'columns']
  return [...shape.slice(0, -2).map((_, i) => `dim ${i}`), 'rows', 'columns']
}

const SINGULAR: Record<string, string> = { channels: 'channel', heads: 'head', batch: 'item', frames: 'frame' }

/** "channel" for "channels", "head" for "heads"; other names unchanged. */
export const singular = (axis: string) => SINGULAR[axis] ?? axis

/** "[1 batch × 8 channels × 28 height × 28 width]" */
export function describeShape(shape: number[], layout: string[]): string {
  return `[${shape.map((size, i) => `${size} ${layout[i] ?? ''}`.trim()).join(' × ')}]`
}

export function tensorTarget(
  nodeId: string,
  output: number,
  shape: number[],
  label: string,
  attention?: AttentionCandidate,
): ViewerTarget {
  return { kind: 'tensor', nodeId, output, shape, layout: layoutFor(shape, attention), attention: Boolean(attention), label }
}
