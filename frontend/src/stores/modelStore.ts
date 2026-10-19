/**
 * Shared model state: the editable configuration, the current normalized graph,
 * and the selection. Selection holds semantic ids only, so every renderer
 * (2D now, 3D later) and the inspector can map it to their own visuals.
 */
import { create } from 'zustand'
import { findNode, unitCount } from '../lib/graph'
import { EXAMPLE_CONFIG, type DenseNetworkConfig } from '../types/architecture'
import type { ModelGraph } from '../types/graph'
import type { ForwardResponse } from '../types/training'

export type Selection =
  | { kind: 'layer'; nodeId: string }
  | { kind: 'unit'; nodeId: string; index: number }
  /** A graph edge; indices name one unit-to-unit connection within it. */
  | { kind: 'connection'; edgeId: string; sourceIndex: number | null; targetIndex: number | null }
  | null

export type ModelStatus =
  | { state: 'idle' }
  | { state: 'loading' }
  | { state: 'ready' }
  | { state: 'error'; error: unknown }

export type ViewMode = '2d' | '3d'

/** A forward pass result, tagged with the weights it was computed from. */
export interface ForwardSnapshot {
  modelId: string
  inputs: number[]
  response: ForwardResponse
  weightsVersion: number
}

interface ModelState {
  viewMode: ViewMode
  setViewMode: (mode: ViewMode) => void
  config: DenseNetworkConfig
  graph: ModelGraph | null
  modelId: string | null
  /** The configuration the current model was built from. */
  modelConfig: DenseNetworkConfig | null
  status: ModelStatus
  selection: Selection
  /** Increases whenever the current model's weights change (e.g. after training). */
  weightsVersion: number
  forward: ForwardSnapshot | null

  updateConfig: (update: (config: DenseNetworkConfig) => DenseNetworkConfig) => void
  resetToExample: () => void
  setLoading: () => void
  setModel: (modelId: string, graph: ModelGraph, config?: DenseNetworkConfig) => void
  setError: (error: unknown) => void
  select: (selection: Selection) => void
  markWeightsChanged: (modelId: string) => void
  setForward: (snapshot: ForwardSnapshot | null) => void
}

export const useModelStore = create<ModelState>()((set) => ({
  viewMode: '2d',
  setViewMode: (viewMode) => set({ viewMode }),
  config: EXAMPLE_CONFIG,
  graph: null,
  modelId: null,
  modelConfig: null,
  status: { state: 'idle' },
  selection: null,
  weightsVersion: 0,
  forward: null,

  updateConfig: (update) => set((state) => ({ config: update(state.config) })),
  resetToExample: () => set({ config: EXAMPLE_CONFIG }),
  setLoading: () => set({ status: { state: 'loading' } }),
  setModel: (modelId, graph, config) =>
    set((state) => ({
      modelId,
      graph,
      modelConfig: config ?? null,
      status: { state: 'ready' },
      selection: isSelectionValid(state.selection, graph) ? state.selection : null,
      // A new model has new weights; old forward results describe another network.
      weightsVersion: state.weightsVersion + 1,
      forward: state.forward?.modelId === modelId ? state.forward : null,
    })),
  setError: (error) => set({ status: { state: 'error', error } }),
  select: (selection) => set({ selection }),
  markWeightsChanged: (modelId) =>
    set((state) => (state.modelId === modelId ? { weightsVersion: state.weightsVersion + 1 } : {})),
  setForward: (forward) => set({ forward }),
}))

/** The forward result for the current weights, or null if absent or stale. */
export function currentForward(state: Pick<ModelState, 'forward' | 'modelId' | 'weightsVersion'>) {
  const f = state.forward
  return f && f.modelId === state.modelId && f.weightsVersion === state.weightsVersion ? f : null
}

/** Whether a selection still refers to something in `graph`. */
export function isSelectionValid(selection: Selection, graph: ModelGraph): boolean {
  if (selection === null) return true
  if (selection.kind === 'layer') return findNode(graph, selection.nodeId) !== undefined
  if (selection.kind === 'unit') {
    const node = findNode(graph, selection.nodeId)
    const count = node ? unitCount(node) : null
    return count !== null && selection.index < count
  }
  const edge = graph.edges.find((e) => e.id === selection.edgeId)
  if (!edge) return false
  const inBounds = (nodeId: string, index: number | null) => {
    if (index === null) return true
    const node = findNode(graph, nodeId)
    const count = node ? unitCount(node) : null
    return count !== null && index < count
  }
  return inBounds(edge.source, selection.sourceIndex) && inBounds(edge.target, selection.targetIndex)
}
