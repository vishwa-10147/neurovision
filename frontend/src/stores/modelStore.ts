/**
 * Shared model state: the editable configuration, the current normalized graph,
 * and the selection. Selection holds semantic ids only, so every renderer
 * (2D now, 3D later) and the inspector can map it to their own visuals.
 */
import { create } from 'zustand'
import { findNode, unitCount } from '../lib/graph'
import { EXAMPLE_CONFIG, type DenseNetworkConfig } from '../types/architecture'
import type { ModelGraph } from '../types/graph'

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

interface ModelState {
  viewMode: ViewMode
  setViewMode: (mode: ViewMode) => void
  config: DenseNetworkConfig
  graph: ModelGraph | null
  modelId: string | null
  status: ModelStatus
  selection: Selection

  updateConfig: (update: (config: DenseNetworkConfig) => DenseNetworkConfig) => void
  resetToExample: () => void
  setLoading: () => void
  setModel: (modelId: string, graph: ModelGraph) => void
  setError: (error: unknown) => void
  select: (selection: Selection) => void
}

export const useModelStore = create<ModelState>()((set) => ({
  viewMode: '2d',
  setViewMode: (viewMode) => set({ viewMode }),
  config: EXAMPLE_CONFIG,
  graph: null,
  modelId: null,
  status: { state: 'idle' },
  selection: null,

  updateConfig: (update) => set((state) => ({ config: update(state.config) })),
  resetToExample: () => set({ config: EXAMPLE_CONFIG }),
  setLoading: () => set({ status: { state: 'loading' } }),
  setModel: (modelId, graph) =>
    set((state) => ({
      modelId,
      graph,
      status: { state: 'ready' },
      selection: isSelectionValid(state.selection, graph) ? state.selection : null,
    })),
  setError: (error) => set({ status: { state: 'error', error } }),
  select: (selection) => set({ selection }),
}))

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
