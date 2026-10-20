/**
 * Model Explorer state: the imported model and its own selection and view mode,
 * kept apart from the Architecture Studio so neither overwrites the other.
 */
import { create } from 'zustand'
import type { ModelGraph } from '../types/graph'
import type { ImportReport, ImportResponse, RunResponse } from '../types/imports'
import type { ViewerTarget } from '../types/maps'
import { isSelectionValid, type Selection, type ViewMode } from './modelStore'

interface ExplorerState {
  modelId: string | null
  graph: ModelGraph | null
  report: ImportReport | null
  selection: Selection
  viewMode: ViewMode
  /** The latest run of the imported model, with the real values it produced. */
  run: RunResponse | null
  /** What the tensor viewer shows (maps, attention, or filters), if open. */
  viewer: ViewerTarget | null

  setImported: (result: ImportResponse) => void
  select: (selection: Selection) => void
  setViewMode: (mode: ViewMode) => void
  setRun: (run: RunResponse | null) => void
  openViewer: (target: ViewerTarget | null) => void
}

export const useExplorerStore = create<ExplorerState>()((set) => ({
  modelId: null,
  graph: null,
  report: null,
  selection: null,
  viewMode: '2d',
  run: null,
  viewer: null,

  setImported: (result) =>
    set({
      modelId: result.model_id,
      graph: result.graph,
      report: result.report,
      selection: null,
      run: null,
      viewer: null,
    }),
  openViewer: (viewer) => set({ viewer }),
  select: (selection) =>
    set((state) => (state.graph && !isSelectionValid(selection, state.graph) ? {} : { selection })),
  setViewMode: (viewMode) => set({ viewMode }),
  // A new run can change tensor shapes, so an open tensor view closes; filters stay.
  setRun: (run) =>
    set((state) =>
      run === null || run.model_id === state.modelId
        ? { run, viewer: state.viewer?.kind === 'filters' ? state.viewer : null }
        : {},
    ),
}))

/** Output tensors of the latest run, grouped by the node that produced them. */
export function runValuesByNode(run: RunResponse | null) {
  if (!run) return null
  const byNode: Record<string, RunResponse['outputs']> = {}
  for (const tensor of [...run.intermediates, ...run.outputs]) {
    if (tensor.node_id) (byNode[tensor.node_id] ??= []).push(tensor)
  }
  return byNode
}
