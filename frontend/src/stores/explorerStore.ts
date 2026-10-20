/**
 * Model Explorer state: the imported model and its own selection and view mode,
 * kept apart from the Architecture Studio so neither overwrites the other.
 */
import { create } from 'zustand'
import type { ModelGraph } from '../types/graph'
import type { ImportReport, ImportResponse, RunResponse } from '../types/imports'
import { isSelectionValid, type Selection, type ViewMode } from './modelStore'

interface ExplorerState {
  modelId: string | null
  graph: ModelGraph | null
  report: ImportReport | null
  selection: Selection
  viewMode: ViewMode
  /** The latest run of the imported model, with the real values it produced. */
  run: RunResponse | null

  setImported: (result: ImportResponse) => void
  select: (selection: Selection) => void
  setViewMode: (mode: ViewMode) => void
  setRun: (run: RunResponse | null) => void
}

export const useExplorerStore = create<ExplorerState>()((set) => ({
  modelId: null,
  graph: null,
  report: null,
  selection: null,
  viewMode: '2d',
  run: null,

  setImported: (result) =>
    set({ modelId: result.model_id, graph: result.graph, report: result.report, selection: null, run: null }),
  select: (selection) =>
    set((state) => (state.graph && !isSelectionValid(selection, state.graph) ? {} : { selection })),
  setViewMode: (viewMode) => set({ viewMode }),
  setRun: (run) => set((state) => (run === null || run.model_id === state.modelId ? { run } : {})),
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
