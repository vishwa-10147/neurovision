/**
 * Model Explorer state: the imported model and its own selection and view mode,
 * kept apart from the Architecture Studio so neither overwrites the other.
 */
import { create } from 'zustand'
import type { ModelGraph } from '../types/graph'
import type { ImportReport, ImportResponse } from '../types/imports'
import { isSelectionValid, type Selection, type ViewMode } from './modelStore'

interface ExplorerState {
  modelId: string | null
  graph: ModelGraph | null
  report: ImportReport | null
  selection: Selection
  viewMode: ViewMode

  setImported: (result: ImportResponse) => void
  select: (selection: Selection) => void
  setViewMode: (mode: ViewMode) => void
}

export const useExplorerStore = create<ExplorerState>()((set) => ({
  modelId: null,
  graph: null,
  report: null,
  selection: null,
  viewMode: '2d',

  setImported: (result) =>
    set({ modelId: result.model_id, graph: result.graph, report: result.report, selection: null }),
  select: (selection) =>
    set((state) => (state.graph && !isSelectionValid(selection, state.graph) ? {} : { selection })),
  setViewMode: (viewMode) => set({ viewMode }),
}))
