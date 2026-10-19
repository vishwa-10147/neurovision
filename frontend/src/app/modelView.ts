/**
 * What a graph view needs from whichever page hosts it: the model's id, its
 * current selection, and the 2D/3D mode. The Architecture Studio and the Model
 * Explorer each provide their own, so they never overwrite each other's state.
 */
import { createContext, useContext } from 'react'
import type { ForwardSnapshot, Selection, ViewMode } from '../stores/modelStore'

export interface ModelView {
  modelId: string | null
  /** Increases when the model's weights change, so cached tensors refresh. */
  weightsVersion: number
  /** Measured values from a forward pass of the current weights, if any. */
  forward: ForwardSnapshot | null
  selection: Selection
  select: (selection: Selection) => void
  viewMode: ViewMode
  setViewMode: (mode: ViewMode) => void
}

export const ModelViewContext = createContext<ModelView | null>(null)

export function useModelView(): ModelView {
  const view = useContext(ModelViewContext)
  if (!view) throw new Error('useModelView must be used inside a ModelViewContext provider.')
  return view
}
