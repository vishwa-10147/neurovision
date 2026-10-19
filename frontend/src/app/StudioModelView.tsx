import { useMemo, type ReactNode } from 'react'
import { currentForward, useModelStore } from '../stores/modelStore'
import { ModelViewContext, type ModelView } from './modelView'

/** Provides the Architecture Studio's educational model to graph views. */
export function StudioModelView({ children }: { children: ReactNode }) {
  const modelId = useModelStore((s) => s.modelId)
  const weightsVersion = useModelStore((s) => s.weightsVersion)
  const forward = useModelStore(currentForward)
  const selection = useModelStore((s) => s.selection)
  const select = useModelStore((s) => s.select)
  const viewMode = useModelStore((s) => s.viewMode)
  const setViewMode = useModelStore((s) => s.setViewMode)
  const value = useMemo<ModelView>(
    () => ({ modelId, weightsVersion, forward, selection, select, viewMode, setViewMode }),
    [modelId, weightsVersion, forward, selection, select, viewMode, setViewMode],
  )
  return <ModelViewContext.Provider value={value}>{children}</ModelViewContext.Provider>
}
