import { Inspector } from '../../components/inspector/Inspector'
import { ErrorState, LoadingState } from '../../components/states/StateViews'
import { totalParameters } from '../../lib/graph'
import { useModelStore } from '../../stores/modelStore'
import { Network2D } from '../visualization-2d/Network2D'
import { ArchitectureEditor } from './ArchitectureEditor'
import { useModelSync } from './useModelSync'

export function ArchitectureStudio() {
  const { validation, retry } = useModelSync()
  const graph = useModelStore((s) => s.graph)
  const status = useModelStore((s) => s.status)
  const errors = validation.ok ? {} : validation.errors
  const parameters = graph ? totalParameters(graph) : null

  let overlay: string | null = null
  if (!validation.ok && graph) overlay = 'Fix the highlighted fields. The diagram shows the last valid network.'
  else if (status.state === 'loading' && graph) overlay = 'Updating…'

  return (
    <section aria-labelledby="studio-title" className="studio">
      <header className="studio-header">
        <h1 id="studio-title">Architecture Studio</h1>
        {graph && (
          <p className="muted" aria-live="polite">
            {graph.model.name} · {graph.nodes.length} layers
            {parameters !== null && ` · ${parameters.toLocaleString()} trainable parameters`}
          </p>
        )}
      </header>

      <div className="studio-grid">
        <ArchitectureEditor errors={errors} />

        <div className="studio-canvas card">
          {status.state === 'error' ? (
            <ErrorState error={status.error} onRetry={retry} />
          ) : graph ? (
            <Network2D graph={graph} overlay={overlay} />
          ) : validation.ok ? (
            <LoadingState label="Building the network graph…" />
          ) : (
            <ErrorState error={new Error('Fix the highlighted fields to build the network.')} />
          )}
        </div>

        <Inspector graph={graph} />
      </div>
    </section>
  )
}
