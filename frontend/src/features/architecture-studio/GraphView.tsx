import { Box, Square } from 'lucide-react'
import { lazy, Suspense, useState } from 'react'
import { ErrorBoundary } from '../../components/states/ErrorBoundary'
import { ErrorState, LoadingState } from '../../components/states/StateViews'
import { useModelStore } from '../../stores/modelStore'
import type { ModelGraph } from '../../types/graph'
import { Network2D } from '../visualization-2d/Network2D'
import { hasWebGL } from '../visualization-3d/environment'

// three.js is large; load it only when the 3D view is first opened.
const Network3D = lazy(() => import('../visualization-3d/Network3D'))

/** Switches between the 2D and 3D renderers; both read the same graph and selection. */
export function GraphView({ graph, overlay }: { graph: ModelGraph; overlay: string | null }) {
  const viewMode = useModelStore((s) => s.viewMode)
  const setViewMode = useModelStore((s) => s.setViewMode)
  const [webgl] = useState(hasWebGL)

  return (
    <div className="graph-view">
      <div className="view-switch" role="group" aria-label="View">
        <button type="button" className="chip" aria-pressed={viewMode === '2d'} onClick={() => setViewMode('2d')}>
          <Square size={14} aria-hidden="true" />
          2D
        </button>
        <button type="button" className="chip" aria-pressed={viewMode === '3d'} onClick={() => setViewMode('3d')}>
          <Box size={14} aria-hidden="true" />
          3D
        </button>
      </div>
      {viewMode === '2d' ? (
        <Network2D graph={graph} overlay={overlay} />
      ) : !webgl ? (
        <div className="graph-view-fallback">
          <ErrorState
            error={new Error('The 3D view needs WebGL, which this browser has turned off or does not support.')}
          />
          <button type="button" className="button" onClick={() => setViewMode('2d')}>
            <Square size={16} aria-hidden="true" />
            Switch to 2D view
          </button>
        </div>
      ) : (
        <ErrorBoundary
          fallback={(error, reset) => (
            <div className="graph-view-fallback">
              <ErrorState error={new Error('The 3D view stopped working.')} onRetry={reset} />
              <p className="muted">{error instanceof Error ? error.message : null}</p>
              <button type="button" className="button" onClick={() => setViewMode('2d')}>
                <Square size={16} aria-hidden="true" />
                Switch to 2D view
              </button>
            </div>
          )}
        >
          <Suspense fallback={<LoadingState label="Loading the 3D view…" />}>
            <Network3D graph={graph} overlay={overlay} />
          </Suspense>
        </ErrorBoundary>
      )}
    </div>
  )
}
