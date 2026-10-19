import { Box, Square } from 'lucide-react'
import { lazy, Suspense, useMemo, useState } from 'react'
import { ErrorBoundary } from '../../components/states/ErrorBoundary'
import { ErrorState, LoadingState } from '../../components/states/StateViews'
import { useModelView } from '../../app/modelView'
import type { ModelGraph } from '../../types/graph'
import { Network2D } from '../visualization-2d/Network2D'
import { hasWebGL } from '../visualization-3d/environment'
import { MAX_3D_NODES } from '../visualization-3d/scene'

// three.js is large; load it only when the 3D view is first opened.
const Network3D = lazy(() => import('../visualization-3d/Network3D'))

/** Switches between the 2D and 3D renderers; both read the same graph and selection. */
export function GraphView({ graph, overlay }: { graph: ModelGraph; overlay: string | null }) {
  const { viewMode, setViewMode, forward } = useModelView()
  const [webgl] = useState(hasWebGL)
  // First (only) example's post-activation values for each layer.
  const activations = useMemo(
    () =>
      forward
        ? Object.fromEntries(Object.entries(forward.response.activations).map(([id, v]) => [id, v.post[0]]))
        : null,
    [forward],
  )

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
        <Network2D graph={graph} overlay={overlay} activations={activations} />
      ) : !webgl || graph.nodes.length > MAX_3D_NODES ? (
        <div className="graph-view-fallback">
          <ErrorState
            error={
              new Error(
                webgl
                  ? `This graph has ${graph.nodes.length.toLocaleString()} nodes; the 3D view shows at most ${MAX_3D_NODES.toLocaleString()}. Use the 2D view for graphs this large.`
                  : 'The 3D view needs WebGL, which this browser has turned off or does not support.',
              )
            }
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
