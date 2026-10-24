/**
 * The network a live run is training, drawn with the same 2D/3D renderers as
 * the Model Explorer and colored by the statistics of the latest logged step.
 */
import { useEffect, useMemo, useState } from 'react'
import { getModelGraph } from '../../api/models'
import { ModelViewContext, type ModelView } from '../../app/modelView'
import { ErrorState, LoadingState } from '../../components/states/StateViews'
import { formatValue } from '../../lib/values'
import type { Selection, ViewMode } from '../../stores/modelStore'
import type { StepHealth } from '../../types/events'
import type { ModelGraph } from '../../types/graph'
import { GraphView } from '../architecture-studio/GraphView'
import { indexModules, liveColors, MODES, statsForNode, type ColorMode } from './graphMatching'

export function LiveGraph({
  modelId,
  runName,
  health,
  activations,
}: {
  modelId: string
  runName: string
  health: { step: number; stats: StepHealth } | null
  activations: Record<string, number[]>
}) {
  const [graph, setGraph] = useState<ModelGraph | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [attempt, setAttempt] = useState(0)
  // null until the user picks: then the first view with data is shown.
  const [chosen, setChosen] = useState<ColorMode | null>(null)
  const [selection, setSelection] = useState<Selection>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('2d')

  useEffect(() => {
    const controller = new AbortController()
    getModelGraph(modelId, controller.signal).then(
      (g) => {
        setGraph(g)
        setError(null)
      },
      (caught) => {
        if (!controller.signal.aborted) setError(caught)
      },
    )
    return () => controller.abort()
  }, [modelId, attempt])

  const index = useMemo(() => (graph ? indexModules(graph) : null), [graph])
  const stats = health?.stats ?? null
  const mode: ColorMode =
    chosen ??
    (index &&
    liveColors(index, 'activations', stats, activations).matched.length === 0 &&
    Object.keys(stats?.parameters ?? {}).length > 0
      ? 'gradients'
      : 'activations')
  const colors = useMemo(
    () => (index ? liveColors(index, mode, stats, activations) : null),
    [index, mode, stats, activations],
  )
  const view = useMemo<ModelView>(
    () => ({
      modelId,
      weightsVersion: 0,
      forward: null,
      selection,
      select: setSelection,
      viewMode,
      setViewMode,
      runValues: null,
    }),
    [modelId, selection, viewMode],
  )
  const selectedId = selection && 'nodeId' in selection ? selection.nodeId : null
  const selected = graph && selectedId ? statsForNode(graph, selectedId, stats, activations) : null

  return (
    <section className="card live-graph" aria-label="Live network graph">
      <h3>Network graph, live</h3>
      {error !== null && <ErrorState error={error} onRetry={() => setAttempt((n) => n + 1)} />}
      {graph === null && error === null && <LoadingState label="Loading the model's graph…" />}
      {graph && colors && (
        <>
          <p className="muted">
            The network being trained in “{runName}”: {graph.nodes.length.toLocaleString()} operators, exported
            from the training script. {health ? `Colored by step ${health.step.toLocaleString()}; ` : 'Waiting for per-layer statistics; '}
            {colors.matched.length} reported layer{colors.matched.length === 1 ? '' : 's'} placed on the graph by module
            name. Operators whose module reported nothing for this view stay uncolored.
          </p>
          <div className="chip-row" role="group" aria-label="Color by">
            {(Object.keys(MODES) as ColorMode[]).map((m) => (
              <button key={m} type="button" className="chip" aria-pressed={mode === m} onClick={() => setChosen(m)}>
                {MODES[m].label}
              </button>
            ))}
          </div>
          <ModelViewContext.Provider value={view}>
            <div className="live-graph-canvas">
              <GraphView
                graph={graph}
                overlay={null}
                values={colors.values}
                valuesLabel={{
                  name: MODES[mode].name,
                  legend: MODES[mode].legend,
                  source: 'Values are the statistics your training script reported; select an operator to see them.',
                }}
              />
            </div>
          </ModelViewContext.Provider>
          {colors.unmatched.length > 0 && (
            <p className="muted">
              Not found in the graph: {colors.unmatched.join(', ')} (no operator is named after that module).
            </p>
          )}
          {selected?.node && (
            <dl className="metric-tiles" aria-label="Selected layer">
              <div>
                <dt>{selected.node.op_type}</dt>
                <dd className="mono">{selected.module ?? selected.node.name}</dd>
              </div>
              {selected.activation && (
                <>
                  <div>
                    <dt>Output mean / std</dt>
                    <dd>
                      {formatValue(selected.activation.mean, 3)} / {formatValue(selected.activation.std, 3)}
                    </dd>
                  </div>
                  {selected.activation.dead_fraction !== null && (
                    <div>
                      <dt>Dead units</dt>
                      <dd>{Math.round(selected.activation.dead_fraction * 100)}%</dd>
                    </div>
                  )}
                </>
              )}
              {selected.parameters && (
                <>
                  <div>
                    <dt>Gradient norm</dt>
                    <dd>{selected.parameters.grad_norm === null ? '–' : formatValue(selected.parameters.grad_norm, 3)}</dd>
                  </div>
                  <div>
                    <dt>Update ratio</dt>
                    <dd>
                      {selected.parameters.update_ratio === null ? '–' : formatValue(selected.parameters.update_ratio, 2)}
                    </dd>
                  </div>
                </>
              )}
              {selected.probe && (
                <div>
                  <dt>Probe values</dt>
                  <dd>{selected.probe.length} sent</dd>
                </div>
              )}
              {!selected.activation && !selected.parameters && !selected.probe && (
                <div>
                  <dt>Statistics</dt>
                  <dd>none reported for this layer</dd>
                </div>
              )}
            </dl>
          )}
        </>
      )}
    </section>
  )
}
