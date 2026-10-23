import { useCallback, useMemo, useState } from 'react'
import { runImported } from '../../api/imports'
import { ModelViewContext, type ModelView } from '../../app/modelView'
import { Inspector } from '../../components/inspector/Inspector'
import { EmptyState, ErrorState } from '../../components/states/StateViews'
import { runValuesByNode, useExplorerStore } from '../../stores/explorerStore'
import { GraphView } from '../architecture-studio/GraphView'
import { ImportReportPanel } from './ImportReportPanel'
import { RunPanel } from './RunPanel'
import { TensorViewer } from './TensorViewer'
import { UploadPanel } from './UploadPanel'

export function ModelExplorer() {
  const { modelId, graph, report, selection, viewMode, select, setViewMode, setImported, run, viewer, openViewer } =
    useExplorerStore()
  const runValues = useMemo(() => runValuesByNode(run), [run])
  const [capturing, setCapturing] = useState(false)
  const [captureError, setCaptureError] = useState<unknown>(null)
  const captureAndRun = useCallback(async (nodeId: string) => {
    const state = useExplorerStore.getState()
    if (!state.modelId || !state.lastInputs) return null
    const capture = state.addCapture(nodeId)
    setCapturing(true)
    setCaptureError(null)
    try {
      const next = await runImported(state.modelId, state.lastInputs, undefined, capture)
      useExplorerStore.getState().setRun(next)
      return next
    } catch (error) {
      setCaptureError(error)
      return null
    } finally {
      setCapturing(false)
    }
  }, [])
  const attention = report?.attention
  // Imported weights never change, so the weights version stays 0.
  const view = useMemo<ModelView>(
    () => ({
      modelId,
      weightsVersion: 0,
      forward: null,
      selection,
      select,
      viewMode,
      setViewMode,
      runValues,
      explorer: { run, attention: attention ?? [], openViewer, captureAndRun, capturing },
    }),
    [modelId, selection, select, viewMode, setViewMode, runValues, run, attention, openViewer, captureAndRun, capturing],
  )
  // Filters need only the model; tensor maps need a run the worker still keeps.
  const showViewer = modelId && viewer && (viewer.kind === 'filters' || run)

  return (
    <ModelViewContext.Provider value={view}>
      <section aria-labelledby="explorer-title" className="studio">
        <header className="studio-header">
          <h1 id="explorer-title">Model Explorer</h1>
          <p className="muted">Inspect and run trained ONNX models.</p>
        </header>
        <div className="studio-grid">
          <div className="studio-side">
            <UploadPanel onImported={setImported} />
            {graph && report && (
              <ImportReportPanel
                graph={graph}
                report={report}
                run={run}
                onView={openViewer}
                onCapture={captureAndRun}
                capturing={capturing}
              />
            )}
          </div>
          <div className="explorer-center">
          <div className="studio-canvas card">
            {graph ? (
              <GraphView graph={graph} overlay={null} />
            ) : (
              <div className="state-padded">
                <EmptyState title="No model imported yet">
                  Import an .onnx file or try a sample. Models exported from PyTorch, TensorFlow, or
                  scikit-learn to ONNX work here.
                </EmptyState>
              </div>
            )}
          </div>
          {showViewer && (
            <TensorViewer modelId={modelId} runId={run?.run_id ?? ''} target={viewer} onClose={() => openViewer(null)} />
          )}
          </div>
          <div className="studio-side">
            {/* Keyed by model, so input drafts start fresh for each import. */}
            {modelId && report && <RunPanel key={modelId} modelId={modelId} report={report} />}
            {captureError !== null && <ErrorState error={captureError} />}
            <Inspector graph={graph} />
          </div>
        </div>
      </section>
    </ModelViewContext.Provider>
  )
}
