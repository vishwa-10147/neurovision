import { useMemo } from 'react'
import { ModelViewContext, type ModelView } from '../../app/modelView'
import { Inspector } from '../../components/inspector/Inspector'
import { EmptyState } from '../../components/states/StateViews'
import { runValuesByNode, useExplorerStore } from '../../stores/explorerStore'
import { GraphView } from '../architecture-studio/GraphView'
import { ImportReportPanel } from './ImportReportPanel'
import { RunPanel } from './RunPanel'
import { UploadPanel } from './UploadPanel'

export function ModelExplorer() {
  const { modelId, graph, report, selection, viewMode, select, setViewMode, setImported, run } = useExplorerStore()
  const runValues = useMemo(() => runValuesByNode(run), [run])
  // Imported weights never change, so the weights version stays 0.
  const view = useMemo<ModelView>(
    () => ({ modelId, weightsVersion: 0, forward: null, selection, select, viewMode, setViewMode, runValues }),
    [modelId, selection, select, viewMode, setViewMode, runValues],
  )

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
            {graph && report && <ImportReportPanel graph={graph} report={report} />}
          </div>
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
          <div className="studio-side">
            {/* Keyed by model, so input drafts start fresh for each import. */}
            {modelId && report && <RunPanel key={modelId} modelId={modelId} report={report} />}
            <Inspector graph={graph} />
          </div>
        </div>
      </section>
    </ModelViewContext.Provider>
  )
}
