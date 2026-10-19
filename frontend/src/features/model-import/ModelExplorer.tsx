import { useMemo } from 'react'
import { ModelViewContext, type ModelView } from '../../app/modelView'
import { Inspector } from '../../components/inspector/Inspector'
import { EmptyState } from '../../components/states/StateViews'
import { useExplorerStore } from '../../stores/explorerStore'
import { GraphView } from '../architecture-studio/GraphView'
import { ImportReportPanel } from './ImportReportPanel'
import { UploadPanel } from './UploadPanel'

export function ModelExplorer() {
  const { modelId, graph, report, selection, viewMode, select, setViewMode, setImported } = useExplorerStore()
  // Imported models never run, so there are no measured values or weight changes.
  const view = useMemo<ModelView>(
    () => ({ modelId, weightsVersion: 0, forward: null, selection, select, viewMode, setViewMode }),
    [modelId, selection, select, viewMode, setViewMode],
  )

  return (
    <ModelViewContext.Provider value={view}>
      <section aria-labelledby="explorer-title" className="studio">
        <header className="studio-header">
          <h1 id="explorer-title">Model Explorer</h1>
          <p className="muted">Inspect the structure and weights of trained ONNX models.</p>
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
          <Inspector graph={graph} />
        </div>
      </section>
    </ModelViewContext.Provider>
  )
}
