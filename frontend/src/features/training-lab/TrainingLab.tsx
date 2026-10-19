import { Network } from 'lucide-react'
import { lazy, Suspense, useEffect, useMemo, useState, type ReactNode } from 'react'
import { cancelTraining, getMetrics, startTraining } from '../../api/training'
import { ErrorBoundary } from '../../components/states/ErrorBoundary'
import { EmptyState, ErrorState, LoadingState } from '../../components/states/StateViews'
import { totalParameters } from '../../lib/graph'
import { useModelStore } from '../../stores/modelStore'
import { useTrainingStore } from '../../stores/trainingStore'
import { FIXED_CLASSES, TrainingConfigSchema } from '../../types/training'
import { useModelSync } from '../architecture-studio/useModelSync'
import { TrainingForm, type TrainingErrors } from './TrainingForm'
import { ActivationStrips, TrainingProgress } from './TrainingResults'

const MetricCharts = lazy(() => import('./MetricCharts'))
import { useTrainingStream } from './useTrainingStream'

export function TrainingLab({ onOpenStudio }: { onOpenStudio: () => void }) {
  // Builds the model from the current architecture if the studio has not yet.
  const { validation, retry } = useModelSync()
  const graph = useModelStore((s) => s.graph)
  const modelId = useModelStore((s) => s.modelId)
  const modelStatus = useModelStore((s) => s.status)
  const modelConfig = useModelStore((s) => s.modelConfig)
  const config = useTrainingStore((s) => s.config)
  const captureLayers = useTrainingStore((s) => s.captureLayers)
  const job = useTrainingStore((s) => s.job)
  const history = useTrainingStore((s) => s.history)
  const activations = useTrainingStore((s) => s.activations)
  const stream = useTrainingStore((s) => s.stream)
  const beginJob = useTrainingStore((s) => s.beginJob)
  const [starting, setStarting] = useState(false)
  const [requestError, setRequestError] = useState<unknown>(null)

  const currentJob = job && job.modelId === modelId ? job : null
  const running = currentJob?.status === 'running'
  const { reconnect } = useTrainingStream(currentJob?.jobId ?? null, running)

  const layers = useMemo(
    () => (graph?.nodes ?? []).filter((n) => n.op_type !== 'Input').map((n) => ({ id: n.id, name: n.name })),
    [graph],
  )
  const layerNames = useMemo(() => Object.fromEntries(layers.map((l) => [l.id, l.name])), [layers])
  const outputUnits = modelConfig?.output_units ?? 2

  // After a page reload or a new model, pick up an existing job's metrics.
  useEffect(() => {
    if (!modelId || currentJob) return
    const controller = new AbortController()
    getMetrics(modelId, controller.signal)
      .then((metrics) => {
        if (metrics.job) beginJob(metrics.job, metrics.history)
      })
      .catch(() => {
        // No previous job to show; the form is ready to start one.
      })
    return () => controller.abort()
  }, [modelId, currentJob, beginJob])

  const { request, errors } = useMemo(() => {
    const request = {
      ...config,
      // Default: every learned layer; never ask for layers this model lacks.
      capture_layers: (captureLayers ?? layers.map((l) => l.id)).filter((id) => id in layerNames),
    }
    const errors: TrainingErrors = {}
    const parsed = TrainingConfigSchema.safeParse(request)
    if (!parsed.success) for (const issue of parsed.error.issues) errors[issue.path.join('.')] ??= issue.message
    const fixed = FIXED_CLASSES[request.dataset]
    if (fixed !== undefined && outputUnits > fixed) {
      errors.dataset = `This dataset has ${fixed} classes, but the network has ${outputUnits} outputs. Use spiral, or set 1 or 2 output units.`
    }
    return { request, errors }
  }, [config, captureLayers, layers, layerNames, outputUnits])

  if (!graph || !modelId) {
    if (modelStatus.state === 'error') return <ErrorState error={modelStatus.error} onRetry={retry} />
    if (!validation.ok) {
      return (
        <Page>
          <EmptyState title="The architecture has errors">
            <p>Fix them in the Architecture Studio before training.</p>
            <button type="button" className="button" onClick={onOpenStudio}>
              <Network size={16} aria-hidden="true" />
              Open Architecture Studio
            </button>
          </EmptyState>
        </Page>
      )
    }
    return (
      <Page>
        <LoadingState label="Preparing the network…" />
      </Page>
    )
  }

  const start = async () => {
    setStarting(true)
    setRequestError(null)
    try {
      beginJob(await startTraining(modelId, TrainingConfigSchema.parse(request)))
    } catch (error) {
      setRequestError(error)
    } finally {
      setStarting(false)
    }
  }

  const cancel = async () => {
    if (!currentJob) return
    try {
      await cancelTraining(currentJob.jobId)
    } catch (error) {
      setRequestError(error)
    }
  }

  const parameters = totalParameters(graph)

  return (
    <Page
      subtitle={`${graph.model.name} · ${graph.nodes.length} layers${parameters !== null ? ` · ${parameters.toLocaleString()} parameters` : ''}`}
      onOpenStudio={onOpenStudio}
    >
      <div className="training-grid">
        <TrainingForm
          errors={errors}
          layers={layers}
          outputUnits={outputUnits}
          running={running}
          starting={starting}
          onStart={() => void start()}
          onCancel={() => void cancel()}
        />
        <div className="training-main">
          {requestError !== null && <ErrorState error={requestError} />}
          {currentJob ? (
            <div className="card">
              <TrainingProgress job={currentJob} history={history} />
              {currentJob.failure && <ErrorState error={new Error(currentJob.failure)} />}
              {stream === 'error' && running && (
                <ErrorState error={new Error('Lost the live connection to this training job.')} onRetry={reconnect} />
              )}
            </div>
          ) : (
            <div className="card">
              <EmptyState title="No training yet">
                Choose settings and press Start training. Loss and accuracy appear here as each epoch
                finishes.
              </EmptyState>
            </div>
          )}
          {history.length > 0 && (
            <ErrorBoundary fallback={(_, reset) => <ErrorState error={new Error('The charts could not be shown.')} onRetry={reset} />}>
              <Suspense fallback={<LoadingState label="Loading charts…" />}>
                <MetricCharts history={history} />
              </Suspense>
            </ErrorBoundary>
          )}
          <ActivationStrips activations={activations} layerNames={layerNames} />
        </div>
      </div>
    </Page>
  )
}

function Page({
  children,
  subtitle,
  onOpenStudio,
}: {
  children: ReactNode
  subtitle?: string
  onOpenStudio?: () => void
}) {
  return (
    <section aria-labelledby="training-title" className="training">
      <header className="studio-header">
        <h1 id="training-title">Training Lab</h1>
        {subtitle && <p className="muted">{subtitle}</p>}
        {onOpenStudio && (
          <button type="button" className="link-button" onClick={onOpenStudio}>
            Edit architecture
          </button>
        )}
      </header>
      {children}
    </section>
  )
}
