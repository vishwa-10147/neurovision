/**
 * Neural Lab: configure one of NeuroVision's built-in PyTorch models and train
 * it live. The server runs the bundled script, which streams to Live Runs
 * (graph, metrics, health, visuals, system use, controls) like any script.
 */
import { Play, Radio, Square } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { launchLab, listLabJobs, listTemplates, stopLabJob, type LabJob, type ParamSpec, type Template } from '../../api/lab'
import { EmptyState, ErrorState, LoadingState } from '../../components/states/StateViews'

const JOB_POLL_MS = 2000
const STATUS_LABELS = { running: 'Training', finished: 'Finished', failed: 'Failed', stopped: 'Stopped' } as const

export function NeuralLab({ onWatch }: { onWatch: (runName: string) => void }) {
  const [templates, setTemplates] = useState<{ list: Template[]; enabled: boolean; python: string | null } | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [selected, setSelected] = useState<string>('cnn')
  const [values, setValues] = useState<Record<string, Record<string, number | string>>>({})
  const [jobs, setJobs] = useState<LabJob[]>([])
  const [launching, setLaunching] = useState(false)
  const [launchError, setLaunchError] = useState<unknown>(null)

  const load = useCallback(() => {
    const controller = new AbortController()
    listTemplates(controller.signal).then(
      (t) => {
        setTemplates({ list: t.templates, enabled: t.enabled, python: t.python })
        setError(null)
      },
      (caught) => !controller.signal.aborted && setError(caught),
    )
    return () => controller.abort()
  }, [])
  // load() only sets state after the request resolves.
  // oxlint-disable-next-line react/set-state-in-effect
  useEffect(load, [load])

  useEffect(() => {
    const controller = new AbortController()
    const poll = () => listLabJobs(controller.signal).then((r) => setJobs(r.jobs), () => undefined)
    void poll()
    const timer = window.setInterval(() => void poll(), JOB_POLL_MS)
    return () => {
      window.clearInterval(timer)
      controller.abort()
    }
  }, [])

  const template = templates?.list.find((t) => t.id === selected) ?? null
  const params = template ? { ...defaults(template), ...values[template.id] } : {}
  const setParam = (name: string, value: number | string) =>
    setValues((v) => ({ ...v, [selected]: { ...v[selected], [name]: value } }))

  const train = async () => {
    if (!template) return
    setLaunching(true)
    setLaunchError(null)
    try {
      const job = await launchLab(template.id, params)
      setJobs((j) => [job, ...j.filter((x) => x.job_id !== job.job_id)])
    } catch (caught) {
      setLaunchError(caught)
    } finally {
      setLaunching(false)
    }
  }

  return (
    <section aria-labelledby="lab-title" className="training">
      <header className="studio-header">
        <h1 id="lab-title">Neural Lab</h1>
        <p className="muted">Configure a real PyTorch model and watch it train live: graph, metrics, visuals and health.</p>
      </header>
      {error !== null && <ErrorState error={error} onRetry={load} />}
      {!templates && error === null && <LoadingState label="Loading models…" />}
      {templates && !templates.enabled && (
        <div className="card">
          <EmptyState title="The Neural Lab is turned off on this server">
            It starts training processes, so it is on by default only in development. Set
            NEUROVISION_LAB_ENABLED=true on the backend to turn it on.
          </EmptyState>
        </div>
      )}
      {templates?.enabled && (
        <div className="training-grid">
          <div className="card run-list lab-templates">
            <h2 className="run-list-title">Models</h2>
            <ul aria-label="Model templates">
              {templates.list.map((t) => (
                <li key={t.id}>
                  <button type="button" className="run-item" aria-pressed={t.id === selected} onClick={() => setSelected(t.id)}>
                    <span className="run-name">{t.label}</span>
                    <span className="muted run-meta">{t.family}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
          <div className="training-main">
            {template && (
              <form
                className="card training-form"
                noValidate
                onSubmit={(e) => {
                  e.preventDefault()
                  void train()
                }}
              >
                <h2 className="run-title">{template.label}</h2>
                <p className="muted">{template.description}</p>
                <p className="muted">
                  In Live runs you will see: the network&apos;s graph, loss and metrics, health alerts, CPU/GPU use and ETA
                  {template.shows.length ? `, plus ${template.shows.join('; ')}` : ''}.
                </p>
                <div className="lab-params">
                  {template.params.map((spec) => (
                    <ParamField key={spec.name} spec={spec} value={params[spec.name]} onChange={(v) => setParam(spec.name, v)} />
                  ))}
                </div>
                {launchError !== null && <ErrorState error={launchError} />}
                <div className="training-actions">
                  <button type="submit" className="button button-primary" disabled={launching}>
                    <Play size={16} aria-hidden="true" />
                    {launching ? 'Starting…' : 'Train live'}
                  </button>
                </div>
                {templates.python && <p className="muted lab-python">Runs with {templates.python}</p>}
              </form>
            )}
            <section className="card" aria-label="Lab runs">
              <h3>Lab runs</h3>
              {jobs.length === 0 ? (
                <p className="muted">Nothing trained yet. Started models appear here and in Live runs.</p>
              ) : (
                <ul className="lab-jobs">
                  {jobs.map((job) => (
                    <li key={job.job_id} className="lab-job">
                      <div className="lab-job-head">
                        <span className="run-name">{job.run_name}</span>
                        <span className={`status-badge status-${job.status === 'running' ? 'running' : job.status === 'finished' ? 'completed' : job.status === 'failed' ? 'failed' : 'cancelled'}`}>
                          {STATUS_LABELS[job.status]}
                        </span>
                        <button type="button" className="button" onClick={() => onWatch(job.run_name)}>
                          <Radio size={14} aria-hidden="true" /> Watch in Live runs
                        </button>
                        {job.status === 'running' && (
                          <button type="button" className="button" onClick={() => void stopLabJob(job.job_id).then((j) => setJobs((all) => all.map((x) => (x.job_id === j.job_id ? j : x))))}>
                            <Square size={14} aria-hidden="true" /> Stop
                          </button>
                        )}
                      </div>
                      {job.output.length > 0 && (
                        <details open={job.status === 'failed'}>
                          <summary className="muted">Script output</summary>
                          <pre className="code-block">{job.output.slice(-12).join('\n')}</pre>
                        </details>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </div>
        </div>
      )}
    </section>
  )
}

function defaults(template: Template): Record<string, number | string> {
  return Object.fromEntries(template.params.map((p) => [p.name, p.default]))
}

function ParamField({ spec, value, onChange }: { spec: ParamSpec; value: number | string; onChange: (v: number | string) => void }) {
  if (spec.kind === 'choice') {
    return (
      <label className="field">
        <span>{spec.label}</span>
        <select value={String(value)} onChange={(e) => onChange(e.target.value)}>
          {(spec.choices ?? []).map((c) => (
            <option key={c} value={c}>
              {c.toUpperCase()}
            </option>
          ))}
        </select>
      </label>
    )
  }
  return (
    <label className="field">
      <span>{spec.label}</span>
      <input
        type="number"
        value={value}
        min={spec.min ?? undefined}
        max={spec.max ?? undefined}
        step={spec.kind === 'int' ? 1 : 'any'}
        onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))}
      />
      {spec.help && <small className="muted">{spec.help}</small>}
    </label>
  )
}
