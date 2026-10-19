import { RefreshCw } from 'lucide-react'
import { API_BASE_URL } from '../../api/client'
import type { BackendHealth } from '../../hooks/useBackendHealth'

interface DashboardProps {
  health: BackendHealth
  onRecheck: () => void
}

export function Dashboard({ health, onRecheck }: DashboardProps) {
  return (
    <section aria-labelledby="dashboard-title" className="page">
      <h1 id="dashboard-title">Dashboard</h1>
      <p className="muted">
        Explore neural-network architecture, signal propagation, and training.
      </p>

      <div className="card">
        <div className="card-header">
          <h2>Backend service</h2>
          <button
            type="button"
            onClick={onRecheck}
            disabled={health.state === 'checking'}
            className="button"
          >
            <RefreshCw size={16} aria-hidden="true" />
            Check again
          </button>
        </div>

        {health.state === 'checking' && <p className="muted">Checking {API_BASE_URL}…</p>}

        {health.state === 'online' && (
          <dl className="facts">
            <dt>Status</dt>
            <dd>{health.data.status}</dd>
            <dt>Service</dt>
            <dd>{health.data.service}</dd>
            <dt>Version</dt>
            <dd>{health.data.version}</dd>
            <dt>Environment</dt>
            <dd>{health.data.environment}</dd>
            <dt>Last checked</dt>
            <dd>{health.checkedAt.toLocaleTimeString()}</dd>
          </dl>
        )}

        {health.state === 'offline' && (
          <div className="error" role="alert">
            <p>{health.message}</p>
            <p className="muted">
              Start it from <code>backend/</code> with the virtual environment active:{' '}
              <code>uvicorn app.main:app --reload --port 8000</code>. If it runs elsewhere,
              set <code>VITE_API_BASE_URL</code>.
            </p>
          </div>
        )}
      </div>
    </section>
  )
}
