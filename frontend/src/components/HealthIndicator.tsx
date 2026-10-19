import type { BackendHealth } from '../hooks/useBackendHealth'

const LABELS: Record<BackendHealth['state'], string> = {
  checking: 'Checking backend…',
  online: 'Backend online',
  offline: 'Backend offline',
}

export function HealthIndicator({ health }: { health: BackendHealth }) {
  return (
    <div className="health" role="status" aria-live="polite" data-state={health.state}>
      <span className="health-dot" aria-hidden="true" />
      <span>{LABELS[health.state]}</span>
    </div>
  )
}
