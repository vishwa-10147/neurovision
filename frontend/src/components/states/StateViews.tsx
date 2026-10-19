import { AlertTriangle, Inbox, Loader2, RefreshCw } from 'lucide-react'
import type { ReactNode } from 'react'
import { ApiError } from '../../api/client'

export function LoadingState({ label }: { label: string }) {
  return (
    <div className="state" role="status" aria-live="polite">
      <Loader2 size={18} className="spin" aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="state state-empty">
      <Inbox size={20} aria-hidden="true" />
      <div>
        <p className="state-title">{title}</p>
        {children && <div className="muted">{children}</div>}
      </div>
    </div>
  )
}

interface ErrorStateProps {
  error: unknown
  onRetry?: () => void
}

export function ErrorState({ error, onRetry }: ErrorStateProps) {
  const apiError = error instanceof ApiError ? error : null
  const message =
    error instanceof Error ? error.message : 'Something went wrong. Please try again.'

  return (
    <div className="state state-error" role="alert">
      <AlertTriangle size={20} aria-hidden="true" />
      <div className="state-body">
        <p className="state-title">{message}</p>
        {apiError?.details.length ? (
          <ul className="state-details">
            {apiError.details.map((detail, i) => (
              <li key={i}>
                {detail.field && <code>{detail.field}</code>} {detail.message}
              </li>
            ))}
          </ul>
        ) : null}
        {apiError?.hint && <p className="muted">{apiError.hint}</p>}
        {apiError?.code && <p className="muted state-code">Error code: {apiError.code}</p>}
        {onRetry && (
          <button type="button" className="button" onClick={onRetry}>
            <RefreshCw size={16} aria-hidden="true" />
            Try again
          </button>
        )}
      </div>
    </div>
  )
}
