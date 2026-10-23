/**
 * Two-way control of an external run (ADR 0008). Buttons only issue commands;
 * the script applies them at its next run.poll() and reports back, and the
 * list below shows what it actually did.
 */
import { Pause, Play, Save, Square } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { sendCommand } from '../../api/runs'
import { ErrorState } from '../../components/states/StateViews'
import { formatValue } from '../../lib/values'
import { useLiveRunStore, type WatchedRun } from '../../stores/liveRunStore'
import type { CommandKind, RunCommand } from '../../types/events'

/** At most this many recent commands are listed. */
const RECENT_COMMANDS = 6

const LABELS: Record<CommandKind, string> = {
  pause: 'Pause',
  resume: 'Resume',
  stop: 'Stop',
  checkpoint: 'Save checkpoint',
  set_lr: 'Set learning rate',
}

const STATUS_TEXT: Record<RunCommand['status'], string> = {
  pending: 'Waiting for the script…',
  applied: 'Applied',
  rejected: 'Rejected',
  expired: 'Expired',
}

export function RunControls({ run }: { run: WatchedRun }) {
  const commands = useLiveRunStore((s) => s.commands)
  const recordCommand = useLiveRunStore((s) => s.recordCommand)
  const [error, setError] = useState<unknown>(null)
  const [sending, setSending] = useState<CommandKind | null>(null)
  const [confirmStop, setConfirmStop] = useState(false)
  const [rate, setRate] = useState('')

  const running = run.status === 'running'
  const can = (kind: CommandKind) => run.controls.includes(kind)
  const recent = Object.values(commands)
    .sort((a, b) => b.command_id - a.command_id)
    .slice(0, RECENT_COMMANDS)
  const waiting = (kind: CommandKind) => recent.some((c) => c.kind === kind && c.status === 'pending')

  if (run.controls.length === 0) {
    if (!running) return null
    return (
      <p className="muted run-controls-off">
        This run cannot be controlled from here. Start it with <code>start_run(..., control=True)</code> to
        pause, stop, save checkpoints or change the learning rate from the dashboard.
      </p>
    )
  }

  const send = async (kind: CommandKind, value?: number) => {
    setSending(kind)
    setError(null)
    try {
      recordCommand(await sendCommand(run.runId, kind, value))
    } catch (caught) {
      setError(caught)
    } finally {
      setSending(null)
    }
  }

  const parsedRate = Number(rate)
  const rateValid = rate.trim() !== '' && Number.isFinite(parsedRate) && parsedRate > 0 && parsedRate <= 10
  const submitRate = (event: FormEvent) => {
    event.preventDefault()
    if (rateValid) void send('set_lr', parsedRate).then(() => setRate(''))
  }

  return (
    <section className="run-controls" aria-label="Run controls">
      <div className="button-row">
        {run.paused
          ? can('resume') && (
              <button
                type="button"
                className="button button-primary"
                disabled={!running || sending !== null || waiting('resume')}
                onClick={() => void send('resume')}
              >
                <Play size={16} aria-hidden="true" /> Resume
              </button>
            )
          : can('pause') && (
              <button
                type="button"
                className="button"
                disabled={!running || sending !== null || waiting('pause')}
                onClick={() => void send('pause')}
              >
                <Pause size={16} aria-hidden="true" /> Pause
              </button>
            )}
        {can('checkpoint') && (
          <button
            type="button"
            className="button"
            disabled={!running || sending !== null}
            onClick={() => void send('checkpoint')}
          >
            <Save size={16} aria-hidden="true" /> Save checkpoint
          </button>
        )}
        {can('stop') &&
          (confirmStop ? (
            <span className="confirm-stop">
              <button
                type="button"
                className="button button-danger"
                disabled={!running || sending !== null}
                onClick={() => {
                  setConfirmStop(false)
                  void send('stop')
                }}
              >
                <Square size={16} aria-hidden="true" /> Confirm stop
              </button>
              <button type="button" className="button" onClick={() => setConfirmStop(false)}>
                Keep running
              </button>
            </span>
          ) : (
            <button
              type="button"
              className="button button-danger"
              disabled={!running || sending !== null || waiting('stop')}
              onClick={() => setConfirmStop(true)}
            >
              <Square size={16} aria-hidden="true" /> Stop
            </button>
          ))}
      </div>
      {can('set_lr') && (
        <form className="lr-form" onSubmit={submitRate}>
          <div className={`field${rate !== '' && !rateValid ? ' has-error' : ''}`}>
            <label htmlFor="run-lr">Learning rate</label>
            <input
              id="run-lr"
              type="number"
              inputMode="decimal"
              step="any"
              min="0"
              max="10"
              placeholder={run.learningRate !== null ? formatValue(run.learningRate, 6) : 'e.g. 0.001'}
              value={rate}
              onChange={(e) => setRate(e.target.value)}
              disabled={!running}
              aria-describedby="run-lr-hint"
            />
          </div>
          <button type="submit" className="button" disabled={!running || !rateValid || sending !== null}>
            Set
          </button>
          <p id="run-lr-hint" className="muted">
            Now {run.learningRate !== null ? formatValue(run.learningRate, 6) : 'unknown'}. Applied to every
            optimizer parameter group; a scheduler may override it.
          </p>
        </form>
      )}
      {error !== null && <ErrorState error={error} />}
      {recent.length > 0 && (
        <ul className="command-list" aria-label="Recent commands">
          {recent.map((command) => (
            <li key={command.command_id} className={`command command-${command.status}`}>
              <span className="command-kind">
                {LABELS[command.kind]}
                {command.value !== null && ` → ${formatValue(command.value, 6)}`}
              </span>
              <span className="command-status">{STATUS_TEXT[command.status]}</span>
              {command.message && <span className="muted command-message">{command.message}</span>}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
