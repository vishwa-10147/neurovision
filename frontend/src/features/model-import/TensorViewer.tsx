import { X } from 'lucide-react'
import { useEffect, useId, useState } from 'react'
import { getTensorGrid, getTensorMap } from '../../api/imports'
import { Heatmap } from '../../components/heatmap/Heatmap'
import { ErrorState, LoadingState } from '../../components/states/StateViews'
import type { Scale } from '../../lib/colors'
import { formatValue } from '../../lib/values'
import type { TensorGrid, TensorMap, ViewerTarget } from '../../types/maps'
import { FiltersView } from './FiltersView'
import { describeShape, singular } from './viewerTargets'

type Loaded<T> = { key: string; data?: T; error?: unknown }

const THUMB_PX = 72
const DETAIL_PX = 360

/** Fetch something keyed by `key`; state is set only after the request settles. */
function useKeyed<T>(key: string | null, load: (signal: AbortSignal) => Promise<T>) {
  const [result, setResult] = useState<Loaded<T> | null>(null)
  useEffect(() => {
    if (!key) return
    const controller = new AbortController()
    load(controller.signal).then(
      (data) => setResult({ key, data }),
      (error: unknown) => {
        if (!controller.signal.aborted) setResult({ key, error })
      },
    )
    return () => controller.abort()
    // `load` is rebuilt every render; `key` captures everything it depends on.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  if (!key) return { state: 'idle' as const }
  if (result?.key !== key) return { state: 'loading' as const }
  return result.error !== undefined ? { state: 'error' as const, error: result.error } : { state: 'ready' as const, data: result.data as T }
}

export function TensorViewer({
  modelId,
  runId,
  target,
  onClose,
}: {
  modelId: string
  runId: string
  target: ViewerTarget
  onClose: () => void
}) {
  const id = useId()
  return (
    <section className="card tensor-viewer" aria-labelledby={`${id}-title`}>
      <header className="tensor-viewer-header">
        <h2 id={`${id}-title`}>{target.kind === 'filters' ? 'Convolution filters' : target.attention ? 'Attention weights' : 'Tensor maps'}</h2>
        <span className="muted mono">{target.label}</span>
        <button type="button" className="icon-button" aria-label="Close the viewer" onClick={onClose}>
          <X size={16} aria-hidden="true" />
        </button>
      </header>
      {target.kind === 'filters' ? (
        <FiltersView modelId={modelId} target={target} />
      ) : (
        <TensorMaps key={`${runId}|${target.nodeId}|${target.output}`} modelId={modelId} runId={runId} target={target} />
      )}
    </section>
  )
}

function TensorMaps({
  modelId,
  runId,
  target,
}: {
  modelId: string
  runId: string
  target: Extract<ViewerTarget, { kind: 'tensor' }>
}) {
  const { shape, layout, attention } = target
  const rank = shape.length
  const hasGrid = rank >= 3
  // Leading dimensions before the grid axis (e.g. the batch index).
  const leadCount = Math.max(0, rank - (hasGrid ? 3 : 2))
  const [lead, setLead] = useState<number[]>(() => Array(leadCount).fill(0))
  const [item, setItem] = useState(0)
  const itemAxis = hasGrid ? layout[rank - 3] : null
  const rowName = layout[rank - 2]
  const colName = layout[rank - 1]
  const base = { runId, nodeId: target.nodeId, output: target.output }

  const grid = useKeyed<TensorGrid>(hasGrid ? `grid|${lead.join(',')}` : null, (signal) =>
    getTensorGrid(modelId, { ...base, fixed: lead }, signal),
  )
  const detailFixed = hasGrid ? [...lead, item] : lead
  const detail = useKeyed<TensorMap>(`map|${detailFixed.join(',')}`, (signal) =>
    getTensorMap(modelId, { ...base, fixed: detailFixed }, signal),
  )

  // Attention rows are probability distributions: a fixed 0–1 scale keeps
  // heads comparable. Other tensors share one symmetric scale across the grid.
  const scaleFrom = (min: number | null, max: number | null): Scale =>
    attention
      ? { kind: 'sequential', min: 0, max: 1 }
      : { kind: 'diverging', limit: Math.max(Math.abs(min ?? 0), Math.abs(max ?? 0)) }
  const gridScale = grid.state === 'ready' ? scaleFrom(grid.data.min, grid.data.max) : null
  const detailScale =
    gridScale ?? (detail.state === 'ready' ? scaleFrom(detail.data.min, detail.data.max) : null)

  return (
    <div className="tensor-maps">
      <p className="muted">
        Shape {describeShape(shape, layout)}
        {attention
          ? '. Detected as attention by pattern (a softmax over a square [..., S, S] tensor), so each row is one query position’s weights over key positions.'
          : rank === 4
            ? '. Dimension roles assume the usual [batch, channels, height, width] order.'
            : '.'}
      </p>

      {leadCount > 0 && (
        <div className="dims" role="group" aria-label="Leading indices">
          {lead.map((value, axis) => (
            <label key={axis} className="dim">
              <span className="muted">{layout[axis]}</span>
              <input
                type="number"
                min={0}
                max={shape[axis] - 1}
                value={value}
                aria-label={`${layout[axis]} index`}
                onChange={(e) => {
                  const next = [...lead]
                  next[axis] = Math.min(shape[axis] - 1, Math.max(0, Number(e.target.value) || 0))
                  setLead(next)
                }}
              />
            </label>
          ))}
        </div>
      )}

      {hasGrid && itemAxis && (
        <div>
          <h3>
            All {itemAxis} <span className="muted">(select one for detail)</span>
          </h3>
          {grid.state === 'loading' && <LoadingState label={`Loading ${itemAxis}…`} />}
          {grid.state === 'error' && <ErrorState error={grid.error} />}
          {grid.state === 'ready' && gridScale && (
            <>
              <ul className="thumbs" aria-label={`${itemAxis} overview`}>
                {grid.data.items.map((thumb) => {
                  const size = Math.max(thumb.values.length, thumb.values[0]?.length ?? 0, 1)
                  return (
                    <li key={thumb.index}>
                      <button
                        type="button"
                        className="thumb"
                        aria-pressed={thumb.index === item}
                        aria-label={`${singular(itemAxis)} ${thumb.index}`}
                        onClick={() => setItem(thumb.index)}
                      >
                        <Heatmap
                          values={thumb.values}
                          scale={gridScale}
                          cellSize={Math.max(2, Math.floor(THUMB_PX / size))}
                          label={`${singular(itemAxis)} ${thumb.index} thumbnail`}
                        />
                        <span>{thumb.index}</span>
                      </button>
                    </li>
                  )
                })}
              </ul>
              {grid.data.items_total > grid.data.items.length && (
                <p className="muted">
                  Showing the first {grid.data.items.length} of {grid.data.items_total} {itemAxis}.
                </p>
              )}
            </>
          )}
        </div>
      )}

      <div>
        <h3>
          {hasGrid && itemAxis ? `${singular(itemAxis)} ${item}` : 'Map'}{' '}
          <span className="muted">
            ({rowName} × {colName})
          </span>
        </h3>
        {detail.state === 'loading' && <LoadingState label="Loading the map…" />}
        {detail.state === 'error' && <ErrorState error={detail.error} />}
        {detail.state === 'ready' && detailScale && (
          <>
            <Heatmap
              values={detail.data.values}
              scale={detailScale}
              cellSize={Math.min(24, Math.max(1, Math.floor(DETAIL_PX / Math.max(detail.data.values.length, detail.data.values[0]?.length ?? 1))))}
              label={`${hasGrid && itemAxis ? `${singular(itemAxis)} ${item}` : 'Map'}, ${detail.data.rows} by ${detail.data.cols}`}
              rowName={rowName}
              colName={colName}
              stride={detail.data.stride}
              interactive
            />
            <MapLegend scale={detailScale} map={detail.data} attention={attention} />
          </>
        )}
      </div>
    </div>
  )
}

function MapLegend({ scale, map, attention }: { scale: Scale; map: TensorMap; attention: boolean }) {
  return (
    <div className="legend map-legend" aria-label="Map legend">
      <span className="legend-item">
        <span className={`activation-scale${scale.kind === 'sequential' ? ' activation-scale-positive' : ''}`} aria-hidden="true" />
        {scale.kind === 'sequential'
          ? 'Weight from 0 (neutral) to 1 (blue), the same scale for every head'
          : `Orange negative, blue positive; full color at ±${formatValue(scale.limit, 3)} (shared across all maps)`}
      </span>
      <p className="legend-note">
        This map: min {map.min === null ? '—' : formatValue(map.min, 4)}, max{' '}
        {map.max === null ? '—' : formatValue(map.max, 4)}.
        {map.stride > 1 &&
          ` ${map.rows} × ${map.cols} values are shown as ${map.stride} × ${map.stride} block averages to stay within 65,536 cells.`}
        {map.non_finite_count > 0 && ` ${map.non_finite_count} NaN/∞ values are shown in red.`}
        {attention && ' Rows sum to 1 for a softmax over keys.'}
      </p>
    </div>
  )
}
