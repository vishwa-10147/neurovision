import { Heatmap } from '../../components/heatmap/Heatmap'
import { ErrorState, LoadingState } from '../../components/states/StateViews'
import { useTensor } from '../../hooks/useTensor'
import { formatValue } from '../../lib/values'
import type { ViewerTarget } from '../../types/maps'

const MAX_FILTERS = 16
const MAX_IN_CHANNELS = 8
const KERNEL_PX = 48

/**
 * Convolution weights [out channels, in channels, kH, kW] as a grid of
 * kernels: one row per output filter, one column per input channel.
 * Uses the bounded tensor endpoint (first 4,096 values).
 */
export function FiltersView({
  modelId,
  target,
}: {
  modelId: string
  target: Extract<ViewerTarget, { kind: 'filters' }>
}) {
  const tensor = useTensor(modelId, target.tensorId, 0)
  const [outChannels, inChannels, kh, kw] = target.shape
  const perFilter = inChannels * kh * kw

  if (target.shape.length !== 4) return <p className="muted">Filters need a 4D weight [out, in, height, width].</p>
  if (perFilter > 4096) {
    return <p className="muted">Each filter has {perFilter.toLocaleString()} weights, more than the 4,096 a view can load.</p>
  }
  if (tensor.state === 'idle' || tensor.state === 'loading') return <LoadingState label="Loading weights…" />
  if (tensor.state === 'error') return <ErrorState error={tensor.error} />

  const values = tensor.data.values
  const filters = Math.min(outChannels, MAX_FILTERS, Math.floor(values.length / perFilter))
  const channels = Math.min(inChannels, MAX_IN_CHANNELS)
  const kernel = (o: number, i: number) =>
    Array.from({ length: kh }, (_, r) => values.slice(o * perFilter + i * kh * kw + r * kw, o * perFilter + i * kh * kw + (r + 1) * kw))
  let limit = 0
  for (let o = 0; o < filters; o++)
    for (let i = 0; i < channels; i++) for (const row of kernel(o, i)) for (const v of row) if (v !== null) limit = Math.max(limit, Math.abs(v))
  const cellSize = Math.max(4, Math.floor(KERNEL_PX / Math.max(kh, kw)))

  return (
    <div className="tensor-maps">
      <p className="muted">
        Weight {target.label}: {outChannels} filters × {inChannels} input channels × {kh} × {kw}. Each kernel
        is what a filter multiplies with one input channel. Orange negative, blue positive; full color at ±
        {formatValue(limit, 3)}.
      </p>
      <table className="filters">
        <thead>
          <tr>
            <th scope="col">Filter</th>
            {Array.from({ length: channels }, (_, i) => (
              <th key={i} scope="col">
                in {i}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: filters }, (_, o) => (
            <tr key={o}>
              <th scope="row">{o}</th>
              {Array.from({ length: channels }, (_, i) => (
                <td key={i}>
                  <Heatmap
                    values={kernel(o, i)}
                    scale={{ kind: 'diverging', limit }}
                    cellSize={cellSize}
                    label={`Filter ${o}, input channel ${i}`}
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {(filters < outChannels || channels < inChannels) && (
        <p className="muted">
          Showing {filters} of {outChannels} filters and {channels} of {inChannels} input channels.
        </p>
      )}
    </div>
  )
}
