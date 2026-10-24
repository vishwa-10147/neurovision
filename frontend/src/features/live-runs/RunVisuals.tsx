/**
 * Custom visuals a training script attached to its latest step: scatter
 * plots (e.g. a latent space) and matrices (hidden states, attention,
 * generated images). Drawn exactly as sent; nothing is interpolated.
 */
import { Heatmap } from '../../components/heatmap/Heatmap'
import { classColor } from '../../lib/classColors'
import type { Scale } from '../../lib/colors'
import type { Visual } from '../../types/events'
import { scatterBounds, toRows } from './visuals'

const PLOT = 280
const PAD = 28

export function RunVisuals({ visuals }: { visuals: { step: number; items: Record<string, Visual> } | null }) {
  if (!visuals) return null
  return (
    <section className="card run-visuals" aria-label="Custom visuals">
      <h3>Visuals from the script</h3>
      <p className="muted">Sent with step {visuals.step.toLocaleString()}, exactly as measured by the training script.</p>
      <div className="visual-grid">
        {Object.entries(visuals.items).map(([name, visual]) => (
          <figure key={name} className="visual">
            {visual.kind === 'scatter' ? <ScatterPlot name={name} visual={visual} /> : <MatrixView name={name} visual={visual} />}
            <figcaption>{name}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  )
}

function ScatterPlot({ name, visual }: { name: string; visual: Extract<Visual, { kind: 'scatter' }> }) {
  const b = scatterBounds(visual.x, visual.y)
  const sx = (v: number) => PAD + ((v - b.minX) / b.spanX) * (PLOT - 2 * PAD)
  const sy = (v: number) => PLOT - PAD - ((v - b.minY) / b.spanY) * (PLOT - 2 * PAD)
  const classes = visual.labels ? new Set(visual.labels).size : 0
  return (
    <svg
      width={PLOT}
      height={PLOT}
      viewBox={`0 0 ${PLOT} ${PLOT}`}
      role="img"
      aria-label={`${name}: ${visual.x.length} points${classes ? ` in ${classes} classes` : ''}`}
    >
      <rect x={PAD} y={PAD} width={PLOT - 2 * PAD} height={PLOT - 2 * PAD} className="visual-frame" />
      {visual.x.map((x, i) => (
        <circle
          key={i}
          cx={sx(x)}
          cy={sy(visual.y[i])}
          r={3}
          fill={visual.labels ? classColor(visual.labels[i]) : 'var(--accent)'}
          fillOpacity={0.8}
        />
      ))}
      <text x={PLOT / 2} y={PLOT - 6} textAnchor="middle" className="visual-axis">
        {visual.x_label ?? 'x'} ({b.minX.toPrecision(2)} … {b.maxX.toPrecision(2)})
      </text>
      <text x={10} y={PLOT / 2} textAnchor="middle" className="visual-axis" transform={`rotate(-90 10 ${PLOT / 2})`}>
        {visual.y_label ?? 'y'} ({b.minY.toPrecision(2)} … {b.maxY.toPrecision(2)})
      </text>
    </svg>
  )
}

function MatrixView({ name, visual }: { name: string; visual: Extract<Visual, { kind: 'matrix' }> }) {
  const min = Math.min(...visual.values)
  const max = Math.max(...visual.values)
  const scale: Scale =
    visual.scale === 'sequential'
      ? { kind: 'sequential', min, max: max > min ? max : min + 1 }
      : { kind: 'diverging', limit: Math.max(Math.abs(min), Math.abs(max)) }
  return (
    <>
      <Heatmap
        values={toRows(visual.values, visual.rows, visual.cols)}
        scale={scale}
        // Wide matrices may use up to twice the plot width.
        cellSize={Math.max(2, Math.floor(Math.min((2 * PLOT) / visual.cols, PLOT / visual.rows)))}
        label={`${name}, ${visual.rows} by ${visual.cols}`}
        rowName={visual.row_label ?? 'row'}
        colName={visual.col_label ?? 'column'}
        interactive
      />
      <p className="muted visual-range">
        {visual.rows} × {visual.cols}
        {visual.row_label && visual.col_label ? ` (${visual.row_label} × ${visual.col_label})` : ''}, values{' '}
        {min.toPrecision(3)} … {max.toPrecision(3)}
      </p>
    </>
  )
}
