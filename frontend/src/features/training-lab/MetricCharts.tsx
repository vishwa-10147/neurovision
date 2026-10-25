// Recharts is large; this module is lazy-loaded by the Training Lab.
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { formatValue } from '../../lib/values'
import type { EpochMetrics } from '../../types/training'
import {
  chartRows,
  COMPARE_COLORS,
  comparisonRows,
  groupMetrics,
  type ComparedRun,
  type ComparisonRow,
} from '../live-runs/series'
import type { RunPoint } from '../../types/runs'

// A fixed height: percentage heights collapse inside auto-sized grid cards.
const CHART_HEIGHT = 220

export default function MetricCharts({ history }: { history: EpochMetrics[] }) {
  return (
    <div className="charts">
      <MetricChart
        title="Loss"
        history={history}
        lines={[
          { key: 'loss', name: 'Training' },
          { key: 'val_loss', name: 'Validation' },
        ]}
      />
      <MetricChart
        title="Accuracy"
        history={history}
        percent
        lines={[
          { key: 'accuracy', name: 'Training' },
          { key: 'val_accuracy', name: 'Validation' },
        ]}
      />
    </div>
  )
}

/**
 * One chart per metric family (e.g. loss with val_loss), by step. Given
 * `runs` instead of `points`, every run's curves are overlaid, one color per run.
 */
export function RunMetricCharts(props: { points: RunPoint[] } | { runs: ComparedRun[] }) {
  if ('runs' in props) return <ComparedMetricCharts runs={props.runs} />
  const { points } = props
  const groups = groupMetrics(points)
  const rows = chartRows(points)
  return (
    <div className="charts">
      {groups.map((group) => (
        <MetricChart
          key={group.title}
          title={group.title}
          history={rows}
          xKey="step"
          xLabel={(step) => `Step ${step}`}
          percent={group.percent}
          // A function, not a string key: metric names may contain dots.
          lines={group.keys.map((key) => ({ key, name: key, value: (row: RunPoint) => row.metrics[key] }))}
          unit="step"
        />
      ))}
    </div>
  )
}

function ComparedMetricCharts({ runs }: { runs: ComparedRun[] }) {
  const groups = groupMetrics(runs.flatMap((run) => run.points))
  const rows = comparisonRows(runs)
  const logged = runs.map((run) => new Set(run.points.flatMap((p) => Object.keys(p.metrics))))
  return (
    <div className="charts">
      {groups.map((group) => (
        <MetricChart
          key={group.title}
          title={group.title}
          history={rows}
          xKey="step"
          xLabel={(step) => `Step ${step}`}
          percent={group.percent}
          // Runs log different steps; draw each curve across the gaps.
          connectNulls
          // The comparison lists the runs and their colors once, above the charts.
          legend={false}
          lines={runs.flatMap((run, r) =>
            group.keys
              .filter((key) => logged[r].has(key))
              .map((key) => ({
                key: `${run.runId}:${key}`,
                name: `${run.name} · ${key}`,
                value: (row: ComparisonRow) => row.byRun[run.runId]?.[key],
                color: COMPARE_COLORS[r % COMPARE_COLORS.length],
                // Validation and other variants are dashed in the run's color.
                dash: key === group.keys[0] ? undefined : '5 4',
              })),
          )}
          unit="step"
        />
      ))}
    </div>
  )
}

function MetricChart<Row extends object>({
  title,
  history,
  lines,
  percent = false,
  xKey = 'epoch',
  xLabel = (epoch) => `Epoch ${epoch}`,
  unit = 'epoch',
  connectNulls = false,
  legend = true,
}: {
  title: string
  history: Row[]
  lines: {
    key: string
    name: string
    value?: (row: Row) => number | undefined
    color?: string
    dash?: string
  }[]
  percent?: boolean
  xKey?: string
  xLabel?: (x: unknown) => string
  unit?: string
  connectNulls?: boolean
  legend?: boolean
}) {
  const colors = ['var(--chart-train)', 'var(--chart-val)', 'var(--accent)', 'var(--warn)']
  const tick = (v: number) => (percent ? `${Math.round(v * 100)}%` : formatValue(v, 2))
  return (
    <figure className="chart card" aria-label={`${title} by ${unit}`}>
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <LineChart data={history} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis dataKey={xKey} stroke="var(--muted)" fontSize={12} />
          <YAxis stroke="var(--muted)" fontSize={12} tickFormatter={tick} domain={percent ? [0, 1] : ['auto', 'auto']} width={48} />
          <Tooltip
            contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
            formatter={(value) => (typeof value === 'number' ? tick(value) : String(value))}
            labelFormatter={xLabel}
          />
          {legend && <Legend />}
          {lines.map((line, i) => (
            <Line
              key={line.key}
              type="monotone"
              dataKey={line.value ?? line.key}
              name={line.name}
              stroke={line.color ?? colors[i % colors.length]}
              strokeDasharray={line.dash}
              connectNulls={connectNulls}
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </figure>
  )
}
