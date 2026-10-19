// Recharts is large; this module is lazy-loaded by the Training Lab.
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { formatValue } from '../../lib/values'
import type { EpochMetrics } from '../../types/training'

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

function MetricChart({
  title,
  history,
  lines,
  percent = false,
}: {
  title: string
  history: EpochMetrics[]
  lines: { key: keyof EpochMetrics; name: string }[]
  percent?: boolean
}) {
  const colors = ['var(--chart-train)', 'var(--chart-val)']
  const tick = (v: number) => (percent ? `${Math.round(v * 100)}%` : formatValue(v, 2))
  return (
    <figure className="chart card" aria-label={`${title} by epoch`}>
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <LineChart data={history} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis dataKey="epoch" stroke="var(--muted)" fontSize={12} />
          <YAxis stroke="var(--muted)" fontSize={12} tickFormatter={tick} domain={percent ? [0, 1] : ['auto', 'auto']} width={48} />
          <Tooltip
            contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
            formatter={(value) => (typeof value === 'number' ? tick(value) : String(value))}
            labelFormatter={(epoch) => `Epoch ${epoch}`}
          />
          <Legend />
          {lines.map((line, i) => (
            <Line
              key={line.key}
              type="monotone"
              dataKey={line.key}
              name={line.name}
              stroke={colors[i]}
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
