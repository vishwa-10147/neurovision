/**
 * A decision tree: splits as "feature ≤ threshold", leaves as their majority
 * class (or regression value), colored by class and purity.
 */
import { classColor } from '../../lib/classColors'
import { layoutTree, majority, type TreeNodeLike } from './treeLayout'

const COLUMN = 64
const ROW = 70
const BOX_W = 58
const BOX_H = 38

export function TreeDiagram({
  nodes,
  title = 'Decision tree',
  description,
  featureNames = [],
  regression = false,
}: {
  nodes: TreeNodeLike[]
  title?: string
  description: string
  /** Names for feature indices; f0, f1, … otherwise. */
  featureNames?: string[]
  /** Leaves hold one value instead of per-class weights. */
  regression?: boolean
}) {
  const layout = layoutTree(nodes)
  const byId = new Map(layout.nodes.map((n) => [n.id, n]))
  const width = layout.width * COLUMN + 20
  const height = layout.depth * ROW + 10
  const cx = (x: number) => 10 + x * COLUMN + COLUMN / 2
  const cy = (depth: number) => 8 + depth * ROW
  const feature = (i: number) => featureNames[i] ?? `f${i}`
  return (
    <section className="card tree-card" aria-label={title}>
      <h3>{title}</h3>
      <p className="muted">{description}</p>
      <div className="tree-scroll">
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title} with ${nodes.length} nodes`}>
          {layout.nodes.map((n) =>
            n.left !== null && n.right !== null
              ? [n.left, n.right].map((child) => {
                  const c = byId.get(child)
                  return c ? (
                    <line key={`${n.id}-${child}`} x1={cx(n.x)} y1={cy(n.depth) + BOX_H} x2={cx(c.x)} y2={cy(c.depth)} className="tree-edge" />
                  ) : null
                })
              : null,
          )}
          {layout.nodes.map((n) => {
            const { label, share } = majority(n.counts)
            const split = n.feature !== null && n.threshold !== null
            const leafText = regression ? `value ${(n.counts[0] ?? 0).toPrecision(3)}` : `class ${label}`
            const fill = regression || n.counts.length === 0 ? 'var(--surface-2)' : classColor(label, 0.15 + 0.6 * share)
            return (
              <g key={n.id} transform={`translate(${cx(n.x) - BOX_W / 2} ${cy(n.depth)})`}>
                <rect width={BOX_W} height={BOX_H} rx={6} fill={fill} className="tree-node" />
                <text x={BOX_W / 2} y={n.samples != null ? 15 : 23} textAnchor="middle" className="tree-text">
                  {split ? `${feature(n.feature as number)} ≤ ${(n.threshold as number).toFixed(2)}` : leafText}
                </text>
                {n.samples != null && (
                  <text x={BOX_W / 2} y={30} textAnchor="middle" className="tree-text muted-text">
                    n={n.samples}
                  </text>
                )}
                <title>
                  {[
                    split ? 'Split' : 'Leaf',
                    n.samples != null ? `${n.samples} samples` : null,
                    n.impurity != null ? `Gini ${n.impurity.toFixed(3)}` : null,
                    n.counts.length ? `per class ${n.counts.map((c) => +c.toFixed(3)).join(' / ')}` : null,
                  ]
                    .filter(Boolean)
                    .join(', ')}
                </title>
              </g>
            )
          })}
        </svg>
      </div>
    </section>
  )
}
