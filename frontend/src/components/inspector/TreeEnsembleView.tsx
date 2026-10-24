/** The trees inside an ONNX-ML TreeEnsemble operator (e.g. a scikit-learn forest). */
import { useState } from 'react'
import { z } from 'zod'
import { TreeDiagram } from '../tree/TreeDiagram'

const TreeEnsembleSchema = z.object({
  kind: z.enum(['classifier', 'regressor']),
  trees: z.number().int(),
  outputs: z.number().int(),
  nodes_total: z.number().int(),
  shown: z.array(
    z.array(
      z.object({
        id: z.number().int(),
        depth: z.number().int(),
        feature: z.number().int().nullable(),
        threshold: z.number().nullable(),
        mode: z.string(),
        left: z.number().int().nullable(),
        right: z.number().int().nullable(),
        values: z.array(z.number()),
      }),
    ),
  ),
})

export function TreeEnsembleView({ metadata }: { metadata: unknown }) {
  const [index, setIndex] = useState(0)
  const parsed = TreeEnsembleSchema.safeParse(metadata)
  if (!parsed.success || parsed.data.shown.length === 0) return null
  const ensemble = parsed.data
  const tree = ensemble.shown[Math.min(index, ensemble.shown.length - 1)]
  const regression = ensemble.kind === 'regressor'
  return (
    <div className="tree-ensemble">
      <p className="muted">
        {ensemble.trees.toLocaleString()} tree{ensemble.trees === 1 ? '' : 's'}, {ensemble.nodes_total.toLocaleString()}{' '}
        nodes in total{ensemble.trees > ensemble.shown.length ? `; the first ${ensemble.shown.length} are drawn` : ''}.
      </p>
      {ensemble.shown.length > 1 && (
        <div className="chip-row" role="group" aria-label="Tree">
          {ensemble.shown.map((_, i) => (
            <button key={i} type="button" className="chip" aria-pressed={i === index} onClick={() => setIndex(i)}>
              Tree {i + 1}
            </button>
          ))}
        </div>
      )}
      <TreeDiagram
        title={`Tree ${index + 1}`}
        nodes={tree.map((n) => ({ ...n, counts: n.values, samples: null, impurity: null }))}
        regression={regression}
        description={
          regression
            ? 'Inputs go left when feature ≤ threshold; leaves hold the value this tree adds.'
            : 'Inputs go left when feature ≤ threshold (fN is input column N). Leaves show the class with the largest weight; boosted models add raw scores per class across trees.'
        }
      />
    </div>
  )
}
