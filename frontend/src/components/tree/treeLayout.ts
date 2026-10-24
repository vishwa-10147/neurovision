/** Layout and summary helpers for drawing decision trees. Pure, so unit-tested. */

/** A tree node from scikit-learn (Classical ML Lab) or an ONNX tree ensemble. */
export interface TreeNodeLike {
  id: number
  depth: number
  /** Split feature and threshold (go left when feature ≤ threshold); null for a leaf. */
  feature: number | null
  threshold: number | null
  left: number | null
  right: number | null
  /** Training samples reaching the node, when known. */
  samples?: number | null
  /** Per class: training samples (scikit-learn) or leaf weights (ONNX). */
  counts: number[]
  impurity?: number | null
}

export interface TreeLayout {
  nodes: (TreeNodeLike & { x: number })[]
  width: number
  depth: number
}

/** Leaves get consecutive x positions; a parent sits between its children. */
export function layoutTree(nodes: TreeNodeLike[]): TreeLayout {
  const byId = new Map(nodes.map((n) => [n.id, n]))
  const x = new Map<number, number>()
  let next = 0
  const place = (id: number): number => {
    const node = byId.get(id)
    if (!node) return next
    if (node.left === null || node.right === null || !byId.has(node.left)) {
      x.set(id, next++)
      return x.get(id) as number
    }
    const left = place(node.left)
    const right = place(node.right)
    x.set(id, (left + right) / 2)
    return x.get(id) as number
  }
  if (nodes.length) place(nodes[0].id)
  return {
    nodes: nodes.map((n) => ({ ...n, x: x.get(n.id) ?? 0 })),
    width: Math.max(1, next),
    depth: Math.max(0, ...nodes.map((n) => n.depth)) + 1,
  }
}

/** The majority class of a node and its share. */
export function majority(counts: number[]): { label: number; share: number } {
  // Leaf weights can be negative (boosting scores); then purity is unknown.
  const total = counts.reduce((a, b) => a + Math.max(0, b), 0) || 1
  let label = 0
  counts.forEach((c, i) => {
    if (c > counts[label]) label = i
  })
  return { label, share: Math.max(0, counts[label] ?? 0) / total }
}
