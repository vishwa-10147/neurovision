/**
 * Placing a live run's per-layer statistics on its linked ONNX graph. Pure, so
 * it is unit-tested.
 *
 * PyTorch's exporter names each ONNX node by the module that produced it, e.g.
 * "/layer1/layer1.0/relu_2/Relu" for the third call of `layer1[0].relu`. The
 * SDK reports statistics by module name ("layer1.0.relu"), so the two meet on
 * the module path. Matching is by name only and is labeled as such.
 */
import type { StepHealth } from '../../types/events'
import type { ModelGraph } from '../../types/graph'

/** "/layer1/layer1.0/relu_2/Relu" → "layer1.0.relu_2"; null for unscoped nodes. */
export function modulePath(nodeName: string): string | null {
  const segments = nodeName.split('/').filter(Boolean)
  if (segments.length < 2) return null
  let path = ''
  for (const segment of segments.slice(0, -1)) {
    // Children of containers repeat the parent path ("layer1", "layer1.0").
    path = path && !segment.startsWith(`${path}.`) ? `${path}.${segment}` : segment
  }
  return path
}

/** Without the exporter's "_2" suffix for a module called more than once. */
const withoutCallSuffix = (path: string) => path.replace(/_\d+$/, '')

export interface ModuleIndex {
  /** Node ids per module path, in graph order. */
  byPath: Map<string, string[]>
  /** Node ids that own weights. */
  weighted: Set<string>
}

export function indexModules(graph: ModelGraph): ModuleIndex {
  const byPath = new Map<string, string[]>()
  const weighted = new Set<string>()
  for (const node of graph.nodes) {
    const path = modulePath(node.name)
    if (path !== null) byPath.set(path, [...(byPath.get(path) ?? []), node.id])
    if (node.parameter_summary && node.parameter_summary.parameter_count > 0) weighted.add(node.id)
  }
  return { byPath, weighted }
}

/** Nodes produced by a module: its own operators, else those of its children. */
export function nodesOf(index: ModuleIndex, module: string): string[] {
  // Every call of the module ("block.1", then "block.1_1"), in graph order.
  const called = [...index.byPath]
    .filter(([path]) => path === module || withoutCallSuffix(path) === module)
    .flatMap(([, ids]) => ids)
  if (called.length > 0) return called
  return [...index.byPath].filter(([path]) => path.startsWith(`${module}.`)).flatMap(([, ids]) => ids)
}

export type ColorMode = 'activations' | 'gradients' | 'dead'

export const MODES: Record<ColorMode, { label: string; name: string; legend: string }> = {
  activations: {
    label: 'Activation size',
    name: 'activation RMS',
    legend: "Operator color: size (RMS) of the layer's output in the latest logged step, scaled across layers",
  },
  gradients: {
    label: 'Gradient norm',
    name: 'gradient norm',
    legend: "Operator color: the layer's gradient norm in the latest logged step, scaled across layers",
  },
  dead: {
    label: 'Dead units',
    name: 'dead fraction',
    legend: 'Operator color: share of units that output zero for the whole probe batch (ReLU family)',
  },
}

export interface LiveColors {
  values: Record<string, number>
  /** Reported layer names placed on the graph, and those that matched nothing. */
  matched: string[]
  unmatched: string[]
}

const rms = (values: number[]) => Math.sqrt(values.reduce((sum, v) => sum + v * v, 0) / Math.max(1, values.length))

/**
 * One value per graph node for the chosen mode. An activation lands on the
 * last operator of its module (its output); a gradient norm on every operator
 * of the module that owns weights.
 */
export function liveColors(
  index: ModuleIndex,
  mode: ColorMode,
  health: StepHealth | null,
  activations: Record<string, number[]>,
): LiveColors {
  const values: Record<string, number> = {}
  const matched: string[] = []
  const unmatched: string[] = []
  const place = (module: string, value: number, which: 'last' | 'weighted') => {
    const nodes = nodesOf(index, module)
    const targets = which === 'last' ? nodes.slice(-1) : nodes.filter((id) => index.weighted.has(id))
    if (targets.length === 0) {
      unmatched.push(module)
      return
    }
    matched.push(module)
    for (const id of targets) values[id] = value
  }
  if (mode === 'activations') {
    const fromHealth = Object.entries(health?.activations ?? {}).map(
      ([module, a]) => [module, Math.sqrt(a.mean ** 2 + a.std ** 2)] as const,
    )
    // Values a probe sent take precedence over the health summary of the same layer.
    const fromProbe = Object.entries(activations).map(([module, v]) => [module, rms(v)] as const)
    for (const [module, value] of new Map([...fromHealth, ...fromProbe])) place(module, value, 'last')
  } else if (mode === 'gradients') {
    for (const [module, p] of Object.entries(health?.parameters ?? {})) {
      if (p.grad_norm !== null) place(module, p.grad_norm, 'weighted')
    }
  } else {
    for (const [module, a] of Object.entries(health?.activations ?? {})) {
      if (a.dead_fraction !== null) place(module, a.dead_fraction, 'last')
    }
  }
  return { values, matched, unmatched }
}

/** The module a node belongs to, and the statistics reported for it. */
export function statsForNode(
  graph: ModelGraph,
  nodeId: string,
  health: StepHealth | null,
  activations: Record<string, number[]>,
) {
  const node = graph.nodes.find((n) => n.id === nodeId)
  const path = node ? modulePath(node.name) : null
  if (!node || path === null) return { node, module: null, activation: null, parameters: null, probe: null }
  // Walk up the module path until something was reported.
  const parts = path.split('.')
  for (let n = parts.length; n > 0; n--) {
    const candidate = parts.slice(0, n).join('.')
    const keys = [candidate, withoutCallSuffix(candidate)]
    const activation = keys.map((k) => health?.activations[k]).find(Boolean) ?? null
    const parameters = keys.map((k) => health?.parameters[k]).find(Boolean) ?? null
    const probe = keys.map((k) => activations[k]).find(Boolean) ?? null
    if (activation || parameters || probe) {
      return { node, module: candidate, activation, parameters, probe }
    }
  }
  return { node, module: path, activation: null, parameters: null, probe: null }
}
