import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Heatmap } from '../../components/heatmap/Heatmap'
import { useExplorerStore } from '../../stores/explorerStore'
import { fakeBackend, tensorResponse, type FakeBackend } from '../../test/fakeBackend'
import { ModelGraphSchema } from '../../types/graph'
import type { ViewerTarget } from '../../types/maps'
import { ModelExplorer } from './ModelExplorer'
import { TensorViewer } from './TensorViewer'
import { tensorTarget } from './viewerTargets'

let backend: FakeBackend
const initialExplorer = useExplorerStore.getState()

const square = (n: number, at: (r: number, c: number) => number) =>
  Array.from({ length: n }, (_, r) => Array.from({ length: n }, (_, c) => at(r, c)))

const viewBase = (shape: number[], fixed: number[], values: number[][]) => {
  const flat = values.flat()
  return {
    model_id: 'onnx_v',
    run_id: 'run_1',
    node_id: 'node_0',
    tensor: 't',
    shape,
    fixed,
    min: Math.min(...flat),
    max: Math.max(...flat),
    non_finite_count: 0,
  }
}

/** Serve grids and maps of a [1, items, n, n] tensor whose item i is filled with `at(i, r, c)`. */
function serveViews(items: number, n: number, at: (i: number, r: number, c: number) => number) {
  const shape = [1, items, n, n]
  backend.use((method, path, _body, params) => {
    const fixed = params.getAll('fixed').map(Number)
    if (method === 'GET' && path === '/models/onnx_v/grids') {
      const maps = Array.from({ length: items }, (_, i) => square(n, (r, c) => at(i, r, c)))
      return Response.json({
        ...viewBase(shape, fixed, maps.flat()),
        axis: 1,
        items_total: items,
        rows: n,
        cols: n,
        items: maps.map((values, index) => ({ index, stride: 1, values, ...viewBase(shape, fixed, values) })),
      })
    }
    if (method === 'GET' && path === '/models/onnx_v/maps') {
      const values = square(n, (r, c) => at(fixed[1], r, c))
      return Response.json({ ...viewBase(shape, fixed, values), rows: n, cols: n, stride: 1, values })
    }
    return undefined
  })
}

function renderViewer(target: ViewerTarget) {
  return render(<TensorViewer modelId="onnx_v" runId="run_1" target={target} onClose={() => {}} />)
}

beforeEach(() => {
  useExplorerStore.setState(initialExplorer, true)
  backend = fakeBackend()
})

afterEach(() => vi.unstubAllGlobals())

describe('Feature maps', () => {
  it('shows every channel and loads the selected one in detail', async () => {
    serveViews(3, 4, (i, r, c) => i - r + c)
    renderViewer(tensorTarget('node_0', 0, [1, 3, 4, 4], 'conv_out'))

    const overview = await screen.findByRole('list', { name: 'channels overview' })
    expect(within(overview).getAllByRole('button')).toHaveLength(3)
    expect(screen.getByText(/assume the usual \[batch, channels, height, width\] order/)).toBeInTheDocument()

    await userEvent.click(within(overview).getByRole('button', { name: 'channel 2' }))
    expect(within(overview).getByRole('button', { name: 'channel 2' })).toHaveAttribute('aria-pressed', 'true')
    await screen.findByRole('img', { name: 'channel 2, 4 by 4' })
    const detailCalls = backend.calls.filter((c) => c.path === '/models/onnx_v/maps')
    expect(detailCalls.at(-1)?.params.getAll('fixed')).toEqual(['0', '2'])
    expect(detailCalls.at(-1)?.query).toMatchObject({ run_id: 'run_1', node_id: 'node_0', output: '0' })
    // One symmetric scale for the whole grid: values span -3 … 5.
    expect(screen.getByLabelText('Map legend')).toHaveTextContent('full color at ±5.000 (shared across all maps)')
  })

  it('can show each map on its own scale so weak channels stay visible', async () => {
    serveViews(3, 4, (i, r, c) => i - r + c)
    renderViewer(tensorTarget('node_0', 0, [1, 3, 4, 4], 'conv_out'))
    await screen.findByRole('img', { name: 'channel 0, 4 by 4' })
    expect(screen.getByLabelText('Map legend')).toHaveTextContent('±5.000 (shared across all maps)')
    await userEvent.click(screen.getByRole('button', { name: 'Each map on its own scale' }))
    // Channel 0 spans -3 … 3 on its own.
    expect(screen.getByLabelText('Map legend')).toHaveTextContent('±3.000 for this map (each map uses its own range)')
    expect(screen.getByRole('button', { name: 'Each map on its own scale' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('can scale an attention head to its own largest weight', async () => {
    serveViews(2, 3, (i, r, c) => (c === (r + i) % 3 ? 0.8 : 0.1))
    const attention = { node_id: 'node_0', name: 'softmax', tensor: 'w', shape: ['batch', 2, 'seq', 'seq'], layout: ['batch', 'heads', 'query', 'key'] }
    renderViewer(tensorTarget('node_0', 0, [1, 2, 3, 3], 'w', attention))
    await screen.findByRole('list', { name: 'heads overview' })
    await userEvent.click(screen.getByRole('button', { name: 'Each map on its own scale' }))
    expect(await screen.findByLabelText('Map legend')).toHaveTextContent("to 0.800 (blue), this head's largest weight")
  })

  it('labels attention by query and key on a fixed 0–1 scale', async () => {
    serveViews(2, 3, (i, r, c) => (c === (r + i) % 3 ? 0.8 : 0.1))
    const attention = { node_id: 'node_0', name: 'softmax', tensor: 'w', shape: ['batch', 2, 'seq', 'seq'], layout: ['batch', 'heads', 'query', 'key'] }
    renderViewer(tensorTarget('node_0', 0, [1, 2, 3, 3], 'w', attention))

    expect(screen.getByRole('heading', { name: 'Attention weights' })).toBeInTheDocument()
    const overview = await screen.findByRole('list', { name: 'heads overview' })
    expect(within(overview).getAllByRole('button').map((b) => b.getAttribute('aria-label'))).toEqual(['head 0', 'head 1'])
    expect(await screen.findByRole('heading', { name: /head 0 \(query × key\)/ })).toBeInTheDocument()
    const legend = await screen.findByLabelText('Map legend')
    expect(legend).toHaveTextContent('Weight from 0 (neutral) to 1 (blue)')
    expect(legend).toHaveTextContent('Rows sum to 1 for a softmax over keys.')
    expect(screen.getByText(/Detected as attention by pattern/)).toBeInTheDocument()
  })

  it('says to run again when the run is no longer kept', async () => {
    backend.use((_method, path) =>
      path === '/models/onnx_v/grids' || path === '/models/onnx_v/maps'
        ? Response.json(
            { error: { code: 'run_expired', message: 'That run is no longer kept. Run the model again to view its tensors.', details: [] } },
            { status: 410 },
          )
        : undefined,
    )
    renderViewer(tensorTarget('node_0', 0, [1, 3, 4, 4], 'conv_out'))
    expect(await screen.findAllByText(/no longer kept\. Run the model again/)).not.toHaveLength(0)
  })
})

describe('Convolution filters', () => {
  it('draws one kernel per filter and input channel and says what is left out', async () => {
    // 20 filters × 1 input channel × 3 × 3 = 180 weights.
    const values = Array.from({ length: 180 }, (_, i) => (i % 9) - 4)
    backend.use((_method, path) =>
      path === '/models/onnx_v/tensors/w0' ? tensorResponse('w0', [20, 1, 3, 3], values) : undefined,
    )
    renderViewer({ kind: 'filters', nodeId: 'node_0', tensorId: 'w0', shape: [20, 1, 3, 3], label: 'conv.weight' })

    const table = await screen.findByRole('table')
    expect(within(table).getAllByRole('img')).toHaveLength(16)
    expect(within(table).getByRole('img', { name: 'Filter 15, input channel 0' })).toBeInTheDocument()
    expect(screen.getByText('Showing 16 of 20 filters and 1 of 1 input channels.')).toBeInTheDocument()
    expect(screen.getByText(/full color at ±\s*4\.000/)).toBeInTheDocument()
  })
})

describe('Heatmap', () => {
  it('reads values with the arrow keys', async () => {
    render(
      <Heatmap
        values={[
          [0.1, 0.2],
          [0.3, null],
        ]}
        scale={{ kind: 'sequential', min: 0, max: 1 }}
        cellSize={10}
        label="Test map"
        rowName="query"
        colName="key"
        interactive
      />,
    )
    const map = screen.getByRole('img', { name: 'Test map' })
    map.focus()
    await userEvent.keyboard('{ArrowDown}')
    expect(screen.getByText('query 0, key 0: 0.1000')).toBeInTheDocument()
    await userEvent.keyboard('{ArrowDown}{ArrowRight}')
    expect(screen.getByText('query 1, key 1: NaN/∞')).toBeInTheDocument()
    await userEvent.keyboard('{ArrowRight}')
    expect(screen.getByText('query 1, key 1: NaN/∞')).toBeInTheDocument()
  })
})

// An imported graph with one attention softmax and one convolution.
const graph = ModelGraphSchema.parse({
  schema_version: '1.0',
  model: {
    id: 'onnx_v',
    name: 'tiny',
    source_format: 'onnx',
    capabilities: { graph: true, inference: true, training: false, activations: true, gradients: false, attention: true },
  },
  nodes: [
    { id: 'input_0', name: 'x', op_type: 'Input', output_tensors: ['x'], output_shapes: [['batch', 1, 3, 3]] },
    {
      id: 'node_0',
      name: 'conv',
      op_type: 'Conv',
      input_tensors: ['x', 'conv.weight'],
      output_tensors: ['c'],
      output_shapes: [['batch', 2, 3, 3]],
      metadata: { onnx: { initializers: [{ name: 'conv.weight', tensor_id: 'w0', shape: [2, 1, 3, 3], dtype: 'float32', external: false }] } },
    },
    { id: 'node_1', name: 'softmax', op_type: 'Softmax', input_tensors: ['c'], output_tensors: ['w'], output_shapes: [['batch', 2, 3, 3]] },
    { id: 'output_0', name: 'w', op_type: 'Output', input_tensors: ['w'] },
  ],
  edges: [
    { id: 'e_0', source: 'input_0', target: 'node_0', tensor_name: 'x' },
    { id: 'e_1', source: 'node_0', target: 'node_1', tensor_name: 'c' },
    { id: 'e_2', source: 'node_1', target: 'output_0', tensor_name: 'w' },
  ],
})
const candidate = { node_id: 'node_1', name: 'softmax', tensor: 'w', shape: ['batch', 2, 3, 3], layout: ['batch', 'heads', 'query', 'key'] }
const report = {
  format: 'onnx',
  file_name: 'tiny.onnx',
  file_bytes: 512,
  operator_count: 2,
  edge_count: 3,
  initializer_count: 1,
  parameter_count: 18,
  external_tensor_count: 0,
  unsupported_operators: [],
  opsets: [{ domain: 'ai.onnx', version: 17 }],
  warnings: [],
  inputs: [{ name: 'x', node_id: 'input_0', dtype: 'float', shape: ['batch', 1, 3, 3] }],
  inference_blockers: [],
  attention: [candidate],
}
const measured = (name: string, nodeId: string) => ({
  name,
  node_id: nodeId,
  shape: [1, 2, 3, 3],
  dtype: 'float32',
  count: 18,
  summary: { min: 0, max: 1, mean: 0.5, std: 0.1 },
  non_finite_count: 0,
  values: Array(18).fill(0.5),
  truncated: false,
})
const run = (retained: boolean) => ({
  model_id: 'onnx_v',
  run_id: 'run_1',
  retained,
  inputs: [{ name: 'x', node_id: 'input_0', shape: [1, 1, 3, 3], dtype: 'float32' }],
  runtime: 'onnxruntime 1.31.0',
  elapsed_ms: 0.2,
  outputs: [measured('w', 'node_1')],
  intermediates: [measured('c', 'node_0')],
  notes: [],
})

const inspector = () => within(screen.getByRole('complementary', { name: 'Inspector' }))
const attentionRegion = () => within(screen.getByRole('region', { name: 'Attention' }))

function setupExplorer(retained = true) {
  backend.use((method, path) => (method === 'GET' && path === '/import/onnx/samples' ? Response.json([]) : undefined))
  backend.use((method, path) => (method === 'POST' && path === '/models/onnx_v/run' ? Response.json(run(retained)) : undefined))
  serveViews(2, 3, () => 1 / 3)
  useExplorerStore.getState().setImported({ model_id: 'onnx_v', graph, report: report as never })
  render(<ModelExplorer />)
}

describe('Viewer entry points', () => {
  it('lists attention candidates and opens them after a run', async () => {
    setupExplorer()
    expect(attentionRegion().getByRole('button', { name: 'View attention' })).toBeDisabled()
    expect(attentionRegion().getByText('Run the model first to see the weights.')).toBeInTheDocument()
    expect(attentionRegion().getByText(/Found by pattern/)).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Run' }))
    await screen.findByLabelText('Run results')
    await userEvent.click(attentionRegion().getByRole('button', { name: 'View attention' }))
    const viewer = screen.getByRole('region', { name: 'Attention weights' })
    expect(await within(viewer).findByRole('list', { name: 'heads overview' })).toBeInTheDocument()

    await userEvent.click(within(viewer).getByRole('button', { name: 'Close the viewer' }))
    expect(screen.queryByRole('region', { name: 'Attention weights' })).not.toBeInTheDocument()
  })

  it('opens maps, inputs, and filters from the inspector', async () => {
    setupExplorer()
    await userEvent.click(screen.getByRole('button', { name: 'Run' }))
    await screen.findByLabelText('Run results')

    await userEvent.click(document.querySelector('[data-node-id="node_0"] rect') as Element)
    await userEvent.click(inspector().getByRole('button', { name: 'View as maps' }))
    expect(await screen.findByRole('list', { name: 'channels overview' })).toBeInTheDocument()

    backend.use((_method, path) =>
      path === '/models/onnx_v/tensors/w0' ? tensorResponse('w0', [2, 1, 3, 3], Array.from({ length: 18 }, (_, i) => i - 9)) : undefined,
    )
    await userEvent.click(inspector().getByRole('button', { name: 'View filters' }))
    expect(await screen.findByRole('region', { name: 'Convolution filters' })).toBeInTheDocument()

    await userEvent.click(document.querySelector('[data-node-id="node_1"] rect') as Element)
    expect(inspector().getByRole('note')).toHaveTextContent('Likely attention weights')
    await userEvent.click(inspector().getByRole('button', { name: 'View attention heads' }))
    expect(screen.getByRole('region', { name: 'Attention weights' })).toBeInTheDocument()

    await userEvent.click(document.querySelector('[data-node-id="input_0"] rect') as Element)
    expect(inspector().getByLabelText('Input in the last run')).toHaveTextContent('x [1 × 1 × 3 × 3] float32')
  })

  it('offers no map views when the run was too large to keep', async () => {
    setupExplorer(false)
    await userEvent.click(screen.getByRole('button', { name: 'Run' }))
    await screen.findByLabelText('Run results')
    expect(attentionRegion().getByRole('button', { name: 'View attention' })).toBeDisabled()
    await userEvent.click(document.querySelector('[data-node-id="node_0"] rect') as Element)
    expect(inspector().getByLabelText('Output in the last run')).toBeInTheDocument()
    expect(inspector().queryByRole('button', { name: 'View as maps' })).not.toBeInTheDocument()
  })
})
