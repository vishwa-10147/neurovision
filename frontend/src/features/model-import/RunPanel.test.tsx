import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useExplorerStore } from '../../stores/explorerStore'
import { fakeBackend, type FakeBackend } from '../../test/fakeBackend'
import { ModelGraphSchema } from '../../types/graph'
import { ModelExplorer } from './ModelExplorer'
import { toRunInput, topValues } from './runInputs'

const initialExplorer = useExplorerStore.getState()
let backend: FakeBackend

// A runnable imported graph: x [batch, 4] → Gemm → Relu → output.
const graph = ModelGraphSchema.parse({
  schema_version: '1.0',
  model: {
    id: 'onnx_run',
    name: 'tiny_mlp',
    source_format: 'onnx',
    capabilities: { graph: true, inference: true, training: false, activations: true, gradients: false, attention: false },
  },
  nodes: [
    { id: 'input_0', name: 'x', op_type: 'Input', output_tensors: ['x'], output_shapes: [['batch', 4]], metadata: { onnx: { dtype: 'float' } } },
    { id: 'node_0', name: 'fc1', op_type: 'Gemm', input_tensors: ['x'], output_tensors: ['h1'], input_shapes: [['batch', 4]], output_shapes: [['batch', 3]] },
    { id: 'node_1', name: 'relu', op_type: 'Relu', input_tensors: ['h1'], output_tensors: ['y'], input_shapes: [['batch', 3]], output_shapes: [['batch', 3]] },
    { id: 'output_0', name: 'y', op_type: 'Output', input_tensors: ['y'], input_shapes: [['batch', 3]] },
  ],
  edges: [
    { id: 'e_0', source: 'input_0', target: 'node_0', tensor_name: 'x' },
    { id: 'e_1', source: 'node_0', target: 'node_1', tensor_name: 'h1' },
    { id: 'e_2', source: 'node_1', target: 'output_0', tensor_name: 'y' },
  ],
})
const report = {
  format: 'onnx',
  file_name: 'tiny_mlp.onnx',
  file_bytes: 512,
  operator_count: 2,
  edge_count: 3,
  initializer_count: 0,
  parameter_count: 0,
  external_tensor_count: 0,
  unsupported_operators: [],
  opsets: [{ domain: 'ai.onnx', version: 17 }],
  warnings: [],
  inputs: [{ name: 'x', node_id: 'input_0', dtype: 'float', shape: ['batch', 4] }],
  inference_blockers: [] as string[],
  attention: [],
}
const tensor = (name: string, nodeId: string, values: number[], shape = [1, 3]) => ({
  name,
  node_id: nodeId,
  shape,
  dtype: 'float32',
  count: values.length,
  summary: { min: Math.min(...values), max: Math.max(...values), mean: values.reduce((a, b) => a + b) / values.length, std: 0.5 },
  non_finite_count: 0,
  values,
  truncated: false,
})
const runResponse = {
  model_id: 'onnx_run',
  runtime: 'onnxruntime 1.31.0',
  elapsed_ms: 0.42,
  outputs: [tensor('y', 'node_1', [0.1, 0.9, 0.0])],
  intermediates: [tensor('h1', 'node_0', [0.1, 0.9, -0.4])],
  notes: ['This run also started the inference worker process.'],
  run_id: 'run_test',
  retained: true,
  inputs: [{ name: 'x', node_id: 'input_0', shape: [1, 4], dtype: 'float32' }],
}

const inspector = () => within(screen.getByRole('complementary', { name: 'Inspector' }))

function setup(blockers: string[] = []) {
  useExplorerStore.getState().setImported({
    model_id: 'onnx_run',
    graph,
    report: { ...report, inference_blockers: blockers } as never,
  })
  render(<ModelExplorer />)
}

beforeEach(() => {
  useExplorerStore.setState(initialExplorer, true)
  backend = fakeBackend()
  backend.use((method, path) => (method === 'GET' && path === '/import/onnx/samples' ? Response.json([]) : undefined))
  backend.use((method, path) => (method === 'POST' && path === '/models/onnx_run/run' ? Response.json(runResponse) : undefined))
})

afterEach(() => vi.unstubAllGlobals())

describe('Running imported models', () => {
  it('explains why a model cannot run', () => {
    setup(['It uses non-standard operators ONNX Runtime cannot run: Scale.'])
    const panel = screen.getByRole('region', { name: 'Run the model' })
    expect(panel).toHaveTextContent('This model cannot be run here:')
    expect(panel).toHaveTextContent('Scale')
    expect(within(panel).queryByRole('button', { name: 'Run' })).not.toBeInTheDocument()
  })

  it('runs with generated inputs and shows real outputs', async () => {
    setup()
    expect(screen.getByLabelText('x dimension 1 (fixed)')).toBeDisabled()
    const batch = screen.getByLabelText('x dimension 0')
    await userEvent.clear(batch)
    await userEvent.type(batch, '3')
    await userEvent.click(screen.getByRole('button', { name: 'Run' }))

    const results = await screen.findByLabelText('Run results')
    expect(backend.calls.find((c) => c.path === '/models/onnx_run/run')?.body).toEqual({
      inputs: { x: { shape: [3, 4], fill: 'random', seed: 0 } },
    })
    expect(results).toHaveTextContent('0.42 ms in onnxruntime 1.31.0 · 1 intermediate tensors captured')
    expect(results).toHaveTextContent('This run also started the inference worker process.')
    expect(results).toHaveTextContent('Top 3 of the first row (index: value): 1: 0.9000, 0: 0.1000, 2: 0.0000')
  })

  it('shades operators and shows their measured outputs in the inspector', async () => {
    setup()
    await userEvent.click(screen.getByRole('button', { name: 'Run' }))
    await screen.findByLabelText('Run results')

    const gemm = document.querySelector('[data-node-id="node_0"] rect')
    expect(gemm).toHaveAttribute('data-value')
    expect(screen.getByText(/Operator color: size \(RMS\) of its output in the last run/)).toBeInTheDocument()

    await userEvent.click(gemm as Element)
    const measured = inspector().getByLabelText('Output in the last run')
    expect(measured).toHaveTextContent('h1 [1 × 3] float32 · 3 values')
    expect(measured).toHaveTextContent('0.1000, 0.9000, -0.4000')

    await userEvent.click(document.querySelector('[data-edge-id="e_1"]') as Element)
    expect(inspector().getByLabelText('Values in the last run')).toHaveTextContent('h1')
  })

  it('sends typed values and checks their count first', async () => {
    setup()
    await userEvent.selectOptions(screen.getByLabelText('Values for x'), 'custom')
    const box = screen.getByLabelText('Typed values for x')
    await userEvent.type(box, '1, 2, 3')
    expect(screen.getByText(/needs 4 values; you entered 3/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Run' })).toBeDisabled()

    await userEvent.type(box, ', 4')
    await userEvent.click(screen.getByRole('button', { name: 'Run' }))
    await screen.findByLabelText('Run results')
    expect(backend.calls.find((c) => c.path === '/models/onnx_run/run')?.body).toEqual({
      inputs: { x: { shape: [1, 4], values: [1, 2, 3, 4] } },
    })
  })

  it('shows runtime failures from the server', async () => {
    backend.use((method, path) =>
      method === 'POST' && path === '/models/onnx_run/run'
        ? Response.json(
            { error: { code: 'inference_timeout', message: 'The model did not finish within 10 seconds and was stopped.', details: [] } },
            { status: 504 },
          )
        : undefined,
    )
    setup()
    await userEvent.click(screen.getByRole('button', { name: 'Run' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('did not finish within 10 seconds')
  })
})

describe('run input helpers', () => {
  const spec = { name: 'x', node_id: 'input_0', dtype: 'float', shape: ['batch', 4] as (string | number)[] }

  it('validates dimensions and totals', () => {
    expect(toRunInput(spec, { dims: ['0', '4'], mode: 'random', seed: '0', custom: '' })).toHaveProperty('error')
    // Numbers follow the user's locale (e.g. 1,00,00,000 in en-IN).
    expect(toRunInput(spec, { dims: ['100000', '100'], mode: 'zeros', seed: '0', custom: '' })).toEqual({
      error: `'x' would have ${(10_000_000).toLocaleString()} values; the limit is ${(1_000_000).toLocaleString()}.`,
    })
    expect(toRunInput(spec, { dims: ['2', '4'], mode: 'ones', seed: '9', custom: '' })).toEqual({
      input: { shape: [2, 4], fill: 'ones', seed: 0 },
    })
    expect(toRunInput(spec, { dims: ['1', '4'], mode: 'custom', seed: '0', custom: '1 2 x 4' })).toEqual({
      error: "Values for 'x' must be numbers.",
    })
  })

  it('finds top values only for [batch, classes] outputs', () => {
    const t = { ...runResponse.outputs[0], values: [0.2, null, 0.7, 0.1] as (number | null)[], shape: [1, 4], count: 4 }
    expect(topValues(t)).toEqual([[2, 0.7], [0, 0.2], [3, 0.1]])
    expect(topValues({ ...t, shape: [1, 2, 2] })).toBeNull()
  })
})
