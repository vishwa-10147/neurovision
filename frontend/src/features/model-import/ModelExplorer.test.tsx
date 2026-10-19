import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import onnxGraph from '../../../../shared/schemas/examples/onnx-import.graph.json'
import { clearTensorCache } from '../../hooks/useTensor'
import { useExplorerStore } from '../../stores/explorerStore'
import { useModelStore } from '../../stores/modelStore'
import { fakeBackend, tensorResponse, type FakeBackend } from '../../test/fakeBackend'
import { ModelExplorer } from './ModelExplorer'

const initialExplorer = useExplorerStore.getState()
const initialModel = useModelStore.getState()
let backend: FakeBackend

// The shared fixture: Conv (with a weight initializer) feeding an unsupported custom op.
const graph = {
  ...onnxGraph,
  model: { ...onnxGraph.model, id: 'onnx_1' },
  nodes: onnxGraph.nodes.map((n) =>
    n.id === 'conv_0'
      ? {
          ...n,
          metadata: {
            onnx: {
              domain: 'ai.onnx',
              initializers: [{ tensor_id: 't0', name: 'conv.weight', shape: [8, 3, 3, 3], dtype: 'float', external: false }],
            },
          },
        }
      : n,
  ),
}
const report = {
  format: 'onnx',
  file_name: 'small_cnn.onnx',
  file_bytes: 4096,
  operator_count: 2,
  edge_count: 1,
  initializer_count: 1,
  parameter_count: 216,
  external_tensor_count: 0,
  unsupported_operators: [{ op_type: 'MyCustomOp', domain: 'com.example', count: 1 }],
  opsets: [{ domain: 'ai.onnx', version: 17 }],
  warnings: onnxGraph.warnings,
}
const imported = { model_id: 'onnx_1', graph, report }
const samples = [
  { name: 'mlp', title: 'Tiny MLP', description: 'Gemm → ReLU → Gemm → Softmax.' },
  { name: 'cnn', title: 'Small CNN', description: 'Conv stages and a classifier.' },
]

const inspector = () => within(screen.getByRole('complementary', { name: 'Inspector' }))

beforeEach(() => {
  useExplorerStore.setState(initialExplorer, true)
  useModelStore.setState(initialModel, true)
  clearTensorCache()
  backend = fakeBackend()
  backend.use((method, path) => (method === 'GET' && path === '/import/onnx/samples' ? Response.json(samples) : undefined))
  backend.use((method, path) =>
    method === 'POST' && path.startsWith('/import/onnx') ? Response.json(imported, { status: 201 }) : undefined,
  )
  backend.use((_, path) =>
    path === '/models/onnx_1/tensors/t0'
      ? tensorResponse('t0', [8, 3, 3, 3], Array.from({ length: 216 }, (_, i) => (i - 100) / 100))
      : undefined,
  )
})

afterEach(() => vi.unstubAllGlobals())

async function importCnnSample() {
  render(<ModelExplorer />)
  await userEvent.click(await screen.findByRole('button', { name: /Small CNN/ }))
  await screen.findByRole('region', { name: 'Import warnings' })
}

describe('Model Explorer', () => {
  it('starts empty and lists samples from the backend', async () => {
    render(<ModelExplorer />)
    expect(screen.getByText('No model imported yet')).toBeInTheDocument()
    expect(await screen.findByRole('button', { name: /Tiny MLP/ })).toBeInTheDocument()
  })

  it('imports a sample and reports capabilities honestly', async () => {
    await importCnnSample()
    expect(backend.calls.some((c) => c.method === 'POST' && c.path === '/import/onnx/samples/cnn')).toBe(true)
    const report = screen.getByLabelText('Import report')
    expect(report).toHaveTextContent('Filesmall_cnn.onnx (4.0 KB)')
    expect(report).toHaveTextContent('Initializers1 (216 values)')
    expect(report).toHaveTextContent('Graph inspection: available')
    expect(report).toHaveTextContent('Running the model: not available — Running imported models is not supported yet.')
    expect(within(report).getByText('MyCustomOp')).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Import warnings' })).toHaveTextContent(/MyCustomOp/)
  })

  it('draws operators as labeled blocks, marking unsupported ones', async () => {
    await importCnnSample()
    const conv = screen.getByRole('button', { name: '/conv/Conv, Conv operator' })
    expect(conv).toHaveClass('is-block')
    const custom = screen.getByRole('button', { name: '/custom/MyOp, MyCustomOp operator, unsupported' })
    expect(custom).toHaveClass('is-unsupported')
    expect(document.querySelectorAll('circle[data-unit]')).toHaveLength(0)
    expect(screen.getByText(/Each block is one operator/)).toBeInTheDocument()
  })

  it('inspects an operator and loads its weights on request', async () => {
    await importCnnSample()
    const conv = screen.getByRole('button', { name: '/conv/Conv, Conv operator' })
    conv.focus()
    await userEvent.keyboard('{Enter}')

    const panel = screen.getByRole('complementary', { name: 'Inspector' })
    expect(inspector().getByRole('heading', { name: 'Conv' })).toBeInTheDocument()
    expect(panel).toHaveTextContent('kernel_shape = [3,3]')
    expect(panel).toHaveTextContent('[N × 3 × H × W]')
    expect(panel).toHaveTextContent('conv.weight [8 × 3 × 3 × 3] float · 216 values')
    expect(backend.calls.some((c) => c.path.includes('/tensors/'))).toBe(false)

    await userEvent.click(inspector().getByRole('button', { name: 'Show values' }))
    expect(await inspector().findByText(/min -1\.0000, max 1\.1500/)).toBeInTheDocument()
    expect(panel).toHaveTextContent('First 12 of 216 values')
  })

  it('explains unsupported operators and edge tensors', async () => {
    await importCnnSample()
    await userEvent.click(document.querySelector('[data-node-id="custom_1"] rect') as Element)
    expect(inspector().getByRole('note')).toHaveTextContent('Not a standard ONNX operator (domain com.example)')

    await userEvent.click(document.querySelector('[data-edge-id="e_conv_0_custom_1"]') as Element)
    expect(inspector().getByRole('heading', { name: 'conv_out' })).toBeInTheDocument()
    expect(screen.getByRole('complementary', { name: 'Inspector' })).toHaveTextContent('Shapeunknown')
  })

  it('uploads a chosen file as the raw request body', async () => {
    render(<ModelExplorer />)
    const file = new File([new Uint8Array([8, 7, 18, 1])], 'my model.onnx')
    await userEvent.upload(screen.getByLabelText('ONNX model file'), file)
    await screen.findByLabelText('Import report')
    const call = backend.calls.find((c) => c.path === '/import/onnx')
    expect(call?.query).toEqual({ filename: 'my model.onnx' })
    expect(call?.body).toBe(file)
  })

  it('rejects other file types and oversized files before uploading', async () => {
    render(<ModelExplorer />)
    const input = screen.getByLabelText('ONNX model file')
    await userEvent.upload(input, new File(['x'], 'weights.pt'), { applyAccept: false })
    expect(screen.getByRole('alert')).toHaveTextContent('Choose an .onnx file')

    const big = new File(['x'], 'big.onnx')
    Object.defineProperty(big, 'size', { value: 65 * 1024 * 1024 })
    await userEvent.upload(input, big)
    expect(screen.getByRole('alert')).toHaveTextContent('the limit is 64 MB')
    expect(backend.calls.some((c) => c.path === '/import/onnx')).toBe(false)
  })

  it('shows the server explanation when a file cannot be parsed', async () => {
    backend.use((method, path) =>
      method === 'POST' && path === '/import/onnx'
        ? Response.json(
            { error: { code: 'invalid_model', message: 'The file is not a valid ONNX model; it could not be parsed.', details: [] } },
            { status: 422 },
          )
        : undefined,
    )
    render(<ModelExplorer />)
    await userEvent.upload(screen.getByLabelText('ONNX model file'), new File(['garbage'], 'broken.onnx'))
    expect(await screen.findByRole('alert')).toHaveTextContent('it could not be parsed')
  })

  it('keeps its selection separate from the Architecture Studio', async () => {
    await importCnnSample()
    await userEvent.click(document.querySelector('[data-node-id="conv_0"] rect') as Element)
    await waitFor(() => expect(useExplorerStore.getState().selection).toEqual({ kind: 'layer', nodeId: 'conv_0' }))
    expect(useModelStore.getState().selection).toBeNull()
  })
})
