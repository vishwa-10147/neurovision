import { beforeEach, describe, expect, it } from 'vitest'
import { denseGraph } from '../test/denseGraph'
import { EXAMPLE_CONFIG } from '../types/architecture'
import { isSelectionValid, useModelStore } from './modelStore'

const graph = denseGraph(EXAMPLE_CONFIG) // 2 -> 8 -> 6 -> 2
const initial = useModelStore.getState()

beforeEach(() => useModelStore.setState(initial, true))

describe('selection validity', () => {
  it.each([
    [{ kind: 'layer', nodeId: 'hidden_1' }, true],
    [{ kind: 'layer', nodeId: 'hidden_9' }, false],
    [{ kind: 'unit', nodeId: 'hidden_1', index: 7 }, true],
    [{ kind: 'unit', nodeId: 'hidden_1', index: 8 }, false],
    [{ kind: 'connection', edgeId: 'e_hidden_1_hidden_2', sourceIndex: 7, targetIndex: 5 }, true],
    [{ kind: 'connection', edgeId: 'e_hidden_1_hidden_2', sourceIndex: 7, targetIndex: 6 }, false],
    [{ kind: 'connection', edgeId: 'e_hidden_1_hidden_2', sourceIndex: null, targetIndex: null }, true],
    [{ kind: 'connection', edgeId: 'missing', sourceIndex: null, targetIndex: null }, false],
  ] as const)('%j → %s', (selection, expected) => {
    expect(isSelectionValid(selection, graph)).toBe(expected)
  })
})

describe('setModel', () => {
  it('keeps a selection that still exists in the new graph', () => {
    useModelStore.getState().select({ kind: 'unit', nodeId: 'hidden_1', index: 3 })
    useModelStore.getState().setModel('m2', graph)
    expect(useModelStore.getState().selection).toEqual({ kind: 'unit', nodeId: 'hidden_1', index: 3 })
  })

  it('clears a selection the new graph no longer contains', () => {
    useModelStore.getState().select({ kind: 'unit', nodeId: 'hidden_1', index: 7 })
    const narrower = denseGraph({ ...EXAMPLE_CONFIG, hidden_layers: [{ units: 4, activation: 'relu' }] })
    useModelStore.getState().setModel('m3', narrower)
    expect(useModelStore.getState().selection).toBeNull()
  })
})
