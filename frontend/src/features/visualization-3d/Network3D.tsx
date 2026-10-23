import { Canvas, type ThreeEvent } from '@react-three/fiber'
import { Minus, Plus, RotateCcw } from 'lucide-react'
import { useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { useModelView } from '../../app/modelView'
import { showsUnits } from '../../lib/graph'
import type { Selection } from '../../stores/modelStore'
import type { ModelGraph } from '../../types/graph'
import { CameraRig, type CameraCommands } from './CameraRig'
import { usePrefersReducedMotion, useThemeColors } from './environment'
import { highlightFor, MAX_3D_CONNECTIONS, MAX_3D_LABELS, MAX_3D_UNITS, planScene } from './scene'
import { Connections, LayerLinks, Layers, Units } from './SceneObjects'

const FOV = 45
/** Above this many layers, layer selection uses a dropdown instead of chips. */
const MAX_LAYER_CHIPS = 12
const DRAG_THRESHOLD_PX = 4
const ORBIT_STEP = Math.PI / 24
const PAN_STEP = 0.05

interface Network3DProps {
  graph: ModelGraph
  overlay?: string | null
  /** One measured value per operator; shaded blocks, as in the 2D view. */
  blockValues?: Record<string, number> | null
}

/** Default export so the studio can lazy-load three.js only when 3D is opened. */
export default function Network3D({ graph, overlay, blockValues = null }: Network3DProps) {
  const scene = useMemo(() => planScene(graph), [graph])
  const operators = !showsUnits(graph)
  const { selection, select } = useModelView()
  const highlight = useMemo(() => highlightFor(scene, selection), [scene, selection])
  const colors = useThemeColors()
  const reducedMotion = usePrefersReducedMotion()
  const commands = useRef<CameraCommands | null>(null)
  const [hovered, setHovered] = useState<number | null>(null)

  // Orbit drags end with a click event; ignore clicks that were really drags.
  const pointerStart = useRef<{ x: number; y: number } | null>(null)
  const wasDrag = (event: { clientX: number; clientY: number }) => {
    const start = pointerStart.current
    return start !== null && Math.hypot(event.clientX - start.x, event.clientY - start.y) > DRAG_THRESHOLD_PX
  }
  const pick = (next: Selection, event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    if (!wasDrag(event.nativeEvent)) select(next)
  }

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget) return
    const c = commands.current
    if (!c) return
    const actions: Record<string, () => void> = {
      ArrowLeft: () => (event.shiftKey ? c.pan(PAN_STEP, 0) : c.orbit(ORBIT_STEP, 0)),
      ArrowRight: () => (event.shiftKey ? c.pan(-PAN_STEP, 0) : c.orbit(-ORBIT_STEP, 0)),
      ArrowUp: () => (event.shiftKey ? c.pan(0, -PAN_STEP) : c.orbit(0, ORBIT_STEP)),
      ArrowDown: () => (event.shiftKey ? c.pan(0, PAN_STEP) : c.orbit(0, -ORBIT_STEP)),
      '+': () => c.zoom(1.25),
      '=': () => c.zoom(1.25),
      '-': () => c.zoom(0.8),
      '0': () => c.reset(),
      Escape: () => select(null),
    }
    const action = actions[event.key]
    if (action) {
      event.preventDefault()
      action()
    }
  }

  const hoveredUnit = hovered !== null ? scene.units[hovered] : undefined
  const hoveredLabel = hoveredUnit
    ? `${scene.layerById.get(hoveredUnit.nodeId)?.name} · unit ${hoveredUnit.index + 1}`
    : null

  return (
    <div className="network3d">
      <div className="toolbar" role="toolbar" aria-label="3D view controls">
        <button type="button" className="icon-button" aria-label="Zoom in" title="Zoom in (+)" onClick={() => commands.current?.zoom(1.25)}>
          <Plus size={16} aria-hidden="true" />
        </button>
        <button type="button" className="icon-button" aria-label="Zoom out" title="Zoom out (−)" onClick={() => commands.current?.zoom(0.8)}>
          <Minus size={16} aria-hidden="true" />
        </button>
        <button type="button" className="icon-button" aria-label="Reset camera" title="Reset camera (0)" onClick={() => commands.current?.reset()}>
          <RotateCcw size={16} aria-hidden="true" />
        </button>
        <span className="toolbar-hint">Drag to orbit · right-drag or Shift+arrows to pan · scroll to zoom</span>
      </div>

      <div
        className="network3d-canvas"
        tabIndex={0}
        role="img"
        aria-label="3D network view. Arrow keys orbit, Shift plus arrow keys pan, plus and minus zoom, 0 resets the camera, Escape clears the selection. Use the layer buttons below to select layers."
        onKeyDown={onKeyDown}
        onPointerDown={(event: PointerEvent) => {
          pointerStart.current = { x: event.clientX, y: event.clientY }
        }}
        style={{ cursor: hovered !== null ? 'pointer' : undefined }}
      >
        <Canvas
          frameloop="demand"
          dpr={[1, 2]}
          camera={{ fov: FOV, position: [0, 0, 20] }}
          raycaster={{ params: { Line: { threshold: 0.06 }, Points: { threshold: 0.1 }, Mesh: {}, LOD: {}, Sprite: {} } }}
          onPointerMissed={(event) => {
            if (!wasDrag(event)) select(null)
          }}
        >
          <ambientLight intensity={0.75} />
          <directionalLight position={[10, 15, 10]} intensity={1.6} />
          <Connections scene={scene} highlight={highlight} colors={colors} onPick={pick} />
          <LayerLinks scene={scene} highlight={highlight} colors={colors} onPick={pick} />
          <Units scene={scene} highlight={highlight} colors={colors} onPick={pick} onHover={setHovered} />
          <Layers scene={scene} highlight={highlight} colors={colors} onPick={pick} blockValues={blockValues} />
          <CameraRig scene={scene} ref={commands} reducedMotion={reducedMotion} />
        </Canvas>
        {hoveredLabel && <div className="network3d-hover">{hoveredLabel}</div>}
      </div>

      {overlay && <div className="network2d-overlay">{overlay}</div>}

      {scene.layers.length > MAX_LAYER_CHIPS ? (
        // Many layers: a dropdown keeps the keyboard path without crowding the canvas.
        <label className="layer-jump">
          <span className="muted">Select:</span>
          <select
            value={selection?.kind === 'layer' ? selection.nodeId : ''}
            onChange={(e) => select(e.target.value ? { kind: 'layer', nodeId: e.target.value } : null)}
          >
            <option value="">Choose a {operators ? 'node' : 'layer'}…</option>
            {scene.layers.map((layer) => (
              <option key={layer.nodeId} value={layer.nodeId}>
                {layer.label === layer.name ? layer.name : `${layer.label} — ${layer.name}`}
              </option>
            ))}
          </select>
        </label>
      ) : (
        <div className="layer-jump" role="group" aria-label="Select a layer">
          <span className="muted">Layers:</span>
          {scene.layers.map((layer) => {
            const pressed = selection?.kind === 'layer' && selection.nodeId === layer.nodeId
            return (
              <button
                key={layer.nodeId}
                type="button"
                className="chip"
                aria-pressed={pressed}
                onClick={() => select({ kind: 'layer', nodeId: layer.nodeId })}
              >
                {layer.name}
              </button>
            )
          })}
        </div>
      )}

      {operators ? (
        <div className="legend" aria-label="3D legend">
          <span className="legend-item"><span className="swatch-block legend-input-bg" aria-hidden="true" />Graph input</span>
          <span className="legend-item"><span className="swatch-block legend-hidden-bg" aria-hidden="true" />Operator</span>
          <span className="legend-item"><span className="swatch-block legend-output-bg" aria-hidden="true" />Graph output</span>
          <span className="legend-item"><span className="swatch-block swatch-warn" aria-hidden="true" />Unsupported operator</span>
          <p className="legend-note">
            Each block is one operator and each link a tensor between operators. Labels show
            operator types{scene.layers.length > MAX_3D_LABELS ? ' (only for the selection in graphs this large)' : ''}.
            {blockValues && Object.keys(blockValues).length > 0
              ? 'Shaded blocks carry measured values, from neutral to blue, on the same scale as the 2D view.'
              : 'Shown for inspection only: values are not computed here; the 2D view colors operators after a run.'}
          </p>
        </div>
      ) : (
      <div className="legend" aria-label="3D legend">
        <span className="legend-item"><span className="swatch legend-input-bg" aria-hidden="true" />Input unit</span>
        <span className="legend-item"><span className="swatch legend-hidden-bg" aria-hidden="true" />Hidden unit</span>
        <span className="legend-item"><span className="swatch legend-output-bg" aria-hidden="true" />Output unit</span>
        <span className="legend-item"><span className="swatch swatch-line" aria-hidden="true" />Selected / related (larger, brighter; everything else dims)</span>
        <p className="legend-note">
          {scene.detail.unitsAsBlocks
            ? `${scene.detail.unitCount.toLocaleString()} units exceed the 3D limit of ${MAX_3D_UNITS.toLocaleString()}, so layers are drawn as blocks and edges as layer links. `
            : scene.detail.aggregatedConnections && scene.detail.connectionCount > MAX_3D_CONNECTIONS
              ? `${scene.detail.connectionCount.toLocaleString()} connections exceed the 3D limit of ${MAX_3D_CONNECTIONS.toLocaleString()}, so each layer pair is drawn as one link. `
              : scene.detail.aggregatedConnections
                ? 'Some layer sizes are unknown, so edges are drawn as layer links. '
                : `Lines show which units connect (${scene.detail.connectionCount.toLocaleString()} total). `}
          Colors show layer roles only, not weights or activations. Forward-pass values appear in
          the 2D view and the inspector; select a connection to see its weight.
        </p>
      </div>
      )}
    </div>
  )
}
