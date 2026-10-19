import { Canvas, type ThreeEvent } from '@react-three/fiber'
import { Minus, Plus, RotateCcw } from 'lucide-react'
import { useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { useModelStore, type Selection } from '../../stores/modelStore'
import type { ModelGraph } from '../../types/graph'
import { CameraRig, type CameraCommands } from './CameraRig'
import { usePrefersReducedMotion, useThemeColors } from './environment'
import { highlightFor, MAX_3D_CONNECTIONS, MAX_3D_UNITS, planScene } from './scene'
import { Connections, LayerLinks, Layers, Units } from './SceneObjects'

const FOV = 45
const DRAG_THRESHOLD_PX = 4
const ORBIT_STEP = Math.PI / 24
const PAN_STEP = 0.05

interface Network3DProps {
  graph: ModelGraph
  overlay?: string | null
}

/** Default export so the studio can lazy-load three.js only when 3D is opened. */
export default function Network3D({ graph, overlay }: Network3DProps) {
  const scene = useMemo(() => planScene(graph), [graph])
  const selection = useModelStore((s) => s.selection)
  const select = useModelStore((s) => s.select)
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
          <Layers scene={scene} highlight={highlight} colors={colors} onPick={pick} />
          <CameraRig scene={scene} ref={commands} reducedMotion={reducedMotion} />
        </Canvas>
        {hoveredLabel && <div className="network3d-hover">{hoveredLabel}</div>}
      </div>

      {overlay && <div className="network2d-overlay">{overlay}</div>}

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
          Color and thickness do not encode weights: weight values become available when PyTorch
          models are added (Milestone 4).
        </p>
      </div>
    </div>
  )
}
