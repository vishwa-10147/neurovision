import { Html } from '@react-three/drei'
import { useThree, type ThreeEvent } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { Selection } from '../../stores/modelStore'
import type { ThemeColors } from './environment'
import { LABEL_CLEARANCE, MAX_3D_LABELS, UNIT_RADIUS, type Highlight3D, type Scene3D } from './scene'

type Pick = (selection: Selection, event: ThreeEvent<MouseEvent>) => void

const DIM_TOWARD_SURFACE = 0.65

function roleColor(colors: ThemeColors, role: 'input' | 'hidden' | 'output') {
  return colors[role]
}

/** All units in one instanced mesh: one draw call regardless of unit count. */
export function Units({
  scene,
  highlight,
  colors,
  onPick,
  onHover,
}: {
  scene: Scene3D
  highlight: Highlight3D
  colors: ThemeColors
  onPick: Pick
  onHover: (unitId: number | null) => void
}) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const invalidate = useThree((s) => s.invalidate)

  useLayoutEffect(() => {
    const target = mesh.current
    if (!target) return
    const matrix = new THREE.Matrix4()
    const color = new THREE.Color()
    const surface = new THREE.Color(colors.surface)
    const accent = new THREE.Color(colors.accent)
    scene.units.forEach((unit, id) => {
      const emphasis = highlight.units.get(id)
      const scale = emphasis === 'selected' ? 1.45 : emphasis === 'related' ? 1.15 : 1
      matrix.makeScale(scale, scale, scale).setPosition(...unit.position)
      target.setMatrixAt(id, matrix)
      color.set(roleColor(colors, unit.role))
      if (emphasis === 'selected') color.lerp(accent, 0.55)
      else if (highlight.active && !emphasis) color.lerp(surface, DIM_TOWARD_SURFACE)
      target.setColorAt(id, color)
    })
    target.instanceMatrix.needsUpdate = true
    if (target.instanceColor) target.instanceColor.needsUpdate = true
    target.computeBoundingSphere()
    invalidate()
  }, [scene, highlight, colors, invalidate])

  if (scene.units.length === 0) return null
  return (
    <instancedMesh
      // Recreate when the instance count changes; InstancedMesh capacity is fixed.
      key={scene.units.length}
      ref={mesh}
      args={[undefined, undefined, scene.units.length]}
      onClick={(event) => {
        if (event.instanceId === undefined) return
        const unit = scene.units[event.instanceId]
        onPick({ kind: 'unit', nodeId: unit.nodeId, index: unit.index }, event)
      }}
      onPointerMove={(event) => {
        event.stopPropagation()
        onHover(event.instanceId ?? null)
      }}
      onPointerOut={() => onHover(null)}
    >
      <sphereGeometry args={[UNIT_RADIUS, 16, 12]} />
      <meshStandardMaterial roughness={0.45} metalness={0.05} />
    </instancedMesh>
  )
}

/** Unit-to-unit connections as one LineSegments object with vertex colors. */
export function Connections({
  scene,
  highlight,
  colors,
  onPick,
}: {
  scene: Scene3D
  highlight: Highlight3D
  colors: ThemeColors
  onPick: Pick
}) {
  const connections = scene.connections
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    if (!connections) return g
    const positions = new Float32Array(connections.length * 6)
    connections.forEach((c, i) => {
      positions.set(scene.units[c.source].position, i * 6)
      positions.set(scene.units[c.target].position, i * 6 + 3)
    })
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(connections.length * 6), 3))
    return g
  }, [connections, scene.units])
  useLayoutEffect(() => () => geometry.dispose(), [geometry])

  const invalidate = useThree((s) => s.invalidate)
  useLayoutEffect(() => {
    if (!connections) return
    const attribute = geometry.getAttribute('color') as THREE.BufferAttribute
    const base = new THREE.Color(colors.connection)
    const dim = base.clone().lerp(new THREE.Color(colors.surface), 0.6)
    const accent = new THREE.Color(colors.accent)
    const related = accent.clone().lerp(base, 0.35)
    connections.forEach((_, i) => {
      const emphasis = highlight.connections.get(i)
      const c = emphasis === 'selected' ? accent : emphasis === 'related' ? related : highlight.active ? dim : base
      attribute.setXYZ(i * 2, c.r, c.g, c.b)
      attribute.setXYZ(i * 2 + 1, c.r, c.g, c.b)
    })
    attribute.needsUpdate = true
    invalidate()
  }, [connections, geometry, highlight, colors, invalidate])

  if (!connections || connections.length === 0) return null
  return (
    <lineSegments
      geometry={geometry}
      onClick={(event) => {
        if (event.index === undefined) return
        const c = connections[Math.floor(event.index / 2)]
        if (!c) return
        onPick({ kind: 'connection', edgeId: c.edgeId, sourceIndex: c.sourceIndex, targetIndex: c.targetIndex }, event)
      }}
    >
      <lineBasicMaterial vertexColors />
    </lineSegments>
  )
}

/** Layer-level links: always clickable, drawn prominently when aggregated. */
export function LayerLinks({
  scene,
  highlight,
  colors,
  onPick,
}: {
  scene: Scene3D
  highlight: Highlight3D
  colors: ThemeColors
  onPick: Pick
}) {
  if (!scene.detail.aggregatedConnections) return null
  return (
    <>
      {scene.layerLinks.map((link) => {
        const emphasis = highlight.links.get(link.edgeId)
        const length = Math.hypot(link.to[0] - link.from[0], link.to[1] - link.from[1], link.to[2] - link.from[2])
        const mid: [number, number, number] = [
          (link.from[0] + link.to[0]) / 2,
          (link.from[1] + link.to[1]) / 2,
          (link.from[2] + link.to[2]) / 2,
        ]
        const direction = new THREE.Vector3(...link.to).sub(new THREE.Vector3(...link.from)).normalize()
        const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction)
        return (
          <mesh
            key={link.edgeId}
            position={mid}
            quaternion={quaternion}
            onClick={(event) =>
              onPick({ kind: 'connection', edgeId: link.edgeId, sourceIndex: null, targetIndex: null }, event)
            }
          >
            <cylinderGeometry args={[emphasis === 'selected' ? 0.22 : 0.14, emphasis === 'selected' ? 0.22 : 0.14, length, 10]} />
            <meshStandardMaterial
              color={emphasis ? colors.accent : colors.connection}
              transparent
              opacity={emphasis ? 0.9 : highlight.active ? 0.3 : 0.6}
            />
          </mesh>
        )
      })}
    </>
  )
}

/** Labels, selection frames, and blocks for layers without individual units. */
export function Layers({
  scene,
  highlight,
  colors,
  onPick,
}: {
  scene: Scene3D
  highlight: Highlight3D
  colors: ThemeColors
  onPick: Pick
}) {
  // Hundreds of DOM labels are unreadable and slow; then label only the selection.
  const labelAll = scene.layers.length <= MAX_3D_LABELS
  return (
    <>
      {scene.layers.map((layer) => {
        const selected = highlight.layers.has(layer.nodeId)
        const frame: [number, number, number] = [
          layer.size[0] + 0.4,
          layer.size[1] + 0.4,
          layer.size[2] + 0.4,
        ]
        return (
          <group key={layer.nodeId} position={layer.center}>
            {layer.unitIds === null && (
              <mesh onClick={(event) => onPick({ kind: 'layer', nodeId: layer.nodeId }, event)}>
                <boxGeometry args={layer.size} />
                <meshStandardMaterial
                  color={layer.unsupported ? colors.warn : roleColor(colors, layer.role)}
                  transparent
                  opacity={highlight.active && !selected ? 0.35 : 0.85}
                />
              </mesh>
            )}
            {selected && (
              <lineSegments>
                <edgesGeometry args={[new THREE.BoxGeometry(...frame)]} />
                <lineBasicMaterial color={colors.accent} />
              </lineSegments>
            )}
            {(labelAll || selected) && (
              <Html
                position={[0, layer.size[1] / 2 + LABEL_CLEARANCE, 0]}
                center
                className="label3d"
                zIndexRange={[10, 0]}
              >
                {layer.label}
              </Html>
            )}
          </group>
        )
      })}
    </>
  )
}
