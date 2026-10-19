import { OrbitControls } from '@react-three/drei'
import { useThree } from '@react-three/fiber'
import { useEffect, useImperativeHandle, useMemo, useRef, type ComponentRef, type Ref } from 'react'
import * as THREE from 'three'
import { defaultCamera, type Scene3D } from './scene'

type OrbitControlsImpl = ComponentRef<typeof OrbitControls>

export interface CameraCommands {
  zoom: (factor: number) => void
  orbit: (azimuth: number, polar: number) => void
  pan: (dx: number, dy: number) => void
  reset: () => void
}

/**
 * Orbit, zoom, and pan via drei's OrbitControls, plus imperative commands for
 * the toolbar and keyboard. Damping is nonessential motion, so it is turned
 * off when the user prefers reduced motion.
 */
export function CameraRig({
  scene,
  ref,
  reducedMotion,
}: {
  scene: Scene3D
  ref: Ref<CameraCommands>
  reducedMotion: boolean
}) {
  // Three.js objects are mutable by design; read them from the store when acting.
  const get = useThree((s) => s.get)
  const controls = useRef<OrbitControlsImpl>(null)
  // Whether the user has moved the camera since the last reset.
  const interacted = useRef(false)
  const size = useThree((s) => s.size)
  const radius = scene.bounds.radius

  const commands = useMemo((): CameraCommands => {
    const camera = () => get().camera as THREE.PerspectiveCamera
    const target = () => controls.current?.target ?? new THREE.Vector3()
    /** Apply a camera change; `moved` marks it as the user's own view. */
    const commit = (moved = true) => {
      interacted.current = moved
      controls.current?.update()
      get().invalidate()
    }
    return {
      reset: () => {
        const cam = camera()
        const { width, height } = get().size
        const view = defaultCamera(scene, cam.fov, height > 0 ? width / height : 1)
        cam.position.set(...view.position)
        // Clip planes follow the framing distance so the scene is never culled.
        const distance = cam.position.distanceTo(new THREE.Vector3(...view.target))
        cam.near = Math.max(0.01, (distance - radius) / 10)
        cam.far = distance + radius * 15
        cam.updateProjectionMatrix()
        cam.lookAt(...view.target)
        target().set(...view.target)
        commit(false)
      },
      zoom: (factor) => {
        const cam = camera()
        const offset = cam.position.clone().sub(target())
        const length = THREE.MathUtils.clamp(offset.length() / factor, radius * 0.2, radius * 12)
        cam.position.copy(target()).add(offset.setLength(length))
        commit()
      },
      orbit: (azimuth, polar) => {
        const cam = camera()
        const spherical = new THREE.Spherical().setFromVector3(cam.position.clone().sub(target()))
        spherical.theta += azimuth
        spherical.phi = THREE.MathUtils.clamp(spherical.phi + polar, 0.05, Math.PI - 0.05)
        cam.position.copy(target()).add(new THREE.Vector3().setFromSpherical(spherical))
        commit()
      },
      pan: (dx, dy) => {
        const cam = camera()
        const distance = cam.position.distanceTo(target())
        const right = new THREE.Vector3().setFromMatrixColumn(cam.matrix, 0).multiplyScalar(dx * distance)
        const up = new THREE.Vector3().setFromMatrixColumn(cam.matrix, 1).multiplyScalar(dy * distance)
        const shift = right.add(up)
        cam.position.add(shift)
        target().add(shift)
        commit()
      },
    }
  }, [get, scene, radius])

  useImperativeHandle(ref, () => commands, [commands])
  // Reframe whenever the scene changes (e.g. layers added or widened).
  useEffect(() => commands.reset(), [commands])
  // Refit on resize too, unless the user has already chosen their own view.
  useEffect(() => {
    if (!interacted.current) commands.reset()
  }, [commands, size.width, size.height])

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      onStart={() => {
        interacted.current = true
      }}
      enableDamping={!reducedMotion}
      dampingFactor={0.12}
      minDistance={radius * 0.2}
      maxDistance={radius * 12}
    />
  )
}
