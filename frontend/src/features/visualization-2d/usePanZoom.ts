import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent, type RefObject } from 'react'
import type { Bounds } from './layout'
import { fitTransform, resetTransform, zoomAround, type ViewTransform } from './viewport'

// Used until the element is measured (and in test environments without layout).
const FALLBACK_SIZE = { width: 800, height: 500 }
const DRAG_THRESHOLD_PX = 3
const PAN_STEP_PX = 40

export function usePanZoom(svgRef: RefObject<SVGSVGElement | null>, bounds: Bounds) {
  const [size, setSize] = useState<{ width: number; height: number } | null>(null)
  const viewport = size ?? FALLBACK_SIZE

  // Refit whenever the content bounds change or the viewport is first measured.
  const fitKey = `${bounds.minX},${bounds.minY},${bounds.maxX},${bounds.maxY}|${size ? 'sized' : 'unsized'}`
  const [view, setView] = useState(() => ({
    key: fitKey,
    transform: fitTransform(bounds, viewport.width, viewport.height),
  }))
  if (view.key !== fitKey) {
    setView({ key: fitKey, transform: fitTransform(bounds, viewport.width, viewport.height) })
  }
  const transform = view.transform
  const setTransform = useCallback(
    (update: (t: ViewTransform) => ViewTransform) =>
      setView((v) => ({ ...v, transform: update(v.transform) })),
    [],
  )

  useEffect(() => {
    const svg = svgRef.current
    if (!svg || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      if (width > 0 && height > 0) setSize({ width, height })
    })
    observer.observe(svg)
    return () => observer.disconnect()
  }, [svgRef])

  // Wheel zoom needs a non-passive listener so the page does not scroll.
  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const rect = svg.getBoundingClientRect()
      const factor = Math.exp(-event.deltaY * 0.0015)
      setTransform((t) => zoomAround(t, factor, event.clientX - rect.left, event.clientY - rect.top))
    }
    svg.addEventListener('wheel', onWheel, { passive: false })
    return () => svg.removeEventListener('wheel', onWheel)
  }, [svgRef, setTransform])

  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null)
  const dragged = useRef(false)

  const onPointerDown = (event: PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return
    drag.current = { x: event.clientX, y: event.clientY, moved: false }
    dragged.current = false
  }
  const onPointerMove = (event: PointerEvent<SVGSVGElement>) => {
    const start = drag.current
    if (!start) return
    const dx = event.clientX - start.x
    const dy = event.clientY - start.y
    if (!start.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) return
    if (!start.moved) event.currentTarget.setPointerCapture?.(event.pointerId)
    start.moved = true
    drag.current = { x: event.clientX, y: event.clientY, moved: true }
    setTransform((t) => ({ ...t, x: t.x + dx, y: t.y + dy }))
  }
  const onPointerUp = () => {
    dragged.current = drag.current?.moved ?? false
    drag.current = null
  }

  const center = () => ({ x: viewport.width / 2, y: viewport.height / 2 })
  const zoomBy = (factor: number) => {
    const c = center()
    setTransform((t) => zoomAround(t, factor, c.x, c.y))
  }
  const fit = () => setTransform(() => fitTransform(bounds, viewport.width, viewport.height))
  const reset = () => setTransform(() => resetTransform(bounds, viewport.width, viewport.height))

  const onKeyDown = (event: KeyboardEvent<SVGSVGElement>) => {
    const pan = (dx: number, dy: number) => setTransform((t) => ({ ...t, x: t.x + dx, y: t.y + dy }))
    const actions: Record<string, () => void> = {
      '+': () => zoomBy(1.25),
      '=': () => zoomBy(1.25),
      '-': () => zoomBy(0.8),
      f: fit,
      '0': reset,
      ArrowLeft: () => pan(PAN_STEP_PX, 0),
      ArrowRight: () => pan(-PAN_STEP_PX, 0),
      ArrowUp: () => pan(0, PAN_STEP_PX),
      ArrowDown: () => pan(0, -PAN_STEP_PX),
    }
    // Only handle keys aimed at the canvas itself, not at a focused layer.
    if (event.target !== event.currentTarget) return
    const action = actions[event.key]
    if (action) {
      event.preventDefault()
      action()
    }
  }

  return {
    transform,
    zoomIn: () => zoomBy(1.25),
    zoomOut: () => zoomBy(0.8),
    fit,
    reset,
    /** True if the last pointer gesture was a pan, so it should not select. */
    wasDragged: () => dragged.current,
    svgHandlers: { onPointerDown, onPointerMove, onPointerUp, onPointerCancel: onPointerUp, onKeyDown },
  }
}
