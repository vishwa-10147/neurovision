import type { Bounds } from './layout'

export interface ViewTransform {
  x: number
  y: number
  k: number
}

// Low enough that fit-to-view can show long imported graphs whole.
export const MIN_ZOOM = 0.05
export const MAX_ZOOM = 4
const FIT_PADDING = 40

export const clampZoom = (k: number) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, k))

/** Scale and center `bounds` inside a viewport of `width` × `height`. */
export function fitTransform(bounds: Bounds, width: number, height: number): ViewTransform {
  const w = Math.max(bounds.maxX - bounds.minX, 1)
  const h = Math.max(bounds.maxY - bounds.minY, 1)
  const k = clampZoom(
    Math.min((width - FIT_PADDING * 2) / w, (height - FIT_PADDING * 2) / h, 1.5),
  )
  return centerAt(bounds, width, height, k)
}

/** Natural size (zoom 1) centered in the viewport. */
export function resetTransform(bounds: Bounds, width: number, height: number): ViewTransform {
  return centerAt(bounds, width, height, 1)
}

function centerAt(bounds: Bounds, width: number, height: number, k: number): ViewTransform {
  const cx = (bounds.minX + bounds.maxX) / 2
  const cy = (bounds.minY + bounds.maxY) / 2
  return { x: width / 2 - cx * k, y: height / 2 - cy * k, k }
}

/** Zoom by `factor` keeping the screen point (px, py) fixed. */
export function zoomAround(t: ViewTransform, factor: number, px: number, py: number): ViewTransform {
  const k = clampZoom(t.k * factor)
  const ratio = k / t.k
  return { k, x: px - (px - t.x) * ratio, y: py - (py - t.y) * ratio }
}
