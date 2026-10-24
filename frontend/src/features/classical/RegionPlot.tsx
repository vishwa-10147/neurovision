/**
 * Decision regions and points on one canvas. Region cells are colored by the
 * predicted class, more opaque where the model is more confident.
 */
import { useEffect, useRef } from 'react'
import type { ClassicResult, Frame } from '../../api/classical'
import { classColor } from '../../lib/classColors'

const SIZE = 440

export function RegionPlot({ result, frame }: { result: ClassicResult; frame: Frame | null }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const pca = result.pca
  const projected = pca && !pca.mean ? pca.projected : null
  const points = projected ?? result.train
  const bounds = result.bounds ?? boundsOf(points.x, points.y)

  useEffect(() => {
    const context = canvas.current?.getContext('2d')
    if (!context) return
    const [x0, x1, y0, y1] = bounds
    const sx = (v: number) => ((v - x0) / (x1 - x0)) * SIZE
    const sy = (v: number) => SIZE - ((v - y0) / (y1 - y0)) * SIZE
    context.clearRect(0, 0, SIZE, SIZE)

    if (frame && frame.grid_class.length) {
      const n = result.grid
      // Integer cell edges: adjacent translucent cells must not overlap.
      const edge = (k: number) => Math.round((k * SIZE) / n)
      frame.grid_class.forEach((label, i) => {
        const row = Math.floor(i / n)
        const col = i % n
        context.fillStyle = classColor(label, 0.12 + 0.4 * frame.grid_confidence[i])
        // Row 0 is the bottom of the plot.
        const top = SIZE - edge(row + 1)
        context.fillRect(edge(col), top, edge(col + 1) - edge(col), SIZE - edge(row) - top)
      })
    }

    const dot = (x: number, y: number, label: number, hollow: boolean) => {
      context.beginPath()
      context.arc(sx(x), sy(y), 3.6, 0, Math.PI * 2)
      if (hollow) {
        context.strokeStyle = classColor(label)
        context.lineWidth = 1.6
        context.stroke()
      } else {
        context.fillStyle = classColor(label)
        context.fill()
        context.strokeStyle = 'rgba(0,0,0,0.35)'
        context.lineWidth = 0.6
        context.stroke()
      }
    }
    const colors = frame?.assignments ?? points.labels
    points.x.forEach((x, i) => dot(x, points.y[i], colors[i] ?? 0, false))
    if (!projected) result.test.x.forEach((x, i) => dot(x, result.test.y[i], result.test.labels[i], true))

    context.strokeStyle = 'rgba(0,0,0,0.85)'
    context.lineWidth = 1.4
    for (const [x, y] of result.support_vectors ?? []) {
      context.beginPath()
      context.arc(sx(x), sy(y), 6.5, 0, Math.PI * 2)
      context.stroke()
    }
    for (const [i, [x, y]] of (frame?.centroids ?? []).entries()) {
      context.strokeStyle = 'black'
      context.lineWidth = 4
      cross(context, sx(x), sy(y))
      context.strokeStyle = classColor(i)
      context.lineWidth = 2
      cross(context, sx(x), sy(y))
    }
    if (pca?.mean && pca.axes) {
      const [mx, my] = pca.mean
      pca.axes.forEach(([ax, ay], i) => {
        context.strokeStyle = i === 0 ? 'rgba(200,40,40,0.95)' : 'rgba(40,40,200,0.95)'
        context.lineWidth = 3
        context.beginPath()
        context.moveTo(sx(mx - 2 * ax), sy(my - 2 * ay))
        context.lineTo(sx(mx + 2 * ax), sy(my + 2 * ay))
        context.stroke()
      })
    }
  }, [result, frame, points, projected, bounds, pca])

  return (
    <canvas
      ref={canvas}
      width={SIZE}
      height={SIZE}
      className="region-plot"
      role="img"
      aria-label={describe(result, frame, Boolean(projected))}
    />
  )
}

function cross(context: CanvasRenderingContext2D, x: number, y: number) {
  context.beginPath()
  context.moveTo(x - 8, y - 8)
  context.lineTo(x + 8, y + 8)
  context.moveTo(x + 8, y - 8)
  context.lineTo(x - 8, y + 8)
  context.stroke()
}

function boundsOf(x: number[], y: number[]): [number, number, number, number] {
  const pad = (a: number[]) => (Math.max(...a) - Math.min(...a)) * 0.08 || 1
  return [Math.min(...x) - pad(x), Math.max(...x) + pad(x), Math.min(...y) - pad(y), Math.max(...y) + pad(y)]
}

function describe(result: ClassicResult, frame: Frame | null, projected: boolean): string {
  if (projected) return `${result.pca?.projected.x.length ?? 0} digits projected on the first two principal components`
  const parts = [`${result.train.x.length} training and ${result.test.x.length} test points`]
  if (frame?.grid_class.length) parts.push(`decision regions at ${frame.label}`)
  if (result.support_vectors) parts.push(`${result.support_vectors.length} support vectors ringed`)
  if (frame?.centroids) parts.push(`${frame.centroids.length} centroids`)
  return parts.join(', ')
}
