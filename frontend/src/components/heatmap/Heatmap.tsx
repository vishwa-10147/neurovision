import { useEffect, useRef, useState, type KeyboardEvent, type MouseEvent } from 'react'
import { currentPalette, scaleColor, type Scale } from '../../lib/colors'
import { formatValue } from '../../lib/values'

export interface HeatmapProps {
  values: (number | null)[][]
  scale: Scale
  /** Accessible description, e.g. "Head 2 attention, 7 by 7". */
  label: string
  /** Pixels per cell (canvas size = cells × cellSize). */
  cellSize: number
  /** Axis names for the readout, e.g. "query" and "key". */
  rowName?: string
  colName?: string
  /** Each cell covers stride × stride original positions (block average). */
  stride?: number
  /** Show a value readout and allow arrow-key navigation. */
  interactive?: boolean
}

/**
 * A 2D map drawn on a canvas, one rectangle per cell. With `interactive`,
 * hovering or using arrow keys shows the exact value of a cell.
 */
export function Heatmap({
  values,
  scale,
  label,
  cellSize,
  rowName = 'row',
  colName = 'column',
  stride = 1,
  interactive = false,
}: HeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rows = values.length
  const cols = values[0]?.length ?? 0
  const [cursor, setCursor] = useState<[number, number] | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')
    if (!canvas || !context) return
    const palette = currentPalette()
    context.clearRect(0, 0, canvas.width, canvas.height)
    values.forEach((row, r) =>
      row.forEach((value, c) => {
        const [red, green, blue] = scaleColor(value, scale, palette)
        context.fillStyle = `rgb(${red}, ${green}, ${blue})`
        context.fillRect(c * cellSize, r * cellSize, cellSize, cellSize)
      }),
    )
    if (interactive && cursor) {
      context.strokeStyle = 'white'
      context.lineWidth = 2
      context.strokeRect(cursor[1] * cellSize + 1, cursor[0] * cellSize + 1, cellSize - 2, cellSize - 2)
    }
  }, [values, scale, cellSize, cursor, interactive])

  const cellAt = (event: MouseEvent<HTMLCanvasElement>): [number, number] | null => {
    const rect = event.currentTarget.getBoundingClientRect()
    const scaleX = rect.width ? (cols * cellSize) / rect.width : 1
    const scaleY = rect.height ? (rows * cellSize) / rect.height : 1
    const r = Math.floor(((event.clientY - rect.top) * scaleY) / cellSize)
    const c = Math.floor(((event.clientX - rect.left) * scaleX) / cellSize)
    return r >= 0 && r < rows && c >= 0 && c < cols ? [r, c] : null
  }

  const onKeyDown = (event: KeyboardEvent<HTMLCanvasElement>) => {
    const moves: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    }
    const move = moves[event.key]
    if (!move) return
    event.preventDefault()
    const [r, c] = cursor ?? [0, 0]
    setCursor(
      cursor
        ? [Math.min(rows - 1, Math.max(0, r + move[0])), Math.min(cols - 1, Math.max(0, c + move[1]))]
        : [0, 0],
    )
  }

  const position = (index: number) =>
    stride > 1 ? `${index * stride}–${index * stride + stride - 1}` : String(index)
  const readout = cursor
    ? `${rowName} ${position(cursor[0])}, ${colName} ${position(cursor[1])}: ${
        values[cursor[0]][cursor[1]] === null ? 'NaN/∞' : formatValue(values[cursor[0]][cursor[1]] as number, 4)
      }${stride > 1 ? ' (block average)' : ''}`
    : null

  return (
    <div className="heatmap">
      <canvas
        ref={canvasRef}
        width={cols * cellSize}
        height={rows * cellSize}
        role="img"
        aria-label={label}
        tabIndex={interactive ? 0 : undefined}
        onMouseMove={interactive ? (e) => setCursor(cellAt(e)) : undefined}
        onKeyDown={interactive ? onKeyDown : undefined}
      />
      {interactive && (
        <p className="heatmap-readout mono" aria-live="polite">
          {readout ?? 'Hover the map, or focus it and use the arrow keys, to read values.'}
        </p>
      )}
    </div>
  )
}
