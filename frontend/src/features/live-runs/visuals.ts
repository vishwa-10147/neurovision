/** Pure helpers for custom visuals, so they are unit-tested. */

/** Row-major values to rows. */
export function toRows(values: number[], rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, (_, r) => values.slice(r * cols, (r + 1) * cols))
}

/** Data bounds with a small margin; a zero span becomes 1 so points stay drawable. */
export function scatterBounds(x: number[], y: number[]) {
  const minX = Math.min(...x)
  const maxX = Math.max(...x)
  const minY = Math.min(...y)
  const maxY = Math.max(...y)
  return { minX, maxX, minY, maxY, spanX: maxX - minX || 1, spanY: maxY - minY || 1 }
}
