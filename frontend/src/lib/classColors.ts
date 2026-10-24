/**
 * Colors for class labels: a fixed 10-color qualitative palette (Tableau 10
 * order), so neighboring labels stay distinguishable; it repeats after 10.
 */
const PALETTE: [number, number, number][] = [
  [31, 119, 180],
  [214, 39, 40],
  [44, 160, 44],
  [148, 103, 189],
  [255, 127, 14],
  [23, 190, 207],
  [227, 119, 194],
  [140, 86, 75],
  [188, 189, 34],
  [127, 127, 127],
]

export function classColor(label: number, alpha = 1): string {
  const [r, g, b] = PALETTE[((label % PALETTE.length) + PALETTE.length) % PALETTE.length]
  return `rgb(${r} ${g} ${b} / ${alpha})`
}
