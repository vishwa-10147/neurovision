/** Numeric color scales for canvas drawing, from the theme's CSS variables. */

export type Rgb = [number, number, number]

export function parseColor(value: string, fallback: Rgb): Rgb {
  const hex = value.trim().match(/^#([0-9a-f]{6})$/i)
  if (!hex) return fallback
  const n = parseInt(hex[1], 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

export function themeRgb(name: string, fallback: Rgb): Rgb {
  if (typeof document === 'undefined') return fallback
  return parseColor(getComputedStyle(document.documentElement).getPropertyValue(name), fallback)
}

const mix = (a: Rgb, b: Rgb, t: number): Rgb => [
  Math.round(a[0] + (b[0] - a[0]) * t),
  Math.round(a[1] + (b[1] - a[1]) * t),
  Math.round(a[2] + (b[2] - a[2]) * t),
]

export type Scale =
  /** Signed values: orange below zero, blue above, neutral at zero, ±limit at full color. */
  | { kind: 'diverging'; limit: number }
  /** Values in [min, max], from neutral to blue (e.g. attention probabilities). */
  | { kind: 'sequential'; min: number; max: number }

export interface Palette {
  negative: Rgb
  zero: Rgb
  positive: Rgb
  missing: Rgb
}

export function currentPalette(): Palette {
  return {
    negative: themeRgb('--act-neg', [255, 138, 76]),
    zero: themeRgb('--act-zero', [42, 50, 71]),
    positive: themeRgb('--act-pos', [79, 157, 255]),
    missing: themeRgb('--bad', [255, 107, 107]),
  }
}

/** The color for one value on a scale; null (NaN/∞) gets the "missing" color. */
export function scaleColor(value: number | null, scale: Scale, palette: Palette): Rgb {
  if (value === null || !Number.isFinite(value)) return palette.missing
  if (scale.kind === 'diverging') {
    const t = scale.limit > 0 ? Math.min(1, Math.abs(value) / scale.limit) : 0
    return mix(palette.zero, value < 0 ? palette.negative : palette.positive, t)
  }
  const span = scale.max - scale.min
  const t = span > 0 ? Math.min(1, Math.max(0, (value - scale.min) / span)) : 0
  return mix(palette.zero, palette.positive, t)
}
