/**
 * Compact display for measured values. Very large or very small non-zero
 * values use exponent notation, so a tiny gradient never reads as exactly 0.
 */
export function formatValue(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return String(v)
  const magnitude = Math.abs(v)
  if (magnitude >= 1000 || (magnitude > 0 && magnitude < 10 ** -digits)) return v.toExponential(2)
  return v.toFixed(digits)
}

/**
 * Fill for a measured value, scaled by the largest magnitude in its group:
 * blue for positive, orange for negative, neutral at zero.
 */
export function activationFill(value: number, maxAbs: number): string {
  const strength = maxAbs > 0 ? Math.round((Math.abs(value) / maxAbs) * 100) : 0
  const hue = value >= 0 ? 'var(--act-pos)' : 'var(--act-neg)'
  return `color-mix(in srgb, ${hue} ${strength}%, var(--act-zero))`
}
