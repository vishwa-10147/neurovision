import { useEffect, useMemo, useState } from 'react'

/** Whether this browser can create a WebGL context. */
export function hasWebGL(): boolean {
  try {
    const canvas = document.createElement('canvas')
    return Boolean(canvas.getContext('webgl2') ?? canvas.getContext('webgl'))
  } catch {
    return false
  }
}

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia?.(query).matches ?? false)
  useEffect(() => {
    const list = window.matchMedia?.(query)
    if (!list) return
    const onChange = () => setMatches(list.matches)
    list.addEventListener('change', onChange)
    return () => list.removeEventListener('change', onChange)
  }, [query])
  return matches
}

export const usePrefersReducedMotion = () => useMediaQuery('(prefers-reduced-motion: reduce)')

export interface ThemeColors {
  input: string
  hidden: string
  output: string
  connection: string
  accent: string
  surface: string
  muted: string
}

/** Scene colors from the CSS custom properties, updated when the theme flips. */
export function useThemeColors(): ThemeColors {
  const dark = useMediaQuery('(prefers-color-scheme: dark)')
  // Re-read the CSS variables whenever the color scheme flips.
  return useMemo(() => readColors(dark), [dark])
}

// `_dark` only keys the memo; the values come from the active stylesheet.
function readColors(_dark: boolean): ThemeColors {
  const style = getComputedStyle(document.documentElement)
  const get = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback
  return {
    input: get('--unit-input', '#4fc3d9'),
    hidden: get('--unit-hidden', '#9b8cf2'),
    output: get('--unit-output', '#f0a35e'),
    connection: get('--connection', '#3a4766'),
    accent: get('--accent', '#7aa2ff'),
    surface: get('--surface', '#121826'),
    muted: get('--muted', '#94a0b8'),
  }
}
