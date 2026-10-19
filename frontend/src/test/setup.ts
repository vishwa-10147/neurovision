import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// jsdom has no WebGL (or 2D canvas); report "unsupported" quietly instead of
// logging "not implemented" errors. Tests that need a context stub it.
HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext

afterEach(() => {
  cleanup()
})
