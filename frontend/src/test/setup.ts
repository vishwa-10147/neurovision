import '@testing-library/jest-dom/vitest'
import { cleanup, configure } from '@testing-library/react'
import { afterEach } from 'vitest'

// findBy*/waitFor give up after 1 s by default. Lazy chunks such as Recharts
// can take longer to load on a busy machine, which made a few tests fail
// intermittently. Waits still resolve as soon as their condition holds.
configure({ asyncUtilTimeout: 5000 })

// jsdom has no WebGL (or 2D canvas); report "unsupported" quietly instead of
// logging "not implemented" errors. Tests that need a context stub it.
HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext

afterEach(() => {
  cleanup()
})
