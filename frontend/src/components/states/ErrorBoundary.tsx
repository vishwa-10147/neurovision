import { Component, type ReactNode } from 'react'

interface Props {
  /** Rendered instead of the children after they throw. */
  fallback: (error: unknown, reset: () => void) => ReactNode
  children: ReactNode
}

/** Contains a render error to one part of the page instead of blanking the app. */
export class ErrorBoundary extends Component<Props, { error: unknown }> {
  state = { error: null as unknown }

  static getDerivedStateFromError(error: unknown) {
    return { error }
  }

  componentDidCatch(error: unknown) {
    console.error('View crashed:', error)
  }

  render() {
    if (this.state.error !== null) {
      return this.props.fallback(this.state.error, () => this.setState({ error: null }))
    }
    return this.props.children
  }
}
