import type { Section } from '../../app/sections'
import { EmptyState } from '../../components/states/StateViews'

export function PlannedSection({ section }: { section: Section }) {
  return (
    <section aria-labelledby="planned-title" className="page">
      <h1 id="planned-title">{section.label}</h1>
      <div className="card">
        <span className="badge">Not built yet · {section.milestone}</span>
        <EmptyState title={section.summary}>
          This section has no working functionality yet.
        </EmptyState>
      </div>
    </section>
  )
}
