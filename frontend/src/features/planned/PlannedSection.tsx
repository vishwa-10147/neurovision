import type { Section } from '../../app/sections'

export function PlannedSection({ section }: { section: Section }) {
  return (
    <section aria-labelledby="planned-title" className="page">
      <h1 id="planned-title">{section.label}</h1>
      <div className="card empty">
        <span className="badge">Not built yet · {section.milestone}</span>
        <p>{section.summary}</p>
        <p className="muted">This section has no working functionality yet.</p>
      </div>
    </section>
  )
}
