import { useState } from 'react'
import { SECTIONS, type SectionId } from './app/sections'
import { HealthIndicator } from './components/HealthIndicator'
import { ArchitectureStudio } from './features/architecture-studio/ArchitectureStudio'
import { Dashboard } from './features/dashboard/Dashboard'
import { PlannedSection } from './features/planned/PlannedSection'
import { useBackendHealth } from './hooks/useBackendHealth'

export default function App() {
  const [active, setActive] = useState<SectionId>('dashboard')
  const { health, recheck } = useBackendHealth()
  const section = SECTIONS.find((s) => s.id === active) ?? SECTIONS[0]

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">NeuroVision</div>
        <nav aria-label="Main">
          <ul>
            {SECTIONS.map(({ id, label, icon: Icon, milestone }) => (
              <li key={id}>
                <button
                  type="button"
                  className="nav-item"
                  aria-current={id === active ? 'page' : undefined}
                  onClick={() => setActive(id)}
                >
                  <Icon size={18} aria-hidden="true" />
                  <span>{label}</span>
                  {milestone && <span className="nav-tag">planned</span>}
                </button>
              </li>
            ))}
          </ul>
        </nav>
        <HealthIndicator health={health} />
      </aside>
      <main className="content">
        {section.id === 'dashboard' ? (
          <Dashboard health={health} onRecheck={recheck} />
        ) : section.id === 'studio' ? (
          <ArchitectureStudio />
        ) : (
          <PlannedSection section={section} />
        )}
      </main>
    </div>
  )
}
