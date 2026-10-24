import { useState } from 'react'
import { SECTIONS, type SectionId } from './app/sections'
import { HealthIndicator } from './components/HealthIndicator'
import { ArchitectureStudio } from './features/architecture-studio/ArchitectureStudio'
import { ClassicalLab } from './features/classical/ClassicalLab'
import { Dashboard } from './features/dashboard/Dashboard'
import { LiveRuns } from './features/live-runs/LiveRuns'
import { NeuralLab } from './features/neural-lab/NeuralLab'
import { ModelExplorer } from './features/model-import/ModelExplorer'
import { PlannedSection } from './features/planned/PlannedSection'
import { TrainingLab } from './features/training-lab/TrainingLab'
import { useBackendHealth } from './hooks/useBackendHealth'
import { useLiveRunStore } from './stores/liveRunStore'

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
        ) : section.id === 'training' ? (
          <TrainingLab onOpenStudio={() => setActive('studio')} />
        ) : section.id === 'classical' ? (
          <ClassicalLab />
        ) : section.id === 'lab' ? (
          <NeuralLab
            onWatch={(runName) => {
              useLiveRunStore.getState().expect(runName)
              setActive('runs')
            }}
          />
        ) : section.id === 'runs' ? (
          <LiveRuns />
        ) : section.id === 'explorer' ? (
          <ModelExplorer />
        ) : (
          <PlannedSection section={section} />
        )}
      </main>
    </div>
  )
}
