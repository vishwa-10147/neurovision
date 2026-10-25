import { FolderOpen, Save } from 'lucide-react'
import { useEffect, useState } from 'react'
import { listProjects, loadProject, saveProject, type ProjectSummary } from '../../api/projects'
import { useModelStore } from '../../stores/modelStore'

export function ProjectPanel() {
  const config = useModelStore((state) => state.config)
  const loadConfig = useModelStore((state) => state.loadConfig)
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [selected, setSelected] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  const refresh = async () => {
    try {
      setProjects((await listProjects()).projects)
    } catch {
      // The architecture editor remains useful when persistence is unavailable.
    }
  }

  useEffect(() => {
    const controller = new AbortController()
    listProjects(controller.signal)
      .then((result) => setProjects(result.projects))
      .catch(() => undefined)
    return () => controller.abort()
  }, [])

  const save = async () => {
    setBusy(true)
    setMessage(null)
    try {
      await saveProject(config)
      await refresh()
      setMessage('Project saved.')
    } catch {
      setMessage('Could not save this project. Check that the backend is running.')
    } finally {
      setBusy(false)
    }
  }

  const open = async () => {
    if (!selected) return
    setBusy(true)
    setMessage(null)
    try {
      const project = await loadProject(selected)
      loadConfig(project.config)
      setMessage(`Opened ${project.name}.`)
    } catch {
      setMessage('Could not open that project.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="card project-panel" aria-labelledby="projects-title">
      <div className="card-header">
        <div>
          <h2 id="projects-title">Projects</h2>
          <p className="muted">Versioned architecture files survive a backend restart.</p>
        </div>
        <button type="button" className="button" onClick={save} disabled={busy}>
          <Save size={16} aria-hidden="true" />
          Save
        </button>
      </div>
      <div className="project-panel-row">
        <label htmlFor="saved-project">Open saved project</label>
        <select id="saved-project" value={selected} onChange={(event) => setSelected(event.target.value)}>
          <option value="">Choose a project…</option>
          {projects.map((project) => (
            <option key={project.project_id} value={project.project_id}>
              {project.name}
            </option>
          ))}
        </select>
        <button type="button" className="button" onClick={open} disabled={!selected || busy}>
          <FolderOpen size={16} aria-hidden="true" />
          Open
        </button>
      </div>
      {message && <p className="muted" role="status">{message}</p>}
    </section>
  )
}
