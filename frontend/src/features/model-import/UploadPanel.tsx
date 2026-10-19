import { FileUp, Sparkles } from 'lucide-react'
import { useEffect, useId, useRef, useState, type DragEvent } from 'react'
import { importSample, listSamples, uploadOnnx } from '../../api/imports'
import { ErrorState, LoadingState } from '../../components/states/StateViews'
import { checkModelFile, MAX_UPLOAD_BYTES, type ImportResponse, type SampleModel } from '../../types/imports'

type Status = { state: 'idle' } | { state: 'importing'; label: string } | { state: 'error'; error: unknown }

export function UploadPanel({ onImported }: { onImported: (result: ImportResponse) => void }) {
  const [status, setStatus] = useState<Status>({ state: 'idle' })
  const [dragging, setDragging] = useState(false)
  const [samples, setSamples] = useState<SampleModel[] | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const id = useId()

  useEffect(() => {
    const controller = new AbortController()
    listSamples(controller.signal).then(setSamples, () => setSamples([]))
    return () => controller.abort()
  }, [])

  const run = async (label: string, task: () => Promise<ImportResponse>) => {
    setStatus({ state: 'importing', label })
    try {
      onImported(await task())
      setStatus({ state: 'idle' })
    } catch (error) {
      setStatus({ state: 'error', error })
    }
  }

  const importFile = (file: File | undefined) => {
    if (!file) return
    const problem = checkModelFile(file)
    if (problem) {
      setStatus({ state: 'error', error: new Error(problem) })
      return
    }
    void run(`Importing ${file.name}…`, () => uploadOnnx(file))
  }

  const onDrop = (event: DragEvent) => {
    event.preventDefault()
    setDragging(false)
    importFile(event.dataTransfer.files[0])
  }

  const busy = status.state === 'importing'
  return (
    <section className="editor upload-panel" aria-labelledby={`${id}-title`}>
      <h2 id={`${id}-title`}>Import a model</h2>
      <div
        className={`drop-zone${dragging ? ' is-dragging' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <FileUp size={22} aria-hidden="true" />
        <p>Drop an .onnx file here, or</p>
        <button type="button" className="button" disabled={busy} onClick={() => inputRef.current?.click()}>
          Choose a file
        </button>
        <input
          ref={inputRef}
          id={`${id}-file`}
          className="visually-hidden"
          type="file"
          accept=".onnx"
          aria-label="ONNX model file"
          onChange={(e) => {
            importFile(e.target.files?.[0])
            e.target.value = ''
          }}
        />
        <p className="muted">
          Up to {MAX_UPLOAD_BYTES / 1024 / 1024} MB. The file is parsed for inspection only; it is not
          run, and external data files it references are not read.
        </p>
      </div>

      {status.state === 'importing' && <LoadingState label={status.label} />}
      {status.state === 'error' && <ErrorState error={status.error} />}

      <div className="samples">
        <h3>
          <Sparkles size={14} aria-hidden="true" /> Or try a sample
        </h3>
        {samples === null ? (
          <LoadingState label="Loading samples…" />
        ) : samples.length === 0 ? (
          <p className="muted">Samples are unavailable while the backend is offline.</p>
        ) : (
          <ul>
            {samples.map((sample) => (
              <li key={sample.name}>
                <button
                  type="button"
                  className="sample-button"
                  disabled={busy}
                  onClick={() => void run(`Importing the ${sample.title} sample…`, () => importSample(sample.name))}
                >
                  <strong>{sample.title}</strong>
                  <span className="muted">{sample.description}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  )
}
