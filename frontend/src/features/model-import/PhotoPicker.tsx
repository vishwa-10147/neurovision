import { useState } from 'react'
import type { Draft, Normalization, Photo } from './runInputs'

/**
 * Pick a photo for an image input. It is decoded in the browser, scaled to
 * cover height × width and center-cropped (like torchvision's Resize +
 * CenterCrop); only the resulting pixels are sent, as the run's values.
 */
export function PhotoPicker({
  inputName,
  size,
  draft,
  onChange,
}: {
  inputName: string
  size: [number, number] | null
  draft: Draft
  onChange: (changes: Partial<Draft>) => void
}) {
  const [preview, setPreview] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  if (!size) return null
  const [height, width] = size

  const choose = async (file: File | undefined) => {
    if (!file) return
    setError(null)
    try {
      const bitmap = await createImageBitmap(file)
      const canvas = document.createElement('canvas')
      canvas.width = width
      canvas.height = height
      const context = canvas.getContext('2d')
      if (!context) throw new Error('no 2D canvas')
      const scale = Math.max(width / bitmap.width, height / bitmap.height)
      const w = bitmap.width * scale
      const h = bitmap.height * scale
      context.drawImage(bitmap, (width - w) / 2, (height - h) / 2, w, h)
      bitmap.close()
      const photo: Photo = { name: file.name, width, height, rgba: context.getImageData(0, 0, width, height).data }
      setPreview(canvas.toDataURL())
      onChange({ photo })
    } catch {
      setError('This file could not be read as an image.')
      onChange({ photo: null })
    }
  }

  return (
    <div className="photo-picker">
      <label className="field">
        <span>
          Photo, cropped to {width} × {height}
        </span>
        <input
          type="file"
          accept="image/*"
          aria-label={`Photo for ${inputName}`}
          onChange={(e) => void choose(e.target.files?.[0])}
        />
      </label>
      <label className="field">
        <span>Scaling</span>
        <select
          value={draft.normalization}
          aria-label={`Scaling for ${inputName}`}
          onChange={(e) => onChange({ normalization: e.target.value as Normalization })}
        >
          <option value="imagenet">ImageNet mean/std (torchvision models)</option>
          <option value="unit">0 to 1</option>
        </select>
      </label>
      {error && <p className="field-error">{error}</p>}
      {preview && draft.photo && (
        <figure className="photo-preview">
          <img src={preview} alt={`${draft.photo.name} as fed to the model`} width={96} height={96} />
          <figcaption className="muted">{draft.photo.name}</figcaption>
        </figure>
      )}
    </div>
  )
}
