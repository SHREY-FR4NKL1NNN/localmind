import { useRef, useState } from 'react'
import { postQuery, postQueryDecomposed } from '../api'

// Longest edge (px) the uploaded image is downscaled to before sending — keeps
// the base64 payload small and inference fast without losing useful detail.
const MAX_IMAGE_DIM = 1024

// Read a File and re-encode it to PNG via a canvas, returning a raw base64
// string (no data-URL prefix) plus a preview URL. The re-encode matters: the
// backend's image decoder (Ollama / stb_image) can't read WEBP/AVIF, so we
// normalise whatever the browser can display into PNG, which it always can.
function readImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const img = new Image()
      img.onload = () => {
        let { naturalWidth: w, naturalHeight: h } = img
        if (Math.max(w, h) > MAX_IMAGE_DIM) {
          const scale = MAX_IMAGE_DIM / Math.max(w, h)
          w = Math.round(w * scale)
          h = Math.round(h * scale)
        }
        const canvas = document.createElement('canvas')
        canvas.width = w
        canvas.height = h
        canvas.getContext('2d').drawImage(img, 0, 0, w, h)
        const pngDataUrl = canvas.toDataURL('image/png')
        resolve({ base64: pngDataUrl.split(',')[1], dataUrl: pngDataUrl })
      }
      img.onerror = () => reject(new Error('Could not decode image'))
      img.src = String(reader.result)
    }
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

export default function QueryInput({ onResult, onStream }) {
  const [query, setQuery] = useState('')
  const [decompose, setDecompose] = useState(false)
  const [streaming, setStreaming] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [image, setImage] = useState(null) // { base64, dataUrl }
  const [dragOver, setDragOver] = useState(false)
  const fileRef = useRef(null)

  async function handleFiles(files) {
    const file = files && files[0]
    if (!file || !file.type.startsWith('image/')) return
    try {
      setImage(await readImage(file))
    } catch {
      setError('Could not read that image.')
    }
  }

  async function handleSubmit() {
    const trimmed = query.trim()
    if (!trimmed || loading) return
    setError('')

    // Streaming is always the decomposed flow; hand the request to the hook.
    if (streaming) {
      onStream({ query: trimmed, imageBase64: image?.base64 ?? null, id: Date.now() })
      return
    }

    setLoading(true)
    try {
      const result = decompose
        ? await postQueryDecomposed(trimmed, image?.base64 ?? null)
        : await postQuery(trimmed)
      onResult(result)
      setQuery('')
    } catch {
      setError('Could not reach the backend. Is the API running on :8000?')
    } finally {
      setLoading(false)
    }
  }

  const empty = query.trim().length === 0

  return (
    <div className="card qinput">
      <textarea
        className="qinput__text"
        rows={3}
        value={query}
        placeholder="Ask anything — LocalMind routes to the right expert"
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit()
        }}
      />

      <div className="qinput__row">
        <div
          className={`dropzone${dragOver ? ' dropzone--over' : ''}${image ? ' dropzone--filled' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            handleFiles(e.dataTransfer.files)
          }}
          onClick={() => !image && fileRef.current?.click()}
        >
          {image ? (
            <>
              <img src={image.dataUrl} alt="upload preview" className="dropzone__thumb" />
              <button
                type="button"
                className="dropzone__clear"
                aria-label="Remove image"
                onClick={(e) => {
                  e.stopPropagation()
                  setImage(null)
                  if (fileRef.current) fileRef.current.value = ''
                }}
              >
                ×
              </button>
            </>
          ) : (
            <span className="dropzone__hint">
              <span className="dropzone__icon" aria-hidden>▦</span>
              Drop image
            </span>
          )}
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>

        <div className="qinput__controls">
          <button
            type="button"
            className={`stream-pill${streaming ? ' stream-pill--on' : ''}`}
            aria-pressed={streaming}
            onClick={() => setStreaming((v) => !v)}
          >
            <span className="stream-pill__dot" />
            Stream
          </button>

          <button
            className={`btn-route${loading ? ' btn-route--loading' : ''}`}
            onClick={handleSubmit}
            disabled={empty}
          >
            <span className="btn-route__label">Route query →</span>
          </button>
        </div>
      </div>

      {!streaming && (
        <label className="toggle toggle--inline" title="Split the query into sub-tasks and synthesize.">
          <input
            type="checkbox"
            checked={decompose}
            onChange={(e) => setDecompose(e.target.checked)}
          />
          Decompose (MoE)
        </label>
      )}

      {error && <span className="error-text">{error}</span>}
    </div>
  )
}
