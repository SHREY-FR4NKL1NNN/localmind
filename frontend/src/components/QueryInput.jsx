import { useRef, useState } from 'react'
import { postQuery, postQueryDecomposed } from '../api'

// Read a File into a raw base64 string (no data-URL prefix) plus a preview URL.
function readImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const dataUrl = String(reader.result)
      const base64 = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl
      resolve({ base64, dataUrl })
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

    // Streaming is always the decomposed flow; hand the request to StreamingPanel.
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

  return (
    <div className="card query-input">
      <h2 className="card__title">Ask LocalMind</h2>
      <textarea
        rows={4}
        value={query}
        placeholder="Type a query — LocalMind will classify and route it…"
        onChange={(e) => setQuery(e.target.value)}
        disabled={loading}
      />

      <div
        className={`dropzone${dragOver ? ' dropzone--over' : ''}`}
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
        onClick={() => fileRef.current?.click()}
      >
        {image ? (
          <div className="dropzone__preview">
            <img src={image.dataUrl} alt="upload preview" className="thumb" />
            <button
              type="button"
              className="btn btn--ghost"
              onClick={(e) => {
                e.stopPropagation()
                setImage(null)
                if (fileRef.current) fileRef.current.value = ''
              }}
            >
              Remove image
            </button>
          </div>
        ) : (
          <span className="dropzone__hint">
            Drop an image here or click to upload (routes to LLaVA vision expert)
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

      <div className="query-input__row">
        <button className="btn" onClick={handleSubmit} disabled={loading}>
          {loading ? 'Routing…' : streaming ? 'Stream' : 'Submit'}
        </button>
        <label className="toggle" title="Stream every expert's tokens live over SSE.">
          <input
            type="checkbox"
            checked={streaming}
            onChange={(e) => setStreaming(e.target.checked)}
            disabled={loading}
          />
          Use streaming
        </label>
        <label
          className="toggle"
          title="Split the query into sub-tasks, route each to its own expert in parallel, then synthesize."
        >
          <input
            type="checkbox"
            checked={decompose}
            onChange={(e) => setDecompose(e.target.checked)}
            disabled={loading || streaming}
          />
          Decompose (MoE)
        </label>
        {error && <span className="error-text">{error}</span>}
      </div>
    </div>
  )
}
