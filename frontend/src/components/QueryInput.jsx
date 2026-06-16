import { useState } from 'react'
import { postQuery } from '../api'

export default function QueryInput({ onResult }) {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit() {
    const trimmed = query.trim()
    if (!trimmed || loading) return

    setLoading(true)
    setError('')
    try {
      const result = await postQuery(trimmed)
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
      <div className="query-input__row">
        <button className="btn" onClick={handleSubmit} disabled={loading}>
          {loading ? 'Routing…' : 'Submit'}
        </button>
        {error && <span className="error-text">{error}</span>}
      </div>
    </div>
  )
}
