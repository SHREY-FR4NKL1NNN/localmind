import { useState } from 'react'
import { postQuery, postQueryDecomposed } from '../api'

export default function QueryInput({ onResult }) {
  const [query, setQuery] = useState('')
  const [decompose, setDecompose] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit() {
    const trimmed = query.trim()
    if (!trimmed || loading) return

    setLoading(true)
    setError('')
    try {
      const result = decompose
        ? await postQueryDecomposed(trimmed)
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
      <div className="query-input__row">
        <button className="btn" onClick={handleSubmit} disabled={loading}>
          {loading ? 'Routing…' : 'Submit'}
        </button>
        <label className="toggle" title="Split the query into sub-tasks, route each to its own expert in parallel, then synthesize.">
          <input
            type="checkbox"
            checked={decompose}
            onChange={(e) => setDecompose(e.target.checked)}
            disabled={loading}
          />
          Decompose (MoE)
        </label>
        {error && <span className="error-text">{error}</span>}
      </div>
    </div>
  )
}
