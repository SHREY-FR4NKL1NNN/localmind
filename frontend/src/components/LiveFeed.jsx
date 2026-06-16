import { useEffect, useState } from 'react'
import { getHistory, routeColor, routeLabel } from '../api'

function formatTime(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '--:--:--'
  return d.toLocaleTimeString([], { hour12: false })
}

function truncate(text, n) {
  if (!text) return ''
  return text.length > n ? `${text.slice(0, n)}…` : text
}

export default function LiveFeed() {
  const [history, setHistory] = useState([])
  const [error, setError] = useState(false)

  useEffect(() => {
    let active = true
    async function load() {
      try {
        const data = await getHistory()
        if (active) {
          // Backend returns newest-first; keep the most recent 20.
          setHistory(data.slice(0, 20))
          setError(false)
        }
      } catch {
        if (active) setError(true)
      }
    }
    load()
    const id = setInterval(load, 3000)
    return () => {
      active = false
      clearInterval(id)
    }
  }, [])

  return (
    <div className="card">
      <h2 className="card__title">Live routing feed</h2>
      {error && (
        <p className="error-text">Could not load history — is the backend up?</p>
      )}
      {!error && history.length === 0 && (
        <p className="placeholder">No routing decisions yet.</p>
      )}
      <div className="feed">
        {history.map((row, i) => (
          <div className="feed-row" key={`${row.timestamp}-${i}`}>
            <span className="feed-row__time">{formatTime(row.timestamp)}</span>
            <span
              className="badge badge--sm"
              style={{ background: routeColor(row.route) }}
            >
              {routeLabel(row.route)}
            </span>
            <span className="feed-row__meta">c {row.complexity.toFixed(2)}</span>
            <span className="feed-row__meta">{row.latency_ms} ms</span>
            <span className="feed-row__query">{truncate(row.query, 70)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
