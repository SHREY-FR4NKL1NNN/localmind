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
    <div className="card feed">
      <div className="feed__head">
        <span className="feed__title">Activity</span>
        <span className="feed__count">{history.length}</span>
      </div>

      {error && <p className="error-text">Could not load history — is the backend up?</p>}

      {!error && history.length === 0 && (
        <div className="feed__empty">
          <span className="feed__empty-icon" aria-hidden>◴</span>
          <span>No queries yet</span>
        </div>
      )}

      <div className="feed__rows">
        {history.map((row, i) => (
          <div className="feed-row" key={`${row.timestamp}-${i}`}>
            <span className="dot dot--exp" style={{ '--exp': routeColor(row.route) }} />
            <span className="feed-row__time">{formatTime(row.timestamp)}</span>
            <span className="pill pill--exp" style={{ '--exp': routeColor(row.route) }}>
              {routeLabel(row.route)}
            </span>
            <span className="feed-row__query">{truncate(row.query, 60)}</span>
            <span className="feed-row__latency">{row.latency_ms} ms</span>
          </div>
        ))}
      </div>
    </div>
  )
}
