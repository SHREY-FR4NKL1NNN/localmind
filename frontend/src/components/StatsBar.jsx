import { useEffect, useState } from 'react'
import { expertColor, expertLabel, getExpertStats, getStats } from '../api'

function formatMs(ms) {
  if (!ms) return '0 ms'
  if (ms > 1000) return `${(ms / 1000).toFixed(1)} s`
  return `${Math.round(ms)} ms`
}

function Stat({ label, value }) {
  return (
    <div className="strip__stat">
      <span className="strip__label">{label}</span>
      <span className="strip__value">{value}</span>
    </div>
  )
}

// Proportional 4-segment bar of expert activations, each in its expert colour.
function ActivationBar({ experts }) {
  const entries = Object.entries(experts || {})
  const total = entries.reduce((s, [, v]) => s + v.count, 0)
  if (total === 0) {
    return <div className="actbar actbar--empty" title="No expert activations yet" />
  }
  return (
    <div className="actbar" title="Expert activation share">
      {entries.map(([name, v]) => (
        <span
          key={name}
          className="actbar__seg"
          style={{ width: `${(v.count / total) * 100}%`, '--exp': expertColor(name) }}
          title={`${expertLabel(name)} · ${v.pct}%`}
        />
      ))}
    </div>
  )
}

// `health` is owned by App (single poller) so the dot never disagrees with the
// header / banner.
export default function StatsBar({ health }) {
  const [stats, setStats] = useState(null)
  const [experts, setExperts] = useState(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    let active = true
    async function load() {
      try {
        const [s, e] = await Promise.all([getStats(), getExpertStats()])
        if (active) {
          setStats(s)
          setExperts(e.experts || {})
          setError(false)
        }
      } catch {
        if (active) setError(true)
      }
    }
    load()
    const id = setInterval(load, 5000)
    return () => {
      active = false
      clearInterval(id)
    }
  }, [])

  const reachable = health?.ollama === 'reachable'

  return (
    <div className="strip">
      <div className="strip__health">
        <span className={`dot ${reachable ? 'dot--ok' : 'dot--bad'}`} />
        <span className="strip__label">{reachable ? 'Online' : 'Offline'}</span>
      </div>
      <span className="strip__divider" />
      <Stat label="Queries" value={stats?.total_queries ?? '—'} />
      <span className="strip__divider" />
      <Stat label="Decomposed" value={stats?.decomposed_queries ?? '—'} />
      <span className="strip__divider" />
      <Stat label="Avg latency" value={stats ? formatMs(stats.avg_total_latency_ms) : '—'} />
      <span className="strip__divider" />
      <div className="strip__stat strip__stat--grow">
        <span className="strip__label">Expert activation</span>
        <ActivationBar experts={experts} />
      </div>
      {error && <span className="error-text strip__err">stats down</span>}
    </div>
  )
}
