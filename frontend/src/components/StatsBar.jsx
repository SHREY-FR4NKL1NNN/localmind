import { useEffect, useState } from 'react'
import { getStats, ROUTE_COLORS } from '../api'

function formatMs(ms) {
  if (!ms) return '0 ms'
  return `${Math.round(ms)} ms`
}

function formatSaved(ms) {
  if (ms > 1000) return `${(ms / 1000).toFixed(1)} s`
  return `${Math.round(ms)} ms`
}

function StatCard({ label, value, color }) {
  return (
    <div className="stat-card">
      <div className="stat-card__label">{label}</div>
      <div className="stat-card__value" style={color ? { color } : undefined}>
        {value}
      </div>
    </div>
  )
}

// `health` is owned by App (single poller) and passed down so the status dot
// and the warning banner never disagree.
export default function StatsBar({ health }) {
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    let active = true
    async function load() {
      try {
        const data = await getStats()
        if (active) {
          setStats(data)
          setError(false)
        }
      } catch {
        if (active) setError(true)
      }
    }
    load()
    const id = setInterval(load, 4000)
    return () => {
      active = false
      clearInterval(id)
    }
  }, [])

  const reachable = health?.ollama === 'reachable'

  return (
    <div className="statsbar">
      <div className="statsbar__head">
        <span className={`dot ${reachable ? 'dot--ok' : ''}`} />
        <span className="health-label">
          {reachable
            ? 'Ollama reachable'
            : health
              ? 'Ollama unreachable'
              : 'Checking Ollama…'}
        </span>
        {error && (
          <span className="error-text">· stats unavailable (backend down?)</span>
        )}
      </div>

      <div className="statsbar__grid">
        <StatCard label="Total queries" value={stats?.total_queries ?? '—'} />
        <StatCard
          label="Mistral %"
          value={stats ? `${stats.mistral_pct}%` : '—'}
          color={ROUTE_COLORS.mistral}
        />
        <StatCard
          label="DeepSeek R1 %"
          value={stats ? `${stats.deepseek_pct}%` : '—'}
          color={ROUTE_COLORS.deepseek}
        />
        <StatCard
          label="Compute saved"
          value={stats ? formatSaved(stats.total_compute_saved_ms) : '—'}
        />
        <StatCard
          label="Avg Mistral latency"
          value={stats ? formatMs(stats.avg_latency_mistral_ms) : '—'}
        />
        <StatCard
          label="Avg DeepSeek latency"
          value={stats ? formatMs(stats.avg_latency_deepseek_ms) : '—'}
        />
      </div>
    </div>
  )
}
