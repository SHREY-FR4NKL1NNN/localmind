import { routeColor, routeLabel } from '../api'

function ScoreBar({ label, value, color }) {
  const pct = Math.round((value ?? 0) * 100)
  return (
    <div className="score">
      <div className="score__head">
        <span>{label}</span>
        <span>{(value ?? 0).toFixed(2)}</span>
      </div>
      <div className="score__track">
        <div
          className="score__fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  )
}

export default function ResponsePanel({ result }) {
  if (!result) {
    return (
      <div className="card">
        <h2 className="card__title">Latest result</h2>
        <p className="placeholder">
          Submit a query to see how LocalMind classifies and routes it.
        </p>
      </div>
    )
  }

  const color = routeColor(result.route)
  const isDeepseek = result.route === 'deepseek'

  return (
    <div className="card">
      <h2 className="card__title">Latest result</h2>

      <span className="badge badge--lg" style={{ background: color }}>
        {routeLabel(result.route)}
      </span>

      {result.error && (
        <div className="banner" style={{ marginTop: 14 }}>
          {result.error}
        </div>
      )}

      <blockquote className="reasoning">{result.reasoning}</blockquote>

      <div className="scores">
        <ScoreBar
          label="Complexity"
          value={result.complexity}
          color={routeColor('deepseek')}
        />
        <ScoreBar
          label="Privacy"
          value={result.privacy}
          color={routeColor('mistral')}
        />
      </div>

      <div className="metrics">
        <div>
          <div className="metric__label">Latency</div>
          <div className="metric__value">{result.latency_ms} ms</div>
        </div>
        <div>
          <div className="metric__label">Model</div>
          <div className="metric__value">{result.model}</div>
        </div>
        <div>
          <div className="metric__label">Compute saved</div>
          <div className="metric__value">
            {isDeepseek ? '—' : `${result.compute_saved_ms} ms`}
          </div>
        </div>
      </div>

      <div className="response-box">
        {result.response || '(no response text)'}
      </div>
    </div>
  )
}
