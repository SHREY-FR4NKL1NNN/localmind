import { expertColor, expertLabel } from '../api'

function isError(text) {
  return typeof text === 'string' && text.startsWith('[error]')
}

function SubtaskCard({ entry, index }) {
  const color = expertColor(entry.expert)
  const errored = isError(entry.response)
  return (
    <div className="subtask">
      <div className="subtask__head">
        <span className="subtask__num">{index + 1}</span>
        <span className="badge badge--sm" style={{ background: color }}>
          {expertLabel(entry.expert)}
        </span>
        {entry.hard_routed && <span className="chip">vision hard-route</span>}
        {entry.depth > 0 && <span className="chip">nested · depth {entry.depth}</span>}
        <span className="subtask__metrics">
          c {entry.complexity.toFixed(2)} · p {entry.privacy.toFixed(2)} ·{' '}
          {entry.latency_ms} ms
        </span>
      </div>
      <div className="subtask__text">{entry.subtask}</div>
      <blockquote className="reasoning reasoning--sm">{entry.reasoning}</blockquote>
      <div className={`response-box${errored ? ' response-box--error' : ''}`}>
        {entry.response || '(no response text)'}
      </div>
    </div>
  )
}

export default function DecomposedPanel({ result }) {
  const subtasks = result.subtasks || []
  const synthesis = result.synthesis

  return (
    <div className="card">
      <h2 className="card__title">Decomposed result (MoE)</h2>

      <span className="badge badge--lg" style={{ background: '#7F77DD' }}>
        {result.decomposed
          ? `${subtasks.length} sub-task${subtasks.length === 1 ? '' : 's'}`
          : 'Single ask — not decomposed'}
      </span>

      {synthesis && (
        <div className="synthesis">
          <div className="synthesis__head">
            <span className="synthesis__label">Unified answer</span>
            <span className="subtask__metrics">
              synthesized by {synthesis.model} · {synthesis.latency_ms} ms
            </span>
          </div>
          <div
            className={`response-box${
              isError(synthesis.response) ? ' response-box--error' : ''
            }`}
          >
            {synthesis.response || '(no response text)'}
          </div>
        </div>
      )}

      <div className="subtasks">
        {subtasks.map((entry, i) => (
          <SubtaskCard key={`${entry.subtask}-${i}`} entry={entry} index={i} />
        ))}
      </div>
    </div>
  )
}
