import { useEffect, useRef, useState } from 'react'
import { expertColor, expertLabel, expertSize } from '../api'

function isError(text) {
  return typeof text === 'string' && text.startsWith('[error]')
}

// DeepSeek's collapsible reasoning trace. Auto-expanded while thinking tokens
// stream, auto-collapses when the expert finishes. Never renders if no thinking
// tokens arrived (as on Ollama builds that emit no <think> tags).
function ReasoningTrace({ text, done }) {
  const [open, setOpen] = useState(true)
  useEffect(() => {
    if (done) setOpen(false)
  }, [done])
  if (!text) return null
  return (
    <div className={`trace${open ? ' trace--open' : ''}`}>
      <button type="button" className="trace__toggle" onClick={() => setOpen((v) => !v)}>
        <span className="trace__chev">{open ? '▾' : '▸'}</span>
        ◆ Reasoning trace
      </button>
      <div className="trace__body">{text}</div>
    </div>
  )
}

function StatusBadge({ data }) {
  if (data.errored) return <span className="status status--error">Error</span>
  if (data.done) return <span className="status status--done">Done</span>
  return <span className="status status--live">Streaming…</span>
}

function ExpertPanel({ subtask, data }) {
  const color = expertColor(subtask.expert)
  const isDeepseek = subtask.expert === 'deepseek-r1:7b'
  const d = data || { response: '', thinking: '', done: false, latency: null, errored: false }

  const bodyRef = useRef(null)
  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight
  }, [d.response])

  return (
    <div className={`xpanel${d.errored ? ' xpanel--error' : ''}`} style={{ '--exp': color }}>
      <div className="xpanel__head">
        <span className="dot dot--exp" />
        <span className="xpanel__name">{expertLabel(subtask.expert)}</span>
        <span className="xpanel__size">{expertSize(subtask.expert)}</span>
        {subtask.hard_routed && <span className="pill pill--exp">vision</span>}
        <span className="xpanel__spacer" />
        {d.done && d.latency != null && <span className="xpanel__latency">{d.latency} ms</span>}
        <StatusBadge data={d} />
      </div>

      {isDeepseek && <ReasoningTrace text={d.thinking} done={d.done} />}

      <div className={`xpanel__body${d.errored ? ' xpanel__body--error' : ''}`} ref={bodyRef}>
        {d.response || (d.done ? '(no response)' : '')}
      </div>
    </div>
  )
}

function SparsityBadge({ sparsity }) {
  if (!sparsity) return null
  const pct = Math.round((1 - sparsity.sparsity_ratio) * 100)
  return (
    <div className="sparsity">
      <span className="sparsity__dots">
        {sparsity.activated_expert_names.map((n) => (
          <span key={n} className="dot dot--exp" style={{ '--exp': expertColor(n) }} title={expertLabel(n)} />
        ))}
      </span>
      <span className="sparsity__text">
        {sparsity.experts_activated} / {sparsity.experts_available} experts activated
      </span>
      <span className="sparsity__sep">·</span>
      <span className="sparsity__muted">{pct}% sparse</span>
      {sparsity.vision_activated && (
        <span className="pill pill--exp" style={{ '--exp': 'var(--expert-llava)' }}>vision</span>
      )}
    </div>
  )
}

function CombinerPanel({ text, active, skipped, done }) {
  if (skipped) return null
  if (!active && !done) return null
  return (
    <div className="combiner">
      <div className="combiner__head">
        <span className="combiner__label">◎ Combined response</span>
      </div>
      <div className="combiner__body">{text || '…'}</div>
    </div>
  )
}

// Shimmer placeholder shown after sparsity fires but before the combiner starts.
function CombinerSkeleton() {
  return (
    <div className="combiner">
      <div className="combiner__head">
        <span className="combiner__label">◎ Combined response</span>
      </div>
      <div className="skeleton">
        <span className="skeleton__line" />
        <span className="skeleton__line skeleton__line--short" />
      </div>
    </div>
  )
}

export default function StreamingPanel({ state }) {
  const { subtasks, byIndex, sparsity, combinerText, combinerActive, combinerSkipped, status } = state

  const showSkeleton =
    sparsity && !combinerSkipped && !combinerActive && status !== 'done'

  return (
    <div className="card stream">
      {status === 'error' && (
        <div className="error-card">
          <p className="error-card__msg">⚠ {state.error}</p>
        </div>
      )}

      {subtasks.length === 0 && status === 'streaming' && (
        <p className="placeholder">Routing…</p>
      )}

      <div className="xpanels">
        {subtasks.map((st) => (
          <ExpertPanel key={st.subtask_index} subtask={st} data={byIndex[st.subtask_index]} />
        ))}
      </div>

      <SparsityBadge sparsity={sparsity} />

      {showSkeleton && <CombinerSkeleton />}
      <CombinerPanel
        text={combinerText}
        active={combinerActive}
        skipped={combinerSkipped}
        done={status === 'done'}
      />
    </div>
  )
}
