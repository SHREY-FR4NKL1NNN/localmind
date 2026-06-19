import { useEffect, useReducer, useRef, useState } from 'react'
import { expertColor, expertLabel, streamDecomposed } from '../api'

// ---- streaming state reducer -------------------------------------------------
// One reducer owns the whole live view. SSE events arrive fast and out of order
// across experts, so each event is a small, order-independent state update.

const initialState = {
  status: 'idle', // idle | streaming | done | error
  subtasks: [], // routing decisions from gate_complete
  byIndex: {}, // subtask_index -> { expert, response, thinking, done, latency }
  sparsity: null,
  combinerText: '',
  combinerActive: false,
  combinerSkipped: null,
  summary: null,
  error: null,
}

function reducer(state, action) {
  switch (action.type) {
    case 'reset':
      return { ...initialState, status: 'streaming' }
    case 'gate': {
      const byIndex = {}
      for (const st of action.subtasks) {
        byIndex[st.subtask_index] = {
          expert: st.expert,
          response: '',
          thinking: '',
          done: false,
          latency: null,
        }
      }
      return { ...state, subtasks: action.subtasks, byIndex }
    }
    case 'token': {
      const cur = state.byIndex[action.idx]
      if (!cur) return state
      const next = action.isThinking
        ? { ...cur, thinking: cur.thinking + action.token }
        : { ...cur, response: cur.response + action.token }
      return { ...state, byIndex: { ...state.byIndex, [action.idx]: next } }
    }
    case 'expert_done': {
      const cur = state.byIndex[action.idx]
      if (!cur) return state
      // full_response is authoritative (and, for DeepSeek, excludes the trace).
      const next = { ...cur, done: true, latency: action.latency, response: action.full }
      return { ...state, byIndex: { ...state.byIndex, [action.idx]: next } }
    }
    case 'sparsity':
      return { ...state, sparsity: action.sparsity }
    case 'combiner_token':
      return {
        ...state,
        combinerActive: true,
        combinerText: state.combinerText + action.token,
      }
    case 'done':
      return {
        ...state,
        status: 'done',
        summary: action.summary,
        combinerSkipped: action.summary.combiner_skipped,
      }
    case 'error':
      return { ...state, status: 'error', error: action.message }
    default:
      return state
  }
}

// ---- per-expert panel --------------------------------------------------------

function ExpertPanel({ subtask, state }) {
  const color = expertColor(subtask.expert)
  const isDeepseek = subtask.expert === 'deepseek-r1:7b'
  const data = state || { response: '', thinking: '', done: false, latency: null }

  // DeepSeek's reasoning panel starts expanded while thinking streams, then
  // auto-collapses once the expert finishes.
  const [thinkOpen, setThinkOpen] = useState(true)
  useEffect(() => {
    if (data.done) setThinkOpen(false)
  }, [data.done])

  const bodyRef = useRef(null)
  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight
  }, [data.response])

  return (
    <div className="expanel" style={{ borderTopColor: color }}>
      <div className="expanel__head">
        <span className="badge badge--sm" style={{ background: color }}>
          {expertLabel(subtask.expert)}
        </span>
        {subtask.hard_routed && <span className="chip">vision hard-route</span>}
        <span className="expanel__task">{subtask.subtask}</span>
        <span className="expanel__status">
          {data.done ? (
            <span className="latency-badge">{data.latency} ms</span>
          ) : (
            <span className="streaming-dot">streaming…</span>
          )}
        </span>
      </div>

      {isDeepseek && data.thinking && (
        <div className="think">
          <button
            type="button"
            className="think__toggle"
            onClick={() => setThinkOpen((v) => !v)}
          >
            {thinkOpen ? '▾' : '▸'} Thinking{data.done ? '' : '…'}
          </button>
          {thinkOpen && <div className="think__body">{data.thinking}</div>}
        </div>
      )}

      <div className="expanel__body" ref={bodyRef}>
        {data.response || (data.done ? '(no response)' : '')}
      </div>
    </div>
  )
}

// ---- sparsity badge ----------------------------------------------------------

function SparsityBadge({ sparsity }) {
  if (!sparsity) return null
  return (
    <div className="sparsity">
      <span className="sparsity__main">
        {sparsity.experts_activated} / {sparsity.experts_available} experts activated
      </span>
      <span className="sparsity__ratio">
        sparsity {sparsity.sparsity_ratio.toFixed(2)}
      </span>
      <span className="sparsity__vision">
        {sparsity.vision_activated ? '👁 vision used' : 'no vision'}
      </span>
      <span className="sparsity__names">
        {sparsity.activated_expert_names.map((n) => expertLabel(n)).join(' · ')}
      </span>
    </div>
  )
}

// ---- combiner panel ----------------------------------------------------------

function CombinerPanel({ text, active, skipped, done }) {
  if (skipped) return null
  if (!active && !done) return null
  return (
    <div className="combiner">
      <div className="combiner__head">
        <span className="combiner__label">Unified answer</span>
        <span className="subtask__metrics">synthesized by Llama 3.2</span>
      </div>
      <div className="response-box">{text || '…'}</div>
    </div>
  )
}

// ---- main panel --------------------------------------------------------------

export default function StreamingPanel({ request }) {
  const [state, dispatch] = useReducer(reducer, initialState)

  useEffect(() => {
    if (!request) return undefined
    const controller = new AbortController()
    dispatch({ type: 'reset' })

    streamDecomposed({
      query: request.query,
      imageBase64: request.imageBase64,
      signal: controller.signal,
      onEvent: (event, data) => {
        switch (event) {
          case 'gate_complete':
            dispatch({ type: 'gate', subtasks: data.subtasks })
            break
          case 'expert_token':
            dispatch({
              type: 'token',
              idx: data.subtask_index,
              isThinking: data.is_thinking,
              token: data.token,
            })
            break
          case 'expert_done':
            dispatch({
              type: 'expert_done',
              idx: data.subtask_index,
              full: data.full_response,
              latency: data.latency_ms,
            })
            break
          case 'sparsity':
            dispatch({ type: 'sparsity', sparsity: data })
            break
          case 'combiner_token':
            dispatch({ type: 'combiner_token', token: data.token })
            break
          case 'done':
            dispatch({ type: 'done', summary: data })
            break
          default:
            break
        }
      },
    }).catch((err) => {
      if (!controller.signal.aborted) {
        dispatch({ type: 'error', message: String(err.message || err) })
      }
    })

    return () => controller.abort()
  }, [request])

  if (!request) {
    return (
      <div className="card">
        <h2 className="card__title">Streaming result</h2>
        <p className="placeholder">
          Submit a query with streaming enabled to watch every expert respond live.
        </p>
      </div>
    )
  }

  return (
    <div className="card">
      <h2 className="card__title">Streaming result (MoE)</h2>

      {state.error && <div className="banner">⚠️ {state.error}</div>}

      {state.subtasks.length === 0 && state.status === 'streaming' && (
        <p className="placeholder">Routing…</p>
      )}

      <SparsityBadge sparsity={state.sparsity} />

      <div className="expanels">
        {state.subtasks.map((st) => (
          <ExpertPanel
            key={st.subtask_index}
            subtask={st}
            state={state.byIndex[st.subtask_index]}
          />
        ))}
      </div>

      <CombinerPanel
        text={state.combinerText}
        active={state.combinerActive}
        skipped={state.combinerSkipped}
        done={state.status === 'done'}
      />
    </div>
  )
}
