import { useEffect, useReducer } from 'react'
import { streamDecomposed } from '../api'

// Owns the full live state of one decomposed streaming request. Lifted out of
// the view so both the GatingDiagram (routing) and the StreamingPanel (tokens)
// can read the same source of truth. SSE events arrive fast and interleaved
// across experts, so each event is a small, order-independent state update.

export const initialStreamState = {
  status: 'idle', // idle | streaming | done | error
  subtasks: [], // routing decisions from gate_complete
  byIndex: {}, // subtask_index -> { expert, response, thinking, done, latency, errored }
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
      return { ...initialStreamState, status: 'streaming' }
    case 'gate': {
      const byIndex = {}
      for (const st of action.subtasks) {
        byIndex[st.subtask_index] = {
          expert: st.expert,
          response: '',
          thinking: '',
          done: false,
          latency: null,
          errored: false,
        }
      }
      return { ...state, subtasks: action.subtasks, byIndex }
    }
    case 'token': {
      const cur = state.byIndex[action.idx]
      if (!cur) return state
      const errored = cur.errored || action.token.startsWith('[error]')
      const next = action.isThinking
        ? { ...cur, thinking: cur.thinking + action.token }
        : { ...cur, response: cur.response + action.token, errored }
      return { ...state, byIndex: { ...state.byIndex, [action.idx]: next } }
    }
    case 'expert_done': {
      const cur = state.byIndex[action.idx]
      if (!cur) return state
      // full_response is authoritative (and, for DeepSeek, excludes the trace).
      const next = {
        ...cur,
        done: true,
        latency: action.latency,
        response: action.full,
        errored: action.full.startsWith('[error]'),
      }
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

export function useDecomposedStream(request) {
  const [state, dispatch] = useReducer(reducer, initialStreamState)

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

  return state
}
