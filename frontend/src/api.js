// Thin client for the LocalMind backend API plus shared route presentation
// helpers. Every request throws on a non-2xx response or network failure so
// callers can surface a clear "backend unreachable" message in the UI.

// Backend base URL. In production builds (e.g. on Vercel) this comes from the
// VITE_API_URL env var — point it at the public backend (the Azure Container
// Apps deployment). Locally it falls back to the IPv4 loopback (127.0.0.1, not
// "localhost", to avoid a possible IPv6 (::1) resolution stall on Windows); the
// backend listens on 127.0.0.1:8000.
export const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

// Single source of truth for expert presentation, keyed by the Ollama model tag
// the gate emits. Colours are CSS custom properties (defined in index.css) so
// components never hardcode hex — they reference the design-system token via the
// returned `var(--expert-*)` string, which resolves inside inline styles.
export const EXPERT_META = {
  'llama3.2': { label: 'Llama 3.2', size: '3.2B', color: 'var(--expert-llama)', token: '--expert-llama' },
  mistral: { label: 'Mistral 7B', size: '7B', color: 'var(--expert-mistral)', token: '--expert-mistral' },
  'deepseek-r1:7b': { label: 'DeepSeek R1', size: '7B', color: 'var(--expert-deepseek)', token: '--expert-deepseek' },
  llava: { label: 'MiniCPM-V', size: '8B', color: 'var(--expert-llava)', token: '--expert-llava' },
}

// The four experts in canonical display order (used by the header dots).
export const EXPERTS = ['llama3.2', 'mistral', 'deepseek-r1:7b', 'llava']

export function expertLabel(expert) {
  return EXPERT_META[expert]?.label || expert
}

// Returns a `var(--expert-*)` token string for use as a dynamic inline style.
export function expertColor(expert) {
  return EXPERT_META[expert]?.color || 'var(--lm-text-muted)'
}

export function expertSize(expert) {
  return EXPERT_META[expert]?.size || ''
}

// The single-route flow reports "mistral"/"deepseek"; map those onto the same
// expert colours so the whole app stays consistent.
export function routeLabel(route) {
  return route === 'deepseek' ? 'DeepSeek R1' : 'Mistral 7B'
}

export function routeColor(route) {
  return route === 'deepseek' ? 'var(--expert-deepseek)' : 'var(--expert-mistral)'
}

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) {
    throw new Error(`Request to ${path} failed with status ${res.status}`)
  }
  return res.json()
}

export function postQuery(query) {
  return request('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })
}

// Tiered MoE flow: decompose the query into sub-tasks, route each to an expert,
// run them in parallel, and synthesize a unified answer. Optional base64 image
// is forwarded so vision sub-tasks can hard-route to the vision expert.
export function postQueryDecomposed(query, imageBase64 = null) {
  return request('/query/decomposed', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, image_base64: imageBase64 }),
  })
}

// Parse one raw SSE event block ("event: x\ndata: {...}") into { event, data }.
function parseSSEEvent(raw) {
  let event = 'message'
  const dataLines = []
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (dataLines.length === 0) return null
  try {
    return { event, data: JSON.parse(dataLines.join('\n')) }
  } catch {
    return null
  }
}

// Streaming tiered flow over Server-Sent Events. EventSource can't POST a body,
// so we read the fetch ReadableStream manually and split it on the SSE record
// separator (\n\n). `onEvent(type, data)` fires per event as it arrives.
// Returns when the stream closes; throws on a failed connection.
export async function streamDecomposed({ query, imageBase64 = null, onEvent, signal }) {
  const res = await fetch(`${API_BASE}/query/decomposed/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query, image_base64: imageBase64 }),
    signal,
  })
  if (!res.ok || !res.body) {
    throw new Error(`Stream request failed with status ${res.status}`)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const parsed = parseSSEEvent(rawEvent)
      if (parsed) onEvent(parsed.event, parsed.data)
    }
  }
}

export function getStats() {
  return request('/stats')
}

export function getExpertStats() {
  return request('/expert-stats')
}

export function getHistory() {
  return request('/history')
}

export function getHealth() {
  return request('/health')
}
