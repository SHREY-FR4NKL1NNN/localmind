// Thin client for the LocalMind backend API plus shared route presentation
// helpers. Every request throws on a non-2xx response or network failure so
// callers can surface a clear "backend unreachable" message in the UI.

// IPv4 loopback to avoid a possible IPv6 (::1) resolution stall for "localhost"
// on Windows; the backend listens on 127.0.0.1:8000.
export const API_BASE = 'http://127.0.0.1:8000'

// Route presentation: teal for the lightweight Mistral path, purple for the
// high-capability DeepSeek R1 path. Both chosen to read well on light and dark.
export const ROUTE_COLORS = {
  mistral: '#1D9E75',
  deepseek: '#7F77DD',
}

export function routeLabel(route) {
  return route === 'deepseek' ? 'DeepSeek R1' : 'Mistral 7B'
}

export function routeColor(route) {
  return ROUTE_COLORS[route] || '#6b7280'
}

// Expert presentation for the decomposed (MoE) flow. Keyed by the Ollama model
// tag the gate emits. Mistral/DeepSeek reuse the route colors above; Llama 3.2
// (fast tier) and LLaVA (vision tier) get their own so all four are distinct.
export const EXPERT_META = {
  'llama3.2': { label: 'Llama 3.2', color: '#E0922F' },
  mistral: { label: 'Mistral 7B', color: ROUTE_COLORS.mistral },
  'deepseek-r1:7b': { label: 'DeepSeek R1', color: ROUTE_COLORS.deepseek },
  llava: { label: 'LLaVA', color: '#C2569E' },
}

export function expertLabel(expert) {
  return EXPERT_META[expert]?.label || expert
}

export function expertColor(expert) {
  return EXPERT_META[expert]?.color || '#6b7280'
}

async function request(path, options) {
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
// is forwarded so vision sub-tasks can hard-route to LLaVA.
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
    headers: { 'Content-Type': 'application/json' },
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

export function getHistory() {
  return request('/history')
}

export function getHealth() {
  return request('/health')
}
