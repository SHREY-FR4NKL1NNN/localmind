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

export function getStats() {
  return request('/stats')
}

export function getHistory() {
  return request('/history')
}

export function getHealth() {
  return request('/health')
}
