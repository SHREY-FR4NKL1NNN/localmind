import { useEffect, useState } from 'react'
import StatsBar from './components/StatsBar'
import QueryInput from './components/QueryInput'
import ResponsePanel from './components/ResponsePanel'
import DecomposedPanel from './components/DecomposedPanel'
import StreamingPanel from './components/StreamingPanel'
import LiveFeed from './components/LiveFeed'
import { getHealth } from './api'

export default function App() {
  const [latest, setLatest] = useState(null)
  const [streamRequest, setStreamRequest] = useState(null)
  const [health, setHealth] = useState(null)

  // Single source of truth for Ollama health: polled here every 10s and shared
  // with StatsBar (status dot) and the warning banner below.
  useEffect(() => {
    let active = true
    async function load() {
      try {
        const data = await getHealth()
        if (active) setHealth(data)
      } catch {
        if (active) setHealth({ status: 'error', ollama: 'unreachable', models: [] })
      }
    }
    load()
    const id = setInterval(load, 10000)
    return () => {
      active = false
      clearInterval(id)
    }
  }, [])

  const unreachable = health && health.ollama !== 'reachable'

  return (
    <div className="app">
      <header className="app__header">
        <h1>LocalMind</h1>
        <p>Smart local LLM routing — Mistral 7B for the simple, DeepSeek R1 for the hard.</p>
      </header>

      <StatsBar health={health} />

      {unreachable && (
        <div className="banner">
          ⚠️ Ollama is unreachable. Start it with <code>ollama serve</code> and
          ensure <code>mistral</code> and <code>deepseek-r1:7b</code> are pulled.
        </div>
      )}

      <div className="app__columns">
        <div className="app__left">
          <QueryInput
            onResult={(r) => {
              setStreamRequest(null)
              setLatest(r)
            }}
            onStream={(req) => {
              setLatest(null)
              setStreamRequest(req)
            }}
          />
          {streamRequest ? (
            <StreamingPanel request={streamRequest} />
          ) : Array.isArray(latest?.subtasks) ? (
            <DecomposedPanel result={latest} />
          ) : (
            <ResponsePanel result={latest} />
          )}
        </div>
        <LiveFeed />
      </div>
    </div>
  )
}
