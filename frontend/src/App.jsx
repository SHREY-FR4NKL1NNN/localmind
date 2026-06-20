import { useEffect, useState } from 'react'
import StatsBar from './components/StatsBar'
import QueryInput from './components/QueryInput'
import ResponsePanel from './components/ResponsePanel'
import DecomposedPanel from './components/DecomposedPanel'
import StreamingPanel from './components/StreamingPanel'
import GatingDiagram from './components/GatingDiagram'
import LiveFeed from './components/LiveFeed'
import AboutDrawer from './components/AboutDrawer'
import { EXPERTS, expertColor, expertLabel, getHealth } from './api'
import { useDecomposedStream } from './hooks/useDecomposedStream'

function EmptyResponse() {
  return (
    <div className="card response-empty">
      <span className="response-empty__icon" aria-hidden>◇</span>
      <span>Your response will appear here</span>
    </div>
  )
}

export default function App() {
  const [latest, setLatest] = useState(null)
  const [streamRequest, setStreamRequest] = useState(null)
  const [health, setHealth] = useState(null)
  const [bannerDismissed, setBannerDismissed] = useState(false)
  const [aboutOpen, setAboutOpen] = useState(false)

  const streamState = useDecomposedStream(streamRequest)
  const streaming = !!streamRequest

  // Single source of truth for Ollama health, polled every 10s.
  useEffect(() => {
    let active = true
    async function load() {
      try {
        const data = await getHealth()
        if (active) {
          setHealth(data)
          if (data.ollama === 'reachable') setBannerDismissed(false)
        }
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
  const reachable = health?.ollama === 'reachable'

  const gateSubtasks = streaming
    ? streamState.subtasks
    : Array.isArray(latest?.subtasks)
      ? latest.subtasks
      : []
  const gateQuery = streaming ? streamRequest.query : latest?.query || ''

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__inner">
          <div className="topbar__brand">
            <span className="wordmark">LocalMind</span>
            <span className="topbar__sep" />
            <span className="topbar__subtitle">MoE-inspired local LLM routing</span>
          </div>
          <div className="topbar__right">
            <div className="expert-dots">
              {EXPERTS.map((e) => (
                <span
                  key={e}
                  className="dot dot--exp dot--lg"
                  style={{ '--exp': expertColor(e) }}
                  title={expertLabel(e)}
                />
              ))}
            </div>
            <div className="health" title={reachable ? 'Ollama reachable' : 'Ollama unreachable'}>
              <span className={`dot dot--lg ${reachable ? 'dot--ok' : 'dot--bad'}`} />
            </div>
            <button type="button" className="topbar__about" onClick={() => setAboutOpen(true)}>
              About
            </button>
          </div>
        </div>
      </header>

      {unreachable && !bannerDismissed && (
        <div className="warn-banner">
          <span>⚠ Ollama unreachable — start with: <code>ollama serve</code></span>
          <button type="button" className="warn-banner__close" aria-label="Dismiss" onClick={() => setBannerDismissed(true)}>
            ×
          </button>
        </div>
      )}

      <main className="layout">
        <section className="layout__left">
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

          <GatingDiagram query={gateQuery} subtasks={gateSubtasks} />

          <div className="response-area">
            {streaming ? (
              <StreamingPanel state={streamState} />
            ) : latest ? (
              Array.isArray(latest.subtasks) ? (
                <DecomposedPanel result={latest} />
              ) : (
                <ResponsePanel result={latest} />
              )
            ) : (
              <EmptyResponse />
            )}
          </div>
        </section>

        <aside className="layout__right">
          <StatsBar health={health} />
          <LiveFeed />
        </aside>
      </main>

      <AboutDrawer open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </div>
  )
}
