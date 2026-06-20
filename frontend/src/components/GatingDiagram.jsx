import { expertColor, expertLabel } from '../api'
import { useMediaQuery } from '../hooks/useMediaQuery'

// Animated SVG routing visualization. Renders Query → Gate → Experts with
// traveling dots on the connecting paths. Reads gate_complete data; derives
// whether decomposition happened on the client (the stream doesn't send a
// `decomposed` flag, and the backend is out of scope here) by comparing the
// lone sub-task to the original query.

function truncate(text, n) {
  if (!text) return ''
  return text.length > n ? `${text.slice(0, n - 1)}…` : text
}

// One connecting path. A dim static line (permanent routing evidence) plus a
// bright traveling dot overlay that animates once via stroke-dashoffset.
function FlowPath({ d, color, delay }) {
  const style = { '--gd-color': color, '--gd-delay': `${delay}ms` }
  return (
    <g>
      <path className="gd-line" d={d} style={style} />
      <path className="gd-flow" d={d} pathLength="100" style={style} />
    </g>
  )
}

// Build the geometry for a horizontal (desktop) or vertical (mobile) layout.
function buildLayout(subtasks, decomposed, vertical) {
  const n = subtasks.length
  if (vertical) {
    const vb = { w: 340, h: 260 }
    const query = { x: 90, y: 10, w: 160, h: 44 }
    const gate = { cx: 170, cy: 120, r: 28 }
    const gap = 8
    const ew = Math.max(60, (vb.w - 24 - (n - 1) * gap) / n)
    const totalW = n * ew + (n - 1) * gap
    const startX = (vb.w - totalW) / 2
    const ey = 196
    const eh = 52
    const experts = subtasks.map((st, i) => ({
      st,
      x: startX + i * (ew + gap),
      y: ey,
      w: ew,
      h: eh,
      anchor: { x: startX + i * (ew + gap) + ew / 2, y: ey },
    }))
    const queryAnchor = { x: query.x + query.w / 2, y: query.y + query.h }
    return { vb, query, gate, experts, queryAnchor, gatePre: { x: gate.cx, y: gate.cy - gate.r }, gatePost: { x: gate.cx, y: gate.cy + gate.r }, vertical }
  }
  const vb = { w: 640, h: 180 }
  const query = { x: 14, y: 62, w: 140, h: 56 }
  const gate = { cx: 320, cy: 90, r: 30 }
  const ew = 168
  const ex = vb.w - ew - 8
  const eh = 44
  const gapE = 12
  const totalH = n * eh + (n - 1) * gapE
  const startY = gate.cy - totalH / 2
  const experts = subtasks.map((st, i) => ({
    st,
    x: ex,
    y: startY + i * (eh + gapE),
    w: ew,
    h: eh,
    anchor: { x: ex, y: startY + i * (eh + gapE) + eh / 2 },
  }))
  const queryAnchor = { x: query.x + query.w, y: query.y + query.h / 2 }
  return { vb, query, gate, experts, queryAnchor, gatePre: { x: gate.cx - gate.r, y: gate.cy }, gatePost: { x: gate.cx + gate.r, y: gate.cy }, vertical }
}

function curve(a, b, vertical) {
  if (vertical) {
    const my = (a.y + b.y) / 2
    return `M ${a.x} ${a.y} C ${a.x} ${my} ${b.x} ${my} ${b.x} ${b.y}`
  }
  const mx = (a.x + b.x) / 2
  return `M ${a.x} ${a.y} C ${mx} ${a.y} ${mx} ${b.y} ${b.x} ${b.y}`
}

// Faint structural ghost shown before the first query.
function GhostDiagram({ vertical }) {
  const L = buildLayout([{ subtask: '', expert: 'llama3.2' }, { subtask: '', expert: 'mistral' }], true, vertical)
  return (
    <div className="gating gating--ghost">
      <svg viewBox={`0 0 ${L.vb.w} ${L.vb.h}`} className="gating__svg" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Routing diagram placeholder">
        <rect className="gd-node-ghost" x={L.query.x} y={L.query.y} width={L.query.w} height={L.query.h} rx="10" />
        <text className="gd-ghost-label" x={L.query.x + L.query.w / 2} y={L.query.y + L.query.h / 2} textAnchor="middle" dominantBaseline="middle">Query</text>
        <circle className="gd-node-ghost" cx={L.gate.cx} cy={L.gate.cy} r={L.gate.r} />
        <text className="gd-ghost-label" x={L.gate.cx} y={L.gate.cy} textAnchor="middle" dominantBaseline="middle">Gate</text>
        {L.experts.map((e, i) => (
          <g key={i}>
            <rect className="gd-node-ghost" x={e.x} y={e.y} width={e.w} height={e.h} rx="8" />
            <text className="gd-ghost-label" x={e.x + e.w / 2} y={e.y + e.h / 2} textAnchor="middle" dominantBaseline="middle">Expert</text>
          </g>
        ))}
        <path className="gd-line gd-line--ghost" d={curve(L.queryAnchor, L.gatePre, vertical)} />
        {L.experts.map((e, i) => (
          <path key={i} className="gd-line gd-line--ghost" d={curve(L.gatePost, e.anchor, vertical)} />
        ))}
      </svg>
      <p className="gating__hint">Submit a query to see routing</p>
    </div>
  )
}

export default function GatingDiagram({ query, subtasks }) {
  const vertical = useMediaQuery('(max-width: 768px)')

  if (!subtasks || subtasks.length === 0) {
    return <GhostDiagram vertical={vertical} />
  }

  const decomposed = !(
    subtasks.length === 1 && (subtasks[0].subtask || '').trim() === (query || '').trim()
  )

  // Direct route: a single sub-task equal to the query — skip the gate node.
  if (!decomposed) {
    const e = subtasks[0]
    const color = expertColor(e.expert)
    const L = buildLayout(subtasks, false, vertical)
    return (
      <div className="gating" key={query}>
        <svg viewBox={`0 0 ${L.vb.w} ${L.vb.h}`} className="gating__svg" preserveAspectRatio="xMidYMid meet">
          <FlowPath d={curve(L.queryAnchor, L.experts[0].anchor, vertical)} color={color} delay={100} />
          <g className="gd-node">
            <rect x={L.query.x} y={L.query.y} width={L.query.w} height={L.query.h} rx="10" className="gd-node-rect" />
            <text className="gd-node-kicker" x={L.query.x + 12} y={L.query.y + 18}>Query</text>
            <text className="gd-node-text" x={L.query.x + 12} y={L.query.y + 38}>{truncate(query, 40)}</text>
          </g>
          <g className="gd-node" style={{ '--exp': color }}>
            <rect x={L.experts[0].x} y={L.experts[0].y} width={L.experts[0].w} height={L.experts[0].h} rx="8" className="gd-expert-rect" />
            <rect x={L.experts[0].x} y={L.experts[0].y} width="4" height={L.experts[0].h} className="gd-expert-bar" />
            <text className="gd-expert-name" x={L.experts[0].x + 14} y={L.experts[0].y + 18}>{expertLabel(e.expert)}</text>
            <text className="gd-node-text" x={L.experts[0].x + 14} y={L.experts[0].y + 36}>{truncate(e.subtask, 30)}</text>
          </g>
        </svg>
        <p className="gating__hint">Direct route — no decomposition needed</p>
      </div>
    )
  }

  const L = buildLayout(subtasks, true, vertical)
  return (
    <div className="gating" key={query}>
      <svg viewBox={`0 0 ${L.vb.w} ${L.vb.h}`} className="gating__svg" preserveAspectRatio="xMidYMid meet">
        {/* lines first so nodes paint over their endpoints */}
        <FlowPath d={curve(L.queryAnchor, L.gatePre, vertical)} color="var(--expert-llama)" delay={100} />
        {L.experts.map((e, i) => (
          <FlowPath key={`f${i}`} d={curve(L.gatePost, e.anchor, vertical)} color={expertColor(e.st.expert)} delay={250 + i * 150} />
        ))}

        {/* Query node */}
        <g className="gd-node">
          <rect x={L.query.x} y={L.query.y} width={L.query.w} height={L.query.h} rx="10" className="gd-node-rect" />
          <text className="gd-node-kicker" x={L.query.x + 12} y={L.query.y + 18}>Query</text>
          <text className="gd-node-text" x={L.query.x + 12} y={L.query.y + 38}>{truncate(query, 40)}</text>
        </g>

        {/* Gate node */}
        <g className="gd-node">
          <text className="gd-node-kicker" x={L.gate.cx} y={L.gate.cy - L.gate.r - 8} textAnchor="middle">Gate</text>
          <circle cx={L.gate.cx} cy={L.gate.cy} r={L.gate.r} className="gd-gate" />
          <text className="gd-gate-model" x={L.gate.cx} y={L.gate.cy + 4} textAnchor="middle">llama3.2</text>
        </g>

        {/* Expert nodes */}
        {L.experts.map((e, i) => {
          const color = expertColor(e.st.expert)
          return (
            <g className="gd-node gd-node--expert" key={`e${i}`} style={{ '--exp': color, '--gd-delay': `${250 + i * 150}ms` }}>
              <rect x={e.x} y={e.y} width={e.w} height={e.h} rx="8" className="gd-expert-rect" />
              <rect x={e.x} y={e.y} width="4" height={e.h} className="gd-expert-bar" />
              <text className="gd-expert-name" x={e.x + 14} y={e.y + 17}>{expertLabel(e.st.expert)}</text>
              <text className="gd-node-text" x={e.x + 14} y={e.y + 33}>{truncate(e.st.subtask, 30)}</text>
              <text className="gd-expert-cx" x={e.x + e.w - 10} y={e.y + 17} textAnchor="end">
                {Number(e.st.complexity).toFixed(2)} complexity
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
