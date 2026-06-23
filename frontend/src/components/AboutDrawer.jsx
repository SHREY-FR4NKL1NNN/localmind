// Slide-in panel from the right (not a modal). Overlay dims and closes on click.
export default function AboutDrawer({ open, onClose }) {
  return (
    <div className={`drawer-root${open ? ' drawer-root--open' : ''}`} aria-hidden={!open}>
      <div className="drawer-overlay" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-label="About LocalMind">
        <div className="drawer__head">
          <span className="drawer__title">LocalMind</span>
          <button type="button" className="drawer__close" aria-label="Close" onClick={onClose}>
            ×
          </button>
        </div>

        <p className="drawer__lede">MoE-inspired routing layer over local models — every query is
          decomposed, gated to the right expert, run in parallel, and synthesized.</p>

        <div className="drawer__section">
          <span className="drawer__kicker">Architecture</span>
          <ul className="drawer__list">
            <li>4 experts: Llama 3.2, Mistral, DeepSeek R1, MiniCPM-V</li>
            <li>Rule-based, explainable gate (Llama 3.2)</li>
            <li>Decompose → gate → parallel → combine</li>
            <li>Fully local — nothing leaves the machine</li>
          </ul>
        </div>

        <div className="drawer__section">
          <span className="drawer__kicker">Repository</span>
          <a
            className="drawer__link"
            href="https://github.com/SHREY-FR4NKL1NNN/localmind"
            target="_blank"
            rel="noreferrer"
          >
            github.com/SHREY-FR4NKL1NNN/localmind ↗
          </a>
        </div>

        <div className="drawer__foot">Built for Polaris Fellowship 2026</div>
      </aside>
    </div>
  )
}
