import { useEffect, useRef, useState } from 'react'
import TerminalPane from './components/TerminalPane'

const panes = [
  { jobId: 'qwen', title: 'Qwen Baseline', tone: 'cyan' },
  { jobId: 'rl-agent', title: 'RL Agent', tone: 'green' },
]

function formatMs(value) {
  if (value == null) {
    return '--'
  }
  if (value < 1000) {
    return `${Math.round(value)} ms`
  }
  return `${(value / 1000).toFixed(2)} s`
}

function buildRunStats(telemetry, issuedAt) {
  if (!telemetry || !issuedAt) {
    return {
      responseMs: null,
      completionMs: null,
      waiting: true,
    }
  }

  const responseMs = telemetry.lastOutputAt && telemetry.lastOutputAt >= issuedAt ? telemetry.lastOutputAt - issuedAt : null
  const finishedAt = telemetry.session?.finished_at ? telemetry.session.finished_at * 1000 : null
  const completionMs = finishedAt && finishedAt >= issuedAt ? finishedAt - issuedAt : null

  return {
    responseMs,
    completionMs,
    waiting: responseMs == null && completionMs == null,
  }
}

function App() {
  const [split, setSplit] = useState(50)
  const [dragging, setDragging] = useState(false)
  const [command, setCommand] = useState('')
  const [comparisonRun, setComparisonRun] = useState(null)
  const [telemetry, setTelemetry] = useState({
    qwen: null,
    'rl-agent': null,
  })
  const workspaceRef = useRef(null)
  const leftPaneRef = useRef(null)
  const rightPaneRef = useRef(null)

  useEffect(() => {
    if (!dragging) {
      return undefined
    }

    const handlePointerMove = (event) => {
      const bounds = workspaceRef.current?.getBoundingClientRect()
      if (!bounds) {
        return
      }

      const next = ((event.clientX - bounds.left) / bounds.width) * 100
      const clamped = Math.min(75, Math.max(25, next))
      setSplit(clamped)
    }

    const handlePointerUp = () => {
      setDragging(false)
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerUp)

    return () => {
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', handlePointerUp)
    }
  }, [dragging])

  const handleBroadcast = async (event) => {
    event.preventDefault()
    const value = command.trim()
    if (!value) {
      return
    }

    const issuedAt = Date.now()
    setComparisonRun({
      command: value,
      issuedAt,
    })

    await Promise.allSettled([
      leftPaneRef.current?.submit(value),
      rightPaneRef.current?.submit(value),
    ])
    setCommand('')
  }

  const handleTelemetryChange = (payload) => {
    setTelemetry((previous) => ({
      ...previous,
      [payload.jobId]: payload,
    }))
  }

  const qwenStats = buildRunStats(telemetry.qwen, comparisonRun?.issuedAt)
  const agentStats = buildRunStats(telemetry['rl-agent'], comparisonRun?.issuedAt)

  let comparisonHeadline = 'Send a shared command to compare runtime.'
  if (comparisonRun) {
    if (qwenStats.completionMs != null && agentStats.completionMs != null) {
      const fasterJob = qwenStats.completionMs <= agentStats.completionMs ? panes[0].title : panes[1].title
      const delta = Math.abs(qwenStats.completionMs - agentStats.completionMs)
      comparisonHeadline = `${fasterJob} finished ${formatMs(delta)} faster.`
    } else if (qwenStats.responseMs != null && agentStats.responseMs != null) {
      const fasterJob = qwenStats.responseMs <= agentStats.responseMs ? panes[0].title : panes[1].title
      const delta = Math.abs(qwenStats.responseMs - agentStats.responseMs)
      comparisonHeadline = `${fasterJob} responded ${formatMs(delta)} faster.`
    } else {
      comparisonHeadline = `Running shared command: ${comparisonRun.command}`
    }
  }

  return (
    <main className="desktop">
      <div className="desktop__glow" />

      <section className="comparison-bar">
        <div className="comparison-bar__copy">
          <span className="comparison-bar__eyebrow">Runtime compare</span>
          <strong>{comparisonHeadline}</strong>
          <small>{comparisonRun ? `Command: ${comparisonRun.command}` : 'Broadcast one command to both panes.'}</small>
        </div>

        <div className="comparison-bar__stats">
          <article className="comparison-card comparison-card--cyan">
            <span>{panes[0].title}</span>
            <strong>{formatMs(qwenStats.completionMs ?? qwenStats.responseMs)}</strong>
            <small>{qwenStats.completionMs != null ? 'completion time' : 'first output latency'}</small>
          </article>

          <article className="comparison-card comparison-card--green">
            <span>{panes[1].title}</span>
            <strong>{formatMs(agentStats.completionMs ?? agentStats.responseMs)}</strong>
            <small>{agentStats.completionMs != null ? 'completion time' : 'first output latency'}</small>
          </article>
        </div>
      </section>

      <section ref={workspaceRef} className="workspace">
        <div className="workspace__pane" style={{ width: `${split}%` }}>
          <TerminalPane ref={leftPaneRef} {...panes[0]} onTelemetryChange={handleTelemetryChange} />
        </div>

        <button
          type="button"
          className={dragging ? 'workspace__divider is-dragging' : 'workspace__divider'}
          onPointerDown={() => setDragging(true)}
          aria-label="Resize terminal panes"
          aria-valuemin={25}
          aria-valuemax={75}
          aria-valuenow={Math.round(split)}
          aria-orientation="vertical"
        >
          <span />
        </button>

        <div className="workspace__pane" style={{ width: `${100 - split}%` }}>
          <TerminalPane ref={rightPaneRef} {...panes[1]} onTelemetryChange={handleTelemetryChange} />
        </div>
      </section>

      <form className="broadcast-bar" onSubmit={handleBroadcast}>
        <label className="broadcast-bar__label" htmlFor="broadcast-input">
          Shared input
        </label>
        <div className="broadcast-bar__field">
          <span className="broadcast-bar__prompt">$</span>
          <input
            id="broadcast-input"
            value={command}
            onChange={(event) => setCommand(event.target.value)}
            placeholder="Send the same command to both terminals"
            spellCheck="false"
          />
          <button type="submit">Send to both</button>
        </div>
      </form>
    </main>
  )
}

export default App
