import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from 'react'
import { useTerminalSession } from '../hooks/useTerminalSession'

function formatTime(timestamp) {
  if (!timestamp) {
    return 'Idle'
  }
  return new Date(timestamp * 1000).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function statusLabel(status) {
  if (status === 'running') {
    return 'Running'
  }
  if (status === 'failed') {
    return 'Failed'
  }
  if (status === 'exited') {
    return 'Completed'
  }
  return 'Starting'
}

const TerminalPane = forwardRef(function TerminalPane({ jobId, title, tone, onTelemetryChange }, ref) {
  const { session, buffer, connectionState, error, lastOutputAt, restart, resize, sendInput, start, stop } =
    useTerminalSession(jobId)
  const viewportRef = useRef(null)
  const scrollRef = useRef(null)

  useImperativeHandle(
    ref,
    () => ({
      submit: async (value) => {
        await sendInput(value, true)
      },
    }),
    [sendInput],
  )

  useEffect(() => {
    onTelemetryChange?.({
      jobId,
      session,
      connectionState,
      error,
      lastOutputAt,
    })
  }, [connectionState, error, jobId, lastOutputAt, onTelemetryChange, session])

  useEffect(() => {
    const container = scrollRef.current
    if (container) {
      container.scrollTop = container.scrollHeight
    }
  }, [buffer])

  useEffect(() => {
    const element = viewportRef.current
    if (!element) {
      return undefined
    }

    let frameId = 0
    const measure = () => {
      cancelAnimationFrame(frameId)
      frameId = requestAnimationFrame(() => {
        const style = getComputedStyle(element)
        const fontSize = parseFloat(style.fontSize) || 15
        const lineHeight = parseFloat(style.lineHeight) || 24
        const cols = Math.max(48, Math.floor(element.clientWidth / (fontSize * 0.61)))
        const rows = Math.max(14, Math.floor(element.clientHeight / lineHeight))
        resize(cols, rows)
      })
    }

    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(element)

    return () => {
      cancelAnimationFrame(frameId)
      observer.disconnect()
    }
  }, [resize])

  const footerMeta = useMemo(
    () => [
      session?.status ? statusLabel(session.status) : 'Connecting',
      session?.started_at ? `Started ${formatTime(session.started_at)}` : null,
      session?.exit_code != null ? `Exit ${session.exit_code}` : null,
      connectionState === 'connected' ? 'WS live' : connectionState,
    ].filter(Boolean),
    [connectionState, session],
  )

  return (
    <article className={`terminal-pane terminal-pane--${tone}`}>
      <header className="terminal-pane__header">
        <div className="terminal-pane__heading">
          <div className="terminal-pane__title-row">
            <span className="terminal-pane__dot" />
            <h2>{title}</h2>
            <span className={`status-chip status-chip--${session?.status || 'starting'}`}>
              {statusLabel(session?.status)}
            </span>
          </div>
          <p>{session?.command || 'Waiting for backend session...'}</p>
          <small>{session?.cwd || 'No working directory available yet.'}</small>
        </div>

        <div className="terminal-pane__actions">
          <button type="button" onClick={start}>
            Attach
          </button>
          <button type="button" onClick={restart}>
            Restart
          </button>
          <button type="button" onClick={stop}>
            Stop
          </button>
        </div>
      </header>

      <div ref={viewportRef} className="terminal-pane__viewport">
        <div ref={scrollRef} className="terminal-pane__scroll">
          <pre className="terminal-pane__buffer">{buffer || 'Starting session...\n'}</pre>
          {session?.status === 'running' ? <span className="terminal-pane__cursor" aria-hidden="true" /> : null}
        </div>
      </div>

      <footer className="terminal-pane__footer">
        <div className="terminal-pane__meta">
          {footerMeta.map((item) => (
            <span key={item}>{item}</span>
          ))}
          {error ? <span className="terminal-pane__error">{error}</span> : null}
        </div>
      </footer>
    </article>
  )
})

export default TerminalPane
