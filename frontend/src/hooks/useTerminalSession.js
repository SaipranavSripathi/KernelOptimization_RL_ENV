import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createOrAttachSession,
  openTerminalSocket,
  resizeTerminalSession,
  sendTerminalInput,
  stopTerminalSession,
} from '../api/terminal'

const BUFFER_LIMIT = 160000

function trimBuffer(text) {
  return text.length > BUFFER_LIMIT ? text.slice(-BUFFER_LIMIT) : text
}

export function useTerminalSession(jobId) {
  const [session, setSession] = useState(null)
  const [buffer, setBuffer] = useState('')
  const [connectionState, setConnectionState] = useState('connecting')
  const [error, setError] = useState('')
  const [lastOutputAt, setLastOutputAt] = useState(null)

  const socketRef = useRef(null)
  const resizeRef = useRef({ cols: null, rows: null })

  const attachSocket = useCallback((sessionId) => {
    if (socketRef.current) {
      socketRef.current.close()
    }

    const socket = openTerminalSocket(sessionId)
    socketRef.current = socket
    setConnectionState('connecting')

    socket.addEventListener('open', () => {
      setConnectionState('connected')
    })

    socket.addEventListener('message', (event) => {
      const payload = JSON.parse(event.data)

      if (payload.type === 'snapshot') {
        setSession(payload.session)
        setBuffer(payload.buffer || '')
        return
      }

      if (payload.type === 'output') {
        setLastOutputAt(Date.now())
        setBuffer((previous) => trimBuffer(previous + payload.data))
        return
      }

      if (payload.type === 'exit') {
        setSession((previous) =>
          previous
            ? {
                ...previous,
                status: payload.status,
                exit_code: payload.exit_code,
                finished_at: payload.finished_at,
              }
            : previous,
        )
      }
    })

    socket.addEventListener('close', () => {
      setConnectionState('disconnected')
    })

    socket.addEventListener('error', () => {
      setConnectionState('error')
    })
  }, [])

  const bootSession = useCallback(
    async (restart = false) => {
      try {
        setError('')
        const payload = await createOrAttachSession(jobId, { restart })
        setSession(payload.session)
        setBuffer(payload.buffer || '')
        attachSocket(payload.session.id)
      } catch (caughtError) {
        setError(caughtError.message)
        setConnectionState('error')
      }
    },
    [attachSocket, jobId],
  )

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      void bootSession(false)
    }, 0)

    return () => {
      window.clearTimeout(timeoutId)
      if (socketRef.current) {
        socketRef.current.close()
      }
    }
  }, [bootSession])

  const restart = useCallback(() => bootSession(true), [bootSession])

  const stop = useCallback(async () => {
    if (!session?.id) {
      return
    }
    try {
      await stopTerminalSession(session.id)
    } catch (caughtError) {
      setError(caughtError.message)
    }
  }, [session])

  const sendInput = useCallback(
    async (value, appendNewline = true) => {
      if (!session?.id || !value.trim()) {
        return
      }
      try {
        await sendTerminalInput(session.id, value, appendNewline)
      } catch (caughtError) {
        setError(caughtError.message)
      }
    },
    [session],
  )

  const resize = useCallback(
    async (cols, rows) => {
      if (!session?.id) {
        return
      }

      const previous = resizeRef.current
      if (previous.cols === cols && previous.rows === rows) {
        return
      }
      resizeRef.current = { cols, rows }

      try {
        await resizeTerminalSession(session.id, cols, rows)
      } catch {
        // Ignore resize errors so rendering stays responsive.
      }
    },
    [session],
  )

  return {
    buffer,
    connectionState,
    error,
    lastOutputAt,
    restart,
    resize,
    sendInput,
    session,
    start: () => bootSession(false),
    stop,
  }
}
