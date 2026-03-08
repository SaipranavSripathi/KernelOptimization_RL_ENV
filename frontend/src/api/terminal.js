const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const API_BASE = RAW_API_BASE.replace(/\/+$/, '')
const WS_BASE = API_BASE.replace(/^http/, 'ws')

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  })

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`
    try {
      const payload = await response.json()
      message = payload.detail || message
    } catch {
      // Keep default error message when the payload is not JSON.
    }
    throw new Error(message)
  }

  return response.json()
}

export function createOrAttachSession(jobId, { restart = false } = {}) {
  return request('/terminal/sessions', {
    method: 'POST',
    body: JSON.stringify({ job_id: jobId, restart }),
  })
}

export function stopTerminalSession(sessionId) {
  return request(`/terminal/sessions/${sessionId}/stop`, {
    method: 'POST',
  })
}

export function sendTerminalInput(sessionId, data, appendNewline = true) {
  return request(`/terminal/sessions/${sessionId}/input`, {
    method: 'POST',
    body: JSON.stringify({ data, append_newline: appendNewline }),
  })
}

export function resizeTerminalSession(sessionId, cols, rows) {
  return request(`/terminal/sessions/${sessionId}/resize`, {
    method: 'POST',
    body: JSON.stringify({ cols, rows }),
  })
}

export function openTerminalSocket(sessionId) {
  return new WebSocket(`${WS_BASE}/terminal/sessions/${sessionId}/stream`)
}
