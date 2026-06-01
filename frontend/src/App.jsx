import { useState, useRef, useEffect } from 'react'
import './App.css'

function Sidebar({ onLoad, onIngest, status }) {
  const [files, setFiles] = useState([])

  return (
    <aside className="sidebar">
      <div className="sidebar-title">RAG System</div>

      <section className="sidebar-section">
        <label className="section-label">Documents</label>
        <label className="file-drop">
          <input
            type="file"
            multiple
            accept=".pdf,.docx,.txt"
            onChange={e => setFiles(Array.from(e.target.files))}
          />
          <span className="file-icon">📄</span>
          <span>{files.length > 0 ? `${files.length} file(s) selected` : 'Click to upload PDF / DOCX / TXT'}</span>
        </label>
        <button
          className="btn btn-primary"
          onClick={() => onIngest(files)}
          disabled={files.length === 0}
        >
          Ingest Files
        </button>
      </section>

      <div className="divider"><span>or</span></div>

      <button className="btn btn-secondary" onClick={onLoad}>
        Load Existing Chunks
      </button>

      {status && (
        <div className={`status-box ${status.type}`}>
          {status.message}
        </div>
      )}
    </aside>
  )
}

function Message({ role, content, sources }) {
  return (
    <div className={`message ${role}`}>
      <div className="bubble">
        <p className="message-text">{content}</p>
        {sources && sources.length > 0 && (
          <details className="sources">
            <summary>Sources ({sources.length})</summary>
            <div className="source-list">
              {sources.map((s, i) => (
                <div key={i} className="source-item">
                  <span className="source-ref">[{i + 1}]</span>
                  <div>
                    <div className="source-meta">{s.source} — p.{s.page}</div>
                    <div className="source-text">{s.text}…</div>
                  </div>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="message assistant">
      <div className="bubble typing">
        <span /><span /><span />
      </div>
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [docStatus, setDocStatus] = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleLoad = async () => {
    setDocStatus({ type: 'loading', message: 'Loading chunks…' })
    try {
      const res = await fetch('/load', { method: 'POST' })
      const data = await res.json()
      if (data.error) setDocStatus({ type: 'error', message: data.error })
      else setDocStatus({ type: 'success', message: `✓ Loaded ${data.chunks} chunks. Ready to chat.` })
    } catch (e) {
      setDocStatus({ type: 'error', message: e.message })
    }
  }

  const handleIngest = async (files) => {
    setDocStatus({ type: 'loading', message: `Ingesting ${files.length} file(s)…` })
    const form = new FormData()
    files.forEach(f => form.append('files', f))
    try {
      const res = await fetch('/ingest', { method: 'POST', body: form })
      const data = await res.json()
      if (data.error) setDocStatus({ type: 'error', message: data.error })
      else setDocStatus({ type: 'success', message: `✓ Ingested ${data.chunks} chunks. Ready to chat.` })
    } catch (e) {
      setDocStatus({ type: 'error', message: e.message })
    }
  }

  const sendQuery = async () => {
    if (!input.trim() || loading) return
    const query = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: query }])
    setLoading(true)

    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      const data = await res.json()
      const reply = data.error
        ? { role: 'assistant', content: `Error: ${data.error}` }
        : { role: 'assistant', content: data.answer, sources: data.sources }
      setMessages(prev => [...prev, reply])
    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${e.message}` }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="layout">
      <Sidebar onLoad={handleLoad} onIngest={handleIngest} status={docStatus} />

      <main className="chat-pane">
        <div className="messages-scroll">
          {messages.length === 0 && (
            <div className="empty">
              <div className="empty-icon">💬</div>
              <p>Load documents from the sidebar, then ask a question.</p>
            </div>
          )}
          {messages.map((m, i) => <Message key={i} {...m} />)}
          {loading && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>

        <div className="input-bar">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendQuery()}
            placeholder="Ask a question about your documents…"
            disabled={loading}
          />
          <button
            className="btn btn-primary send-btn"
            onClick={sendQuery}
            disabled={loading || !input.trim()}
          >
            Send
          </button>
        </div>
      </main>
    </div>
  )
}
