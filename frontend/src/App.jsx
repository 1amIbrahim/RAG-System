import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }, [text])
  return (
    <button className={`copy-btn ${copied ? 'copied' : ''}`} onClick={handleCopy} title="Copy response">
      {copied ? (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"/></svg>
      ) : (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
      )}
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

function SourceCard({ source, index }) {
  const [open, setOpen] = useState(false)
  const filename = source.source.split(/[/\\]/).pop()
  return (
    <div className={`source-card ${open ? 'open' : ''}`}>
      <button className="source-card-header" onClick={() => setOpen(o => !o)}>
        <span className="source-badge">[{index + 1}]</span>
        <span className="source-filename" title={source.source}>{filename}</span>
        <span className="source-page">p.{source.page}</span>
        <svg className="source-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <polyline points="6 9 12 15 18 9"/>
        </svg>
      </button>
      {open && <div className="source-body">{source.text}…</div>}
    </div>
  )
}

function Message({ role, content, sources, timestamp }) {
  const [sourcesOpen, setSourcesOpen] = useState(false)
  return (
    <div className={`message ${role}`}>
      <div className="avatar">{role === 'assistant' ? '✦' : 'You'}</div>
      <div className="bubble-wrap">
        <div className="bubble">
          {role === 'assistant' ? (
            <div className="message-md">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
            </div>
          ) : (
            <p className="message-text">{content}</p>
          )}
        </div>
        <div className="message-meta">
          <span className="timestamp">{formatTime(timestamp)}</span>
          {role === 'assistant' && <CopyButton text={content} />}
        </div>
        {sources && sources.length > 0 && (
          <div className="sources-section">
            <button className="sources-toggle" onClick={() => setSourcesOpen(o => !o)}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/>
              </svg>
              {sourcesOpen ? 'Hide' : 'Show'} {sources.length} source{sources.length !== 1 ? 's' : ''}
              <svg className={`toggle-chevron ${sourcesOpen ? 'open' : ''}`} width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <polyline points="6 9 12 15 18 9"/>
              </svg>
            </button>
            {sourcesOpen && (
              <div className="source-list">
                {sources.map((s, i) => <SourceCard key={i} source={s} index={i} />)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="message assistant">
      <div className="avatar">✦</div>
      <div className="bubble-wrap">
        <div className="bubble typing">
          <span /><span /><span />
        </div>
      </div>
    </div>
  )
}

const SUGGESTIONS = [
  'What are the main topics covered?',
  'Summarize the key findings',
  'What are the most important conclusions?',
  'List the key concepts mentioned',
]

function EmptyState({ onSuggest }) {
  return (
    <div className="empty">
      <div className="empty-icon-wrap">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/>
        </svg>
      </div>
      <h3 className="empty-title">Ask your documents anything</h3>
      <p className="empty-desc">Load or upload documents from the sidebar, then try one of these:</p>
      <div className="suggestions">
        {SUGGESTIONS.map(s => (
          <button key={s} className="suggestion-chip" onClick={() => onSuggest(s)}>
            {s}
          </button>
        ))}
      </div>
    </div>
  )
}

function FileItem({ name, onRemove }) {
  const ext = name.split('.').pop().toUpperCase()
  const extColors = { PDF: '#ef4444', DOCX: '#3b82f6', TXT: '#10b981' }
  return (
    <div className="file-item">
      <span className="file-ext" style={{ background: extColors[ext] || '#6b7280' }}>{ext}</span>
      <span className="file-name" title={name}>{name}</span>
      <button className="file-remove" onClick={onRemove} title="Remove">×</button>
    </div>
  )
}

function Sidebar({ onLoad, onIngest, status, isReady }) {
  const [files, setFiles] = useState([])
  const [dragging, setDragging] = useState(false)
  const ingesting = status?.type === 'loading'

  const addFiles = (newFiles) => {
    setFiles(prev => {
      const names = new Set(prev.map(f => f.name))
      return [...prev, ...newFiles.filter(f => !names.has(f.name))]
    })
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    addFiles(Array.from(e.dataTransfer.files).filter(f => /\.(pdf|docx|txt)$/i.test(f.name)))
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
          </svg>
        </div>
        <span className="sidebar-title">RAG System</span>
        {isReady && <span className="ready-dot" title="Documents loaded" />}
      </div>

      <section className="sidebar-section">
        <label className="section-label">Upload Documents</label>
        <label
          className={`file-drop ${dragging ? 'drag-over' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <input type="file" multiple accept=".pdf,.docx,.txt" onChange={e => addFiles(Array.from(e.target.files))} />
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
          </svg>
          <span className="file-drop-text">{dragging ? 'Drop files here' : 'Drag & drop or click'}</span>
          <span className="file-drop-hint">PDF · DOCX · TXT</span>
        </label>

        {files.length > 0 && (
          <div className="file-list">
            {files.map((f, i) => (
              <FileItem key={i} name={f.name} onRemove={() => setFiles(p => p.filter((_, j) => j !== i))} />
            ))}
          </div>
        )}

        <button className="btn btn-primary" onClick={() => onIngest(files)} disabled={files.length === 0 || ingesting}>
          {ingesting ? <><span className="btn-spinner" />Ingesting…</> : 'Ingest Files'}
        </button>
      </section>

      <div className="divider"><span>or</span></div>

      <button className="btn btn-secondary" onClick={onLoad} disabled={ingesting}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
        </svg>
        Load Existing Chunks
      </button>

      {status && status.type !== 'loading' && (
        <div className={`status-box ${status.type}`}>{status.message}</div>
      )}
      {status && status.type === 'loading' && (
        <div className="status-box loading">
          <span className="status-spinner" />
          {status.message}
        </div>
      )}
    </aside>
  )
}

function useAutoResize(value) {
  const ref = useRef(null)
  useEffect(() => {
    if (!ref.current) return
    ref.current.style.height = 'auto'
    ref.current.style.height = Math.min(ref.current.scrollHeight, 160) + 'px'
  }, [value])
  return ref
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [docStatus, setDocStatus] = useState(null)
  const [isReady, setIsReady] = useState(false)
  const bottomRef = useRef(null)
  const textareaRef = useAutoResize(input)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleLoad = async () => {
    setDocStatus({ type: 'loading', message: 'Loading chunks…' })
    try {
      const res = await fetch('/load', { method: 'POST' })
      const data = await res.json()
      if (data.error) { setDocStatus({ type: 'error', message: data.error }); setIsReady(false) }
      else { setDocStatus({ type: 'success', message: `Loaded ${data.chunks} chunks` }); setIsReady(true) }
    } catch (e) { setDocStatus({ type: 'error', message: e.message }); setIsReady(false) }
  }

  const handleIngest = async (files) => {
    setDocStatus({ type: 'loading', message: `Ingesting ${files.length} file(s)…` })
    const form = new FormData()
    files.forEach(f => form.append('files', f))
    try {
      const res = await fetch('/ingest', { method: 'POST', body: form })
      const data = await res.json()
      if (data.error) { setDocStatus({ type: 'error', message: data.error }); setIsReady(false) }
      else { setDocStatus({ type: 'success', message: `Ingested ${data.chunks} chunks` }); setIsReady(true) }
    } catch (e) { setDocStatus({ type: 'error', message: e.message }); setIsReady(false) }
  }

  const sendQuery = async (override) => {
    const query = (override ?? input).trim()
    if (!query || loading) return
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: query, timestamp: new Date() }])
    setLoading(true)
    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      const data = await res.json()
      const reply = data.error
        ? { role: 'assistant', content: `**Error:** ${data.error}`, timestamp: new Date() }
        : { role: 'assistant', content: data.answer, sources: data.sources, timestamp: new Date() }
      setMessages(prev => [...prev, reply])
    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', content: `**Error:** ${e.message}`, timestamp: new Date() }])
    } finally {
      setLoading(false)
    }
  }

  const userCount = messages.filter(m => m.role === 'user').length

  return (
    <div className="layout">
      <Sidebar onLoad={handleLoad} onIngest={handleIngest} status={docStatus} isReady={isReady} />

      <main className="chat-pane">
        <header className="chat-header">
          <div className="chat-header-left">
            <span className="chat-title">Chat</span>
            {userCount > 0 && <span className="chat-count">{userCount} question{userCount !== 1 ? 's' : ''}</span>}
          </div>
          {messages.length > 0 && (
            <button className="btn-ghost" onClick={() => setMessages([])}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M10 11v6M14 11v6M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2"/>
              </svg>
              Clear
            </button>
          )}
        </header>

        <div className="messages-scroll">
          {messages.length === 0 && <EmptyState onSuggest={sendQuery} />}
          {messages.map((m, i) => <Message key={i} {...m} />)}
          {loading && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>

        <div className="input-bar">
          <div className="input-wrapper">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery() } }}
              placeholder="Ask a question about your documents…"
              disabled={loading}
              rows={1}
            />
            <button
              className="send-btn"
              onClick={() => sendQuery()}
              disabled={loading || !input.trim()}
              aria-label="Send"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
              </svg>
            </button>
          </div>
          <p className="input-hint">Enter to send · Shift+Enter for new line</p>
        </div>
      </main>
    </div>
  )
}
