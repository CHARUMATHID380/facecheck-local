'use client'

import { ChangeEvent, useCallback, useEffect, useRef, useState } from 'react'
import {
  Check,
  LockKeyhole,
  RotateCcw,
  ScanFace,
  Upload,
  UserRound,
  X,
} from 'lucide-react'
import Topbar from '@/components/topbar'

// ─── Types ────────────────────────────────────────────────────────────────────

type Candidate = {
  face_id: string
  name: string
  distance: number
  similarity: number
}

type IdentifyResult = {
  identified: boolean
  name: string
  face_id: string | null
  similarity: number
  distance: number
  candidates: Candidate[]
  message: string
}

type Toast = { kind: 'ok' | 'err' | 'info'; text: string } | null
type Tone = 'idle' | 'success' | 'danger' | 'loading'

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function IdentifyPage() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [threshold, setThreshold] = useState(0.40)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<IdentifyResult | null>(null)
  const [tone, setTone] = useState<Tone>('idle')
  const [toast, setToast] = useState<Toast>(null)
  const [enrolledCount, setEnrolledCount] = useState<number | null>(null)

  const previewRef = useRef('')
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  // ── show self-dismissing toast ────────────────────────────────────────────
  const showToast = useCallback((kind: Toast['kind'], text: string) => {
    setToast({ kind: kind!, text })
    if (toastTimer.current) clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(null), 5000)
  }, [])

  // ── fetch enrolled count for info strip ──────────────────────────────────
  useEffect(() => {
    fetch('/api/enrolled-faces', { cache: 'no-store' })
      .then(r => r.json())
      .then((d: unknown) => Array.isArray(d) && setEnrolledCount(d.length))
      .catch(() => {})
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current)
      if (toastTimer.current) clearTimeout(toastTimer.current)
    }
  }, [])

  // ── image picker ──────────────────────────────────────────────────────────
  const handleImageChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (!file.type.startsWith('image/')) {
      showToast('err', 'Please choose a JPG, PNG, or WEBP image.')
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      showToast('err', 'Image must be under 10 MB.')
      return
    }
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    const url = URL.createObjectURL(file)
    previewRef.current = url
    setPreviewUrl(url)
    setImageFile(file)
    setResult(null)
    setTone('idle')
  }

  // ── identify ──────────────────────────────────────────────────────────────
  const handleIdentify = async () => {
    if (!imageFile) { showToast('err', 'Please choose a photo first.'); return }

    setLoading(true)
    setTone('loading')
    setResult(null)
    setToast(null)

    try {
      const body = new FormData()
      body.append('image', imageFile)
      body.append('threshold', String(threshold))
      body.append('top_k', '5')

      const res = await fetch('/api/identify', { method: 'POST', body })
      const data = await res.json()

      if (!res.ok) {
        showToast('err', data.error ?? 'Identification failed.')
        setTone('danger')
        return
      }

      const r = data as IdentifyResult
      setResult(r)
      setTone(r.identified ? 'success' : 'danger')
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      showToast('err', msg)
      setTone('danger')
    } finally {
      setLoading(false)
    }
  }

  // ─── Derived display values ───────────────────────────────────────────────
  const resultHeading = (() => {
    if (tone === 'loading') return 'Analysing…'
    if (!result) return 'Waiting for photo'
    return result.identified ? result.name : 'Unknown'
  })()

  const resultDetail = (() => {
    if (tone === 'loading') return 'Running face detection and matching.'
    if (!result) return 'Upload a photo and click Identify.'
    return result.message
  })()

  const simPct = result?.similarity ?? 0
  const isBadSim = !result?.identified

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <main className="app-shell">
      <Topbar />

      {/* Hero */}
      <section className="page-hero">
        <div className="eyebrow"><span className="eyebrow-dot" /> Face identification</div>
        <h1>Upload a face.<br /><em>Find the match.</em></h1>
        <p>Submit a photo and the system compares it against every enrolled person. If nobody matches above the threshold, the face is labelled unknown.</p>
      </section>

      <section className="page-workspace" aria-label="Identification workspace">

        {/* ── Left: form ── */}
        <div>
          <div className="panel form-panel">
            <h2>Identify a face</h2>
            <p>Upload the photo you want to identify against the database.</p>

            {/* Image picker */}
            <label
              className={`upload-zone${previewUrl ? ' has-image' : ''}`}
              htmlFor="identify-image"
              aria-label="Upload photo to identify"
            >
              {previewUrl
                ? <><img src={previewUrl} alt="Photo to identify" /><span className="replace-hint"><RotateCcw size={13} /> Choose a different photo</span></>
                : <><span className="upload-icon"><Upload size={19} /></span><strong>Choose a photo to identify</strong><span>JPG, PNG or WEBP · up to 10 MB</span></>
              }
            </label>
            <input id="identify-image" type="file" accept="image/*" onChange={handleImageChange} />

            <div className="form-divider" />

            {/* Threshold slider */}
            <div className="field">
              <label htmlFor="threshold-slider" style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Match threshold</span>
                <strong style={{ color: 'var(--primary)' }}>{(threshold * 100).toFixed(0)}% similarity</strong>
              </label>
              <input
                id="threshold-slider"
                type="range"
                min={0.20} max={0.70} step={0.01}
                value={threshold}
                onChange={e => setThreshold(parseFloat(e.target.value))}
                style={{ accentColor: 'var(--primary)', width: '100%', cursor: 'pointer' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '.65rem', color: 'var(--muted-foreground)', marginTop: 2 }}>
                <span>Strict (20%)</span>
                <span>Default (40%)</span>
                <span>Lenient (70%)</span>
              </div>
            </div>

            {/* Info strip */}
            {enrolledCount !== null && (
              <div className="toast info" style={{ marginBottom: 18 }}>
                <UserRound size={14} />
                {enrolledCount === 0
                  ? <>No faces enrolled yet. <a href="/enroll" style={{ color: 'var(--primary)', fontWeight: 700 }}>Enroll someone first →</a></>
                  : <>{enrolledCount} face{enrolledCount !== 1 ? 's' : ''} in the database ready to match against.</>
                }
              </div>
            )}

            <button
              className="btn-submit"
              onClick={handleIdentify}
              disabled={loading || !imageFile}
            >
              {loading
                ? <><span style={{ width: 16, height: 16, border: '2px solid rgba(255,255,255,.4)', borderTopColor: '#fff', borderRadius: '50%', display: 'inline-block', animation: 'spin .7s linear infinite' }} /> Identifying…</>
                : <><ScanFace size={15} /> Identify face</>
              }
            </button>

            {/* Toast */}
            {toast && (
              <div className={`toast ${toast.kind}`} style={{ marginTop: 16 }} role="alert">
                {toast.kind === 'ok' ? <Check size={15} /> : <X size={15} />}
                {toast.text}
              </div>
            )}
          </div>
        </div>

        {/* ── Right: result ── */}
        <aside className="result-column-page">
          <div className="id-result-card">

            {/* Status badge */}
            <div className="result-top">
              <span className="result-kicker">IDENTIFICATION RESULT</span>
              <span className={`result-badge${tone === 'success' ? ' success' : tone === 'danger' ? ' danger' : ''}`}>
                <span />
                {tone === 'success' ? 'Identified' : tone === 'danger' ? 'Unknown' : tone === 'loading' ? 'Processing' : 'Standby'}
              </span>
            </div>

            {/* Symbol */}
            <div className={`id-symbol${tone === 'success' ? ' success' : tone === 'danger' ? ' danger' : tone === 'loading' ? ' loading' : ''}`} aria-hidden="true">
              {tone === 'success' ? <Check size={30} /> : tone === 'danger' ? <X size={30} /> : <UserRound size={30} />}
            </div>

            <h2>{resultHeading}</h2>
            <p>{resultDetail}</p>

            {/* Similarity bar */}
            {result && (
              <div className="sim-bar-wrap">
                <div className="sim-bar-label">
                  <span>Similarity</span>
                  <span style={{ color: isBadSim ? '#c98025' : 'var(--primary)', fontWeight: 800 }}>
                    {simPct.toFixed(1)}%
                  </span>
                </div>
                <div className="sim-bar-track">
                  <div
                    className={`sim-bar-fill${isBadSim ? ' danger' : ''}`}
                    style={{ width: `${simPct}%` }}
                    role="meter"
                    aria-valuenow={simPct}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  />
                </div>
              </div>
            )}

            <div className="result-divider" />

            {/* Meta */}
            <div className="meta-row">
              <span>Threshold</span>
              <strong>{(threshold * 100).toFixed(0)}% similarity</strong>
            </div>
            <div className="meta-row">
              <span>Distance</span>
              <strong>{result ? result.distance.toFixed(3) : '—'}</strong>
            </div>
            <div className="meta-row">
              <span>Processing</span>
              <strong><span className="local-dot" /> On this device</strong>
            </div>

            {/* Candidates */}
            {result && result.candidates.length > 0 && (
              <div className="candidates">
                <div className="result-divider" style={{ marginBottom: 12 }} />
                <div className="candidates-title">TOP CANDIDATES</div>
                {result.candidates.map((c, i) => (
                  <div
                    key={c.face_id}
                    className={`candidate-row${i === 0 && result.identified ? ' best' : ''}`}
                  >
                    <span className="candidate-name">
                      {i === 0 && result.identified && <Check size={11} style={{ marginRight: 4, verticalAlign: -1, color: 'var(--primary)' }} />}
                      {c.name}
                    </span>
                    <span style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                      <span className="candidate-sim">{c.similarity.toFixed(1)}%</span>
                      <span className="candidate-dist">d={c.distance.toFixed(3)}</span>
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Privacy note */}
          <div className="trust-card">
            <LockKeyhole size={17} />
            <div>
              <strong>Your privacy comes first</strong>
              <p>Images are processed locally. No face data is sent to any external server.</p>
            </div>
          </div>

          {/* Tips */}
          <div className="tips-card">
            <span className="tips-title">For best results</span>
            <div><span>01</span><p>Use a clear, front-facing photo with even lighting.</p></div>
            <div><span>02</span><p>Make sure the person is already enrolled in the database.</p></div>
            <div><span>03</span><p>Lower the threshold to be stricter; raise it to be more lenient.</p></div>
          </div>
        </aside>
      </section>

      <footer>
        <span><LockKeyhole size={13} /> No images are uploaded to any server</span>
        <span>Matching against local face_database.json</span>
      </footer>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </main>
  )
}
