'use client'

import { ChangeEvent, useCallback, useEffect, useRef, useState } from 'react'
import {
  Check,
  FileImage,
  LockKeyhole,
  RotateCcw,
  Trash2,
  Upload,
  UserPlus,
  UserRound,
  X,
} from 'lucide-react'
import Topbar from '@/components/topbar'

// ─── Types ───────────────────────────────────────────────────────────────────

type EnrollResult = {
  face_id: string
  name: string
  enrolled_at: string
  message: string
}

type EnrolledFace = {
  face_id: string
  name: string
  enrolled_at: string
}

type Toast = { kind: 'ok' | 'err' | 'info'; text: string } | null

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatDate(iso: string) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short',
    })
  } catch {
    return iso
  }
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function EnrollPage() {
  const [name, setName] = useState('')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<EnrollResult | null>(null)
  const [toast, setToast] = useState<Toast>(null)
  const [enrolled, setEnrolled] = useState<EnrolledFace[]>([])
  const [listLoading, setListLoading] = useState(true)
  const [replace, setReplace] = useState(false)

  const previewRef = useRef('')
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  // ── show a self-dismissing toast ─────────────────────────────────────────
  const showToast = useCallback((kind: Toast['kind'], text: string) => {
    setToast({ kind: kind!, text })
    if (toastTimer.current) clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(null), 5000)
  }, [])

  // ── fetch enrolled list ───────────────────────────────────────────────────
  const fetchEnrolled = useCallback(async () => {
    setListLoading(true)
    try {
      const res = await fetch('/api/enrolled-faces', { cache: 'no-store' })
      if (!res.ok) throw new Error('Failed to load enrolled faces')
      const data: EnrolledFace[] = await res.json()
      setEnrolled(data)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      showToast('err', msg)
    } finally {
      setListLoading(false)
    }
  }, [showToast])

  useEffect(() => {
    void fetchEnrolled()
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current)
      if (toastTimer.current) clearTimeout(toastTimer.current)
    }
  }, [fetchEnrolled])

  // ── image picker ─────────────────────────────────────────────────────────
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
  }

  // ── enroll ────────────────────────────────────────────────────────────────
  const handleEnroll = async () => {
    if (!imageFile) { showToast('err', 'Please choose a photo first.'); return }
    if (!name.trim()) { showToast('err', 'Please enter a name.'); return }

    setLoading(true)
    setResult(null)
    setToast(null)

    try {
      const body = new FormData()
      body.append('image', imageFile)
      body.append('name', name.trim())
      body.append('replace', String(replace))

      const res = await fetch('/api/enroll', { method: 'POST', body })
      const data = await res.json()

      if (!res.ok) {
        showToast('err', data.error ?? 'Enrollment failed.')
        return
      }

      setResult(data as EnrollResult)
      showToast('ok', `'${data.name}' enrolled successfully.`)
      // reset form
      setName('')
      setImageFile(null)
      if (previewRef.current) URL.revokeObjectURL(previewRef.current)
      setPreviewUrl('')
      previewRef.current = ''
      setReplace(false)
      await fetchEnrolled()
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      showToast('err', msg)
    } finally {
      setLoading(false)
    }
  }

  // ── delete one face ───────────────────────────────────────────────────────
  const handleDelete = async (face_id: string, faceName: string) => {
    if (!window.confirm(`Remove '${faceName}' from the database?`)) return
    try {
      const res = await fetch(`/api/delete-face/${face_id}`, { method: 'DELETE' })
      const data = await res.json()
      if (!res.ok) { showToast('err', data.error ?? 'Delete failed.'); return }
      showToast('ok', data.message)
      await fetchEnrolled()
    } catch (e: unknown) {
      showToast('err', e instanceof Error ? e.message : String(e))
    }
  }

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <main className="app-shell">
      <Topbar />

      {/* Hero */}
      <section className="page-hero">
        <div className="eyebrow"><span className="eyebrow-dot" /> Face enrollment</div>
        <h1>Add a face.<br /><em>Build your database.</em></h1>
        <p>Upload a clear, front-facing photo and give the person a name. Their face embedding is stored locally for identification.</p>
      </section>

      <section className="page-workspace" aria-label="Enrollment workspace">

        {/* ── Left: form ── */}
        <div>
          <div className="panel form-panel">
            <h2>Enroll a new person</h2>
            <p>One person per photo. Front-facing, good lighting.</p>

            {/* Image picker */}
            <label
              className={`upload-zone${previewUrl ? ' has-image' : ''}`}
              htmlFor="enroll-image"
              aria-label="Upload reference photo"
            >
              {previewUrl
                ? <><img src={previewUrl} alt="Selected photo" /><span className="replace-hint"><RotateCcw size={13} /> Choose a different photo</span></>
                : <><span className="upload-icon"><Upload size={19} /></span><strong>Choose a photo</strong><span>JPG, PNG or WEBP · up to 10 MB</span></>
              }
            </label>
            <input id="enroll-image" type="file" accept="image/*" onChange={handleImageChange} />

            <div className="form-divider" />

            {/* Name field */}
            <div className="field">
              <label htmlFor="enroll-name">Full name</label>
              <input
                id="enroll-name"
                type="text"
                placeholder="e.g. Alice Johnson"
                value={name}
                onChange={e => setName(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && void handleEnroll()}
                maxLength={100}
                autoComplete="off"
              />
            </div>

            {/* Replace toggle */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 20 }}>
              <input
                id="replace-toggle"
                type="checkbox"
                checked={replace}
                onChange={e => setReplace(e.target.checked)}
                style={{ width: 15, height: 15, accentColor: 'var(--primary)', cursor: 'pointer' }}
              />
              <label htmlFor="replace-toggle" style={{ fontSize: '.74rem', color: 'var(--muted-foreground)', cursor: 'pointer' }}>
                Overwrite if this name already exists
              </label>
            </div>

            <button
              className="btn-submit"
              onClick={handleEnroll}
              disabled={loading || !imageFile || !name.trim()}
            >
              {loading
                ? <><span style={{ width: 16, height: 16, border: '2px solid rgba(255,255,255,.4)', borderTopColor: '#fff', borderRadius: '50%', display: 'inline-block', animation: 'spin .7s linear infinite' }} /> Enrolling…</>
                : <><UserPlus size={15} /> Enroll person</>
              }
            </button>

            {/* Toast */}
            {toast && (
              <div className={`toast ${toast.kind}`} style={{ marginTop: 16 }} role="alert">
                {toast.kind === 'ok' ? <Check size={15} /> : <X size={15} />}
                {toast.text}
              </div>
            )}

            {/* Success detail */}
            {result && (
              <div style={{ marginTop: 18 }}>
                <div className="form-divider" />
                <p style={{ fontSize: '.72rem', color: 'var(--muted-foreground)', margin: '0 0 6px' }}>ENROLLED</p>
                <div className="enrolled-row" style={{ background: '#e2f4ed', borderColor: '#c0e5d5' }}>
                  <span className="enrolled-name">{result.name}</span>
                  <span className="enrolled-date">{formatDate(result.enrolled_at)}</span>
                </div>
                <p style={{ fontSize: '.67rem', color: '#78918b', marginTop: 6 }}>ID: {result.face_id}</p>
              </div>
            )}
          </div>
        </div>

        {/* ── Right: enrolled list ── */}
        <aside className="result-column-page">
          <div className="id-result-card">
            <div className="result-top">
              <span className="result-kicker">ENROLLED DATABASE</span>
              <span className="result-badge">
                <span />
                {enrolled.length} {enrolled.length === 1 ? 'person' : 'people'}
              </span>
            </div>

            <div className="id-symbol" style={{ width: 56, height: 56, marginTop: 28, marginBottom: 14 }} aria-hidden="true">
              <FileImage size={26} />
            </div>
            <h2 style={{ textAlign: 'center', fontFamily: 'var(--font-serif)', fontSize: '1.6rem', fontWeight: 400, letterSpacing: '-.04em', margin: 0 }}>
              {listLoading ? 'Loading…' : enrolled.length === 0 ? 'No faces yet' : `${enrolled.length} enrolled`}
            </h2>
            <p style={{ textAlign: 'center', fontSize: '.75rem', color: 'var(--muted-foreground)', marginTop: 6, minHeight: 36 }}>
              {enrolled.length === 0 && !listLoading
                ? 'Enroll your first person using the form.'
                : 'All face embeddings are stored locally.'}
            </p>

            <div className="result-divider" />

            {listLoading ? (
              <div className="empty-state">Loading enrolled faces…</div>
            ) : enrolled.length === 0 ? (
              <div className="empty-state">No faces enrolled yet.</div>
            ) : (
              <div className="enrolled-list">
                {enrolled.map(f => (
                  <div className="enrolled-row" key={f.face_id}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                      <span style={{ display: 'grid', placeItems: 'center', width: 30, height: 30, background: 'var(--muted)', borderRadius: 8 }}>
                        <UserRound size={15} color="var(--primary)" />
                      </span>
                      <div>
                        <div className="enrolled-name">{f.name}</div>
                        <div className="enrolled-date">{formatDate(f.enrolled_at)}</div>
                      </div>
                    </div>
                    <button
                      className="enrolled-delete"
                      onClick={() => void handleDelete(f.face_id, f.name)}
                      aria-label={`Remove ${f.name}`}
                      title={`Remove ${f.name}`}
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Privacy note */}
          <div className="trust-card">
            <LockKeyhole size={17} />
            <div>
              <strong>Stored locally</strong>
              <p>Face embeddings never leave your machine. No cloud upload, no external storage.</p>
            </div>
          </div>
        </aside>
      </section>

      <footer>
        <span><LockKeyhole size={13} /> No images are uploaded to any server</span>
        <span>Embeddings stored in face_database.json</span>
      </footer>

      {/* Spinner keyframe */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </main>
  )
}
