'use client'

import { ChangeEvent, useCallback, useEffect, useRef, useState } from 'react'
import { Camera, Check, FileImage, LockKeyhole, Play, RotateCcw, ShieldCheck, Square, Upload, UserRound, Video, X } from 'lucide-react'
import Link from 'next/link'

type FaceApi = {
  nets: {
    tinyFaceDetector: { loadFromUri: (uri: string) => Promise<void> }
    faceLandmark68Net: { loadFromUri: (uri: string) => Promise<void> }
    faceRecognitionNet: { loadFromUri: (uri: string) => Promise<void> }
  }
  TinyFaceDetectorOptions: new (options: { inputSize: number; scoreThreshold: number }) => unknown
  bufferToImage: (file: File) => Promise<unknown>
  detectSingleFace: (input: unknown, options: unknown) => {
    withFaceLandmarks: () => { withFaceDescriptor: () => Promise<any> }
  }
  resizeResults: (detection: any, dimensions: { width: number; height: number }) => any
  euclideanDistance: (first: Float32Array, second: Float32Array) => number
}
type ResultTone = 'idle' | 'success' | 'danger'

const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights'
const MATCH_THRESHOLD = 0.55

export default function Page() {
  const [faceApi, setFaceApi] = useState<FaceApi | null>(null)
  const [referenceUrl, setReferenceUrl] = useState('')
  const [referenceDescriptor, setReferenceDescriptor] = useState<Float32Array | null>(null)
  const [referenceStatus, setReferenceStatus] = useState('Add a clear, front-facing photo to begin.')
  const [cameraStatus, setCameraStatus] = useState('Camera is ready when you are.')
  const [result, setResult] = useState('Waiting for recognition')
  const [resultDetail, setResultDetail] = useState('Start your camera after adding a reference photo.')
  const [tone, setTone] = useState<ResultTone>('idle')
  const [loading, setLoading] = useState(false)
  const [cameraRunning, setCameraRunning] = useState(false)
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [error, setError] = useState('')

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const referenceUrlRef = useRef('')

  const loadFaceApi = useCallback(async () => {
    if (faceApi && modelsLoaded) return faceApi
    setLoading(true)
    setCameraStatus('Loading recognition models…')
    const script = document.createElement('script')
    script.src = 'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js'
    script.async = true
    await new Promise<void>((resolve, reject) => {
      script.onload = () => resolve()
      script.onerror = () => reject(new Error('Unable to load recognition models'))
      document.head.appendChild(script)
    })
    const api = (window as Window & { faceapi?: FaceApi }).faceapi
    if (!api) throw new Error('Recognition library unavailable')
    await Promise.all([
      api.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      api.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      api.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
    ])
    setFaceApi(api)
    setModelsLoaded(true)
    setLoading(false)
    return api
  }, [faceApi, modelsLoaded])

  const recognizeFace = useCallback(async () => {
    const api = faceApi
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!api || !video || !canvas || !referenceDescriptor || video.readyState < 2) return
    const detection = await api.detectSingleFace(video, new api.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 })).withFaceLandmarks().withFaceDescriptor()
    const context = canvas.getContext('2d')
    if (!context) return
    context.clearRect(0, 0, canvas.width, canvas.height)
    if (!detection) {
      setResult('No face detected')
      setResultDetail('Move into the frame and keep your face well lit.')
      setTone('idle')
      return
    }
    const resized = api.resizeResults(detection, { width: canvas.width, height: canvas.height })
    const box = resized.detection.box
    const distance = api.euclideanDistance(referenceDescriptor, detection.descriptor)
    const confidence = Math.max(0, Math.min(100, Math.round((1 - distance) * 100)))
    const isMatch = distance < MATCH_THRESHOLD
    context.strokeStyle = isMatch ? '#16a394' : '#c98025'
    context.lineWidth = 4
    context.strokeRect(box.x, box.y, box.width, box.height)
    setTone(isMatch ? 'success' : 'danger')
    setResult(isMatch ? 'Match found' : 'No match')
    setResultDetail(`${confidence}% similarity · ${isMatch ? 'The faces appear to belong to the same person.' : 'Try a clearer reference photo or better lighting.'}`)
  }, [faceApi, referenceDescriptor])

  const stopCamera = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    const context = canvasRef.current?.getContext('2d')
    if (context && canvasRef.current) context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    setCameraRunning(false)
    setCameraStatus('Camera is ready when you are.')
    setResult('Waiting for recognition')
    setResultDetail('Start your camera after adding a reference photo.')
    setTone('idle')
  }, [])

  const handleReference = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setReferenceStatus('Please choose an image file.')
      return
    }
    if (referenceUrlRef.current) URL.revokeObjectURL(referenceUrlRef.current)
    const url = URL.createObjectURL(file)
    referenceUrlRef.current = url
    setReferenceUrl(url)
    setReferenceDescriptor(null)
    setReferenceStatus('Analyzing reference photo…')
    setError('')
    try {
      const api = await loadFaceApi()
      const image = await api.bufferToImage(file)
      const detection = await api.detectSingleFace(image, new api.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 })).withFaceLandmarks().withFaceDescriptor()
      if (!detection) {
        setReferenceStatus('No face found. Try a clearer, front-facing photo.')
        return
      }
      setReferenceDescriptor(detection.descriptor)
      setReferenceStatus('Reference ready for verification.')
      setCameraStatus('Reference ready. Camera can now be started.')
    } catch {
      setReferenceStatus('Could not process this image. Please try another one.')
      setError('Recognition models could not be loaded. Check your connection and try again.')
    } finally {
      setLoading(false)
    }
  }

  const startCamera = async () => {
    if (!referenceDescriptor) {
      setError('Add a reference photo before starting recognition.')
      return
    }
    try {
      setError('')
      const api = await loadFaceApi()
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
      streamRef.current = stream
      const video = videoRef.current
      if (!video) return
      video.srcObject = stream
      await new Promise<void>((resolve) => { video.onloadedmetadata = () => resolve() })
      await video.play()
      if (canvasRef.current) {
        canvasRef.current.width = video.videoWidth
        canvasRef.current.height = video.videoHeight
      }
      setCameraRunning(true)
      setCameraStatus('Camera is live. Looking for a face…')
      setResult('Looking for a face')
      setResultDetail('Keep your face inside the frame.')
      intervalRef.current = setInterval(() => { void recognizeFace() }, 700)
      void api
    } catch {
      setError('Camera access failed. Allow camera permissions and try again.')
      setCameraStatus('Camera permission is needed to continue.')
      setTone('danger')
    }
  }

  useEffect(() => () => {
    stopCamera()
    if (referenceUrlRef.current) URL.revokeObjectURL(referenceUrlRef.current)
  }, [stopCamera])

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Verifai home"><span className="brand-mark"><ShieldCheck size={18} /></span><span>verifai</span></a>
        <nav className="nav-links" aria-label="Main navigation">
          <Link href="/enroll" className="nav-link"><span style={{display:'inline-flex',alignItems:'center',gap:5}}><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><line x1="19" y1="8" x2="19" y2="14"/><line x1="22" y1="11" x2="16" y2="11"/></svg>Enroll</span></Link>
          <Link href="/identify" className="nav-link"><span style={{display:'inline-flex',alignItems:'center',gap:5}}><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="3"/></svg>Identify</span></Link>
        </nav>
        <div className="privacy-pill"><LockKeyhole size={14} /> Local-only processing</div>
      </header>

      <section className="hero" id="top">
        <div className="eyebrow"><span className="eyebrow-dot" /> Private identity check</div>
        <h1>Verify a face.<br /><em>Keep the data.</em></h1>
        <p>Compare a reference photo with your camera in seconds. Everything stays in your browser — no uploads, no accounts, no guesswork.</p>
      </section>

      <section className="workspace" aria-label="Face verification workspace">
        <div className="step-column">
          <div className="step-label"><span>01</span><div><strong>Reference photo</strong><small>Tell us who to look for</small></div></div>
          <div className="panel reference-panel">
            <div className="panel-head"><div><h2>Upload a clear photo</h2><p>Front-facing works best, with one person in frame.</p></div><FileImage className="panel-icon" size={21} /></div>
            <label className={`upload-zone ${referenceUrl ? 'has-image' : ''}`} htmlFor="reference-image">
              {referenceUrl ? <img src={referenceUrl} alt="Selected reference" /> : <><span className="upload-icon"><Upload size={20} /></span><strong>Choose a reference photo</strong><span>JPG, PNG or WEBP · up to 10 MB</span></>}
              {referenceUrl && <span className="replace-hint"><RotateCcw size={14} /> Choose a different photo</span>}
            </label>
            <input id="reference-image" type="file" accept="image/*" onChange={handleReference} />
            <div className={`inline-status ${referenceDescriptor ? 'ready' : ''}`}><span className="status-dot" />{referenceStatus}</div>
          </div>
          <div className="connector" aria-hidden="true" />
          <div className="step-label"><span>02</span><div><strong>Live verification</strong><small>Check against your camera</small></div></div>
          <div className="panel camera-panel">
            <div className="camera-stage">
              <video ref={videoRef} muted playsInline aria-label="Live camera preview" />
              <canvas ref={canvasRef} aria-hidden="true" />
              {!cameraRunning && <div className="camera-empty"><span className="camera-icon"><Camera size={24} /></span><strong>Your camera preview will appear here</strong><span>We only access it after you start verification.</span></div>}
              <div className="stage-label"><span className={`live-dot ${cameraRunning ? 'active' : ''}`} />{cameraRunning ? 'LIVE' : 'READY'}</div>
            </div>
            <div className="camera-controls"><div className="camera-status"><span className="status-dot" />{cameraStatus}</div><div className="button-group"><button className="button primary" onClick={startCamera} disabled={cameraRunning || loading || !referenceDescriptor}><Play size={15} fill="currentColor" /> Start camera</button><button className="button secondary" onClick={stopCamera} disabled={!cameraRunning}><Square size={14} fill="currentColor" /> Stop</button></div></div>
          </div>
        </div>

        <aside className="result-column">
          <div className="result-card">
            <div className="result-top"><span className="result-kicker">VERIFICATION STATUS</span><span className={`result-badge ${tone}`}><span />{tone === 'success' ? 'Verified' : tone === 'danger' ? 'Review' : 'Standby'}</span></div>
            <div className={`result-symbol ${tone}`} aria-hidden="true">{tone === 'success' ? <Check size={34} /> : tone === 'danger' ? <X size={34} /> : <UserRound size={34} />}</div>
            <h2>{result}</h2><p>{resultDetail}</p>
            <div className="result-divider" /><div className="result-meta"><span>Threshold</span><strong>55% similarity</strong></div><div className="result-meta"><span>Processing</span><strong><span className="local-dot" /> On this device</strong></div>
          </div>
          <div className="trust-card"><ShieldCheck size={18} /><div><strong>Your privacy comes first</strong><p>Face data is processed locally and never leaves this browser tab.</p></div></div>
          {error && <div className="error-card" role="alert"><X size={16} />{error}</div>}
          <div className="tips-card"><span className="tips-title">For the best result</span><div><span>01</span><p>Use even lighting and face the camera directly.</p></div><div><span>02</span><p>Remove sunglasses, masks, or anything covering your face.</p></div></div>
        </aside>
      </section>

      <footer><span><LockKeyhole size={13} /> No images or video are uploaded</span><span>Built for private, one-to-one checks</span></footer>
    </main>
  )
}
