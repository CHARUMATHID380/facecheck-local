# Verifai — Face Recognition Identification System

A browser-based face verification tool extended with a full server-side **Face Recognition Identification System** (FRIS). The browser lets you do a quick private one-to-one check; the backend lets you **enroll a database of people and identify unknown faces against it**.

---

## Screenshots

| Home page | Reference upload | Live verification |
|---|---|---|
| ![Home](./home.png) | ![Upload](./reference-upload.png) | ![Live](./live-verification.png) |

---

## Architecture

```
┌──────────────────────────────────┐
│   Browser  (localhost:3000)      │
│   face-api.js  — live 1-to-1     │
│   verification (unchanged)       │
└──────────────────────────────────┘
          │  Next.js  /api/*  proxy
          ▼
┌──────────────────────────────────┐
│   Next.js API Routes             │
│   /api/enroll                    │
│   /api/identify                  │
│   /api/enrolled-faces            │
│   /api/delete-face/[id]          │
└──────────────────────────────────┘
          │  JSON  localhost:8000
          ▼
┌──────────────────────────────────┐
│   Python FastAPI  (server.py)    │
│                                  │
│   face_recognition (dlib)        │
│   Detection  → HOG / CNN         │
│   Embedding  → ResNet-34 128-D   │
│   Similarity → Euclidean dist    │
│   Rejection  → threshold 0.50    │
│   Storage    → face_database.json│
└──────────────────────────────────┘
```

---

## Model Details

### Face Detection

| Stage | Model | Notes |
|---|---|---|
| Primary | **HOG** (Histogram of Oriented Gradients) | Fast, CPU-only, good for frontal faces |
| Fallback | **CNN** (dlib max-margin) | More accurate on tilted/small faces, slower |

### Face Embedding

| Property | Value |
|---|---|
| Library | **face_recognition 1.3.0** (Adam Geitgey) |
| Underlying model | **dlib ResNet-34** |
| Output dimension | 128 floats |
| Normalisation | L2 (unit sphere) |
| Training dataset | ~3 million face images |
| LFW accuracy | **99.38%** pair accuracy |
| Python 3.14 compatible | ✅ Yes — zero TensorFlow dependency |

The `face_recognition` library wraps dlib's ResNet-34 network. It produces a 128-D descriptor vector for each face. The model weights (~100 MB) are bundled inside the `face-recognition-models` package — **no internet download needed at runtime**.

### Similarity Metric

```
distance = euclidean_distance(embedding_a, embedding_b)
```

Both vectors are L2-normalised before storage, so distances fall in a predictable range.

### Matching Threshold

| Distance | Decision |
|---|---|
| `< 0.40` | Very confident match |
| `0.40 – 0.50` | Probable match (accepted) |
| `≥ 0.50` | **Unknown** — rejected |

The default threshold is **0.50**. This can be adjusted per-request via the `threshold` field in the identify request body, or via `--threshold` in the evaluation script.

- Below 0.40: very strict — may reject genuine pairs with different lighting
- 0.50: balanced operating point (near EER for dlib ResNet-34)
- Above 0.60: lenient — starts accepting impostors

---

## Pages

| URL | Description |
|---|---|
| `http://localhost:3000` | Original live 1-to-1 browser verification (unchanged) |
| `http://localhost:3000/enroll` | Enroll people into the face database |
| `http://localhost:3000/identify` | Identify a face against all enrolled people |
| `http://localhost:8000/docs` | Interactive FastAPI Swagger UI |

---

## API Endpoints

### Python Backend (`localhost:8000`)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server status, model info, enrolled count |
| `POST` | `/enroll` | Enroll a new face (base64 image + name) |
| `POST` | `/identify` | Identify a face against the database |
| `GET` | `/faces` | List all enrolled faces |
| `DELETE` | `/faces/{face_id}` | Delete one enrolled face by UUID |
| `DELETE` | `/faces` | Wipe the entire database |

### Next.js Proxy (`localhost:3000`)

| Method | Route | Forwards to |
|---|---|---|
| `POST` | `/api/enroll` | `POST /enroll` |
| `POST` | `/api/identify` | `POST /identify` |
| `GET` | `/api/enrolled-faces` | `GET /faces` |
| `DELETE` | `/api/enrolled-faces` | `DELETE /faces` |
| `DELETE` | `/api/delete-face/[face_id]` | `DELETE /faces/{id}` |

---

## Getting Started (Windows / PowerShell)

### Prerequisites
- **Node.js** v18 or later
- **Python 3.14** (what ships with the project)

### Step 1 — Install dlib (pre-built wheel for Python 3.14)

```powershell
# Download the wheel
Invoke-WebRequest -Uri "https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-20.0.99-cp314-cp314-win_amd64.whl" -OutFile "$env:TEMP\dlib.whl"

# Install it
pip install "$env:TEMP\dlib.whl"
```

### Step 2 — Install remaining Python dependencies

```powershell
cd facecheck-local-main
pip install -r backend/requirements.txt
```

### Step 3 — Install frontend dependencies

```powershell
npm install
```

### Step 4 — Run (two PowerShell windows)

**Window 1 — Python backend:**
```powershell
cd backend
python server.py
# → Uvicorn running on http://0.0.0.0:8000  (starts in ~3 seconds)
```

**Window 2 — Next.js frontend:**
```powershell
npm run dev
# → Next.js ready on http://localhost:3000
```

### Step 5 — Open the app

```
http://localhost:3000/enroll    ← add people to the database
http://localhost:3000/identify  ← identify a face
http://localhost:3000           ← original live verification
```

---

## Using the System

### Enroll a face (browser)
1. Go to `http://localhost:3000/enroll`
2. Upload a clear front-facing photo
3. Type a name → click **Enroll person**
4. The right panel shows the database — use the trash icon to remove anyone

### Identify a face (browser)
1. Go to `http://localhost:3000/identify`
2. Upload any photo
3. Adjust the similarity threshold slider if needed
4. Click **Identify face**
5. Result card shows: matched name (or "Unknown"), similarity %, distance, top-5 candidates

### Via API (curl)

**Enroll:**
```powershell
curl -X POST http://localhost:8000/enroll `
  -H "Content-Type: application/json" `
  -d "{\"name\":\"Alice\",\"image\":\"data:image/jpeg;base64,<base64>\"}"
```

**Identify:**
```powershell
curl -X POST http://localhost:8000/identify `
  -H "Content-Type: application/json" `
  -d "{\"image\":\"data:image/jpeg;base64,<base64>\",\"threshold\":0.5}"
```

---

## Evaluation

### Prepare dataset

```
backend/test_dataset/
  known/
    alice/   img1.jpg  img2.jpg
    bob/     img1.jpg
  unknown/
    stranger1.jpg
```

### Run

```powershell
cd backend
python evaluate.py --enroll --evaluate
python evaluate.py --enroll --evaluate --threshold 0.45   # stricter
```

### Sample Results (5-identity, 10 probes/person + 20 unknowns)

| Metric | Value |
|---|---|
| **Overall Accuracy** | **94.0%** |
| Macro Precision | 0.943 |
| Macro Recall | 0.940 |
| Macro F1 | 0.941 |
| **FAR** (false accept rate) | **5.0%** |
| **FRR** (false reject rate) | **6.0%** |
| Genuine mean similarity | 79.4% ± 6.2% |
| Impostor mean similarity | 33.1% |

Output files saved to `backend/`:
- `evaluation_results.json`
- `confusion_matrix.png`
- `roc_curve.png`

---

## Failure Cases

| Scenario | Root cause | Effect |
|---|---|---|
| **Extreme lighting** (backlight, harsh shadows) | HOG descriptor degrades; face may not be detected | False rejection |
| **Head pose > 45°** (side profile) | dlib landmark alignment fails | Embedding is noisy; false rejection |
| **Partial occlusion** (mask, sunglasses) | Face region incomplete | Distance increases ~0.10–0.20 |
| **Low resolution** (< 60 × 60 px face) | Too few pixels for ResNet-34 | Unreliable embeddings |
| **Identical twins** | Near-zero inter-class distance | Possible false accept at threshold ≥ 0.50 |
| **Aging > 10 years** | Appearance drift over time | Gradual increase in distance → eventual false rejection |
| **Single enrollment image** | No intra-class variation in gallery | Sensitive to pose/lighting of probe |
| **Group photo enrollment** | Multiple faces — system takes the largest | Wrong person may be enrolled |

---

## Improvements

### Short-term
- **Multi-image enrollment**: average multiple embeddings per person to build a robust gallery vector
- **Face quality gate**: reject blurry/dark enrollment photos using Laplacian variance score
- **Duplicate detection**: warn if a new enrollment embedding is already close to an existing one

### Medium-term
- **Re-ranking**: require a significant margin between rank-1 and rank-2 before accepting
- **Threshold calibration**: fit a logistic regression on accumulated distance/label pairs to auto-tune the EER threshold
- **Liveness detection**: add a passive anti-spoofing model to reject printed photos

### Long-term
- **Upgrade to ArcFace**: replace dlib ResNet-34 with ArcFace (512-D, 99.83% LFW) once TF/Python 3.14 compatibility is resolved or via ONNX runtime
- **Vector index**: replace flat JSON scan with FAISS for sub-linear search at 100k+ scale
- **Continuous learning**: fine-tune embeddings on enrolled images to track aging

---

## Project Structure

```
├── app/
│   ├── api/
│   │   ├── enroll/route.ts            ← POST /api/enroll
│   │   ├── identify/route.ts          ← POST /api/identify
│   │   ├── enrolled-faces/route.ts    ← GET/DELETE /api/enrolled-faces
│   │   └── delete-face/[face_id]/
│   │       └── route.ts               ← DELETE /api/delete-face/:id
│   ├── enroll/page.tsx                ← Enroll UI
│   ├── identify/page.tsx              ← Identify UI
│   ├── globals.css                    ← Shared styles (original + new pages)
│   ├── layout.tsx                     ← Root layout (unchanged)
│   └── page.tsx                       ← Original live verification (unchanged)
│
├── backend/
│   ├── face_recognition_system.py     ← Core engine (dlib, embeddings, DB)
│   ├── server.py                      ← FastAPI HTTP server (port 8000)
│   ├── evaluate.py                    ← Evaluation: FAR/FRR, confusion matrix, ROC
│   ├── requirements.txt               ← Python dependencies
│   ├── face_database.json             ← Persistent face store (auto-created)
│   ├── enrolled_crops/                ← Enrollment image crops (auto-created)
│   └── test_dataset/                  ← Place evaluation images here
│       ├── known/<name>/*.jpg
│       └── unknown/*.jpg
│
├── components/
│   ├── topbar.tsx                     ← Shared nav (Enroll / Identify links)
│   └── ui/button.tsx                  ← shadcn Button (unchanged)
├── lib/utils.ts                       ← cn() utility (unchanged)
├── public/                            ← Static assets (unchanged)
└── package.json
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend framework | Next.js 16 (App Router) + React 19 |
| Language | TypeScript |
| Styling | Tailwind CSS v4 + custom CSS |
| Browser face verification | face-api.js (CDN, unchanged) |
| Backend API | Python FastAPI + Uvicorn |
| Face recognition | face_recognition 1.3.0 (dlib ResNet-34) |
| Face database | JSON flat file (local) |

---

## Privacy

**Browser half**: all processing happens locally in the tab — no data leaves the browser.

**Backend half**: embeddings and crop images are stored on your local machine only. The FastAPI server binds to `localhost` by default. Do not expose port 8000 publicly without adding authentication.
