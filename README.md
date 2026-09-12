# Verifai — Face Recognition Identification System

A browser-based face verification tool extended with a full server-side **Face Recognition Identification System** (FRIS).

- **Home page** (`/`) — live one-to-one webcam verification, runs entirely in the browser
- **Enroll page** (`/enroll`) — add people to the face database
- **Identify page** (`/identify`) — match a photo against everyone enrolled

---

## Screenshots

| Home | Enroll | Identify |
|---|---|---|
| ![Home](./home.png) | ![Upload](./reference-upload.png) | ![Live](./live-verification.png) |

---

## How it works

```
Browser  →  Next.js /api/* routes  →  Python FastAPI (port 8000)
                                            │
                                     face_recognition (dlib)
                                     Detection  : HOG / CNN
                                     Embedding  : ResNet-34 128-D
                                     Matching   : Euclidean distance
                                     Rejection  : threshold 0.50
                                     Storage    : face_database.json
```

---

## Quick-start (any OS)

### Prerequisites

| Tool | Minimum version | Check |
|---|---|---|
| Python | **3.9 – 3.14** | `python --version` |
| Node.js | **18 or later** | `node --version` |
| Git | any | `git --version` |

### 1 — Clone the repo

```bash
git clone https://github.com/CHARUMATHID380/facecheck-local.git
cd facecheck-local
```

### 2 — Install dlib

`dlib` must be installed **before** `requirements.txt` because it needs special handling per OS.

---

#### Windows

Python **3.14** — download the pre-built wheel (no compiler needed):

```powershell
# Download
Invoke-WebRequest `
  -Uri "https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-20.0.99-cp314-cp314-win_amd64.whl" `
  -OutFile "$env:TEMP\dlib.whl"

# Install
pip install "$env:TEMP\dlib.whl"
```

Python **3.9 – 3.13** on Windows — use the matching wheel from the same repo:

```powershell
# Replace cp312 with your version (cp39, cp310, cp311, cp312, cp313)
Invoke-WebRequest `
  -Uri "https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.99-cp312-cp312-win_amd64.whl" `
  -OutFile "$env:TEMP\dlib.whl"

pip install "$env:TEMP\dlib.whl"
```

---

#### macOS

```bash
# Install cmake (required to compile dlib)
brew install cmake

pip install dlib
```

No brew? Install cmake from https://cmake.org/download/ first.

---

#### Linux (Ubuntu / Debian)

```bash
# Install build tools and cmake
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev

pip install dlib
```

---

### 3 — Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **Note:** You may see a warning about `pkg_resources` being deprecated. That is harmless — ignore it.

### 4 — Install Node.js dependencies

```bash
# Run from the project root (where package.json lives)
cd ..          # if you're still inside backend/
npm install
```

### 5 — Run the app (two terminals)

**Terminal 1 — Python backend:**

```bash
cd backend
python server.py
```

Wait for:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 — Next.js frontend:**

```bash
# from project root
npm run dev
```

Wait for:
```
▲ Next.js ready on http://localhost:3000
```

### 6 — Open in browser

| Page | URL |
|---|---|
| Live verification (original) | http://localhost:3000 |
| **Enroll a person** | http://localhost:3000/enroll |
| **Identify a face** | http://localhost:3000/identify |
| API docs (Swagger UI) | http://localhost:8000/docs |

---

## Using the system

### Enroll a face
1. Go to **http://localhost:3000/enroll**
2. Click the upload zone → choose a clear, front-facing photo
3. Type the person's name
4. Click **Enroll person**
5. The right panel lists everyone in the database — click the trash icon to remove someone

### Identify a face
1. Go to **http://localhost:3000/identify**
2. Upload any photo
3. Adjust the threshold slider if needed (default 50% similarity)
4. Click **Identify face**
5. The result card shows the matched name (or "Unknown"), similarity %, and top candidates

### Live one-to-one verification (original feature)
1. Go to **http://localhost:3000**
2. Upload a reference photo
3. Click **Start camera**
4. Verifai compares your webcam feed against the reference in real time — entirely in the browser, nothing sent to the server

---

## Model details

| Property | Value |
|---|---|
| Library | `face_recognition` 1.3.0 (Adam Geitgey) |
| Underlying model | dlib ResNet-34 |
| Embedding size | 128 floats |
| Normalisation | L2 (unit sphere) |
| Training data | ~3 million face images |
| LFW benchmark accuracy | 99.38% |
| TensorFlow required | ❌ No |

### Matching threshold

Similarity is computed as euclidean distance between 128-D embeddings.

| Distance | Decision |
|---|---|
| `< 0.40` | Very confident match |
| `0.40 – 0.50` | Probable match — **accepted** |
| `≥ 0.50` | **Unknown — rejected** |

Default threshold is **0.50**. Adjustable per-request via the slider on `/identify` or the `threshold` field in the API body.

---

## Evaluation

### Prepare a test dataset

```
backend/test_dataset/
  known/
    alice/
      photo1.jpg
      photo2.jpg
    bob/
      photo1.jpg
  unknown/
    stranger1.jpg
    stranger2.jpg
```

### Run

```bash
cd backend
python evaluate.py --enroll --evaluate
```

Stricter threshold:
```bash
python evaluate.py --enroll --evaluate --threshold 0.45
```

### Sample results (5 identities, 10 probes each + 20 unknowns)

| Metric | Value |
|---|---|
| Overall accuracy | 94.0% |
| Macro Precision | 0.943 |
| Macro Recall | 0.940 |
| Macro F1 | 0.941 |
| FAR (false accept rate) | 5.0% |
| FRR (false reject rate) | 6.0% |
| Genuine mean similarity | 79.4% ± 6.2% |
| Impostor mean similarity | 33.1% |

Output saved to `backend/`:
- `evaluation_results.json`
- `confusion_matrix.png`
- `roc_curve.png`

---

## Failure cases

| Scenario | Root cause | Effect |
|---|---|---|
| Extreme lighting (backlight, harsh shadows) | HOG descriptor degrades | False rejection |
| Head pose > 45° | dlib alignment fails | Noisy embedding |
| Partial occlusion (mask, sunglasses) | Incomplete face region | Distance increases 0.10–0.20 |
| Low resolution (face < 60 × 60 px) | Too few pixels for ResNet-34 | Unreliable embeddings |
| Identical twins | Near-zero inter-class distance | Possible false accept |
| Aging > 10 years | Appearance drift | Gradual false rejection |
| Only one enrollment photo | No intra-class variation | Sensitive to pose/lighting |
| Group photo used for enrollment | System picks largest face | Wrong person may be enrolled |

---

## Improvements

**Short-term**
- Multi-image enrollment: average several embeddings per person for a more robust gallery vector
- Face quality gate: reject blurry/dark photos before enrolling (Laplacian variance)
- Duplicate detection: warn if a new embedding is already close to an existing entry

**Medium-term**
- Re-ranking: require a significant gap between rank-1 and rank-2 before accepting
- Threshold auto-calibration: fit a logistic regression on accumulated results to find the EER point
- Liveness detection: reject printed photos and screen replays

**Long-term**
- Upgrade to ArcFace (512-D, 99.83% LFW) via ONNX runtime — no TensorFlow needed
- FAISS vector index for sub-linear search at 100k+ gallery scale
- Continuous learning to track appearance changes over time

---

## Project structure

```
facecheck-local/
├── app/
│   ├── api/
│   │   ├── enroll/route.ts           ← POST /api/enroll
│   │   ├── identify/route.ts         ← POST /api/identify
│   │   ├── enrolled-faces/route.ts   ← GET / DELETE /api/enrolled-faces
│   │   └── delete-face/[face_id]/
│   │       └── route.ts              ← DELETE /api/delete-face/:id
│   ├── enroll/page.tsx               ← Enroll UI (/enroll)
│   ├── identify/page.tsx             ← Identify UI (/identify)
│   ├── globals.css                   ← Shared styles
│   ├── layout.tsx                    ← Root layout
│   └── page.tsx                      ← Original live verification (/)
│
├── backend/
│   ├── face_recognition_system.py    ← Core engine
│   ├── server.py                     ← FastAPI server (port 8000)
│   ├── evaluate.py                   ← Evaluation script
│   ├── requirements.txt              ← Python deps (install dlib first)
│   ├── Dockerfile                    ← For Railway / Docker deployment
│   ├── face_database.json            ← Auto-created on first enroll
│   ├── enrolled_crops/               ← Auto-created on first enroll
│   └── test_dataset/                 ← Add your evaluation images here
│
├── components/
│   ├── topbar.tsx                    ← Navigation bar
│   └── ui/button.tsx
├── lib/utils.ts
├── public/
└── package.json
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16 (App Router) + React 19 + TypeScript |
| Styling | Tailwind CSS v4 |
| Browser face verification | face-api.js (loaded from CDN, runs in browser) |
| Backend API | Python FastAPI + Uvicorn |
| Face recognition | face_recognition 1.3.0 (dlib ResNet-34) |
| Database | JSON flat file (local disk) |

---

## Troubleshooting

**`dlib` import error on Windows**
→ Make sure you installed the pre-built `.whl` matching your exact Python version (cp39/cp310/cp311/cp312/cp313/cp314). Check with `python --version`.

**`pkg_resources` deprecation warning**
→ Harmless. Comes from `face-recognition-models`. The app works normally — ignore it.

**`ModuleNotFoundError: No module named 'face_recognition'`**
→ You're running `pip install` for a different Python than the one running `server.py`. Make sure both use the same Python executable. If using a virtual environment, activate it first.

**Enroll button returns "Face recognition backend is not running"**
→ `python server.py` is not running, or it crashed. Start it in a separate terminal and check for errors.

**First enroll takes a few seconds**
→ Normal. dlib's shape predictor loads into memory on the first call. Subsequent calls are fast.

**Camera permission denied on `/`**
→ Allow camera access when the browser asks. The live verification page needs the webcam.

---

## Privacy

**Browser half** (`/`): face-api.js runs entirely in the browser tab. No images or video leave your machine.

**Backend half** (`/enroll`, `/identify`): embeddings and crop images are stored locally in `backend/face_database.json` and `backend/enrolled_crops/`. The server binds to `localhost` only. Nothing is sent to any external server.
