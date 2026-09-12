"""
Face Recognition Identification System
=======================================
Core engine using the `face_recognition` library (dlib / HOG + ResNet).

No TensorFlow. No PyTorch. Works on Python 3.14.

Pipeline
--------
  Detection  : dlib HOG-based detector (fast, CPU-only)
               fallback: dlib CNN detector (more accurate, slower)
  Landmarks  : dlib 68-point shape predictor
  Embedding  : dlib ResNet-34 face descriptor (128-D float64 vector)
  Similarity : Euclidean distance between L2-normalised embeddings
  Rejection  : distance > THRESHOLD → label as "unknown"

Model
-----
  face_recognition v1.3.0 uses dlib's ResNet-34 network trained on a
  3-million-image dataset.  On the Labeled Faces in the Wild (LFW)
  benchmark it achieves ~99.38% accuracy.

Threshold
---------
  THRESHOLD = 0.50  (euclidean distance on 128-D unit vectors)
  Values BELOW the threshold → match.
  Values AT OR ABOVE the threshold → "unknown".

  Rule of thumb:
    distance < 0.40  → very confident match
    0.40–0.50        → probable match
    > 0.50           → reject as unknown
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Suppress pkg_resources deprecation warning from face_recognition_models
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message="pkg_resources")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).parent
DB_PATH   = BASE_DIR / "face_database.json"
CROPS_DIR = BASE_DIR / "enrolled_crops"
CROPS_DIR.mkdir(exist_ok=True)

# Euclidean-distance rejection threshold (128-D descriptor space)
THRESHOLD = 0.50

# How many jitter augmentations when computing enrollment embedding
# Higher = more accurate but slower.  1 = fast, 10 = high accuracy.
NUM_JITTERS = 1

# Detection model: "hog" (fast, CPU) or "cnn" (accurate, slower)
DETECTION_MODEL = "hog"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import
# ---------------------------------------------------------------------------

def _fr():
    """Return the face_recognition module (imported lazily)."""
    import face_recognition  # type: ignore
    return face_recognition


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _load_db() -> dict[str, Any]:
    if DB_PATH.exists():
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not read DB (%s); starting fresh.", e)
    return {}


def _save_db(db: dict[str, Any]) -> None:
    tmp = DB_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    tmp.replace(DB_PATH)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _decode_image(source: str | bytes | np.ndarray) -> np.ndarray:
    """Return an RGB uint8 numpy array from base64 string, bytes, or array."""
    if isinstance(source, np.ndarray):
        arr = source
    elif isinstance(source, bytes):
        arr = np.array(Image.open(io.BytesIO(source)).convert("RGB"))
    elif isinstance(source, str):
        b64 = source.split(",", 1)[1] if source.startswith("data:") else source
        arr = np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
    else:
        raise ValueError(f"Unsupported image type: {type(source)}")

    # face_recognition expects uint8 RGB
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def _save_crop(image: np.ndarray, name: str, face_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
    path = CROPS_DIR / f"{safe}_{face_id[:8]}.jpg"
    Image.fromarray(image).save(str(path), format="JPEG", quality=92)
    return str(path)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_embedding(image: np.ndarray) -> np.ndarray:
    """
    Detect the face in `image` and return its 128-D L2-normalised descriptor.
    Raises ValueError if no face is detected.
    """
    fr = _fr()

    locations = fr.face_locations(image, model=DETECTION_MODEL)
    if not locations:
        # Retry with CNN detector if HOG found nothing
        locations = fr.face_locations(image, model="cnn")
    if not locations:
        raise ValueError(
            "No face detected in the image. "
            "Use a clear, front-facing photo with good lighting."
        )

    # Use the largest detected face (most prominent)
    locations_sorted = sorted(
        locations,
        key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]),
        reverse=True,
    )

    encodings = fr.face_encodings(
        image,
        known_face_locations=[locations_sorted[0]],
        num_jitters=NUM_JITTERS,
        model="large",
    )
    if not encodings:
        raise ValueError("Could not compute face embedding.")

    emb = np.array(encodings[0], dtype=np.float64)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm
    return emb


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors (lower = more similar)."""
    return float(np.linalg.norm(a - b))


def similarity_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Convert euclidean distance to a 0-100% similarity score.
    At distance=0 → 100%, at distance=THRESHOLD → 0%.
    Clamped to [0, 100].
    """
    dist = euclidean_distance(a, b)
    sim = max(0.0, min(100.0, (1.0 - dist / THRESHOLD) * 100.0))
    return float(sim)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enroll_face(
    image_source: str | bytes | np.ndarray,
    name: str,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    """
    Enroll a new face into the database.

    Parameters
    ----------
    image_source : base64 data-URI, raw bytes, or numpy RGB array
    name         : human-readable label
    replace      : overwrite existing entry with same name if True

    Returns
    -------
    dict: face_id, name, enrolled_at, message
    """
    if not name or not name.strip():
        raise ValueError("Name must not be empty.")
    name = name.strip()

    db = _load_db()
    existing = [fid for fid, m in db.items() if m["name"].lower() == name.lower()]

    if existing and not replace:
        raise ValueError(
            f"'{name}' is already enrolled (id={existing[0]}). "
            "Set replace=True to overwrite."
        )

    image = _decode_image(image_source)
    embedding = get_embedding(image)

    face_id   = str(uuid.uuid4())
    crop_path = _save_crop(image, name, face_id)

    for old in existing:
        del db[old]

    db[face_id] = {
        "face_id":     face_id,
        "name":        name,
        "embedding":   embedding.tolist(),
        "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "crop_path":   crop_path,
    }
    _save_db(db)
    log.info("Enrolled '%s' (id=%s).", name, face_id)

    return {
        "face_id":     face_id,
        "name":        name,
        "enrolled_at": db[face_id]["enrolled_at"],
        "message":     f"Successfully enrolled '{name}'.",
    }


def identify_face(
    image_source: str | bytes | np.ndarray,
    *,
    top_k: int = 3,
    threshold: float = THRESHOLD,
) -> dict[str, Any]:
    """
    Identify a face against the enrolled database.

    Returns
    -------
    dict: identified, name, face_id, similarity, distance, candidates, message
    """
    db = _load_db()
    if not db:
        return {
            "identified": False,
            "name":       "unknown",
            "face_id":    None,
            "similarity": 0.0,
            "distance":   999.0,
            "candidates": [],
            "message":    "Database is empty. Enroll some faces first.",
        }

    image = _decode_image(image_source)
    probe = get_embedding(image)

    scores: list[dict[str, Any]] = []
    for fid, meta in db.items():
        enrolled = np.array(meta["embedding"], dtype=np.float64)
        dist = euclidean_distance(probe, enrolled)
        sim  = similarity_score(probe, enrolled)
        scores.append({
            "face_id":    fid,
            "name":       meta["name"],
            "distance":   round(dist, 4),
            "similarity": round(sim, 2),
        })

    scores.sort(key=lambda x: x["distance"])
    best       = scores[0]
    candidates = scores[:top_k]

    if best["distance"] >= threshold:
        return {
            "identified": False,
            "name":       "unknown",
            "face_id":    None,
            "similarity": best["similarity"],
            "distance":   best["distance"],
            "candidates": candidates,
            "message": (
                f"No confident match (best distance {best['distance']:.3f} "
                f">= threshold {threshold:.2f}). Labelled unknown."
            ),
        }

    return {
        "identified": True,
        "name":       best["name"],
        "face_id":    best["face_id"],
        "similarity": best["similarity"],
        "distance":   best["distance"],
        "candidates": candidates,
        "message": (
            f"Identified as '{best['name']}' — "
            f"{best['similarity']:.1f}% similarity (distance {best['distance']:.3f})."
        ),
    }


def list_enrolled_faces() -> list[dict[str, Any]]:
    db = _load_db()
    return [
        {"face_id": fid, "name": m["name"], "enrolled_at": m.get("enrolled_at", "")}
        for fid, m in db.items()
    ]


def delete_face(face_id: str) -> dict[str, str]:
    db = _load_db()
    if face_id not in db:
        raise KeyError(f"No face with id '{face_id}'.")
    name = db[face_id]["name"]
    del db[face_id]
    _save_db(db)
    log.info("Deleted '%s' (id=%s).", name, face_id)
    return {"message": f"Deleted '{name}' (id={face_id})."}


def clear_database() -> dict[str, str]:
    _save_db({})
    log.warning("Database cleared.")
    return {"message": "All enrolled faces deleted."}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Face Recognition System")
    print(f"  Library   : face_recognition (dlib ResNet-34)")
    print(f"  Threshold : {THRESHOLD}  (euclidean distance)")
    print(f"  Detector  : {DETECTION_MODEL}")
    print(f"  Database  : {DB_PATH}")
    print()
    faces = list_enrolled_faces()
    print(f"  Enrolled  : {len(faces)} face(s)")
    for f in faces:
        print(f"    - {f['name']}  ({f['face_id'][:8]}…)  {f['enrolled_at']}")
