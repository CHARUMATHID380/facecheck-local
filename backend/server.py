"""
Face Recognition API Server
============================
FastAPI application that exposes the face recognition system over HTTP.
The Next.js frontend proxies calls to this server via /api/* routes.

Endpoints
---------
POST   /enroll              Enroll a new face
POST   /identify            Identify a face against the database
GET    /faces               List all enrolled faces
DELETE /faces/{face_id}     Delete a specific enrolled face
DELETE /faces               Clear the entire database
GET    /health              Health check

Run with:
    python server.py
    # or
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from face_recognition_system import (
    THRESHOLD as COSINE_THRESHOLD,
    clear_database,
    delete_face,
    enroll_face,
    identify_face,
    list_enrolled_faces,
    DETECTION_MODEL,
)
MODEL_NAME      = "dlib ResNet-34 (face_recognition)"
DETECTOR_BACKEND = DETECTION_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Face Recognition API",
    description="Enrollment and identification of faces using ArcFace embeddings.",
    version="1.0.0",
)

# Allow the Next.js dev server (port 3000) and any localhost origin to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class EnrollRequest(BaseModel):
    image: str = Field(
        ...,
        description="Base64-encoded image string (data-URI or raw base64).",
    )
    name: str = Field(..., min_length=1, max_length=100, description="Person's name.")
    replace: bool = Field(
        False,
        description="If True, overwrite an existing entry with the same name.",
    )


class IdentifyRequest(BaseModel):
    image: str = Field(
        ...,
        description="Base64-encoded image string (data-URI or raw base64).",
    )
    top_k: int = Field(3, ge=1, le=20, description="Number of candidates to return.")
    threshold: float = Field(
        COSINE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Cosine-distance rejection threshold (0–1).",
    )


class EnrollResponse(BaseModel):
    face_id: str
    name: str
    enrolled_at: str
    message: str


class IdentifyResponse(BaseModel):
    identified: bool
    name: str
    face_id: Optional[str]
    similarity: float
    distance: float
    candidates: list[dict[str, Any]]
    message: str


class FaceEntry(BaseModel):
    face_id: str
    name: str
    enrolled_at: str


class DeleteResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    model: str
    detector: str
    threshold: float
    enrolled_count: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return server status and configuration summary."""
    faces = list_enrolled_faces()
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        detector=DETECTOR_BACKEND,
        threshold=COSINE_THRESHOLD,
        enrolled_count=len(faces),
    )


@app.post(
    "/enroll",
    response_model=EnrollResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Enrollment"],
)
def enroll(req: EnrollRequest) -> EnrollResponse:
    """
    Enroll a new face into the database.
    Accepts a base64-encoded image and a name.
    """
    try:
        result = enroll_face(req.image, req.name, replace=req.replace)
        return EnrollResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        log.exception("Unexpected error during enrollment.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enrollment failed: {e}",
        )


@app.post("/identify", response_model=IdentifyResponse, tags=["Identification"])
def identify(req: IdentifyRequest) -> IdentifyResponse:
    """
    Identify a face by matching it against the enrolled database.
    Returns 'unknown' if the best match distance exceeds the threshold.
    """
    try:
        result = identify_face(req.image, top_k=req.top_k, threshold=req.threshold)
        return IdentifyResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        log.exception("Unexpected error during identification.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Identification failed: {e}",
        )


@app.get("/faces", response_model=list[FaceEntry], tags=["Enrollment"])
def get_faces() -> list[FaceEntry]:
    """Return a list of all enrolled faces (no raw embedding data)."""
    faces = list_enrolled_faces()
    return [FaceEntry(**f) for f in faces]


@app.delete("/faces/{face_id}", response_model=DeleteResponse, tags=["Enrollment"])
def remove_face(face_id: str) -> DeleteResponse:
    """Delete a specific enrolled face by its UUID."""
    try:
        result = delete_face(face_id)
        return DeleteResponse(**result)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        log.exception("Unexpected error during deletion.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {e}",
        )


@app.delete("/faces", response_model=DeleteResponse, tags=["Enrollment"])
def wipe_database() -> DeleteResponse:
    """Delete ALL enrolled faces. Use with caution."""
    result = clear_database()
    return DeleteResponse(**result)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # reload=True causes DeepFace to import at startup (slow)
        log_level="info",
    )
