"""
FastAPI backend for managing 3D model metadata and files. 
I have already created the endpoint for the backend in GCP Cloud Run
"""

from __future__ import annotations

import json
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse

from app.data_manager import DataManager


app = FastAPI(title="3D Model Library Backend")


@app.on_event("startup")
def startup_event() -> None:
    # Create a single DataManager instance and keep it for the lifetime of the service
    app.state.data_manager = DataManager()


@app.on_event("shutdown")
def shutdown_event() -> None:
    dm: DataManager = app.state.data_manager
    dm.close()


def _get_dm() -> DataManager:
    return app.state.data_manager


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/models", response_model=List[Dict])
def list_models() -> List[Dict]:
    """
    Return all models metadata from SQLite database.
    (Currently still uses SQLite inside the container; can be migrated to Cloud SQL later.)
    """
    dm = _get_dm()
    return dm.get_all_models()


@app.get("/models/{model_id}", response_model=Dict)
def get_model(model_id: str) -> Dict:
    """
    Get single model's metadata and ensure the file exists (locally or via GCS download).
    """
    dm = _get_dm()
    models = [m for m in dm.get_all_models() if m["id"] == model_id]
    if not models:
        raise HTTPException(status_code=404, detail="Model not found")

    model = models[0]
    # Ensure the file exists locally; it will be pulled from GCS when needed
    path = dm.get_model_path(model_id)
    if not path:
        raise HTTPException(status_code=404, detail="Model file not found (local + GCS)")

    model["local_path"] = str(path)
    return model


@app.get("/models/{model_id}/content")
def download_model_content(model_id: str):
    """
    Return the binary content of the specified model file for client-side caching.
    If GCS is configured, return a signed URL (redirect) to avoid Cloud Run size limits.
    """
    dm = _get_dm()
    # First, try GCS signed URL if configured
    if getattr(dm, "gcs_storage", None) is not None:
        try:
            # Look up filename from DB
            models = [m for m in dm.get_all_models() if m["id"] == model_id]
            if not models:
                raise HTTPException(status_code=404, detail="Model not found")
            filename = models[0]["filename"]
            signed_url = dm.gcs_storage.generate_signed_url(filename=filename, expiration_seconds=3600)
            return RedirectResponse(url=signed_url, status_code=307)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Model file not found in GCS")
        except Exception as e:
            # Fall back to local delivery if signed URL fails
            print(f"Signed URL generation failed for {model_id}: {e}")

    # Fallback: serve local file (for local dev or when GCS not configured)
    path = dm.get_model_path(model_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        str(path),
        media_type="application/octet-stream",
        filename=path.name,
    )


@app.post("/models")
async def upload_model(
    model_id: str = Form(...),
    name: str = Form(...),
    tags: str = Form("[]"),
    file: UploadFile = File(...),
):

    # Upload a new 3D model to the database and the local directory
    dm = _get_dm()

    try:
        content = await file.read()
        tags_list = json.loads(tags) if tags else []
        if not isinstance(tags_list, list):
            raise ValueError("tags must be a JSON array")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad input: {e}")

    ok = dm.save_model_to_gallery(
        model_id=model_id,
        filename=file.filename,
        name=name,
        tags=tags_list,
        model_data=content,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save model")

    return JSONResponse({"success": True, "id": model_id, "filename": file.filename})



