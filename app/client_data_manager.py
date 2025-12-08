"""
Client-side DataManager: acts as a thin cache in front of the FastAPI backend.

Responsibilities:
- Fetch model metadata and files from the FastAPI backend over HTTP
- Temporarily cache downloaded 3D model files (e.g. `.obj`, `.glb`) under the local
  `assets/models/` directory
- Provide a DataManager-like interface for the UI (get_all_models / search_models / get_model_path)
- Allow the application to clear the local cache on exit
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional

from app.backend_client import BackendAPIClient


class ClientDataManager:
    """Client-side data manager using FastAPI backend + local cache."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        assets_dir: Optional[str] = None,
    ):
        # API client (talks to FastAPI backend)
        self.api = BackendAPIClient(api_url=api_url)

        # Local cache directory: reuse assets/models so thumbnail logic keeps working
        if assets_dir is None:
            project_root = Path(__file__).parent.parent
            self.assets_dir = project_root / "assets"
        else:
            self.assets_dir = Path(assets_dir)

        self.models_dir = self.assets_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache of model metadata to avoid repeated HTTP calls
        self._all_models: List[Dict] = []

    # metadata 

    def get_all_models(self) -> List[Dict]:
        """Get all models from backend."""
        self._all_models = self.api.list_models()
        return self._all_models

    def search_models(self, query: str) -> List[Dict]:
        """
        Perform a simple client-side search (based on name / tags),
        so we do not need a dedicated search endpoint on the backend yet.
        """
        query = (query or "").strip()
        if not query:
            if not self._all_models:
                self.get_all_models()
            return self._all_models

        if not self._all_models:
            self.get_all_models()

        q = query.lower()
        results: List[Dict] = []
        for m in self._all_models:
            name = str(m.get("display_name") or m.get("name") or "")
            tags = m.get("tags") or []
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = [tags]
            text_hit = q in name.lower()
            tag_hit = any(q in str(t).lower() for t in tags)
            if text_hit or tag_hit:
                results.append(m)
        return results

    # model files 

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """
        Ensure the model file exists locally (cache).

        - Fetch model metadata from the backend (to obtain filename)
        - If `assets/models/filename` already exists, return it
        - Otherwise download the file bytes from FastAPI, write them locally, then return the path
        """
        meta = self.api.get_model(model_id)
        filename = meta.get("filename")
        if not filename:
            return None

        local_path = self.models_dir / filename
        if local_path.exists():
            return local_path

        content = self.api.download_model_content(model_id)
        try:
            with open(local_path, "wb") as f:
                f.write(content)
        except OSError as e:
            print(f"Error writing model cache file {local_path}: {e}")
            return None

        return local_path

    # cache management 

    def clear_cache(self) -> None:
        """Delete all cached model files from assets/models."""
        if not self.models_dir.exists():
            return
        for pattern in ("*.obj", "*.glb"):
            for path in self.models_dir.glob(pattern):
                try:
                    path.unlink()
                except OSError:
                    pass

    def close(self) -> None:
        """Cleanup resources."""
        try:
            self.api.close()
        except Exception:
            pass


