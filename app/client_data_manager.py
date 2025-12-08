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
from typing import List, Dict, Optional, Any

from app.backend_client import BackendAPIClient
from sentence_transformers import SentenceTransformer

class ClientDataManager:
    """Client-side data manager using FastAPI backend + local cache."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        assets_dir: Optional[str] = None,
        load_without_test = True
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

        # sentence transformer, loading mini models
        self.miniM_model = SentenceTransformer("all-MiniLM-L6-v2") # 'all-mpnet-base-v2' for more accuracy?
        self.name_order = [] # to be easier to use for scene_brain
        self.vector_database = None  # as it shouldn't be a standard vector []

        if load_without_test == True:
            # combine name and tags
            meta_json_path = "app/assets/metadata.json.backup"
            if os.path.exists(meta_json_path):
                self.name_order, self.vector_database = self.concatenate_name_tags(meta_json_path)
            else:
                print("Error: could not load meta.json")

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

    """ Infrastructure and Vector Database """
    def concatenate_name_tags(self, metajson_location: str = None):
        name_tags = []
        self.name_order = []

        # default
        if metajson_location is None:

            for m in self._all_models:
                model_name = str(m.get("display_name") or m.get("name") or "")
                tags = m.get("tags") or []
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except Exception:
                        tags = [tags]

                name_tags.append(model_name.lower() + " " + " ".join(tags))
                self.name_order.append(model_name.lower())

        else:

            # from the data_manager.py initial migration from json
            # Migrate from JSON

            with open(metajson_location, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            for entry in metadata:
                #model_id = entry.get('id')
                model_name = entry.get('display_name') or entry.get('name')
                #display_name = entry.get('name', filename)  # Use 'name' from JSON as display_name
                tags = entry.get('tags', [])

                # concatenate filename and tags
                name_tags.append(model_name.lower() + " " + " ".join(tags))
                self.name_order.append(model_name.lower())

        embeddings = self.miniM_model.encode_document(name_tags)
            # document recommended to use for encode for "your corpus"
        self.vector_database = embeddings
        #print(self.vector_database)

        return self.name_order, self.vector_database
