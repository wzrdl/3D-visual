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
import re
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import httpx

from app.backend_client import BackendAPIClient

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
        self.name_order: List[str] = []
        self.vector_database = None

        # Semantic search using sentence-transformers
        self._stopwords: List[str] = self._load_stopwords()
        # Semantic search members
        self._embedding_ids: List[str] = []
        self._model_lookup: Dict[str, Dict] = {}
        self._encoder = None
        self._semantic_embeddings = None  # ndarray
        self._semantic_ids: List[str] = []
        self._semantic_model_name = os.getenv("SEMANTIC_MODEL_NAME", "paraphrase-MiniLM-L3-v2")
        self._failed_downloads: set[str] = set()
        self._failed_logged: set[str] = set()

        if load_without_test:
            # Attempt to prime cache from local backup for offline scenarios
            meta_json_path = "app/assets/metadata.json.backup"
            if os.path.exists(meta_json_path):
                self.name_order, self.vector_database = self.concatenate_name_tags(meta_json_path)
            # Build vector index from backend (best-effort)
            self._init_vector_index()

    # metadata 

    def _load_stopwords(self) -> List[str]:
        """Load optional stopwords list from assets/stopwords.txt"""
        stopwords_path = Path(__file__).parent.parent / "assets" / "stopwords.txt"
        if not stopwords_path.exists():
            return []
        try:
            with open(stopwords_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception:
            return []

    def _infer_placement_type(self, display_name: str, tags: List[str]) -> str:
        """Heuristically infer placement type from name/tags."""
        text = f"{display_name} {' '.join(tags)}".lower()
        mapping = {
            "floating": ["cloud", "bird", "plane", "helicopter", "balloon", "star"],
            "character": ["person", "human", "man", "woman", "child", "soldier", "knight", "ninja", "zombie", "elf", "wizard", "goblin", "cowboy", "doctor", "pirate", "viking", "chef"],
            "prop": ["cup", "book", "pen", "plate", "vase", "clock", "phone", "pillow", "blanket", "lamp"],
            "ground": ["tree", "rock", "stone", "car", "chair", "table", "desk", "building", "house", "fence", "wall", "bench"],
        }
        for ptype, keywords in mapping.items():
            if any(k in text for k in keywords):
                return ptype
        return "ground"

    def _prepare_model_record(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize model dict and inject placement_type tag."""
        tags = model.get("tags") or []
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = [tags]
        if not isinstance(tags, list):
            tags = []
        display_name = str(model.get("display_name") or model.get("name") or "")
        placement_type = self._infer_placement_type(display_name, tags)
        if placement_type not in tags:
            tags.append(placement_type)
        return {
            **model,
            "display_name": display_name,
            "name": display_name,
            "tags": tags,
            "placement_type": placement_type,
        }

    def get_all_models(self) -> List[Dict]:
        """Get all models from backend and normalize."""
        self._all_models = [self._prepare_model_record(m) for m in self.api.list_models()]
        # Reset semantic caches so downstream (viewer/scene) can pick up new models
        self._semantic_embeddings = None
        self._encoder = None
        self._semantic_ids = []
        self._embedding_ids = []
        self._model_lookup = {m["id"]: m for m in self._all_models}
        # Best-effort rebuild (fail silently if encoder unavailable)
        try:
            self._init_vector_index()
        except Exception:
            pass
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

    # -------- Lightweight semantic search (semantic encoder) --------
    def _build_vector_corpus(self) -> Tuple[List[str], List[str]]:
        """Build corpus strings and id list for semantic embedding."""
        if not self._all_models:
            self.get_all_models()
        corpus: List[str] = []
        ids: List[str] = []
        self._model_lookup = {m["id"]: m for m in self._all_models}

        for model in self._all_models:
            name_raw = model.get("display_name", "") or model.get("name", "")
            name_norm = self._normalize_text(name_raw)
            if name_norm:
                corpus.append(name_norm)
                ids.append(model["id"])
        return corpus, ids

    def _init_vector_index(self):
        """Create in-memory semantic vector index for search."""
        corpus, ids = self._build_vector_corpus()
        if not corpus:
            return
        self._embedding_ids = ids
        self._init_semantic_embeddings()

    def _init_semantic_embeddings(self):
        """Build lightweight sentence embeddings if model is available."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self._semantic_model_name)
            texts = [self._normalize_text(self._model_lookup[mid]["display_name"]) for mid in self._embedding_ids]
            self._semantic_embeddings = np.array(model.encode(texts, normalize_embeddings=True))
            self._semantic_ids = list(self._embedding_ids)
            self._encoder = model
        except Exception:
            self._encoder = None
            self._semantic_embeddings = None

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Run lightweight semantic search using sentence-transformers cosine similarity.

        Returns a list of dicts: {"model": model_dict, "score": float}
        """
        if not query or not query.strip():
            return []
        if self._semantic_embeddings is None or self._encoder is None:
            self._init_vector_index()
        if self._semantic_embeddings is None or self._encoder is None:
            return []

        normalized_query = self._normalize_text(query)
        if not normalized_query:
            return []
        results: List[Dict] = []
        try:
            variants = self._synonym_expand(normalized_query)
            q_emb = self._encoder.encode(variants, normalize_embeddings=True)
            scores = None
            for emb in q_emb:
                s = cosine_similarity([emb], self._semantic_embeddings).flatten()
                scores = s if scores is None else np.maximum(scores, s)
            if scores is None:
                return []
            ranked_indices = np.argsort(scores)[::-1]
            for idx in ranked_indices[:top_k]:
                model_id = self._semantic_ids[idx]
                model = self._model_lookup.get(model_id)
                if not model:
                    continue
                results.append({"model": model, "score": float(scores[idx])})
        except Exception:
            return []

        return results

    def _normalize_text(self, text: str) -> str:
        """
        Lowercase and replace any non-alphanumeric (including underscores) with space,
        so tokens like 'car' survive from names such as 'model_064_a_ferrari_sf90_car'.
        """
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
        return " ".join(text.split())

    def _synonym_expand(self, query: str) -> List[str]:
        """Expand query with lightweight synonyms for better recall."""
        base = query.strip().lower()
        synonyms_map = {
            "zoo": ["animal", "wildlife", "lion", "tiger", "bear", "snake", "frog", "bird"],
            "animal": ["wildlife", "creature", "beast"],
            "car": ["vehicle", "auto", "automobile", "racing car"],
            "house": ["home", "building"],
        }
        extras = synonyms_map.get(base, [])
        return [base] + extras if extras else [base]

    def get_model_embeddings(self, model_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Return normalized semantic embeddings for given model ids (best-effort).

        Used by layout stage to compute pairwise similarity without tags.
        """
        try:
            self._init_vector_index()
        except Exception:
            return {}

        if getattr(self, "_semantic_embeddings", None) is None or getattr(self, "_semantic_ids", None) is None:
            return {}

        id_to_idx = {mid: idx for idx, mid in enumerate(self._semantic_ids)}
        vectors: Dict[str, np.ndarray] = {}
        for mid in model_ids:
            idx = id_to_idx.get(mid)
            if idx is None:
                continue
            vec = self._semantic_embeddings[idx]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vectors[mid] = vec / norm
        return vectors

    # model files   

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """
        Ensure the model file exists locally (cache).

        - Fetch model metadata from the backend (to obtain filename)
        - If `assets/models/filename` already exists, return it
        - Otherwise download the file bytes from FastAPI, write them locally, then return the path
        """
        if model_id in self._failed_downloads:
            return None

        try:
            meta = self.api.get_model(model_id)
        except Exception as e:
            if model_id not in self._failed_logged:
                print(f"[ClientDataManager] Skip metadata for {model_id}: {e}")
                self._failed_logged.add(model_id)
            self._failed_downloads.add(model_id)
            return None

        filename = meta.get("filename")
        if not filename:
            return None

        local_path = self.models_dir / filename
        if local_path.exists():
            return local_path

        try:
            content = self.api.download_model_content(model_id)
        except Exception as e:
            if model_id not in self._failed_logged:
                print(f"[ClientDataManager] Skip download for {model_id}: {e}")
                self._failed_logged.add(model_id)
            self._failed_downloads.add(model_id)
            return None

        try:
            with open(local_path, "wb") as f:
                f.write(content)
        except OSError as e:
            print(f"Error writing model cache file {local_path}: {e}")
            self._failed_downloads.add(model_id)
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

    # ============= Legacy Test Support =============
    # The following method is kept for backward compatibility with existing tests.
    # New code should use semantic_search() instead.
    
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

        # Build semantic embeddings for the provided corpus (legacy test support)
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self._semantic_model_name)
            self.vector_database = model.encode(name_tags, normalize_embeddings=True)
            self._encoder = model
        except Exception:
            self.vector_database = None
        return self.name_order, self.vector_database
