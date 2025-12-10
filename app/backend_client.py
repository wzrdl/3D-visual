"""
This backend client is used to talk to the FastAPI backend
All the endpoints are already created in the backend, we just need to call them here
"""

from __future__ import annotations

import json
import os
from typing import Optional, Dict, Any, List

import httpx


class BackendAPIClient:
    """Simple blocking client for the 3D backend."""

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        base = api_url or os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")
        self.base_url = base.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(timeout=60.0)


    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_models(self) -> List[Dict[str, Any]]:
        resp = self._client.get(f"{self.base_url}/models", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def get_model(self, model_id: str) -> Dict[str, Any]:
        resp = self._client.get(
            f"{self.base_url}/models/{model_id}", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    def download_model_content(self, model_id: str) -> bytes:
        resp = self._client.get(
            f"{self.base_url}/models/{model_id}/content",
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.content

    def upload_model(
        self,
        name: str,
        tags: List[str],
        file_bytes: bytes,
        filename: str,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        This function is used to upload the model to the backend.
        """
        data = {
            "name": name,
            "tags": json.dumps(tags),
        }
        if model_id is not None:
            data["model_id"] = model_id

        files = {
            "file": (filename, file_bytes, "application/octet-stream"),
        }

        resp = self._client.post(
            f"{self.base_url}/models",
            headers=self._headers(),
            data=data,
            files=files,
        )
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        if self._client:
            self._client.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


