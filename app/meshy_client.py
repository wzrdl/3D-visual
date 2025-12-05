"""
Meshy Text-to-3D API client.

Implements a minimal workflow based on the official docs:
https://docs.meshy.ai/en/api/text-to-3d

Workflow we support here:
- Create a Text-to-3D preview task (`mode="preview"`) from a text prompt
- Poll the task status until it finishes
- Return the OBJ download URL when available
"""

from __future__ import annotations

import os
import time
from typing import Optional, Dict, Any

import httpx
import asyncio
from pathlib import Path


_DOTENV_LOADED = False


def _load_dotenv_from_project_root() -> None:
    """
    Minimal .env loader for local development.

    - Looks for a file named ".env" in the project root (parent of the app/ package).
    - Parses KEY=VALUE lines and populates os.environ if the key is not already set.

    This avoids adding an external dependency like python-dotenv while still
    allowing you to keep secrets (e.g. MESHY_API_KEY) in a .env file.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    try:
        project_root = Path(__file__).parent.parent
        env_path = project_root / ".env"
        if not env_path.exists():
            _DOTENV_LOADED = True
            return

        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        # Fail silently; we do not want dotenv loading to break the app
        print(f"[Meshy] Warning: failed to load .env: {e}")
    finally:
        _DOTENV_LOADED = True


class MeshyClient:
    """Async client for Meshy Text-to-3D API."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        ai_model: Optional[str] = None,
    ):
        """
        Initialize Meshy client.

        - api_url: base URL for Meshy API, defaults to https://api.meshy.ai
        - api_key: Meshy API key. If not provided, reads MESHY_API_KEY from env.
        - ai_model: Meshy model id to use (meshy-4, meshy-5, latest). If not
          provided, reads MESHY_TEXT3D_MODEL from env and falls back to "meshy-5".
        """
        # Ensure .env (if present) is loaded before reading environment variables
        _load_dotenv_from_project_root()

        base = api_url or os.getenv("MESHY_API_URL", "https://api.meshy.ai")
        self.base_url = base.rstrip("/")
        self.api_key = (api_key or os.getenv("MESHY_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError(
                "Meshy API key is not configured. "
                "Set the MESHY_API_KEY environment variable with your Meshy API key."
            )

        # Choose which Meshy model to use for Text-to-3D.
        # Valid values (per docs): "meshy-4", "meshy-5", "latest".
        # We intentionally default to "meshy-5" so that we do NOT use the
        # latest experimental model unless explicitly requested.
        env_model = os.getenv("MESHY_TEXT3D_MODEL", "").strip()
        chosen = (ai_model or env_model or "meshy-5").strip()
        if chosen not in {"meshy-4", "meshy-5", "latest"}:
            print(
                f"[Meshy] Warning: invalid MESHY_TEXT3D_MODEL='{chosen}', "
                "falling back to 'meshy-5'. Valid values are 'meshy-4', 'meshy-5', 'latest'."
            )
            chosen = "meshy-5"
        self.ai_model = chosen

        # Single async client reused across requests
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=60.0,
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _create_preview_task(self, prompt: str) -> str:
        """
        Create a Text-to-3D preview task and return its task id.

        Endpoint:
        POST /openapi/v2/text-to-3d

        Request body (JSON):
        {
          "mode": "preview",
          "prompt": "a monster mask",
          ...
        }

        Response:
        { "result": "<task_id>" }
        """
        payload = {
            "mode": "preview",
            "prompt": prompt,
            # You can tweak these defaults if desired
            "art_style": "realistic",
            "should_remesh": True,
            # Explicitly choose which Meshy model to use instead of defaulting to "latest"
            "ai_model": self.ai_model,
        }
        resp = await self._client.post(
            "/openapi/v2/text-to-3d",
            headers=self._headers(),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        task_id = data.get("result")
        if not isinstance(task_id, str):
            raise RuntimeError(f"Unexpected response from Meshy API: {data}")
        return task_id

    async def _get_task(self, task_id: str) -> Dict[str, Any]:
        """
        Fetch a Text-to-3D task object.

        Endpoint (per docs):
        GET /openapi/v2/text-to-3d/{task_id}
        """
        resp = await self._client.get(
            f"/openapi/v2/text-to-3d/{task_id}",
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def _wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 5.0,
        timeout: float = 15 * 60.0,
    ) -> Dict[str, Any]:
        """
        Poll a Meshy task until it finishes or times out.

        Returns the final task object.
        """
        start = time.monotonic()
        last_status = None
        while True:
            task = await self._get_task(task_id)
            status = task.get("status")
            if status != last_status:
                # basic logging to stdout so we can see progress in console
                print(f"[Meshy] Task {task_id} status: {status}")
                last_status = status

            if status in ("SUCCEEDED", "FAILED", "CANCELED"):
                return task

            if time.monotonic() - start > timeout:
                raise TimeoutError(f"Meshy task {task_id} timed out after {timeout} seconds")

            await asyncio.sleep(poll_interval)

    async def generate_model(self, prompt: str) -> Dict[str, Any]:
        """
        High-level helper:
        - Create preview task
        - Wait for completion
        - Return dict with success flag and OBJ download URL (if available)

        Returns dict:
        {
          "success": bool,
          "task_id": str,
          "obj_url": Optional[str],
          "task": Dict[str, Any],
          "error": Optional[str],
        }
        """
        try:
            task_id = await self._create_preview_task(prompt)
            task = await self._wait_for_task(task_id)
        except Exception as e:
            return {
                "success": False,
                "task_id": None,
                "obj_url": None,
                "task": {},
                "error": str(e),
            }

        status = task.get("status")
        if status != "SUCCEEDED":
            return {
                "success": False,
                "task_id": task_id,
                "obj_url": None,
                "task": task,
                "error": f"Task finished with status={status}",
            }

        model_urls = task.get("model_urls") or {}
        obj_url = model_urls.get("obj")

        if not obj_url:
            return {
                "success": False,
                "task_id": task_id,
                "obj_url": None,
                "task": task,
                "error": "OBJ URL not available in Meshy response",
            }

        return {
            "success": True,
            "task_id": task_id,
            "obj_url": obj_url,
            "task": task,
            "error": None,
        }

    async def download_model(self, url: str, save_path: str) -> bool:
        """
        Download model from given URL to local path.
        """
        try:
            async with self._client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(save_path, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"[Meshy] Error downloading model from {url}: {e}")
            return False

    async def close(self) -> None:
        """Close underlying HTTP client."""
        if self._client:
            await self._client.aclose()

    async def __aenter__(self) -> "MeshyClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


