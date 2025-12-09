"""
Meshy Text-to-3D API client.

I refrence the official doc from here:
https://docs.meshy.ai/en/api/text-to-3d

Workflow we support here:
First, Create a Text-to-3D preview task (`mode="preview"`) from a text prompt
Second, Poll the task status until it finishes
Third, Return the OBJ download URL when available
"""

from __future__ import annotations

import os
import time
from typing import Optional, Dict, Any, List

import httpx
import asyncio
from pathlib import Path


_DOTENV_LOADED = False


def _load_dotenv_from_project_root() -> None:
    """Minimal .env loader: read .env and set os.environ values."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        _DOTENV_LOADED = True
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")
    _DOTENV_LOADED = True


class MeshyClient:
    """Async client for Meshy Text-to-3D API."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        ai_model: Optional[str] = None,
    ):
        # The official url is "https://api.meshy.ai"
        _load_dotenv_from_project_root()

        base = api_url or os.getenv("MESHY_API_URL", "https://api.meshy.ai")
        self.base_url = base.rstrip("/")
        self.api_key = (api_key or os.getenv("MESHY_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError(
                "Meshy API key is not configured. "
                "Set the MESHY_API_KEY environment variable with your Meshy API key."
            )
        # TODO : Add the logic to choose the model
        env_model = os.getenv("MESHY_TEXT3D_MODEL", "").strip()
        chosen = (ai_model or env_model or "meshy-5").strip()
        if chosen not in {"meshy-4", "meshy-5", "latest"}:
            print(
                f"[Meshy] Warning: invalid MESHY_TEXT3D_MODEL='{chosen}', "
                "falling back to 'meshy-5'. Valid values are 'meshy-4', 'meshy-5', 'latest'."
            )
            chosen = "meshy-5"
        self.ai_model = chosen

        # TODO : the timeout is 60 seconds, we can change it if needed
        # The main window may stuck right now when we are waiting for the model to be generated
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
        """
        payload = {
            "mode": "preview",
            "prompt": prompt,
            # TODO : We can change the art style if needed
            "art_style": "realistic",
            "should_remesh": True,
            "ai_model": self.ai_model,
        }
        resp = await self._client.post(
            "/openapi/v2/text-to-3d",
            headers=self._headers(),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        # The task id is the response from the API
        task_id = data.get("result")
        if not isinstance(task_id, str):
            raise RuntimeError(f"Unexpected response from Meshy API: {data}")
        return task_id

    async def _create_refine_task(self, model_id: str) -> str:
        """
        Create a Text-to-3D refine task and return its task id.
        """
        payload = {
            "mode": "refine",
            "model_id": model_id,
            "enable_pbr": True
        }
        resp = await self._client.post(
            "/openapi/v2/text-to-3d",
            headers=self._headers(),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        # The task id is the response from the API
        task_id = data.get("result")
        if not isinstance(task_id, str):
            raise RuntimeError(f"Unexpected response from Meshy API: {data}")
        return task_id

    async def _get_task(self, task_id: str) -> Dict[str, Any]:
        """
        Fetch a Text-to-3D task object.
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
        # TODO : The timeout is 15 minutes, we can change it if needed
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
                print(f"[Meshy] Task {task_id} status: {status}")
                last_status = status

            if status in ("SUCCEEDED", "FAILED", "CANCELED"):
                return task

            if time.monotonic() - start > timeout:
                raise TimeoutError(f"Meshy task {task_id} timed out after {timeout} seconds")

            await asyncio.sleep(poll_interval)
    async def refine_model(self, model_id: str) -> Dict[str, Any]:
        """
        This function will get the refined model from the API

        The date response looks like:
        {
            "id": str,
            "model_urls": {
                "glb": str,
                "fbx": str,
                "obj": str,
                "mtl": str,
                "usdz": str
            },
            "thumbnail_url": str,
            "prompt": str,
            "art_style": str,
            "progress": int,
            "started_at": int,
            "created_at": int,
            "finished_at": int,
            "status": str,
            "texture_urls": [
                {
                "base_color": str
                }
            ],
            "preceding_tasks": int,
            "task_error": {
                "message": None | str
            }
        }
        """

        try: 
            task_id = await self._create_refine_task(model_id)
        except Exception as e:
            return {
                "success": False,
                "task_id": None,
                "model_id": model_id,
                "task": None,
                "error": str(e),
            }

        task = await self._wait_for_task(task_id)
        if task.get("status") != "SUCCEEDED":
            return {
                "model_id": model_id,
                "task_id": task_id,
                "model_urls": None,
                "task": task,
                "error": f"Task finished with status={task.get('status')}",
            }
        return {
            "task_id": task_id,
            "model_id": model_id,
            "model_urls": task.get("model_urls"),
            "task": task,
            "error": None,
            "thumbnail_url": task.get("thumbnail_url"),
            "texture_urls": task.get("texture_urls"),
        }
    async def generate_model(self, prompt: str) -> Dict[str, Any]:
        """
        The main function to generate the model
        We will check the task status until it is succeeded or failed
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

        # We need to check if the OBJ URL is available
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
    async def download_refine_model(self, 
                                    model_urls: Dict[str, Any], 
                                    texture_urls: List[Dict[str, Any]], 
                                    thumbnail_url: str, 
                                    save_dir: str) -> bool:
        """
        Download all refined model files including textures and thumbnail.
        
        Args:
            model_urls: Dict with format URLs (glb, fbx, obj, etc.)
            texture_urls: List of texture dicts with base_color URLs
            thumbnail_url: Thumbnail image URL
            save_dir: Directory to save all downloaded files
        
        Returns:
            True if all downloads succeeded, False otherwise
        """
        from pathlib import Path
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Collect all URLs to download
        all_urls: List[tuple] = []
        
        # Add model format URLs
        for fmt, url in model_urls.items():
            if url:
                all_urls.append((url, f"model.{fmt}"))
        
        # Add texture URLs
        for i, tex in enumerate(texture_urls or []):
            if isinstance(tex, dict):
                base_color = tex.get("base_color")
                if base_color:
                    all_urls.append((base_color, f"texture_{i}.png"))
        
        # Add thumbnail
        if thumbnail_url:
            all_urls.append((thumbnail_url, "thumbnail.png"))
        
        try:
            for url, filename in all_urls:
                file_path = save_path / filename
                async with self._client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(file_path, "wb") as f:
                        async for chunk in resp.aiter_bytes():
                            f.write(chunk)
            return True
        except Exception as e:
            print(f"[Meshy] Error downloading refined model: {e}")
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()

    async def __aenter__(self) -> "MeshyClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


