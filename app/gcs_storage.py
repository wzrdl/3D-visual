"""
Simple wrapper around Google Cloud Storage for storing and fetching 3D model files.

This is intentionally small and optional:
- If you don't configure a GCS bucket, nothing in the app will change.
- If you set the env vars below, models will be pulled from GCS on demand.

Required environment variables:
- GCS_MODELS_BUCKET: name of the Cloud Storage bucket that holds model files
- (optional) GCS_MODELS_PREFIX: path prefix inside the bucket, default "models/"
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from google.cloud import storage  # type: ignore[import]


class GCSModelStorage:
    """Handles downloading model files from a Google Cloud Storage bucket."""

    def __init__(self, bucket_name: str, base_prefix: str = "models/"):
        self.bucket_name = bucket_name
        # Ensure prefix ends with a trailing slash if non-empty
        if base_prefix and not base_prefix.endswith("/"):
            base_prefix = base_prefix + "/"
        self.base_prefix = base_prefix

        # Lazily created client & bucket (can be reused across downloads)
        self._client: Optional[storage.Client] = None
        self._bucket: Optional[storage.Bucket] = None

    @property
    def client(self) -> storage.Client:
        if self._client is None:
            # This uses Application Default Credentials.
            # On local dev you typically set GOOGLE_APPLICATION_CREDENTIALS
            # pointing to your service-account JSON file.
            self._client = storage.Client()
        return self._client

    @property
    def bucket(self) -> storage.Bucket:
        if self._bucket is None:
            self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket

    def _blob_path_for_filename(self, filename: str) -> str:
        return f"{self.base_prefix}{filename}"

    def download_model_if_needed(self, filename: str, local_dir: Path) -> Path:
        """
        Ensure that `filename` exists in `local_dir`.

        - If the file already exists locally, just return the local path.
        - If not, download it from GCS bucket and return the path.
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / filename

        if local_path.exists():
            return local_path

        blob_path = self._blob_path_for_filename(filename)
        blob = self.bucket.blob(blob_path)

        if not blob.exists():
            raise FileNotFoundError(
                f"GCS object not found: gs://{self.bucket_name}/{blob_path}"
            )

        blob.download_to_filename(str(local_path))
        return local_path

    def upload_model(self, local_path: Path, filename: str) -> None:
        """
        Upload a local model file to the configured GCS bucket.

        The object will be stored at: <base_prefix>/<filename>
        """
        if not local_path.exists():
            raise FileNotFoundError(f"Local model file not found: {local_path}")

        blob_path = self._blob_path_for_filename(filename)
        blob = self.bucket.blob(blob_path)
        blob.upload_from_filename(str(local_path))
        print(f"Uploaded model to gs://{self.bucket_name}/{blob_path}")


