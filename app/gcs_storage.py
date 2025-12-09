"""
Simple wrapper around Google Cloud Storage for storing and fetching 3D model files.

Required environment variables:
- GCS_MODELS_BUCKET: name of the Cloud Storage bucket that holds model files
- GCS_MODELS_PREFIX: path prefix inside the bucket, default "models/"
"""

from __future__ import annotations

from datetime import timedelta
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
            # On local dev we typically set GOOGLE_APPLICATION_CREDENTIALS
            # pointing to the service-account JSON file.
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

    def generate_signed_url(self, filename: str, expiration_seconds: int = 3600) -> str:
        """
        Generate a time-limited signed URL for downloading a model directly from GCS.
        """
        blob_path = self._blob_path_for_filename(filename)
        blob = self.bucket.blob(blob_path)
        if not blob.exists():
            raise FileNotFoundError(f"GCS object not found: gs://{self.bucket_name}/{blob_path}")
        return blob.generate_signed_url(expiration=timedelta(seconds=expiration_seconds))

    def list_model_files(self):
        """
        List filenames (relative to base_prefix) that exist in the bucket.

        This is used to backfill the local database with objects that already
        live in GCS. We only return leaf objects; "directories" are skipped.
        """
        blobs = self.client.list_blobs(self.bucket_name, prefix=self.base_prefix)
        filenames = []
        for blob in blobs:
            # Skip placeholders / directory entries
            if blob.name.endswith("/"):
                continue

            # Strip prefix so the caller receives just the filename
            if self.base_prefix and blob.name.startswith(self.base_prefix):
                filename = blob.name[len(self.base_prefix) :]
            else:
                filename = blob.name

            # Defensive: skip empty names
            if not filename:
                continue

            filenames.append(
                {
                    "filename": filename,
                    # Allow optional custom metadata on the blob; falls back to defaults
                    "metadata": blob.metadata or {},
                }
            )

        return filenames


