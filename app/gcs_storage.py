"""
This file is used to manage the model files in the GCS bucket
It is used to download the model files from the GCS bucket to the local directory
and upload the model files from the local directory to the GCS bucket
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Optional

from google.cloud import storage


class GCSModelStorage:
    """Handles downloading model files from a Google Cloud Storage bucket."""

    def __init__(self, bucket_name: str, base_prefix: str = "models/"):
        self.bucket_name = bucket_name
        if base_prefix and not base_prefix.endswith("/"):
            base_prefix = base_prefix + "/"
        self.base_prefix = base_prefix

        # Eagerly create GCS client/bucket; no lazy loading needed.
        self.client: storage.Client = storage.Client()
        self.bucket: storage.Bucket = self.client.bucket(self.bucket_name)

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
        Upload a local model file to the configured GCS bucket, and the object will be 
        stored at: <base_prefix>/<filename>
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
        List filenames that exist in the bucket.
        This is used to backfill the local database with objects that already live in GCS.
        """
        blobs = self.client.list_blobs(self.bucket_name, prefix=self.base_prefix)
        filenames = []
        for blob in blobs:
            if blob.name.endswith("/"):
                continue

            if self.base_prefix and blob.name.startswith(self.base_prefix):
                filename = blob.name[len(self.base_prefix) :]
            else:
                filename = blob.name

            if not filename:
                continue

            filenames.append(
                {
                    "filename": filename,
                    "metadata": blob.metadata or {},
                }
            )

        return filenames


