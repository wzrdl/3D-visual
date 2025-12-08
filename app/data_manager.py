"""
DataManager - handles the local dataset stuff
Uses SQLite for persistent storage of 3D model metadata
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# for thumbnails
import pyvista as pv

from numpy import add
from app.schema import (
    CREATE_TABLE_MODELS,
    CREATE_INDEX_DISPLAY_NAME,
    CREATE_INDEX_TAGS,
    SCHEMA_VERSION,
)

try:
    # Optional import – only used when GCS is configured
    from app.gcs_storage import GCSModelStorage
except Exception:  # pragma: no cover - keep optional
    GCSModelStorage = None  # type: ignore[assignment]


class DataManager:
    """manages 3D models and their metadata using SQLite"""
    
    def __init__(self, assets_dir: Optional[str] = None):
        """setup paths and initialize database, we already have a assets directory"""
        if assets_dir is None:
            project_root = Path(__file__).parent.parent
            self.assets_dir = project_root / "assets"
        else:
            self.assets_dir = Path(assets_dir)
        
        self.models_dir = self.assets_dir / "models"
        self.db_path = self.assets_dir / "models.db"

        # We migrate from metadata.json to SQLite database
        self.metadata_file = self.assets_dir / "metadata.json" 
        
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Optional: configure Google Cloud Storage for model files
        self._init_gcs_storage()

        # Initialize database connection, we only need the database file path
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        # Run migrations
        self._run_migrations()
        self._migrate_from_json()

        # Sync remote objects into the local DB if a GCS bucket is configured
        self._sync_models_from_gcs()

        #cleanup
        self.remove_nonexistent_models()
    def _init_gcs_storage(self) -> None:
        """
        Initialize optional Google Cloud Storage model backend.

        If the environment variable GCS_MODELS_BUCKET is set (and the optional
        dependency google-cloud-storage is installed), DataManager will try
        to fetch model files from that bucket whenever they are missing locally.
        """
        self.gcs_storage: Optional["GCSModelStorage"] = None

        bucket = os.getenv("GCS_MODELS_BUCKET", "").strip()
        if not bucket:
            return

        if GCSModelStorage is None:
            print(
                "GCS_MODELS_BUCKET is set but google-cloud-storage is not available. "
                "Install it and re-run, or unset GCS_MODELS_BUCKET to use local files only."
            )
            return

        prefix = os.getenv("GCS_MODELS_PREFIX", "models/").strip() or "models/"
        try:
            self.gcs_storage = GCSModelStorage(bucket_name=bucket, base_prefix=prefix)
            print(f"Using GCS bucket '{bucket}' for model files (prefix='{prefix}')")
        except Exception as e:
            # Fail gracefully and fall back to local-only mode
            print(f"Error initializing GCS model storage: {e}")
            self.gcs_storage = None
    
    def _run_migrations(self):
        """create tables and indexes if they don't exist"""
        cursor = self.conn.cursor()
        try:
            cursor.execute(CREATE_TABLE_MODELS)
            cursor.execute(CREATE_INDEX_DISPLAY_NAME)
            cursor.execute(CREATE_INDEX_TAGS)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error running migrations: {e}")
            self.conn.rollback()

    # Since we are using SQLite, we need to migrate from metadata.json to SQLite database
    # Once the migration is done, we can remove the metadata.json file
    # Ignore this part in the future, we only need to run it once
    def _migrate_from_json(self):
        """migrate data from metadata.json if it exists and database is empty"""
        if not self.metadata_file.exists():
            return
        
        # Check if database already has data
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM models")
        count = cursor.fetchone()[0]
        
        if count > 0:
            # Database already has data, skip migration
            # We don't want to overwrite the existing data
            return
        
        # Migrate from JSON
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            if not isinstance(metadata, list):
                return
            
            cursor = self.conn.cursor()
            for entry in metadata:
                model_id = entry.get('id')
                filename = entry.get('filename')
                display_name = entry.get('name', filename)  # Use 'name' from JSON as display_name
                tags = entry.get('tags', [])
                
                # To prevent SQL injection, we use the question mark placeholder
                # This is a safer way to insert values into the database
                if model_id and filename:
                    cursor.execute(
                        "INSERT INTO models (id, filename, display_name, tags) VALUES (?, ?, ?, ?)",
                        (model_id, filename, display_name, json.dumps(tags))
                    )
            
            self.conn.commit()
            print(f"Migrated {len(metadata)} models from metadata.json to SQLite")
        except (json.JSONDecodeError, IOError, sqlite3.Error) as e:
            print(f"Error migrating from JSON: {e}")
            self.conn.rollback()

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """get file path for a model by id

        Behavior:
        - If the model file exists locally under assets/models, return that.
        - If not and GCS is configured, try to download it from the bucket,
          cache it under assets/models, then return the local path.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT filename FROM models WHERE id = ?", (model_id,))
        row = cursor.fetchone()

        if not row:
            return None

        filename = row["filename"]
        local_path = self.models_dir / filename

        # Already present on disk
        if local_path.exists():
            return local_path

        # Try to fetch from GCS if configured
        if getattr(self, "gcs_storage", None) is not None:
            try:
                local_path = self.gcs_storage.download_model_if_needed(  # type: ignore[union-attr]
                    filename=filename,
                    local_dir=self.models_dir,
                )
                return local_path
            except Exception as e:
                print(f"Error downloading model '{filename}' from GCS: {e}")
                return None

        # No GCS configured and file missing
        return None

    def remove_nonexistent_models(self):
        """remove all models from models.db that don't exist in the assets folder        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, filename FROM models")
        all_models = cursor.fetchall()

        deleted_number = 0

        for row in all_models:
            model_id = row['id']
            file_path = self.get_model_path(model_id)

            # Check if file_path is None or doesn't exist
            if file_path is None or not file_path.exists():
                cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
                deleted_number += 1

        self.conn.commit()
        print(f"Removed {deleted_number} models from models.db")

    def get_all_models(self) -> List[Dict]:
        """get all models, sorted by creation date.
            However, we don't need to run this query often"""

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM models ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    def search_models(self, query: str) -> List[Dict]:
        """search models by display_name or tags (type-ahead search)
            This is a very common query, so we might use this frequently when we need to search for a model
        """

        #TODO: add a prompt instead of the name of the model

        query_lower = f"%{query.lower()}%"
        cursor = self.conn.cursor()
        
        # Search in display_name and tags
        cursor.execute(
            """SELECT * FROM models 
               WHERE LOWER(display_name) LIKE ? 
               OR LOWER(tags) LIKE ?
               ORDER BY display_name""",
            (query_lower, query_lower)
        )
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            # Parse tags JSON and check if query matches any tag
            tags = json.loads(row['tags'])
            if (query.lower() in row['display_name'].lower() or
                any(query.lower() in tag.lower() for tag in tags)):
                results.append(self._row_to_dict(row))
        
        return results
    
    def add_model(self, model_id: str, filename: str, name: str,
                  tags: List[str]) -> bool:
        """add a new model to metadata"""
        cursor = self.conn.cursor()
        
        # Check if id already exists
        cursor.execute("SELECT id FROM models WHERE id = ?", (model_id,))
        if cursor.fetchone():
            print(f"Model with ID '{model_id}' already exists")
            return False
        
        try:
            cursor.execute(
                """INSERT INTO models (id, filename, display_name, tags, modified_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (model_id, filename, name, json.dumps(tags), datetime.now().isoformat())
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error adding model: {e}")
            self.conn.rollback()
            return False
    
    def save_model_to_gallery(self, model_id: str, filename: str, name: str,
                             tags: List[str], model_data: bytes) -> bool:
        """When the user adds a model, we need to save the model file to the gallery,
           and also add the model into the database
        """
        cursor = self.conn.cursor()
        
        try:
            # Start transaction
            cursor.execute("BEGIN TRANSACTION")

            # 1) Write model file locally
            model_path = self.models_dir / filename
            with open(model_path, "wb") as f:
                f.write(model_data)

            # 2) Optionally upload to GCS so the model is available in the shared bucket
            if getattr(self, "gcs_storage", None) is not None:
                try:
                    self.gcs_storage.upload_model(model_path, filename)  # type: ignore[union-attr]
                except Exception as upload_err:
                    # Depending on requirements we could roll back on this error; for now just warn
                    print(f"Warning: failed to upload model to GCS: {upload_err}")

            # 3) Insert metadata into SQLite
            cursor.execute(
                """
                INSERT INTO models (id, filename, display_name, tags, modified_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (model_id, filename, name, json.dumps(tags), datetime.now().isoformat()),
            )

            # Commit transaction
            self.conn.commit()
            return True
        except (IOError, sqlite3.Error) as e:
            print(f"Error saving model to gallery: {e}")
            self.conn.rollback()
            # Clean up file if it was created
            model_path = self.models_dir / filename
            if model_path.exists():
                model_path.unlink()
            return False

    def _sync_models_from_gcs(self) -> None:
        """
        Backfill the local SQLite metadata with objects that already exist in GCS.

        This makes sure the /models endpoint returns entries for models that were
        uploaded directly to the bucket (e.g., via UI or another service) and not
        through this API.
        """
        if getattr(self, "gcs_storage", None) is None:
            return

        try:
            remote_files = self.gcs_storage.list_model_files()  # type: ignore[union-attr]
        except Exception as e:
            print(f"Warning: failed to list models from GCS: {e}")
            return

        if not remote_files:
            return

        cursor = self.conn.cursor()
        cursor.execute("SELECT id, filename FROM models")
        existing = cursor.fetchall()

        # Build quick lookup sets
        existing_ids = {row["id"] for row in existing}
        existing_filenames = {row["filename"] for row in existing}

        new_rows = []
        for entry in remote_files:
            filename = entry.get("filename")
            meta = entry.get("metadata", {}) or {}

            if not filename or filename in existing_filenames:
                continue

            # Derive identifiers; fall back to filename stem
            stem = Path(filename).stem
            model_id = meta.get("id") or stem
            display_name = meta.get("display_name") or meta.get("name") or stem
            tags_raw = meta.get("tags", "[]")

            # Try to parse tags from metadata; default to empty list
            tags_list: List[str]
            try:
                tags_list = json.loads(tags_raw) if isinstance(tags_raw, str) else list(tags_raw)
                if not isinstance(tags_list, list):
                    tags_list = []
            except Exception:
                tags_list = []

            # Avoid id collision even if filename is new
            if model_id in existing_ids:
                # generate a new id based on filename stem to avoid conflict
                model_id = f"{stem}_{len(existing_ids) + len(new_rows) + 1}"

            new_rows.append(
                (model_id, filename, display_name, json.dumps(tags_list), datetime.now().isoformat())
            )

        if not new_rows:
            return

        try:
            cursor.executemany(
                """
                INSERT INTO models (id, filename, display_name, tags, modified_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                new_rows,
            )
            self.conn.commit()
            print(f"Synced {len(new_rows)} model(s) from GCS into local DB")
        except sqlite3.Error as e:
            print(f"Error syncing models from GCS: {e}")
            self.conn.rollback()
    
    def get_next_id(self) -> str:
        """generate next available model id, since we don't have a primary key, 
            we need to generate a unique id for each model"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM models WHERE id LIKE 'model_%'")
        rows = cursor.fetchall()
        
        if not rows:
            return "model_001"
        
        existing_ids = []
        for row in rows:
            id_str = row['id']
            try:
                num = int(id_str.replace('model_', ''))
                existing_ids.append(num)
            except ValueError:
                pass
        
        if existing_ids:
            next_num = max(existing_ids) + 1
        else:
            next_num = 1
        
        return f"model_{next_num:03d}"

    def import_new_models_from_folder(self, default_tags: Optional[List[str]] = None) -> int:
        
        default_tags = default_tags or []

        # Collect existing filenames from the database to avoid duplicates
        cursor = self.conn.cursor()
        cursor.execute("SELECT filename FROM models")
        existing_filenames = {row["filename"] for row in cursor.fetchall()}

        # Find all .obj and .glb files in the models directory
        model_files = [
            p for p in self.models_dir.glob("*.obj") if p.is_file()
        ] + [
            p for p in self.models_dir.glob("*.glb") if p.is_file()
        ]

        imported_count = 0
        for model_path in model_files:
            filename = model_path.name
            if filename in existing_filenames:
                # Already in database, skip
                continue

            model_id = self.get_next_id()
            display_name = model_path.stem  # filename without extension
            tags = list(default_tags)

            try:
                cursor.execute(
                    """
                    INSERT INTO models (id, filename, display_name, tags, modified_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (model_id, filename, display_name, json.dumps(tags), datetime.now().isoformat()),
                )
                imported_count += 1
                print(f"Imported model: id={model_id}, file={filename}")
            except sqlite3.Error as e:
                print(f"Error importing model from file '{filename}': {e}")
                self.conn.rollback()
                # continue with the next file

        if imported_count > 0:
            self.conn.commit()

        print(f"Total new models imported: {imported_count}")
        return imported_count
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """We want to convert database row to dictionary for easier access"""
        return {
            'id': row['id'],
            'filename': row['filename'],
            'name': row['display_name'],  # Keep 'name' key for compatibility
            'display_name': row['display_name'],
            'tags': json.loads(row['tags']),
            'created_at': row['created_at'],
            'modified_at': row['modified_at'],
        }

    def close(self):
        """close database connection"""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """cleanup on deletion"""
        self.close()


def _cli_import_new_models():
    
    dm = DataManager()
    try:
        imported = dm.import_new_models_from_folder()
        print(f"Finished importing. New models added: {imported}")
    finally:
        dm.close()


if __name__ == "__main__":
    _cli_import_new_models()