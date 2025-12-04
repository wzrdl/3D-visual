## Backend Data Flow & Architecture

This document describes how data flows through the backend (`server/`) and how it interacts with
Google Cloud Storage (GCS) and the desktop client.

### Components

- `server/app.py`  
  FastAPI application exposing HTTP endpoints for:
  - Listing models (`GET /models`)
  - Fetching a single model (`GET /models/{model_id}`)
  - Downloading model file content (`GET /models/{model_id}/content`)
  - Uploading a new model (`POST /models`)

- `app/data_manager.py`  
  Backend `DataManager` used only inside the FastAPI service. It:
  - Stores model metadata in a local SQLite database (`assets/models.db`)
  - Stores model files under `assets/models/`
  - Optionally mirrors model files to a GCS bucket via `GCSModelStorage`

- `app/gcs_storage.py`  
  Thin wrapper around `google-cloud-storage` for:
  - Downloading model files from GCS to the local `assets/models/` directory
  - Uploading local model files to the configured bucket

### Backend Read Flow (Cloud Run / local FastAPI)

1. **Client requests model metadata**
   - Client calls `GET /models` or `GET /models/{model_id}` on the FastAPI backend.
   - `server/app.py` uses `DataManager.get_all_models()` or `DataManager.get_model_path()` to
     read from the SQLite database.

2. **Backend ensures the model file exists**
   - When a specific model is requested, `DataManager.get_model_path(model_id)`:
     - Looks up the `filename` in SQLite.
     - Builds `assets/models/<filename>` as the expected local path.
     - If the file exists locally, that path is returned.
     - If the file is missing and GCS is configured (`GCS_MODELS_BUCKET` and `GCS_MODELS_PREFIX`):
       - `GCSModelStorage.download_model_if_needed()` downloads
         `gs://<bucket>/<prefix>/<filename>` to `assets/models/<filename>`, then returns the path.

3. **Backend returns data to the client**
   - `GET /models` returns a JSON list of metadata objects (id, filename, display_name, tags, etc.).
   - `GET /models/{model_id}` returns metadata for a single model (and ensures file availability).
   - `GET /models/{model_id}/content` returns the `.obj` file as a binary `FileResponse`, which
     the desktop client caches locally.

### Backend Write Flow (Upload New Model)

When a client uploads a new model via `POST /models`:

1. **FastAPI receives multipart form-data**
   - Fields:
     - `model_id`: unique identifier (e.g. `"model_123"`)
     - `name`: human-readable display name
     - `tags`: JSON string of tags (e.g. `["human", "soldier"]`)
     - `file`: the `.obj` file content

2. **`DataManager.save_model_to_gallery(...)` is called**
   - Inside `app/data_manager.py`:
     - Starts a database transaction.
     - Writes the model file to `assets/models/<filename>`.
     - If GCS is configured:
       - Uses `GCSModelStorage.upload_model()` to upload the file to
         `gs://<bucket>/<prefix>/<filename>`.
     - Inserts model metadata (id, filename, display_name, tags, timestamps) into SQLite.
     - Commits the transaction.

3. **Backend responds**
   - On success, returns JSON:
     - `{"success": true, "id": "<model_id>", "filename": "<filename>"}`.
   - On failure, returns a `500` error with an appropriate message.

### Desktop Client Interaction (High Level)

Although the desktop client code lives in `app/`, it is important to understand how it uses
this backend:

- The desktop app uses `ClientDataManager` (`app/client_data_manager.py`) together with
  `BackendAPIClient` (`app/backend_client.py`).
- `ClientDataManager`:
  - Calls the backend (`/models`, `/models/{id}`, `/models/{id}/content`) to obtain metadata and
    file bytes.
  - Caches downloaded `.obj` files under its own local `assets/models/` directory so that the
    PyQt 3D viewer can load them from disk.
  - Clears this cache when the application closes.

In summary:
- **Authoritative data (metadata + model files) lives on the backend (SQLite + GCS).**
- **The desktop app is a thin client with a short-lived local cache.**

## Backend Deployment (GCP Cloud Run)

The backend in this repo is designed to be deployed as a container to Google Cloud Run.
Below are the concrete steps that match the current production setup.

### Prerequisites

- GCP project: `d-models-480102`
- GCS bucket: `my-3d-model-bucket` (contains `models/` folder with `.obj` files)
- Google Cloud SDK (`gcloud`) installed and authenticated

### 1. Set project and enable services

```bash
gcloud config set project d-models-480102

gcloud services enable run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com
```

### 2. Build the container image

From the project root (`3D-visual/`), run:

```bash
gcloud builds submit --tag gcr.io/d-models-480102/3d-visual-backend
```

This uses the top-level `Dockerfile` and pushes the image to
`gcr.io/d-models-480102/3d-visual-backend`.

### 3. Deploy to Cloud Run

Deploy the image as a Cloud Run service (region: `us-east1`):

```bash
gcloud run deploy d3-visual-backend \
  --image gcr.io/d-models-480102/3d-visual-backend \
  --platform managed \
  --region us-east1 \
  --allow-unauthenticated \
  --set-env-vars GCS_MODELS_BUCKET=my-3d-model-bucket,GCS_MODELS_PREFIX=models/
```

Cloud Run will respond with a service URL similar to:

```text
https://d3-visual-backend-588296003116.us-east1.run.app
```

This URL is what the desktop client uses as `BACKEND_API_URL`.

