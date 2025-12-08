# 3D Model Generator & Cloud Library

## Overview

This project is a desktop 3D model browser & viewer (PyQt + PyVista) backed by a cloud-hosted
model library:

- **Backend (FastAPI + SQLite + GCS)** lives in the same repo under `server/` and `app/data_manager.py`.
- **Desktop client (PyQt)** lives under `app/` and talks to the backend over HTTP.
- The desktop app keeps only a **local cache** of model files; the **authoritative data** is stored
  in **SQLite (metadata) + Google Cloud Storage (model files)**.

For backend details and data flow, see `server/README.md`.

## Project Structure (High Level)

```text
3D-visual/
├── app/
│   ├── __init__.py
│   ├── main_window.py        # Main application window (tabs)
│   ├── pages.py              # Gallery / AI Generation / Viewer pages
│   ├── viewer.py             # 3D model viewer widget (PyVista)
│   ├── client_data_manager.py# Client-side DataManager (HTTP + local cache)
│   ├── backend_client.py     # HTTP client for FastAPI backend
│   ├── api_client.py         # (Reserved) API client for AI generation
│   ├── data_manager.py       # Backend DataManager (SQLite + GCS), used by FastAPI
│   ├── gcs_storage.py        # GCS helper (download/upload model files)
│   └── schema.py             # SQLite schema definitions
├── server/
│   ├── app.py                # FastAPI backend entrypoint
│   ├── requirements.txt      # Backend-only dependencies
│   └── README.md             # Backend data flow & architecture
├── assets/
│   ├── models.db             # SQLite database (backend, auto-created)
│   ├── models/               # Local model files directory (also used as cache)
│   └── metadata.json.backup  # Legacy metadata (migrated to SQLite)
├── Dockerfile                # Container image for deploying backend to Cloud Run
├── main.py                   # Desktop application entry point
├── requirements.txt          # Shared Python dependencies (client + backend)
└── README.md                 # This file
```

## Running the Desktop Client (Local)

1. **Create and activate a virtual environment**

   ```bash
   cd 3D-visual
   python -m venv venv
   # On macOS / Linux:
   source venv/bin/activate
   # On Windows (PowerShell):
   venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Point the client to the deployed backend**

   The desktop client uses the environment variable `BACKEND_API_URL` to know where the FastAPI
   backend is running. In your setup the backend is already deployed to Cloud Run, so you only need
   to set this once:

   ```bash
   # macOS / Linux
   export BACKEND_API_URL="https://d3-visual-backend-588296003116.us-east1.run.app"

   # Windows (PowerShell)
   $env:BACKEND_API_URL="https://d3-visual-backend-588296003116.us-east1.run.app"
   ```

4. **Run the desktop application**

   ```bash
   python main.py
   ```

   - The **Gallery** tab fetches model metadata from the backend (`GET /models`).
   - When you click a model, the client downloads the model file (e.g. `.obj`, `.glb`) via
     `GET /models/{id}/content` and caches it under `assets/models/` before loading it in the
     3D viewer.
   - When the app closes, the client clears its local cache (cached model files).

<!-- Backend setup and deployment are documented in server/README.md -->
