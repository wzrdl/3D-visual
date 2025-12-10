# 3D Model Generator & Cloud Library

## Overview

This project is a desktop 3D model browser & viewer (PyQt + PyVista) backed by a cloud-hosted model library.

- **Desktop Client (PyQt)**: Lives under `app/`. It allows users to browse models, view them in 3D, and generate new models using AI (Meshy API). It communicates with the backend via HTTP.
- **Backend (FastAPI + SQLite + GCS)**: Lives under `server/` and uses `app/data_manager.py`. It manages the authoritative model metadata (SQLite) and files (Google Cloud Storage).

For backend details and deployment instructions, please refer to [server/README.md](server/README.md).

## Project Structure

```text
3D-visual/
├── app/
│   ├── main_window.py        # Main application window (UI)
│   ├── pages.py              # UI Pages: Gallery, AI Generation, Viewer
│   ├── viewer.py             # Core 3D viewer widget (PyVista)
│   ├── scene_viewer.py       # Advanced scene visualization logic
│   ├── meshy_client.py       # Client for Meshy AI (Text-to-3D)
│   ├── client_data_manager.py# Manages data flow for the client (Caching + Backend API)
│   ├── backend_client.py     # HTTP Client for communicating with the backend
│   └── ...
├── assets/
│   ├── models/               # Local cache for 3D model files
│   └── ...
├── main.py                   # Desktop application entry point
├── requirements.txt          # Python dependencies
├── .env.example              # Template for environment variables
└── README.md                 # This file
```

## Setup & Installation

### 1. Prerequisites

- Python 3.12 or higher
- A [Meshy API Key](https://meshy.ai/) (for AI 3D model generation)

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

The application requires specific environment variables to function correctly. You can set them in your terminal or create a `.env` file in the project root.

**Step A: Create a `.env` file for MeshyAI**

1.  Create a file named `.env` in the root directory (`3D-visual/`).
2.  Add the following content (replace with your actual keys):

    ```ini
    # Your Meshy API Key for Text-to-3D generation
    MESHY_API_KEY=msy_your_api_key_here
    ```

**Step B: Set Backend Environment Variables Manually**

*Windows (PowerShell):*
```powershell
$env:BACKEND_API_URL="https://d3-visual-backend-588296003116.us-east1.run.app"
```

*macOS / Linux:*
```bash
export BACKEND_API_URL="https://d3-visual-backend-588296003116.us-east1.run.app"
```

### 5. Run the Application

Once the environment is configured, launch the desktop client:

```bash
python main.py
```

## Features

- **Gallery**: Browse 3D models stored in the cloud. Metadata is fetched from the backend, and files are downloaded/cached on demand.
- **3D Viewer**: Interactive 3D visualization using PyVista. Supports orbit, zoom, and pan.
- **AI Generation**: Generate 3D models from text prompts using the Meshy API. Generated models can be viewed locally.
- **Scene Generator**: Compose scenes with multiple objects (experimental).

## Troubleshooting

- **Missing API Key**: If the AI generation fails, ensure `MESHY_API_KEY` is set correctly in your `.env` file or environment variables.
- **Backend Connection**: If the Gallery is empty or shows errors, verify that `BACKEND_API_URL` is reachable.
- **Qt/PyVista Issues**: If the window doesn't appear or crashes, ensure your graphics drivers are up to date and that you have installed the correct PyQt6 dependencies from `requirements.txt`.
- **MacOS problem**: If the program crashed, just open it again, the macOS have some VTK problem that I can't fix.