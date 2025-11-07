# 3D Model Generator & Local Library



## Project Structure

```
3D-visual/
├── app/
│   ├── __init__.py
│   ├── main_window.py      # Main application window
│   ├── viewer.py            # 3D model viewer widget
│   ├── api_client.py        # API client for model generation
│   ├── data_manager.py      # SQLite database manager
│   └── schema.py            # Database schema definitions
├── assets/
│   ├── models.db            # SQLite database (auto-created)
│   ├── models/              # 3D model files directory
│   └── metadata.json.backup # Legacy metadata (migrated to SQLite)
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
└── README.md
```

## Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd 3D-visual
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   ```
    # Activate virtual environment
    # On macOS/Linux:
   ```
   source venv/bin/activate
   ```
   # On Windows:
   ```
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```


