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

## Troubleshooting

### PyQt6 DLL Load Error on Windows

If you encounter the error `ImportError: DLL load failed while importing QtWidgets: 找不到指定的程序` on Windows, this is typically caused by missing Visual C++ Redistributables. Follow these steps:

1. **Install Visual C++ Redistributables:**
   - Download and install the latest **Visual C++ Redistributable** from Microsoft:
     - [VC++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)
   - Run the installer and restart your computer if prompted.

2. **Alternative: Reinstall PyQt6:**
   ```bash
   pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
   pip install --upgrade --force-reinstall PyQt6
   ```

3. **If the issue persists:**
   - Ensure you're using a 64-bit Python installation (PyQt6 requires 64-bit Python on Windows)
   - Try creating a fresh virtual environment:
     ```bash
     python -m venv venv_new
     venv_new\Scripts\activate
     pip install -r requirements.txt
     ```


