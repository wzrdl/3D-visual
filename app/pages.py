"""
Page classes for the application
Each page is a separate class with its own functionality
"""
import os
import asyncio
import re
from pathlib import Path

from PyQt6.QtGui import QImage, QIcon
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QTextEdit, QLabel, QHBoxLayout, QGridLayout, QListView
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from app.viewer import ThreeDViewer
from app.meshy_client import MeshyClient
from app.client_data_manager import ClientDataManager


import shutil
import pathlib


class BasePage(QWidget):
    """Base class that all pages inherit from"""
    
    def __init__(self, parent=None):
        """Create a new page"""
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Build the UI - each page needs to implement this"""
        raise NotImplementedError("Subclasses must implement setup_ui()")


class GenerationWorker(QThread):
    """Worker thread for AI generation to avoid freezing the UI"""
    status_update = pyqtSignal(str)
    finished_success = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, meshy_client, data_manager, prompt: str, parent=None):
        super().__init__(parent)
        self.client = meshy_client
        self.data_manager = data_manager
        self.prompt = prompt

    def run(self):
        """Run the async task in a new event loop on this thread"""
        try:
            asyncio.run(self._run_async())
        except Exception as e:
            self.error_occurred.emit(f"Worker error: {str(e)}")

    def _next_model_id(self) -> str:
        """Generate next model id (thread-safe enough for this context)"""
        try:
            models = self.data_manager.get_all_models()
            ids = []
            for m in models:
                mid = str(m.get("id") or "")
                if mid.startswith("model_"):
                    try:
                        ids.append(int(mid.replace("model_", "")))
                    except ValueError:
                        continue
            if not ids:
                return "model_001"
            return f"model_{max(ids) + 1:03d}"
        except Exception:
            return "model_999"

    def _build_filename(self, model_id: str, prompt: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", prompt.strip().lower()) or "model"
        if len(safe) > 40:
            safe = safe[:40]
        return f"{model_id}_{safe}.obj"

    async def _run_async(self):
        self.status_update.emit("Calling Meshy to generate 3D model (this may take 2-5 minutes)...")
        
        # 1. Generate
        result = await self.client.generate_model(self.prompt)
        if not result.get("success"):
            error = result.get("error") or "unknown error"
            self.error_occurred.emit(f"Meshy generation failed: {error}")
            return

        obj_url = result.get("obj_url")
        if not obj_url:
            self.error_occurred.emit("Meshy result does not contain an OBJ download URL.")
            return

        # 2. Download
        assets_dir = Path(__file__).parent.parent / "assets"
        models_dir = assets_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        model_id = self._next_model_id()
        filename = self._build_filename(model_id, self.prompt)
        local_path = models_dir / filename

        self.status_update.emit("Downloading OBJ model from Meshy...")
        ok = await self.client.download_model(obj_url, str(local_path))
        if not ok:
            self.error_occurred.emit("Failed to download OBJ from Meshy.")
            return

        # 3. Upload/Save
        try:
            file_bytes = local_path.read_bytes()
        except OSError as e:
            self.error_occurred.emit(f"Failed to read downloaded file: {e}")
            return

        display_name = self.prompt.strip() or "AI Generated Model"
        tags = ["ai", "meshy"]

        self.status_update.emit("Uploading model to backend...")
        try:
            # Note: This calls requests/httpx synchronously usually, which is fine in a worker thread
            resp = self.data_manager.api.upload_model(
                name=display_name,
                tags=tags,
                file_bytes=file_bytes,
                filename=filename,
                model_id=model_id,
            )
            print(f"Upload response: {resp}")
        except Exception as e:
            self.error_occurred.emit(f"Error uploading to backend: {e}")
            return

        self.status_update.emit("Generation complete!")
        self.finished_success.emit()


class GalleryPage(BasePage):
    """The gallery page where you can browse and search through your 3D models"""
    
    def __init__(self, data_manager, viewer_page_callback=None, parent=None):
        """Create the gallery page. Needs a data manager to load models and a callback to open them in the viewer"""
        self.data_manager = data_manager
        self.viewer_page_callback = viewer_page_callback
        self.all_models = []
        super().__init__(parent)
        self.load_models()
    
    def setup_ui(self):
        """Put together the search bar and model list"""
        layout = QGridLayout(self) # was QVBox Layout
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setMaxLength(50)
        self.search_bar.setPlaceholderText("Search models by name or tags...")
        self.search_bar.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.search_bar.textChanged.connect(self.filter_models)
        layout.addWidget(self.search_bar)

        # Model list widget
        self.model_list = QListWidget()
        self.model_list.setAlternatingRowColors(True)
        self.model_list.itemClicked.connect(self.on_model_selected)
        self.model_list.setFlow(QListView.Flow.LeftToRight)
        self.model_list.setWrapping(True)
        self.model_list.setResizeMode(QListWidget.ResizeMode.Adjust)

        layout.addWidget(self.model_list)
    
    def load_models(self):
        """Grab all models from the database and show them in the list"""

        # Currently we just get all models from the database and show them in the list
        # TODO: We may find a better way to load models in the future
        self.all_models = self.data_manager.get_all_models()
        self.populate_model_list(self.all_models)
    
    def populate_model_list(self, models):
        """Fill the list with model names"""
        self.model_list.clear()

        # setup for the gallery view
        self.model_list.setWordWrap(True)
        self.model_list.setIconSize(QSize(150, 150))
        self.model_list.setGridSize(QSize(175, 175))
        self.model_list.setViewMode(QListWidget.ViewMode.IconMode)

        for model in models:
            # model name
            item = QListWidgetItem()
            item.setText(model['display_name'])
            item.setData(Qt.ItemDataRole.UserRole, model)

            # model thumbnail - use cross-platform path
            model_name = model['filename']
            project_root = Path(__file__).parent.parent
            thumbnail_path = project_root / "assets" / "thumbnails" / (Path(model_name).stem + ".png")

            # Ensure we have a local copy of the model before generating thumbnail.
            # This will download from backend/GCS if needed.
            if not thumbnail_path.exists():
                model_id = model.get("id")
                model_path_fs: str | None = None
                try:
                    if model_id is not None and hasattr(self.data_manager, "get_model_path"):
                        path_obj = self.data_manager.get_model_path(model_id)
                        if path_obj is not None:
                            model_path_fs = str(path_obj)
                        else:
                            # mark failure to avoid retry loop
                            model_path_fs = None
                except Exception as e:
                    print(f"Error resolving model path for thumbnail (id={model_id}): {e}")
                    model_path_fs = None

                if model_path_fs and os.path.exists(model_path_fs):
                    try:
                        thumbnail_object = ThreeDViewer()
                        thumbnail_object.generate_thumbnail(model_path_fs)
                    except Exception as e:
                        print(f"Error generating thumbnail for '{model_name}': {e}")
                else:
                    # Skip thumbnail generation if file missing/unreachable
                    model_path_fs = None

            if thumbnail_path.exists():
                thumbnail = QIcon(str(thumbnail_path))
                item.setIcon(thumbnail)

            self.model_list.addItem(item)

    
    def filter_models(self, query: str):
        """Update the list as the user types in the search box"""
        query = query.strip()
        if not query:
            self.populate_model_list(self.all_models)
        else:
            filtered_models = self.data_manager.search_models(query)
            self.populate_model_list(filtered_models)
    
    def on_model_selected(self, item: QListWidgetItem):
        """When someone clicks a model, open it in the 3D viewer"""
        model = item.data(Qt.ItemDataRole.UserRole)
        if not model:
            return
        
        # Get model path using DataManager
        model_path = self.data_manager.get_model_path(model['id'])
        if not model_path or not model_path.exists():
            print(f"Model file not found: {model_path}")
            return
        
        # Call callback to switch to viewer page and load model
        if self.viewer_page_callback:
            self.viewer_page_callback(model_path, model['display_name'])


class AIGenerationPage(BasePage):
    """Page where you can type a description and have AI generate a 3D model"""
    
    def __init__(self, data_manager: ClientDataManager, gallery_page: GalleryPage | None = None, parent=None):
        """Create the AI generation page

        Args:
            data_manager: client-side data manager used to talk to the backend
            gallery_page: optional reference to gallery page so we can refresh it after upload
        """
        self.data_manager = data_manager
        self.gallery_page = gallery_page
        self.meshy_client: MeshyClient | None = None
        super().__init__(parent)
    
    def setup_ui(self):
        """Build the prompt input and generate button"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title/Header
        title_label = QLabel("AI 3D Model Generation")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel("Enter a text prompt to generate a 3D model using AI")
        desc_label.setStyleSheet("color: #666; margin-bottom: 20px;")
        layout.addWidget(desc_label)

        # Text input area (like ChatGPT)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(
            "Describe the 3D model you want to generate...\n\n"
            "Example: 'A red sports car' or 'A medieval castle with towers'"
        )
        self.prompt_input.setMinimumHeight(200)
        layout.addWidget(self.prompt_input)

        # Generate button
        self.generate_button = QPushButton("Generate")
        self.generate_button.setMinimumHeight(40)
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        # TODO: Connect to API client in Week 5
        self.generate_button.clicked.connect(self.on_generate_clicked)
        layout.addWidget(self.generate_button)

        # Status label for basic feedback
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #444; margin-top: 8px;")
        layout.addWidget(self.status_label)

        # Add stretch to push content to top
        layout.addStretch()
    
    def get_prompt(self) -> str:
        """Returns whatever text the user typed in"""
        return self.prompt_input.toPlainText().strip()

    def _ensure_meshy_client(self) -> MeshyClient | None:
        """Create Meshy client if needed, handling missing API key gracefully."""
        if self.meshy_client is not None:
            return self.meshy_client
        try:
            self.meshy_client = MeshyClient()
            return self.meshy_client
        except Exception as e:
            # Most likely missing MESHY_API_KEY or network issues
            msg = f"Meshy client init failed: {e}"
            print(msg)
            self.status_label.setText(
                "Error: Meshy API is not configured. Please set the MESHY_API_KEY environment variable."
            )
            return None

    def on_generate_clicked(self):
        """When the generate button is pressed"""
        prompt = self.get_prompt()
        if not prompt:
            self.status_label.setText("Please enter a text description to generate a 3D model.")
            return

        client = self._ensure_meshy_client()
        if not client:
            return

        # Disable button to prevent double-click
        self.generate_button.setEnabled(False)
        self.generate_button.setText("Generating... (Please Wait)")
        self.status_label.setText("Starting generation worker...")

        # Create and start worker
        self.worker = GenerationWorker(client, self.data_manager, prompt, self)
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.error_occurred.connect(self.on_generation_error)
        self.worker.finished_success.connect(self.on_generation_success)
        self.worker.start()

    def on_generation_error(self, msg: str):
        self.status_label.setText(f"Error: {msg}")
        self._reset_ui()

    def on_generation_success(self):
        self.status_label.setText("Success! Model added to Gallery.")
        self._reset_ui(success=True)
        
        # Refresh gallery
        try:
            self.data_manager.get_all_models()
            if self.gallery_page:
                self.gallery_page.load_models()
        except Exception as e:
            print(f"Error refreshing gallery: {e}")

    def _reset_ui(self, success=False):
        self.generate_button.setEnabled(True)
        self.generate_button.setText("Generate Again" if success else "Generate")

    def clear_prompt(self):
        """Wipe out the text in the input box"""
        self.prompt_input.clear()


class ViewerPage(BasePage):
    """The 3D viewer page where models get displayed"""
    
    def __init__(self, parent=None):
        """Create the viewer page"""
        self.viewer = None
        super().__init__(parent)
    
    def setup_ui(self):
        """Set up buttons in the viewer tab"""
        button_layout = QHBoxLayout(self)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(0)

        # button_style_template copied from generate_button to be consistent
        button_style_template = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """
        # Download button -- copied from generate_button
        self.download_button = QPushButton("Download")
        self.download_button.setMinimumHeight(40)
        self.download_button.setStyleSheet(button_style_template)

        # Toggle Light button
        self.light_button = QPushButton("Light: On")
        self.light_button.setMinimumHeight(40)
        self.light_button.setStyleSheet(button_style_template)

        # Toggle Gallery button
        self.gallery_button = QPushButton("Add to Gallery")
        self.gallery_button.setMinimumHeight(40)
        self.gallery_button.setStyleSheet(button_style_template)

        # combining the buttons in a layout
        button_layout.addWidget(self.download_button)
        button_layout.addWidget(self.light_button)
        button_layout.addWidget(self.gallery_button)

        # making a widget hold the layout so we can nest layouts
        button_layout_widget = QWidget(self)
        button_layout_widget.setLayout(button_layout)

        """Set up the 3D viewer widget"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 3D viewer - QtInteractor IS a QWidget, so we can add it directly
        self.viewer = ThreeDViewer(self)
        # Set minimum size to ensure viewer is visible
        self.viewer.setMinimumSize(400, 400)
        self.viewer.show_grid()

        # Add viewer directly to layout (QtInteractor is already a widget)
        layout.addWidget(button_layout_widget) # adding the top row of buttons
        layout.addWidget(self.viewer, stretch=1)

        self.download_button.clicked.connect(self.clicked_download_button)
        self.light_button.clicked.connect(self.toggle_light_button)
        self.gallery_button.clicked.connect(self.clicked_gallery_button)
    
    def load_model(self, model_path: str):
        """Load a model file and show it in the viewer"""
        """ takes in the path to load and the name to set for a file download name"""
        if not self.viewer:
            return
        
        try:
            # Clear previous model
            self.viewer.clear()
            # Load new model
            self.model_path = str(model_path)  # so that download function can access it
            mesh = ThreeDViewer.load_model(model_path)
            
            # Normalize dimensions to avoid huge models
            mesh = ThreeDViewer.normalize_mesh(mesh)
            
            self.viewer.add_mesh(mesh)

            # Top-down light from above (positive Z direction)
            self.light = ThreeDViewer.setup_light()
            self.viewer.add_light(self.light)

            # Default camera: similar to SceneViewer, from upper-right diagonal
            cx, cy, cz = mesh.center
            bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
            extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1.0)
            distance = max(extent * 2.0, 2.5)
            camera_height = distance * 0.7
            camera_dist = distance * 0.7
            # From upper-left diagonal (negative X, positive Z)
            camera_pos = (cx - camera_dist, cy + camera_height, cz + camera_dist)
            # Use +Y as the up direction to keep the model upright in the view
            self.viewer.camera_position = [camera_pos, (cx, cy, cz), (0, 1, 0)]

            print(f"Loaded model: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

        # resetting button text upon new model loading
        self.download_button.setText("Download")
        self.gallery_button.setText("Add to Gallery")
        self.light_button.setText("Light: On")

    # makes the file and display name from the model path
    def file_name_from_model_path(self):
        """ takes the model path and gets the [name].obj from it """

        if self.model_path != None:
            file_name = self.model_path.split('/')[-1]
            return file_name
        else:
            return None

    # function for the next button click
    def file_name_exists(self, file_name):
        """ changes the filename so that it does not produce errors
            It does so by checking the last value for if it is a number or not """

        if file_name[-5].isdigit() == True:
            number = int(file_name[-5]) + 1
            string_number = str(number)
            file_name = file_name[:-5] + string_number + file_name[-4:]
        else:
            file_name = file_name[:-4] + "1" + file_name[-4:]

        return file_name

        # for buttons
    def clicked_download_button(self):
        """When the download button is pressed"""
        print("download clicked")
        self.download_button.setText("File Downloaded")
        file_name = self.file_name_from_model_path()

        download_path = pathlib.Path.home() / 'Downloads' # works for windows computer

        if os.path.exists(str(download_path / file_name)):
            file_name = self.file_name_exists(file_name)

        shutil.copyfile(self.model_path, file_name)
        shutil.move(file_name, download_path)

        return

    def toggle_light_button(self):
        """Toggles the light on or off"""

        if self.light.on:
            self.light.switch_off()
            self.light_button.setText("Light: Off")
        else:
            self.light.switch_on()
            self.light_button.setText("Light: On")
        return

    def clicked_gallery_button(self):
        """When the gallery button is pressed - placeholder for future implementation"""
        # TODO: Implement gallery save functionality when needed
        self.gallery_button.setText("Feature coming soon")
        return

    def clear(self):
        """Remove everything from the viewer"""
        if self.viewer:
            self.viewer.clear()
    
    def cleanup(self):
        """Free up OpenGL stuff when closing the window"""
        if self.viewer:
            try:
                self.viewer.clear()
                if hasattr(self.viewer, 'render_window') and self.viewer.render_window:
                    try:
                        self.viewer.render_window.Finalize()
                    except:
                        pass
                self.viewer.close()
            except Exception:
                pass


class SceneGeneratorPage(BasePage):
    """
    Scene generation page - implements the full pipeline "text input → 3D scene".

    Features:
    1. Accept natural language scene descriptions
    2. Invoke SceneBrain for semantic parsing
    3. Invoke LayoutEngine for spatial layout
    4. Render with SceneViewer
    5. Provide a debug visualization toggle
    """
    
    def __init__(self, data_manager=None, parent=None):
        """
        Initialize the scene generation page.

        Args:
            data_manager: Data manager for accessing the model library
            parent: Parent widget
        """
        self.data_manager = data_manager
        self._scene_brain = None
        self._layout_engine = None
        self._scene_nodes = []
        super().__init__(parent)
    
    def setup_ui(self):
        """Build the UI"""
        from PyQt6.QtWidgets import (
            QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
            QPushButton, QCheckBox, QSplitter, QFrame, QGroupBox
        )
        from PyQt6.QtCore import Qt
        
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # ===== Left control panel =====
        left_panel = QFrame()
        left_panel.setMaximumWidth(400)
        left_panel.setMinimumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(15)
        
        # Title
        title_label = QLabel("🎬 Smart Scene Composer")
        title_label.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            color: #2196F3;
            padding: 10px 0;
        """)
        left_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(
            "Enter a natural language description and the system will automatically:\n"
            "• Semantic analysis → understand objects and counts\n"
            "• Spatial layout → force-directed anti-overlap\n"
            "• Hierarchy constraints → anchor-based parent-child"
        )
        desc_label.setStyleSheet("color: #666; margin-bottom: 10px; line-height: 1.4;")
        desc_label.setWordWrap(True)
        left_layout.addWidget(desc_label)
        
        # Input area
        input_group = QGroupBox("Scene description")
        input_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        input_layout = QVBoxLayout(input_group)
        
        self.scene_input = QTextEdit()
        self.scene_input.setPlaceholderText(
            "Example inputs:\n\n"
            "• 5 trees and 3 rocks\n"
            "• A room with a table and two chairs\n"
            "• 3 soldiers and a knight standing guard\n"
            "• A desk with a lamp"
        )
        self.scene_input.setMinimumHeight(150)
        self.scene_input.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
                background-color: #fafafa;
            }
            QTextEdit:focus {
                border-color: #2196F3;
                background-color: white;
            }
        """)
        input_layout.addWidget(self.scene_input)
        left_layout.addWidget(input_group)
        
        # Generate button
        self.generate_btn = QPushButton("🚀 Generate Scene")
        self.generate_btn.setMinimumHeight(45)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.generate_btn.clicked.connect(self.on_generate_clicked)
        left_layout.addWidget(self.generate_btn)
        
        # Debug controls
        debug_group = QGroupBox("Debug options")
        debug_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        debug_layout = QVBoxLayout(debug_group)
        
        self.debug_checkbox = QCheckBox("Show debug info (AABB boxes & parent-child lines)")
        self.debug_checkbox.setStyleSheet("color: #555; padding: 5px;")
        self.debug_checkbox.stateChanged.connect(self.on_debug_toggled)
        debug_layout.addWidget(self.debug_checkbox)
        
        self.screenshot_btn = QPushButton("📷 Save Screenshot")
        self.screenshot_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #43A047;
            }
        """)
        self.screenshot_btn.clicked.connect(self.on_screenshot_clicked)
        debug_layout.addWidget(self.screenshot_btn)
        
        left_layout.addWidget(debug_group)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("""
            color: #666;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 6px;
        """)
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        
        # Spacer
        left_layout.addStretch()
        
        # ===== Right 3D view =====
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 3D scene viewer (with data_manager for remote fetch)
        from app.scene_viewer import SceneViewer
        self.scene_viewer = SceneViewer(self, data_manager=self.data_manager)
        self.scene_viewer.set_status_callback(self._set_status_text)
        self.scene_viewer.setMinimumSize(600, 500)
        right_layout.addWidget(self.scene_viewer)
        
        # Assemble main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)
    
    def _ensure_engines(self):
        """Ensure SceneBrain and LayoutEngine are initialized"""
        # Scene generation logic lives in SceneViewer now; nothing to init here
        pass
    
    def on_generate_clicked(self):
        """
        Handler for the generate button click.

        Steps:
        1. Read user input
        2. SceneBrain semantic analysis
        3. LayoutEngine spatial layout
        4. SceneViewer rendering
        """
        text = self.scene_input.toPlainText().strip()
        if not text:
            self._set_status_text("⚠️ Please enter a scene description")
            return
        
        # Disable button
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("⏳ Generating...")
        self.status_label.setText("🔍 Analyzing scene description...")
        
        try:
            # Delegate generation and rendering to SceneViewer
            self.scene_viewer.generate_scene(
                text=text,
                debug_enabled=self.debug_checkbox.isChecked()
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._set_status_text(f"❌ Generation failed: {str(e)}")
        finally:
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("🚀 Generate Scene")
    
    def on_debug_toggled(self, state):
        """Handle debug checkbox state change"""
        from PyQt6.QtCore import Qt
        self.scene_viewer.debug_mode = (state == Qt.CheckState.Checked.value)
        
        # If a scene already exists, let SceneViewer re-render
        try:
            if getattr(self.scene_viewer, "_scene_nodes", []):
                models_dir = self.scene_viewer._get_models_dir()
                debug_data = (
                    self.scene_viewer._layout_engine.debug_data
                    if self.scene_viewer.debug_mode
                    else None
                )
                self.scene_viewer.render_scene(
                    self.scene_viewer._scene_nodes,
                    models_dir,
                    debug_data
                )
        except Exception:
            pass
    
    def on_screenshot_clicked(self):
        """Handle screenshot button click"""
        try:
            filepath = self.scene_viewer.take_scene_screenshot()
            self.status_label.setText(f"📸 Screenshot saved: {filepath}")
        except Exception as e:
            self.status_label.setText(f"❌ Screenshot failed: {str(e)}")
    
    def _set_status_text(self, text: str):
        """Unified status update helper"""
        self.status_label.setText(text)
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'scene_viewer') and self.scene_viewer:
            try:
                self.scene_viewer.clear_scene()
                if hasattr(self.scene_viewer, 'render_window') and self.scene_viewer.render_window:
                    try:
                        self.scene_viewer.render_window.Finalize()
                    except:
                        pass
                self.scene_viewer.close()
            except Exception:
                pass